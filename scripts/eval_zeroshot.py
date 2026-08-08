#!/usr/bin/env python3
"""Score zero-shot cloning: clone held-out sentences, then measure the result.

Protocol, following what published zero-shot TTS evaluations do so the numbers
mean something next to theirs:

  1. Pick a speaker with several utterances. One becomes the reference clip; the
     rest are held out.
  2. For each held-out utterance, synthesize *its text* in the cloned voice.
  3. Score: WER against that text, and speaker similarity against the held-out
     recording — never against the reference clip, which would be SIM-r.
  4. Score the **real held-out recordings the same way** and print them as the
     ground-truth row. Without it neither column is interpretable: TTS routinely
     beats human speech on ASR-WER, and the similarity ceiling is set by the
     speaker and encoder, not by 1.0.

    python scripts/eval_zeroshot.py --speakers 3 --utterances 4
"""
from __future__ import annotations

import argparse
import io
import platform
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import (EvalTable, SystemScores,  # noqa: E402
                                  ctc_transcriber, score_intelligibility,
                                  sim_o, whisper_transcriber)
from speechforge.synth import clone, load_backend  # noqa: E402


def hardware_label() -> str:
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        elif platform.system() == "Linux":
            cpu = ""
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        else:
            cpu = ""
    except Exception:
        cpu = ""
    return f"{cpu or platform.machine()} ({platform.system()})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="chatterbox")
    p.add_argument("--corpus", default="edacc",
                   choices=["edacc", "librispeech_dummy"],
                   help="librispeech_dummy is a smoke test: it has one speaker "
                        "with multiple utterances, so no cross-speaker floor")
    p.add_argument("--second-asr", action="store_true",
                   help="also score with HuBERT-CTC. Scoring TTS only with "
                        "Whisper measures partly Whisper's own tail")
    p.add_argument("--asr-model", default="small",
                   help="large-v3 for publishable numbers; small to iterate")
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--utterances", type=int, default=3,
                   help="held-out utterances per speaker")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out-dir", default=None, help="keep the synthesized wavs")
    args = p.parse_args()

    print(f"study=eval_zeroshot  hardware={hardware_label()}  "
          f"backend={args.backend}  asr={args.asr_model}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    speakers = corpora.load(args.corpus, args.speakers, args.utterances)
    print(f"corpus={args.corpus}  speakers={len(speakers)}  "
          f"{args.utterances} held-out utterances each")
    print(f"speaker ids: {list(speakers)}\n")

    backend = load_backend(args.backend, device="cpu")
    backend.set_threads(args.threads)
    print(f"backend: {backend.describe()}  sr={backend.sample_rate}")

    encoder = None
    try:
        from speechforge.similarity import load_encoder
        encoder = load_encoder()
        print(f"speaker encoder: {encoder.describe()}")
    except Exception as exc:
        print(f"speaker encoder unavailable ({exc}); SIM column will be blank")

    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")
    transcribe2 = None
    if args.second_asr:
        try:
            transcribe2 = ctc_transcriber()
            print("second ASR: facebook/hubert-large-ls960-ft (CTC)")
        except Exception as exc:
            print(f"second ASR unavailable ({exc}); WER-2 will be blank")
    print()

    synth_pairs, real_pairs = [], []
    synth_sims, real_sims = [], []
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for spk, utts in speakers.items():
        ref_audio, ref_sr, _ref_text = utts[0]
        held = utts[1:]
        print(f"speaker {spk}: cloning from {len(ref_audio)/ref_sr:.1f}s, "
              f"{len(held)} held-out")

        for j, (ho_audio, ho_sr, text) in enumerate(held):
            wav, sr, report = clone(backend, text, ref_audio, ref_sr)
            if report["warnings"]:
                print(f"  ref warnings: {report['warnings']}")
            if out_dir:
                import soundfile as sf
                sf.write(str(out_dir / f"{spk}_{j}.wav"), wav, sr)

            synth_pairs.append((wav, sr, text))
            real_pairs.append((ho_audio, ho_sr, text))

            if encoder is not None:
                # Held-out utterance, with the reference passed so the harness
                # can refuse if they are ever the same array.
                synth_sims.append(sim_o(encoder, wav, sr, ho_audio, ho_sr,
                                        reference_audio=ref_audio))
                # Ground truth for this column: a *different* real recording of
                # the same speaker against the held-out one.
                other = held[(j + 1) % len(held)] if len(held) > 1 else None
                if other is not None:
                    real_sims.append(sim_o(encoder, other[0], other[1],
                                           ho_audio, ho_sr))

    print("\nscoring...")
    table = EvalTable(corpus=f"{args.corpus} (held-out)",
                      hardware=hardware_label())
    table.set_ground_truth(SystemScores(
        name="ground truth",
        wer=score_intelligibility(transcribe, real_pairs),
        wer_second=(score_intelligibility(transcribe2, real_pairs)
                    if transcribe2 else None),
        sim_o=float(np.nanmean(real_sims)) if real_sims else None,
        n_utterances=len(real_pairs),
        notes="real recordings, scored identically"))
    table.add(SystemScores(
        name=args.backend,
        wer=score_intelligibility(transcribe, synth_pairs),
        wer_second=(score_intelligibility(transcribe2, synth_pairs)
                    if transcribe2 else None),
        sim_o=float(np.nanmean(synth_sims)) if synth_sims else None,
        n_utterances=len(synth_pairs)))

    print()
    print(table.render())


if __name__ == "__main__":
    main()
