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
from speechforge.evaluate import (EvalTable, ScorerUnavailable,  # noqa: E402
                                  SystemScores, ctc_transcriber,
                                  mos_predictor, require_scorer,
                                  score_intelligibility, score_naturalness,
                                  sim_o, whisper_transcriber)
from speechforge.prosody import pitch_gap_st, prosody_scorer  # noqa: E402
from speechforge.synth import clone, load_backend  # noqa: E402
from speechforge.text import apply_case  # noqa: E402


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


def _round(v, p=3):
    return None if v is None or v != v else round(float(v), p)


def _parse_number(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="chatterbox")
    p.add_argument("--corpus", default="edacc",
                   choices=["edacc", "librispeech", "librispeech_dummy"],
                   help="librispeech_dummy is a smoke test: it has one speaker "
                        "with multiple utterances, so no cross-speaker floor")
    p.add_argument("--temperature", type=float, default=None,
                   help="override the backend's decode temperature. Chatterbox "
                        "defaults to 0.8, which is a lot of sampling entropy "
                        "for an evaluation")
    p.add_argument("--cfg-weight", type=float, default=None,
                   help="override classifier-free-guidance weight (default 0.5)")
    p.add_argument("--gen", action="append", default=[], metavar="KEY=VALUE",
                   help="any other backend setting, repeatable, e.g. --gen "
                        "alpha=0 --gen beta=0 for StyleTTS 2. Values parse as "
                        "numbers where they can. They reach the backend's "
                        "describe(), and so the provenance banner")
    p.add_argument("--mos", action="store_true",
                   help="also predict MOS (UTMOS). A smoke alarm for broken "
                        "audio, not a ranking signal \u2014 it correlates poorly "
                        "at system level and saturates on good audio")
    p.add_argument("--second-asr", action="store_true",
                   help="also score with HuBERT-CTC. Scoring TTS only with "
                        "Whisper measures partly Whisper's own tail")
    p.add_argument("--prosody", action="store_true",
                   help="also measure prosody: pitch range, pitch level against "
                        "the held-out recording, speaking rate, each beside the "
                        "same measurement on the real recordings. Needs the "
                        "[prosody] extra; see speechforge/prosody.py for what it "
                        "does and does not capture")
    p.add_argument("--no-sim", action="store_true",
                   help="skip speaker similarity. Without it, a speaker encoder "
                        "that fails to load stops the run before any synthesis "
                        "— as does any scorer asked for with --mos or "
                        "--second-asr — rather than leaving its column blank")
    p.add_argument("--asr-model", default="small",
                   help="large-v3 for publishable numbers; small to iterate")
    p.add_argument("--speakers", type=int, default=2)
    p.add_argument("--utterances", type=int, default=3,
                   help="held-out utterances per speaker")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--scan", type=int, default=None,
                   help="edacc only: how many streamed rows to search for "
                        "speakers. Utterances come grouped by conversation, so "
                        "reaching N speakers needs ~N/2 conversations")
    p.add_argument("--seed", type=int, default=1234,
                   help="seed each synthesis. Chatterbox samples at "
                        "temperature 0.8, so unseeded runs are not "
                        "reproducible: two identical runs of this study gave "
                        "AUROC 0.591 and 0.553. Pass -1 to sample freely and "
                        "measure the spread instead")
    p.add_argument("--out-dir", default=None, help="keep the synthesized wavs")
    p.add_argument("--ref-seconds", type=float, default=None,
                   help="reference-clip cap in seconds (default 10). The "
                        "default is borrowed from other models' docs, not "
                        "measured on this one")
    p.add_argument("--per-utterance", default=None, metavar="PATH",
                   help="write per-utterance WER and text length as JSON. Lets "
                        "a corpus-level WER be decomposed: a few long "
                        "utterances failing badly looks identical to uniform "
                        "mediocrity in the aggregate")
    p.add_argument("--text-case", choices=["raw", "sentence"], default="raw",
                   help="how corpus text is presented to the model. LibriSpeech "
                        "and EdAcc ship unpunctuated UPPER CASE, which is "
                        "out-of-distribution for an LLM-based TTS: CosyVoice 2 "
                        "measured WER 0.316 on raw caps vs 0.048 on sentence "
                        "case. Default 'raw' keeps earlier runs comparable. "
                        "WER scoring is case-insensitive, so this cannot flatter "
                        "a system through the metric.")
    args = p.parse_args()

    print(f"study=eval_zeroshot  hardware={hardware_label()}  "
          f"backend={args.backend}  asr={args.asr_model}  "
          f"seed={args.seed}  text_case={args.text_case}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    speakers = corpora.load(args.corpus, args.speakers, args.utterances,
                            **({'scan': args.scan}
                               if args.scan and args.corpus == 'edacc' else {}))
    print(f"corpus={args.corpus}  speakers={len(speakers)}  "
          f"{args.utterances} held-out utterances each")
    print(f"speaker ids: {list(speakers)}\n")

    gen = {}
    if args.temperature is not None:
        gen["temperature"] = args.temperature
    if args.cfg_weight is not None:
        gen["cfg_weight"] = args.cfg_weight
    for item in args.gen:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--gen expects KEY=VALUE, got {item!r}")
        gen[key] = _parse_number(value)
    backend = load_backend(args.backend, device="cpu", **gen)
    backend.set_threads(args.threads)
    print(f"backend: {backend.describe()}  sr={backend.sample_rate}")

    # Every scorer is loaded before any synthesis, and one that was asked for
    # and fails stops the run (require_scorer): a blank column in a finished
    # table is easy to miss, a failed run is not.
    def _encoder():
        from speechforge.similarity import load_encoder
        return load_encoder()

    encoder = None
    if args.no_sim:
        print("speaker encoder: off (--no-sim); SIM column will be blank")
    else:
        encoder = require_scorer("speaker encoder", _encoder, "pass --no-sim")
        print(f"speaker encoder: {encoder.describe()}")

    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")
    predict_mos = None
    if args.mos:
        predict_mos = require_scorer("MOS predictor", mos_predictor,
                                     "drop --mos")
        print("naturalness: UTMOS (SpeechMOS) — sanity check only")

    transcribe2 = None
    if args.second_asr:
        transcribe2 = require_scorer("second ASR", ctc_transcriber,
                                     "drop --second-asr")
        print("second ASR: facebook/hubert-large-ls960-ft (CTC)")

    measure_prosody = None
    if args.prosody:
        measure_prosody = require_scorer("prosody (pYIN)", prosody_scorer,
                                         "drop --prosody")
        print("prosody: pYIN F0 (librosa) — pitch range, pitch gap, words/s")
    print()

    synth_pairs, real_pairs, reports = [], [], []
    synth_sims, real_sims = [], []
    pros_rows = []          # one per synthesized utterance, when --prosody
    out_dir = Path(args.out_dir) if args.out_dir else None
    real_cache = {}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for spk, utts in speakers.items():
        # The reference transcript is not spare: CosyVoice 2 conditions on it,
        # and the corpora already carry it, so no extra work is needed to supply
        # it. Backends that ignore it are unaffected.
        ref_audio, ref_sr, ref_text = utts[0]
        held = utts[1:]
        print(f"speaker {spk}: cloning from {len(ref_audio)/ref_sr:.1f}s, "
              f"{len(held)} held-out")

        for j, (ho_audio, ho_sr, text) in enumerate(held):
            # Presentation applies to what the MODEL reads, on both the synthesis
            # text and the reference transcript. `text` itself stays raw for
            # scoring below — and WER normalizes case anyway.
            wav, sr, report = clone(backend, apply_case(text, args.text_case),
                                        ref_audio, ref_sr,
                                        seed=None if args.seed < 0 else args.seed,
                                        ref_seconds=args.ref_seconds,
                                        reference_text=apply_case(
                                            ref_text, args.text_case))
            reports.append(report)
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

            if measure_prosody is not None:
                # Real recordings are measured once each and reused: every
                # held-out clip is both a target here and, for its neighbour,
                # the "other recording" of the pitch-gap ground truth.
                def real_prosody(k):
                    key = (spk, k)
                    if key not in real_cache:
                        a, s, t = held[k]
                        real_cache[key] = measure_prosody(a, s, t)
                    return real_cache[key]

                syn = measure_prosody(wav, sr, text)
                real = real_prosody(j)
                other_p = real_prosody((j + 1) % len(held)) if len(held) > 1 else None
                pros_rows.append({
                    "synth": syn, "real": real,
                    "gap_synth": pitch_gap_st(syn.median_f0_hz, real.median_f0_hz),
                    # Ground truth for the gap: another real recording of the
                    # same speaker, as for SIM-o.
                    "gap_real": (pitch_gap_st(other_p.median_f0_hz, real.median_f0_hz)
                                 if other_p is not None else float("nan")),
                })

    if args.per_utterance:
        # Decompose the aggregate. Whisper only, since the point is the shape of
        # the error distribution rather than a second opinion on its level.
        import json

        from speechlens.metrics import wer as _wer
        rows = []
        for i, ((swav, ssr, text), (rwav, rsr, _t), rep) in enumerate(zip(
                synth_pairs, real_pairs, reports)):
            row = {
                "words": len(text.split()),
                "ref_s": rep["duration_s"],
                "ref_original_s": rep["original_duration_s"],
                "audio_s": round(len(swav) / ssr, 2),
                "real_audio_s": round(len(rwav) / rsr, 2),
                "wer_synth": round(_wer(text, transcribe(swav, ssr)), 4),
                "wer_real": round(_wer(text, transcribe(rwav, rsr)), 4),
            }
            if pros_rows:
                pr = pros_rows[i]
                row.update({"prosody_synth": pr["synth"].to_dict(),
                            "prosody_real": pr["real"].to_dict(),
                            "pitch_gap_synth": _round(pr["gap_synth"]),
                            "pitch_gap_real": _round(pr["gap_real"])})
            rows.append(row)
        Path(args.per_utterance).write_text(json.dumps(rows, indent=2))
        print(f"per-utterance detail -> {args.per_utterance}")

    print("\nscoring...")

    def pmean(side, field):
        vals = [getattr(r[side], field) for r in pros_rows]
        vals = [v for v in vals if v is not None and v == v]
        return float(np.mean(vals)) if vals else None

    def gmean(key):
        vals = [r[key] for r in pros_rows if r[key] == r[key]]
        return float(np.mean(vals)) if vals else None

    table = EvalTable(corpus=f"{args.corpus} (held-out)",
                      hardware=hardware_label())
    table.set_ground_truth(SystemScores(
        name="ground truth",
        wer=score_intelligibility(transcribe, real_pairs),
        wer_second=(score_intelligibility(transcribe2, real_pairs)
                    if transcribe2 else None),
        sim_o=float(np.nanmean(real_sims)) if real_sims else None,
        mos=(score_naturalness(predict_mos, [(a, s) for a, s, _t in real_pairs])
             if predict_mos else None),
        n_utterances=len(real_pairs),
        notes="real recordings, scored identically",
        pitch_range_st=pmean("real", "pitch_range_st"),
        pitch_gap_st=gmean("gap_real"),
        words_per_s=pmean("real", "words_per_s"),
        jump_fraction=pmean("real", "jump_fraction")))
    table.add(SystemScores(
        name=args.backend,
        wer=score_intelligibility(transcribe, synth_pairs),
        wer_second=(score_intelligibility(transcribe2, synth_pairs)
                    if transcribe2 else None),
        sim_o=float(np.nanmean(synth_sims)) if synth_sims else None,
        mos=(score_naturalness(predict_mos, [(a, s) for a, s, _t in synth_pairs])
             if predict_mos else None),
        n_utterances=len(synth_pairs),
        pitch_range_st=pmean("synth", "pitch_range_st"),
        pitch_gap_st=gmean("gap_synth"),
        words_per_s=pmean("synth", "words_per_s"),
        jump_fraction=pmean("synth", "jump_fraction")))

    print()
    print(table.render())


if __name__ == "__main__":
    try:
        main()
    except ScorerUnavailable as exc:
        raise SystemExit(f"error: {exc}")
