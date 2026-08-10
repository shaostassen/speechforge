#!/usr/bin/env python3
"""Does synthesize-then-convert actually produce a golden speaker?

The learner is an accented EdAcc speaker; the pronunciation source is a native
LibriSpeech speaker. For each held-out learner utterance we build four signals
and score them all the same way:

  learner-real     the learner's own recording          identity ceiling
  native-tts       TTS from the native reference        pronunciation ceiling
  learner-tts      TTS cloned from the learner          what plain cloning gives
  golden           native-tts converted to the learner  the thing under test

`learner-tts` is the baseline that matters. If plain cloning already sounds like
the learner *and* pronounces correctly, the conversion stage is unnecessary — so
the interesting question is not "does golden work" but "does golden beat cloning".

Every number is a similarity against a **held-out** learner utterance, never the
reference clip, and WER against the utterance's own text.

    python scripts/eval_golden.py --speakers 4 --utterances 3
"""
from __future__ import annotations

import argparse
import platform
import shlex
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import cosine, whisper_transcriber  # noqa: E402
from speechforge.golden import (ChatterboxVCBackend, golden_speech,  # noqa: E402
                                trade_off)
from speechforge.reference import prepare  # noqa: E402
from speechforge.synth import clone, load_backend  # noqa: E402


def hardware_label() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speakers", type=int, default=4, help="accented learners")
    p.add_argument("--utterances", type=int, default=3)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scan", type=int, default=9000)
    args = p.parse_args()

    print(f"study=eval_golden  hardware={hardware_label()}  "
          f"asr={args.asr_model}  seed={args.seed}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    import soundfile as sf

    from speechlens.metrics import wer

    learners = corpora.load("edacc", args.speakers, args.utterances,
                            scan=args.scan)
    natives = corpora.load("librispeech", 1, 1)
    native_ref, native_sr, _ = next(iter(natives.values()))[0]
    print(f"learners (accented): {list(learners)}")
    print(f"pronunciation source: LibriSpeech {list(natives)[0]}, "
          f"{len(native_ref)/native_sr:.1f}s\n")

    tts = load_backend("chatterbox", device="cpu")
    tts.set_threads(args.threads)
    vc = ChatterboxVCBackend(device="cpu")
    print(f"tts: {tts.describe()}\nvc : {vc.describe()}")

    from speechforge.similarity import load_encoder
    enc = load_encoder()
    print(f"encoder: {enc.describe()}")

    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")
    print()

    rows = []
    for spk, utts in learners.items():
        learner_ref, lref_sr, _ = utts[0]
        # Stage the learner's clip once: the converter takes a path, and
        # re-preparing it per utterance would be wasted work and one more place
        # for the two runs to diverge.
        prepped, pref_sr, _rep = prepare(learner_ref, lref_sr,
                                         target_sr=tts.sample_rate,
                                         target_s=getattr(tts, "ref_seconds", None) or 10.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, prepped, pref_sr)

        print(f"{spk}: learner ref {len(prepped)/pref_sr:.1f}s, "
              f"{len(utts) - 1} held-out")

        held = utts[1:]
        for j, (held_audio, held_sr, text) in enumerate(held):
            g = golden_speech(tts, vc, text, native_ref, native_sr, tmp.name,
                              seed=args.seed)
            learner_tts, lt_sr, _ = clone(tts, text, learner_ref, lref_sr,
                                          seed=args.seed)

            # Every similarity is against the HELD-OUT utterance, never the
            # reference clip the models were conditioned on.
            e_held = enc(held_audio, held_sr)
            sim = lambda a, s: cosine(enc(a, s), e_held)  # noqa: E731

            rows.append({
                "speaker": spk,
                "words": len(text.split()),
                # A DIFFERENT real recording of the same speaker, cyclically —
                # comparing utts[1] every time would compare the first held-out
                # utterance against itself and report a spurious 1.0.
                "sim_learner_real": (sim(held[(j + 1) % len(held)][0],
                                         held[(j + 1) % len(held)][1])
                                     if len(held) > 1 else float("nan")),
                "sim_native_tts": sim(g.source_audio, g.source_sample_rate),
                "sim_learner_tts": sim(learner_tts, lt_sr),
                "sim_golden": sim(g.audio, g.sample_rate),
                "wer_native_tts": wer(text, transcribe(g.source_audio,
                                                       g.source_sample_rate)),
                "wer_learner_tts": wer(text, transcribe(learner_tts, lt_sr)),
                "wer_golden": wer(text, transcribe(g.audio, g.sample_rate)),
                "wer_real": wer(text, transcribe(held_audio, held_sr)),
            })

    def m(key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 68)
    print("GOLDEN SPEAKER — identity vs pronunciation")
    print("=" * 68)
    print(f"{'signal':>16} {'SIM to learner':>15} {'WER':>7}")
    print(f"{'learner real':>16} {'(held-out ref)':>15} {m('wer_real'):>7.3f}")
    print(f"{'native tts':>16} {m('sim_native_tts'):>15.3f} "
          f"{m('wer_native_tts'):>7.3f}")
    print(f"{'learner tts':>16} {m('sim_learner_tts'):>15.3f} "
          f"{m('wer_learner_tts'):>7.3f}")
    print(f"{'golden':>16} {m('sim_golden'):>15.3f} {m('wer_golden'):>7.3f}")

    t = trade_off(m("sim_golden"), m("sim_native_tts"),
                  m("wer_golden"), m("wer_native_tts"))
    print(f"\nidentity gain (golden - native tts) : {t['identity_gain']:+.3f}")
    print(f"WER cost      (golden - native tts) : {t['wer_cost']:+.3f}")
    print(f"verdict: {t['verdict']}")

    print(f"\nvs plain cloning: golden SIM {m('sim_golden'):.3f} vs "
          f"learner-tts {m('sim_learner_tts'):.3f}  "
          f"({m('sim_golden') - m('sim_learner_tts'):+.3f})")
    print(f"                  golden WER {m('wer_golden'):.3f} vs "
          f"learner-tts {m('wer_learner_tts'):.3f}  "
          f"({m('wer_golden') - m('wer_learner_tts'):+.3f})")
    print("\nIf plain cloning already matches the learner and pronounces well, "
          "the\nconversion stage is not earning its place.")


if __name__ == "__main__":
    main()
