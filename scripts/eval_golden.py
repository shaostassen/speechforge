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


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import cosine, whisper_transcriber  # noqa: E402
from speechforge.golden import (ChatterboxVCBackend, blend_rate,  # noqa: E402
                                golden_speech, golden_table,
                                mean_ignoring_nan, trade_off)
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
    p.add_argument("--per-utterance", metavar="PATH",
                   help="dump every row as JSON. Worth doing: the first run of "
                        "this study printed only aggregates, so a reporting bug "
                        "cost the whole run instead of one re-render.")
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
    print(f"tts: {tts.describe()} @ {tts.sample_rate} Hz")
    print(f"vc : {vc.describe()} @ {vc.sample_rate} Hz")
    if tts.sample_rate != vc.sample_rate:
        print(f"  !! backends disagree on sample rate — the learner reference is "
              f"prepared at the TTS rate ({tts.sample_rate}) and handed to the "
              f"converter, so identity numbers are suspect")

    from speechforge.similarity import load_encoder
    enc = load_encoder()
    print(f"encoder: {enc.describe()}")

    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")
    print()

    # The source speaker's REAL recording, embedded once — it is the same clip for
    # every utterance. This is what makes "did the output shed the source" a
    # matched question: scoring golden against it is synthetic-vs-real with
    # different text, exactly like scoring golden against the learner's held-out
    # audio. Scoring against the native TTS instead is not matched, because golden
    # is a conversion of that waveform and inherits its prosody and vocoder.
    e_native_real = enc(native_ref, native_sr)

    rows = []
    for spk, utts in learners.items():
        learner_ref, lref_sr, _ = utts[0]
        # Stage the learner's clip once: the converter takes a path, and
        # re-preparing it per utterance would be wasted work and one more place
        # for the two runs to diverge.
        # Prepared to the *TTS* backend's rate and cap even though this clip is
        # the converter's target. That is deliberate — the cloning arm prepares
        # identically, and the two arms must see the same reference for the
        # comparison to mean anything. But it silently assumes the two backends
        # agree on sample rate; the banner prints both so a disagreement shows up
        # in the log instead of skewing a result.
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

            # Embed each clip once and reuse. Every similarity to the learner is
            # against the HELD-OUT utterance, never the reference clip the models
            # were conditioned on.
            e_held = enc(held_audio, held_sr)
            e_src = enc(g.source_audio, g.source_sample_rate)
            e_gold = enc(g.audio, g.sample_rate)
            e_ltts = enc(learner_tts, lt_sr)
            other_audio, other_sr, _ = held[(j + 1) % len(held)]
            e_other = enc(other_audio, other_sr) if len(held) > 1 else None

            rows.append({
                "speaker": spk,
                "words": len(text.split()),
                # A DIFFERENT real recording of the same speaker, cyclically —
                # comparing utts[1] every time would compare the first held-out
                # utterance against itself and report a spurious 1.0.
                "sim_learner_real": (cosine(e_other, e_held)
                                     if e_other is not None else float("nan")),
                "sim_native_tts": cosine(e_src, e_held),
                "sim_learner_tts": cosine(e_ltts, e_held),
                "sim_golden": cosine(e_gold, e_held),
                # Identity SHED: how much the output still resembles the voice it
                # was converted away from. Without this a blend of the two voices
                # passes every other check.
                "sim_golden_to_source": cosine(e_gold, e_src),
                # The anchor that makes the line above readable: two clones of
                # DIFFERENT speakers from the SAME TTS and vocoder. Whatever
                # similarity shared synthesis artefacts produce on their own shows
                # up here, so only the excess above it can be retained identity.
                "sim_ltts_to_source": cosine(e_ltts, e_src),
                # Identity RETAINED, the matched version: golden against the
                # source speaker's real audio. Same kind of comparison as
                # sim_golden (synthetic vs real, different text, no shared
                # prosody), so the two read on one scale.
                "sim_golden_to_source_real": cosine(e_gold, e_native_real),
                # Floor control: a clone of the LEARNER has no source speaker in
                # it at all, so whatever this scores is the level two unrelated
                # voices reach under this encoder and pipeline.
                "sim_ltts_to_source_real": cosine(e_ltts, e_native_real),
                "wer_native_tts": wer(text, transcribe(g.source_audio,
                                                       g.source_sample_rate)),
                "wer_learner_tts": wer(text, transcribe(learner_tts, lt_sr)),
                "wer_golden": wer(text, transcribe(g.audio, g.sample_rate)),
                "wer_real": wer(text, transcribe(held_audio, held_sr)),
            })

    def m(key):
        return mean_ignoring_nan(rows, key)

    if args.per_utterance:
        import json
        # Written before the table, so a rendering failure cannot destroy the
        # measurement that took an hour to produce.
        Path(args.per_utterance).write_text(json.dumps(rows, indent=1))
        print(f"per-utterance rows -> {args.per_utterance}")

    print(f"\nn = {len(rows)} held-out utterances, {len(learners)} learners")
    print(golden_table(rows))

    frac = blend_rate(rows)
    t = trade_off(m("sim_golden"), m("sim_native_tts"),
                  m("wer_golden"), m("wer_native_tts"),
                  sim_to_source=m("sim_golden_to_source"),
                  blend_fraction=frac)
    print(f"\nidentity acquired  SIM(golden, learner)    : {t['sim_to_learner']:.3f}"
          f"   (source started at {t['sim_source_to_learner']:.3f})")
    print(f"identity gain (golden - native tts)        : {t['identity_gain']:+.3f}")
    print(f"WER cost      (golden - native tts)        : {t['wer_cost']:+.3f}")
    # Per utterance, not from the means: the means can favour the learner while
    # most individual utterances do not.
    n_blend = round(frac * len(rows))
    print(f"closer to source than learner              : {n_blend}/{len(rows)} "
          f"utterances ({frac:.0%})")
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
