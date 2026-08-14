#!/usr/bin/env python3
"""Is correcting two words better than regenerating all thirty?

`eval_golden.py` regenerates the whole utterance in the learner's voice. This
regenerates only the words a calibrated per-word reliability policy flags, and
splices them into the learner's own recording — so most of the output is genuinely
their speech, with their prosody and pacing intact.

Three arms, all scored the same way:

  learner real     the recording, untouched            what they actually said
  golden whole     the whole utterance regenerated     eval_golden.py's output
  golden targeted  only flagged spans regenerated      the thing under test

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
It measures **cost**: how much audio is regenerated, how many splice seams that
creates, and how much speaker identity survives. Those are the engineering claims,
and they are answerable here.

It does **not** measure whether the right words were chosen. That needs labels for
which words the learner actually mispronounced, and EdAcc does not carry them — the
same "ground truth is impossible by construction" wall `golden.py` hit.

And **WER on the targeted arm is close to circular**: flagged audio is replaced with
speech synthesized *from the reference transcript*, so an ASR reading that transcript
back is being graded on text the pipeline was handed. A WER improvement here is
mostly arithmetic, not evidence that anyone's pronunciation improved. It is reported
because leaving it out would hide that the replacement can still damage the
surrounding audio — a WER that fails to improve is informative; one that improves is
not.

    python scripts/eval_targeted.py --speakers 6 --utterances 3
"""
from __future__ import annotations

import argparse
import json
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
from speechforge.golden import ChatterboxVCBackend, golden_speech  # noqa: E402
from speechforge.reference import prepare  # noqa: E402
from speechforge.synth import load_backend  # noqa: E402
from speechforge.target import (replacement_fraction, select_spans,  # noqa: E402
                                splice)
from speechforge.text import apply_case  # noqa: E402


def hardware_label() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def reference_words_for_span(ref_text: str, hyp_words, lo: int, hi: int):
    """Map a span of ASR words back to the words the learner *meant* to say.

    The span is chosen from ASR output, but the correction has to speak the
    reference text — synthesizing the ASR's own guess would reproduce the
    misrecognition it was flagged for. `align` gives the Levenshtein path, so each
    hypothesis index maps to a reference index where one exists.
    """
    from speechlens.metrics import align

    ref_tokens = ref_text.split()
    hyp_tokens = [w["word"].strip() for w in hyp_words]
    picked = []
    for op, ri, hi_ in align(ref_tokens, hyp_tokens):
        if hi_ is not None and lo <= hi_ < hi and ri is not None:
            picked.append(ri)
    if not picked:
        return ""
    # Contiguous reference slice: a span with a deletion in the middle would
    # otherwise drop the missing word from the correction entirely.
    return " ".join(ref_tokens[min(picked):max(picked) + 1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speakers", type=int, default=6)
    p.add_argument("--utterances", type=int, default=3)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scan", type=int, default=9000)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="reliability below this flags a word for correction")
    p.add_argument("--pad-words", type=int, default=1,
                   help="widen each span so cuts land on unsuspect words")
    p.add_argument("--text-case", choices=["raw", "sentence"], default="sentence")
    p.add_argument("--per-utterance", metavar="PATH")
    args = p.parse_args()

    print(f"study=eval_targeted  hardware={hardware_label()}  "
          f"asr={args.asr_model}  seed={args.seed}  "
          f"threshold={args.threshold}  pad_words={args.pad_words}  "
          f"text_case={args.text_case}  date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    import soundfile as sf

    from speechlens.metrics import wer
    from speechlens.pipeline import SpeechLens

    learners = corpora.load("edacc", args.speakers, args.utterances,
                            scan=args.scan)
    natives = corpora.load("librispeech", 1, 1)
    native_ref, native_sr, _ = next(iter(natives.values()))[0]
    print(f"learners: {list(learners)}")
    print(f"pronunciation source: LibriSpeech {list(natives)[0]}\n")

    tts = load_backend("chatterbox", device="cpu")
    tts.set_threads(args.threads)
    vc = ChatterboxVCBackend(device="cpu")
    from speechforge.similarity import load_encoder
    enc = load_encoder()
    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")

    # Word-level reliability comes from the other project, policy and all.
    # Word timestamps are OFF by default there and are what the whole selector
    # depends on: without them speechlens returns no per-word scores rather than
    # guessing, so this study would skip every utterance and report n=0.
    from speechlens.asr import RobustnessConfig
    cfg = RobustnessConfig(word_timestamps=True)
    lens = SpeechLens(model_size=args.asr_model, device="cpu",
                      compute_type="int8")
    print(f"tts: {tts.describe()}\nvc : {vc.describe()}\nenc: {enc.describe()}\n")

    rows = []
    for spk, utts in learners.items():
        learner_ref, lref_sr, _ = utts[0]
        prepped, pref_sr, _ = prepare(learner_ref, lref_sr,
                                      target_sr=tts.sample_rate,
                                      target_s=getattr(tts, "ref_seconds", None) or 10.0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, prepped, pref_sr)
        held = utts[1:]
        print(f"{spk}: {len(held)} held-out", flush=True)

        for j, (audio, sr, text) in enumerate(held):
            # Identity is always scored against a DIFFERENT real recording of the
            # same speaker. The targeted output literally contains this utterance's
            # audio, so scoring against it would return ~1.0 and mean nothing —
            # the same SIM-r trap as scoring a clone against its own reference.
            other_audio, other_sr, _ = held[(j + 1) % len(held)]
            e_other = enc(other_audio, other_sr) if len(held) > 1 else None

            # word_timestamps defaults to False, and without it speechlens returns
            # no per-word scores at all — correctly refusing to invent them, which
            # would have silently skipped every utterance in this study.
            # (array, sr) tuple, not a bare array: analyze() routes tuples through
            # from_array and everything else through load_audio, which hands a
            # numpy array straight to soundfile and raises.
            res = lens.analyze((audio, sr), language="en", cfg=cfg,
                               policy="auto")
            # AnalysisResult exposes dicts, not objects: the scored words live in
            # transcript["segments"][i]["words"], and `reliability` is a coverage
            # summary rather than the per-word scores.
            words = [w for s in res.transcript["segments"]
                     for w in (s.get("words") or [])]
            if not words or "reliability" not in words[0]:
                print(f"  no per-word reliability "
                      f"({res.reliability or 'no policy selected'}); skipping")
                continue

            spans = select_spans(words, threshold=args.threshold,
                                 pad_words=args.pad_words)
            cost = replacement_fraction(spans, total_words=max(len(words), 1),
                                        total_duration_s=len(audio) / sr)

            row = {"speaker": spk, "words": len(text.split()),
                   "flagged_spans": len(spans),
                   "word_fraction": cost["word_fraction"],
                   "duration_fraction": cost["duration_fraction"],
                   "splice_points": cost["splice_points"],
                   "wer_real": wer(text, transcribe(audio, sr))}

            if spans:
                reps = []
                for s in spans:
                    span_text = reference_words_for_span(text, words,
                                                         s.start_word, s.end_word)
                    if not span_text:
                        continue
                    g = golden_speech(tts, vc, apply_case(span_text, args.text_case),
                                      native_ref, native_sr, tmp.name,
                                      seed=args.seed)
                    reps.append((s.start_s, s.end_s, g.audio, g.sample_rate))
                targeted = splice(audio, sr, reps) if reps else audio
            else:
                targeted = audio      # nothing flagged: the learner keeps their audio

            gw = golden_speech(tts, vc, apply_case(text, args.text_case),
                               native_ref, native_sr, tmp.name, seed=args.seed)

            row["wer_targeted"] = wer(text, transcribe(targeted, sr))
            row["wer_golden_whole"] = wer(text, transcribe(gw.audio, gw.sample_rate))
            if e_other is not None:
                row["sim_targeted"] = cosine(enc(targeted, sr), e_other)
                row["sim_golden_whole"] = cosine(enc(gw.audio, gw.sample_rate), e_other)
                row["sim_real"] = cosine(enc(audio, sr), e_other)
            rows.append(row)

    if args.per_utterance:
        Path(args.per_utterance).write_text(json.dumps(rows, indent=1))
        print(f"\nper-utterance rows -> {args.per_utterance}")

    def m(k):
        v = [r[k] for r in rows if k in r and r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 70)
    print(f"TARGETED vs WHOLE-UTTERANCE CORRECTION — n={len(rows)}")
    print("=" * 70)
    print(f"{'arm':>16} {'SIM to learner':>15} {'WER':>8}")
    print(f"{'learner real':>16} {m('sim_real'):>15.3f} {m('wer_real'):>8.3f}"
          f"   <- identity ceiling")
    print(f"{'golden whole':>16} {m('sim_golden_whole'):>15.3f} "
          f"{m('wer_golden_whole'):>8.3f}")
    print(f"{'golden targeted':>16} {m('sim_targeted'):>15.3f} "
          f"{m('wer_targeted'):>8.3f}")
    print(f"\nregenerated: {m('word_fraction'):.1%} of words, "
          f"{m('duration_fraction'):.1%} of audio, "
          f"{m('splice_points'):.1f} seams per utterance")
    print(f"utterances with nothing flagged: "
          f"{sum(1 for r in rows if r['flagged_spans'] == 0)}/{len(rows)}")
    print("\nWER on the targeted arm is near-circular — flagged audio is replaced "
          "with\nspeech synthesized from the reference transcript. Read it only as "
          "a check\nthat splicing did not damage the surrounding audio.")


if __name__ == "__main__":
    main()
