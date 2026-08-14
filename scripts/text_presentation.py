#!/usr/bin/env python3
"""Does the *shape* of the input text change how well a TTS is understood?

The audit behind the largest correction in this project. LibriSpeech and EdAcc
distribute transcripts as unpunctuated UPPER CASE — a convention for *scoring* —
and evaluation scripts routinely hand that straight to a generative model as
input. Measured on LibriSpeech test-clean, that cost Chatterbox 0.090 WER and
CosyVoice 2 0.213, so "give every model the same text" is not the neutral choice
it appears to be: identical treatment, unequal handicap.

Design, and why each part is there:

  **Paired.** The same utterance, the same reference, the same seed, synthesized
  once per presentation. Only the case of the input string differs, so a
  per-utterance difference cannot be sampling noise or utterance difficulty.

  **A real-audio row.** Real speech does not read text, so its WER must be
  identical across arms. It is the control that proves the mechanism rather than
  assuming it — if that row moves, something other than presentation changed.

  **Duration ratio.** The failure mode is not uniform degradation, it is
  autoregressive over-generation: on CosyVoice 2, corr(duration ratio, WER) was
  +0.861 with outputs up to 2.42x the expected length. A WER-only report would
  hide the mechanism.

  **Per-utterance output.** Aggregates have hidden the truth four separate times
  in this project; the rows are written before anything is summarized.

    python scripts/text_presentation.py --backend chatterbox --corpus librispeech
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import whisper_transcriber  # noqa: E402
from speechforge.synth import clone, load_backend  # noqa: E402
from speechforge.text import apply_case  # noqa: E402

PRESENTATIONS = ("raw", "sentence")


def hardware_label() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def summarize(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="chatterbox")
    p.add_argument("--corpus", default="librispeech",
                   choices=["edacc", "librispeech", "librispeech_dummy"])
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--scan", type=int, default=9000)
    p.add_argument("--ref-seconds", type=float, default=None,
                   help="reference cap. Leave unset to use the backend's own "
                        "measured preference — a borrowed cap has already "
                        "produced a spurious 5.9x WER difference here")
    p.add_argument("--per-utterance", metavar="PATH")
    args = p.parse_args()

    print(f"study=text_presentation  hardware={hardware_label()}  "
          f"backend={args.backend}  corpus={args.corpus}  "
          f"asr={args.asr_model}  seed={args.seed}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    from speechlens.metrics import wer

    speakers = corpora.load(args.corpus, args.speakers, args.utterances,
                            **({"scan": args.scan}
                               if args.corpus == "edacc" else {}))
    print(f"speakers: {list(speakers)}\n")

    backend = load_backend(args.backend, device="cpu")
    backend.set_threads(args.threads)
    print(f"backend: {backend.describe()}  sr={backend.sample_rate}  "
          f"needs_reference_text={getattr(backend, 'needs_reference_text', False)}")
    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")
    print()

    rows = []
    for spk, utts in speakers.items():
        ref_audio, ref_sr, ref_text = utts[0]
        held = utts[1:]
        print(f"{spk}: {len(held)} held-out", flush=True)

        for ho_audio, ho_sr, text in held:
            row = {"speaker": spk, "words": len(text.split()),
                   # Presentation-independent by construction: the control.
                   "wer_real": wer(text, transcribe(ho_audio, ho_sr)),
                   "real_audio_s": len(ho_audio) / ho_sr}
            for mode in PRESENTATIONS:
                wav, sr, _ = clone(
                    backend, apply_case(text, mode), ref_audio, ref_sr,
                    seed=None if args.seed < 0 else args.seed,
                    ref_seconds=args.ref_seconds,
                    # The reference transcript gets the same treatment as the
                    # synthesis text; presenting them differently would confound
                    # the very thing under test.
                    reference_text=apply_case(ref_text, mode))
                # Scored against the ORIGINAL text either way. WER normalizes
                # case and punctuation, so presentation cannot flatter the metric.
                row[f"wer_{mode}"] = wer(text, transcribe(wav, sr))
                row[f"dur_ratio_{mode}"] = (len(wav) / sr) / (len(ho_audio) / ho_sr)
            row["delta"] = row["wer_sentence"] - row["wer_raw"]
            rows.append(row)

    if args.per_utterance:
        Path(args.per_utterance).write_text(json.dumps(rows, indent=1))
        print(f"\nper-utterance rows -> {args.per_utterance}")

    print("\n" + "=" * 66)
    print(f"TEXT PRESENTATION — {args.backend} on {args.corpus}, n={len(rows)}")
    print("=" * 66)
    print(f"{'presentation':>14} {'WER':>8} {'dur ratio':>11}")
    print(f"{'real audio':>14} {summarize(rows, 'wer_real'):>8.3f} "
          f"{1.0:>11.2f}   <- must not move between arms")
    for mode in PRESENTATIONS:
        print(f"{mode:>14} {summarize(rows, f'wer_{mode}'):>8.3f} "
              f"{summarize(rows, f'dur_ratio_{mode}'):>11.2f}")

    d = summarize(rows, "delta")
    better = sum(1 for r in rows if r["delta"] < 0)
    worse = sum(1 for r in rows if r["delta"] > 0)
    raw, sen = summarize(rows, "wer_raw"), summarize(rows, "wer_sentence")
    print(f"\nsentence case − raw caps: {d:+.3f} WER"
          + (f"  ({raw / sen:.1f}x better)" if sen > 0 and raw > sen else ""))
    print(f"improved on {better}/{len(rows)} utterances, worse on {worse}")
    print("\nThe real-audio row is the control: it reads no text, so if it "
          "differs\nbetween arms the difference is not presentation.")


if __name__ == "__main__":
    main()
