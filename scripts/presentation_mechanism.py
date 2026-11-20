#!/usr/bin/env python3
"""Where does the text-presentation effect live in a phoneme-input TTS?

Chatterbox and CosyVoice 2 read text; StyleTTS 2 reads espeak phonemes, so case
can reach it only through the grapheme-to-phoneme front end. And espeak does
treat capitals differently: in LibriSpeech test-clean, raw capitals change the
phonemes of 57% of transcripts (scripts/styletts2_checks.py phonemizer). Two
kinds of change dominate — function words losing their weak forms (TO -> tuː
rather than tə), and short words **spelled out as initialisms** (IT -> "I.T.",
US -> "U.S.").

The second kind is an unambiguous error, so this tests a sharp prediction: if
the effect lives in the front end, raw-caps damage should concentrate in the
utterances that contain a spelled-out word.

Spelled out is defined without a word list: a capitalized word whose phonemes,
stress marks aside, equal its letters read one by one, and differ from its
lower-case reading. That catches IT and US, and DR read as "D.R." instead of
"doctor", without flagging TO or HAS.

Reads the per-utterance rows of a text_presentation.py run. Those rows carry no
text, so the corpus is reloaded in the same order and the word counts are
checked against the rows before anything is attributed.

    python scripts/presentation_mechanism.py runs/present_styletts2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.backends.styletts2 import phonemize  # noqa: E402

_MARKS = "ˈˌ. "


def _bare(p: str) -> str:
    return "".join(ch for ch in p if ch not in _MARKS)


def spelled_words(words):
    """The subset of ``words`` espeak reads letter by letter in capitals."""
    import re
    cand = sorted({w for w in words if re.fullmatch(r"[A-Z]{2,4}", w)})
    if not cand:
        return set()
    caps = phonemize(cand)
    low = phonemize([w.lower() for w in cand])
    letters = phonemize([".".join(w) + "." for w in cand])
    return {w for w, c, lo, le in zip(cand, caps, low, letters)
            if _bare(c) == _bare(le) and _bare(c) != _bare(lo)}


def sign_test(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rows", help="per-utterance JSON from text_presentation.py")
    p.add_argument("--corpus", default="librispeech")
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    args = p.parse_args()

    rows = json.loads(Path(args.rows).read_text())
    speakers = corpora.load(args.corpus, args.speakers, args.utterances)
    texts = [(spk, t) for spk, utts in speakers.items() for _a, _s, t in utts[1:]]
    if len(texts) != len(rows):
        raise SystemExit(f"{len(rows)} rows but {len(texts)} utterances reloaded")
    for (spk, t), r in zip(texts, rows):
        if r["speaker"] != spk or r["words"] != len(t.split()):
            raise SystemExit(f"row/utterance misalignment at speaker {spk}: "
                             f"{r['words']} words in the row, {len(t.split())} reloaded")
    print(f"aligned {len(rows)} rows to their transcripts "
          f"(speaker and word count match on every row)\n")

    vocab = {w for _s, t in texts for w in t.split()}
    spelled = spelled_words(vocab)
    print(f"words espeak spells out in capitals, in these {len(rows)} transcripts: "
          f"{sorted(spelled) or 'none'}")
    for (_s, t), r in zip(texts, rows):
        r["n_spelled"] = sum(1 for w in t.split() if w in spelled)

    def report(name, sub):
        if not sub:
            print(f"{name:>22}  n=0")
            return
        raw = np.mean([r["wer_raw"] for r in sub])
        sen = np.mean([r["wer_sentence"] for r in sub])
        better = sum(r["delta"] < 0 for r in sub)
        worse = sum(r["delta"] > 0 for r in sub)
        print(f"{name:>22}  n={len(sub):<3} raw {raw:.3f}  sentence {sen:.3f}  "
              f"delta {sen - raw:+.3f}  better/worse {better}/{worse}  "
              f"sign p={sign_test(better, worse):.3g}")

    print("\nmean per-utterance WER, split by whether the raw text contains a "
          "spelled-out word:")
    report("has a spelled word", [r for r in rows if r["n_spelled"] > 0])
    report("no spelled word", [r for r in rows if r["n_spelled"] == 0])
    report("all", rows)
    ns = np.array([r["n_spelled"] for r in rows], float)
    d = np.array([r["delta"] for r in rows], float)
    if ns.std() > 0 and d.std() > 0:
        print(f"\ncorr(spelled words, sentence - raw WER) = {np.corrcoef(ns, d)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
