"""Which words actually need correcting, and how much audio that costs.

`golden.py` regenerates a whole utterance. A pronunciation tutor should not: the
learner mispronounced two words out of thirty, and replacing all thirty throws away
their prosody, their pacing, and twenty-eight words of their own correct speech —
while giving the identity-transfer stage twenty-eight more chances to drift.

The selector is already built and validated in the other project: speechlens
produces **calibrated per-word reliability** with risk-targeted thresholds. A word
the ASR is confident about was pronounced clearly enough to recognise; a word it is
not confident about is the candidate for correction.

What this module deliberately does NOT claim
--------------------------------------------
Low ASR confidence is **not** a mispronunciation detector. It marks words that were
hard to recognise, and the causes are entangled: mispronunciation, yes, but also
background noise, clipping, disfluency, an unusual name, or a word the language
model found improbable in context. Phase 2 of this project measured the related
negative — confidence does not separate synthetic from real speech — which is a
direct reminder that this signal means less than its name suggests.

So the honest framing is *triage*, not diagnosis: it narrows thirty words to two
candidates, and the measurable question is whether correcting only those beats
correcting everything on the trade-off already established — same intelligibility,
less identity drift, because less audio was regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Span:
    """A contiguous run of words selected for correction, with its audio bounds."""
    start_word: int
    end_word: int          # exclusive
    start_s: float
    end_s: float
    words: List[str] = field(default_factory=list)
    reliabilities: List[float] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {"start_word": self.start_word, "end_word": self.end_word,
                "start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3),
                "duration_s": round(self.duration_s, 3),
                "words": list(self.words),
                "min_reliability": (round(min(self.reliabilities), 3)
                                    if self.reliabilities else None)}


def select_spans(words: Sequence[dict], threshold: float,
                 pad_words: int = 0,
                 merge_gap_words: int = 1) -> List[Span]:
    """Pick the low-reliability words, as spans rather than isolated words.

    ``words`` are speechlens per-word dicts: ``word``, ``start``, ``end`` and a
    reliability score under ``reliability`` (calibrated) or ``prob`` (raw).

    Spans, not words, because splicing single words back into running speech is
    where concatenative synthesis audibly fails — a replaced word carries its own
    onset and offset, and the seams land mid-phrase. Neighbouring selections are
    merged across small gaps and optionally padded, so the cut points fall between
    words that were not themselves suspect.

    ``merge_gap_words=1`` merges "bad good bad" into one span: three words replaced
    as a unit beat two splices with a single good word marooned between them.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    def score(w):
        v = w.get("reliability", w.get("prob"))
        if v is None:
            raise KeyError("word has neither 'reliability' nor 'prob'; the "
                           "selector needs a per-word score")
        return float(v)

    flagged = [i for i, w in enumerate(words) if score(w) < threshold]
    if not flagged:
        return []

    groups: List[List[int]] = [[flagged[0]]]
    for i in flagged[1:]:
        # +1 is adjacency; the gap allowance is on top of that.
        if i - groups[-1][-1] <= merge_gap_words + 1:
            groups[-1].append(i)
        else:
            groups.append([i])

    spans = []
    for g in groups:
        lo = max(0, g[0] - pad_words)
        hi = min(len(words), g[-1] + 1 + pad_words)
        chunk = list(words[lo:hi])
        spans.append(Span(start_word=lo, end_word=hi,
                          start_s=float(chunk[0]["start"]),
                          end_s=float(chunk[-1]["end"]),
                          words=[w["word"] for w in chunk],
                          reliabilities=[score(w) for w in chunk]))
    # Padding can make neighbours overlap; overlapping spans would double-replace
    # the same audio.
    return _merge_overlaps(spans)


def _merge_overlaps(spans: List[Span]) -> List[Span]:
    out: List[Span] = []
    for s in spans:
        if out and s.start_word <= out[-1].end_word:
            prev = out[-1]
            merged = Span(start_word=prev.start_word,
                          end_word=max(prev.end_word, s.end_word),
                          start_s=prev.start_s, end_s=max(prev.end_s, s.end_s),
                          words=prev.words + s.words[max(0, prev.end_word - s.start_word):],
                          reliabilities=prev.reliabilities + s.reliabilities)
            out[-1] = merged
        else:
            out.append(s)
    return out


def replacement_fraction(spans: Sequence[Span], total_words: int,
                         total_duration_s: float) -> dict:
    """How much of the utterance a targeted correction would regenerate.

    The quantity the whole idea rests on. If targeting replaces 90% of the audio it
    has not earned its complexity over regenerating the utterance outright, and
    that should be visible before any listening test.
    """
    if total_words <= 0:
        raise ValueError("total_words must be positive")
    if total_duration_s <= 0:
        raise ValueError("total_duration_s must be positive")
    w = sum(s.end_word - s.start_word for s in spans)
    d = sum(s.duration_s for s in spans)
    return {"spans": len(spans),
            "words_replaced": w,
            "word_fraction": round(w / total_words, 3),
            "seconds_replaced": round(d, 3),
            "duration_fraction": round(min(d / total_duration_s, 1.0), 3),
            # Stated, not assumed: every span boundary is a splice, and splices
            # are where concatenation artefacts live.
            "splice_points": 2 * len(spans)}
