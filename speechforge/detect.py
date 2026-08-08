"""Does ASR confidence react to synthetic speech, holding correctness fixed?

The question: when Whisper transcribes a word *correctly*, does it feel
differently confident about a synthesized voice than a real one? Restricting to
correct words is what makes the question sharp — otherwise any difference is
just a restatement of "synthetic speech has higher WER".

Two possible answers, both worth having:

- **Confidence separates them.** Then per-word confidence is a synthetic-speech
  indicator, falling out of machinery that already exists for a different
  purpose, and needing no detector to be trained.
- **It does not.** Then that is a fact about what Whisper's confidence actually
  measures — acoustic-lexical fit rather than anything about provenance.

Either way it quantifies a bias the evaluation harness has to carry: if synthetic
speech is *systematically* easier for an ASR to be confident about, WER-based
intelligibility flatters it, and by how much is measurable rather than assumed.

The extraction here is pure and testable; the model work lives in scripts/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class WordSample:
    """One transcribed word with its confidence and whether it was right."""
    word: str
    confidence: float
    correct: bool
    synthetic: bool


def words_from_result(result_dict: dict) -> List[dict]:
    """Flatten a speechlens AnalysisResult dict to its words, in reading order.

    Returns ``[]`` when word timestamps were not requested, rather than raising:
    a caller sweeping several conditions should get an empty condition, not a
    crash halfway through a long run.
    """
    out: List[dict] = []
    for seg in result_dict.get("transcript", {}).get("segments", []):
        out.extend(seg.get("words") or [])
    return out


def label_words(words: Sequence[dict], reference: str,
                synthetic: bool) -> List[WordSample]:
    """Pair each transcribed word with its correctness against ``reference``.

    Correctness comes from a Levenshtein alignment in speechlens, so a word is
    "correct" only if it survived alignment unchanged — substitutions and
    insertions both count as wrong. Words that normalize to nothing (bare
    punctuation) carry no evidence and are dropped rather than guessed.
    """
    from speechlens.metrics import labels_for_words

    texts = [w["word"] for w in words]
    labels = labels_for_words(reference, texts)
    out: List[WordSample] = []
    for w, lab in zip(words, labels):
        if lab is None:
            continue
        out.append(WordSample(word=w["word"].strip(),
                              confidence=float(w["prob"]),
                              correct=bool(lab), synthetic=synthetic))
    return out


def correct_only(samples: Sequence[WordSample]) -> List[WordSample]:
    """Keep only correctly-transcribed words.

    The whole point of the study: comparing confidence on words the ASR got
    right removes accuracy as an explanation for any difference found.
    """
    return [s for s in samples if s.correct]


def confidence_contrast(samples: Sequence[WordSample]) -> Dict[str, float]:
    """Compare confidence between real and synthetic, on correct words only.

    ``auroc_real_higher`` is the probability that a randomly chosen real word
    scores above a randomly chosen synthetic one. 0.5 means confidence carries
    no information about provenance; far from 0.5 in either direction means it
    does — and *which* direction matters, so it is not folded into an absolute
    value.
    """
    from speechlens.confidence import auroc

    correct = correct_only(samples)
    real = [s.confidence for s in correct if not s.synthetic]
    synth = [s.confidence for s in correct if s.synthetic]
    if not real or not synth:
        return {"n_real": float(len(real)), "n_synth": float(len(synth)),
                "auroc_real_higher": float("nan"),
                "mean_real": float("nan"), "mean_synth": float("nan"),
                "delta": float("nan")}

    scores = real + synth
    is_real = [True] * len(real) + [False] * len(synth)
    mean_real = sum(real) / len(real)
    mean_synth = sum(synth) / len(synth)
    return {
        "n_real": float(len(real)),
        "n_synth": float(len(synth)),
        "auroc_real_higher": auroc(scores, is_real),
        "mean_real": mean_real,
        "mean_synth": mean_synth,
        "delta": mean_real - mean_synth,
    }


def calibrated_contrast(samples: Sequence[WordSample],
                        policy) -> Optional[Dict[str, float]]:
    """The same contrast after calibration, using a speechlens policy.

    Calibration is monotone, so it cannot change the AUROC — that is the point of
    reporting both. What it can change is the *accept rate*: if a policy fitted
    on real speech accepts synthetic words at a different rate, then a reliability
    gate tuned on real audio behaves differently on synthetic audio, which is a
    concrete deployment consequence rather than a curiosity.
    """
    if policy is None:
        return None
    correct = correct_only(samples)
    if not correct:
        return None

    def accept_rate(synthetic: bool) -> float:
        subset = [s for s in correct if s.synthetic is synthetic]
        if not subset:
            return float("nan")
        ok = sum(1 for s in subset
                 if policy.calibrator.predict(s.confidence) >= policy.threshold)
        return ok / len(subset)

    return {"accept_real": accept_rate(False),
            "accept_synth": accept_rate(True)}
