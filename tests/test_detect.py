"""The synthetic-vs-real confidence contrast, on constructed samples.

The model work is in scripts/; what is pinned here is the extraction and the
contrast maths, since a subtle error there would produce a plausible-looking
finding about synthetic speech that is really a bug.
"""
import math

import pytest

from speechforge.detect import (WordSample, calibrated_contrast,
                                confidence_contrast, correct_only,
                                words_from_result)

HAVE_SPEECHLENS = True
try:
    import speechlens.metrics  # noqa: F401
except ImportError:
    HAVE_SPEECHLENS = False

needs_speechlens = pytest.mark.skipif(
    not HAVE_SPEECHLENS, reason="needs the [eval] extra (speechlens)")


def _s(conf, correct=True, synthetic=False, word="w"):
    return WordSample(word=word, confidence=conf, correct=correct,
                      synthetic=synthetic)


# --- flattening -------------------------------------------------------------

def test_words_are_flattened_in_reading_order():
    result = {"transcript": {"segments": [
        {"words": [{"word": " a", "prob": 0.9}, {"word": " b", "prob": 0.8}]},
        {"words": [{"word": " c", "prob": 0.7}]},
    ]}}
    assert [w["word"] for w in words_from_result(result)] == [" a", " b", " c"]


def test_missing_word_timestamps_yields_empty_not_an_error():
    # A sweep over conditions should get an empty condition, not a crash
    # halfway through a long run.
    assert words_from_result({"transcript": {"segments": [{"text": "hi"}]}}) == []
    assert words_from_result({}) == []


# --- filtering --------------------------------------------------------------

def test_correct_only_drops_the_wrong_words():
    samples = [_s(0.9), _s(0.2, correct=False), _s(0.8)]
    assert len(correct_only(samples)) == 2


def test_contrast_uses_only_correct_words():
    # A wrong word with an extreme confidence must not move the result, or the
    # study collapses into a restatement of "synthetic speech has higher WER".
    base = [_s(0.9, synthetic=False), _s(0.9, synthetic=True)]
    with_noise = base + [_s(0.01, correct=False, synthetic=True)]
    assert (confidence_contrast(base)["mean_synth"]
            == pytest.approx(confidence_contrast(with_noise)["mean_synth"]))


# --- the contrast -----------------------------------------------------------

@needs_speechlens
def test_no_difference_gives_chance_auroc():
    samples = [_s(0.9, synthetic=False), _s(0.9, synthetic=True),
               _s(0.8, synthetic=False), _s(0.8, synthetic=True)]
    out = confidence_contrast(samples)
    assert out["auroc_real_higher"] == pytest.approx(0.5)
    assert out["delta"] == pytest.approx(0.0)


@needs_speechlens
def test_real_scoring_higher_pushes_auroc_up():
    samples = [_s(0.95, synthetic=False), _s(0.93, synthetic=False),
               _s(0.60, synthetic=True), _s(0.55, synthetic=True)]
    out = confidence_contrast(samples)
    assert out["auroc_real_higher"] == pytest.approx(1.0)
    assert out["delta"] > 0


@needs_speechlens
def test_synthetic_scoring_higher_pushes_auroc_down():
    # Direction is preserved rather than folded into an absolute value: "ASR is
    # MORE confident on synthetic speech" is a different finding from the
    # reverse, and it is the one that would bias WER-based evaluation.
    samples = [_s(0.50, synthetic=False), _s(0.55, synthetic=False),
               _s(0.95, synthetic=True), _s(0.97, synthetic=True)]
    out = confidence_contrast(samples)
    assert out["auroc_real_higher"] == pytest.approx(0.0)
    assert out["delta"] < 0


@needs_speechlens
def test_one_sided_data_is_nan_not_a_fake_half():
    out = confidence_contrast([_s(0.9, synthetic=False)])
    assert math.isnan(out["auroc_real_higher"])
    assert out["n_synth"] == 0


@needs_speechlens
def test_counts_are_reported():
    samples = [_s(0.9), _s(0.8), _s(0.7, synthetic=True),
               _s(0.1, correct=False, synthetic=True)]
    out = confidence_contrast(samples)
    assert out["n_real"] == 2 and out["n_synth"] == 1


# --- labelling --------------------------------------------------------------

@needs_speechlens
def test_label_words_marks_the_wrong_word():
    from speechforge.detect import label_words
    words = [{"word": " the", "prob": 0.99}, {"word": " cat", "prob": 0.4}]
    out = label_words(words, "the dog", synthetic=True)
    assert [s.correct for s in out] == [True, False]
    assert all(s.synthetic for s in out)


@needs_speechlens
def test_label_words_drops_punctuation_only_tokens():
    from speechforge.detect import label_words
    words = [{"word": " hi", "prob": 0.9}, {"word": " --", "prob": 0.3}]
    out = label_words(words, "hi", synthetic=False)
    assert len(out) == 1 and out[0].word == "hi"


# --- calibrated accept rates ------------------------------------------------

class _FakePolicy:
    """Accepts a calibrated score at or above 0.8; calibration is identity."""
    threshold = 0.8

    class calibrator:
        @staticmethod
        def predict(x):
            return x


def test_accept_rates_are_reported_per_condition():
    samples = [_s(0.9, synthetic=False), _s(0.7, synthetic=False),
               _s(0.9, synthetic=True), _s(0.9, synthetic=True)]
    out = calibrated_contrast(samples, _FakePolicy())
    assert out["accept_real"] == pytest.approx(0.5)
    assert out["accept_synth"] == pytest.approx(1.0)


def test_no_policy_means_no_calibrated_result():
    assert calibrated_contrast([_s(0.9)], None) is None


def test_no_correct_words_means_no_calibrated_result():
    assert calibrated_contrast([_s(0.9, correct=False)], _FakePolicy()) is None
