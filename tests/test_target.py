"""Targeted correction: which words get replaced, and what that costs."""
import pytest

from speechforge.target import (Span, replacement_fraction, select_spans)


def _words(scores, dur=0.4):
    """Evenly spaced words carrying the given reliability scores."""
    return [{"word": f"w{i}", "start": i * dur, "end": (i + 1) * dur,
             "reliability": s} for i, s in enumerate(scores)]


def test_nothing_selected_when_every_word_is_reliable():
    assert select_spans(_words([0.9, 0.95, 0.99]), threshold=0.5) == []


def test_a_single_bad_word_becomes_one_span():
    spans = select_spans(_words([0.9, 0.2, 0.9]), threshold=0.5)
    assert len(spans) == 1
    assert spans[0].start_word == 1 and spans[0].end_word == 2
    assert spans[0].words == ["w1"]


def test_adjacent_bad_words_merge_into_one_span():
    spans = select_spans(_words([0.9, 0.2, 0.1, 0.9]), threshold=0.5)
    assert len(spans) == 1
    assert (spans[0].start_word, spans[0].end_word) == (1, 3)


def test_a_lone_good_word_between_two_bad_ones_is_absorbed():
    # "bad good bad" — replacing three words as a unit beats two splices with a
    # single good word marooned between them, because every boundary is a seam.
    spans = select_spans(_words([0.9, 0.2, 0.8, 0.2, 0.9]), threshold=0.5,
                         merge_gap_words=1)
    assert len(spans) == 1
    assert (spans[0].start_word, spans[0].end_word) == (1, 4)


def test_distant_bad_words_stay_separate():
    spans = select_spans(_words([0.2, 0.9, 0.9, 0.9, 0.2]), threshold=0.5,
                         merge_gap_words=1)
    assert len(spans) == 2


def test_padding_does_not_produce_overlapping_spans():
    # Padding is what makes cut points land on unsuspect words, but it can push
    # neighbours into each other — and overlapping spans would replace the same
    # audio twice.
    spans = select_spans(_words([0.2, 0.9, 0.9, 0.2]), threshold=0.5,
                         pad_words=1, merge_gap_words=0)
    assert len(spans) == 1
    for a, b in zip(spans, spans[1:]):
        assert a.end_word <= b.start_word


def test_padding_is_clamped_at_the_utterance_edges():
    spans = select_spans(_words([0.2, 0.9]), threshold=0.5, pad_words=5)
    assert spans[0].start_word == 0 and spans[0].end_word == 2


def test_raw_prob_is_accepted_when_calibrated_reliability_is_absent():
    words = [{"word": "a", "start": 0.0, "end": 0.4, "prob": 0.2}]
    assert len(select_spans(words, threshold=0.5)) == 1


def test_a_word_with_no_score_is_refused():
    words = [{"word": "a", "start": 0.0, "end": 0.4}]
    with pytest.raises(KeyError, match="reliability"):
        select_spans(words, threshold=0.5)


def test_threshold_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_spans(_words([0.5]), threshold=1.5)


def test_span_times_follow_the_selected_words():
    spans = select_spans(_words([0.9, 0.2, 0.2, 0.9], dur=0.5), threshold=0.5)
    assert spans[0].start_s == pytest.approx(0.5)
    assert spans[0].end_s == pytest.approx(1.5)
    assert spans[0].duration_s == pytest.approx(1.0)


# --- the cost of targeting ---------------------------------------------------

def test_replacement_fraction_reports_what_targeting_actually_saves():
    spans = select_spans(_words([0.9] * 28 + [0.2, 0.2]), threshold=0.5)
    out = replacement_fraction(spans, total_words=30, total_duration_s=12.0)
    assert out["words_replaced"] == 2
    assert out["word_fraction"] == pytest.approx(2 / 30, abs=1e-3)
    assert out["spans"] == 1


def test_splice_points_are_reported_because_seams_are_the_risk():
    # Two spans mean four boundaries, and concatenation artefacts live exactly
    # there. A caller choosing between targeted and whole-utterance correction
    # needs this number, not just the fraction replaced.
    spans = select_spans(_words([0.2, 0.9, 0.9, 0.9, 0.2]), threshold=0.5)
    assert replacement_fraction(spans, 5, 2.0)["splice_points"] == 4


def test_duration_fraction_cannot_exceed_one():
    # Word timings come from an ASR and can overlap or overrun; a fraction above
    # 1.0 would read as "replaced more than all of it".
    spans = [Span(0, 2, 0.0, 30.0)]
    assert replacement_fraction(spans, 2, 10.0)["duration_fraction"] == 1.0


def test_degenerate_totals_are_refused():
    with pytest.raises(ValueError):
        replacement_fraction([], total_words=0, total_duration_s=1.0)
    with pytest.raises(ValueError):
        replacement_fraction([], total_words=1, total_duration_s=0.0)


# --- splicing corrected audio back in ---------------------------------------

import numpy as np  # noqa: E402

from speechforge.target import splice  # noqa: E402


def test_replaced_region_carries_the_new_audio():
    base = np.full(16000, 0.5, dtype=np.float32)
    rep = np.full(4000, -0.5, dtype=np.float32)
    out = splice(base, 16000, [(0.25, 0.5, rep, 16000)], crossfade_ms=0.0)
    # Middle of the replacement, clear of any boundary.
    assert out[4000 + 2000] == pytest.approx(-0.5, abs=1e-5)


def test_audio_outside_the_span_is_untouched():
    base = np.full(16000, 0.5, dtype=np.float32)
    out = splice(base, 16000, [(0.5, 0.75, np.zeros(100, np.float32), 16000)],
                 crossfade_ms=0.0)
    assert out[:8000] == pytest.approx(np.full(8000, 0.5), abs=1e-6)


def test_no_click_at_the_seam():
    # The failure this exists to prevent: butt-joining two waveforms mid-cycle is
    # a step discontinuity, and a repair tool that clicks at every repair is worse
    # than one that repairs nothing.
    base = np.full(16000, 1.0, dtype=np.float32)
    rep = np.full(4000, -1.0, dtype=np.float32)
    smooth = splice(base, 16000, [(0.25, 0.5, rep, 16000)], crossfade_ms=20.0)
    abrupt = splice(base, 16000, [(0.25, 0.5, rep, 16000)], crossfade_ms=0.0)
    assert np.max(np.abs(np.diff(smooth))) < np.max(np.abs(np.diff(abrupt)))


def test_crossfade_never_consumes_a_whole_chunk():
    # A fade longer than the audio it spans would delete the replacement.
    base = np.full(2000, 0.5, dtype=np.float32)
    rep = np.full(20, -0.5, dtype=np.float32)      # 1.25 ms at 16 kHz
    out = splice(base, 16000, [(0.05, 0.06, rep, 16000)], crossfade_ms=500.0)
    assert len(out) > 0 and np.all(np.isfinite(out))


def test_overlapping_replacements_are_refused():
    # Two corrections claiming the same moment has no defined answer.
    base = np.zeros(16000, dtype=np.float32)
    r = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="overlapping"):
        splice(base, 16000, [(0.1, 0.5, r, 16000), (0.3, 0.6, r, 16000)])


def test_replacement_at_a_different_rate_is_resampled():
    # Backends synthesize at their own rate (24 kHz Chatterbox, 22.05 kHz Piper);
    # splicing without resampling would shift pitch and tempo of the repair.
    base = np.zeros(16000, dtype=np.float32)
    rep = np.full(24000, 0.3, dtype=np.float32)     # 1.0 s at 24 kHz
    out = splice(base, 16000, [(0.25, 0.5, rep, 24000)], crossfade_ms=0.0)
    # 0.75 s of base + ~1.0 s of resampled replacement
    assert len(out) / 16000 == pytest.approx(1.75, abs=0.05)


def test_spans_out_of_order_are_handled():
    base = np.full(16000, 0.5, dtype=np.float32)
    a = np.full(100, 0.1, dtype=np.float32)
    b = np.full(100, 0.2, dtype=np.float32)
    out = splice(base, 16000, [(0.6, 0.7, b, 16000), (0.1, 0.2, a, 16000)],
                 crossfade_ms=0.0)
    assert np.all(np.isfinite(out))


def test_bounds_beyond_the_audio_are_clamped():
    base = np.full(1000, 0.5, dtype=np.float32)
    out = splice(base, 16000, [(0.5, 99.0, np.zeros(10, np.float32), 16000)],
                 crossfade_ms=0.0)
    assert np.all(np.isfinite(out))


def test_no_replacements_returns_the_original():
    base = np.full(1000, 0.5, dtype=np.float32)
    assert splice(base, 16000, []) == pytest.approx(base, abs=1e-6)


# --- mapping ASR spans back to the words the learner meant to say ------------

def _ref_words_for_span(*a, **k):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_targeted",
        str(__import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts" / "eval_targeted.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_words_for_span(*a, **k)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("speechlens") is None,
    reason="needs speechlens for Levenshtein alignment")
def test_correction_speaks_the_reference_word_not_the_misrecognition():
    # The whole point: the span was flagged *because* the ASR misheard it, so
    # synthesizing the ASR's own guess would faithfully reproduce the error.
    ref = "SHE SELLS SEA SHELLS"
    hyp = [{"word": w} for w in ["SHE", "CELLS", "SEA", "SHELLS"]]
    out = _ref_words_for_span(ref, hyp, 1, 2)
    assert out == "SELLS"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("speechlens") is None,
    reason="needs speechlens for Levenshtein alignment")
def test_a_deleted_word_inside_the_span_is_still_spoken():
    # The learner dropped a word entirely. A correction that also drops it has
    # corrected nothing, so the reference slice is taken contiguously.
    ref = "THE QUICK BROWN FOX JUMPS"
    hyp = [{"word": w} for w in ["THE", "QUICK", "FOX", "JUMPS"]]
    out = _ref_words_for_span(ref, hyp, 1, 3)
    assert "BROWN" in out
