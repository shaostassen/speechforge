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
