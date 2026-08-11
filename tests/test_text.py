"""Text presentation, which turned out to be worth 6.6x WER on one backend."""
import pytest

from speechforge.text import apply_case, sentence_case


def test_corpus_caps_become_prose():
    assert (sentence_case("STUFF IT INTO YOU HIS BELLY COUNSELLED HIM")
            == "Stuff it into you his belly counselled him.")


def test_already_cased_prose_keeps_its_capitals():
    # Re-casing text a corpus already punctuated would destroy proper nouns and
    # acronyms — "Dr. Watson from NASA" must not become "Dr. watson from nasa".
    assert (sentence_case("Dr. Watson from NASA arrived")
            == "Dr. Watson from NASA arrived.")


def test_existing_terminal_punctuation_is_not_doubled():
    assert sentence_case("Who is there?") == "Who is there?"
    assert sentence_case("HELLO THERE.") == "Hello there."


def test_empty_and_whitespace_survive():
    assert sentence_case("") == ""
    assert sentence_case("   ") == ""


def test_raw_mode_changes_nothing():
    # The default must be untouched: every number recorded before this existed
    # was measured on raw corpus text, and silently normalizing would invalidate
    # the comparison without anyone noticing.
    t = "STUFF IT INTO YOU"
    assert apply_case(t, "raw") is t


def test_sentence_mode_dispatches():
    assert apply_case("HELLO WORLD", "sentence") == "Hello world."


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown text case mode"):
        apply_case("hi", "titlecase")


def test_scoring_is_unaffected_by_the_choice():
    # The whole design rests on this: changing input case cannot flatter a system
    # through the metric, because WER normalizes both sides anyway.
    pytest.importorskip("speechlens")
    from speechlens.metrics import wer
    ref = "STUFF IT INTO YOU HIS BELLY COUNSELLED HIM"
    assert wer(ref, sentence_case(ref)) == 0.0
