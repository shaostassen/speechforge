"""The evaluation harness, including the guards that stop it flattering a system.

Fakes stand in for the ASR and the speaker encoder, so this runs offline with no
weights. What is being tested is mostly *refusals* — the table declining to
render without ground truth, and SIM declining to compare a clone against the
clip it was conditioned on. Those are the load-bearing parts.
"""
import math

import numpy as np
import pytest

from speechforge.evaluate import (EvalTable, GroundTruthMissing,
                                  ScorerUnavailable, SystemScores, cosine,
                                  require_scorer, score_intelligibility, sim_o)


# --- cosine -----------------------------------------------------------------

def test_identical_vectors_are_one():
    v = [0.1, 0.4, -0.2, 0.9]
    assert cosine(v, v) == pytest.approx(1.0)


def test_opposite_vectors_are_minus_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_scale_invariant():
    a, b = [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
    assert cosine(a, b) == pytest.approx(1.0)


def test_zero_vector_is_nan_not_zero():
    # A zero embedding means the encoder failed. Returning 0.0 would read as
    # "completely different speakers" — a broken measurement posing as a result.
    assert math.isnan(cosine([0.0, 0.0], [1.0, 1.0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


# --- SIM-o guard ------------------------------------------------------------

def _embed_by_frequency(audio, sr):
    """Fake encoder: embeds the dominant frequency, so 'same speaker' is testable."""
    spec = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    hz = float(np.fft.rfftfreq(len(audio), 1 / sr)[int(np.argmax(spec))])
    return np.array([hz, hz ** 0.5, 1.0])


def _tone(hz, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_sim_o_high_for_the_same_voice():
    synth, held = _tone(200.0), _tone(200.0, seconds=1.5)
    assert sim_o(_embed_by_frequency, synth, 16000, held, 16000) > 0.99


def test_sim_o_lower_for_a_different_voice():
    synth, held = _tone(200.0), _tone(600.0, seconds=1.5)
    same = sim_o(_embed_by_frequency, synth, 16000, _tone(200.0, 1.5), 16000)
    diff = sim_o(_embed_by_frequency, synth, 16000, held, 16000)
    assert diff < same


def test_comparing_against_the_reference_clip_raises():
    # SIM-r inflation: measuring how well the model copied one specific
    # recording rather than whether it captured the speaker.
    ref = _tone(200.0, seconds=1.2)
    with pytest.raises(ValueError, match="SIM-r"):
        sim_o(_embed_by_frequency, _tone(200.0), 16000, ref, 16000,
              reference_audio=ref)


def test_held_out_clip_that_merely_resembles_the_reference_is_allowed():
    # The guard must key on being the *same* array, not on sounding similar,
    # or it would reject every legitimate same-speaker comparison.
    ref = _tone(200.0, seconds=1.2)
    held = _tone(200.0, seconds=1.4)
    assert sim_o(_embed_by_frequency, _tone(200.0), 16000, held, 16000,
                 reference_audio=ref) > 0.99


# --- intelligibility --------------------------------------------------------
# WER comes from speechlens rather than being reimplemented here, so only these
# three need the `eval` extra. A module-level importorskip would skip the whole
# file, including the cosine and table tests that need nothing — so the marker
# is applied per test.
try:
    import speechlens.metrics  # noqa: F401
    HAVE_SPEECHLENS = True
except ImportError:
    HAVE_SPEECHLENS = False

needs_speechlens = pytest.mark.skipif(
    not HAVE_SPEECHLENS, reason="needs the [eval] extra (speechlens)")


@needs_speechlens
def test_perfect_transcription_gives_zero_wer():
    pairs = [(_tone(200.0), 16000, "hello world"),
             (_tone(300.0), 16000, "the quick brown fox")]

    # Keyed on call order, not on audio length: both tones are 1 s at 16 kHz,
    # so an earlier version of this fake keyed on len(audio), matched both, and
    # returned the same text twice.
    texts = iter([p[2] for p in pairs])
    score = score_intelligibility(lambda a, s: next(texts), pairs)
    assert score == pytest.approx(0.0)


@needs_speechlens
def test_wer_is_computed_on_the_concatenation():
    # Per-utterance averaging over-weights short utterances; published TTS
    # numbers use the corpus-level figure.
    pairs = [(_tone(200.0), 16000, "one"),
             (_tone(200.0), 16000, "two three four five six seven eight")]
    # Miss the single-word utterance entirely, get the long one right.
    def transcribe(a, s):
        transcribe.n += 1
        return "" if transcribe.n == 1 else "two three four five six seven eight"
    transcribe.n = 0
    score = score_intelligibility(transcribe, pairs)
    assert score == pytest.approx(1 / 8)      # 1 error out of 8 reference words


@needs_speechlens
def test_empty_input_is_nan():
    assert math.isnan(score_intelligibility(lambda a, s: "", []))


# --- the table --------------------------------------------------------------

def test_render_without_ground_truth_raises():
    t = EvalTable()
    t.add(SystemScores("some-model", wer=0.03, sim_o=0.66, n_utterances=100))
    with pytest.raises(GroundTruthMissing, match="ground-truth"):
        t.render()


def test_ground_truth_row_is_marked_in_the_output():
    t = EvalTable(corpus="librispeech", hardware="9950X")
    t.set_ground_truth(SystemScores("ground truth", wer=0.021, sim_o=0.69,
                                    n_utterances=100))
    t.add(SystemScores("chatterbox", wer=0.034, sim_o=0.66, n_utterances=100))
    out = t.render()
    assert "read every row against this" in out
    assert "ground truth" in out and "chatterbox" in out
    assert "librispeech" in out and "9950X" in out


def test_wer_below_human_is_called_out():
    # The trap this whole module exists to prevent.
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.021, n_utterances=50))
    t.add(SystemScores("suspiciously-good", wer=0.014, n_utterances=50))
    out = t.render()
    assert "not an intelligibility win" in out


def test_wer_above_human_is_not_flagged():
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.021, n_utterances=50))
    t.add(SystemScores("normal", wer=0.034, n_utterances=50))
    assert "not an intelligibility win" not in t.render()


def test_missing_metrics_render_as_dashes_not_zeros():
    # A missing measurement must not look like a measured zero.
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.021, n_utterances=10))
    t.add(SystemScores("partial", wer=0.03, n_utterances=10))   # no sim, no mos
    out = t.render()
    assert "—" in out
    assert "0.000" not in out


def test_sim_above_ground_truth_is_flagged():
    # A clone matching the speaker better than their own other recordings do
    # means the comparison is wrong — almost always SIM-r against the reference.
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, sim_o=0.90,
                                    n_utterances=10))
    t.add(SystemScores("too-good", wer=0.03, sim_o=0.95, n_utterances=10))
    assert "SIM above ground truth" in t.render()


def test_sim_below_ground_truth_is_not_flagged():
    # The published 0.69-0.76 ceiling must NOT be used as a threshold: with this
    # encoder legitimate same-speaker scores run 0.89-0.98, so warning on 0.73
    # would fire on every real run and become noise.
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, sim_o=0.979,
                                    n_utterances=10))
    t.add(SystemScores("legit", wer=0.03, sim_o=0.927, n_utterances=10))
    out = t.render()
    assert "SIM above ground truth" not in out
    assert "encoder-specific" in out


def test_mos_is_labelled_as_non_ranking():
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, mos=4.1,
                                    n_utterances=10))
    t.add(SystemScores("m", wer=0.03, mos=4.3, n_utterances=10))
    out = t.render()
    assert "cannot rank" in out


def test_scores_are_json_ready():
    import json
    json.dumps(SystemScores("x", wer=0.03, sim_o=0.66).to_dict())


# --- naturalness ------------------------------------------------------------

def test_score_naturalness_averages_the_clips():
    from speechforge.evaluate import score_naturalness
    clips = [(_tone(200.0), 16000), (_tone(300.0), 16000)]
    scores = iter([4.0, 3.0])
    assert score_naturalness(lambda a, s: next(scores), clips) == pytest.approx(3.5)


def test_score_naturalness_is_nan_with_no_clips():
    from speechforge.evaluate import score_naturalness
    assert math.isnan(score_naturalness(lambda a, s: 4.0, []))


def test_mos_column_still_renders_when_unavailable():
    # The predictor needs torch and a hub download. An unavailable predictor has
    # to leave a blank column, never a zero — a 0.0 MOS would read as "terrible
    # audio" rather than "not measured".
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, mos=None,
                                    n_utterances=10))
    t.add(SystemScores("m", wer=0.03, mos=None, n_utterances=10))
    out = t.render()
    assert "—" in out and "0.00" not in out


# --- a requested scorer is measured or the run stops -------------------------

def test_require_scorer_returns_what_loaded():
    sentinel = object()
    assert require_scorer("speaker encoder", lambda: sentinel, "pass --no-sim") is sentinel


def test_require_scorer_refuses_rather_than_blanking():
    # The old behaviour printed a warning and returned nothing, and the table
    # still rendered — with an empty column that looked like a finished run.
    def blocked():
        raise OSError("403 from cas-server.example")

    with pytest.raises(ScorerUnavailable) as err:
        require_scorer("speaker encoder", blocked, "pass --no-sim")
    msg = str(err.value)
    assert "speaker encoder" in msg            # which scorer
    assert "OSError" in msg and "403" in msg   # why
    assert "pass --no-sim" in msg              # how to run without it on purpose
    assert isinstance(err.value.__cause__, OSError)


def test_require_scorer_catches_import_failures_too():
    # A missing extra surfaces as ImportError at load time; that is the same
    # blank column and gets the same refusal.
    def missing():
        raise ImportError("No module named 'torch'")

    with pytest.raises(ScorerUnavailable, match="ImportError"):
        require_scorer("second ASR", missing, "drop --second-asr")


# --- the prosody block ----------------------------------------------------------

def _table(**synth):
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, n_utterances=10,
                                    pitch_range_st=6.0, pitch_gap_st=0.8,
                                    words_per_s=2.7, jump_fraction=0.001))
    t.add(SystemScores("m", wer=0.03, n_utterances=10, **synth))
    return t


def test_no_prosody_block_unless_prosody_was_measured():
    # Every table rendered before prosody existed must render as it did.
    t = EvalTable()
    t.set_ground_truth(SystemScores("ground truth", wer=0.02, n_utterances=10))
    t.add(SystemScores("m", wer=0.03, n_utterances=10))
    assert "pitch range" not in t.render()


def test_prosody_block_reads_against_the_ground_truth_row():
    out = _table(pitch_range_st=4.5, pitch_gap_st=1.2, words_per_s=2.9,
                 jump_fraction=0.002).render()
    block = out[out.index("pitch range"):]
    assert "6.00" in block and "4.50" in block      # human row and system row
    assert block.index("ground truth") < block.index("4.50")
    assert "read every row against this" in block
    assert "unreliable" not in block


def test_prosody_flags_a_broken_pitch_track():
    out = _table(pitch_range_st=9.0, pitch_gap_st=1.0, words_per_s=2.8,
                 jump_fraction=0.08).render()
    assert "pitch tracking unreliable" in out


def test_prosody_to_dict_round_trips_the_fields():
    d = SystemScores("m", pitch_range_st=5.0, words_per_s=3.0).to_dict()
    assert d["pitch_range_st"] == 5.0 and d["words_per_s"] == 3.0
    assert d["pitch_gap_st"] is None

