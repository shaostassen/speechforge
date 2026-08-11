"""Golden-speaker pipeline and its trade-off reporting, with fake models.

The interesting part is the reporting: this pipeline has two failure modes that
look like success if you read one metric, so the tests mostly pin that both get
named.
"""
import numpy as np
import pytest

from speechforge.golden import (CeilingMissing, GoldenResult, blend_rate,
                                golden_speech, golden_table, mean_ignoring_nan,
                                trade_off)


def _speech(seconds=6.0, sr=16000, amp=0.4, f0=180.0):
    t = np.arange(int(seconds * sr)) / sr
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amp * env * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


class FakeTTS:
    sample_rate = 24000
    align_ms = None
    ref_seconds = None

    def __init__(self):
        self.calls = []

    def synthesize(self, text, reference_path, seed=None, reference_text=None):
        self.calls.append({"text": text, "seed": seed})
        t = np.arange(int(2.0 * self.sample_rate)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), self.sample_rate

    def set_threads(self, n): pass

    def describe(self): return "fake-tts"


class FakeVC:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def convert(self, audio, sr, target_ref_path, seed=None):
        self.calls.append({"n": len(audio), "sr": sr,
                           "target": target_ref_path, "seed": seed})
        # Return something distinguishable from the input.
        t = np.arange(int(2.0 * self.sample_rate)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32), self.sample_rate

    def describe(self): return "fake-vc"


# --- pipeline ---------------------------------------------------------------

def test_pipeline_runs_tts_then_conversion(tmp_path):
    ref = tmp_path / "learner.wav"
    import soundfile as sf
    sf.write(str(ref), _speech(5.0), 16000)

    tts, vc = FakeTTS(), FakeVC()
    out = golden_speech(tts, vc, "hello world", _speech(5.0), 16000, str(ref))

    assert tts.calls[0]["text"] == "hello world"
    assert vc.calls[0]["target"] == str(ref)
    assert isinstance(out, GoldenResult)


def test_intermediate_is_kept_not_discarded(tmp_path):
    # The stage-1 audio is the pronunciation baseline. Without it, a WER change
    # in the output cannot be attributed to conversion rather than to synthesis.
    import soundfile as sf
    ref = tmp_path / "learner.wav"
    sf.write(str(ref), _speech(5.0), 16000)

    out = golden_speech(FakeTTS(), FakeVC(), "hi", _speech(5.0), 16000, str(ref))
    assert out.source_audio is not None
    assert out.source_sample_rate == 24000
    assert not np.array_equal(out.audio, out.source_audio)


def test_seed_reaches_both_stochastic_stages(tmp_path):
    # Two sampling stages means two ways for a recorded number to be
    # irreproducible.
    import soundfile as sf
    ref = tmp_path / "learner.wav"
    sf.write(str(ref), _speech(5.0), 16000)

    tts, vc = FakeTTS(), FakeVC()
    golden_speech(tts, vc, "hi", _speech(5.0), 16000, str(ref), seed=99)
    assert tts.calls[0]["seed"] == 99
    assert vc.calls[0]["seed"] == 99


def test_result_is_json_ready(tmp_path):
    import json
    import soundfile as sf
    ref = tmp_path / "learner.wav"
    sf.write(str(ref), _speech(5.0), 16000)

    out = golden_speech(FakeTTS(), FakeVC(), "hi", _speech(5.0), 16000, str(ref))
    d = out.to_dict()
    json.dumps(d)
    assert d["duration_s"] == pytest.approx(2.0, abs=0.1)


# --- trade-off reporting ----------------------------------------------------

def test_no_identity_movement_is_named():
    # Failure mode 1: conversion did nothing. WER looks great precisely because
    # nothing happened, so a WER-only reading would call this a success.
    out = trade_off(sim_to_learner=0.60, sim_source_to_learner=0.90,
                    wer_output=0.10, wer_source=0.10, sim_to_source=0.95,
                    blend_fraction=1.0)
    assert out["identity_gain"] < 0
    assert "did not move identity" in out["verdict"]


def test_damaged_speech_is_named_even_when_identity_moved():
    # Failure mode 2: it sounds like the learner and no longer says the words.
    # A similarity-only reading would call this a success.
    out = trade_off(sim_to_learner=0.92, sim_source_to_learner=0.55,
                    wer_output=0.40, wer_source=0.10, sim_to_source=0.50,
                    blend_fraction=0.0)
    assert out["identity_gain"] > 0
    assert "damaged the speech" in out["verdict"]


def test_blend_is_named_even_though_identity_moved_and_wer_held():
    # Failure mode 3, the one an unmeasured "identity shed" axis hid entirely:
    # the output moved toward the learner AND kept its pronunciation, but still
    # resembles the source voice as much as the target. That is a blend of two
    # people, which sounds like neither, and both other axes call it a success.
    out = trade_off(sim_to_learner=0.82, sim_source_to_learner=0.60,
                    wer_output=0.11, wer_source=0.10, sim_to_source=0.88,
                    blend_fraction=0.4)
    assert out["identity_gain"] > 0
    assert out["wer_cost"] < 0.03
    assert out["resembles_source_more"] is True
    assert "blend" in out["verdict"]


def test_success_requires_all_three_axes():
    out = trade_off(sim_to_learner=0.90, sim_source_to_learner=0.60,
                    wer_output=0.11, wer_source=0.10, sim_to_source=0.55,
                    blend_fraction=0.0)
    assert "largely intact" in out["verdict"]


def test_modest_cost_is_distinguished_from_damage():
    out = trade_off(sim_to_learner=0.90, sim_source_to_learner=0.60,
                    wer_output=0.16, wer_source=0.10, sim_to_source=0.55,
                    blend_fraction=0.0)
    assert "modest" in out["verdict"]


def test_identity_shed_cannot_be_silently_omitted():
    # It was omitted once: the study passed source-to-learner similarity into a
    # parameter named sim_to_source, so shed went unmeasured while the module
    # docstring claimed to report it. A required argument makes that impossible.
    with pytest.raises(TypeError):
        trade_off(0.9, 0.6, 0.11, 0.10)   # missing shed AND blend rate


def test_no_single_figure_of_merit_is_produced():
    # Weighting "sounds like them" against "still says the words" is a product
    # decision. A combined score would bury it, so the API must not offer one.
    out = trade_off(0.9, 0.6, 0.11, 0.10, 0.55, 0.0)
    assert "score" not in out and "overall" not in out
    assert {"sim_to_learner", "sim_source_to_learner", "sim_to_source",
            "identity_gain", "resembles_source_more", "blend_fraction",
            "wer_output", "wer_source", "wer_cost", "verdict"} == set(out)


# --- the table must anchor its own similarity column -------------------------

def _rows(n=4, ceiling=0.93):
    return [{"sim_learner_real": ceiling, "sim_native_tts": 0.68,
             "sim_learner_tts": 0.84, "sim_golden": 0.81,
             "wer_real": 0.18, "wer_native_tts": 0.19,
             "wer_learner_tts": 0.26, "wer_golden": 0.18} for _ in range(n)]


def test_table_prints_the_measured_ceiling_not_a_placeholder():
    # The bug this pins: the first version of the study computed the real-vs-real
    # ceiling and then printed the literal string "(held-out ref)" in its place,
    # discarding the one number that gives the SIM column a scale.
    text = golden_table(_rows())
    assert "0.930" in text
    assert "held-out ref" not in text


def test_table_refuses_to_render_without_a_ceiling():
    # Same guardrail as EvalTable's GroundTruthMissing: an unanchored similarity
    # column invites reading 0.84 as good when the ceiling might be 0.97.
    rows = _rows()
    for r in rows:
        r["sim_learner_real"] = float("nan")
    with pytest.raises(CeilingMissing, match="no scale"):
        golden_table(rows)


def test_ceiling_survives_speakers_that_could_not_supply_one():
    # A single-utterance speaker contributes NaN. That must not poison the mean
    # for the speakers that did supply a second recording.
    rows = _rows(3) + [dict(_rows(1)[0], sim_learner_real=float("nan"))]
    assert "0.930" in golden_table(rows)


def test_blend_rate_catches_what_the_means_hide():
    # The real n=72 shape: mean similarity favours the learner (0.824 vs 0.802)
    # while a majority of individual utterances are closer to the source. Two
    # heavily overlapping distributions produce a mean gap that no single
    # utterance need exhibit, so a verdict read off the means calls this a
    # success. Half the rows lean each way by different margins.
    rows = ([{"sim_golden": 0.95, "sim_golden_to_source": 0.60}] * 3 +
            [{"sim_golden": 0.70, "sim_golden_to_source": 0.88}] * 5)
    assert mean_ignoring_nan(rows, "sim_golden") > \
        mean_ignoring_nan(rows, "sim_golden_to_source")   # means say "fine"
    assert blend_rate(rows) == 0.625                       # utterances say otherwise

    out = trade_off(sim_to_learner=mean_ignoring_nan(rows, "sim_golden"),
                    sim_source_to_learner=0.60, wer_output=0.10,
                    wer_source=0.10,
                    sim_to_source=mean_ignoring_nan(rows, "sim_golden_to_source"),
                    blend_fraction=blend_rate(rows))
    assert out["resembles_source_more"] is False   # the mean test misses it
    assert "majority" in out["verdict"]            # the per-utterance test does not


def test_blend_rate_refuses_when_shed_was_never_measured():
    with pytest.raises(CeilingMissing, match="blend rate is undefined"):
        blend_rate([{"sim_golden": 0.8}])


def test_shed_is_flagged_uninterpretable_without_a_synthetic_anchor():
    # SIM(golden, native tts) compares two clips from the same TTS and vocoder,
    # so shared synthesis artefacts inflate it regardless of retained identity.
    # Without a same-pipeline different-speaker anchor the number must not be
    # presented as if it meant something.
    text = golden_table(_rows())
    assert "UNINTERPRETABLE" in text


def test_shed_is_reported_against_the_anchor_when_present():
    rows = [dict(r, sim_golden_to_source=0.80, sim_ltts_to_source=0.77)
            for r in _rows()]
    text = golden_table(rows)
    assert "UNINTERPRETABLE" not in text
    assert "0.770" in text          # the anchor
    assert "+0.030" in text         # only the excess can be retained identity


def test_mean_ignores_nan_and_reports_nan_when_empty():
    assert mean_ignoring_nan([{"x": 1.0}, {"x": float("nan")}], "x") == 1.0
    empty = mean_ignoring_nan([{"x": float("nan")}], "x")
    assert empty != empty
