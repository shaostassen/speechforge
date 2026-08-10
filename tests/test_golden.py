"""Golden-speaker pipeline and its trade-off reporting, with fake models.

The interesting part is the reporting: this pipeline has two failure modes that
look like success if you read one metric, so the tests mostly pin that both get
named.
"""
import numpy as np
import pytest

from speechforge.golden import GoldenResult, golden_speech, trade_off


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

    def synthesize(self, text, reference_path, seed=None):
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
    out = trade_off(sim_to_learner=0.60, sim_to_source=0.90,
                    wer_output=0.10, wer_source=0.10)
    assert out["identity_gain"] < 0
    assert "did not move identity" in out["verdict"]


def test_damaged_speech_is_named_even_when_identity_moved():
    # Failure mode 2: it sounds like the learner and no longer says the words.
    # A similarity-only reading would call this a success.
    out = trade_off(sim_to_learner=0.92, sim_to_source=0.55,
                    wer_output=0.40, wer_source=0.10)
    assert out["identity_gain"] > 0
    assert "damaged the speech" in out["verdict"]


def test_success_requires_both_axes():
    out = trade_off(sim_to_learner=0.90, sim_to_source=0.60,
                    wer_output=0.11, wer_source=0.10)
    assert "largely intact" in out["verdict"]


def test_modest_cost_is_distinguished_from_damage():
    out = trade_off(sim_to_learner=0.90, sim_to_source=0.60,
                    wer_output=0.16, wer_source=0.10)
    assert "modest" in out["verdict"]


def test_no_single_figure_of_merit_is_produced():
    # Weighting "sounds like them" against "still says the words" is a product
    # decision. A combined score would bury it, so the API must not offer one.
    out = trade_off(0.9, 0.6, 0.11, 0.10)
    assert "score" not in out and "overall" not in out
    assert {"sim_to_learner", "sim_to_source", "identity_gain",
            "wer_output", "wer_source", "wer_cost", "verdict"} == set(out)
