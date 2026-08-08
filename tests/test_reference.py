"""Reference-clip preparation, tested against signals we construct.

This is the module that decides clone quality, and none of it needs a model, so
all of it is pinned here. Runs offline with no weights, like the ASR project.
"""
import numpy as np
import pytest

from speechforge.reference import (MIN_S, ReferenceReport, prepare, resample,
                                   trim_silence)

SR = 16000


def _speech(seconds=6.0, sr=SR, amp=0.4, f0=180.0):
    """Amplitude-modulated tone. Not speech, but it has speech-like envelope."""
    t = np.arange(int(seconds * sr)) / sr
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amp * env * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _with_silence(y, lead_s=0.0, tail_s=0.0, sr=SR):
    return np.concatenate([
        np.zeros(int(lead_s * sr), np.float32), y,
        np.zeros(int(tail_s * sr), np.float32)]).astype(np.float32)


# --- trimming ---------------------------------------------------------------

def test_trims_leading_and_trailing_silence():
    y = _with_silence(_speech(4.0), lead_s=1.0, tail_s=2.0)
    out, lead, tail = trim_silence(y, SR)
    assert lead == pytest.approx(1.0, abs=0.05)
    assert tail == pytest.approx(2.0, abs=0.05)
    assert len(out) / SR == pytest.approx(4.0, abs=0.1)


def test_all_silent_input_is_left_alone_not_emptied():
    # Returning an empty array here would turn a bad clip into a crash later.
    y = np.zeros(SR, np.float32)
    out, lead, tail = trim_silence(y, SR)
    assert len(out) == len(y)
    assert (lead, tail) == (0.0, 0.0)


def test_trim_handles_empty_input():
    out, lead, tail = trim_silence(np.array([], np.float32), SR)
    assert len(out) == 0 and lead == 0.0 and tail == 0.0


# --- duration policy --------------------------------------------------------

def test_long_reference_is_truncated_not_kept():
    # The counterintuitive rule: more reference audio makes cloning worse.
    y = _speech(40.0)
    out, sr, rep = prepare(y, SR)
    assert len(out) / sr == pytest.approx(10.0, abs=0.2)
    assert any("truncated" in w for w in rep.warnings)


def test_truncation_keeps_the_beginning():
    # The tail is where recordings trail off or clip a word.
    y = np.concatenate([_speech(11.0, amp=0.4),
                        _speech(11.0, amp=0.4, f0=400.0)])
    out, sr, _rep = prepare(y, SR)
    # Dominant frequency should be the first segment's, not the second's.
    spec = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(len(out), 1 / sr)[int(np.argmax(spec))]
    assert abs(peak_hz - 180.0) < abs(peak_hz - 400.0)


def test_short_reference_warns_but_still_returns_audio():
    y = _speech(1.5)
    out, _sr, rep = prepare(y, SR)
    assert len(out) > 0
    assert any("under-determined" in w for w in rep.warnings)


def test_trailing_silence_is_called_out_specifically():
    # For continuation-style models this region becomes the onset of the
    # generated speech, so it deserves its own warning.
    y = _with_silence(_speech(5.0), tail_s=1.5)
    _out, _sr, rep = prepare(y, SR)
    assert any("trailing" in w for w in rep.warnings)


# --- level ------------------------------------------------------------------

def test_clipping_is_flagged():
    y = np.clip(_speech(5.0, amp=3.0), -1.0, 1.0)
    _out, _sr, rep = prepare(y, SR)
    assert any("clipping" in w for w in rep.warnings)


def test_very_quiet_reference_is_flagged():
    _out, _sr, rep = prepare(_speech(5.0, amp=0.01), SR)
    assert any("quiet" in w for w in rep.warnings)


def test_output_is_peak_normalized():
    out, _sr, _rep = prepare(_speech(5.0, amp=0.05), SR)
    assert float(np.max(np.abs(out))) == pytest.approx(0.95, abs=1e-3)


def test_empty_input_reports_rather_than_raising():
    out, _sr, rep = prepare(np.array([], np.float32), SR)
    assert len(out) == 0
    assert rep.warnings == ["reference is empty"]


# --- resampling -------------------------------------------------------------

def test_resample_changes_length_proportionally():
    y = _speech(2.0)
    out = resample(y, SR, 24000)
    assert len(out) == pytest.approx(len(y) * 24000 / SR, rel=0.01)


def test_resample_is_a_noop_at_the_same_rate():
    y = _speech(1.0)
    assert np.array_equal(resample(y, SR, SR), y)


def test_resample_preserves_tone_frequency():
    # A naive decimation would alias; this checks the band-limited path.
    y = _speech(2.0, f0=300.0)
    out = resample(y, SR, 24000)
    spec = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(len(out), 1 / 24000)[int(np.argmax(spec))]
    assert peak_hz == pytest.approx(300.0, abs=15.0)


def test_prepare_resamples_when_asked():
    out, sr, rep = prepare(_speech(5.0), SR, target_sr=24000)
    assert sr == 24000
    assert rep.sample_rate == 24000
    assert len(out) / sr == pytest.approx(5.0, abs=0.2)


def test_report_is_json_ready():
    import json
    _out, _sr, rep = prepare(_speech(5.0), SR)
    assert isinstance(rep, ReferenceReport)
    json.dumps(rep.to_dict())
