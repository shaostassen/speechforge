"""Prosody measurement, on signals whose pitch is known by construction.

The summary maths runs without a tracker. The tracked tests build harmonic
"voices" with a specified F0 contour, so the right answer is known exactly, and
need librosa (skipped without it).
"""
import numpy as np
import pytest

from speechforge.prosody import (FMAX, FMIN, MIN_VOICED, adaptive_range,
                                 jump_fraction, measure, pitch_gap_st,
                                 semitones, summarize_f0)

HAVE_LIBROSA = True
try:
    import librosa  # noqa: F401
except ImportError:
    HAVE_LIBROSA = False

needs_librosa = pytest.mark.skipif(not HAVE_LIBROSA,
                                   reason="needs librosa (the [prosody] extra)")
SR = 16000


def voice(f0_hz, sr=SR, harmonics=8, lead_s=0.0, tail_s=0.0):
    """A harmonic signal following ``f0_hz`` (one value per sample), with optional
    silence either side. Harmonic amplitudes fall as 1/k, roughly like a voice."""
    f0 = np.asarray(f0_hz, dtype=np.float64)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))
    y = 0.3 * y / np.max(np.abs(y))
    pad = lambda s: np.zeros(int(s * sr))  # noqa: E731
    return np.concatenate([pad(lead_s), y, pad(tail_s)]).astype(np.float32)


# --- summary maths, no tracker ------------------------------------------------

def test_semitones_of_an_octave_is_twelve():
    assert semitones(220.0, 110.0) == pytest.approx(12.0)


def test_flat_track_has_zero_range():
    med, rng = summarize_f0(np.full(200, 150.0))
    assert med == pytest.approx(150.0)
    assert rng == pytest.approx(0.0, abs=1e-9)


def test_unvoiced_frames_are_ignored():
    f0 = np.full(200, 150.0)
    f0[::2] = np.nan
    assert summarize_f0(f0)[0] == pytest.approx(150.0)


def test_too_little_voicing_gives_nan_not_a_number():
    med, rng = summarize_f0(np.r_[np.full(MIN_VOICED - 1, 150.0),
                                  np.full(100, np.nan)])
    assert np.isnan(med) and np.isnan(rng)


def _contaminated(frac):
    rng_ = np.random.default_rng(0)
    clean = 150.0 * 2 ** (rng_.normal(0, 2.0, 2000) / 12)   # ~2 st spread
    bad = clean.copy()
    bad[::int(round(1 / frac))] *= 2.0                      # octave errors
    return clean, bad


@pytest.mark.parametrize("frac", [0.02, 0.05])
def test_range_moves_far_less_than_a_standard_deviation_under_octave_errors(frac):
    # The failure this metric is designed around: a tracker jumping an octave on
    # a few frames must not read as a much livelier voice. Measured: at 2% a
    # standard deviation rises 32% and the range 4%; at 5%, 62% against 12%.
    clean, bad = _contaminated(frac)
    rise = lambda a, b: b / a - 1  # noqa: E731
    sd = lambda f: np.std(semitones(f, np.median(f)))  # noqa: E731
    r_rise = rise(summarize_f0(clean)[1], summarize_f0(bad)[1])
    assert r_rise < 0.15
    assert r_rise < rise(sd(clean), sd(bad)) / 4


def test_jump_fraction_sees_the_octave_errors_the_range_cannot_fully_absorb():
    # At 10% contamination the range breaks too (+63%), so the error rate itself
    # has to be visible. A real contour is smooth from frame to frame, so this
    # one is: +-2 semitones over a few hundred milliseconds.
    t = np.arange(2000)
    clean = 150.0 * 2 ** (2.0 * np.sin(2 * np.pi * t / 40) / 12)
    bad = clean.copy()
    bad[::10] *= 2.0
    assert jump_fraction(clean) == 0.0
    # Each bad frame is a jump in and a jump out: 2 of every ~10 transitions.
    assert jump_fraction(bad) == pytest.approx(0.2, abs=0.01)


def test_jump_fraction_ignores_unvoiced_gaps():
    f0 = np.array([150.0, np.nan, 300.0, 300.0])   # no adjacent voiced pair jumps
    assert jump_fraction(f0) == 0.0
    assert np.isnan(jump_fraction(np.full(5, np.nan)))


def test_second_pass_range_follows_the_speaker():
    # A low voice: the second pass starts at 0.75 x Q1, well clear of the 50 Hz
    # first-pass floor where creaky frames used to be pinned an octave down.
    f0 = np.r_[np.full(50, 90.0), np.full(50, 110.0)]
    lo, hi = adaptive_range(f0)
    assert lo == pytest.approx(0.75 * 90.0)
    assert hi == pytest.approx(1.5 * 110.0)


def test_second_pass_range_is_bounded_both_ways():
    assert adaptive_range(np.full(100, 55.0))[0] == FMIN       # never below the floor
    # The first pass reports at most FMAX, so the ceiling tops out at 1.5 x FMAX.
    assert adaptive_range(np.full(100, FMAX))[1] == pytest.approx(1.5 * FMAX)
    assert adaptive_range(np.full(100, 480.0))[1] == pytest.approx(720.0)


def test_no_second_pass_without_enough_voicing():
    assert adaptive_range(np.full(MIN_VOICED - 1, 120.0)) is None


def test_pitch_gap_is_symmetric_and_nan_safe():
    assert pitch_gap_st(110.0, 220.0) == pytest.approx(12.0)
    assert pitch_gap_st(220.0, 110.0) == pytest.approx(12.0)
    assert np.isnan(pitch_gap_st(float("nan"), 110.0))


# --- through the tracker ---------------------------------------------------------

@needs_librosa
def test_flat_voice_is_tracked_at_its_pitch_with_no_range():
    p = measure(voice(np.full(SR * 2, 120.0)), SR)
    assert p.median_f0_hz == pytest.approx(120.0, rel=0.01)
    assert p.pitch_range_st < 0.3
    assert p.voiced_fraction > 0.9
    assert p.jump_fraction == 0.0


@needs_librosa
def test_a_one_octave_glide_has_the_range_its_percentiles_predict():
    # F0 rises 100 -> 200 Hz evenly in semitones, so p10-p90 covers 80% of 12 st.
    n = SR * 3
    f0 = 100.0 * 2 ** (np.linspace(0, 12, n) / 12)
    p = measure(voice(f0), SR)
    assert p.pitch_range_st == pytest.approx(9.6, abs=0.4)


@needs_librosa
def test_register_gap_between_two_voices_an_octave_apart():
    lo = measure(voice(np.full(SR * 2, 110.0)), SR)
    hi = measure(voice(np.full(SR * 2, 220.0)), SR)
    assert pitch_gap_st(lo.median_f0_hz, hi.median_f0_hz) == pytest.approx(12.0, abs=0.2)


@needs_librosa
def test_rate_counts_speech_not_the_silence_around_it():
    # 2 s of voice, 1 s of silence either side, 5 words -> 2.5 words/s, not 1.25.
    y = voice(np.full(SR * 2, 150.0), lead_s=1.0, tail_s=1.0)
    p = measure(y, SR, text="one two three four five")
    assert p.speech_s == pytest.approx(2.0, abs=0.05)
    assert p.words_per_s == pytest.approx(2.5, abs=0.1)


@needs_librosa
def test_silence_has_no_pitch():
    p = measure(np.zeros(SR * 2, dtype=np.float32), SR, text="nothing")
    assert np.isnan(p.median_f0_hz) and np.isnan(p.pitch_range_st)


@needs_librosa
def test_sample_rate_does_not_change_the_answer():
    f0 = np.full(24000 * 2, 140.0)
    p24 = measure(voice(f0, sr=24000), 24000)
    p16 = measure(voice(np.full(SR * 2, 140.0)), SR)
    assert p24.median_f0_hz == pytest.approx(p16.median_f0_hz, rel=0.01)


def test_rate_needs_text():
    p = measure(np.zeros(SR, dtype=np.float32), SR,
                track=lambda y, sr: np.full(100, np.nan))
    assert p.words_per_s is None
