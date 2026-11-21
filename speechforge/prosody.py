"""Prosody: the axis WER and speaker similarity cannot see.

Three per-utterance measurements. Each is read against the same measurement on
real speech — the ground-truth row, as for every other column in this harness:

**Pitch range** — the 10th-to-90th-percentile spread of F0 within an
utterance, in semitones. Semitones make it register-free: a low voice and a
high one intoning alike score alike. A monotone voice approaches zero. Below
the ground-truth row means flatter than the speakers themselves read; above it
means more exaggerated. Neither direction is "better" — the target is the
human row.

**Pitch-level gap** — the distance in semitones between the median F0 of the
clone and that of the speaker's held-out recording: whether the clone speaks in
the speaker's register. Its ground truth is a *different* real recording of the
same speaker scored against the held-out one, exactly as for SIM-o, because one
person's two readings do not share a median pitch either.

**Speaking rate** — words per second over the speech span, leading and trailing
silence trimmed with the same rule as reference preparation; pauses inside the
span count. Every row reads the same texts, so rates compare directly with the
human row.

The trap this module is built around: **pitch trackers do not fail
symmetrically on real and synthetic speech.** A tracker that jumps an octave on
vocoder artefacts makes synthesis read as *more* expressive. Two defences:

- the range is a percentile range, not a standard deviation. A frame at
  double F0 is a 12-semitone outlier: with 2% of frames an octave off, a
  standard deviation rises 32% and this range 4%; at 5%, 62% against 12%. At
  10% both break (+105%, +63%) — robust to a few bad frames, not to a broken
  track. So each measurement also reports `jump_fraction`, the share of
  adjacent voiced frames whose F0 moves more than 9 semitones in 10 ms: a voice
  does not do that, and an octave error does;
- `scripts/prosody_checks.py` measures pYIN against an independent tracker,
  WORLD's Harvest, on both real and synthesized speech before any number is
  reported. Trust neither tracker on audio the other has not been checked on.

Not measured: whether intonation is *appropriate*. A natural reading that
differs from the speaker's own scores the same as theirs, which is deliberate —
comparing contours frame by frame against one human rendition would penalise
exactly the variation a generative model is meant to add, and would have no
ground-truth row. Nor naturalness, which needs listeners.

Import-light like the rest of the package: librosa loads inside `f0_track`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from speechforge.reference import resample, trim_silence

TARGET_SR = 16000
HOP = 160            # 10 ms
FRAME = 1024         # 64 ms: at least two periods of the lowest F0 searched
FMIN, FMAX = 50.0, 500.0   # adult speech, with headroom at both ends
MIN_VOICED = 20      # frames (0.2 s); fewer and there is no pitch to summarise
RANGE_PCT = (10.0, 90.0)
JUMP_ST = 9.0        # per 10 ms frame: beyond any voice, short of an octave


@dataclass
class Prosody:
    median_f0_hz: float      # nan when too little is voiced
    pitch_range_st: float    # p10-p90 of F0, semitones; nan likewise
    voiced_fraction: float   # voiced frames / frames in the speech span
    jump_fraction: float     # adjacent voiced frames > JUMP_ST apart; nan if none
    speech_s: float          # after trimming leading and trailing silence
    words_per_s: Optional[float]   # None when no text was given

    def to_dict(self) -> dict:
        def r(v, p):
            return None if v is None or v != v else round(float(v), p)
        return {"median_f0_hz": r(self.median_f0_hz, 1),
                "pitch_range_st": r(self.pitch_range_st, 3),
                "voiced_fraction": r(self.voiced_fraction, 3),
                "jump_fraction": r(self.jump_fraction, 4),
                "speech_s": r(self.speech_s, 3),
                "words_per_s": r(self.words_per_s, 3)}


def semitones(f_hz, ref_hz) -> np.ndarray:
    return 12.0 * np.log2(np.asarray(f_hz, dtype=np.float64) / ref_hz)


def summarize_f0(f0: np.ndarray):
    """``(median_hz, p10-p90 range in semitones)`` of a frame track, NaN = unvoiced.

    Pure numpy, so the summary maths is testable without a tracker.
    """
    v = np.asarray(f0, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < MIN_VOICED:
        return float("nan"), float("nan")
    med = float(np.median(v))
    lo, hi = np.percentile(semitones(v, med), RANGE_PCT)
    return med, float(hi - lo)


def jump_fraction(f0: np.ndarray) -> float:
    """Share of adjacent voiced-frame pairs whose F0 differs by > JUMP_ST.

    An octave error shows up as a 12-semitone step between neighbouring frames;
    no voice moves that far in 10 ms. A tracking-quality diagnostic, not prosody.
    """
    v = np.asarray(f0, dtype=np.float64)
    both = np.isfinite(v[:-1]) & np.isfinite(v[1:]) & (v[:-1] > 0) & (v[1:] > 0)
    if not both.any():
        return float("nan")
    step = np.abs(semitones(v[1:][both], v[:-1][both]))
    return float(np.mean(step > JUMP_ST))


def pitch_gap_st(a_hz: float, b_hz: float) -> float:
    """Absolute distance in semitones between two median pitches."""
    if not (a_hz > 0 and b_hz > 0):     # also false for nan
        return float("nan")
    return float(abs(semitones(a_hz, b_hz)))


def _mono16k(audio, sr: int) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return resample(y, sr, TARGET_SR)


def f0_track(audio, sr: int) -> np.ndarray:
    """F0 in Hz per 10 ms frame (pYIN), NaN where unvoiced.

    The first call in a process pays a one-off JIT compile (~20 s); after that
    a 7 s clip takes about a quarter of a second on one CPU core.
    """
    import librosa

    y = _mono16k(audio, sr)
    if len(y) < FRAME:
        return np.full(0, np.nan)
    f0, voiced, _prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=TARGET_SR,
                                     frame_length=FRAME, hop_length=HOP)
    return np.where(voiced, f0, np.nan)


def measure(audio, sr: int, text: Optional[str] = None, track=f0_track) -> Prosody:
    """Prosody of one utterance. ``text`` is needed for the speaking rate only.

    The pitch is tracked on the speech span, not the whole clip, so leading and
    trailing silence cannot dilute the voiced fraction.
    """
    y = _mono16k(audio, sr)
    span, _lead, _tail = trim_silence(y, TARGET_SR)
    speech_s = len(span) / TARGET_SR
    f0 = track(span, TARGET_SR)
    med, rng = summarize_f0(f0)
    voiced = float(np.mean(np.isfinite(f0))) if len(f0) else 0.0
    rate = None
    if text is not None and speech_s > 0:
        rate = len(text.split()) / speech_s
    return Prosody(med, rng, voiced, jump_fraction(f0), speech_s, rate)


def prosody_scorer():
    """``measure``, once its tracker is known to import. For ``require_scorer``."""
    import librosa  # noqa: F401  (fail here, before any synthesis, if missing)
    return measure
