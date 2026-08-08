"""Prepare a reference clip for cloning. This is where quality is won or lost.

Every documented failure mode of zero-shot cloning traces back to the reference,
not the model:

**Longer is worse, not better.** Both CosyVoice and dots.tts state this outright
in their docs — around 10 s is the sweet spot, and past it inference slows *and*
cloning quality degrades. The intuition that "more reference audio is safer" is
simply wrong for these models. (kNN-VC is the exception: it genuinely wants
minutes, because it is doing nearest-neighbour retrieval rather than conditioning
an embedding.)

**Trailing silence is a specific killer.** Continuation-style models treat the
reference as a literal prefix to the generation, so trailing noise or room tone
becomes the *onset* of your synthesized speech. Leading silence is comparatively
harmless; trailing is not.

**Noise gets baked into the speaker embedding.** The clone reproduces the room,
the compression, and the microphone alongside the voice. A clean 10 s clip beats
two noisy minutes.

Pure NumPy so all of this is testable without a model or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

TARGET_S = 10.0          # documented sweet spot for CosyVoice / dots.tts
MIN_S = 3.0              # below this, speaker identity is under-determined
MAX_S = 30.0             # hard ceiling; most models train on <=30 s utterances

# Silence gate. Relative to peak so it survives level differences between a
# phone recording and a studio one.
SILENCE_DB = -45.0
FRAME_S = 0.02


@dataclass
class ReferenceReport:
    """What was done to the clip, and anything the caller should worry about."""
    duration_s: float
    original_duration_s: float
    sample_rate: int
    trimmed_lead_s: float
    trimmed_tail_s: float
    peak: float
    warnings: list

    def to_dict(self) -> dict:
        return {
            "duration_s": round(self.duration_s, 2),
            "original_duration_s": round(self.original_duration_s, 2),
            "sample_rate": self.sample_rate,
            "trimmed_lead_s": round(self.trimmed_lead_s, 3),
            "trimmed_tail_s": round(self.trimmed_tail_s, 3),
            "peak": round(self.peak, 4),
            "warnings": self.warnings,
        }


def _frame_energy_db(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    n = max(1, int(FRAME_S * sr))
    pad = (-len(y)) % n
    padded = np.concatenate([y, np.zeros(pad, dtype=y.dtype)]) if pad else y
    frames = padded.reshape(-1, n)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(np.max(np.abs(y))) or 1e-12
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1e-12) / peak)
    return db, n


def trim_silence(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float, float]:
    """Drop leading and trailing near-silence. Returns (audio, lead_s, tail_s)."""
    if len(y) == 0:
        return y, 0.0, 0.0
    db, n = _frame_energy_db(y, sr)
    voiced = np.flatnonzero(db > SILENCE_DB)
    if voiced.size == 0:
        return y, 0.0, 0.0            # all quiet: leave it, and warn upstream
    start = int(voiced[0]) * n
    end = min(len(y), (int(voiced[-1]) + 1) * n)
    return y[start:end], start / sr, (len(y) - end) / sr


def prepare(y: np.ndarray, sr: int, target_sr: Optional[int] = None,
            target_s: float = TARGET_S
            ) -> Tuple[np.ndarray, int, ReferenceReport]:
    """Clean a reference clip and report what happened.

    Trims silence, resamples if asked, caps the duration, and peak-normalizes.
    Never raises on bad input — it returns warnings instead, because the caller
    is usually a batch job and a clip that is merely *suspect* should still be
    usable with the doubt recorded.
    """
    y = np.asarray(y, dtype=np.float32).ravel()
    original_s = len(y) / sr if sr else 0.0
    warnings: list = []

    if len(y) == 0:
        return y, sr, ReferenceReport(0.0, 0.0, sr, 0.0, 0.0, 0.0,
                                      ["reference is empty"])

    y, lead, tail = trim_silence(y, sr)
    if tail > 0.25:
        # Worth surfacing: for continuation-style models this trailing region
        # would have become the onset of the generated speech.
        warnings.append(f"trimmed {tail:.2f}s of trailing near-silence")

    if target_sr and target_sr != sr:
        y = resample(y, sr, target_sr)
        sr = target_sr

    duration = len(y) / sr
    if duration > max(target_s, MIN_S):
        # Truncate from the START, keeping the earliest speech: the tail is
        # where a recording is most likely to trail off or clip a word.
        y = y[: int(max(target_s, MIN_S) * sr)]
        warnings.append(f"truncated {duration:.1f}s to {len(y) / sr:.1f}s "
                        f"(longer references degrade cloning, not improve it)")
        duration = len(y) / sr

    if duration < MIN_S:
        warnings.append(f"only {duration:.1f}s of speech after trimming; "
                        f"under {MIN_S:.0f}s the voice is under-determined")

    peak = float(np.max(np.abs(y)))
    if peak >= 0.999:
        warnings.append("reference is clipping; distortion will be cloned")
    if 0.0 < peak < 0.05:
        warnings.append(f"reference is very quiet (peak {peak:.3f}); "
                        "noise will dominate the speaker embedding")
    if peak > 0:
        y = (y / peak * 0.95).astype(np.float32)

    return y, sr, ReferenceReport(duration, original_s, sr, lead, tail, peak,
                                  warnings)


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited resample via scipy's polyphase filter.

    Not naive decimation: dropping samples aliases high-frequency energy down
    into the speech band, and the model will faithfully clone the artifacts.
    """
    if sr_from == sr_to or len(y) == 0:
        return np.asarray(y, dtype=np.float32)
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(y, int(sr_to) // g, int(sr_from) // g).astype(np.float32)
