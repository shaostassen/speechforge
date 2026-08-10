"""Golden-speaker synthesis: a learner's own voice, pronouncing correctly.

The motivating application. Telling a learner they mispronounced something is
half a tool; letting them hear the correction **in their own voice** is the other
half, and the pedagogical literature finds that a well-matched voice is what
actually drives improvement.

Why plain zero-shot TTS does not do this
----------------------------------------
Cloning from an accented reference *preserves the accent* — the model treats the
reference as the target to imitate, pronunciation included. That is a feature for
voice cloning and precisely wrong here. What is wanted is pronunciation from one
source and identity from another, which is **foreign accent conversion**, a
distinct task with no maintained open-source implementation.

The established recipe is *synthesize then convert*: a native-referenced TTS
produces correct pronunciation, then voice conversion moves the timbre onto the
learner. Both stages here are MIT-licensed.

What can actually be measured
-----------------------------
Ground truth is impossible by construction: you cannot record a non-native
speaker producing native pronunciation, so there is no target signal to compare
against. That rules out the obvious evaluation and is why this is research rather
than a library call.

What remains measurable is the **trade-off**, using only machinery this repo
already has:

  identity acquired  SIM(output, learner's held-out audio) — does it sound like them?
  identity shed      SIM(output, the native TTS it came from) — did it stop sounding like that voice?
  pronunciation kept WER(output) against WER(the native TTS input) — did conversion damage the speech?

A method that scores well on identity and badly on WER has re-accented the audio
or destroyed it; one that scores well on WER and badly on identity has simply not
converted. Neither number alone can distinguish success from either failure,
which is the whole reason to report all three.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from speechforge.synth import clone


class CeilingMissing(RuntimeError):
    """Raised when the real-vs-real identity ceiling could not be measured."""


class ChatterboxVCBackend:
    """Voice conversion via Chatterbox — MIT code and MIT weights.

    Separate from the TTS backend because they are separate models with separate
    checkpoints, even though they ship in one package. Loading both costs memory,
    so a caller that only synthesizes should not pay for the converter.
    """

    def __init__(self, device: str = "cpu"):
        from chatterbox.vc import ChatterboxVC
        self._model = ChatterboxVC.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        self._device = device

    def convert(self, audio: np.ndarray, sr: int, target_ref_path: str,
                seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Re-voice ``audio`` to the speaker in ``target_ref_path``.

        Seeded for the same reason synthesis is: any stochastic stage makes a
        recorded number unreproducible, and this pipeline has two of them.
        """
        import soundfile as sf
        if seed is not None:
            import torch
            torch.manual_seed(seed)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, np.asarray(audio, dtype=np.float32).ravel(), sr)
            wav = self._model.generate(tmp.name,
                                       target_voice_path=target_ref_path)
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return arr.astype(np.float32).ravel(), self.sample_rate

    def describe(self) -> str:
        return "chatterbox-vc (MIT)"


@dataclass
class GoldenResult:
    """Output plus every intermediate, because each stage needs scoring."""
    audio: np.ndarray
    sample_rate: int
    # Stage 1: correct pronunciation in the native reference's voice. Kept
    # because it is the pronunciation baseline the final output is judged
    # against — without it, a WER change cannot be attributed to conversion.
    source_audio: np.ndarray = field(repr=False, default=None)
    source_sample_rate: int = 0
    reference_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"sample_rate": self.sample_rate,
                "duration_s": round(len(self.audio) / self.sample_rate, 2)
                if self.sample_rate else None,
                "source_duration_s": (
                    round(len(self.source_audio) / self.source_sample_rate, 2)
                    if self.source_audio is not None and self.source_sample_rate
                    else None),
                "reference_report": self.reference_report}


def golden_speech(tts_backend, vc_backend, text: str,
                  native_ref: np.ndarray, native_sr: int,
                  learner_ref_path: str,
                  seed: Optional[int] = None) -> GoldenResult:
    """Say ``text`` with the native reference's pronunciation, in the learner's voice.

    ``learner_ref_path`` is a path rather than an array because the converter
    takes one, and because the learner's clip should be prepared and written once
    by the caller rather than re-staged per utterance.

    The intermediate is returned, not discarded: a WER change between stage 1 and
    the output is the only way to tell whether conversion preserved the
    pronunciation it was given.
    """
    source, src_sr, report = clone(tts_backend, text, native_ref, native_sr,
                                   seed=seed)
    out, out_sr = vc_backend.convert(source, src_sr, learner_ref_path, seed=seed)
    return GoldenResult(audio=out, sample_rate=out_sr,
                        source_audio=source, source_sample_rate=src_sr,
                        reference_report=report)


def trade_off(sim_to_learner: float, sim_source_to_learner: float,
              wer_output: float, wer_source: float,
              sim_to_source: float) -> dict:
    """Summarize the three-way trade-off, without collapsing it to one score.

    The three arguments carrying similarities are easy to confuse, so, precisely:

      sim_to_learner        SIM(output, learner held-out)   identity acquired
      sim_source_to_learner SIM(native tts, learner held-out)  where it started
      sim_to_source         SIM(output, the native tts)     identity shed

    ``sim_to_source`` is required rather than optional because the module docstring
    promises to report identity shed, and an earlier version of this function
    accepted ``sim_source_to_learner`` under the name ``sim_to_source`` and so
    never measured shed at all. That gap hid a specific failure mode: a converter
    that *blends* the two voices scores a gain toward the learner while still
    resembling the source, producing audio that sounds like neither person.

    Deliberately returns the components and a verdict string rather than a single
    figure of merit. There is no defensible way to weight "sounds like the
    learner" against "still says the words" — that weighting is a product
    decision, and a combined score would bury it.
    """
    identity_gain = sim_to_learner - sim_source_to_learner
    wer_cost = wer_output - wer_source
    # Threshold-free: is the output closer to the voice it was aiming for than to
    # the voice it came from? Anything else would be an invented cutoff.
    resembles_source_more = sim_to_source >= sim_to_learner

    if identity_gain <= 0.0:
        verdict = ("conversion did not move identity toward the learner; "
                   "output still resembles the source voice")
    elif resembles_source_more:
        verdict = ("identity moved toward the learner but the output still "
                   "resembles its source at least as much — a blend, not a "
                   "conversion")
    elif wer_cost > 0.10:
        verdict = ("identity moved, but conversion damaged the speech "
                   "(WER cost > 10 points) — likely re-accented or degraded")
    elif wer_cost > 0.03:
        verdict = "identity moved at a modest intelligibility cost"
    else:
        verdict = "identity moved with pronunciation largely intact"

    return {"sim_to_learner": sim_to_learner,
            "sim_source_to_learner": sim_source_to_learner,
            "sim_to_source": sim_to_source,
            "identity_gain": identity_gain,
            "resembles_source_more": resembles_source_more,
            "wer_output": wer_output,
            "wer_source": wer_source,
            "wer_cost": wer_cost,
            "verdict": verdict}


def mean_ignoring_nan(rows: Sequence[dict], key: str) -> float:
    """Mean of ``key`` across ``rows``, skipping NaN. NaN if nothing is left."""
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    return float(np.mean(vals)) if vals else float("nan")


def golden_table(rows: Sequence[dict]) -> str:
    """Render the four-signal table, with the identity ceiling always shown.

    Refuses to render without a real-vs-real ceiling, for the same reason
    ``EvalTable`` refuses to render without a ground-truth row: an unanchored
    similarity column cannot be read. A clone at 0.84 is excellent if two real
    recordings of that speaker score 0.85 and mediocre if they score 0.97, and
    the number alone does not say which.

    Measuring the ceiling needs two held-out utterances per speaker, so a study
    configured with fewer fails here rather than printing an unreadable column.
    """
    ceiling = mean_ignoring_nan(rows, "sim_learner_real")
    if ceiling != ceiling:
        raise CeilingMissing(
            "no real-vs-real speaker similarity was measured, so the SIM column "
            "has no scale. Two held-out utterances per speaker are needed "
            "(--utterances 3 or more)."
        )

    def m(key):
        return mean_ignoring_nan(rows, key)

    out = ["=" * 68,
           "GOLDEN SPEAKER — identity vs pronunciation",
           "=" * 68,
           f"{'signal':>16} {'SIM to learner':>15} {'WER':>7}",
           f"{'learner real':>16} {ceiling:>15.3f} {m('wer_real'):>7.3f}",
           f"{'native tts':>16} {m('sim_native_tts'):>15.3f} "
           f"{m('wer_native_tts'):>7.3f}",
           f"{'learner tts':>16} {m('sim_learner_tts'):>15.3f} "
           f"{m('wer_learner_tts'):>7.3f}",
           f"{'golden':>16} {m('sim_golden'):>15.3f} {m('wer_golden'):>7.3f}",
           "",
           f"identity ceiling (real vs real, same speaker): {ceiling:.3f}  "
           f"— the SIM column is unreadable without it",
           f"headroom used by golden : {m('sim_golden') / ceiling:.1%} of ceiling",
           f"headroom used by cloning: {m('sim_learner_tts') / ceiling:.1%} of ceiling"]
    return "\n".join(out)
