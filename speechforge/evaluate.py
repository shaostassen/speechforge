"""Scoring synthesized speech, with the traps built into the API.

Three axes, and each has a way of flattering a system that this module tries to
make structurally hard:

**Intelligibility — WER of an ASR system on the synthesized audio.** The trap:
modern zero-shot TTS scores *better* ASR-WER than real human speech, because it
produces cleaner, more canonical articulation than a person does. A published
case has a system at 1.47% against 1.62% for ground truth. So a low WER is not
evidence of good synthesis past the human line — below it the metric has stopped
measuring intelligibility and started measuring "how ASR-friendly is this".
Hence `EvalTable` refuses to render without a ground-truth row.

**Speaker similarity — cosine distance between speaker embeddings.** Two traps.
Compare against the *reference clip itself* and you measure how well the model
copied a specific recording, which inflates the number; the honest comparison is
a held-out utterance from the same speaker (SIM-o, not SIM-r). And the ceiling is
not 1.0 — real human speech from the same speaker scores about 0.69-0.76 against
itself on these metrics, so 0.65 is near parity and 0.9 means you mismeasured.

**Naturalness — predicted MOS.** Reported, and explicitly not used for ranking:
UTMOS-style predictors correlate poorly at the system level and saturate on good
audio. It is a smoke alarm, not a judge.

Import-light: ASR and torch are pulled in lazily inside the scorers, so a run
that only needs the table maths does not need either.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# Published zero-shot TTS papers report a same-speaker similarity ceiling around
# 0.69-0.76. That figure is ENCODER-SPECIFIC and does not transfer: measured with
# `microsoft/wavlm-base-plus-sv`, same-speaker similarity runs 0.89 mean (studio
# audio reaches 0.98) and the different-speaker floor is 0.68.
#
# So this constant is orientation for reading papers, never a threshold. An
# earlier version used it to warn "above 0.73, suspect the measurement", which
# would have fired on every legitimate run with this encoder — a gate that
# always fires is as useless as one that never does. The comparison that
# actually holds is against the ground-truth row, measured with the same
# encoder on the same audio.
PUBLISHED_SIM_CEILING = (0.69, 0.76)


class GroundTruthMissing(RuntimeError):
    """Raised when a table would render without its ground-truth row."""


class ScorerUnavailable(RuntimeError):
    """Raised when a scorer the run asked for cannot be loaded."""


@dataclass
class SystemScores:
    name: str
    wer: Optional[float] = None
    wer_second: Optional[float] = None     # a non-Whisper ASR, see below
    sim_o: Optional[float] = None
    mos: Optional[float] = None
    n_utterances: int = 0
    notes: str = ""
    # Prosody (speechforge.prosody): per-utterance means.
    pitch_range_st: Optional[float] = None
    pitch_gap_st: Optional[float] = None
    words_per_s: Optional[float] = None
    jump_fraction: Optional[float] = None   # tracker diagnostic, not prosody

    def to_dict(self) -> dict:
        return {"name": self.name, "wer": self.wer,
                "wer_second": self.wer_second, "sim_o": self.sim_o,
                "mos": self.mos, "n_utterances": self.n_utterances,
                "notes": self.notes, "pitch_range_st": self.pitch_range_st,
                "pitch_gap_st": self.pitch_gap_st,
                "words_per_s": self.words_per_s,
                "jump_fraction": self.jump_fraction}

    def has_prosody(self) -> bool:
        return any(v is not None for v in (self.pitch_range_st, self.pitch_gap_st,
                                           self.words_per_s))


# Pitch-tracking errors start to distort the pitch range past this share of
# octave-sized frame steps. Each wrong frame is a jump in and a jump out, so
# 4% of steps is ~2% of frames, where the range already reads ~4% high (and at
# 5% of frames, 12% high — tests/test_prosody.py).
JUMP_WARN = 0.04


@dataclass
class EvalTable:
    """Results for several systems, plus the ground truth they are read against.

    The ground-truth row is mandatory. It is the only thing that makes any of
    the other numbers interpretable, and it is the row people omit — so
    `render()` raises rather than quietly producing a table that reads as
    stronger than the evidence supports.
    """
    rows: List[SystemScores] = field(default_factory=list)
    ground_truth: Optional[SystemScores] = None
    corpus: str = ""
    hardware: str = ""

    def add(self, scores: SystemScores) -> None:
        self.rows.append(scores)

    def set_ground_truth(self, scores: SystemScores) -> None:
        self.ground_truth = scores

    def render(self) -> str:
        if self.ground_truth is None:
            raise GroundTruthMissing(
                "no ground-truth row: synthesized speech can score better "
                "ASR-WER than real human speech, so a WER without the human "
                "row beside it cannot be read as intelligibility. Score the "
                "real recordings and call set_ground_truth().")

        head = (f"{'system':>22} {'n':>5} {'WER':>7} {'WER-2':>7} "
                f"{'SIM-o':>7} {'MOS*':>6}")
        lines = []
        if self.corpus or self.hardware:
            lines.append(f"corpus={self.corpus}  hardware={self.hardware}")
        lines += [head, "-" * len(head)]

        def fmt(s: SystemScores) -> str:
            def n(v, p=3):
                return f"{v:>7.{p}f}" if v is not None else f"{'—':>7}"
            mos = f"{s.mos:>6.2f}" if s.mos is not None else f"{'—':>6}"
            return (f"{s.name:>22} {s.n_utterances:>5} {n(s.wer)} "
                    f"{n(s.wer_second)} {n(s.sim_o)} {mos}")

        gt = self.ground_truth
        lines.append(fmt(gt) + "   <- read every row against this")
        for r in self.rows:
            flags = []
            if (r.wer is not None and gt.wer is not None and r.wer < gt.wer):
                flags.append("WER below human: not an intelligibility win")
            # Compared against the measured ground truth, not a published
            # constant: the ceiling depends on the encoder and the recording
            # conditions, so only the same-encoder same-audio number is a
            # meaningful bound. A clone matching a speaker better than their
            # own other recordings do means the comparison is wrong.
            if (r.sim_o is not None and gt.sim_o is not None
                    and r.sim_o > gt.sim_o):
                flags.append("SIM above ground truth: check it is a held-out "
                             "clip, not the reference")
            line = fmt(r)
            if flags:
                line += "   (" + "; ".join(flags) + ")"
            lines.append(line)

        lines.append("")
        lines.append("* MOS is a predicted proxy, reported as a sanity check "
                     "only — it correlates")
        lines.append("  poorly at system level and cannot rank systems.")
        if gt.sim_o is not None:
            lines.append(f"  SIM-o is encoder-specific. The ceiling here is the "
                         f"ground-truth row ({gt.sim_o:.3f}), not the "
                         f"{PUBLISHED_SIM_CEILING[0]:.2f}-"
                         f"{PUBLISHED_SIM_CEILING[1]:.2f} seen in papers, which "
                         f"used a different encoder.")
        if gt.has_prosody() or any(r.has_prosody() for r in self.rows):
            lines += [""] + self._render_prosody()
        return "\n".join(lines)

    def _render_prosody(self) -> List[str]:
        # A separate block, shown only when prosody was measured, so every table
        # rendered before it existed renders exactly as it did.
        head = (f"{'prosody':>22} {'n':>5} {'pitch range':>12} {'pitch gap':>10} "
                f"{'words/s':>8} {'jumps':>7}")
        lines = [head, "-" * len(head)]

        def fmt(s: SystemScores) -> str:
            def n(v, w, p):
                return f"{v:>{w}.{p}f}" if v is not None else f"{'—':>{w}}"
            jumps = (f"{100 * s.jump_fraction:>6.2f}%" if s.jump_fraction is not None
                     else f"{'—':>7}")
            return (f"{s.name:>22} {s.n_utterances:>5} {n(s.pitch_range_st, 12, 2)} "
                    f"{n(s.pitch_gap_st, 10, 2)} {n(s.words_per_s, 8, 2)} {jumps}")

        gt = self.ground_truth
        lines.append(fmt(gt) + "   <- read every row against this")
        for r in self.rows:
            line = fmt(r)
            if r.jump_fraction is not None and r.jump_fraction > JUMP_WARN:
                line += ("   (pitch tracking unreliable on this audio: "
                         "check before reading the pitch columns)")
            lines.append(line)
        lines += [
            "",
            "  pitch range: p10-p90 of F0 within an utterance, semitones. Below the",
            "    ground truth is flatter than the speakers, above is livelier; the",
            "    target is the human row, not a direction.",
            "  pitch gap: semitones between median F0 and the held-out recording's.",
            "    The ground-truth row is another recording of the same speaker.",
            "  words/s: over the speech span, leading and trailing silence trimmed.",
            "  jumps: share of 10 ms steps over 9 semitones — tracker errors, not",
            f"    prosody. Flagged above {100 * JUMP_WARN:.0f}%.",
        ]
        return lines


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, guarding the degenerate cases.

    Returns nan for a zero vector rather than 0.0: a zero embedding means the
    encoder failed, and 0.0 would read as "completely different voices", which
    is a measurement passing itself off as a result.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"embedding shape mismatch: {x.shape} vs {y.shape}")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0.0 or ny == 0.0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def sim_o(embed, synth_audio, synth_sr, held_out_audio, held_out_sr,
          reference_audio: Optional[np.ndarray] = None) -> float:
    """Speaker similarity against a **held-out** utterance, not the reference.

    ``embed`` is any callable ``(audio, sr) -> vector``.

    Passing ``reference_audio`` opts into a guard: if the held-out clip is
    actually the reference clip, this raises. Comparing a clone to the very clip
    it was conditioned on is SIM-r, it inflates the score, and it is an easy
    mistake to make when both live in the same variable.
    """
    if reference_audio is not None:
        ho = np.asarray(held_out_audio).ravel()
        ref = np.asarray(reference_audio).ravel()
        if ho.shape == ref.shape and np.allclose(ho, ref):
            raise ValueError(
                "held-out clip is the reference clip: that measures how well "
                "the model copied one recording (SIM-r), not whether it "
                "captured the speaker (SIM-o). Use a different utterance.")
    return cosine(embed(synth_audio, synth_sr),
                  embed(held_out_audio, held_out_sr))


def score_intelligibility(transcribe, pairs: Sequence[tuple]) -> float:
    """Corpus WER over ``(audio, sr, reference_text)`` triples.

    ``transcribe`` is ``(audio, sr) -> str``. WER is computed on the
    concatenation rather than averaged per utterance, which is the convention
    the ASR project already uses and what published TTS numbers report — a
    per-utterance mean over-weights short utterances.
    """
    from speechlens.metrics import wer

    refs, hyps = [], []
    for audio, sr, ref in pairs:
        refs.append(ref)
        hyps.append(transcribe(audio, sr))
    if not refs:
        return float("nan")
    return wer(" ".join(refs), " ".join(hyps))


def require_scorer(what: str, load, opt_out: str):
    """Load a scorer the run asked for, or stop the run with ``ScorerUnavailable``.

    The trap: a column that comes back blank still renders. The table, the
    provenance banner and the exit code all look like a finished run, and the
    gap is found later, by whoever reads the log closely enough. This harness
    used to warn and carry on; a speaker-encoder download blocked by the network
    then let a study run on with no SIM column, and it was caught only by
    reading the log mid-run. So a requested scorer is either measured or the run
    stops before any synthesis, and leaving a column out is a decision made up
    front — ``opt_out`` says how, and goes in the error.
    """
    try:
        return load()
    except Exception as exc:
        raise ScorerUnavailable(
            f"{what} failed to load ({type(exc).__name__}: {exc}). Fix that, "
            f"or {opt_out} to run without it.") from exc


def whisper_transcriber(model_size: str = "large-v3", device: str = "auto",
                        compute_type: str = "auto", language: str = "en"):
    """A ``(audio, sr) -> str`` backed by speechlens.

    This is the field-standard intelligibility evaluator — F5-TTS's own
    evaluation uses the same faster-whisper large-v3 stack — so numbers from it
    are comparable to published ones.

    Caveat worth stating in any write-up: scoring TTS with Whisper when Whisper
    is also the ASR everyone tunes against measures partly your own tail. Report
    a second, architecturally different ASR alongside it (`wer_second`).
    """
    from speechlens.asr import RobustnessConfig
    from speechlens.pipeline import SpeechLens

    lens = SpeechLens(model_size=model_size, device=device,
                      compute_type=compute_type)
    cfg = RobustnessConfig()

    def transcribe(audio: np.ndarray, sr: int) -> str:
        return lens.analyze((audio, sr), language=language, cfg=cfg).text

    return transcribe


def mos_predictor(device: str = "cpu"):
    """Predicted MOS via UTMOS (SpeechMOS), as a smoke alarm — not a judge.

    Reported because naturalness is the one axis WER and speaker similarity say
    nothing about, and human listening tests are not available here. But the
    literature is clear that predicted MOS correlates poorly with human judgement
    *at the system level* and saturates on good audio, so it cannot rank systems.
    It is useful for one thing: catching output that is broken in a way WER does
    not see — buzzing, truncation, wrong speaking rate.

    Returns a callable ``(audio, sr) -> float`` or raises. A caller that asked
    for MOS should load it through ``require_scorer`` rather than blank the
    column. Loaded through torch.hub, which needs network on first use.
    """
    import torch

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True)
    predictor.eval()
    predictor.to(device)
    target_sr = 16000

    def predict(audio: np.ndarray, sr: int) -> float:
        y = np.asarray(audio, dtype=np.float32).ravel()
        if sr != target_sr:
            from speechforge.reference import resample
            y = resample(y, sr, target_sr)
        with torch.no_grad():
            score = predictor(torch.from_numpy(y).unsqueeze(0).to(device),
                              target_sr)
        return float(score.item())

    return predict


def score_naturalness(predict, clips: Sequence[tuple]) -> float:
    """Mean predicted MOS over ``(audio, sr)`` pairs.

    A mean, deliberately: per-clip predicted MOS is noisier than the differences
    anyone cares about, and reporting a spread would imply a precision the
    predictor does not have.
    """
    scores = [predict(a, sr) for a, sr in clips]
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


def ctc_transcriber(model_name: str = "facebook/hubert-large-ls960-ft",
                    device: str = "cpu"):
    """A second, architecturally different ASR — HuBERT + CTC.

    Why bother, when Whisper is the field standard: scoring TTS with Whisper
    when Whisper is also the model everyone tunes against measures partly your
    own tail. A synthetic voice that happens to suit Whisper's priors scores well
    without being more intelligible to anything else.

    HuBERT-CTC is a genuinely different bet — encoder-only, frame-synchronous,
    no autoregressive language model on top. It has no internal LM to paper over
    unclear articulation, so it is *harsher* and its absolute WER runs higher.
    That is the point: agreement between the two is evidence, and disagreement
    localizes the problem. Read the pair, never either alone.
    """
    import torch
    from transformers import HubertForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = HubertForCTC.from_pretrained(model_name)
    model.eval()
    model.to(device)
    target_sr = 16000

    def transcribe(audio: np.ndarray, sr: int) -> str:
        y = np.asarray(audio, dtype=np.float32).ravel()
        if sr != target_sr:
            from speechforge.reference import resample
            y = resample(y, sr, target_sr)
        inputs = processor(y, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        # Uppercase, unpunctuated output — speechlens.normalize_text folds that
        # against Whisper's cased, punctuated text, so WER stays comparable.
        return processor.batch_decode(ids)[0]

    return transcribe
