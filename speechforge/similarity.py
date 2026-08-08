"""Speaker embeddings, for measuring whether a clone captured the voice.

The field standard for zero-shot TTS is a WavLM-large model fine-tuned for
speaker verification, and cosine distance between its embeddings. Numbers from a
*different* encoder are not comparable to published ones — this is the single
biggest reason the same system gets different SIM figures in different papers —
so the encoder identity is recorded alongside every score.

Two things to keep in mind while reading any number this produces:

**The ceiling is not 1.0.** Real human speech from the same speaker scores about
0.69-0.76 against itself on these metrics. A clone at 0.65 is close to parity; a
clone at 0.90 almost certainly means the comparison was against the reference
clip itself rather than a held-out utterance.

**Embeddings are speaker-verification features, not perceptual ones.** They can
be fooled by channel and recording conditions, which is exactly why the
reference pipeline normalizes level and resamples: an untreated channel
difference shows up as a speaker difference.

torch is imported lazily so the core install stays free of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# WavLM-base-plus fine-tuned for speaker verification. Chosen over the
# `wavlm_large_finetune.pth` checkpoint used in several papers because that one
# is distributed as a bare state dict via a Google Drive link with no licence
# statement, which this project will not depend on. The cost is that scores are
# NOT directly comparable to papers using the large checkpoint — recorded here
# and in every table rather than glossed over.
DEFAULT_MODEL = "microsoft/wavlm-base-plus-sv"
MODEL_SR = 16000


@dataclass
class SpeakerEncoder:
    """Callable ``(audio, sr) -> embedding``, matching what ``sim_o`` expects."""
    model_name: str = DEFAULT_MODEL
    device: str = "cpu"
    _model: object = None
    _extractor: object = None

    def load(self) -> "SpeakerEncoder":
        import torch
        from transformers import (AutoFeatureExtractor,
                                  WavLMForXVector)
        self._extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = WavLMForXVector.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self._torch = torch
        return self

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self._model is None:
            self.load()
        y = np.asarray(audio, dtype=np.float32).ravel()
        if sr != MODEL_SR:
            # The encoder was trained at 16 kHz; feeding it anything else
            # shifts every formant and the "speaker difference" it reports is
            # really a sample-rate difference.
            from speechforge.reference import resample
            y = resample(y, sr, MODEL_SR)
        inputs = self._extractor(y, sampling_rate=MODEL_SR,
                                 return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
        return out.embeddings[0].cpu().numpy().astype(np.float64)

    def describe(self) -> str:
        return f"{self.model_name} @ {MODEL_SR} Hz"


def load_encoder(model_name: str = DEFAULT_MODEL,
                 device: str = "cpu") -> SpeakerEncoder:
    return SpeakerEncoder(model_name=model_name, device=device).load()


def self_similarity_baseline(encoder, utterances, sr: int) -> Optional[float]:
    """Mean pairwise similarity between real recordings of the same speaker.

    **Compute this before reading any clone score.** It is the ceiling for that
    speaker, that encoder, and those recording conditions — the "ground truth"
    row for the similarity column. Published ceilings (~0.69-0.76) come from a
    different encoder and different audio, so they orient but do not substitute.

    Returns None if given fewer than two utterances, since there is no pair.
    """
    from speechforge.evaluate import cosine

    embs = [encoder(u, sr) for u in utterances]
    if len(embs) < 2:
        return None
    scores = [cosine(embs[i], embs[j])
              for i in range(len(embs)) for j in range(i + 1, len(embs))]
    return float(np.mean(scores)) if scores else None
