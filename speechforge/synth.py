"""One interface over the synthesis backends.

Swapping models must be a config change, not a rewrite — partly for the obvious
reason, and partly because the licence situation is unstable enough that a
backend may have to be dropped. Fish Speech went from CC-BY-NC-SA to
research-only; ZipVoice may become viable if its weights licence is clarified.
Nothing above this layer should care.

Backends are imported lazily. Each pins its own torch and transformers versions
and they conflict, so importing this module must not import any of them.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np

from speechforge.reference import prepare


class Backend(Protocol):
    """What a synthesis backend has to provide."""

    sample_rate: int
    # Frame size the model tokenizes the reference at, in ms, or None. Declared
    # by the backend rather than assumed by the caller: it is a property of the
    # model's tokenizer, not of the audio.
    align_ms: Optional[float]

    def synthesize(self, text: str, reference_path: str
                   ) -> Tuple[np.ndarray, int]: ...

    def set_threads(self, n: int) -> None: ...

    def describe(self) -> str: ...


class ChatterboxBackend:
    """Resemble AI Chatterbox — MIT code and MIT weights.

    Two things to know. Every output is watermarked with Resemble's PerTh neural
    watermarker, unconditionally; that is fine but the audio is not pristine, so
    do not use these outputs as a clean signal-analysis baseline.

    And the released package cannot load the 110M `chatterbox-nano` checkpoint:
    `from_pretrained` takes only `device`, with a hardcoded repo id, and Nano
    ships an incompatible file layout. What loads is the 532M default, measured
    at RTF ~1.9-2.6 on a Ryzen 9 9950X — about 2x slower than realtime. See
    docs/VALIDATION.md.
    """

    def __init__(self, device: str = "cpu"):
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        # S3 tokenizes the reference at 25 Hz, i.e. 40 ms per token.
        self.align_ms = 40.0
        self._device = device

    def synthesize(self, text: str, reference_path: str
                   ) -> Tuple[np.ndarray, int]:
        wav = self._model.generate(text, audio_prompt_path=reference_path)
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return arr.astype(np.float32).ravel(), self.sample_rate

    def set_threads(self, n: int) -> None:
        import torch
        torch.set_num_threads(n)

    def describe(self) -> str:
        t3 = getattr(self._model, "t3", None)
        if t3 is None:
            return "chatterbox (default checkpoint)"
        params = sum(p.numel() for p in t3.parameters())
        return f"chatterbox default, t3 backbone {params / 1e6:.0f}M"


BACKENDS = {"chatterbox": ChatterboxBackend}


def load_backend(name: str, device: str = "cpu") -> Backend:
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name](device=device)


def clone(backend: Backend, text: str, reference: np.ndarray, sr: int,
          out_path: Optional[str] = None) -> Tuple[np.ndarray, int, dict]:
    """Synthesize ``text`` in the voice of ``reference``.

    The reference goes through ``reference.prepare`` first — trimmed, resampled,
    length-capped, normalized — because that is where clone quality is actually
    determined. The report comes back with the audio so a batch job can record
    *why* a particular clip produced a poor clone instead of guessing later.
    """
    import soundfile as sf

    ref_audio, ref_sr, report = prepare(
        reference, sr, target_sr=backend.sample_rate,
        align_ms=getattr(backend, "align_ms", None))
    if len(ref_audio) == 0:
        raise ValueError(f"unusable reference: {report.warnings}")

    # Backends take a path, not an array, so stage the cleaned clip.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, ref_audio, ref_sr)
        wav, out_sr = backend.synthesize(text, tmp.name)

    if out_path:
        sf.write(out_path, wav, out_sr)
    return wav, out_sr, report.to_dict()
