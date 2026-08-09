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

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None) -> Tuple[np.ndarray, int]: ...

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

    def __init__(self, device: str = "cpu", **gen_kwargs):
        """``gen_kwargs`` are passed to ``generate`` on every call.

        Exposed so decode settings can be recorded and swept, not because the
        defaults need fixing. ``temperature=0.8`` looked like a lot of sampling
        entropy for an evaluation, and the obvious guess was that it explained
        why WER on LibriSpeech test-clean came out at 11% against the 1.5-3%
        published for zero-shot systems.

        **Measured, and the guess was wrong.** Lowering temperature makes both
        intelligibility and speaker similarity monotonically worse: WER 0.124 →
        0.162 → 0.276 and SIM-o 0.935 → 0.913 → 0.888 at temperature 0.8 → 0.4
        → 0.1. Autoregressive TTS degenerates when sampling gets too greedy, the
        same way greedy LLM decoding falls into repetition. The 0.8 default is
        doing real work.

        Recording which settings produced a number is the point — a WER without
        its decode configuration is not reproducible even with a seed.
        """
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        # S3 tokenizes the reference at 25 Hz, i.e. 40 ms per token.
        self.align_ms = 40.0
        self._device = device
        self._gen_kwargs = dict(gen_kwargs)

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Synthesize, optionally seeded for reproducibility.

        Chatterbox samples at ``temperature=0.8`` by default, so the same text
        and reference produce different audio every call. That was measured, not
        assumed: two identical evaluation runs gave AUROC 0.591 and 0.553 with
        the mean difference changing sign — an effect entirely inside sampling
        noise, which would have been reported as a finding.

        Seeding does not make the model deterministic in any deep sense; it
        makes a *run* reproducible, which is what a recorded number needs. Where
        the quantity of interest is the model's spread, run several seeds and
        report the range instead.
        """
        if seed is not None:
            import torch
            torch.manual_seed(seed)
        wav = self._model.generate(text, audio_prompt_path=reference_path,
                                   **self._gen_kwargs)
        arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
        return arr.astype(np.float32).ravel(), self.sample_rate

    def set_threads(self, n: int) -> None:
        import torch
        torch.set_num_threads(n)

    def describe(self) -> str:
        t3 = getattr(self._model, "t3", None)
        base = ("chatterbox (default checkpoint)" if t3 is None else
                f"chatterbox default, t3 backbone "
                f"{sum(p.numel() for p in t3.parameters()) / 1e6:.0f}M")
        if self._gen_kwargs:
            # Decode settings travel with the description so they land in the
            # provenance banner of every recorded run.
            args = ", ".join(f"{k}={v}" for k, v in
                             sorted(self._gen_kwargs.items()))
            return f"{base} [{args}]"
        return f"{base} [chatterbox defaults]"


BACKENDS = {"chatterbox": ChatterboxBackend}


def load_backend(name: str, device: str = "cpu", **gen_kwargs) -> Backend:
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name](device=device, **gen_kwargs)


def clone(backend: Backend, text: str, reference: np.ndarray, sr: int,
          out_path: Optional[str] = None,
          seed: Optional[int] = None) -> Tuple[np.ndarray, int, dict]:
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
        wav, out_sr = backend.synthesize(text, tmp.name, seed=seed)

    if out_path:
        sf.write(out_path, wav, out_sr)
    return wav, out_sr, report.to_dict()
