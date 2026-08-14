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

import sys
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
    # Preferred reference-clip length in seconds. A model property, measured per
    # backend — the widely-repeated "~10 s is the sweet spot" comes from
    # CosyVoice and dots.tts documentation and does NOT hold for Chatterbox,
    # where 20 s references cut WER by 24% on the utterances it affected.
    ref_seconds: Optional[float]
    # Whether the model needs a TRANSCRIPT of the reference clip, not just its
    # audio. Chatterbox does not; CosyVoice 2's `inference_zero_shot` takes a
    # `prompt_text` and conditions on it. Declared rather than inferred, because
    # a caller that passes nothing gets quietly degraded audio, not an error.
    needs_reference_text: bool

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None,
                   reference_text: Optional[str] = None
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

    def __init__(self, device: str = "cpu", **gen_kwargs):
        """``gen_kwargs`` are passed to ``generate`` on every call.

        Exposed so decode settings can be recorded and swept, not because the
        defaults need fixing. ``temperature=0.8`` looked like a lot of sampling
        entropy for an evaluation, and the obvious guess was that it explained
        why WER on LibriSpeech test-clean came out at 11% against the 1.5-3%
        published for zero-shot systems.

        **Measured, and the guess was wrong.** Below 0.8 both intelligibility and
        speaker similarity degrade sharply — WER 0.124 → 0.162 → 0.276 at
        temperature 0.8 → 0.4 → 0.1 — because autoregressive decoding degenerates
        when sampling gets too greedy, the same way greedy LLM decoding falls into
        repetition. Above 0.8 the curve plateaus (0.124 / 0.106 / 0.119 at
        0.8 / 1.0 / 1.2), so the default sits at the knee. No temperature closes
        the gap to published systems, which was the point of asking.

        Recording which settings produced a number is the point — a WER without
        its decode configuration is not reproducible even with a seed.
        """
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        # S3 tokenizes the reference at 25 Hz, i.e. 40 ms per token.
        self.align_ms = 40.0
        # Measured, not borrowed. Capping references at the commonly-cited 10 s
        # cost 24% relative WER on the half of a LibriSpeech sample whose
        # references exceeded it (0.126 -> 0.097), with the untruncated half
        # unchanged to three decimals as a control. The curve plateaus by 20 s.
        self.ref_seconds = 20.0
        # Conditions on reference audio alone; a transcript would be ignored.
        self.needs_reference_text = False
        self._device = device
        self._gen_kwargs = dict(gen_kwargs)

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None,
                   reference_text: Optional[str] = None
                   ) -> Tuple[np.ndarray, int]:
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


class CosyVoice2Backend:
    """FunAudioLLM CosyVoice 2 (0.5B) — Apache-2.0 code and **Apache-2.0 weights**.

    The best licence-to-quality ratio available: most competitive zero-shot models
    pair permissive code with restricted checkpoints (F5-TTS and OmniVoice ship
    `cc-by-nc-4.0` weights, XTTS-v2 is non-commercial for its outputs too). Both
    halves being Apache-2.0 was verified on the model card, not taken from a
    third-party summary — see docs/LICENSES.md.

    Not pip-installable. It is a recursive repo clone whose `third_party/Matcha-TTS`
    submodule must be on `sys.path`, so ``repo_dir`` points at that checkout. The
    upstream `requirements.txt` also targets CUDA (cu121 index, `tensorrt-cu12`,
    `onnxruntime-gpu`) and includes training-only packages; on a CPU box the
    inference subset is much smaller, because `deepspeed` is training-only,
    `vllm` is imported lazily inside methods, and `pyarrow` belongs to the dataset
    pipeline. Only `openai-whisper` is a genuine top-level inference dependency.

    **It needs the reference transcript.** `inference_zero_shot` takes a
    `prompt_text` alongside the prompt audio and conditions on it, so this backend
    sets ``needs_reference_text = True`` and refuses to guess when it is missing.
    """

    def __init__(self, device: str = "cpu", repo_dir: Optional[str] = None,
                 model_dir: Optional[str] = None, **gen_kwargs):
        import os
        import sys

        repo = repo_dir or os.environ.get("COSYVOICE_REPO", "")
        if not repo or not os.path.isdir(repo):
            raise SystemExit(
                "CosyVoice 2 needs the cloned repo: pass repo_dir= or set "
                "COSYVOICE_REPO. It is not installable from PyPI.")
        # Both the repo root and the Matcha-TTS submodule have to be importable.
        for p in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from cosyvoice.cli.cosyvoice import CosyVoice2

        weights = model_dir or os.environ.get(
            "COSYVOICE_MODEL",
            os.path.join(repo, "pretrained_models", "CosyVoice2-0.5B"))
        self._model = CosyVoice2(weights)
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000))
        # No public statement of a reference tokenizer frame size, and guessing
        # one would silently change how references are trimmed.
        self.align_ms = None
        # CosyVoice's own documentation is the source of the ~10 s figure, so
        # unlike Chatterbox (measured at 20 s) this is the model's stated
        # preference. Still worth measuring rather than trusting.
        self.ref_seconds = 10.0
        self.needs_reference_text = True
        self._device = device
        self._gen_kwargs = dict(gen_kwargs)

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None,
                   reference_text: Optional[str] = None
                   ) -> Tuple[np.ndarray, int]:
        if not reference_text:
            raise ValueError(
                "CosyVoice 2 conditions on a transcript of the reference clip; "
                "pass reference_text. Synthesizing without it degrades the "
                "output silently rather than failing.")
        if seed is not None:
            import torch
            torch.manual_seed(seed)

        # inference_zero_shot is a generator yielding chunks, one per synthesized
        # segment. Concatenate rather than taking the first, or long texts come
        # back truncated with no error.
        chunks = []
        for out in self._model.inference_zero_shot(
                text, reference_text, reference_path,
                stream=False, **self._gen_kwargs):
            wav = out["tts_speech"]
            arr = (wav.detach().cpu().numpy() if hasattr(wav, "detach")
                   else np.asarray(wav))
            chunks.append(arr.astype(np.float32).ravel())
        if not chunks:
            raise RuntimeError("CosyVoice 2 produced no audio")
        return np.concatenate(chunks), self.sample_rate

    def set_threads(self, n: int) -> None:
        import torch
        torch.set_num_threads(n)

    def describe(self) -> str:
        base = "cosyvoice2-0.5B (Apache-2.0 code + weights)"
        if self._gen_kwargs:
            args = ", ".join(f"{k}={v}" for k, v in sorted(self._gen_kwargs.items()))
            return f"{base} [{args}]"
        return f"{base} [cosyvoice defaults]"


class PiperBackend:
    """Piper — a **non-autoregressive** VITS model, here as an experimental control.

    Not a cloning backend: its voice is fixed by the checkpoint, so the reference
    audio is ignored and speaker similarity against a target is meaningless. It
    earns its place for one question the cloning backends cannot answer.

    Both other backends are transformer decoders — Chatterbox's T3, CosyVoice 2's
    Qwen — and both degrade badly on unpunctuated UPPER CASE input (5.1x and 2.0x
    WER on LibriSpeech). The obvious hypothesis is that this is a property of
    *LLM-style* TTS, which reads prose and finds corpus capitals out of
    distribution. Piper tests it: VITS has no autoregressive decoder, so it
    structurally cannot over-generate — the failure that produced outputs 2.42x
    their expected length. If Piper barely moves, the finding sharpens from "text
    presentation matters" to "text presentation matters for LLM-based TTS", which
    is the version that tells you which published comparisons to distrust.

    **Invoked as a subprocess, deliberately.** `piper-tts` >= 1.3 is
    `GPL-3.0-or-later` (the MIT `rhasspy/piper` was archived in October 2025).
    Importing it into this package and distributing the result would raise a real
    copyleft question; a process boundary is the accepted separation, and the CLI
    does everything needed. See docs/LICENSES.md.
    """

    def __init__(self, device: str = "cpu", voice: str = "en_US-lessac-medium",
                 data_dir: Optional[str] = None, **gen_kwargs):
        import shutil
        import subprocess

        self.voice = voice
        self._data_dir = data_dir
        self._python = shutil.which("python3") or sys.executable
        # Fail at construction, not at the first synthesis inside a study.
        probe = subprocess.run([self._python, "-m", "piper", "--help"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            raise SystemExit(
                "piper CLI not available: pip install piper-tts, then "
                f"python3 -m piper.download_voices {voice}\n{probe.stderr[:300]}")

        # Piper voices are 16 kHz (x_low/low) or 22.05 kHz (medium/high); read it
        # from the voice's own config rather than assuming, since a wrong rate
        # silently changes pitch and duration.
        self.sample_rate = self._read_voice_rate()
        self.align_ms = None
        self.ref_seconds = None
        self.needs_reference_text = False
        self._device = device
        self._gen_kwargs = dict(gen_kwargs)

    def _read_voice_rate(self) -> int:
        import glob
        import json
        import os
        roots = [self._data_dir] if self._data_dir else []
        roots += [os.path.expanduser("~/.local/share/piper"), os.getcwd()]
        for root in roots:
            if not root:
                continue
            for cfg in glob.glob(os.path.join(root, "**", f"{self.voice}.onnx.json"),
                                 recursive=True):
                with open(cfg) as fh:
                    return int(json.load(fh)["audio"]["sample_rate"])
        # Medium-quality voices, the common default, are 22.05 kHz.
        return 22050

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None,
                   reference_text: Optional[str] = None
                   ) -> Tuple[np.ndarray, int]:
        """``reference_path`` is accepted and ignored — Piper does not clone.

        Kept in the signature so this satisfies the same Protocol as the cloning
        backends and can be dropped into the same studies unchanged.
        """
        import subprocess
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            cmd = [self._python, "-m", "piper", "-m", self.voice, "-f", out.name]
            if self._data_dir:
                cmd += ["--data-dir", self._data_dir]
            cmd += ["--", text]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"piper failed: {r.stderr[:400]}")
            wav, sr = sf.read(out.name, dtype="float32")
        return np.asarray(wav, dtype=np.float32).ravel(), int(sr)

    def set_threads(self, n: int) -> None:
        # onnxruntime threading lives in the subprocess; nothing to set here, and
        # pretending otherwise would put a false claim in a provenance banner.
        pass

    def describe(self) -> str:
        return (f"piper {self.voice} (VITS, non-autoregressive; "
                f"GPL-3.0 CLI via subprocess)")


BACKENDS = {"chatterbox": ChatterboxBackend, "cosyvoice2": CosyVoice2Backend,
            "piper": PiperBackend}


def load_backend(name: str, device: str = "cpu", **gen_kwargs) -> Backend:
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; have {sorted(BACKENDS)}")
    return BACKENDS[name](device=device, **gen_kwargs)


def clone(backend: Backend, text: str, reference: np.ndarray, sr: int,
          out_path: Optional[str] = None, seed: Optional[int] = None,
          ref_seconds: Optional[float] = None,
          reference_text: Optional[str] = None) -> Tuple[np.ndarray, int, dict]:
    """Synthesize ``text`` in the voice of ``reference``.

    The reference goes through ``reference.prepare`` first — trimmed, resampled,
    length-capped, normalized — because that is where clone quality is actually
    determined. The report comes back with the audio so a batch job can record
    *why* a particular clip produced a poor clone instead of guessing later.

    ``reference_text`` is the transcript of the reference clip. Backends that
    declare ``needs_reference_text`` require it — CosyVoice 2 conditions on it —
    and this refuses rather than proceeding, because the failure mode is degraded
    audio scored as a model result rather than an exception.
    """
    import soundfile as sf

    if getattr(backend, "needs_reference_text", False) and not reference_text:
        raise ValueError(
            f"{backend.describe()} conditions on a transcript of the reference "
            f"clip, but reference_text was not given. Refusing rather than "
            f"synthesizing degraded audio that would be scored as a result.")

    # ref_seconds overrides the default cap. The default (10 s) comes from
    # CosyVoice and dots.tts documentation, which is not evidence about
    # Chatterbox — worth being able to test rather than assume.
    from speechforge.reference import TARGET_S
    cap = ref_seconds
    if cap is None:
        # Backend preference first, then the conservative default.
        cap = getattr(backend, "ref_seconds", None) or TARGET_S
    ref_audio, ref_sr, report = prepare(
        reference, sr, target_sr=backend.sample_rate, target_s=cap,
        align_ms=getattr(backend, "align_ms", None))
    if len(ref_audio) == 0:
        raise ValueError(f"unusable reference: {report.warnings}")

    # Backends take a path, not an array, so stage the cleaned clip.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, ref_audio, ref_sr)
        wav, out_sr = backend.synthesize(text, tmp.name, seed=seed,
                                         reference_text=reference_text)

    if out_path:
        sf.write(out_path, wav, out_sr)
    return wav, out_sr, report.to_dict()
