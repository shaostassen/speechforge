"""StyleTTS 2 — the diffusion backend.

**What the diffusion actually generates.** StyleTTS 2 is routinely called a
"diffusion TTS", and that is true in a narrower sense than it sounds. The
diffusion model does not produce audio or a spectrogram. It samples a 256-d
*style vector* — 128 acoustic dimensions (timbre, recording character) and 128
prosodic ones (rhythm, intonation, energy) — conditioned on the text's PL-BERT
embedding and, for the multi-speaker checkpoint, on the reference clip's style.
Everything downstream — durations, F0, the waveform — is a deterministic,
non-autoregressive network driven by that vector.

Upstream then *blends* the sample with the reference's own style:

    acoustic = alpha * diffused[:128] + (1 - alpha) * reference[:128]
    prosody  = beta  * diffused[128:] + (1 - beta)  * reference[128:]

so ``alpha`` and ``beta`` dial the diffusion's contribution from nothing
(``alpha = beta = 0``: the sample is computed and thrown away) to everything
(``alpha = beta = 1``). The defaults here, 0.3 and 0.7, are the ones in
upstream's LibriTTS demo. That makes the diffusion stage separately measurable,
which is the point of having it as a backend.

**The diffusion is not the only randomness.** With the sample discarded the
*style* is fixed, but the waveform still changes with the seed: the HiFi-GAN
decoder is a neural source-filter, and its ``SineGen`` draws random harmonic
phases and additive excitation noise on every call. This was assumed away in the
first draft of the ablation and caught by checking — two seeds at
``alpha = beta = 0`` gave identical durations and different samples. Note also
that ADPM2 is an *ancestral* sampler, drawing fresh noise at every step, so
changing ``diffusion_steps`` changes the decoder's noise draw as well.

**Not autoregressive.** Durations come from a predictor in one pass, so output
length is fixed by the text before any audio exists. That is the structural
property that rules out the over-generation measured on Chatterbox and
CosyVoice 2 — the same property Piper (VITS) has, from a different family.

**Licences.** Code is MIT. The LibriTTS checkpoint carries a use condition, not
a licence badge: listeners must be told the speech is synthetic unless the
voice's owner gave permission, and per the authors (StyleTTS2 issue #37) that
binds only for voices *outside* open-access datasets. Inference also needs
`phonemizer`, which is **GPL-3.0** — so, exactly as with Piper, it is an
optional extra and is **run as a subprocess** (its own ``phonemize`` CLI) rather
than imported. Upstream's own README notes the GPL dependency; the alternative,
the MIT `styletts2` package, swaps in gruut and the authors warn that the
phoneme mismatch costs quality. See docs/LICENSES.md.

**Setup.** Not on PyPI. ``repo_dir`` (or ``STYLETTS2_REPO``) is a clone of
``yl4579/StyleTTS2`` — it carries the text aligner, pitch extractor and PL-BERT
weights under ``Utils/`` — with the LibriTTS checkpoint from
``yl4579/StyleTTS2-LibriTTS`` placed at ``Models/LibriTTS/``.

Module import is torch-free: torch and the upstream code load in ``__init__``.
"""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The exact options of upstream's inference
#     EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
# as the CLI spells them. Byte-identical to the library call on every
# LibriSpeech test-clean transcript in both presentations; measured, see
# docs/VALIDATION.md. -q keeps warnings off stdout, which is the data channel.
# --preserve-empty-lines because the CLI otherwise DROPS empty input lines, which
# shifts every later text onto the wrong phonemes — found by the line-count check
# below, which is why it exists.
PHONEMIZE_ARGS = ("-q", "-l", "en-us", "-b", "espeak",
                  "--preserve-punctuation", "--with-stress",
                  "--preserve-empty-lines")

# Upstream trims this many samples off every output: "weird pulse at the end of
# the model". Kept, so the audio matches the reference implementation.
END_PULSE_SAMPLES = 50

DEFAULTS = {"alpha": 0.3, "beta": 0.7, "diffusion_steps": 5,
            "embedding_scale": 1.0}

# oneDNN (mkldnn) is switched off around this model's computation, and only
# there. With it on, resident memory grew ~150-200 MB per synthesis without
# bound — 2.0 -> 5.7 GB over 24 LibriSpeech utterances — while the count and
# size of live tensors stayed exactly constant. The growth needs varying input
# shapes (a repeated shape plateaus) and survives malloc_trim, a fixed mmap
# threshold, a single malloc arena and ONEDNN_PRIMITIVE_CACHE_CAPACITY=0; it
# disappears with mkldnn off. Every utterance has a new length, so a study of a
# few hundred clips was OOM-killed. Scoped with a context manager rather than
# the global flag, because the evaluation encoders run torch in the same process
# and must not have their numerics changed by loading a backend.
USE_MKLDNN = False


# --- phonemization (GPL boundary) -------------------------------------------

def phonemize(texts: Sequence[str], executable: Optional[str] = None,
              runner: Callable = subprocess.run) -> List[str]:
    """Phonemize via the ``phonemize`` CLI, one line in and one line out per text.

    A subprocess, deliberately: phonemizer is GPL-3.0 and importing it would put
    copyleft code in this package's process. The CLI reads newline-separated
    lines, so a newline inside a text would silently split it into two outputs
    and misalign every result after it — they are flattened to spaces first,
    and the line count is checked on the way back.
    """
    exe = executable or _phonemize_executable()
    lines = [" ".join(t.split()) for t in texts]
    r = runner([exe, *PHONEMIZE_ARGS], input="\n".join(lines) + "\n",
               capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"phonemize failed: {r.stderr[:400]}")
    out = r.stdout.split("\n")
    if out and out[-1] == "":
        out = out[:-1]
    if len(out) != len(lines):
        raise RuntimeError(
            f"phonemize returned {len(out)} lines for {len(lines)} inputs; "
            f"refusing to guess the alignment")
    return out


def _phonemize_executable() -> str:
    # Next to the running interpreter first: the CLI is installed into the same
    # environment as this package, and PATH often points at a system Python.
    local = os.path.join(os.path.dirname(sys.executable), "phonemize")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    found = shutil.which("phonemize")
    if found:
        return found
    raise SystemExit(
        "phonemize CLI not found: pip install -e '.[styletts2]' and install "
        "espeak-ng (apt install espeak-ng). It is GPL-3.0 and invoked as a "
        "subprocess; see docs/LICENSES.md.")


# --- tokenization -------------------------------------------------------------

def tokenize(phonemes: str, symbols: Dict[str, int],
             word_tokenize: Callable[[str], List[str]]) -> Tuple[List[int], List[str]]:
    """Upstream's text path: NLTK word tokenization, rejoined, mapped per char.

    Returns the token ids — with the leading pad token upstream inserts — and
    the characters that had no id. Upstream's ``TextCleaner`` prints and drops
    those; here they come back so a caller can see that the model was given
    less than the text.
    """
    ps = " ".join(word_tokenize(phonemes))
    ids, dropped = [0], []
    for ch in ps:
        if ch in symbols:
            ids.append(symbols[ch])
        else:
            dropped.append(ch)
    return ids, dropped


# --- long inputs ----------------------------------------------------------------

_BOUNDARIES = (".!?", ";:,—", " ")


def plan_chunks(text: str, measure: Callable[[str], int], limit: int,
                ) -> List[str]:
    """Split ``text`` so that every piece measures within ``limit`` tokens.

    PL-BERT has 512 position embeddings; a longer input is an index error deep
    inside the model rather than a clear failure. Long LibriSpeech utterances do
    exceed it. Splits prefer sentence ends, then clause punctuation, then any
    word boundary — each time at the boundary nearest the middle, so the pieces
    stay balanced and recursion stays shallow. Unpunctuated corpus capitals only
    ever split at spaces, which is itself a presentation effect worth knowing.
    """
    text = " ".join(text.split())
    if not text:
        return []
    if measure(text) <= limit:
        return [text]
    cut = _best_cut(text)
    if cut is None:
        raise ValueError(
            f"cannot split a single word of {measure(text)} tokens under the "
            f"{limit}-token limit: {text[:60]!r}")
    return (plan_chunks(text[:cut], measure, limit)
            + plan_chunks(text[cut:], measure, limit))


def _best_cut(text: str) -> Optional[int]:
    mid = len(text) / 2
    for marks in _BOUNDARIES:
        # A cut is the index just after a boundary character followed by a space
        # (or the space itself, for the word-boundary class).
        cands = [i + 1 for i, ch in enumerate(text[:-1])
                 if ch in marks and (ch == " " or text[i + 1] == " ")]
        cands = [c for c in cands if 0 < c < len(text)]
        if cands:
            return min(cands, key=lambda c: abs(c - mid))
    return None


# --- checkpoint loading -----------------------------------------------------------

# What the text-aligner checkpoint (Utils/ASR/epoch_00080.pth) needs, found by
# disassembling its pickle with pickletools rather than by trial. Its 'model'
# entry is plain tensors; the rest is pickled training state.
_REAL_GLOBALS = {
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("builtins", "dict"),
    ("__builtin__", "dict"),
}
# Training state the inference path never reads: replaced with inert stand-ins.
# `getattr` is the dangerous one — in a full unpickler it is the standard
# building block of an arbitrary-code-execution payload.
_INERT_GLOBALS = {
    ("builtins", "getattr"),
    ("__builtin__", "getattr"),
    ("torch.optim.adamw", "AdamW"),
    ("torch.optim.lr_scheduler", "OneCycleLR"),
}


class _Inert:
    """Stand-in for pickled training objects: accepts any state, keeps none."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _inert_call(*args, **kwargs):
    return _Inert()


class RestrictedUnpickler(pickle.Unpickler):
    """An allowlist unpickler, the pattern from the Python ``pickle`` docs.

    PyTorch >= 2.6 loads with ``weights_only=True``, and this checkpoint cannot
    pass it: its pickle uses a ``POP`` opcode the weights-only unpickler does not
    implement, so no allowlist rescues it. The only remaining stock option is
    ``weights_only=False``, which executes whatever the file says. This sits in
    between: real constructors for tensors and containers, inert stubs for the
    training state, and refusal for anything else.
    """

    def find_class(self, module: str, name: str):
        if (module, name) in _REAL_GLOBALS:
            return super().find_class(module, name)
        if (module, name) in _INERT_GLOBALS:
            return _inert_call if name == "getattr" else _Inert
        raise pickle.UnpicklingError(
            f"refusing to load global {module}.{name} from a checkpoint")


restricted_pickle = types.ModuleType("restricted_pickle")
restricted_pickle.Unpickler = RestrictedUnpickler
restricted_pickle.load = pickle.load   # torch never calls it, but probes for it
restricted_pickle.__name__ = "restricted_pickle"


def check_state_dict_load(result, name: str) -> None:
    """Fail on any missing or unexpected key.

    Upstream loads PL-BERT with ``strict=False`` and falls back to non-strict
    loading for the main model. If a library update renames a parameter, those
    weights silently stay at their random initialization and the model still
    produces audio — a broken backend that scores as a bad one.
    """
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            f"{name}: {len(missing)} missing and {len(unexpected)} unexpected "
            f"keys (first: {(missing + unexpected)[:3]}); refusing a partial load")


def validate_settings(alpha: float, beta: float, diffusion_steps: int,
                      embedding_scale: float) -> None:
    for k, v in (("alpha", alpha), ("beta", beta)):
        if not 0.0 <= v <= 1.0:
            # Outside [0, 1] the blend extrapolates past both the diffused and
            # the reference style; nothing upstream validates it either way.
            raise ValueError(f"{k}={v} is not a blend weight in [0, 1]")
    if int(diffusion_steps) != diffusion_steps or diffusion_steps < 1:
        raise ValueError(f"diffusion_steps={diffusion_steps} must be a positive integer")
    if embedding_scale <= 0:
        raise ValueError(f"embedding_scale={embedding_scale} must be positive")


# --- the backend ----------------------------------------------------------------------

class StyleTTS2Backend:
    """StyleTTS 2, LibriTTS multi-speaker checkpoint — zero-shot, style diffusion.

    ``gen_kwargs`` are ``alpha``, ``beta``, ``diffusion_steps`` and
    ``embedding_scale`` (classifier-free guidance on the text embedding), with
    upstream's LibriTTS demo values as defaults. All four always appear in
    ``describe()``: they are the knobs a diffusion study varies, so a recorded
    number without them is not reproducible.
    """

    def __init__(self, device: str = "cpu", repo_dir: Optional[str] = None,
                 checkpoint: Optional[str] = None, **gen_kwargs):
        unknown = set(gen_kwargs) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown StyleTTS 2 settings {sorted(unknown)}; "
                             f"have {sorted(DEFAULTS)}")
        self._settings = {**DEFAULTS, **gen_kwargs}
        self._settings["diffusion_steps"] = int(self._settings["diffusion_steps"])
        validate_settings(**self._settings)

        repo = repo_dir or os.environ.get("STYLETTS2_REPO", "")
        if not repo or not os.path.isdir(repo):
            raise SystemExit(
                "StyleTTS 2 needs the cloned repo: pass repo_dir= or set "
                "STYLETTS2_REPO to a checkout of yl4579/StyleTTS2. It is not "
                "installable from PyPI.")
        self._repo = os.path.abspath(repo)
        ckpt = checkpoint or os.environ.get(
            "STYLETTS2_CHECKPOINT",
            os.path.join(self._repo, "Models", "LibriTTS", "epochs_2nd_00020.pth"))
        if not os.path.isfile(ckpt):
            raise SystemExit(
                f"StyleTTS 2 checkpoint not found at {ckpt}; download "
                "Models/LibriTTS/{config.yml,epochs_2nd_00020.pth} from the "
                "yl4579/StyleTTS2-LibriTTS Hugging Face repo into the clone.")
        self._ckpt = ckpt

        # Fail at construction, not at the first synthesis inside a study.
        self._phonemize_exe = _phonemize_executable()
        try:
            import nltk
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            raise SystemExit("NLTK tokenizer data missing: "
                             "python -m nltk.downloader punkt_tab")
        from nltk.tokenize import word_tokenize
        self._word_tokenize = word_tokenize

        self._load(device)
        self.sample_rate = 24000
        # Nothing tokenizes the reference: two style encoders pool a mel
        # spectrogram over its whole length.
        self.align_ms = None
        # Measured, not borrowed — see docs/VALIDATION.md. The style encoders
        # average over time and nothing treats the reference as a prefix, so the
        # failure that made a 10 s cap catastrophic for CosyVoice 2 (cutting mid-
        # phrase into a continuation model) has no mechanism here.
        self.ref_seconds = 30.0
        self.needs_reference_text = False
        self._device = device
        self.last_synthesis: dict = {}

    # -- loading -----------------------------------------------------------------

    def _load(self, device: str) -> None:
        import importlib.util

        import torch
        import yaml

        if self._repo not in sys.path:
            sys.path.insert(0, self._repo)

        def from_file(name, rel):
            # Upstream's top-level module names (`models`, `text_utils`) are
            # generic enough to collide with other packages; load by path under
            # private names instead of trusting import order.
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(self._repo, rel))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        models = from_file("_styletts2_models", "models.py")
        text_utils = from_file("_styletts2_text_utils", "text_utils.py")
        from Modules.diffusion.sampler import (ADPM2Sampler, DiffusionSampler,
                                               KarrasSchedule)
        from Utils.ASR.models import ASRCNN

        cfg_path = os.path.join(os.path.dirname(self._ckpt), "config.yml")
        with open(cfg_path) as fh:
            config = yaml.safe_load(fh)

        def rel(p):
            return os.path.join(self._repo, p)

        # Text aligner: built here rather than through upstream's loader, which
        # calls torch.load in a way PyTorch 2.6 refuses for this file.
        with open(rel(config["ASR_config"])) as fh:
            asr = ASRCNN(**yaml.safe_load(fh)["model_params"])
        asr_state = torch.load(rel(config["ASR_path"]), map_location="cpu",
                               weights_only=False,
                               pickle_module=restricted_pickle)["model"]
        check_state_dict_load(asr.load_state_dict(asr_state), "text aligner")

        # Pitch extractor: plain tensors, loads under weights_only=True.
        pitch = models.load_F0_models(rel(config["F0_path"]))

        bert = self._load_plbert(rel(config["PLBERT_dir"]))

        mp = _munch(config["model_params"])
        self._decoder_type = mp.decoder.type
        model = models.build_model(mp, asr, pitch, bert)

        params = torch.load(self._ckpt, map_location="cpu")["net"]
        loaded = []
        for key in model:
            if key not in params:
                continue
            state = params[key]
            if all(k.startswith("module.") for k in state):
                # DataParallel prefix, which upstream strips on failure.
                state = {k[len("module."):]: v for k, v in state.items()}
            check_state_dict_load(model[key].load_state_dict(state), key)
            loaded.append(key)
        # Inference touches these modules; each must have come from the file.
        needed = {"bert", "bert_encoder", "predictor", "decoder", "text_encoder",
                  "predictor_encoder", "style_encoder", "diffusion"}
        if needed - set(loaded):
            raise RuntimeError(f"checkpoint lacks {sorted(needed - set(loaded))}")
        for key in model:
            model[key].eval().to(device)

        self._model = model
        self._torch = torch
        self._symbols = dict(text_utils.dicts)
        with open(rel(os.path.join(config["PLBERT_dir"], "config.yml"))) as fh:
            self._max_tokens = int(
                yaml.safe_load(fh)["model_params"]["max_position_embeddings"])
        # Upstream's empirical schedule for this checkpoint, unchanged.
        self._sampler = DiffusionSampler(
            model.diffusion.diffusion, sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False)
        import torchaudio
        self._to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    @staticmethod
    def _load_plbert(plbert_dir: str):
        import torch
        import yaml
        from transformers import AlbertConfig
        # Upstream's subclass returns last_hidden_state alone, which is what
        # bert_encoder consumes; a plain AlbertModel returns an output object.
        from Utils.PLBERT.util import CustomAlbert

        with open(os.path.join(plbert_dir, "config.yml")) as fh:
            plcfg = yaml.safe_load(fh)
        bert = CustomAlbert(AlbertConfig(**plcfg["model_params"]))
        steps = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(plbert_dir)
                 if f.startswith("step_")]
        state = torch.load(os.path.join(plbert_dir, f"step_{max(steps)}.t7"),
                           map_location="cpu")["net"]
        state = {k[len("module.encoder."):]: v for k, v in state.items()
                 if k.startswith("module.encoder.")}
        # A non-persistent buffer in current transformers, so not loadable;
        # upstream deletes it too.
        state.pop("embeddings.position_ids", None)
        check_state_dict_load(bert.load_state_dict(state, strict=False), "PL-BERT")
        return bert

    # -- synthesis -----------------------------------------------------------------

    def reference_style(self, reference_path: str):
        """Upstream's ``compute_style``: 24 kHz, trimmed at 30 dB, two encoders."""
        import librosa
        torch = self._torch
        wave, _sr = librosa.load(reference_path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        mel = self._to_mel(torch.from_numpy(audio).float())
        mel = ((torch.log(1e-5 + mel.unsqueeze(0)) + 4.0) / 4.0).to(self._device)
        with torch.no_grad():
            return torch.cat([self._model.style_encoder(mel.unsqueeze(1)),
                              self._model.predictor_encoder(mel.unsqueeze(1))],
                             dim=1)

    def _phonemes(self, text: str, cache: dict) -> str:
        # Chunk planning measures a text by phonemizing it, and synthesis needs
        # the same phonemes; the cache keeps that to one subprocess per text.
        if text not in cache:
            cache[text] = phonemize([text], self._phonemize_exe)[0]
        return cache[text]

    def synthesize(self, text: str, reference_path: str,
                   seed: Optional[int] = None,
                   reference_text: Optional[str] = None
                   ) -> Tuple[np.ndarray, int]:
        """``reference_text`` is accepted and ignored: style comes from audio alone.

        One seed drives both random draws in upstream's order — the diffusion
        noise, then the decoder's excitation — so a seeded run reproduces
        upstream's output exactly. Two consequences for studies: at the same seed
        and step count, arms that differ only in ``alpha``/``beta`` share the
        decoder's noise draw (the blend happens after sampling), which makes
        their per-utterance differences paired; and no setting makes the output
        seed-independent, ``alpha = beta = 0`` included.
        """
        if not text or not text.strip():
            raise ValueError("empty text")
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        cache: dict = {}
        chunks = plan_chunks(
            text, lambda t: len(tokenize(self._phonemes(t, cache), self._symbols,
                                         self._word_tokenize)[0]),
            self._max_tokens)
        phonemes = [self._phonemes(c, cache) for c in chunks]
        pieces, dropped, n_tokens = [], [], []
        with torch.backends.mkldnn.flags(enabled=USE_MKLDNN):
            ref_s = self.reference_style(reference_path)
            for ps in phonemes:
                ids, lost = tokenize(ps, self._symbols, self._word_tokenize)
                dropped.extend(lost)
                n_tokens.append(len(ids))
                pieces.append(self._infer(ids, ref_s))
        self.last_synthesis = {"chunks": len(chunks), "tokens": n_tokens,
                               "dropped_symbols": "".join(dropped),
                               "phonemes": phonemes}
        return np.concatenate(pieces).astype(np.float32), self.sample_rate

    def _infer(self, ids: List[int], ref_s) -> np.ndarray:
        """Upstream's ``inference``, line for line, with the settings injected."""
        torch = self._torch
        m = self._model
        s = self._settings
        dev = self._device
        tokens = torch.LongTensor(ids).to(dev).unsqueeze(0)
        with torch.no_grad():
            lengths = torch.LongTensor([tokens.shape[-1]]).to(dev)
            mask = torch.gt(torch.arange(lengths.max()).unsqueeze(0)
                            .expand(lengths.shape[0], -1).type_as(lengths) + 1,
                            lengths.unsqueeze(1)).to(dev)
            t_en = m.text_encoder(tokens, lengths, mask)
            bert_dur = m.bert(tokens, attention_mask=(~mask).int())
            d_en = m.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self._sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(dev),
                embedding=bert_dur, embedding_scale=s["embedding_scale"],
                features=ref_s, num_steps=s["diffusion_steps"]).squeeze(1)
            acoustic = s["alpha"] * s_pred[:, :128] + (1 - s["alpha"]) * ref_s[:, :128]
            prosody = s["beta"] * s_pred[:, 128:] + (1 - s["beta"]) * ref_s[:, 128:]

            d = m.predictor.text_encoder(d_en, prosody, lengths, mask)
            x, _ = m.predictor.lstm(d)
            duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze(0)).clamp(min=1)

            aln = torch.zeros(int(lengths), int(pred_dur.sum()))
            c = 0
            for i in range(aln.size(0)):
                aln[i, c:c + int(pred_dur[i])] = 1
                c += int(pred_dur[i])
            aln = aln.unsqueeze(0).to(dev)

            en = d.transpose(-1, -2) @ aln
            if self._decoder_type == "hifigan":
                en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
            f0, n = m.predictor.F0Ntrain(en, prosody)
            asr = t_en @ aln
            if self._decoder_type == "hifigan":
                asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
            out = m.decoder(asr, f0, n, acoustic.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-END_PULSE_SAMPLES]

    def set_threads(self, n: int) -> None:
        self._torch.set_num_threads(n)

    def describe(self) -> str:
        s = self._settings
        return (f"styletts2 LibriTTS (style diffusion, ADPM2, non-autoregressive; "
                f"phonemizer GPL-3.0 CLI via subprocess) "
                f"[alpha={s['alpha']}, beta={s['beta']}, "
                f"diffusion_steps={s['diffusion_steps']}, "
                f"embedding_scale={s['embedding_scale']}]")


def _munch(d):
    from munch import Munch
    if isinstance(d, dict):
        return Munch({k: _munch(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_munch(v) for v in d]
    return d
