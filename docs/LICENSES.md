# Model licence audit

This repo is public. Voice-cloning models are unusually treacherous on licensing,
because **the restriction usually lives in the weights, not the code** — several
projects ship MIT or Apache-2.0 source with CC-BY-NC checkpoints, because they were
trained on the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset)
in-the-wild dataset. Third-party write-ups routinely call these models "MIT" or
"Apache" on the strength of the repo badge alone, and are wrong about the thing you
would actually be shipping.

So: every model is recorded here with **both** licences, and anything that cannot be
used in a public portfolio project is listed with the reason rather than quietly
omitted. Audited 2026-08-07.

## This project's own licence

**MIT** (`LICENSE`), same as the SpeechLens package it depends on.

Nothing in the dependency set constrains that choice, which is the point of the audit
below. Chatterbox is MIT on code *and* weights, CosyVoice 2 is Apache-2.0 on both, and
the one copyleft dependency — Piper, `GPL-3.0-or-later` since the MIT repo was archived
— is **invoked as a subprocess rather than imported**, so no GPL code is linked into
this package and none is installed unless you ask for the `piper` extra.

The models this project *drives* keep their own terms regardless, and so do their
weights. An MIT licence here covers the harness, not the checkpoints it loads.

## In use

| Model | Code | Weights | Role |
|---|---|---|---|
| **Chatterbox** (Resemble AI) | MIT | **MIT** | Local CPU cloning + voice conversion |
| **CosyVoice 2** (0.5B) | Apache-2.0 | **Apache-2.0** | Second backend — quality comparison |
| **Piper** (`piper-tts` ≥1.3) | **GPL-3.0-or-later** | per-voice, mostly CC BY 4.0 | Non-LLM control for the text-presentation audit |
| **kNN-VC** | MIT | MIT | Training-free, explainable VC baseline |

Chatterbox is the only family found with permissive *weights* and a vendor claim of
better-than-realtime CPU synthesis. CosyVoice 2 has the best licence-to-quality ratio
in the field — genuinely Apache-2.0 on both sides.

**CosyVoice 2's Apache-2.0 weights licence was re-verified on the model card itself**
(`FunAudioLLM/CosyVoice2-0.5B`, checked 2026-08-11), not taken from a third-party
summary — which is the same discipline that caught F5-TTS and OmniVoice shipping
permissive code with `cc-by-nc-4.0` checkpoints. It holds.

Three practical notes, all found by installing it rather than reading about it:

- **It is not on PyPI.** It installs as a recursive `git clone` whose
  `third_party/Matcha-TTS` submodule must be on `sys.path`. The backend takes a
  `repo_dir` for exactly this reason.
- **Its `requirements.txt` targets CUDA** — a `cu121` extra index, `tensorrt-cu12`,
  `onnxruntime-gpu` — and mixes in training-only packages. Installed verbatim on a
  CPU box that is gigabytes of unusable wheels. The inference subset is much smaller:
  `deepspeed` is training-only, `vllm` is imported lazily inside methods, `pyarrow`
  belongs to the dataset pipeline. Only `openai-whisper` is a genuine top-level
  inference dependency, and it needs `setuptools<81` to build (newer setuptools drops
  `pkg_resources`).
- **It needs the reference transcript**, not just reference audio, which is a real
  API difference from Chatterbox rather than a detail — see `needs_reference_text` in
  `speechforge/synth.py`.

Two notes on Chatterbox that are easy to miss:

- **Every output is watermarked** with Resemble's PerTh neural watermarker. Harmless,
  arguably a responsible-AI feature, but worth knowing in a project that also does
  signal analysis — the audio is not pristine.
- **The released package cannot load `chatterbox-nano`**: `from_pretrained` takes only
  `device` with a hardcoded repo id, and Nano ships an incompatible file layout. Its
  MIT weights are real; the loader is the obstacle.

## Excluded, and why

| Model | Problem |
|---|---|
| **XTTS-v2** (Coqui) | **CPML.** Non-commercial for the model *and its outputs* — it prohibits "any direct or indirect payment arising from the use of the model or its output". Also RTF ~1.41 at 32 CPU threads, so not realtime here anyway. |
| **F5-TTS** weights | Code is MIT; **weights are `cc-by-nc-4.0`** by the authors' own README, due to Emilia. Also RTF ~37 on a single CPU thread — Colab-only regardless of licence. |
| **OmniVoice** weights | Code is Apache-2.0; **weights are CC-BY-NC**, again Emilia. The most widely mis-reported licence found. Also no ONNX path and slow on CPU per the maintainer. |
| **Fish Speech / OpenAudio** | **Research-only.** The "Fish Audio Research License" requires a separate written agreement for commercial use and explicitly sweeps in fine-tunes and LoRA derivatives. Note this is a *tightening* — it was CC-BY-NC-SA historically. |
| **IndexTTS-2** | Custom bilibili licence, not OSI-approved. The thresholds (100M MAU / RMB 1B revenue) mean it is almost certainly fine in practice, but it is not open source and would need disclosing. Technically interesting: it decouples speaker and emotion references, which is the cleanest fix for prosody leakage. |
| **Higgs Audio v2/v3** | Repo is Apache-2.0 but the **HuggingFace weights are tagged `license: other`**. Multiple 2026 articles assert Apache-2.0 commercial use; that appears to be wrong for the checkpoints. |
| **Supertonic** | MIT sample code, OpenRAIL-M models, and zero-shot cloning is behind a paid tier — the open repo has no cloning pipeline. |

## Open question

**ZipVoice / ZipVoice-Distill** (`k2-fsa/ZipVoice`) is technically the best fit on
paper: 123M params, official ONNX + INT8, **RTF 1.22 on a single CPU thread**, and
speaker similarity at the human ceiling (0.751 vs 0.755 GT on Seed-TTS test-zh). If its
licence were clear it would displace Chatterbox as the primary local backend.

But **no licence is declared for the weights** — not in the HF metadata, not in the
README. The main checkpoint is Emilia-trained, so assume CC-BY-NC until told otherwise.
There is also a `ZipVoice-LibriTTS` checkpoint, and LibriTTS is CC BY 4.0, so that
variant is *plausibly* clean — unconfirmed.

**Action:** open an issue on the repo asking (a) the weights licence and (b) whether the
LibriTTS checkpoint escapes the Emilia constraint. Revisit this table with the answer.
The repo has also been quiet since 2025-12.

## Standing rule

Licences here drift — Fish Speech is the proof. Re-read the actual `LICENSE` file and
the HuggingFace `cardData.license` field before any public claim or release, and update
this table with the date checked. A repo badge is not evidence.

## Piper relicensed from MIT to GPL-3.0 — caught by re-checking, not assumed

This document existed because licence drift is real, and Piper is the second case
after Fish Speech. The MIT `rhasspy/piper` repository was **archived read-only in
October 2025**; development moved to `OHF-Voice/piper1-gpl`, and what
`pip install piper-tts` fetches today (1.6.1) declares **`GPL-3.0-or-later`**,
verified from PyPI's own metadata rather than a third-party summary or the older
README. The earlier entry in this file said MIT, which was true when written and is
now wrong — the fifth documented constant this project adopted and later found not
to hold.

**Consequence for this repo, and how it is handled.** GPL-3.0 is copyleft: importing
Piper into `speechforge` and distributing the result would raise a real question about
whether `speechforge` must itself be GPL. That question is contested for Python
imports and not one to resolve casually in a portfolio project. So `PiperBackend`
**invokes the `piper` CLI as a subprocess** instead of importing it. A process
boundary is the widely accepted separation for GPL tools — the same relationship a
project has with `ffmpeg` — and it costs almost nothing here, because the CLI
(`python3 -m piper -m VOICE -f OUT.wav -- TEXT`) does exactly what is needed.

Piper is also **not installed by default**: it is an optional extra, so a plain
install of this package pulls no GPL code at all.

**The voices are licensed separately from the code.** `rhasspy/piper-voices`
checkpoints carry their own terms, mostly CC BY 4.0 (attribution) with some CC0 and
some more restrictive, varying per voice and per dataset. As with F5-TTS and
OmniVoice, **the restriction lives in the weights**, so the specific voice used in any
recorded run is named in that run's provenance banner rather than left implicit.
