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
| **StyleTTS 2** (LibriTTS checkpoint) | MIT | **no licence — usage terms only**, see below | Diffusion backend: style diffusion, non-autoregressive |

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

## StyleTTS 2 — usable here, but the weakest weights licence in this table

Audited 2026-11-20 from the README, the `LICENSE` file, the Hugging Face card
(`yl4579/StyleTTS2-LibriTTS`) and issue #37, where the terms were argued out.

- **Code: MIT** (`LICENSE`, and the author in #37: "The codebase itself is MIT").
- **Weights: no licence of any kind is declared.** The HF card has no README and no
  `license` field; #37 is titled "Possibly misleading license info", and a reply there
  points out that the copyright default is all rights reserved and asks for an explicit
  grant. None was given. What exists is a **usage condition** in the README: "Before
  using these pre-trained models, you agree to inform the listeners that the speech
  samples are synthesized by the pre-trained models, unless you have the permission
  to use the voice you synthesize."
- **When it binds is ambiguous, and the two readings disagree for this project.** The
  README narrows it to voices "NOT in the training set, i.e., your reference speakers
  are not from any open access dataset". The author's reason in #37 is the first half:
  LibriTTS is CC BY 4.0, "so anyone has the right to use the voices inside the training
  set", and the rule exists because "the model can also synthesize in voices not seen
  during training". LibriSpeech test-clean speakers are **from an open-access dataset**
  (exempt on the second reading) and **not in the training set** (bound on the first).
- **Commercial use is not addressed** — asked in #37 in 2025, unanswered.

How this repo complies: it takes the conservative reading and treats the rule as
binding. It publishes numbers, not audio; any synthesized sample published from it
must be labelled as synthesized. The weights are downloaded by the user from Hugging
Face and never redistributed here. That makes it usable for this project's
evaluation work. It does **not** make the checkpoint shippable in a product — for
that it is behind Chatterbox (MIT weights) and CosyVoice 2 (Apache-2.0), and would
need the author's explicit grant first.

Two dependencies, handled like Piper's:

- **`phonemizer` is GPL-3.0-or-later**, verified from its PyPI classifiers. Upstream's
  own README says inference "depends on a GPL-licensed package, so it is not included
  directly in this repository". `StyleTTS2Backend` therefore **runs phonemizer's own
  `phonemize` CLI as a subprocess** and never imports it, and it is only installed by
  the `styletts2` extra. The CLI output was checked byte-for-byte against upstream's
  library call on all 2,620 LibriSpeech test-clean transcripts in both presentations:
  zero mismatches. `espeak-ng` behind it is GPL-3.0 too, and is a system binary.
- **The MIT `styletts2` package on PyPI was not used.** It avoids the GPL dependency by
  phonemizing with gruut instead, and the authors warn of "lower quality due to
  mismatch between phonemizer and gruut": the model was trained on espeak phonemes.
  It also pins 2023-era `huggingface-hub`, `accelerate` and `langchain`.

The text aligner, pitch extractor and PL-BERT that inference needs ship inside the MIT
repo under `Utils/`. The checkpoint used in every recorded run is
`epochs_2nd_00020.pth`, SHA-256 `1164ffe1…57dca0a`, which matches the LFS object id
Hugging Face publishes for it.

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

**Re-checked 2026-11-20 — half answered.** The HF card gained a licence section
(last modified 2026-10-10): "Our code is released under the Apache 2.0 License. The
pre-trained model is licensed under the CC-BY-NC due to constraints from its training
data (Emilia)." So (a) is settled for the Emilia checkpoints: **CC-BY-NC, excluded.**
(b) is not. The repo now also hosts `zipvoice_libritts` and `zipvoice_distill_libritts`,
trained on LibriTTS alone, and the sentence speaks of "the pre-trained model" in the
singular while giving Emilia as the reason. Read strictly, it covers every checkpoint;
read by its stated reason, it covers none of the LibriTTS ones. The card's metadata
still has no `license` field. Under the standing rule below, ambiguity is not a grant,
so ZipVoice stays out — which is why the diffusion backend here is StyleTTS 2 rather
than the better-fitting ZipVoice-LibriTTS.

**Action (still open):** open an issue on the repo asking (a) the weights licence and (b) whether the
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
