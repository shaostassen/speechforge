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

## In use

| Model | Code | Weights | Role |
|---|---|---|---|
| **Chatterbox** (Resemble AI) | MIT | **MIT** | Local CPU cloning + voice conversion |
| **CosyVoice 2** (0.5B) | Apache-2.0 | **Apache-2.0** | Quality ceiling, Colab GPU |
| **Piper** (archived MIT build) | MIT | MIT | Non-cloning latency/quality baseline |
| **kNN-VC** | MIT | MIT | Training-free, explainable VC baseline |

Chatterbox is the only family found with permissive *weights* and a vendor claim of
better-than-realtime CPU synthesis. CosyVoice 2 has the best licence-to-quality ratio
in the field — genuinely Apache-2.0 on both sides.

Two notes on Chatterbox that are easy to miss:

- **Every output is watermarked** with Resemble's PerTh neural watermarker. Harmless,
  arguably a responsible-AI feature, but worth knowing in a project that also does
  signal analysis — the audio is not pristine.
- The current `piper` development repo (`OHF-Voice/piper1-gpl`) is **GPL-3.0**. The
  MIT-licensed `rhasspy/piper` was archived 2025-10-06. Use the archived MIT build, or
  accept GPL.

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
