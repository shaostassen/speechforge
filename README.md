# SpeechForge

Zero-shot voice cloning — take one short recording of a voice, synthesize new
sentences in it — with an evaluation harness that can tell you whether it actually
worked.

The second half is the point. Cloning demos are easy to make and hard to trust: the
sample you'd post is the one that came out well. So every number here arrives with the
hardware, model and date that produced it, raw logs committed, and a ground-truth row
in every table.

## Status

Early. What exists: the reference-audio pipeline, the backend-agnostic synthesis layer,
the CPU throughput benchmark, and a licence audit. What doesn't yet: the evaluation
triad and the studies. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for measured
results and [`docs/LICENSES.md`](docs/LICENSES.md) for why these models and not others.

## Why the licences get their own document

Voice-cloning models are unusually treacherous here, because **the restriction usually
lives in the weights, not the code**. F5-TTS and OmniVoice both ship permissive source
with `cc-by-nc-4.0` checkpoints, because they were trained on an in-the-wild dataset.
XTTS-v2 is non-commercial for the model *and its outputs*. Third-party write-ups
routinely repeat the repo badge and are wrong about the thing you'd be shipping.

This repo uses Chatterbox (MIT code, **MIT weights**) and CosyVoice 2 (Apache-2.0 on
both). What was excluded, and why, is recorded rather than omitted.

## The first thing measured was the assumption most likely to be wrong

The plan rested on a vendor claim: Chatterbox-Nano at "3× faster than realtime on 8 CPU
cores". Measured on a Ryzen 9 9950X, the installable checkpoint runs at **RTF ~1.9–2.6
— about 2× slower than realtime**, with threading saturating at 16 threads (32 is
*worse*).

And Chatterbox-Nano turns out not to be loadable from the released package at all: its
weights are published under MIT, but the loader hardcodes a different repo and Nano
ships an incompatible file layout. The claim is untested rather than refuted.

This doesn't block anything — batch evaluation at RTF 2 means a 100-sentence set costs
about 17 minutes — but it does mean no interactive demo on CPU, and it makes ZipVoice's
undeclared weights licence worth chasing. Full detail in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## Reference audio is where quality is decided

Three findings from the model docs that run against intuition, all enforced in
`speechforge/reference.py`:

- **Longer references are worse, not better.** Around 10 s is the documented sweet
  spot; past it, inference slows *and* cloning degrades. Clips are truncated, keeping
  the beginning.
- **Trailing silence is a specific hazard.** Continuation-style models treat the
  reference as a literal prefix, so trailing room tone becomes the onset of the
  generated speech. It gets trimmed and warned about by name.
- **Noise is cloned along with the voice** — it lands in the speaker embedding, so you
  reproduce the room. A clean 10 s clip beats two noisy minutes.

Every clip comes back with a report of what was done and what looked wrong, so a batch
run records *why* a clone was poor instead of leaving you to guess afterwards.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core: reference pipeline + tests, no torch
pip install -e ".[eval]"                # + speechlens (ASR, WER, calibration)

# A synthesis backend. On a machine with no GPU, force CPU wheels or pip pulls
# ~2.5 GB of unusable CUDA libraries:
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchaudio==2.6.0
pip install -e ".[chatterbox]"
```

The core install stays torch-free on purpose: you should be able to score a directory of
wav files without a TTS stack. Backends pin conflicting torch and transformers versions,
so they're optional extras and imported lazily — `speechforge.synth` does not import
torch, and a test pins that.

```bash
pytest -q                                          # offline, no weights
python scripts/bench_synth.py --backend chatterbox --ref ref.wav
```

## Relationship to SpeechLens

The [ASR project](https://github.com/shaostassen/voice_detection_using_machine_learning)
supplies the evaluation half, and not by coincidence: **WER of an ASR system run on
synthesized speech is the field-standard intelligibility metric**, and F5-TTS's own
evaluation script uses the same `faster-whisper-large-v3` stack. That project also
brings word-level alignment, AUROC/ECE/NCE, and calibration — all validated, all reused
rather than rewritten.

It stays a separate repo because it holds a hard no-torch dependency ceiling, which is
what keeps its install lean and its 132 tests running weight-free in two seconds. That
ceiling is load-bearing and this project would break it.

## Ethics

Reference voices are the author's own recordings, plus speakers from
permissively-licensed corpora (LibriSpeech, VCTK) where benchmark comparability requires
it — that is the standard research protocol and those licences permit it. Nothing
scraped, nothing non-consented. Note also that Chatterbox watermarks every output it
produces, unconditionally.
