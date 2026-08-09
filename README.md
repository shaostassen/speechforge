# SpeechForge

Zero-shot voice cloning — take one short recording of a voice, synthesize new
sentences in it — with an evaluation harness that can tell you whether it actually
worked.

The second half is the point. Cloning demos are easy to make and hard to trust: the
sample you'd post is the one that came out well. So every number here arrives with the
hardware, model and date that produced it, raw logs committed, and a ground-truth row
in every table.

## Status

Working end to end, on a small scale. Cloning runs, the evaluation harness scores it,
and two studies have results. 81 tests pass offline with no weights and no network.

What is deliberately *not* claimed: that Chatterbox is good. The samples so far are 12
utterances across 4 speakers — enough to demonstrate the harness and to catch three of
my own measurement errors, not enough to rank a model. Everything measured is in
[`docs/VALIDATION.md`](docs/VALIDATION.md) with raw logs; model licences and exclusions
are in [`docs/LICENSES.md`](docs/LICENSES.md).

## What has been measured

**Cloning works, scored against the right baseline.** On EdAcc — spontaneous, accented,
real recording conditions — cloning from a ~4 s conversational reference:

| | WER (Whisper) | WER (HuBERT-CTC) | SIM-o |
|---|---|---|---|
| ground truth | 0.130 | 0.259 | 0.894 |
| chatterbox | 0.346 | 0.383 | 0.844 |

**The second ASR changes the headline.** Synthesis is 2.7× worse than real speech under
Whisper and 1.5× worse under HuBERT-CTC. Both are true; which you report is decided by
your choice of evaluator. HuBERT has no language model to repair the disfluencies of
real spontaneous speech, so it scores *real* audio far harder; synthetic speech is
cleaner, leaving less to repair, and the two converge. Reporting one ASR alone hides
that entirely — which is why the harness reports two.

**Speaker similarity depends heavily on the reference, and my earlier reading was a
small-sample artefact.** At n=60 the clone-to-real gap is **0.013 on LibriSpeech** and
**0.060 on EdAcc** — 4.6× larger from disfluent conversational references. I previously
recorded a "consistent ~0.05 gap across corpora" and concluded harder references
"degrade the clone far less than expected", both from runs of n=3 and n=12. At n=60 the
conclusion inverts. Absolute SIM values are encoder-specific and inflated by matched
recording conditions, so only the ground-truth row in the same run bounds them.

**On LibriSpeech, WER at scale (n=60):**

| | WER (Whisper) | WER (HuBERT-CTC) | SIM-o |
|---|---|---|---|
| ground truth | **0.021** | 0.016 | 0.964 |
| chatterbox | 0.112 | 0.110 | 0.951 |

The ground-truth 2.1% reproduces the published LibriSpeech test-clean figure, which is
the check that says the harness is measuring the same thing the literature does.
Against that, Chatterbox at 11% is well behind published zero-shot systems, which
report 1.5–3% on this corpus — a gap large enough not to be protocol noise, though the
protocol does differ (cloning from one held-out reference rather than a fixed 3 s prompt).

⚠️ **The two ASRs are not independent on this corpus.** `hubert-large-ls960-ft` is
trained on LibriSpeech, so their agreement here (0.021 vs 0.016) partly reflects domain
match, not corroboration. On EdAcc, where HuBERT is out of domain, they diverged sharply
(0.130 vs 0.259). Read the pair with that in mind.

**ASR confidence does not detect synthetic speech.** Holding the transcribed word
*correct* — so accuracy can't explain the result — real and synthetic confidence
distributions are the same shape at every percentile, and the mean difference changed
sign between two identical runs. Whisper's per-word confidence measures acoustic-lexical
fit, not provenance. It is not a free deepfake detector, and it does not meaningfully
bias WER-based evaluation either.

## Three measurement errors this harness caught in its own results

Kept here because they are the reason to trust anything above.

- **A vendor RTF claim was wrong by ~6×.** Chatterbox-Nano is advertised at 3× faster
  than realtime on 8 CPU cores; the installable checkpoint measured **2× slower** than
  realtime, and Nano turns out not to be loadable from the released package at all.
- **A published similarity ceiling did not transfer.** Papers report ~0.69–0.76; this
  encoder puts legitimate same-speaker pairs at 0.89–0.98. A warning built on the
  published figure would have fired on every valid run.
- **Nothing was reproducible.** Chatterbox samples at `temperature=0.8`, so two
  identical study runs gave AUROC 0.591 and 0.553 with the mean difference flipping
  sign. Synthesis is now seeded by default; pass `--seed -1` when the spread is the
  quantity you actually want.

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
