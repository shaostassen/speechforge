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
and four studies have results. 125 tests pass offline with no weights and no network.

Two backends are wired up (Chatterbox, CosyVoice 2) and, with each at a *measured*
reference cap, both land essentially on the ground-truth row — so what is reported
between them is a trade-off, not a ranking. The largest studies run at n=60-72 across 12
speakers on two corpora, and their main product so far has been **seven measurement
errors of mine that the harness caught**, two of which had already been written up as
findings before being refuted. Everything measured is in
[`docs/VALIDATION.md`](docs/VALIDATION.md) with raw logs; model licences and exclusions
are in [`docs/LICENSES.md`](docs/LICENSES.md).

## What has been measured

**Cloning works, scored against the right baseline.** On EdAcc — spontaneous, accented,
real recording conditions — cloning from a ~4 s conversational reference:

| | WER (Whisper) | WER (HuBERT-CTC) | SIM-o |
|---|---|---|---|
| ground truth | 0.130 | 0.259 | 0.894 |
| chatterbox | 0.346 | 0.383 | 0.844 |

⚠️ Measured on **raw `ALL CAPS`** corpus text, before that was found to cost Chatterbox
~0.090 WER on LibriSpeech (below). The synthesis row here therefore understates the
model; the ground-truth row is unaffected, since real audio does not read text. Kept as
recorded rather than deleted — the comparison *between* presentations is what produced
the finding.

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

**The biggest finding is a mistake of mine: the WER gap was the shape of my input.**
LibriSpeech and EdAcc ship transcripts as unpunctuated `ALL CAPS`. That is a convention
for *scoring*; I passed it straight to a generative model as though it were input. On
LibriSpeech test-clean, n=60, identical utterances in every row:

| | text as the model sees it | WER (Whisper) | WER (HuBERT-CTC) | SIM-o |
|---|---|---|---|---|
| ground truth | — | **0.021** | 0.016 | 0.964 |
| chatterbox | raw `ALL CAPS` | 0.112 | 0.110 | 0.951 |
| **chatterbox** | **sentence case** | **0.022** | **0.022** | 0.949 |

**A 5× reduction from changing nothing but case and a full stop.** At 0.022 Chatterbox
sits on the ground-truth row and inside the 1.5–3% published for zero-shot systems.
SIM-o barely moves (0.951 → 0.949), so it is specifically intelligibility.

This **refutes what this README previously claimed** — that 11% was "a gap large enough
not to be protocol noise". It was protocol noise, and mine. An earlier decomposition had
correctly excluded temperature, text length and reference cap, leaving ~0.095
unexplained; text presentation accounts for ~0.090 of it.

The ground-truth 2.1% reproducing the published figure is what made the error findable:
the harness was measuring correctly the whole time, so the anomaly had to be the input.

**"Give every model the same text" is the wrong kind of fairness.** It looks
even-handed and isn't, because robustness to it differs — the same raw input costs
Chatterbox 0.090 WER and CosyVoice 2 0.213. Identical treatment, unequal handicap.
`--text-case` makes the choice explicit and recorded, defaulting to `raw` so earlier
numbers stay comparable.

**The second backend's "10× worse" was also my configuration.** CosyVoice 2 first
measured 0.212 against Chatterbox's 0.022 — and I declined to publish that ranking
because one asymmetry was mine: a 10 s reference cap taken from CosyVoice's own docs
rather than measured. Splitting the run by whether that cap had truncated the reference:

| CosyVoice 2, 10 s cap | WER mean | WER median | n |
|---|---|---|---|
| reference **hit** the cap | **0.507** | 0.375 | 25 |
| reference **under** the cap | **0.033** | **0.000** | 35 |

`corr(seconds discarded, WER) = +0.870`. Cutting at a hard 10.0 s lands mid-phrase and
the model over-generates — `corr(duration ratio, WER) = +0.861`, outputs up to 2.42×
the expected length. Re-run without truncation, CosyVoice 2 measures **0.036**, and
utterances above WER 0.5 go from 8 to **zero**. With each backend at a cap that was
measured rather than borrowed:

| system | reference cap | WER | WER-2 | SIM-o |
|---|---|---|---|---|
| ground truth | — | 0.021 | 0.016 | **0.964** |
| chatterbox | 20 s (measured optimum) | **0.022** | **0.022** | 0.949 |
| cosyvoice2 | no truncation | 0.036 | 0.034 | **0.958** |

Both sit essentially on the ground-truth row. What is left is a trade-off, not a
ranking: Chatterbox lower WER, CosyVoice 2 higher similarity and nearer the SIM
ceiling. One corpus, one language, n=60.

That cap is the **fourth** constant taken from documentation rather than measured that
turned out not to transfer — after the published SIM ceiling, the vendor RTF claim, and
the original 10 s cap I had already caught once on Chatterbox and then reintroduced
here. Four for four.

CosyVoice 2's first run also logged `no frontend is avaliable`, my trimmed install
having omitted `ttsfrd` and `wetext`. Re-running with `wetext` active gave
**byte-identical** results: normalization is a no-op on plain sentence-case prose, so
that defect was real and immaterial. Two processes producing identical rows also makes
this the project's strongest reproducibility check.

⚠️ **The two ASRs are not independent on this corpus.** `hubert-large-ls960-ft` is
trained on LibriSpeech, so their agreement here (0.021 vs 0.016) partly reflects domain
match, not corroboration. On EdAcc, where HuBERT is out of domain, they diverged sharply
(0.130 vs 0.259). Read the pair with that in mind.

**A learner's own voice, pronouncing correctly — and the conversion stage earns its
place.** The motivating application: synthesize with a native reference, then convert
the timbre onto the learner. Scored against plain cloning, which is the baseline that
matters, at n=72 across 12 accented EdAcc speakers:

| signal | SIM to learner | WER |
|---|---|---|
| learner real *(identity ceiling)* | 0.924 | 0.225 |
| native tts | 0.621 | 0.076 |
| learner tts *(plain cloning)* | 0.856 | 0.087 |
| **golden** | **0.818** | **0.073** |

Conversion costs **−0.003 WER** against the TTS it came from — it does not damage the
speech — and beats plain cloning by 0.014 WER for 0.039 SIM. That WER gap is accent
preservation showing up as a measurement: cloning an accented reference reproduces the
accent, and the ASR pays for it.

**The output carries both voices, and measuring that took three attempts.** Scored
against the source speaker's *real* audio — so that both sides are synthetic-vs-real
with different text and no shared prosody — golden sits **65% of the way toward the
learner and 56% toward the source** on a scale this run measures at its own ends
(0.621 different speaker, 0.924 same speaker). Retained source identity is **+0.114**
above a clone that never touched the source. Partial conversion, not a clean transfer.

Comparing against the source *TTS* instead is the trap: golden is a conversion of that
waveform, so it shares its vocoder and prosody, a bias measured at **+0.070**. That
substitution inflated the per-utterance "closer to the source" rate from **44% to 54%**
and produced a spurious "majority blend" verdict, now retracted. The function that
computes it refuses the unmatched input outright.

**ASR confidence does not detect synthetic speech.** Holding the transcribed word
*correct* — so accuracy can't explain the result — real and synthetic confidence
distributions are the same shape at every percentile, and the mean difference changed
sign between two identical runs. Whisper's per-word confidence measures acoustic-lexical
fit, not provenance. It is not a free deepfake detector, and it does not meaningfully
bias WER-based evaluation either.

## Measurement errors this harness caught in its own results

Kept here because they are the reason to trust anything above.

Four came from trusting a number that was never measured here — the whole category has a perfect failure record:

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
- **A documented reference-length "sweet spot" was catastrophic.** CosyVoice's own docs
  say ~10 s; a hard 10 s cap cut references mid-phrase and drove the model into
  autoregressive over-generation, costing **5.9× WER** (0.212 → 0.036). This is the same
  mistake I had already caught and fixed on Chatterbox, then reintroduced for a new
  backend because the vendor stated it.

Three more came from studies that printed a plausible table while concealing their own
gap — all in the golden-speaker work, all the same shape: a guardrail existed in the
library and the study hand-rolled its own reporting around it.

- **An identity ceiling computed and then discarded**, printing a placeholder string
  where the number belonged, leaving a similarity column with no scale.
- **An axis promised in the docstring and never computed** — source-to-learner
  similarity was passed into a parameter named `sim_to_source`, so "identity shed" went
  unmeasured while the docs claimed otherwise.
- **A verdict read off the means**, which cannot see a majority of individual
  utterances going the other way.

The lesson that generalises: reach for the library's guardrail instead of re-printing
the table by hand, and dump per-utterance rows *before* rendering — the last bug's bias
check then costs a re-render instead of a 90-minute re-run.

And one that is really the same lesson three times: **small samples are for choosing
what to measure, never for a number that gets written down.** A "consistent ~0.05 SIM
gap" (n=12) inverted at n=60. A "54% blend rate" fell to 22% under a measured bias
correction. A probe putting CosyVoice 2 at 0.048 in sentence case (n=9) measured 0.212 at
n=60 — right about the direction, 4.4× wrong about the size. All three were directionally
useful and quantitatively worthless.

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

- **Preferred reference length is model-specific, and the popular figure is wrong for
  Chatterbox.** CosyVoice and dots.tts document ~10 s as the sweet spot; applying that
  here cost **24% relative WER** on references it truncated (0.126 → 0.097 at a 20 s
  cap), with the untouched half unchanged to three decimals as a control. The cap is now
  a backend property, measured rather than borrowed.
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
