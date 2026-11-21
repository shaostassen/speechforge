# Validation log

Same discipline as the SpeechLens project: a number exists here only alongside the
run that produced it — hardware, model, date, and the raw log in
[`validation_runs/`](validation_runs/). No estimated or remembered values.

---

# Run — 2026-08-07 — Chatterbox synthesis throughput on CPU

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X 16-Core (32 threads), Linux x86_64, **no GPU** |
| package | `chatterbox-tts` 0.1.7 (PyPI), `torch` 2.6.0+cpu |
| model | `ResembleAI/chatterbox` default checkpoint — **t3 backbone 532M params** |
| reference | 10 s LibriSpeech clip |
| output | 24 kHz |
| raw log | [`validation_runs/bench_chatterbox_9950x.txt`](validation_runs/bench_chatterbox_9950x.txt) |

This was the first thing measured, deliberately: the plan's local-iteration story rested
on a vendor claim of "3× faster than realtime on 8 CPU cores" for Chatterbox-Nano, and an
unverified assumption at the base of a plan is the expensive kind.

Model load (63 s) excluded; one warm-up pass discarded; three sentences of differing
length. RTF = wall time / duration of audio produced — the same definition the ASR
project used, so the numbers are comparable across both.

| threads | RTF range | × realtime |
|---|---|---|
| 8 | 2.264 – 2.613 | 0.38 – 0.44 |
| **16** | **1.913 – 2.049** | **0.49 – 0.52** |
| 32 | 2.224 – 2.910 | 0.34 – 0.45 |

## Two findings

**1. Synthesis runs about 2× slower than realtime, at best.** Not the claimed 3× faster.
Threading saturates at **16 threads**, and 32 is measurably *worse* — the same
memory-bandwidth ceiling the ASR bench hit, and a reminder that "32 threads available"
is not "32 threads useful".

**2. Chatterbox-Nano cannot currently be loaded from the released package,** so the
vendor's claim remains untested rather than refuted. Evidence:

- `chatterbox-tts` 0.1.7 exposes `from_pretrained(device)` — no model-selection
  argument.
- GitHub `master` is the same, with a hardcoded `REPO_ID = "ResembleAI/chatterbox"`.
- Pointing `REPO_ID` at `ResembleAI/chatterbox-nano` will not work either: Nano ships a
  **different file layout** — `t3_nano_v1.safetensors` plus a `.yaml` config and a
  GPT-2-style tokenizer (`vocab.json` / `merges.txt`), where the loader expects
  `t3_cfg.safetensors` and `tokenizer.json`.

So the 110M Nano weights are published under MIT, but the code to load them is not in
the open package. The 532M default is what is actually runnable today.

## What this changes, and what it doesn't

**It does not block Phase 1.** The plan needs *batch* synthesis for evaluation, not
interactive latency. At RTF ~2, generating a 100-sentence evaluation set of ~5 s
utterances costs roughly 17 minutes on this machine — entirely workable, and the
9950X stays the right place to iterate.

**It does block an interactive demo**, and it makes the ZipVoice licence question
material rather than optional. ZipVoice-Distill reports RTF 1.22 on a *single* CPU
thread — plausibly ~10× faster than what is measured here — and it is the obvious
alternative if real-time local synthesis becomes a requirement. See
[`LICENSES.md`](LICENSES.md) for why it is not yet adopted.

---

# Run — 2026-08-08 — first scored clone, and what the SIM number is worth

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, no GPU |
| synthesis | `chatterbox` default (532M), reference 5.9 s LibriSpeech |
| ASR | `speechlens` / faster-whisper **small**, int8 (iteration setting, not `large-v3`) |
| speaker encoder | `microsoft/wavlm-base-plus-sv` @ 16 kHz |
| protocol | 1 speaker, 3 held-out utterances; synthesize each held-out text, score against that text and that recording |

```
                system     n     WER   SIM-o
          ground truth     3   0.044   0.979   <- read every row against this
            chatterbox     3   0.103   0.927
```

WER 0.103 against a human 0.044 — synthesis is about 2.3× worse, and safely
**above** the human line, so the number is readable as intelligibility. (With
`small` rather than `large-v3`; not comparable to published figures yet.)

The reference pipeline earned its place immediately: it reported
`trimmed 0.43s of trailing near-silence` on the very first real clip. For a
continuation-style model that region would have become the onset of the
generated speech.

## The similarity number needed a floor before it meant anything

SIM-o 0.927 looks superb against the 0.69–0.76 ceiling that zero-shot TTS papers
report — and would have been badly over-read. Ground truth on the same audio and
encoder is **0.979**, so the whole scale is shifted, not the clone.

The deciding measurement is the different-speaker floor, which the LibriSpeech
dummy split cannot provide — it contains exactly one speaker with more than one
utterance. Measured instead on 6 EdAcc speakers:

| | n | mean | min | max |
|---|---|---|---|---|
| same speaker | 16 | **0.890** | 0.764 | 0.987 |
| different speaker | 120 | **0.676** | 0.222 | 0.921 |

Usable range 0.214 — **the classes do separate**, so the encoder works. Two
caveats that travel with every number it produces:

- **34% of different-speaker pairs score above the lowest same-speaker pair.**
  Reliable in aggregate, noisy per utterance. Do not read a single SIM score as
  a verdict on a single clone.
- **The published 0.69–0.76 ceiling does not apply to this encoder.** Papers use
  a WavLM-large checkpoint fine-tuned for verification, distributed as a bare
  state dict with no licence statement — deliberately not a dependency here. The
  cost of that choice is exactly this: absolute SIM values are not comparable to
  published ones, only to our own ground-truth row.

## A gate that would have cried wolf

The first version of `EvalTable` warned "SIM above 0.73 — suspect the
measurement", taken from the published ceiling. With this encoder, legitimate
same-speaker scores run 0.89–0.98, so that warning would have fired on **every
valid run** and become noise to scroll past.

That is the same failure as the ASR project's `low_confidence = 0.55`, which sat
below the model's floor and could never fire — the mirror image, and equally
useless. A warning that always fires carries no information.

Fixed: the check now compares against the **measured ground-truth row**, and
flags only the case that is genuinely wrong — a clone scoring *above* the
speaker's own other recordings, which almost always means SIM-r against the
reference clip. The published range survives as a footnote for reading papers,
not as a threshold.

---

# Run — 2026-08-08 — the real protocol: 4 speakers, two ASRs

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, no GPU |
| corpus | **EdAcc**, 4 speakers from 2 conversations (`C08-A/B`, `C10-A/B`) |
| sample | 12 held-out utterances (3 per speaker), references 3.9–5.3 s |
| synthesis | `chatterbox` default (532M) |
| ASR 1 | faster-whisper **`large-v3`** int8, via speechlens |
| ASR 2 | **`facebook/hubert-large-ls960-ft`** (CTC) — different architecture on purpose |
| encoder | `microsoft/wavlm-base-plus-sv` |
| raw log | [`validation_runs/eval_edacc_9950x.txt`](validation_runs/eval_edacc_9950x.txt) |

```
                system     n     WER   WER-2   SIM-o
          ground truth    12   0.130   0.259   0.894   <- read every row against this
            chatterbox    12   0.346   0.383   0.844
```

## The second ASR changes the conclusion, which is why it exists

The two evaluators **disagree sharply on real speech and agree on synthetic**:

| | Whisper `large-v3` | HuBERT-CTC | ratio |
|---|---|---|---|
| ground truth | 0.130 | **0.259** | HuBERT 2.0× harsher |
| chatterbox | 0.346 | 0.383 | HuBERT 1.1× harsher |
| **synthesis ÷ real** | **2.7×** | **1.5×** | — |

Read the bottom row. **"Synthesis is 2.7× worse than real speech" and "synthesis is
1.5× worse" are both true, and which one you report is decided entirely by your
choice of ASR.** Had this run used Whisper alone — the field-standard choice — it
would have produced the 2.7× figure with no indication that the number was that
soft.

The mechanism is legible. HuBERT-CTC is frame-synchronous with no autoregressive
language model, so it has nothing to paper over the disfluencies, accents and
crosstalk of spontaneous EdAcc speech: WER 0.259 on *real recordings*. Whisper's
internal LM repairs much of that, reaching 0.130. Synthetic speech is cleaner and
more canonical, so there is far less for an LM to repair — and the two ASRs
converge (0.346 vs 0.383, 11% apart).

Which is to say: **the apparent gap between synthetic and real speech is partly an
artefact of how much the evaluator's language model flatters real speech.** This
is the ASR-family-alignment confound, measured rather than cited.

## Speaker similarity is consistent across two very different corpora

| corpus | ground truth | chatterbox | gap |
|---|---|---|---|
| LibriSpeech (studio, read) | 0.979 | 0.927 | 0.052 |
| EdAcc (spontaneous, accented) | 0.894 | 0.844 | 0.050 |

The **absolute** values move a long way between corpora — recording conditions
inflate within-corpus similarity — but the **gap is the same to within 0.002**.
Expressed against the encoder's measured range (different-speaker floor 0.676,
same-speaker mean 0.890), the clone lands about **79% of the way from "different
speaker" to "same speaker"** on EdAcc, and ~83% on LibriSpeech. Consistent.

Two independent checks came out right, which is the main reason to believe the
similarity column at all:

- The EdAcc ground-truth row (**0.894**) reproduces the same-speaker mean measured
  separately on 6 EdAcc speakers (**0.890**) — different utterances, same answer.
- Cloning from spontaneous conversational references (3.9–5.3 s, disfluent,
  possible crosstalk) cost only ~0.05 similarity versus studio read speech. Harder
  references degrade the clone far less than expected.

## The absolute WER is high, for a predicted reason

Ground-truth WER of 0.130 under Whisper is poor for "real human speech", and it is
not a measurement error. EdAcc transcribes disfluencies verbatim; **4.6% of its
reference words are fillers** (`uh` 102, `um` 68 in a 100-utterance sample,
measured before this run) and Whisper performs implicit disfluency removal, so
every one becomes a deletion. Spontaneous accented conversation accounts for the
rest. Both rows pay the same penalty, so the comparison holds — but these absolute
numbers are **not** comparable to LibriSpeech or to published figures.

## Caveats that still bound this result

- **n = 12, 4 speakers, 2 conversations.** Enough to demonstrate the harness and
  see a consistent SIM gap; not enough for a claim about Chatterbox.
- Both ASRs, the encoder, and the corpus are English.
- Predicted MOS remains unwired — the column renders `—`, which is correct.
- This run predates the reference frame-alignment fix, so its syntheses each
  triggered Chatterbox's internal token-truncation path. Effect is sub-frame
  (<40 ms of the prompt) and not expected to matter, but a re-run would be cleaner.

---

# Run — 2026-08-09 — scored eval at scale, seeded, and calibrated against the literature

| field | value |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, no GPU |
| corpus | **real LibriSpeech test-clean** — 12 speakers, 60 held-out utterances |
| synthesis | `chatterbox` default (532M), **seed 1234** (reproducible) |
| ASR 1 / 2 | faster-whisper `large-v3` int8 / `hubert-large-ls960-ft` (CTC) |
| raw log | [`validation_runs/eval_librispeech_9950x.txt`](validation_runs/eval_librispeech_9950x.txt) |

```
                system     n     WER   WER-2   SIM-o
          ground truth    60   0.021   0.016   0.964   <- read every row against this
            chatterbox    60   0.112   0.110   0.951
```

## The harness is calibrated against published work

**Ground-truth WER of 0.021 reproduces the ~2.1% that LibriSpeech test-clean papers
report** (Seed-TTS test-en GT 2.14%; LibriSpeech-PC test-clean 1.87–2.23%). That is
the check that matters: it says this harness measures the same quantity the
literature does, so its other rows can be read against published numbers rather
than only against each other.

Against that baseline, **Chatterbox at 0.112 is well behind published zero-shot
systems**, which report 1.5–3% WER on this corpus. The protocol differs — cloning
from one held-out reference instead of a fixed ~3 s prompt, CPU int8 ASR — but a
4–7× gap is too large to attribute to that.

## Two corrections this run forced

**The "consistent ~0.05 similarity gap across corpora" was a small-sample
artefact.** Recorded earlier from runs of n=3 (LibriSpeech dummy, gap 0.052) and
n=12 (EdAcc, gap 0.050). At n=60 the LibriSpeech gap is **0.013**. Two tiny
samples agreeing is not consistency, and I wrote it up as though it were.

**The two ASRs are not independent on LibriSpeech.** `hubert-large-ls960-ft` is
trained on LibriSpeech 960 h, so evaluating LibriSpeech with it is in-domain: it
scores ground truth at 0.016, *better* than Whisper's 0.021. On EdAcc, out of
domain, it was 2× harsher than Whisper (0.259 vs 0.130). So their close agreement
here (0.112 vs 0.110) is partly domain match rather than corroboration — the
second-ASR check is strongest exactly where the corpora are unfamiliar to both.

## Cross-corpus comparison, both seeded at n=60

EdAcc re-run at the same scale and seed (12 speakers, 5 held-out each; scan
budget raised to 9000 rows, which reaches 6 conversations):

```
corpus=edacc (held-out)
                system     n     WER   WER-2   SIM-o
          ground truth    60   0.149   0.281   0.922
            chatterbox    60   0.369   0.427   0.862
```

| | LibriSpeech | EdAcc |
|---|---|---|
| ground-truth WER (Whisper / HuBERT) | 0.021 / 0.016 | 0.149 / 0.281 |
| chatterbox WER | 0.112 / 0.110 | 0.369 / 0.427 |
| **synthesis ÷ real, Whisper** | **5.3×** | **2.5×** |
| **synthesis ÷ real, HuBERT** | **6.9×** | **1.5×** |
| SIM-o, ground truth → clone | 0.964 → 0.951 | 0.922 → 0.862 |
| **SIM-o gap** | **0.013** | **0.060** |

**The two-ASR divergence is confirmed at scale, and its direction flips with
domain.** On LibriSpeech, HuBERT scores ground truth *better* than Whisper
(0.016 vs 0.021) because `hubert-large-ls960-ft` is trained on LibriSpeech — it
is in-domain, so their agreement is not corroboration. On EdAcc it is 1.9×
harsher (0.281 vs 0.149), having no language model to repair spontaneous
accented speech. The second evaluator earns its place only out of domain, which
is the opposite of where its agreement first looked reassuring.

**Cloning is meaningfully worse from conversational references.** The
clone-to-ground-truth similarity gap is **0.060 on EdAcc against 0.013 on
LibriSpeech** — 4.6× larger. I previously wrote that harder references "degrade
the clone far less than expected", from an n=12 gap of 0.050 read against an
n=3 LibriSpeech gap of 0.052. Both small samples were unreliable, and at n=60
the conclusion inverts: a disfluent 4-second conversational reference costs
substantially more speaker similarity than a clean studio one. Corrected in the
README.

**What did survive scaling:** the synthesis-to-real WER ratio on EdAcc, 2.7×/1.5×
at n=12 and 2.5×/1.5× at n=60.

## Still bounded

- Two corpora, one backend, English, n=60 each.
- Predicted MOS is now wired (`--mos`) but was not enabled for these runs, so
  the column renders `—`.
- **Decode settings were Chatterbox defaults**, including `temperature=0.8` —
  which the sweep below shows is the right choice, not an oversight.

## Where the WER gap comes from — decomposing the aggregate

A corpus WER of 0.112 cannot distinguish "a few utterances fail catastrophically"
from "everything is uniformly mediocre", and those have different causes. Same
seeded run, LibriSpeech n=60, per-utterance detail recorded
([`validation_runs/per_utterance_librispeech.json`](validation_runs/per_utterance_librispeech.json)).

| tercile | words | synthesis WER | real WER |
|---|---|---|---|
| short | 5–14 | **0.145** | 0.019 |
| mid | 14–26 | 0.133 | 0.024 |
| long | 26–48 | **0.094** | 0.021 |

- `corr(words, synthesis WER)` = **−0.149** — longer text is if anything *easier*
- `corr(words, real WER)` = +0.004 — real speech is flat across length, so this is
  not utterance difficulty
- `corr(real WER, synthesis WER)` = +0.319 — only mildly shared

**The long-text hypothesis is refuted.** I expected chunking or length limits to
concentrate the errors in long utterances; the long tercile is the *best* of the
three. Much of the short-utterance penalty is a denominator artefact — one error
in a 6-word reference is WER 0.167, the same error in a 48-word reference is
0.021 — which is also why the per-utterance mean here (0.124) sits above the
corpus-level figure (0.112): averaging per utterance over-weights short ones,
exactly the reason `score_intelligibility` concatenates instead.

**The errors are broad, not concentrated.** Three utterances of 60 exceed WER 0.5
and account for 24% of the total error mass — real catastrophic failures, the
autoregressive instability the literature warns about — but the remaining 76% is
spread across the corpus.

**So the 5–7× gap is the model, not the protocol.** Two candidate explanations
have now been excluded by measurement rather than argument: decode temperature
(the sweep below) and text length (here). A broad, length-independent elevation
in WER is what a genuinely weaker intelligibility profile looks like, and it is
consistent with Chatterbox being built for expressive on-device synthesis rather
than benchmark WER — it does not appear in the Seed-TTS comparison tables at all.

## The reference cap — my own mistake, measured

The last protocol variable, and the only one that was my decision rather than a
vendor default: `reference.prepare` truncated references to 10 s, a figure taken
from CosyVoice and dots.tts documentation and applied to Chatterbox without
checking. Swept at n=60, same seed, with per-utterance reference lengths recorded
so the affected half can be isolated:

| slice | cap 10 s | cap 20 s | cap 30 s | n |
|---|---|---|---|---|
| all utterances | 0.124 | 0.109 | 0.109 | 60 |
| **reference was truncated** | **0.126** | **0.097** | 0.096 | 30 |
| reference untouched (**control**) | 0.122 | 0.122 | 0.122 | 30 |

**I was wrong, and the control proves it is causal.** Utterances whose reference
the cap actually shortened improve by **24% relative** (0.126 → 0.097); utterances
it never touched are unchanged **to three decimal places**, which is what a
correct seeded control looks like. Corpus-level WER goes 0.112 → 0.095. Speaker
similarity is flat throughout (0.951 → 0.950), so this is an intelligibility
effect, not a voice-matching one. The curve plateaus by 20 s.

Consequences:

- **Preferred reference length is a model property, not a constant.** It now lives
  on the backend (`Backend.ref_seconds`), measured at 20 s for Chatterbox, with
  the borrowed 10 s surviving only as the fallback for unmeasured backends —
  the same pattern already used for `align_ms`.
- **The earlier attribution needs softening.** "The gap is the model, not the
  protocol" was too strong: roughly 0.017 of corpus WER was mine. What survives
  is weaker but still substantial — at the best cap, 0.095 is **3–6× the 1.5–3%**
  published for zero-shot systems, and neither temperature nor text length nor
  reference length accounts for that remainder.
- The general lesson is the one this file keeps recording: guidance from one
  model's documentation is not evidence about another. Three borrowed or assumed
  numbers have now been tested here — the published SIM ceiling, the vendor RTF
  claim, and this cap — and all three failed to transfer.

## Temperature sweep — a hypothesis of mine, refuted

LibriSpeech, 6 speakers, 18 held-out utterances, seed 1234, `large-v3`. Same
ground truth throughout (WER 0.031, SIM-o 0.962) since only synthesis changes.

| temperature | chatterbox WER | SIM-o |
|---|---|---|
| 0.1 | 0.276 | 0.888 |
| 0.4 | 0.162 | 0.913 |
| **0.8 (default)** | 0.124 | **0.935** |
| 1.0 | **0.106** | 0.930 |
| 1.2 | 0.119 | 0.934 |

The hypothesis was that `temperature=0.8` is a lot of sampling entropy for an
evaluation, and that it explained the 5–7× WER ratio against published systems.
**Wrong, and in the opposite direction.**

**Below 0.8, degradation is sharp and monotone** — WER 0.124 → 0.162 → 0.276 and
SIM-o 0.935 → 0.913 → 0.888 as temperature falls. A 2.2× WER effect across
seeded runs, which is far too large to be n=18 noise. The mechanism is familiar
from language models: autoregressive decoding degenerates when sampling gets too
greedy, falling into repetition and stalled output. TTS is not exempt.

**Above 0.8 it plateaus.** WER 0.124 / 0.106 / 0.119 at 0.8 / 1.0 / 1.2 is
non-monotone and spans 0.018 — at n=18 that does not support picking a winner,
so **the shallow minimum at 1.0 is not a recommendation.** SIM-o likewise flattens
around 0.93. The defensible statement is that the default sits at the knee of the
curve, with everything above it equivalent within this sample's resolution.

**The gap to published systems survives the whole sweep.** The best WER anywhere
on the curve is 0.106, still 3.5–7× the 1.5–3% published for zero-shot systems on
this corpus. So it is not a configuration artefact — it is the model, the
protocol, or both. That matters because "we misconfigured it" was the comfortable
explanation, and it is now excluded across the full range rather than at one
point.

---

# Study — 2026-08-08 — does ASR confidence react to synthetic speech?

The question: when the ASR transcribes a word **correctly**, is it as confident
about a cloned voice as a real one? Restricting to correct words is what makes it
sharp — otherwise any difference is a restatement of "synthesis has higher WER".

Setup: for each held-out utterance there are two recordings of the *same text*,
real and cloned, so text, speaker and ASR are fixed and only provenance varies.
`large-v3` int8 with word timestamps; per-word probability is the signal.
Script: [`scripts/confidence_vs_synthetic.py`](../scripts/confidence_vs_synthetic.py).

## The answer depends on the corpus, and that is the finding

| | EdAcc, 4 spk, 12 utts | **LibriSpeech, 12 spk, 60 utts** |
|---|---|---|
| correct words (real / synth) | 288 / 294 | **1251 / 1161** |
| mean confidence, real | 0.9489 | **0.9723** |
| mean confidence, synthetic | 0.9475 | **0.9546** |
| **mean delta** | **+0.0014** | **+0.0177** |
| AUROC (real above synth) | 0.597 | 0.607 |
| p5 real / synth | 0.667 / 0.691 | **0.827 / 0.715** |
| accept-rate gap | **+1.2%** | **−3.2%** |

**On clean read speech the effect is clear.** Real words are transcribed with
higher confidence than cloned ones (0.972 vs 0.955), synthetic speech carries the
heavier low tail (p5 0.715 vs 0.827), and a reliability gate fitted on real audio
**rejects synthetic words more often** — 91.6% accepted against 94.9%.

**On spontaneous accented speech the effect vanishes.** Not because synthesis
improved, but because *real* speech got worse: EdAcc's real confidence is 0.949
against LibriSpeech's 0.972, while synthetic sits at ~0.95 in both. Disfluencies,
accents and conversational recording conditions drag real audio down to meet the
synthetic level, and the accept-rate gap even changes sign.

So the honest statement is: **ASR confidence separates real from synthetic speech
only to the extent that the real speech is clean.** The absolute confidence of
synthetic speech is remarkably stable across corpora (0.9475 / 0.9546); it is the
real-speech baseline that moves.

Two consequences that matter for the rest of this repo:

- **Confidence is not a usable synthetic-speech detector.** AUROC ~0.60 on the
  favourable corpus, and no separation at all on the harder one. Whisper's
  per-word confidence measures acoustic-lexical fit, and canonical synthetic
  articulation happens to score close to clean human articulation.
- **The reliability gate is mildly conservative toward synthesis, not
  permissive.** That is the reassuring direction: a gate tuned on real speech
  will flag synthetic output slightly *more*, so it does not wave cloned audio
  through.

## What the earlier, smaller run got wrong

The first EdAcc run reported "no measurable effect" and I was one step from
generalizing it. Two changes overturned that: **5× the sample** (60 utterances,
12 speakers rather than 12 and 4) and **a different corpus**. The EdAcc reading
was not wrong *about EdAcc* — it was wrong as a general claim, and only the
scale-up revealed which.

It also exposed a methodological hole worth stating plainly: the earlier 585
words came from **12 utterances**, so treating them as 585 independent samples
would have badly overstated the evidence. Clustering is why a 0.59 AUROC on that
sample meant much less than it looked like.

## Reproducibility, and a bug it caught

Chatterbox samples at `temperature=0.8`, so identical settings produce different
audio. Measured before the fix — same corpus, same speakers, same flags:

| run | AUROC | mean delta |
|---|---|---|
| 1 (unseeded) | 0.591 | +0.0015 |
| 2 (unseeded) | 0.553 | **−0.0007** (sign flipped) |
| 3 (seeded 1234) | 0.597 | +0.0014 |

**The mean difference changed sign between two identical runs.** The effect I was
about to report was inside sampling noise. Synthesis is now seeded by default
(`--seed 1234`; `-1` to sample freely when the spread is the quantity of
interest), which also means the EdAcc numbers recorded above this section are
single unseeded samples and should be treated as such.

**Verified, not assumed.** Two independent runs of the 60-utterance study at
seed 1234 agree in every digit — 1251/1161 correct words, delta +0.0177,
AUROC 0.607, all five percentiles, both accept rates:

| | run A | run B |
|---|---|---|
| correct words (real / synth) | 1251 / 1161 | 1251 / 1161 |
| mean delta | +0.0177 | +0.0177 |
| AUROC | 0.607 | 0.607 |
| p1 / p5 / p10 / p25 / p50, real | 0.452 / 0.827 / 0.969 / 0.993 / 0.997 | identical |
| accepted real / synthetic | 94.9% / 91.6% | identical |

So the numbers in this file are now reproducible from the recorded command, which
is the property that lets a result be checked rather than believed.

## Follow-ups from the first run

- **The dummy corpus cannot support this protocol.** One speaker with multiple
  utterances means no cross-speaker floor and n=3. Real LibriSpeech test-clean,
  or EdAcc, is needed for any number worth publishing.
- Re-run with `large-v3` for comparability, and add the second non-Whisper ASR.
- Predicted MOS is still unwired; the column renders as `—`, which is correct
  until it is measured.

**Earlier follow-ups**
- Ask Resemble AI how Nano/Turbo are meant to be loaded; re-measure if an entrypoint
  appears. If Nano hits ~0.4 RTF as its parameter count suggests it might, it becomes
  the primary backend.
- Measure the M2 via MLX. Unified memory and Metal make it a genuinely different
  proposition from "CPU", and it is the natural place for interactive work.

---

# Run — 2026-08-11 — golden speaker: does synthesize-then-convert beat plain cloning?

The Phase 3 question, and deliberately not "does golden work". Plain cloning is the
free baseline; conversion has to beat it or it is not earning its place.

| | |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, CPU only |
| TTS / VC | `chatterbox` t3 532M @ 24 kHz / `chatterbox-vc` @ 24 kHz (both MIT) |
| ASR | `faster-whisper large-v3`, int8 |
| speaker encoder | `microsoft/wavlm-base-plus-sv` @ 16 kHz |
| learners | 12 EdAcc speakers, 6 held-out utterances each — **n = 72** |
| pronunciation source | LibriSpeech 1089, 10.4 s (**n = 1**) |
| seed | 1234 |
| raw logs | `eval_golden_n72_anchored_9950x.txt`, `per_utterance_golden_n72_anchored.json` |

```
python scripts/eval_golden.py --speakers 12 --utterances 6 --asr-model large-v3 \
    --threads 16 --seed 1234 --scan 9000 --per-utterance runs/golden_anchor.json
```

| signal | SIM to learner | WER |
|---|---|---|
| learner real *(identity ceiling)* | **0.924** | 0.225 |
| native tts *(pronunciation source)* | 0.626 | 0.208 |
| learner tts *(plain cloning)* | 0.855 | 0.250 |
| **golden** *(convert native tts → learner)* | **0.824** | **0.208** |

## Conversion buys intelligibility for identity, and the trade is favourable

**WER cost of conversion is exactly 0.000** — golden matches the native TTS it came
from to three decimals, so the converter does not damage the speech it is given.
Against plain cloning, golden is **0.042 better on WER** (0.208 vs 0.250) for
**0.031 worse on SIM** (0.824 vs 0.855).

That WER gap is the accent-preservation effect showing up as a measurement: cloning
from an accented reference reproduces the accent, and the ASR pays for it. Golden
reaches **96.4% of the identity plain cloning achieves** (0.824 / 0.855) while
pronouncing like the native source. For a pronunciation-teaching application, which
is the whole motivation, that is the right side of the trade.

Read the SIM column against the ceiling, never in absolute terms: real-vs-real
same-speaker similarity is **0.924** on this encoder and corpus, so cloning uses
92.6% of the available headroom and golden 89.2%.

## Identity shed is NOT cleanly measurable in this design

The third axis promised by `golden.py` does not survive scrutiny, and the honest
result is a bound rather than a value.

`SIM(golden, native tts) = 0.802` looks alarmingly high against any real-speaker
floor. But golden and native tts are two clips from the **same TTS and vocoder**, so
shared synthesis artefacts inflate the number regardless of retained identity. The
run therefore also measures a same-pipeline different-speaker anchor —
`SIM(learner tts, native tts)`, two clones of different targets from one pipeline:

| quantity | value |
|---|---|
| `SIM(golden, native tts)` | 0.802 |
| same-pipeline different-speaker anchor | 0.696 |
| excess over anchor | **+0.106** |

Measured same-pipeline inflation is **+0.070** — synthetic-vs-synthetic
different-speaker pairs score 0.696 where synthetic-vs-real ones score 0.626.

**The +0.106 excess is an upper bound on retained source identity, not an estimate.**
The anchor removes the shared-pipeline component but not a shared-prosody one: golden
is a *conversion of the source waveform*, so it inherits the source's prosody, timing
and phonetic realisation, which speaker embeddings partly capture. `learner tts` is an
independent generation and shares none of that, so the anchor is not matched on
prosody. Separating the two needs a control this run does not have — convert the same
source waveform to a **decoy** target and compare; prosody sharing is then identical
and only target identity differs.

## A "majority blend" finding that did not survive its own bias check

Worth recording because it was one step from being written down as a result.

Per utterance, **39/72 (54%)** of golden outputs are closer to the source voice than
to the learner — a majority-blend outcome, while the *means* (0.824 vs 0.802) favour
the learner comfortably. That per-utterance check is genuinely more sensitive than the
means, and it is now what the verdict uses: two heavily overlapping distributions
produce a mean gap that no individual utterance need exhibit.

But the comparison is not apples-to-apples. `SIM(golden, learner)` is
synthetic-vs-real while `SIM(golden, native tts)` is synthetic-vs-synthetic, and the
latter carries the measured +0.070 inflation. Subtracting it:

| | closer to source |
|---|---|
| raw | 39/72 (54%) |
| after removing the +0.070 same-pipeline inflation | **16/72 (22%)** |

A plausible correction of a *measured* bias moves the statistic from majority to
minority, so **the blend claim is unsupported**. The uniform subtraction is itself
crude, which is the point: a number that flips its conclusion under a rough
correction was never load-bearing.

## Caveats that ship with these numbers

- **WER is not comparable between the real and synthetic arms on this corpus.** EdAcc
  transcripts are verbatim spontaneous speech: **3.1% of the 2582 reference words in
  these texts are bare fillers** (`UM`, `UH`, …), before repetitions and false starts
  like "WHICH IS JUST THE WHICH IS JUST A UH". TTS reads fillers aloud; Whisper tends
  to drop them, so every synthetic arm pays a penalty the real arm does not — which is
  why native tts (0.208) scores *better* than real speech (0.225). The three synthetic
  arms share identical text and evaluator, and every claim above is a comparison
  among them.
- **One pronunciation source** (LibriSpeech 1089). n=1 on that side of the design.
- **Cross-gender conversion is unseparated.** EdAcc speaker gender was not read, so
  same- vs cross-gender conversion — known to differ in difficulty for VC — is pooled.
  Per-speaker rows are in the JSON if this needs revisiting.
- **Ground truth is impossible by construction.** A non-native speaker producing
  native pronunciation cannot be recorded, so there is no target signal. Everything
  here is a trade-off measurement, never an accuracy one.

## Reproducibility

Two independent seeded runs of this study agree to three decimals on every similarity
(0.924 / 0.626 / 0.855 / 0.824 / 0.802) and on WER for three of four arms. Only
`learner tts` WER differed, 0.249 vs 0.250 — consistent with float reduction order
across 16 threads rather than with sampling, since the seeds fix the sampling.

## Three reporting bugs this study found in itself

The runs above are the third, fourth and fifth attempts. Each earlier one produced a
plausible table that was wrong in a way the table itself concealed.

- **The identity ceiling was computed and discarded.** The table printed the literal
  string `(held-out ref)` where real-vs-real similarity belonged, leaving the SIM
  column with no scale — 0.84 reads as excellent against a 0.85 ceiling and mediocre
  against 0.97. `golden_table` now refuses to render without it, exactly as
  `EvalTable` refuses without a ground-truth row.
- **Identity shed was promised and never computed.** The module docstring listed it as
  one of three axes, but the study passed *source-to-learner* similarity into a
  parameter named `sim_to_source`. It is now a required argument.
- **The verdict was read off the means**, which cannot detect a majority blend.

All three are the same failure as the earlier `sim_o`/`cosine` episode: a guardrail
existed in the library and the study reached around it by hand-rolling its own
reporting. `--per-utterance` now dumps every row *before* the table renders, which is
why the 54%→22% bias check above cost a re-render instead of another 90-minute run.

## Operational note

`/tmp` on the 9950X is wiped on reboot, and the box rebooted mid-run — destroying one
in-flight study. Run logs now go to `~/speechforge/runs/`. The earlier results survived
only because they had already been copied into this repo.

## Follow-up

**The decoy-target control**, which converts the same source waveform to a speaker who
is neither the source nor the learner. That makes prosody sharing identical across the
comparison and leaves target identity as the only difference, turning the +0.106 upper
bound into an actual estimate of retained source identity. One extra VC call per
utterance.

---

# Run — 2026-08-11 — the WER gap was text presentation, and it is now closed

The single largest correction in this log. Everything below supersedes the
"unexplained ~0.095" left over from the gap decomposition above.

| | |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, CPU only |
| corpus | LibriSpeech test-clean, 12 speakers × 5 held-out — **n = 60**, identical utterance set across all rows |
| ASR | `faster-whisper large-v3` int8, plus `hubert-large-ls960-ft` CTC |
| encoder | `microsoft/wavlm-base-plus-sv` |
| seed | 1234 |
| raw logs | `eval_chatter_libri_cased_9950x.txt`, `eval_cosy_libri_9950x.txt`, `eval_cosy_libri_cased_9950x.txt` |

## The result

| system | text as the model sees it | WER | WER-2 | SIM-o |
|---|---|---|---|---|
| ground truth | — | **0.021** | 0.016 | 0.964 |
| chatterbox | raw `ALL CAPS` (as the corpus ships) | 0.112 | 0.110 | 0.951 |
| **chatterbox** | **sentence case** | **0.022** | **0.022** | 0.949 |
| cosyvoice2 | raw `ALL CAPS` | 0.425 | 0.430 | 0.954 |
| cosyvoice2 | sentence case | 0.212 | 0.204 | 0.957 |

**Chatterbox goes from 0.112 to 0.022 — a 5× reduction — on a change to nothing
but the case and terminal punctuation of the input string.** At 0.022 it sits on
top of the ground-truth row (0.021) and inside the 1.5–3% band published for
zero-shot systems on this corpus. SIM-o is unmoved (0.951 → 0.949), so this is
specifically an intelligibility effect, not a general quality shift.

**This refutes a claim recorded earlier in this log and in the README**: that
Chatterbox at 11% was "well behind published zero-shot systems … a gap large
enough not to be protocol noise". It was protocol noise. Mine. The decomposition
above correctly excluded temperature, text length and the reference cap, and the
~0.095 residual it could not explain is almost exactly the 0.090 that text
presentation accounts for.

LibriSpeech and EdAcc both distribute transcripts as unpunctuated upper case. That
is a corpus convention for *scoring*, and I passed it straight through to a
generative model as though it were input. WER scoring is case- and
punctuation-insensitive — verified — so the metric never noticed.

## Why "give both models the same text" was the wrong kind of fairness

Feeding every backend the raw corpus string looks scrupulously even-handed. It is
not, because the models are not equally robust to it: the same input costs
Chatterbox 0.090 WER and CosyVoice 2 0.213. Identical treatment, unequal handicap.
The defensible protocol is to give each model input of the kind it was trained on
and to record that choice, which is what `--text-case` now does — defaulting to
`raw` so every number recorded before this remains comparable, and printing the
setting in the provenance banner.

## The CosyVoice 2 row is a floor, not a measurement

**Do not read this as a model comparison.** The run logged `no frontend is
avaliable`: neither `ttsfrd` nor `wetext` was installed, so CosyVoice 2 ran with
its text normalization entirely disabled, which my trimmed CPU install caused. Its
0.212 is a lower bound set by my environment. A re-run with `wetext` active is
recorded separately. Chatterbox has no comparable frontend dependency, so the two
rows are not yet measuring the same thing and no ranking between them is claimed.

## A third small-sample lesson

The probe that found this effect used 3 speakers × 3 utterances and put CosyVoice 2
at **0.048** in sentence case. At n=60 the same configuration measured **0.212** —
4.4× worse. The probe was right about the *direction* and badly wrong about the
*magnitude*, which is the same failure as the "consistent ~0.05 SIM gap" (n=12) and
the "54% blend rate" earlier in this file. n≤12 is good for deciding what to
measure properly and not for any number that gets written down.

## What this invalidates

Every synthesis WER previously recorded in this file used raw `ALL CAPS` text and
therefore understates its backend, by ~0.090 for Chatterbox on LibriSpeech. The
affected numbers are the EdAcc and LibriSpeech rows above and the golden-speaker
study, whose EdAcc text has the same convention. Speaker similarity appears
unaffected (0.951 vs 0.949). Re-running the golden study in sentence case is the
immediate follow-up; the rows are kept rather than deleted, since the comparison
*between* them is what produced this finding.

## The missing normalizer turned out not to matter — and proved reproducibility

Re-ran CosyVoice 2 with `wetext` installed and the log confirming `use wetext
frontend` (raw log `eval_cosy_libri_wetext_9950x.txt`). Result: **0.212 / 0.204 /
0.957 — identical to three decimals**, and the per-utterance JSON is byte-identical
to the run that had no frontend at all.

Two things follow. Text normalization is a **no-op on this corpus**: LibriSpeech
sentence-case English prose contains no numbers, currency or abbreviations for a
normalizer to expand, which is exactly what `wetext` returned when tested directly.
So the earlier `no frontend is avaliable` warning was a real defect in my install
and immaterial to this measurement — worth having checked rather than assumed, and
worth not having published a ranking over.

And two separate processes produced byte-identical rows at seed 1234, which is a
stronger reproducibility check than the earlier golden-speaker pair (that one
differed by 0.001 on one WER, consistent with thread reduction order).

**Where that leaves the comparison.** In sentence case, n=60, identical utterances:
Chatterbox 0.022, CosyVoice 2 0.212. One asymmetry remains and it is mine: the
reference cap is **20 s for Chatterbox, measured** (0.126 → 0.097 at 20 s versus
10 s) and **10 s for CosyVoice 2, taken from its own documentation and not measured
here**. A 10× WER difference is much larger than the ~24% relative the cap moved on
Chatterbox, so it is unlikely to be the explanation — but "unlikely" is not
"measured", and three borrowed numbers have already failed in this project. A
reference-cap sweep for CosyVoice 2 is the outstanding work before any ranking
between these two backends is stated.

---

# Run — 2026-08-11 — golden speaker, re-measured in sentence case with a matched control

Supersedes the earlier golden-speaker run. Same 12 EdAcc learners, same 6 held-out
utterances each (**n = 72**), same seed, same source speaker — two changes:
`--text-case sentence`, and identity retention scored against the source speaker's
**real audio** instead of its TTS. Raw logs `eval_golden_n72_cased_9950x.txt`,
`per_utterance_golden_final.json`.

| signal | SIM to learner | WER (sentence case) | WER (was, raw caps) |
|---|---|---|---|
| learner real *(ceiling)* | **0.924** | 0.225 | 0.225 |
| native tts | 0.621 | **0.076** | 0.208 |
| learner tts *(plain cloning)* | 0.856 | **0.087** | 0.250 |
| **golden** | **0.818** | **0.073** | 0.208 |

## The text-presentation prediction held exactly

Every synthetic arm improved by roughly 2.7-2.9x, and **`learner real` did not move
at all** (0.225 both times) — which is the control that proves the mechanism: real
audio does not read text, so only the arms that were handed ALL-CAPS input could
change. The earlier anomaly where native tts (0.208) scored *worse* than real
spontaneous speech (0.225) is gone; at 0.076 it is now cleanly better, as clean read
speech should be.

## Conversion still earns its place, by a smaller margin

WER cost against the TTS it came from is **−0.003** — conversion does not damage the
speech. Against plain cloning, golden is **0.014 better on WER** for **0.039 worse on
SIM**. Both effects shrank once the text handicap was removed (they were 0.041 and
0.031 in raw caps), which is the expected direction: the accent-preservation penalty
that conversion recovers is smaller when the model is not also fighting its input.

## Identity: the output carries both voices, measured properly this time

| quantity | value |
|---|---|
| acquired — SIM(golden, learner's real held-out) | 0.818 |
| retained — SIM(golden, source speaker's **real** audio) | 0.791 |
| control — SIM(learner clone, source real) | 0.677 |
| excess over control | **+0.114** |
| scale measured in this run | 0.621 different speaker … 0.924 same speaker |

Both numbers are synthetic-vs-real with different text and no shared prosody, so for
the first time they are directly comparable. On the run's own scale the output sits
**65% of the way toward the learner and 56% of the way toward the source**. There is
real retained source identity — +0.114 above a clone that never touched the source —
and it coexists with substantial acquired identity. Partial conversion, not a clean
transfer.

## The "majority blend" claim is retracted

Per utterance, **32/72 (44%)** of outputs are closer to the source than to the
learner — a large minority, not a majority. The earlier 54% came from `blend_rate`
comparing `sim_golden` (synthetic-vs-real) against `sim_golden_to_source`
(synthetic-vs-**synthetic**), and golden is a conversion of that very waveform, so it
shares the source's vocoder and prosody; the same-pipeline bias measured +0.070.

`blend_rate` now accepts only the matched key and raises if handed the unmatched one,
with the 44%-to-54% inflation named in the error. This is the second time this exact
statistic misled me and the third time in this project that a like-for-unlike
similarity comparison produced a false finding — the first was SIM-r vs SIM-o.

## What still bounds this

- **One pronunciation source** (LibriSpeech 1089), n=1 on that side.
- **Cross-gender conversion is pooled**, not separated; per-speaker rows are in the JSON.
- **Ground truth is impossible by construction** — a non-native speaker producing
  native pronunciation cannot be recorded. Every number here is a trade-off, never an
  accuracy.
- EdAcc `learner real` WER stays high (0.225) because the corpus transcribes
  disfluencies verbatim; that arm is not comparable with the synthetic ones, and no
  claim above crosses that line.

---

# Run — 2026-08-12 — the CosyVoice 2 gap was my reference cap, and it is the fourth borrowed number to fail

The comparison this project refused to publish, now measurable. LibriSpeech
test-clean, 12 speakers × 5 held-out (**n = 60**, identical utterances in every
row), sentence case, seed 1234, 9950X. Raw logs `eval_cosy_libri_ref30_9950x.txt`,
`per_utterance_cosy_libri_ref30.json`.

## Found by splitting the existing run, before spending compute

CosyVoice 2's mean WER of 0.212 was not uniform mediocrity. Its **median was
0.032** while its mean was 0.231 — the worst 10% of utterances carried 55% of the
error mass. Splitting the same run by whether my 10 s cap had truncated the
reference:

| CosyVoice 2, 10 s cap | WER mean | WER median | n |
|---|---|---|---|
| reference **hit** the cap | **0.507** | 0.375 | 25 |
| reference **under** the cap | **0.033** | **0.000** | 35 |

`corr(seconds discarded, WER)` = **+0.870**. The mechanism is visible in the audio
lengths: `corr(duration ratio, WER)` = **+0.861**, with outputs up to **2.42× the
expected duration**. A hard cut at 10.0 s lands mid-phrase, and the model then
over-generates — autoregressive looping. Chatterbox at its measured 20 s cap
discards a median of 0.7 s and shows `corr` = +0.126, i.e. nothing.

## Confirmed directly, and the subgroup prediction held

Re-run with `--ref-seconds 30`, which truncates nothing on this corpus (longest
reference used: 22.6 s):

| | WER | WER-2 | SIM-o |
|---|---|---|---|
| ground truth | **0.021** | 0.016 | 0.964 |
| cosyvoice2, 10 s cap | 0.212 | 0.204 | 0.957 |
| **cosyvoice2, no truncation** | **0.036** | **0.034** | 0.958 |

**5.9× better from removing my own truncation.** The untruncated subgroup predicted
0.033; the clean run measured 0.036. The failure mode is gone: utterances above WER
0.5 went 8 → **0**, over-long outputs 13 → **1**, worst duration ratio 2.42 → 1.42.

## The comparison, finally stateable

Each backend at a cap that was **measured rather than borrowed**:

| system | reference cap | WER | WER-2 | SIM-o |
|---|---|---|---|---|
| ground truth | — | 0.021 | 0.016 | **0.964** |
| chatterbox | 20 s (measured optimum) | **0.022** | **0.022** | 0.949 |
| cosyvoice2 | no truncation | 0.036 | 0.034 | **0.958** |

Both sit essentially on the ground-truth row, and **the 10× difference reported
earlier was entirely an artefact of my configuration.** What remains is a genuine
trade-off rather than a ranking: Chatterbox has the lower WER, CosyVoice 2 the
higher speaker similarity — and CosyVoice 2's 0.958 is nearer the ground-truth
ceiling of 0.964. On one corpus, in one language, with n=60. Chatterbox's own sweep
found its curve plateaus by 20 s, so its row is not disadvantaged by using 20 rather
than 30.

## Fourth borrowed number, same failure

`ref_seconds = 10.0` for CosyVoice 2 came **from CosyVoice's own documentation**.
That is the same mistake already made and fixed for Chatterbox — where measurement
showed 20 s beats 10 s by 24% relative — reintroduced for a new backend because the
vendor stated it. The running tally of constants taken from documentation or papers
rather than measured here: published SIM ceiling (0.69–0.76, actual 0.89–0.98),
vendor RTF claim (3× realtime, actual 2× slower), the original 10 s cap, and now
this. **Four for four.**

Refusing to publish the ranking while this was outstanding was correct for the
stated reason — "unlikely is not measured". Ranking on the means would have reported
a well-regarded model as 10× worse than it is.

---

# Run — 2026-08-14 — text presentation replicates on a second corpus

EdAcc re-run in sentence case, matched to the raw-caps run above: same 12 speakers,
5 held-out each (**n = 60**), scan 9000, seed 1234, `large-v3` + HuBERT-CTC. Raw log
`eval_edacc_n60_cased_9950x.txt`, rows `per_utterance_edacc_cased.json`.

| system | text as the model sees it | WER | WER-2 | SIM-o |
|---|---|---|---|---|
| ground truth | — | 0.149 | 0.281 | 0.922 |
| chatterbox | raw `ALL CAPS` | 0.369 | 0.427 | 0.862 |
| **chatterbox** | **sentence case** | **0.093** | **0.126** | 0.858 |

**A 4.0x reduction, replicating the LibriSpeech result on a completely different
domain** — spontaneous, accented, real recording conditions rather than clean read
speech. Both ASRs move together (0.369→0.093 and 0.427→0.126), so it is not a
Whisper artefact.

**The control held exactly.** The ground-truth row is *identical to three decimals
in both runs* (0.149 / 0.281 / 0.922). Real audio does not read text, so it cannot
respond to presentation; had it moved, something other than presentation would have
changed between the runs. Speaker similarity is also near-unmoved (0.862 → 0.858),
matching LibriSpeech (0.951 → 0.949): this is specifically an intelligibility
effect, not a general quality shift.

## The harness caught its own result crossing a line

At 0.093 the synthesis row now sits **below** the 0.149 human row, and the table
flagged it automatically:

```
chatterbox    60   0.093   0.126   0.858    (WER below human: not an intelligibility win)
```

That guardrail exists because modern TTS routinely scores better ASR-WER than real
speech — clean read synthesis has none of the disfluencies, false starts and fillers
that EdAcc transcribes verbatim, so the ASR has less to get wrong. **A WER below the
ground-truth row is not evidence of better intelligibility**, and without the warning
this run would read as "synthesis beats humans".

## An archival defect in the raw-caps run

`eval_edacc_n60_9950x.txt` begins mid-stream: no provenance banner, no `cmd:` line,
and its first speaker row is missing. It lists 11 speakers x 5 held-out = 55 against
a table reporting n=60; twelve speakers reconciles it, and the sentence-case run's
speaker list is exactly those 11 plus `EDACC-C08-A`. So the two runs are matched and
the comparison stands — but that file does not meet this project's own rule that
every recorded run keeps its raw stdout with hardware, model and date. The original
stdout predates the current archiving discipline and is not recoverable. Recorded
here rather than left as a silently incomplete log.

---

# Study — 2026-08-14 — is the text-presentation effect specific to autoregressive TTS?

The first *paired* test of the finding, and a test of what causes it. Earlier
evidence compared separate runs post-hoc; this synthesizes the same utterance twice
with the same seed, varying only the case of the input string, so a per-utterance
difference cannot be sampling noise or utterance difficulty.

| | |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, CPU |
| corpus | LibriSpeech test-clean, 12 speakers x 5 — **n = 60**, identical in both arms |
| ASR | `faster-whisper large-v3` int8 |
| seed | 1234 |
| raw logs | `present_chatterbox_librispeech_9950x.txt`, `present_piper_librispeech_9950x.txt` |

| backend | architecture | raw `CAPS` | sentence | ratio | improved / worse / same | sign test |
|---|---|---|---|---|---|---|
| **chatterbox** | autoregressive (T3) | 0.109 | **0.025** | **4.4x** | 37 / 4 / 19 | **p = 1e-7** |
| **piper** | non-autoregressive (VITS) | 0.051 | 0.036 | 1.4x | 12 / 7 / 41 | p = 0.36 |

**The autoregressive model is hit hard and the non-autoregressive one is not
measurably hit at all.** Chatterbox improves on 37 of 60 utterances against 4 that
worsen — a result no reasonable amount of noise produces. Piper's 12-versus-7 is
indistinguishable from chance, and its 1.4x rests on that.

## The mechanism, not just the correlation

Over-generation — output running well past the duration the text warrants — appears
**only in the autoregressive model, and only on raw capitals**: 2/60 for Chatterbox
on raw, **0/60** once cased, and **0/60 for Piper in both arms**. CosyVoice 2, also
autoregressive, showed the same signature far more strongly (13/60, up to 2.42x
duration, `corr(duration ratio, WER) = +0.861`).

That is what the architecture predicts. An autoregressive decoder conditions each
step on its own previous outputs, so unfamiliar input can push it into a state it
never recovers from; VITS emits a whole utterance in one non-recurrent pass and
structurally *cannot* run away. The effect and its proposed cause are both present
in one family and both absent in the other.

## The control reproduced for the fifth time

The real-audio row measured **0.021** in both studies — identical, and identical to
the ground-truth row of every other LibriSpeech run in this file. Real speech reads
no text, so it cannot respond to presentation; that it never moves is what licenses
attributing the synthetic arms' movement to presentation at all.

## What this does not establish

**One model is not a family.** Piper is a single non-autoregressive system, and a
null result on one model is consistent with the architectural explanation without
demonstrating it. A second non-AR model would strengthen this considerably.

**The absolute WERs are not comparable across the two backends.** Piper does not
clone: it reads in a fixed studio voice, which is an easier task than matching a
target speaker, and that alone could explain its lower raw-caps WER (0.051 against
0.109). Only the *within-backend* raw-to-sentence ratio is a fair comparison, which
is why the ratio is what is reported.

**Piper's voice carries its own licence.** `en_US-lessac-medium`, from
`rhasspy/piper-voices`; the CLI is GPL-3.0-or-later and is invoked as a subprocess
rather than imported. See docs/LICENSES.md.

---

# Study — 2026-08-14 — targeted correction: identity is preserved, intelligibility is not

Regenerate only the words a calibrated reliability policy flags, splice them into the
learner's own recording, and compare against regenerating the whole utterance.

| | |
|---|---|
| hardware | AMD Ryzen 9 9950X, 16 threads, CPU |
| learners | 8 EdAcc speakers, 4 held-out each — **n = 29** scored |
| selector | speechlens per-word reliability, `policy="auto"`, threshold 0.5, pad 1 word |
| correction | `golden_speech` (Chatterbox TTS → ChatterboxVC) on the reference words |
| ASR / encoder | `faster-whisper large-v3` int8 / `wavlm-base-plus-sv` |
| raw logs | `eval_targeted_edacc_9950x.txt`, `per_utterance_targeted.json` |

| arm | SIM to learner | WER |
|---|---|---|
| learner real *(identity ceiling)* | **0.917** | 0.176 |
| golden whole | 0.838 | **0.045** |
| **golden targeted** | **0.909** | **0.253** |

Regenerated **18.4% of words / 19.5% of audio**, 2.3 seams per utterance; 8 of 29
utterances had nothing flagged at all.

## The identity claim holds

Targeting reaches **0.909 against a 0.917 ceiling — 99% of the available identity**,
where whole-utterance regeneration reaches 0.838 (91%). That is the engineering case
for targeting, and it is confirmed: replacing a fifth of the audio costs almost none
of the speaker.

## The intelligibility claim fails, and the split says why

WER on the targeted arm is **worse than leaving the audio alone** (0.253 vs 0.176).
This is the arm whose WER was expected to be near-circular — flagged audio is replaced
by speech synthesized *from the reference transcript*, so an ASR reading it back should
score better almost mechanically. It scores worse. Splitting by whether anything was
actually spliced:

| | n | WER real → targeted | replaced |
|---|---|---|---|
| nothing flagged | 8 | 0.195 → 0.195 (**+0.000**) | 0% |
| spliced | 21 | 0.169 → 0.275 (**+0.106**) | 26.9% |

The untouched group is an exact no-op — identical WER *and* identical similarity —
which rules out the pipeline damaging audio it does not touch, and confirms the
damage is caused by splicing rather than by those utterances being harder.

**And it is not the seams.** Damage scales with how much audio is replaced
(`corr = +0.675`) and barely at all with how many boundaries are created
(`corr = +0.109`). The crossfades are doing their job; the *replacement audio itself*
is the problem.

That is a specific diagnosis, because the same synthesis method scores **0.045** when
it produces a whole utterance. Two or three words synthesized in isolation are not a
small version of that — they get no surrounding context, so their prosody, pacing and
duration do not fit the gap they are dropped into, and the mismatch costs more
recognition than the mispronunciation did.

## What to try next, and what not to conclude

The fix this points at is **synthesize the whole utterance, then splice in only the
corrected region from it**: the fragment then carries context-appropriate prosody and
duration, at the cost of one extra full synthesis per utterance. That is a direct
test of the diagnosis above and is the obvious next iteration.

Not concluded here: whether the *right* words were selected. That needs labels for
which words the learner actually mispronounced, which EdAcc does not carry — the same
"ground truth is impossible by construction" limit `golden.py` documents. Low ASR
confidence marks hard-to-recognise words, and mispronunciation is only one of the
causes; this study measures the cost of acting on that signal, not its accuracy.

---

# Run — 2026-11-20 — StyleTTS 2, a diffusion backend: on the ground-truth row for WER, lowest SIM of the three

| field | value |
|---|---|
| hardware | Intel Xeon @ 2.10 GHz, **4 threads**, Linux x86_64, no GPU — **not the 9950X** the earlier rows ran on |
| package | `torch` 2.6.0+cpu, `transformers` 4.49.0, `yl4579/StyleTTS2` @ `5cedc71`, phonemizer 3.6.0 CLI, espeak-ng 1.51 |
| model | `yl4579/StyleTTS2-LibriTTS` `epochs_2nd_00020.pth`, SHA-256 `1164ffe1…57dca0a` (matches the published LFS id) |
| settings | upstream's LibriTTS demo values: `alpha=0.3`, `beta=0.7`, `diffusion_steps=5`, `embedding_scale=1.0`; reference cap 30 s (truncated nothing — longest reference 22.6 s) |
| protocol | identical to the Chatterbox and CosyVoice 2 rows: LibriSpeech test-clean, 12 speakers × 5 held-out, seed 1234, sentence case |
| ASR / encoder | `faster-whisper large-v3` int8 + `hubert-large-ls960-ft` / `wavlm-base-plus-sv` |
| raw logs | `eval_styletts2_libri_cased_xeon.txt`, `per_utterance_styletts2_libri_cased.json` |

| system | hardware | reference cap | WER | WER-2 | SIM-o |
|---|---|---|---|---|---|
| ground truth | 9950X | — | 0.021 | 0.016 | 0.964 |
| ground truth | **Xeon** | — | **0.019** | 0.016 | 0.964 |
| chatterbox | 9950X | 20 s | 0.022 | 0.022 | 0.949 |
| cosyvoice2 | 9950X | none | 0.036 | 0.034 | 0.958 |
| **styletts2** | **Xeon** | 30 s | **0.019** | **0.015** | **0.931** |

**On the ground-truth row for intelligibility, under both ASRs** — and, unlike the
EdAcc rows that crossed below it, there is no reason here to distrust that: this is
read speech with verbatim-clean transcripts, so a system matching it is matching it.
**And the lowest speaker similarity of the three backends**, 0.033 under the ceiling
against 0.015 for Chatterbox and 0.006 for CosyVoice 2. That gap is not seed luck: a
second seed in the ablation below gives **0.926**, so StyleTTS 2 sits 0.033–0.038 under
the ceiling, and its seed spread (0.005) is a quarter of its distance to Chatterbox.
What separates the three is again a trade-off, now with a third point on it:
StyleTTS 2 buys the best intelligibility with the weakest identity.

A plausible reason, **not tested here**: StyleTTS 2 compresses the reference into a
single 256-d style vector pooled over its whole length, while Chatterbox and
CosyVoice 2 condition on the reference as a sequence — speech tokens and mel frames.
One global vector has less room for a voice. It is also the oldest of the three
(NeurIPS 2023), and its multi-speaker checkpoint was trained on LibriTTS
train-clean-460 alone, about 245 hours.

**No over-generation, as the architecture predicts.** Durations come from a predictor
in one pass, so the output length is fixed before any audio exists. Duration ratio
averages 1.02 with a maximum of 1.26; nothing approaches CosyVoice 2's 2.42×. 45 of 60
utterances have zero errors and none exceeds a WER of 0.2.

## The ground-truth row moved, by one utterance, and that bounds cross-machine reads

Same 60 utterances, same Whisper checkpoint, same int8 compute type: the real-audio WER
is 0.021 on the 9950X and **0.019** on this Xeon. Per utterance, exactly **one of 60**
differs — a 48-word clip on which the 9950X's Whisper made 3 errors and this one made
none. CTranslate2's int8 kernels are CPU-specific, and that is enough to move a
transcript. HuBERT (0.016) and SIM-o (0.964) reproduce to three decimals.

So a WER difference of ~0.002 between rows from *different machines* can be ASR noise
rather than a system difference. The 0.003 between StyleTTS 2 and Chatterbox is the
same size as that shift: it is not a ranking, and this write-up does not make one.
Everything compared within one table below is from one machine and one process.

## What was checked before any number was taken

The backend is not upstream's code, so the first job was showing it computes the same
thing — and three of the checks failed first, which is why they are checks.

- **Bit-exact against upstream.** Upstream's loaders and the inference function from
  its LibriTTS demo notebook, run side by side with the backend at three diffusion
  settings × three texts: **max |difference| = 0** on all nine
  (`styletts2_parity_xeon.txt`).
- **The GPL boundary costs nothing.** phonemizer is GPL-3.0, so the backend runs its
  `phonemize` CLI as a subprocess. Byte-identical to upstream's library call on **all
  2,620** LibriSpeech test-clean transcripts in both presentations
  (`styletts2_phonemizer_xeon.txt`). The first version of the wrapper failed its own
  test: the CLI **silently drops empty input lines**, which would shift every later
  text onto the wrong phonemes. A line-count check refused rather than misaligning;
  `--preserve-empty-lines` fixes it.
- **A checkpoint torch 2.6 will not load safely.** The text aligner's pickle uses a
  `POP` opcode that the `weights_only` unpickler does not implement, so no allowlist
  rescues it — the stock alternative is `weights_only=False`, which executes the file.
  It was disassembled with `pickletools` first (tensors, containers, and pickled
  `AdamW`, `OneCycleLR` and `getattr` training state) and is loaded by a restricted
  unpickler that builds the first two and stubs the third.
- **No silent partial loads.** Upstream loads PL-BERT with `strict=False`. A renamed
  parameter in a newer `transformers` would leave random weights in place and the
  model would still talk. Every load here fails on any missing or unexpected key.
- **A memory leak that killed a run.** With oneDNN on, resident memory grew 150–200 MB
  per synthesis without bound — 2.0 → 5.7 GB over 24 utterances — while the number and
  size of live tensors stayed exactly constant. It needs varying input shapes (one
  repeated shape plateaus) and survives `malloc_trim`, a fixed mmap threshold, a single
  malloc arena and `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0`; it disappears with oneDNN off.
  Every utterance has a new length, so the first headline run was OOM-killed. oneDNN
  is now off inside the backend only, by context manager — the global flag would change
  the evaluation encoders' numerics in the same process. Parity holds under the same
  setting; the fix itself moves samples by at most 0.011 and changed no output length.
  Measured alone on the machine, 24 utterances, each setting in a fresh process
  (`styletts2_memory_xeon.txt`):

  | oneDNN | RSS after load | after 8 | after 24 | live tensors | RTF |
  |---|---|---|---|---|---|
  | on | 2,001 MB | 3,597 MB | **5,707 MB** (+3,706) | 1648 → 1648 | 1.008 |
  | **off** | 2,001 MB | 2,329 MB | **2,393 MB** (+392, flat after ~8) | 1648 → 1648 | **0.917** |

  **The fix costs no speed** — oneDNN off was 9% *faster* on this run. An informal
  reading during the leak hunt had suggested oneDNN on was faster; this controlled
  comparison supersedes it.

## Reproduced three times, once across a container restart

The default setting was synthesized and scored in three separate processes: this
headline run, the `default` arm of the diffusion ablation, and the sentence-case arm of
the text-presentation study — the last two either side of a container restart. Per
utterance, all **60 of 60** agree on WER, synthesized duration and the real-audio WER
in all three (the headline file stores values rounded; the two unrounded files match
exactly). So the default-setting audio is the same in all three studies, and the seed
fixes the output on this machine, decoder noise included.

---

# Study — 2026-11-21 — what the diffusion in StyleTTS 2 does: nothing WER or SIM can see, except at full strength, where identity becomes a draw

StyleTTS 2's diffusion does not generate audio. It samples one 256-d **style vector**
— 128 dimensions of timbre ("acoustic") and 128 of prosody — conditioned on the text
and the reference, and inference blends that sample with the reference's own style:
`alpha` sets how much of the timbre half comes from the sample, `beta` how much of the
prosody half. At `alpha = beta = 0` the sampler still runs but its output is discarded:
the voice is the reference's style vector alone, so that arm is diffusion **off**.

| | |
|---|---|
| hardware | Intel Xeon @ 2.10 GHz, 4 threads, CPU — the same machine and settings as the headline row |
| corpus | LibriSpeech test-clean, 12 speakers × 5 — **n = 60**, identical in every arm; seed 1234; sentence case |
| ASR / encoder | `faster-whisper large-v3` int8 / `wavlm-base-plus-sv` |
| raw logs | `diffusion_ablation_xeon.txt`, `per_utterance_diffusion_ablation.json`. A container restart cut the run after six arms; it resumed from its checkpoint, so those six print there as "done in a previous run" and their own output is in `diffusion_ablation_part1_xeon.txt`. (An earlier attempt, before checkpointing existed, was lost to a restart outright; nothing from it is used.) |

| arm | `alpha` / `beta` | steps | WER | SIM-o | duration ratio (mean / max) | dWER vs off | better / worse | p | dSIM vs off | higher / lower | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | 0.019 | 0.964 | 1.00 | | | | | | |
| **diffusion off** | 0 / 0 | 5 | 0.016 | 0.924 | **1.07 / 1.40** | *control* | | | | | |
| off, seed + 1 | 0 / 0 | 5 | 0.020 | 0.924 | 1.07 / 1.40 | +0.003 | 1 / 3 | 0.63 | −0.000 | 29 / 31 | 0.90 |
| **default** | 0.3 / 0.7 | 5 | 0.019 | 0.931 | 1.02 / 1.26 | +0.002 | 3 / 5 | 0.73 | +0.007 | 35 / 25 | 0.25 |
| default, seed + 1 | 0.3 / 0.7 | 5 | 0.014 | 0.926 | 1.02 / 1.24 | −0.002 | 3 / 2 | 1 | +0.002 | 27 / 33 | 0.52 |
| default, 10 steps | 0.3 / 0.7 | 10 | 0.023 | 0.926 | 1.01 / 1.23 | +0.006 | 2 / 8 | 0.11 | +0.002 | 28 / 32 | 0.70 |
| default, 10 s reference cap | 0.3 / 0.7 | 5 | 0.022 | 0.931 | 1.02 / 1.26 | +0.005 | 3 / 6 | 0.51 | +0.007 | 37 / 23 | 0.09 |
| **full diffusion** | 1 / 1 | 5 | 0.022 | 0.920 | 1.00 / 1.24 | +0.005 | 4 / 7 | 0.55 | −0.004 | 29 / 31 | 0.90 |
| **full diffusion, seed + 1** | 1 / 1 | 5 | 0.024 | **0.893** | 0.98 / 1.20 | +0.007 | 5 / 11 | 0.21 | **−0.031** | **14 / 46** | **4e-5** |

Corpus-level WER; better / worse and higher / lower count utterances against the
diffusion-off arm at seed 1234 (for the seed + 1 rows too), ties dropped; p is a
two-sided exact sign test.

**Intelligibility: no setting differs significantly from diffusion off, and every
difference is seed-sized.** The seed alone, with every setting held, moves corpus WER
by 0.002–0.005 and changes the WER of 4–9 of the 60 utterances. Turning diffusion on,
doubling its steps or running it at full strength moves it by 0.002–0.007 from the
control, and the smallest p is 0.11.

**Identity: the default blend gains nothing measurable, and full diffusion makes the
voice a draw.** The default is +0.007 SIM on one seed and +0.002 on the other, neither
significant. Full diffusion is −0.004 on one seed and **−0.031 on the other, worse on
46 of 60 utterances** (p = 4 × 10⁻⁵). At `alpha = 1` the timbre half of the style
vector is entirely sampled, so how close the voice lands to the reference depends on
the sample; the default's `alpha = 0.3` keeps 70% of the reference's own timbre, and
that is what holds identity steady across seeds. Diffusion off is the most stable of
all: SIM is 0.924 on both seeds. The decoder's own randomness — its random phases,
which change 4 of 60 transcripts — leaves mean identity where it was.

**Speaking rate is the one thing diffusion visibly changes.** With diffusion off, the
output runs **7% longer than the speaker's real reading** of the same text (mean 1.07,
longest 1.40), and identically so on both seeds, because with the reference's prosody
vector the duration predictor is deterministic. Blending in sampled prosody brings
that to 1.02 (default) and 1.00 (full). That is the prosody half doing what it is for,
and it does not register in WER or SIM-o because neither measures prosody.

**Ten diffusion steps instead of five changes nothing** (+0.006 WER, 2 / 8, p = 0.11;
+0.002 SIM).

**A 10 s reference cap costs StyleTTS 2 nothing measurable.** On the 30 utterances
whose reference exceeds 10 s, capping it moves WER 0.018 → 0.023 and SIM 0.936 →
0.935. The same cap cost CosyVoice 2 5.9× in WER, by cutting its reference mid-phrase
and sending the autoregressive decoder into over-generation; StyleTTS 2 pools the
reference into one vector and sets durations in one pass, so a mid-phrase cut has
nothing to run away with. Its 30 s default is therefore not load-bearing — but it was
chosen to truncate nothing, and nothing here says capping helps either.

## What this does not establish

**"The diffusion does nothing" is not the conclusion.** It is that the diffusion does
nothing *these two metrics* can see, at n = 60, on read speech, over two seeds. The
StyleTTS 2 paper's case for diffusion is naturalness and prosodic variety, measured by
listeners; this harness has no naturalness or prosody metric, and the one prosodic
quantity it records — duration — is exactly where diffusion shows up.

**Two seeds bound the spread; they do not measure it.** The full-diffusion result says
identity at `alpha = 1` *can* drop by 0.031 on a seed, not how often.

---

# Study — 2026-11-21 — a second non-autoregressive model *is* hit by raw capitals, through its phonemizer

The 2026-08-14 study named its own gap — "A second non-AR model would strengthen this
considerably" — because its one non-autoregressive model, Piper, was not measurably
hurt by raw `ALL CAPS` input, and the effect was read as specific to autoregressive TTS. StyleTTS 2
is that second model, and a better test than Piper — it clones, so it is doing the same
task as Chatterbox. Same script, same paired design: each utterance synthesized twice
with the same seed, only the case of the input differing.

| | |
|---|---|
| hardware | Intel Xeon @ 2.10 GHz, 4 threads, CPU |
| corpus | LibriSpeech test-clean, 12 speakers × 5 — **n = 60**, identical in both arms; seed 1234 |
| ASR | `faster-whisper large-v3` int8 |
| raw logs | `present_styletts2_librispeech_xeon.txt`, `per_utterance_present_styletts2.json`, `present_mechanism_styletts2_xeon.txt`, `spelled_word_check_xeon.txt` |

| backend | architecture | input | raw `CAPS` | sentence | ratio | improved / worse / same | sign test |
|---|---|---|---|---|---|---|---|
| chatterbox *(9950X)* | autoregressive (T3) | text tokens | 0.109 | 0.025 | 4.4× | 37 / 4 / 19 | p = 1e-7 |
| piper *(9950X)* | non-autoregressive (VITS) | espeak-ng phonemes | 0.051 | 0.036 | 1.4× | 12 / 7 / 41 | p = 0.36 |
| **styletts2** *(Xeon)* | non-autoregressive (style diffusion) | espeak-ng phonemes | **0.044** | **0.022** | **2.0×** | **16 / 2 / 42** | **p = 0.001** |

Mean per-utterance WER, as in the 2026-08-14 table — which is why the sentence-case
figure reads 0.022 here and 0.019 (corpus-level) in the headline: the same audio, 60 of
60 utterances identical. The real-audio control reads 0.020 on this scale.

**StyleTTS 2 is hit, and significantly**: 16 utterances improve in sentence case, 2
worsen. The reading that the effect is specific to autoregressive TTS — in that study
and, until this change, in the README — does not survive its own stated caveat: one
non-AR model was not a family, and the second one breaks the pattern.

**But not by over-generation.** Raw-caps output is *shorter* than sentence case in **59
of 60** utterances, by 13% on average (duration ratio 0.89 against 1.02), nothing
exceeds 1.11× the real reading, and the length change is unrelated to the WER change
(`corr = −0.14`). A one-pass duration predictor cannot run on, as the architecture
predicts; whatever reaches StyleTTS 2 is a different mechanism from the one in
CosyVoice 2.

## The mechanism, predicted before the run

StyleTTS 2 reads phonemes, so case can only reach it through espeak-ng, and espeak does
treat capitals differently: on all 2,620 test-clean transcripts, raw caps change the
phonemes of 57% (`styletts2_phonemizer_xeon.txt`). Counted word by word, the commonest
change is a function word losing its weak form (`TO` as /tuː/ rather than /tə/, 1,118
times), then stress shifts (`I`, `NOT`); and one kind is an unambiguous error: short
words **spelled out as initialisms** — `IT` read as "I.T." (491 times), `US` as "U.S."
(58). So `scripts/presentation_mechanism.py` was written and committed
*before this study ran* with one prediction: if the effect lives in the front end, the
damage concentrates in the utterances that contain a spelled-out word. A word counts as
spelled out if its caps phonemes equal its letters read one by one and differ from its
lower-case reading; no word list. In these 60 transcripts that finds one word, `IT`.

| utterances | n | raw `CAPS` | sentence | delta | improved / worse | sign test |
|---|---|---|---|---|---|---|
| **contain `IT`** | 13 | **0.096** | 0.030 | **−0.067** | **8 / 0** | **p = 0.008** |
| do not | 47 | 0.030 | 0.020 | −0.010 | 8 / 2 | p = 0.11 |
| all | 60 | 0.044 | 0.022 | −0.022 | 16 / 2 | p = 0.001 |

**The prediction holds.** The 13 utterances with a spelled-out word — 22% of the
sample — carry two-thirds of the total improvement, and every one of them that changes
gets better. In the other 47 the remaining difference is a seventh the size per
utterance and not distinguishable from chance at this n; weak forms and stress may
account for it, but this run cannot say.

**And the errors are at the word.** `scripts/spelled_word_check.py` re-synthesizes
those 13 utterances in raw caps — all 13 reproduce the study's WER exactly — and looks
inside them:

- espeak spells `IT` out **in context**, not only in isolation, at **16 of its 17**
  occurrences, counted in the phonemes the backend actually synthesized;
- Whisper hears **9 of the 17** as something other than "it" — "i d", "id", "eye he",
  "aight he", "i it", or nothing;
- those positions hold **15 of the 28** raw-caps errors in these utterances (54%),
  counting the word and insertions directly beside it only. A neighbour pulled into
  the error — "IT is" heard as "eye he has" — is not counted, so that is a floor.

The other 8 spelled occurrences come back as "it": Whisper's language model repairs
"eye-tee" from context. So on this mechanism WER undercounts what was synthesized.

## What this does not establish

**Why Piper escaped.** Piper also reads espeak-ng phonemes, so "phoneme input" alone
does not explain why one non-AR model is hit and the other is not. Piper's front end
(`piper-phonemize`) was not compared with the phonemizer CLI here, and its 12 / 7 is
not evidence of *no* effect, only of none measurable at n = 60.

**Cross-machine absolute WERs.** The Chatterbox and Piper rows ran on the 9950X; only
the within-backend ratio is compared, as before.

**The fix is the same either way.** Sentence case removes the spelled-out initialisms:
espeak reads `it`, and a sentence-initial `It`, as a word. `--text-case sentence` was
already the recommendation; this adds a second, independent reason for it.

---

# Study — 2026-11-21 — prosody, the axis the diffusion ablation could not see: diffusion flattens pitch, and varies it between seeds

The diffusion ablation above ended on "this harness has no naturalness or prosody
metric". This adds one — `speechforge/prosody.py`, `eval_zeroshot.py --prosody` — and
uses it on the question it was missing for.

| | |
|---|---|
| hardware | Intel Xeon @ 2.30 GHz, 4 threads, CPU — **a different host** from the 2.10 GHz runs above |
| package | `librosa` 1.0.1 (pYIN, the harness's tracker); `pyworld` 0.3.5 (WORLD's Harvest — the cross-check only, a dev dependency) |
| corpus | LibriSpeech test-clean, 12 speakers × 5 — **n = 60**; sentence case; seeds 1234 and 1235 |
| arms | the diffusion ablation's own: off (`alpha = beta = 0`), default (0.3 / 0.7), full (1 / 1), each on both seeds |
| raw logs | `diffusion_prosody_xeon.txt`, `per_utterance_diffusion_prosody.json`, `prosody_checks_xeon.txt`, `diffusion_prosody_truth_f0.npz`; the first tracker's run: `diffusion_prosody_first_tracker_xeon.txt`, `per_utterance_diffusion_prosody_first_tracker.json` |

**Same audio as the published WER and SIM.** The script refuses to report unless every
synthesized length matches the diffusion ablation's arm — and all **360 of 360** do,
on a different CPU. With diffusion on, a clip's length depends on the sampled style
vector, so equal lengths mean equal samples; sample values can still differ at
floating-point rounding across hosts, which a length check cannot see.

## What the harness measures now

Per utterance, beside the same measurement on the speaker's real recording of the same
text:

- **pitch range** — 10th-to-90th percentile of F0 within the utterance, in semitones, so
  register cancels. A monotone voice approaches 0. The target is the human row, not a
  direction.
- **pitch gap** — semitones between the clone's median F0 and the held-out recording's:
  the speaker's register. Its ground truth is another recording of the same speaker, as
  for SIM-o.
- **words / s** — over the speech span, with the reference-preparation silence rule.
- **jumps** — the share of 10 ms steps over 9 semitones: tracker errors, not prosody,
  flagged above 4%.

It renders as a separate block under the main table, only when `--prosody` is passed,
so every earlier table renders unchanged. A requested prosody scorer that cannot load
stops the run, like every other scorer.

## The first tracker failed, and the model supplied the truth to show it

Pitch trackers do not fail symmetrically on real and synthetic speech, and an octave
error reads as livelier intonation — so the range is a percentile range rather than a
standard deviation. That defence has a limit, and its own test found it: the first
version claimed the range was unaffected below 10% bad frames. Measured, 2% of frames
an octave off raise a standard deviation 32% and the range 4%; at 5%, 62% against 12%;
at 10% both break. Hence the `jumps` column.

Then the tracker itself failed. On a 4-utterance smoke run, pYIN over a fixed 50–500 Hz
range read two renditions with **identical** pitch contours as **15.98 and 5.70
semitones** of range. More than a tenth of one rendition's voiced frames sat exactly at
the 50 Hz search floor, an octave under the speaker, and the vocoder's noise decided
which frames fell there. The jump detector missed it: the pinned frames sat across
unvoiced gaps, never adjacent to a correct one.

**For synthetic audio the true pitch is known.** StyleTTS 2's vocoder drives its harmonic
source with the F0 contour its predictor outputs, so `diffusion_prosody.py` saves that
contour beside every clip — reading it leaves the audio byte-identical — and
`prosody_checks.py` scores trackers against it. With diffusion off, the contour is also
identical across seeds (60 of 60, as arrays), which gives a second test with a known
answer of zero. Four candidates, fixed before the full run was analysed:

| tracker | GPE | voices unvoiced frames | range error vs truth (mean / mean abs.) | utterances off > 2 st | noise floor, identical contours | median F0 error |
|---|---|---|---|---|---|---|
| pYIN, 50–500 Hz *(first version)* | 1.19% | 6.6% | +0.67 / 1.06 | 18 / 360 | 0.53 st | 0.38 st |
| **pYIN, two-pass** | **0.54%** | **1.9%** | **+0.04 / 0.59** | **8 / 360** | **0.12 st** | 0.42 st |
| Harvest, 50–500 Hz | 7.92% | 38.9% | +3.82 / 3.83 | 121 / 360 | 0.65 st | 0.24 st |
| Harvest, two-pass | 5.71% | 35.0% | +1.38 / 1.46 | 17 / 360 | 0.43 st | 0.24 st |

GPE: frames voiced in both whose F0 is off by more than 20%. Two-pass: a first pass at
50–500 Hz, then a search between 0.75 × Q1 and 1.5 × Q3 of what it found.

**The harness now uses two-pass pYIN.** Its range is unbiased against truth to 0.04 st,
it rarely invents pitch, and its noise floor is a quarter of the first version's.
Harvest voices over a third of the frames the vocoder leaves silent, which inflates its
ranges; it stays as the cross-check, the way HuBERT stays beside Whisper. Harvest is
better on median F0, by 0.2 st — small against the 2.6 st between two recordings of one
speaker.

**What the check cannot cover: real speech, which has no truth contour.** There, 3.9%
of voiced frames sit at the adaptive floor — creak the tracker cannot follow — and the
two tracker families agree on a *single* real utterance's range only loosely
(r = +0.32), while agreeing on the mean over 60 within 0.55 st (9.52 against 8.97). Both
separate two recordings of one speaker from recordings of two speakers by pitch gap
(P = 0.85 and 0.93). Read the prosody columns as means, which is how the table reports
them.

## What diffusion does to prosody

| arm | **true range** (model's contour) | true vs off | pYIN range | pYIN vs the speakers' own reading | pitch gap | words / s |
|---|---|---|---|---|---|---|
| ground truth (real) | — | | **9.52** | | 2.65 | 2.85 |
| **diffusion off** | 7.72 | *control* | 7.90 | −1.63 (35 lower / 25 higher, p = 0.25) | 2.26 | 2.64 |
| off, seed + 1 | 7.72 | identical | 7.94 | −1.58 (35 / 25, p = 0.25) | 2.23 | 2.64 |
| **default** | 6.60 | **−1.13** (48 / 12, p = 3e-6) | 6.69 | **−2.84** (41 / 19, p = 0.006) | 2.30 | 2.74 |
| default, seed + 1 | 6.12 | **−1.60** (52 / 8, p = 5e-9) | 5.92 | **−3.60** (49 / 11, p = 8e-7) | 2.04 | 2.74 |
| **full** | 6.28 | **−1.45** (46 / 14, p = 4e-5) | 6.37 | **−3.16** (44 / 16, p = 4e-4) | 2.55 | 2.77 |
| full, seed + 1 | 5.70 | **−2.03** (50 / 10, p = 2e-7) | 5.53 | **−3.99** (53 / 7, p = 8e-10) | 2.03 | 2.86 |

Ranges in semitones; lower / higher count utterances; sign tests. The pYIN column sits
within ±0.22 st of the true range on every arm, so the differences between arms survive
measurement.

**Diffusion makes StyleTTS 2 flatter, not livelier.** On the contour the model actually
produced, default diffusion narrows the pitch range by 1.1–1.6 semitones against
diffusion off, on 48–52 of 60 utterances; full diffusion by 1.5–2.0. Against the speakers
themselves, diffusion off is not distinguishable from their own reading (−1.6 st under
pYIN, +1.2 under Harvest, both n.s.), and both diffusion settings are flatter than the
speakers under both trackers (pYIN −2.8 to −4.0, Harvest −1.6 to −2.6, every p < 0.01).
So diffusion moves the prosody *away* from the human row.

**What diffusion adds is variety between seeds.** With diffusion off the contour is
identical on both seeds; with it on, never. Two seeds' renditions of one utterance
differ in pitch level by **1.15 st** (default) and **1.67 st** (full), in length by 2.6%
and 3.7%, and in pitch range by only 0.5–0.7 st. The variety is mostly *where* the voice
sits and how fast it goes, not how much it moves.

**Speaking rate moves toward the real reading** — 2.64 words/s with diffusion off,
2.74–2.86 with it, against 2.85 for the speakers — the 7% slow-down the ablation saw as
duration, now as rate.

**The identity loss at full diffusion is not a register shift.** The full-diffusion arm
on seed + 1 lost 0.031 SIM in the ablation; its pitch gap here is 2.03 st, inside the
human row's 2.65. Whatever moved the voice away from the speaker was timbre — the half
of the style vector `alpha` controls — not pitch.

**The first tracker would have doubled the effect.** Under fixed-range pYIN, default
diffusion read as 2.25 st flatter than off (`diffusion_prosody_first_tracker_xeon.txt`);
the truth is 1.13. Fixed-range pYIN misreads diffusion-off audio more often (GPE 1.7%
there against 0.9% on default), which inflated the control. The direction survived; the
size did not.

## What this does not establish

**Flatter is not the same as less natural.** A p10–p90 range is one statistic of
intonation; no listener heard these clips, and nothing here measures whether the
intonation fits the sentence.

**Why diffusion flattens is not tested.** Two seeds, one model, read speech, n = 60.

**The comparison with the speakers rests on a tracker.** For synthetic audio the truth
contour settles the arm differences; for real speech it does not exist, and the two
trackers disagree on the sign of diffusion-off's difference from the speakers. Only the
diffusion settings' difference from the speakers holds under both.
