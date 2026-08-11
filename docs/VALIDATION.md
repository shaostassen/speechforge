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
