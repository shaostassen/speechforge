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
