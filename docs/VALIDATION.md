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

**Follow-ups**
- Ask Resemble AI how Nano/Turbo are meant to be loaded; re-measure if an entrypoint
  appears. If Nano hits ~0.4 RTF as its parameter count suggests it might, it becomes
  the primary backend.
- Measure the M2 via MLX. Unified memory and Metal make it a genuinely different
  proposition from "CPU", and it is the natural place for interactive work.
