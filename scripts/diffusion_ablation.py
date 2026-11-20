#!/usr/bin/env python3
"""What does the diffusion stage of a diffusion TTS actually contribute?

StyleTTS 2's diffusion model samples a style vector, which upstream then blends
with the reference clip's own style: ``alpha`` weights the acoustic half (timbre),
``beta`` the prosodic half. At ``alpha = beta = 0`` the sample is discarded — the
same network runs with the diffusion switched off. That makes a clean ablation
possible on one backend, one corpus and one ASR, which no amount of comparing
different models can give.

Arms, all on the same utterances and references:

  **no diffusion** (0 / 0) — the control.

  **default** (0.3 / 0.7, 5 steps) — upstream's LibriTTS demo values.

  **full diffusion** (1 / 1) — style entirely from the sampler.

  **10 steps** — does more ODE integration change anything at the default blend?

  **seed + 1** for all three blends — the spread a single seed hides. A
  difference between arms smaller than the difference between two seeds of the
  same arm is not a finding. The control's seed+1 arm is the noise floor: the
  diffused style is discarded there, so what moves is the HiFi-GAN decoder's
  own random excitation (SineGen phases and noise). The first draft of this
  script asserted the control was seed-invariant; it is not, and checking
  found it.

Why same-seed arms pair cleanly: alpha and beta are applied *after* sampling,
so at one seed and step count every blend draws the identical diffusion noise
and then the identical decoder noise. Only the style differs. The 10-step arm
draws more noise (ADPM2 is ancestral), so it is not paired in that sense.

  **10 s reference cap** — the figure borrowed from CosyVoice's docs, against
  this backend's 30 s. Only references longer than 10 s can differ, so the
  comparison is reported on that subset as well as overall.

Every arm is scored paired against the control, per utterance, with a sign test;
rows are written before anything is summarized.

    python scripts/diffusion_ablation.py --per-utterance rows.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import date
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import sim_o, whisper_transcriber  # noqa: E402
from speechforge.synth import clone  # noqa: E402
from speechforge.text import apply_case  # noqa: E402

NO_DIFF = {"alpha": 0.0, "beta": 0.0}
FULL = {"alpha": 1.0, "beta": 1.0}

# (id, display name, settings, seed offset, reference cap or None for the
# backend's own). Ids are what --arms takes; names may contain commas.
ARMS = [
    ("nodiff", "no diffusion", NO_DIFF, 0, None),
    ("default", "default", {}, 0, None),
    ("full", "full diffusion", FULL, 0, None),
    ("steps10", "default, 10 steps", {"diffusion_steps": 10}, 0, None),
    ("nodiff_s1", "no diffusion, seed+1", NO_DIFF, 1, None),
    ("default_s1", "default, seed+1", {}, 1, None),
    ("full_s1", "full diffusion, seed+1", FULL, 1, None),
    ("cap10", "default, 10 s ref cap", {}, 0, 10.0),
]
CONTROL = "no diffusion"


def hardware_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def sign_test(pos: int, neg: int) -> float:
    """Two-sided exact binomial sign test; ties are dropped, as is standard."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def corpus_wer(rows, hyp_key: str) -> float:
    from speechlens.metrics import wer
    return wer(" ".join(r["text"] for r in rows),
               " ".join(r[hyp_key] for r in rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="librispeech",
                   choices=["edacc", "librispeech", "librispeech_dummy"])
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--text-case", choices=["raw", "sentence"], default="sentence")
    p.add_argument("--arms", default=None,
                   help="comma-separated arm ids (default: all): "
                        + ", ".join(a[0] for a in ARMS))
    p.add_argument("--per-utterance", metavar="PATH", required=True,
                   help="rows are written here before any summary")
    args = p.parse_args()

    print(f"study=diffusion_ablation  hardware={hardware_label()}  "
          f"backend=styletts2  corpus={args.corpus}  asr={args.asr_model}  "
          f"seed={args.seed}  text_case={args.text_case}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    from speechforge.backends.styletts2 import StyleTTS2Backend
    from speechforge.similarity import load_encoder
    from speechlens.metrics import wer

    arms = [a[1:] for a in ARMS]
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",")]
        unknown = set(wanted) - {a[0] for a in ARMS}
        if unknown:
            raise SystemExit(f"unknown arms {sorted(unknown)}")
        if "nodiff" not in wanted:
            raise SystemExit("the 'nodiff' arm is the control; keep it")
        arms = [a[1:] for a in ARMS if a[0] in wanted]

    speakers = corpora.load(args.corpus, args.speakers, args.utterances)
    print(f"speakers: {list(speakers)}")
    for spk, utts in speakers.items():
        print(f"  {spk}: reference {len(utts[0][0]) / utts[0][1]:.1f}s")
    print()

    encoder = load_encoder()     # no fallback: a blank SIM column is not a result
    print(f"speaker encoder: {encoder.describe()}")
    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")

    # Real audio, scored once: the ground-truth row.
    items = []
    for spk, utts in speakers.items():
        ref_audio, ref_sr, _ = utts[0]
        held = utts[1:]
        for j, (ho, ho_sr, text) in enumerate(held):
            other = held[(j + 1) % len(held)]
            items.append({"speaker": spk, "j": j, "text": text,
                          "ref_s": len(ref_audio) / ref_sr,
                          "real_audio_s": len(ho) / ho_sr,
                          "hyp_real": transcribe(ho, ho_sr),
                          "sim_real": sim_o(encoder, other[0], other[1], ho, ho_sr)})
    print(f"ground truth scored: {len(items)} utterances\n", flush=True)

    import gc

    rows = {}
    for name, settings, seed_off, cap in arms:
        backend = StyleTTS2Backend(device="cpu", **settings)
        backend.set_threads(args.threads)
        print(f"[{name}] {backend.describe()}  ref_cap={cap or backend.ref_seconds}s",
              flush=True)
        arm_rows = []
        k = 0
        for spk, utts in speakers.items():
            ref_audio, ref_sr, _ = utts[0]
            for j, (ho, ho_sr, text) in enumerate(utts[1:]):
                wav, sr, report = clone(backend, apply_case(text, args.text_case),
                                        ref_audio, ref_sr,
                                        seed=args.seed + seed_off, ref_seconds=cap)
                info = backend.last_synthesis
                arm_rows.append({
                    **items[k], "arm": name,
                    "hyp": transcribe(wav, sr),
                    "sim": sim_o(encoder, wav, sr, ho, ho_sr,
                                 reference_audio=ref_audio),
                    "audio_s": len(wav) / sr,
                    "dur_ratio": (len(wav) / sr) / items[k]["real_audio_s"],
                    "ref_used_s": report["duration_s"],
                    "chunks": info["chunks"],
                    "dropped": info["dropped_symbols"]})
                k += 1
        rows[name] = arm_rows
        # One model in memory at a time: this process also holds Whisper
        # large-v3 and WavLM, and the box it was written on has 16 GB.
        del backend
        gc.collect()

    out_rows = []
    for name, rs in rows.items():
        for r in rs:
            out_rows.append({**r, "wer": wer(r["text"], r["hyp"]),
                             "wer_real": wer(r["text"], r["hyp_real"])})
    Path(args.per_utterance).write_text(json.dumps(out_rows, indent=1))
    print(f"\nper-utterance rows -> {args.per_utterance}")

    rows = {name: [] for name in rows}       # rebuilt in order, now with WER
    for r in out_rows:
        rows[r["arm"]].append(r)
    ctrl = rows[CONTROL]
    n = len(ctrl)

    print("\n" + "=" * 92)
    print(f"DIFFUSION ABLATION — styletts2 on {args.corpus}, n={n}, "
          f"{args.text_case} case, ASR {args.asr_model}")
    print("=" * 92)
    print(f"{'arm':>24} {'WER':>7} {'SIM-o':>7} {'dur':>6} {'max dur':>8}"
          f"   {'vs control: dWER':>17} {'+/-':>7} {'p':>7}   {'dSIM':>7} {'+/-':>7} {'p':>7}")
    gt_wer = corpus_wer(ctrl, "hyp_real")
    gt_sim = float(np.mean([r["sim_real"] for r in ctrl]))
    print(f"{'ground truth':>24} {gt_wer:>7.3f} {gt_sim:>7.3f} {1.0:>6.2f} {'':>8}"
          f"   <- read every row against this")
    for name, rs in rows.items():
        w = corpus_wer(rs, "hyp")
        s = float(np.mean([r["sim"] for r in rs]))
        dur = [r["dur_ratio"] for r in rs]
        line = (f"{name:>24} {w:>7.3f} {s:>7.3f} {np.mean(dur):>6.2f} "
                f"{max(dur):>8.2f}")
        if name != CONTROL:
            dw = [a["wer"] - c["wer"] for a, c in zip(rs, ctrl)]
            ds = [a["sim"] - c["sim"] for a, c in zip(rs, ctrl)]
            wb, ww = sum(d < 0 for d in dw), sum(d > 0 for d in dw)
            sh, sl = sum(d > 0 for d in ds), sum(d < 0 for d in ds)
            line += (f"   {w - corpus_wer(ctrl, 'hyp'):>+17.3f} {f'{wb}/{ww}':>7} "
                     f"{sign_test(wb, ww):>7.3g}   {s - float(np.mean([c['sim'] for c in ctrl])):>+7.3f} "
                     f"{f'{sh}/{sl}':>7} {sign_test(sh, sl):>7.3g}")
        print(line)
    print("  +/- for WER: utterances better/worse than control; for SIM: higher/lower.")
    print("  dur = synthesized / real duration; a non-autoregressive model cannot run on.")

    def seed_gap(a, b):
        if a in rows and b in rows:
            ra, rb = rows[a], rows[b]
            flips = sum(1 for x, y in zip(ra, rb) if x["wer"] != y["wer"])
            print(f"seed spread, {a!r} vs {b!r}: WER "
                  f"{corpus_wer(ra, 'hyp'):.3f} vs {corpus_wer(rb, 'hyp'):.3f}, "
                  f"SIM {np.mean([r['sim'] for r in ra]):.3f} vs "
                  f"{np.mean([r['sim'] for r in rb]):.3f}; "
                  f"{flips}/{len(ra)} utterances change WER")
    print()
    seed_gap("no diffusion", "no diffusion, seed+1")
    seed_gap("default", "default, seed+1")
    seed_gap("full diffusion", "full diffusion, seed+1")

    cap_arm = "default, 10 s ref cap"
    if cap_arm in rows:
        idx = [i for i, r in enumerate(rows["default"]) if r["ref_s"] > 10.0]
        sub = lambda name: [rows[name][i] for i in idx]  # noqa: E731
        if idx:
            print(f"reference cap on the {len(idx)} utterances whose reference "
                  f"exceeds 10 s: WER {corpus_wer(sub('default'), 'hyp'):.3f} "
                  f"(30 s) vs {corpus_wer(sub(cap_arm), 'hyp'):.3f} (10 s); SIM "
                  f"{np.mean([r['sim'] for r in sub('default')]):.3f} vs "
                  f"{np.mean([r['sim'] for r in sub(cap_arm)]):.3f}")

    dropped = sorted({ch for r in out_rows for ch in r["dropped"]})
    chunked = sum(1 for r in ctrl if r["chunks"] > 1)
    print(f"inputs split for PL-BERT's 512-token limit: {chunked}/{n}; "
          f"symbols the model could not read: {dropped or 'none'}")


if __name__ == "__main__":
    main()
