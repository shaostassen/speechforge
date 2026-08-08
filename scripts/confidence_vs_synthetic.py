#!/usr/bin/env python3
"""Phase 2: does ASR confidence react to synthetic speech, holding correctness fixed?

For each held-out utterance we have two recordings of the *same text* — the real
one and a clone of it — so the text, the speaker identity and the ASR are all
held constant and only provenance varies. Restricting to words the ASR
transcribed **correctly** removes accuracy as an explanation for any difference.

What comes out:

  auroc_real_higher   P(a random real word scores above a random synthetic one).
                      0.5 = confidence says nothing about provenance.
  delta               mean confidence, real minus synthetic.
  accept rates        what a speechlens reliability policy fitted on real speech
                      does to synthetic words — the deployment consequence.

Also the reason the evaluation harness needs this: if an ASR is systematically
*more* confident on synthetic speech, WER-based intelligibility flatters it, and
this measures by how much.

    python scripts/confidence_vs_synthetic.py --speakers 4 --utterances 3
"""
from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.detect import (calibrated_contrast,  # noqa: E402
                                confidence_contrast, label_words,
                                words_from_result)
from speechforge.synth import clone, load_backend  # noqa: E402


def hardware_label() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip()
    except Exception:
        pass
    return platform.machine()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="chatterbox")
    p.add_argument("--corpus", default="edacc",
                   choices=["edacc", "librispeech", "librispeech_dummy"])
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--speakers", type=int, default=4)
    p.add_argument("--utterances", type=int, default=3)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234,
                   help="seed each synthesis. Chatterbox samples at "
                        "temperature 0.8, so unseeded runs are not "
                        "reproducible: two identical runs of this study gave "
                        "AUROC 0.591 and 0.553. Pass -1 to sample freely and "
                        "measure the spread instead")
    p.add_argument("--policy", default="clean",
                   help="speechlens bundled reliability policy, or '' to skip")
    args = p.parse_args()

    print(f"study=confidence_vs_synthetic  hardware={hardware_label()}  "
          f"backend={args.backend}  asr={args.asr_model}  "
          f"seed={args.seed}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    from speechlens.asr import RobustnessConfig
    from speechlens.pipeline import SpeechLens

    speakers = corpora.load(args.corpus, args.speakers, args.utterances)
    print(f"corpus={args.corpus}  speakers={len(speakers)}  "
          f"{args.utterances} held-out each\n")

    backend = load_backend(args.backend, device="cpu")
    backend.set_threads(args.threads)

    lens = SpeechLens(model_size=args.asr_model, device="cpu",
                      compute_type="int8")
    # Word timestamps are the whole point: per-word probability is the signal.
    cfg = RobustnessConfig(word_timestamps=True)

    policy = None
    if args.policy:
        from speechlens.calibration import load_bundled_policy
        policy = load_bundled_policy(args.policy)
        print(f"reliability policy: {args.policy} "
              f"(threshold {policy.threshold:.3f}, "
              f"target risk {policy.target_risk:.0%})\n")

    samples = []
    for spk, utts in speakers.items():
        ref_audio, ref_sr, _ = utts[0]
        for ho_audio, ho_sr, text in utts[1:]:
            wav, sr, _report = clone(backend, text, ref_audio, ref_sr,
                                       seed=None if args.seed < 0 else args.seed)

            real = lens.analyze((ho_audio, ho_sr), language="en", cfg=cfg)
            synth = lens.analyze((wav, sr), language="en", cfg=cfg)

            samples += label_words(words_from_result(real.to_dict()), text,
                                   synthetic=False)
            samples += label_words(words_from_result(synth.to_dict()), text,
                                   synthetic=True)
        print(f"  {spk}: {len(samples)} labelled words so far")

    out = confidence_contrast(samples)
    print("\n" + "=" * 72)
    print("CONFIDENCE ON CORRECTLY-TRANSCRIBED WORDS — real vs synthetic")
    print("=" * 72)
    print(f"correct words     real={out['n_real']:.0f}  "
          f"synthetic={out['n_synth']:.0f}")
    print(f"mean confidence   real={out['mean_real']:.4f}  "
          f"synthetic={out['mean_synth']:.4f}  delta={out['delta']:+.4f}")
    print(f"AUROC (real above synthetic) = {out['auroc_real_higher']:.3f}")

    print("\nconfidence percentiles (the low tail is where they differ)")
    print(f"{'':>6} {'p1':>8} {'p5':>8} {'p10':>8} {'p25':>8} {'p50':>8}")
    for kind in ("real", "synth"):
        row = "  ".join(f"{out[f'p{q}_{kind}']:>6.3f}" for q in (1, 5, 10, 25, 50))
        print(f"{kind:>6}   {row}")

    a = out["auroc_real_higher"]
    if a != a:
        print("\nreading: undetermined — one side had no correct words")
    else:
        # Deliberately not a single verdict. The ranking statistic and the
        # threshold statistic answer different questions and can point opposite
        # ways: a distribution can rank higher overall while still having more
        # mass below a fixed threshold. An earlier version of this script
        # asserted one conclusion from AUROC alone and contradicted its own
        # accept-rate output.
        print("\nreading:")
        if abs(a - 0.5) < 0.06:
            print("  ranking  — confidence carries essentially no provenance "
                  "information")
        elif a > 0.5:
            print("  ranking  — real words tend to score above synthetic ones")
        else:
            print("  ranking  — synthetic words tend to score above real ones")
        print(f"  centre   — means differ by {out['delta']:+.4f}"
              f"{'  (negligible)' if abs(out['delta']) < 0.01 else ''}")
        lo_r, lo_s = out["p5_real"], out["p5_synth"]
        if abs(lo_r - lo_s) > 0.02:
            which = "real" if lo_r < lo_s else "synthetic"
            print(f"  tail     — {which} speech has the heavier low tail "
                  f"(p5 {lo_r:.3f} vs {lo_s:.3f})")
        else:
            print(f"  tail     — low tails are comparable "
                  f"(p5 {lo_r:.3f} vs {lo_s:.3f})")
        print("\n  These can disagree, and if they do the tail explains it: a "
              "threshold\n  gate cares only about mass below the threshold, "
              "AUROC about the whole order.")

    cal = calibrated_contrast(samples, policy)
    if cal:
        print(f"\nwith the '{args.policy}' policy (fitted on real speech):")
        print(f"  accepted real      {cal['accept_real']:.1%}")
        print(f"  accepted synthetic {cal['accept_synth']:.1%}")
        gap = cal["accept_synth"] - cal["accept_real"]
        print(f"  gap {gap:+.1%} — a reliability gate tuned on real audio "
              f"behaves\n       {'more' if gap > 0 else 'less'} permissively "
              f"on synthetic audio")

    print("\nCaveat: correctness is judged against the reference text, so a "
          "clone that\nsays the right words in an odd way counts as correct. "
          "That is deliberate —\nthe question is about confidence, not "
          "naturalness.")


if __name__ == "__main__":
    main()
