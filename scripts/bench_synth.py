#!/usr/bin/env python3
"""Synthesis throughput per backend, on whatever machine you run it on.

Measure this before believing any vendor RTF claim. The first run of this script
found that the installable Chatterbox is ~2x *slower* than realtime on a Ryzen 9
9950X, against a claimed 3x faster — see docs/VALIDATION.md.

Method, matching the ASR project so numbers are comparable:
  - model load excluded (amortized across a session)
  - one warm-up pass discarded (lazy init, kernel autotune, page faults)
  - several sentences of differing length, since RTF varies with length
  - thread count swept, because more threads is not monotonically better:
    16 beat 32 on the 9950X

    python scripts/bench_synth.py --backend chatterbox --ref ref.wav
"""
from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Short, medium, and a deliberately awkward one. Tongue-twisters are where
# autoregressive TTS destabilizes, so a length sweep alone would flatter it.
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Calibration matters more than raw accuracy when the output is trusted.",
    "She sells sea shells by the sea shore, and the shells she sells are "
    "surely seashells from the shore.",
]


def hardware_label() -> str:
    """One-line hardware string for the validation record."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            cpu = out.stdout.strip()
        elif platform.system() == "Linux":
            cpu = ""
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        else:
            cpu = ""
    except Exception:
        cpu = ""
    cpu = cpu or platform.processor() or platform.machine()
    return f"{cpu} ({platform.system()} {platform.machine()})"


def banner(study: str, backend: str, extra: str = "") -> None:
    print(f"study={study}  hardware={hardware_label()}  backend={backend}  "
          f"date={date.today().isoformat()}{extra}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="chatterbox")
    p.add_argument("--ref", required=True, help="reference audio for cloning")
    p.add_argument("--threads", default="8,16,32",
                   help="comma-separated thread counts to sweep")
    args = p.parse_args()

    from speechforge.synth import load_backend

    banner("bench_synth", args.backend)

    t0 = time.perf_counter()
    backend = load_backend(args.backend, device="cpu")
    print(f"load={time.perf_counter() - t0:.1f}s  "
          f"output_sr={backend.sample_rate}  params={backend.describe()}")

    print("\nwarm-up (discarded)...")
    backend.synthesize(SENTENCES[0], args.ref)

    print(f"\n{'threads':>8} {'sent':>5} {'audio_s':>8} {'wall_s':>7} "
          f"{'RTF':>6} {'x realtime':>11}")
    for nt in [int(t) for t in args.threads.split(",")]:
        backend.set_threads(nt)
        for i, text in enumerate(SENTENCES):
            t0 = time.perf_counter()
            wav, sr = backend.synthesize(text, args.ref)
            wall = time.perf_counter() - t0
            audio_s = len(wav) / sr
            rtf = wall / audio_s
            print(f"{nt:>8} {i:>5} {audio_s:>8.2f} {wall:>7.2f} "
                  f"{rtf:>6.3f} {1 / rtf:>11.2f}")

    print("\nRTF < 1.0 is faster than realtime. Thread count is swept because "
          "it is not\nmonotonic — 16 beat 32 on the 9950X.")


if __name__ == "__main__":
    main()
