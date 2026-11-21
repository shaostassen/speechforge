#!/usr/bin/env python3
"""Which pitch tracker can the prosody columns trust? Measured, not assumed.

For synthetic audio the true pitch is known. StyleTTS 2's vocoder drives its
harmonic source with the F0 contour its predictor outputs, and
scripts/diffusion_prosody.py saves that contour beside every clip it
synthesizes (`.f0.npy`, 12.5 ms frames). So each tracker is scored against
truth, frame by frame, with the standard gross pitch error (GPE: frames voiced
in both where the tracker is off by more than 20%), and per utterance, on the
two numbers the harness reports: median F0 and pitch range.

A second test needs no truth contour, only a known zero. With diffusion off
(alpha = beta = 0) the F0 contour is identical on every seed; only the
vocoder's own noise differs. Whatever a tracker reports as the difference
between two seeds' renditions is its own noise.

Both tests were prompted by a 4-utterance smoke run in which pYIN read one
identical-contour pair as 15.98 and 5.70 semitones of range: over a tenth of
one rendition's voiced frames sat exactly at the 50 Hz search floor, an octave
under the speaker, and decoder noise decided which frames fell there.

Neither test involves what diffusion does, so choosing a tracker by them cannot
tilt the diffusion result. Real speech has no truth contour; there the checks
are whether the pitch gap separates two recordings of one speaker from
recordings of two speakers, and agreement between the two tracker families.

Candidates, fixed before the full run was analysed:
  pyin      pYIN, 50-500 Hz (the harness's first version)
  pyin2     pYIN, speaker-adaptive two-pass range: a first pass at 50-500 Hz,
            then 0.75 x Q1 to 1.5 x Q3 of that pass's voiced F0
  harvest   WORLD's Harvest, 50-500 Hz
  harvest2  Harvest, the same two-pass range

    python scripts/prosody_checks.py --audio-dir runs/prosody/audio --workers 4
"""
from __future__ import annotations

import argparse
import platform
import shlex
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechforge import corpora  # noqa: E402
from speechforge.prosody import (FMAX, FMIN, HOP, TARGET_SR, _mono16k,  # noqa: E402
                                 pitch_gap_st, semitones, summarize_f0)
from speechforge.reference import trim_silence  # noqa: E402

ARMS = ["off", "off_s1", "default", "default_s1", "full", "full_s1"]
TRUTH_HOP_S = 300 / 24000          # StyleTTS 2's F0 frame
TRACK_HOP_S = HOP / TARGET_SR
GROSS = 0.20                       # the standard GPE threshold


def hardware_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def _pyin(y, lo, hi):
    import librosa
    f0, v, _ = librosa.pyin(y, fmin=lo, fmax=hi, sr=TARGET_SR,
                            frame_length=1024, hop_length=HOP)
    return np.where(v, f0, np.nan)


def _harvest(y, lo, hi):
    import pyworld
    f0, _ = pyworld.harvest(np.asarray(y, np.float64), TARGET_SR, f0_floor=lo,
                            f0_ceil=hi, frame_period=1000.0 * TRACK_HOP_S)
    return np.where(f0 > 0, f0, np.nan)


def two_pass(track):
    """A speaker-adaptive search range: 0.75 x Q1 to 1.5 x Q3 of a first pass."""
    def run(y):
        first = track(y, FMIN, FMAX)
        v = first[np.isfinite(first)]
        if v.size < 20:
            return first, FMIN
        lo = max(FMIN, 0.75 * np.percentile(v, 25))
        hi = min(FMAX * 1.5, 1.5 * np.percentile(v, 75))
        return track(y, lo, hi), lo
    return run


CANDIDATES = {
    "pyin": lambda y: (_pyin(y, FMIN, FMAX), FMIN),
    "pyin2": two_pass(_pyin),
    "harvest": lambda y: (_harvest(y, FMIN, FMAX), FMIN),
    "harvest2": two_pass(_harvest),
}


def evaluate_clip(job):
    """Every candidate on one clip: the harness's summary, and frame scores
    against the truth contour when there is one."""
    key, audio, sr, truth = job
    y = _mono16k(audio, sr)
    span, lead_s, _tail = trim_silence(y, TARGET_SR)
    out = {"key": key}
    t_sum = None
    if truth is not None:
        t_sum = summarize_f0(np.where(truth >= FMIN, truth, np.nan))
        out["truth"] = {"median": t_sum[0], "range": t_sum[1]}
    for name, fn in CANDIDATES.items():
        f0, floor = fn(span)
        med, rng = summarize_f0(f0)
        v = f0[np.isfinite(f0)]
        res = {"median": med, "range": rng,
               "pinned": float(np.mean(semitones(v, floor) < 0.5)) if v.size else float("nan")}
        if truth is not None:
            # Tracker frame t sits at lead_s + t * 10 ms in the clip; look up the
            # truth frame covering that instant.
            idx = ((lead_s + np.arange(len(f0)) * TRACK_HOP_S) / TRUTH_HOP_S).astype(int)
            ok = idx < len(truth)
            tr, ft = f0[ok], truth[idx[ok]]
            t_voiced = ft >= FMIN
            t_unvoiced = ft <= 10.0        # the vocoder's own voicing rule
            k_voiced = np.isfinite(tr)
            both = t_voiced & k_voiced
            err = np.abs(tr[both] / ft[both] - 1.0)
            fine = np.abs(semitones(tr[both], ft[both]))[err <= GROSS]
            res.update({"n_both": int(both.sum()), "n_gross": int((err > GROSS).sum()),
                        "n_tv": int(t_voiced.sum()), "n_tv_kv": int(both.sum()),
                        "n_tu": int(t_unvoiced.sum()),
                        "n_tu_kv": int((t_unvoiced & k_voiced).sum()),
                        "fine_sum": float(fine.sum()), "fine_n": int(fine.size)})
        out[name] = res
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True,
                   help="diffusion_prosody.py's audio directory")
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    import librosa
    import pyworld
    import soundfile as sf
    print(f"study=prosody_checks:tracker  hardware={hardware_label()}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}")
    print(f"librosa {librosa.__version__}, pyworld {getattr(pyworld, '__version__', '?')}\n")

    speakers = corpora.load("librispeech", args.speakers, args.utterances)
    spk_ids = list(speakers)
    items = [(spk, j, a, s) for spk, utts in speakers.items()
             for j, (a, s, _t) in enumerate(utts[1:])]
    n = len(items)
    index = {(spk, j): k for k, (spk, j, _a, _s) in enumerate(items)}
    audio_dir = Path(args.audio_dir)
    arms = [a for a in ARMS if (audio_dir / a).is_dir()]

    jobs = [(("real", k), a, s, None) for k, (_spk, _j, a, s) in enumerate(items)]
    for arm in arms:
        for k, (spk, j, _a, _s) in enumerate(items):
            pth = audio_dir / arm / f"{k:02d}_{spk}_{j}.wav"
            wav, sr = sf.read(str(pth), dtype="float32")
            jobs.append(((arm, k), wav, sr, np.load(str(pth.with_suffix(".f0.npy")))))
    print(f"{len(jobs)} clips: {n} real, {len(jobs) - n} synthesized "
          f"({', '.join(arms)}); {args.workers} workers\n", flush=True)

    if args.workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(args.workers) as pool:
            done = pool.map(evaluate_clip, jobs, chunksize=4)
    else:
        done = [evaluate_clip(j) for j in jobs]
    res = {d["key"]: d for d in done}
    real = [res[("real", k)] for k in range(n)]
    synth = [res[(arm, k)] for arm in arms for k in range(n)]

    # --- against the truth contour -----------------------------------------------
    print("=" * 96)
    print(f"AGAINST TRUTH — synthesized clips, the model's own F0 contour (n={len(synth)})")
    print("=" * 96)
    print(f"{'tracker':>9} {'GPE':>7} {'fine':>6} {'recall':>7} {'false-v':>8} "
          f"{'|d med|':>8} {'d range':>8} {'|d range|':>9} {'>2 st':>6}   GPE by arm")
    for name in CANDIDATES:
        rows = [r[name] for r in synth]
        tot = lambda f: sum(r[f] for r in rows)  # noqa: E731
        gpe = tot("n_gross") / max(1, tot("n_both"))
        dmed = [pitch_gap_st(r[name]["median"], r["truth"]["median"]) for r in synth]
        drng = np.array([r[name]["range"] - r["truth"]["range"] for r in synth])
        drng = drng[np.isfinite(drng)]
        by_arm = []
        for arm in arms:
            ar = [res[(arm, k)][name] for k in range(n)]
            by_arm.append(f"{arm} {100 * sum(r['n_gross'] for r in ar) / max(1, sum(r['n_both'] for r in ar)):.1f}")
        print(f"{name:>9} {100 * gpe:>6.2f}% {tot('fine_sum') / max(1, tot('fine_n')):>6.2f} "
              f"{100 * tot('n_tv_kv') / max(1, tot('n_tv')):>6.1f}% "
              f"{100 * tot('n_tu_kv') / max(1, tot('n_tu')):>7.1f}% "
              f"{np.nanmean(dmed):>8.2f} {drng.mean():>+8.2f} {np.abs(drng).mean():>9.2f} "
              f"{int((np.abs(drng) > 2).sum()):>6}   {'  '.join(by_arm)}", flush=True)
    t_range = np.nanmean([r["truth"]["range"] for r in synth])
    print(f"\n  truth: median F0 and p10-p90 range of the predicted contour over frames "
          f">= {FMIN:.0f} Hz; mean range {t_range:.2f} st.")
    print(f"  GPE: frames voiced in both with error > {100 * GROSS:.0f}%. fine: mean |st| "
          f"error on the rest. recall: truth-voiced")
    print("  frames the tracker voices. false-v: frames the vocoder leaves unvoiced "
          "(F0 <= 10 Hz) that it voices.")
    print("  d med / d range: per utterance, tracker minus truth, semitones; >2 st: "
          "utterances whose range is off by more.")

    # --- identical-contour pairs ---------------------------------------------------
    if "off" in arms and "off_s1" in arms:
        print("\n" + "=" * 96)
        print(f"NOISE FLOOR — diffusion off, two seeds: identical F0 contour (n={n} pairs)")
        print("=" * 96)
        for name in CANDIDATES:
            a = [res[("off", k)][name] for k in range(n)]
            b = [res[("off_s1", k)][name] for k in range(n)]
            dr = np.array([abs(x["range"] - y["range"]) for x, y in zip(a, b)])
            dm = np.array([pitch_gap_st(x["median"], y["median"]) for x, y in zip(a, b)])
            dr, dm = dr[np.isfinite(dr)], dm[np.isfinite(dm)]
            pinned = np.nanmean([x["pinned"] for x in a + b])
            print(f"{name:>9}  |d range| mean {dr.mean():.2f} st, 90th pct "
                  f"{np.percentile(dr, 90):.2f}, > 2 st on {int((dr > 2).sum())}/{dr.size};"
                  f"  |d median| {dm.mean():.2f} st;  pinned at floor {100 * pinned:.2f}%")

    # --- real speech -------------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"REAL SPEECH — no truth contour (n={n})")
    print("=" * 96)
    for name in CANDIDATES:
        same, diff = [], []
        for k, (spk, j, _a, _s) in enumerate(items):
            m = len(speakers[spk]) - 1
            q = index.get((spk, (j + 1) % m)) if m > 1 else None
            nxt = spk_ids[(spk_ids.index(spk) + 1) % len(spk_ids)]
            r = index.get((nxt, j))
            if q is not None:
                same.append(pitch_gap_st(real[k][name]["median"], real[q][name]["median"]))
            if r is not None:
                diff.append(pitch_gap_st(real[k][name]["median"], real[r][name]["median"]))
        same, diff = np.array(same), np.array(diff)
        sep = float(np.mean([a < b for a in same for b in diff]))
        other = ("harvest" if name.startswith("pyin") else "pyin") + \
            ("2" if name.endswith("2") else "")
        xm = np.array([pitch_gap_st(real[k][name]["median"], real[k][other]["median"])
                       for k in range(n)])
        xr = np.array([real[k][name]["range"] for k in range(n)])
        yr = np.array([real[k][other]["range"] for k in range(n)])
        ok = np.isfinite(xr) & np.isfinite(yr)
        print(f"{name:>9}  range {np.nanmean(xr):.2f} st;  pinned {100 * np.nanmean([r[name]['pinned'] for r in real]):.2f}%;"
              f"  pitch gap: same speaker {np.nanmean(same):.2f}, different {np.nanmean(diff):.2f}, "
              f"P(same<diff) {sep:.3f};  vs {other}: |d median| {np.nanmedian(xm):.2f} st, "
              f"range corr {np.corrcoef(xr[ok], yr[ok])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
