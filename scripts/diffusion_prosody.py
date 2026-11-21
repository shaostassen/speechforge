#!/usr/bin/env python3
"""What StyleTTS 2's diffusion does to prosody — the axis WER and SIM could not see.

The diffusion ablation (scripts/diffusion_ablation.py) found no setting that
moves WER or SIM-o beyond seed noise, and one effect on duration, and could say
no more because the harness had no prosody measure. This measures the prosody
of the *same audio*: the same arms, seeds and utterances, checked
sample-count by sample-count against that study's per-utterance file
(--check-against). With diffusion on, a clip's length depends on the sampled
style vector, so equal lengths on every clip mean equal samples; across two
CPUs the sample values can still differ at floating-point rounding, which the
length check cannot see. Up to that, these numbers describe the audio whose
WER and SIM are already published.

Per arm, against the ground-truth row and paired per utterance:
  pitch range, pitch gap and words/s (speechforge.prosody), each against the
  speaker's real reading of the same text and against diffusion off.
Per setting, the variety the StyleTTS 2 authors claim for diffusion:
  seed diversity — how far two seeds' renditions of one utterance differ in
  length, pitch range and pitch level.

Three readings of every synthesized clip's pitch: the harness's tracker
(two-pass pYIN), a cross-check from another family (WORLD's Harvest, same
two-pass range rule; pyworld is a dev-only dependency), and the truth — the
F0 contour the model itself drove its vocoder with (below). Real speech has
only the first two. An effect the trackers disagree on is not reported as one,
and for synthetic audio the truth decides. scripts/prosody_checks.py is where
the tracker was chosen; this script reports, per arm, how far it sits from
truth, so a tracker error that differs between arms would show here.

Synthesized audio is written to --audio-dir as 16-bit WAV and reused when
present, so a restart costs only the arm in progress, and the analysis can be
redone without synthesis. Measurements are always taken from the written
files, so fresh and resumed runs measure identical input.

Beside each WAV goes the F0 contour the model itself predicted for it
(`.f0.npy`, one value per 12.5 ms — 300 samples at 24 kHz; the vocoder voices
frames above 10 Hz). The vocoder's harmonic source is driven by exactly that
contour, so for synthetic audio the true pitch is known, and
scripts/prosody_checks.py scores each tracker against it. Capturing it only
reads the predictor's output — the audio is byte-identical with and without
it — and the length check against the published study confirms it again.

    python scripts/diffusion_prosody.py --audio-dir runs/prosody_audio \\
        --check-against docs/validation_runs/per_utterance_diffusion_ablation.json \\
        --per-utterance runs/per_utterance_diffusion_prosody.json
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
from speechforge.prosody import (FMAX, FMIN, adaptive_range, measure,  # noqa: E402
                                 pitch_gap_st, semitones, summarize_f0)
from speechforge.synth import clone  # noqa: E402
from speechforge.text import apply_case  # noqa: E402

NO_DIFF = {"alpha": 0.0, "beta": 0.0}
FULL = {"alpha": 1.0, "beta": 1.0}
# (id, name as in the diffusion ablation, settings, seed offset)
ARMS = [
    ("off", "no diffusion", NO_DIFF, 0),
    ("off_s1", "no diffusion, seed+1", NO_DIFF, 1),
    ("default", "default", {}, 0),
    ("default_s1", "default, seed+1", {}, 1),
    ("full", "full diffusion", FULL, 0),
    ("full_s1", "full diffusion, seed+1", FULL, 1),
]
CONTROL = "off"
SETTINGS = [("no diffusion", "off", "off_s1"), ("default", "default", "default_s1"),
            ("full diffusion", "full", "full_s1")]
TRACKERS = ("pyin", "harvest")


def hardware_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def sign_test(pos: int, neg: int) -> float:
    """Two-sided exact binomial sign test; ties dropped."""
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def harvest_track(y, sr):
    """WORLD's Harvest at the same 10 ms hop and the same two-pass range rule
    as the harness's pYIN, NaN where unvoiced."""
    import pyworld

    def run(lo, hi):
        f0, _t = pyworld.harvest(np.asarray(y, dtype=np.float64), sr,
                                 f0_floor=lo, f0_ceil=hi, frame_period=10.0)
        return np.where(f0 > 0, f0, np.nan)
    first = run(FMIN, FMAX)
    rng = adaptive_range(first)
    return first if rng is None else run(*rng)


def _measure_job(job):
    """One clip: both trackers, and the truth contour for a synthesized one."""
    kind, payload, text = job
    truth, audio_s = None, None
    if kind == "real":
        audio, sr = payload
    else:
        import soundfile as sf
        audio, sr = sf.read(payload, dtype="float32")
        audio_s = len(audio) / sr
        f0 = np.load(str(Path(payload).with_suffix(".f0.npy")))
        truth = summarize_f0(np.where(f0 >= FMIN, f0, np.nan))
    return {**measure_both(audio, sr, text), "audio_s": audio_s, "truth": truth}


def measure_both(audio, sr, text):
    return {"pyin": measure(audio, sr, text),
            "harvest": measure(audio, sr, text, track=harvest_track)}


def finite(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=np.float64)
    return a[np.isfinite(a)]


def mean(xs) -> float:
    a = finite(xs)
    return float(a.mean()) if a.size else float("nan")


def paired(a, b):
    """``(mean a - b, n a>b, n a<b, sign p)`` over pairs where both are finite."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    d = d[np.isfinite(d)]
    up, down = int((d > 0).sum()), int((d < 0).sum())
    return float(d.mean()) if d.size else float("nan"), up, down, sign_test(up, down)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--per-utterance", required=True, metavar="PATH")
    p.add_argument("--check-against", metavar="PATH",
                   help="per-utterance JSON of diffusion_ablation.py; every "
                        "synthesized length must match it exactly")
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=4,
                   help="processes for measurement (synthesis is sequential)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--text-case", choices=["raw", "sentence"], default="sentence")
    args = p.parse_args()

    print(f"study=diffusion_prosody  hardware={hardware_label()}  "
          f"backend=styletts2  corpus=librispeech  seed={args.seed}  "
          f"text_case={args.text_case}  date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")
    import librosa
    import pyworld
    print(f"trackers: pYIN two-pass (librosa {librosa.__version__}, the harness's), "
          f"Harvest two-pass (pyworld {getattr(pyworld, '__version__', '?')}, "
          f"cross-check); truth: the model's own F0 contour\n")
    from multiprocessing import get_context
    pool = get_context("fork").Pool(args.workers)

    speakers = corpora.load("librispeech", args.speakers, args.utterances)
    spk_ids = list(speakers)
    print(f"speakers: {spk_ids}")
    items = []
    for spk, utts in speakers.items():
        for j, (a, s, t) in enumerate(utts[1:]):
            items.append({"speaker": spk, "j": j, "text": t, "audio": a, "sr": s})
    n = len(items)
    held = {spk: utts[1:] for spk, utts in speakers.items()}
    print(f"{n} held-out utterances\n", flush=True)

    # --- real recordings: targets, same-speaker and different-speaker pairs --
    print("measuring real recordings...", flush=True)
    real = pool.map(_measure_job, [("real", (it["audio"], it["sr"]), it["text"])
                                   for it in items])
    index = {(it["speaker"], it["j"]): k for k, it in enumerate(items)}

    def other_same(k):          # another recording of the same speaker
        it = items[k]
        m = len(held[it["speaker"]])
        return index[(it["speaker"], (it["j"] + 1) % m)] if m > 1 else None

    def other_speaker(k):       # same slot, the next speaker
        it = items[k]
        nxt = spk_ids[(spk_ids.index(it["speaker"]) + 1) % len(spk_ids)]
        return index.get((nxt, it["j"]))

    # --- synthesis, reused from disk when present -----------------------------
    audio_dir = Path(args.audio_dir)
    synth = {}
    import soundfile as sf
    for arm, name, settings, seed_off in ARMS:
        d = audio_dir / arm
        d.mkdir(parents=True, exist_ok=True)
        paths = [d / f"{k:02d}_{it['speaker']}_{it['j']}.wav" for k, it in enumerate(items)]
        missing = [k for k, pth in enumerate(paths)
                   if not (pth.exists() and pth.with_suffix(".f0.npy").exists())]
        if missing:
            from speechforge.backends.styletts2 import StyleTTS2Backend
            backend = StyleTTS2Backend(device="cpu", **settings)
            backend.set_threads(args.threads)
            print(f"[{arm}] {backend.describe()}: synthesizing {len(missing)}/{n}",
                  flush=True)
            predictor = backend._model.predictor
            original = predictor.F0Ntrain
            captured = []

            def capture(*a, **k):
                f0, nrg = original(*a, **k)
                captured.append(f0.detach().cpu().numpy().ravel().copy())
                return f0, nrg
            predictor.F0Ntrain = capture
            for k in missing:
                it = items[k]
                ref_audio, ref_sr, _ = speakers[it["speaker"]][0]
                captured.clear()
                wav, sr, _rep = clone(backend, apply_case(it["text"], args.text_case),
                                      ref_audio, ref_sr, seed=args.seed + seed_off)
                np.save(str(paths[k].with_suffix(".f0.npy")),
                        np.concatenate(captured).astype(np.float32))
                tmp = paths[k].with_suffix(".tmp.wav")
                sf.write(str(tmp), wav, sr, subtype="PCM_16")
                tmp.replace(paths[k])      # atomic, and last: a WAV means done
            predictor.F0Ntrain = original
            del backend
            import gc
            gc.collect()
        else:
            print(f"[{arm}] all {n} reused from {d}", flush=True)
        synth[arm] = pool.map(_measure_job, [("synth", str(pth), items[k]["text"])
                                             for k, pth in enumerate(paths)])
        print(f"[{arm}] measured", flush=True)

    # --- same audio as the published WER/SIM? ----------------------------------
    if args.check_against:
        pub = json.loads(Path(args.check_against).read_text())
        print()
        for arm, name, _s, _o in ARMS:
            ref = [r for r in pub if r["arm"] == name]
            if len(ref) != n or any(r["text"] != it["text"] for r, it in zip(ref, items)):
                raise SystemExit(f"[{arm}] published rows do not align with this corpus")
            same = sum(abs(r["audio_s"] - s["audio_s"]) < 1e-9
                       for r, s in zip(ref, synth[arm]))
            print(f"[{arm}] synthesized length identical to the published "
                  f"'{name}' arm on {same}/{n} utterances")
            if same != n:
                raise SystemExit(f"[{arm}] not the published audio; refusing to "
                                 f"attach these numbers to that study")

    # --- per-utterance rows, written before anything is summarized -------------
    out = []
    for k, it in enumerate(items):
        row = {"speaker": it["speaker"], "j": it["j"], "words": len(it["text"].split()),
               "real": {t: real[k][t].to_dict() for t in TRACKERS}}
        for arm, *_ in ARMS:
            tm, trg = synth[arm][k]["truth"]
            row[arm] = {"audio_s": synth[arm][k]["audio_s"],
                        "truth": {"median_f0_hz": None if tm != tm else round(tm, 1),
                                  "pitch_range_st": None if trg != trg else round(trg, 3)},
                        **{t: synth[arm][k][t].to_dict() for t in TRACKERS}}
        out.append(row)
    Path(args.per_utterance).write_text(json.dumps(out, indent=1))
    print(f"\nper-utterance rows -> {args.per_utterance}")

    def col(rows, t, field):
        return [getattr(r[t], field) for r in rows]

    def gaps(rows, t, partner):
        res = []
        for k in range(n):
            q = partner(k) if callable(partner) else k
            src = rows[k][t] if rows is not real else real[k][t]
            res.append(float("nan") if q is None else
                       pitch_gap_st(src.median_f0_hz, real[q][t].median_f0_hz))
        return res

    # --- the tracker, before any prosody claim ----------------------------------
    print("\n" + "=" * 92)
    print("TRACKERS — pYIN against Harvest, and the pitch gap's own scale")
    print("=" * 92)

    def agreement(label, pairs):
        dmed = [abs(float(semitones(a.median_f0_hz, b.median_f0_hz)))
                for a, b in pairs if a.median_f0_hz > 0 and b.median_f0_hz > 0]
        ra = np.array([a.pitch_range_st for a, b in pairs], float)
        rb = np.array([b.pitch_range_st for a, b in pairs], float)
        ok = np.isfinite(ra) & np.isfinite(rb)
        corr = np.corrcoef(ra[ok], rb[ok])[0, 1] if ok.sum() > 2 else float("nan")
        jp = mean([a.jump_fraction for a, _ in pairs])
        jh = mean([b.jump_fraction for _, b in pairs])
        print(f"{label:>14}  n={len(pairs):<4} median F0 |pYIN-Harvest|: median "
              f"{np.median(dmed):.2f} st, 95th pct {np.percentile(dmed, 95):.2f} st, "
              f"> 6 st: {sum(d > 6 for d in dmed)}")
        print(f"{'':>14}  pitch range: pYIN {np.nanmean(ra):.2f} st, Harvest "
              f"{np.nanmean(rb):.2f} st, corr {corr:+.3f};  octave jumps: pYIN "
              f"{100 * jp:.2f}%, Harvest {100 * jh:.2f}%")

    agreement("real", [(r["pyin"], r["harvest"]) for r in real])
    agreement("synthesized", [(s["pyin"], s["harvest"])
                              for arm, *_ in ARMS for s in synth[arm]])
    for t in TRACKERS:
        same = finite(gaps(real, t, other_same))
        diff = finite(gaps(real, t, other_speaker))
        sep = float(np.mean([s < d for s in same for d in diff]))
        print(f"{'pitch gap ' + t:>14}  same speaker {same.mean():.2f} st (n={same.size}), "
              f"different speaker {diff.mean():.2f} st (n={diff.size}); "
              f"P(same < different) = {sep:.3f}")

    # --- prosody per arm ----------------------------------------------------------
    for t in TRACKERS:
        print("\n" + "=" * 92)
        print(f"PROSODY ({'pYIN' if t == 'pyin' else 'Harvest'}) — styletts2 on "
              f"librispeech, n={n}, {args.text_case} case")
        print("=" * 92)
        print(f"{'arm':>24} {'range':>6} {'vs real: d':>11} {'+/-':>7} {'p':>7}"
              f"   {'gap':>5} {'words/s':>8}   {'vs off: d range':>15} {'+/-':>7} {'p':>7}")
        rr = col(real, t, "pitch_range_st")
        print(f"{'ground truth':>24} {mean(rr):>6.2f} {'':>11} {'':>7} {'':>7}   "
              f"{mean(gaps(real, t, other_same)):>5.2f} "
              f"{mean(col(real, t, 'words_per_s')):>8.2f}   <- read every row against this")
        ctrl = col(synth[CONTROL], t, "pitch_range_st")
        for arm, name, *_ in ARMS:
            sr_ = col(synth[arm], t, "pitch_range_st")
            d, up, down, pv = paired(sr_, rr)
            line = (f"{name:>24} {mean(sr_):>6.2f} {d:>+11.2f} {f'{up}/{down}':>7} "
                    f"{pv:>7.2g}   {mean(gaps(synth[arm], t, lambda k: k)):>5.2f} "
                    f"{mean(col(synth[arm], t, 'words_per_s')):>8.2f}")
            if arm != CONTROL:
                d2, u2, w2, p2 = paired(sr_, ctrl)
                line += f"   {d2:>+15.2f} {f'{u2}/{w2}':>7} {p2:>7.2g}"
            print(line)
        print("  range: p10-p90 of F0, semitones. 'vs real' pairs each clone with the "
              "speaker's own")
        print("  reading of the same text (+ = livelier); 'vs off' with diffusion off "
              "at seed 1234.")
        print("  gap: |median F0 - the held-out recording's|, st; ground truth = another "
              "recording.")

    # --- truth: the contour the vocoder was driven with ------------------------------
    print("\n" + "=" * 92)
    print("TRUTH — the model's own F0 contour (synthesized audio only; no tracker)")
    print("=" * 92)
    t_off = np.array([r["truth"][1] for r in synth[CONTROL]])
    print(f"{'arm':>24} {'range':>6}   {'vs off: d range':>15} {'+/-':>7} {'p':>9}"
          f"   {'tracker - truth: pYIN':>21} {'Harvest':>8}")
    for arm, name, *_ in ARMS:
        tr = np.array([r["truth"][1] for r in synth[arm]])
        line = f"{name:>24} {np.nanmean(tr):>6.2f}   "
        if arm != CONTROL:
            d, up, down, pv = paired(tr, t_off)
            line += f"{d:>+15.2f} {f'{up}/{down}':>7} {pv:>9.2g}   "
        else:
            line += f"{'':>15} {'':>7} {'':>9}   "
        err = {t: mean([r[t].pitch_range_st - r["truth"][1] for r in synth[arm]])
               for t in TRACKERS}
        line += f"{err['pyin']:>+21.2f} {err['harvest']:>+8.2f}"
        print(line)
    print("  range: p10-p90 of the predicted contour over frames >= 50 Hz, semitones. "
          "'tracker - truth'")
    print("  is each tracker's mean error on that arm: equal across arms means the arm "
          "differences above")
    print("  survive measurement.")

    # --- seed diversity -------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SEED DIVERSITY — two seeds' renditions of the same utterance")
    print("=" * 92)
    for t in TRACKERS:
        print(f"[{'pYIN' if t == 'pyin' else 'Harvest'}]")
        for label, a, b in SETTINGS:
            la = np.array([r["audio_s"] for r in synth[a]])
            lb = np.array([r["audio_s"] for r in synth[b]])
            dlen = np.abs(la - lb) / ((la + lb) / 2)
            ident = int(np.sum(la == lb))
            dr = finite([abs(x - y) for x, y in zip(col(synth[a], t, "pitch_range_st"),
                                                     col(synth[b], t, "pitch_range_st"))])
            dm = finite([pitch_gap_st(x, y) for x, y in zip(col(synth[a], t, "median_f0_hz"),
                                                             col(synth[b], t, "median_f0_hz"))])
            print(f"{label:>24}  length differs {100 * dlen.mean():5.2f}% "
                  f"(identical on {ident}/{n})   |d range| {dr.mean():.2f} st   "
                  f"|d median F0| {dm.mean():.2f} st")
    print("[truth]")
    for label, a, b in SETTINGS:
        tr = [abs(x["truth"][1] - y["truth"][1]) for x, y in zip(synth[a], synth[b])]
        tm = [pitch_gap_st(x["truth"][0], y["truth"][0]) for x, y in zip(synth[a], synth[b])]
        same = sum(np.array_equal(np.load(str(audio_dir / a / f"{k:02d}_{it['speaker']}_{it['j']}.f0.npy")),
                                  np.load(str(audio_dir / b / f"{k:02d}_{it['speaker']}_{it['j']}.f0.npy")))
                   for k, it in enumerate(items))
        print(f"{label:>24}  contour identical on {same}/{n}   |d range| {mean(tr):.2f} st   "
              f"|d median F0| {mean(tm):.2f} st")
    pool.close()


if __name__ == "__main__":
    main()
