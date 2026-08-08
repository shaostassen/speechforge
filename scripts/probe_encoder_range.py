"""Encoder dynamic range, measured on a corpus that actually has many speakers.

The LibriSpeech dummy split has exactly one speaker with multiple utterances, so
it cannot produce a different-speaker floor at all. EdAcc has many, so use it.

The question is unchanged and decisive: same-speaker similarity came out at
0.977 with this encoder. If *different* speakers also score ~0.95, the metric
has no usable range and the clone's 0.927 is uninterpretable.
"""
import io
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

sys.path.insert(0, "/home/shao/speechforge/repo")

from speechforge.evaluate import cosine
from speechforge.similarity import load_encoder

MIN_S, WANT_SPK, PER_SPK, SCAN = 2.0, 6, 3, 900
SKIP = "IGNORE_TIME_SEGMENT_IN_SCORING"

ds = load_dataset("edinburghcstr/edacc", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

by_spk = defaultdict(list)
seen = 0
for row in ds:
    seen += 1
    if seen > SCAN:
        break
    text = (row.get("text") or "").strip()
    if not text or SKIP in text or len(text.split()) < 3:
        continue
    a = row["audio"]
    raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) / sr < MIN_S:
        continue
    spk = row.get("speaker") or "?"
    if len(by_spk[spk]) < PER_SPK:
        by_spk[spk].append((y.astype(np.float32), int(sr)))
    if sum(1 for v in by_spk.values() if len(v) >= 2) >= WANT_SPK:
        break

by_spk = {k: v for k, v in by_spk.items() if len(v) >= 2}
print(f"speakers={len(by_spk)}  utterances={[len(v) for v in by_spk.values()]}")

enc = load_encoder()
print(f"encoder: {enc.describe()}\n")

emb = {s: [enc(y, sr) for y, sr in u] for s, u in by_spk.items()}

same = [cosine(a, b) for vs in emb.values() for a, b in combinations(vs, 2)]
diff = [cosine(a, b) for s1, s2 in combinations(emb, 2)
        for a in emb[s1] for b in emb[s2]]

print(f"SAME speaker  n={len(same):>4}  mean={np.mean(same):.3f}  "
      f"min={np.min(same):.3f}  max={np.max(same):.3f}")
print(f"DIFF speaker  n={len(diff):>4}  mean={np.mean(diff):.3f}  "
      f"min={np.min(diff):.3f}  max={np.max(diff):.3f}")

gap = float(np.mean(same) - np.mean(diff))
print(f"\nusable range (same - diff) = {gap:.3f}")
overlap = sum(1 for d in diff if d > np.min(same)) / max(len(diff), 1)
print(f"different-speaker pairs scoring above the LOWEST same-speaker pair: "
      f"{overlap:.0%}")
print("\nVERDICT:", "usable — the classes separate"
      if gap > 0.15 else "NOT usable as configured — classes barely separate")
