"""Speaker-grouped corpora for the cloning protocol.

The protocol needs **several utterances per speaker**: one to clone from, the
rest held out for scoring. That requirement rules out more corpora than you'd
expect — the LibriSpeech dummy split, which the ASR project used happily, has
exactly one speaker with more than one utterance, so it can produce neither a
held-out split worth the name nor a different-speaker floor.

Two sources, both loadable without torch (`decode=False` plus soundfile):

- ``edacc`` — Edinburgh International Accents of English. Many speakers, accent
  metadata, spontaneous conversation, real recording conditions. The realistic
  option, and the harder one.
- ``librispeech_dummy`` — tiny, cached, studio-clean. Fine for smoke-testing the
  plumbing, useless for a result. Kept because a fast loop matters.
"""
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Utterance = Tuple[np.ndarray, int, str]          # audio, sample rate, text
SpeakerSet = Dict[str, List[Utterance]]

# EdAcc marks stretches excluded from its own scoring protocol. Left in, they
# compare a transcript against a placeholder string and report a meaningless WER.
EDACC_SKIP = "IGNORE_TIME_SEGMENT_IN_SCORING"

MIN_S = 2.0          # Whisper is unreliable below this; LID more so
MIN_WORDS = 4        # a 1-3 word reference makes WER hopelessly coarse


def _read(row) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    a = row["audio"]
    raw = a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes()
    y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def load_librispeech_dummy(n_speakers: int, per_speaker: int) -> SpeakerSet:
    """Studio-clean, cached, and almost single-speaker. Smoke tests only."""
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    by_spk: SpeakerSet = defaultdict(list)
    for i in range(len(ds)):
        row = ds[i]
        y, sr = _read(row)
        by_spk[str(row.get("speaker_id", "?"))].append((y, sr, row["text"].strip()))
    return _select(by_spk, n_speakers, per_speaker)


def load_edacc(n_speakers: int, per_speaker: int, scan: int = 1500) -> SpeakerSet:
    """Spontaneous, accented, many speakers.

    Streamed, since the corpus is ~40 h. Utterances arrive grouped by
    conversation, so a naive first-N sample yields one speaker — hence the scan
    budget and the grouping.
    """
    from datasets import Audio, load_dataset
    ds = load_dataset("edinburghcstr/edacc", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    by_spk: SpeakerSet = defaultdict(list)
    seen = 0
    for row in ds:
        seen += 1
        if seen > scan:
            break
        text = (row.get("text") or "").strip()
        if not text or EDACC_SKIP in text or len(text.split()) < MIN_WORDS:
            continue
        y, sr = _read(row)
        if len(y) / sr < MIN_S:
            continue
        spk = str(row.get("speaker") or "?")
        if len(by_spk[spk]) < per_speaker + 1:
            by_spk[spk].append((y, sr, text))
        ready = sum(1 for v in by_spk.values() if len(v) >= per_speaker + 1)
        if ready >= n_speakers:
            break
    return _select(by_spk, n_speakers, per_speaker)


def _select(by_spk: SpeakerSet, n_speakers: int, per_speaker: int) -> SpeakerSet:
    """Keep only speakers with enough utterances for a held-out split.

    Speakers with too few are dropped rather than padded by reusing the
    reference clip as a held-out one — that is precisely the SIM-r mistake the
    evaluation harness refuses.
    """
    out: SpeakerSet = {}
    for spk, utts in by_spk.items():
        if len(utts) >= per_speaker + 1:
            out[spk] = utts[: per_speaker + 1]
        if len(out) >= n_speakers:
            break
    return out


CORPORA = {"edacc": load_edacc, "librispeech_dummy": load_librispeech_dummy}


def load(name: str, n_speakers: int, per_speaker: int) -> SpeakerSet:
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; have {sorted(CORPORA)}")
    speakers = CORPORA[name](n_speakers, per_speaker)
    if not speakers:
        raise SystemExit(
            f"{name}: no speaker had {per_speaker + 1} usable utterances "
            f"(need one to clone from plus {per_speaker} held out)")
    return speakers
