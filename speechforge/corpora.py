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
from typing import Dict, List, Optional, Tuple

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


LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def load_librispeech_test_clean(n_speakers: int, per_speaker: int,
                                root: Optional[str] = None) -> SpeakerSet:
    """Real LibriSpeech test-clean — 40 speakers, the standard benchmark.

    Downloaded and parsed directly from OpenSLR rather than through
    `datasets`: the HF loaders for LibriSpeech are script-based and script
    datasets are no longer supported, so the hub path is a dead end. The
    archive is ~346 MB and the on-disk layout is trivial to walk:

        test-clean/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
        test-clean/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

    Speaker identity comes from the directory structure, so the held-out split
    is exact — no guessing from metadata, and no risk of the one-speaker
    problem that makes the dummy split unusable for this protocol.
    """
    import soundfile as sf

    base = Path(root or Path.home() / ".cache" / "speechforge")
    base.mkdir(parents=True, exist_ok=True)
    extracted = base / "LibriSpeech" / "test-clean"

    if not extracted.is_dir():
        import tarfile
        import urllib.request
        archive = base / "test-clean.tar.gz"
        if not archive.is_file():
            print(f"downloading LibriSpeech test-clean (~346 MB) to {archive}")
            urllib.request.urlretrieve(LIBRISPEECH_URL, archive)
        print(f"extracting to {base}")
        with tarfile.open(archive) as tf:
            tf.extractall(base)

    by_spk: SpeakerSet = defaultdict(list)
    for spk_dir in sorted(extracted.iterdir()):
        if not spk_dir.is_dir():
            continue
        for chapter in sorted(spk_dir.iterdir()):
            trans = next(chapter.glob("*.trans.txt"), None)
            if trans is None:
                continue
            texts = {}
            for line in trans.read_text().splitlines():
                uid, _, text = line.partition(" ")
                texts[uid] = text.strip()
            for flac in sorted(chapter.glob("*.flac")):
                text = texts.get(flac.stem)
                if not text or len(text.split()) < MIN_WORDS:
                    continue
                if len(by_spk[spk_dir.name]) >= per_speaker + 1:
                    break
                y, sr = sf.read(str(flac), dtype="float32")
                if len(y) / sr < MIN_S:
                    continue
                by_spk[spk_dir.name].append((y.astype(np.float32), int(sr), text))
        if sum(1 for v in by_spk.values()
               if len(v) >= per_speaker + 1) >= n_speakers:
            break
    return _select(by_spk, n_speakers, per_speaker)


CORPORA = {"edacc": load_edacc,
           "librispeech": load_librispeech_test_clean,
           "librispeech_dummy": load_librispeech_dummy}


def load(name: str, n_speakers: int, per_speaker: int) -> SpeakerSet:
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; have {sorted(CORPORA)}")
    speakers = CORPORA[name](n_speakers, per_speaker)
    if not speakers:
        raise SystemExit(
            f"{name}: no speaker had {per_speaker + 1} usable utterances "
            f"(need one to clone from plus {per_speaker} held out)")
    return speakers
