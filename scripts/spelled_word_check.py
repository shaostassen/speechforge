#!/usr/bin/env python3
"""Are the raw-caps errors *at* the spelled-out word? A word-level check.

presentation_mechanism.py shows raw-caps damage concentrating in the utterances
that contain a word espeak reads letter by letter (IT -> "I.T."). Its input rows
carry no hypotheses, so it cannot show that the errors sit at that word rather
than elsewhere in the same utterances. This re-synthesizes only those
utterances, raw caps, with the study's seed, and:

1. checks each one reproduces the study's WER exactly — otherwise this would be
   a different sample, not a look inside the measured one;
2. checks espeak spelled the word *in context*, not only in isolation, by
   counting it in the phonemes the backend actually synthesized;
3. aligns the transcript to the reference and, per occurrence of the spelled
   word, reports what the ASR heard there, counting as errors at the word a
   substitution or deletion of it and any insertion directly beside it — "I.T."
   heard as "i d" is one substitution plus one insertion, and the alignment may
   place the insertion on either side.

    python scripts/spelled_word_check.py runs/present_styletts2.json
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from speechforge import corpora  # noqa: E402
from speechforge.evaluate import whisper_transcriber  # noqa: E402
from speechforge.synth import clone, load_backend  # noqa: E402
from speechforge.text import apply_case  # noqa: E402
from presentation_mechanism import _bare, spelled_words  # noqa: E402
from speechforge.backends.styletts2 import phonemize  # noqa: E402


def hardware_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rows", help="per-utterance JSON from text_presentation.py")
    p.add_argument("--backend", default="styletts2")
    p.add_argument("--speakers", type=int, default=12)
    p.add_argument("--utterances", type=int, default=5)
    p.add_argument("--asr-model", default="large-v3")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    print(f"study=spelled_word_check  hardware={hardware_label()}  "
          f"backend={args.backend}  asr={args.asr_model}  seed={args.seed}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")

    from speechlens.metrics import EQUAL, INS, align, normalize_text, wer

    rows = json.loads(Path(args.rows).read_text())
    speakers = corpora.load("librispeech", args.speakers, args.utterances)
    items = [(spk, utts[0], t) for spk, utts in speakers.items()
             for _a, _s, t in utts[1:]]
    if len(items) != len(rows) or any(
            r["speaker"] != s or r["words"] != len(t.split())
            for (s, _ref, t), r in zip(items, rows)):
        raise SystemExit("rows do not align with the reloaded corpus")

    spelled = spelled_words({w for _s, _r, t in items for w in t.split()})
    print(f"spelled-out words: {sorted(spelled)}")
    spelled_ph = {_bare(ph) for ph in phonemize(sorted(spelled))}

    backend = load_backend(args.backend, device="cpu")
    backend.set_threads(args.threads)
    transcribe = whisper_transcriber(model_size=args.asr_model, device="cpu",
                                     compute_type="int8")

    total = at_word = n = reproduced = 0
    occurrences = in_context = misheard = 0
    for (spk, (ref_audio, ref_sr, ref_text), text), row in zip(items, rows):
        words = text.split()
        if not any(w in spelled for w in words):
            continue
        n += 1
        # Reference tokens, each tagged with whether its source word is spelled.
        ref_tok, tag = [], []
        for w in words:
            toks = normalize_text(w).split()
            ref_tok += toks
            tag += [w in spelled] * len(toks)
        if ref_tok != normalize_text(text).split():
            raise SystemExit(f"per-word normalization differs from whole-text at {spk}")

        wav, sr, _ = clone(backend, apply_case(text, "raw"), ref_audio, ref_sr,
                           seed=args.seed,
                           reference_text=apply_case(ref_text, "raw"))
        phonemes = backend.last_synthesis.get("phonemes") or []
        if isinstance(phonemes, str):
            phonemes = [phonemes]
        spelled_here = sum(1 for chunk in phonemes for tok in chunk.split()
                           if _bare(tok) in spelled_ph)
        hyp = transcribe(wav, sr)
        w = wer(text, hyp)
        same = abs(w - row["wer_raw"]) < 1e-12
        reproduced += same
        hyp_tok = normalize_text(hyp).split()

        ops = align(ref_tok, hyp_tok)
        errs = sum(op != EQUAL for op, _i, _j in ops)
        here = 0
        heard = []
        for k, (op, i, j) in enumerate(ops):
            if i is None or not tag[i]:
                continue
            got, err = [], 0
            if op != EQUAL:
                err += 1
            # Insertions directly beside this word, on either side.
            before = []
            m = k - 1
            while m >= 0 and ops[m][0] == INS:
                before.insert(0, hyp_tok[ops[m][2]])
                m -= 1
            after = []
            m = k + 1
            while m < len(ops) and ops[m][0] == INS:
                after.append(hyp_tok[ops[m][2]])
                m += 1
            got = before + ([hyp_tok[j]] if j is not None else []) + after
            err += len(before) + len(after)
            here += err
            heard.append(" ".join(got) or "(deleted)")
            misheard += err > 0
        occurrences += sum(tag)
        in_context += min(spelled_here, sum(tag))
        total += errs
        at_word += here

        print(f"\n{spk}  raw WER {w:.3f} (study {row['wer_raw']:.3f}, "
              f"{'reproduced' if same else 'NOT REPRODUCED'})  "
              f"spelled in context {spelled_here}/{sum(tag)}  "
              f"errors {errs}, at the spelled word {here}")
        print(f"  ref: {' '.join(t.upper() if g else t for t, g in zip(ref_tok, tag))}")
        print(f"  hyp: {' '.join(hyp_tok)}")
        print(f"  heard as: {heard}")

    print("\n" + "=" * 66)
    print(f"{n} utterances with a spelled-out word; {reproduced}/{n} reproduce "
          f"the study's raw-caps WER exactly")
    print(f"occurrences of a spelled-out word: {occurrences}; spelled by espeak in "
          f"context: {in_context}; heard as something other than the word: "
          f"{misheard}")
    if total:
        print(f"raw-caps word errors in these utterances: {total}; at a spelled "
              f"word: {at_word} ({at_word / total:.0%})")


if __name__ == "__main__":
    main()
