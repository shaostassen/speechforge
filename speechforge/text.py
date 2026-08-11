"""How corpus text is presented to a synthesis model.

Not a cosmetic detail. LibriSpeech and EdAcc transcripts are distributed in
unpunctuated upper case (`STUFF IT INTO YOU HIS BELLY COUNSELLED HIM`), and that
is *out-of-distribution input* for an LLM-based TTS, which has read ordinary
prose. Measured on CosyVoice 2, LibriSpeech test-clean, 9 utterances:

    ALL CAPS, as the corpus ships it   WER 0.316
    ordinary sentence case             WER 0.048

— a 6.6x difference, better on 8 of 9 utterances, from changing nothing but the
case of the input string. Chatterbox is far less sensitive, which is exactly why
this is dangerous: feeding both models the raw corpus text looks scrupulously
even-handed while handing one of them input it was never trained on.

WER scoring is case- and punctuation-insensitive, so this choice cannot flatter a
system through the metric — it only changes what the model is asked to read.

The default stays `raw` so that previously recorded runs remain comparable, and
the choice is printed in every provenance banner. A silent normalization here
would quietly invalidate every number in docs/VALIDATION.md.
"""
from __future__ import annotations

_TERMINAL = ".!?…\"')]"


def sentence_case(text: str) -> str:
    """Upper-case corpus text -> ordinary prose the model has seen before.

    Lower-cases, capitalizes the first letter, and adds a full stop when the text
    does not already end in terminal punctuation. Text that is already mixed case
    is left alone apart from the terminal stop: re-casing prose that a corpus
    already punctuated would destroy proper nouns and acronyms.
    """
    s = text.strip()
    if not s:
        return s
    # "Already prose" test: any lower-case letter means the corpus punctuated and
    # cased this text itself, so only the terminal stop is ours to add.
    if not any(c.islower() for c in s):
        s = s.lower()
    s = s[:1].upper() + s[1:]
    if s[-1] not in _TERMINAL:
        s += "."
    return s


def apply_case(text: str, mode: str) -> str:
    """``mode`` is 'raw' (untouched, the default) or 'sentence'."""
    if mode == "raw":
        return text
    if mode == "sentence":
        return sentence_case(text)
    raise ValueError(f"unknown text case mode {mode!r}; use 'raw' or 'sentence'")
