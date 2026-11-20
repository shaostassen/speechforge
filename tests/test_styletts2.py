"""The StyleTTS 2 backend's torch-free parts.

No torch, no weights, no network. What a model run cannot check cheaply is
checked here: that the GPL boundary keeps phonemizer out of this process and its
line protocol cannot silently misalign, that long inputs are split rather than
overflowing PL-BERT, that the checkpoint loader refuses anything it was not
built to load, and that partial weight loads fail instead of producing audio.

Parity with upstream's own inference code — bit-exact at three diffusion
settings — needs the weights, so it lives in docs/VALIDATION.md, not here.
"""
import os
import pickle
import shutil
import subprocess
import sys
import types
from collections import OrderedDict

import pytest

from speechforge.backends import styletts2 as st


class FakeRun:
    """Stands in for subprocess.run; records the call, returns canned output."""

    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls = []

    def __call__(self, cmd, input=None, **kwargs):
        self.calls.append({"cmd": cmd, "input": input})
        return types.SimpleNamespace(stdout=self.stdout, stderr=self.stderr,
                                     returncode=self.returncode)


# --- phonemization ---------------------------------------------------------

def test_phonemize_invokes_the_cli_with_upstreams_exact_options():
    run = FakeRun(stdout="a\nb\n")
    st.phonemize(["x", "y"], executable="phonemize", runner=run)
    cmd = run.calls[0]["cmd"]
    assert cmd[0] == "phonemize"
    # The options that make the CLI byte-identical to upstream's library call.
    for opt in ("en-us", "espeak", "--preserve-punctuation", "--with-stress"):
        assert opt in cmd
    assert "-q" in cmd   # warnings must not land on stdout, the data channel


def test_newlines_inside_a_text_cannot_split_it_into_two_outputs():
    # The CLI is line-oriented: an embedded newline would yield two output lines
    # and shift every later text onto the wrong phonemes.
    run = FakeRun(stdout="p1\np2\n")
    st.phonemize(["first\nline", "second"], executable="phonemize", runner=run)
    assert run.calls[0]["input"] == "first line\nsecond\n"


def test_a_line_count_mismatch_is_refused_rather_than_guessed():
    run = FakeRun(stdout="only one\n")
    with pytest.raises(RuntimeError, match="refusing to guess"):
        st.phonemize(["a", "b"], executable="phonemize", runner=run)


def test_cli_failure_is_an_error_not_empty_phonemes():
    run = FakeRun(returncode=1, stderr="espeak not found")
    with pytest.raises(RuntimeError, match="espeak not found"):
        st.phonemize(["a"], executable="phonemize", runner=run)


def test_empty_output_lines_survive_alignment():
    run = FakeRun(stdout="a\n\nc\n")
    assert st.phonemize(["a", "", "c"], executable="phonemize",
                        runner=run) == ["a", "", "c"]


@pytest.mark.skipif(shutil.which("phonemize") is None,
                    reason="phonemizer CLI and espeak-ng not installed")
def test_real_cli_keeps_one_output_per_input():
    out = st.phonemize(["Hello there.", "", "Two words"])
    assert len(out) == 3 and out[1] == "" and out[0] and out[2]


def test_module_import_does_not_pull_in_phonemizer_or_torch():
    # The GPL boundary and the torch-free core, checked in a clean interpreter.
    code = ("import sys, speechforge.backends.styletts2, speechforge.synth; "
            "print('torch' in sys.modules, 'phonemizer' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.stdout.strip() == "False False", out.stderr


# --- tokenization --------------------------------------------------------------

SYMBOLS = {"$": 0, " ": 1, "a": 2, "b": 3, ",": 4}


def test_tokenize_prepends_the_pad_token_like_upstream():
    ids, dropped = st.tokenize("ab", SYMBOLS, str.split)
    assert ids == [0, 2, 3] and dropped == []


def test_unknown_symbols_are_reported_not_silently_dropped():
    # Upstream's TextCleaner prints and drops them; the model then reads less
    # than the text, with nothing recorded.
    ids, dropped = st.tokenize("a`b", SYMBOLS, str.split)
    assert ids == [0, 2, 3] and dropped == ["`"]


def test_word_tokenization_is_rejoined_with_spaces():
    ids, _ = st.tokenize("a,b", SYMBOLS, lambda s: ["a", ",", "b"])
    assert ids == [0, 2, 1, 4, 1, 3]


# --- long inputs ------------------------------------------------------------------

def by_chars(text):
    return len(text)


def test_short_text_is_one_chunk():
    assert st.plan_chunks("a short text", by_chars, 100) == ["a short text"]


def test_empty_text_has_no_chunks():
    assert st.plan_chunks("   ", by_chars, 100) == []


def test_every_chunk_fits_and_nothing_is_lost():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = st.plan_chunks(text, by_chars, 120)
    assert all(by_chars(c) <= 120 for c in chunks)
    assert " ".join(c.strip() for c in chunks) == text


def test_sentence_ends_are_preferred_over_commas_and_spaces():
    text = "one two, three four. five six, seven eight"
    chunks = st.plan_chunks(text, by_chars, 30)
    assert chunks[0].endswith("four.")


def test_clause_punctuation_is_preferred_over_bare_spaces():
    text = "alpha beta gamma, delta epsilon zeta"
    chunks = st.plan_chunks(text, by_chars, 25)
    assert chunks[0].endswith("gamma,")


def test_a_word_too_long_to_split_is_an_error():
    with pytest.raises(ValueError, match="cannot split"):
        st.plan_chunks("x" * 50, by_chars, 10)


# --- checkpoint loading ---------------------------------------------------------------

def restricted_loads(data: bytes):
    import io
    return st.RestrictedUnpickler(io.BytesIO(data)).load()


def test_plain_containers_load_normally():
    obj = {"model": OrderedDict(w=[1.0, 2.0])}
    assert restricted_loads(pickle.dumps(obj)) == obj


@pytest.mark.parametrize("target", [os.system, eval, exec, subprocess.Popen])
def test_anything_outside_the_allowlist_is_refused(target):
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        restricted_loads(pickle.dumps(target))


def test_getattr_is_neutralized_not_honoured():
    # In a full unpickler, getattr is the building block of a code-execution
    # payload. Here it resolves to an inert stand-in.
    class Payload:
        def __reduce__(self):
            return (getattr, ("anything", "upper"))
    out = restricted_loads(pickle.dumps(Payload()))
    assert isinstance(out, st._Inert)


def test_pickled_training_state_becomes_inert():
    # The text-aligner checkpoint pickles its AdamW optimizer and OneCycleLR
    # scheduler alongside the weights; inference needs neither.
    data = (b"\x80\x02}q\x00(X\x05\x00\x00\x00modelq\x01}q\x02"
            b"X\t\x00\x00\x00optimizerq\x03ctorch.optim.adamw\nAdamW\nq\x04)"
            b"\x81q\x05}q\x06X\x02\x00\x00\x00lrq\x07G?PbM\xd2\xf1\xa9\xfcsbu.")
    out = restricted_loads(data)
    assert out["model"] == {} and isinstance(out["optimizer"], st._Inert)


def test_restricted_module_is_shaped_for_torch_load():
    # torch.load(pickle_module=...) subclasses .Unpickler and checks __name__.
    assert issubclass(st.restricted_pickle.Unpickler, pickle.Unpickler)
    assert st.restricted_pickle.__name__ != "dill"


def test_partial_weight_loads_fail():
    # Upstream loads PL-BERT with strict=False: a renamed parameter would leave
    # random weights in place and the model would still talk.
    ok = types.SimpleNamespace(missing_keys=[], unexpected_keys=[])
    st.check_state_dict_load(ok, "fine")
    bad = types.SimpleNamespace(missing_keys=["encoder.w"], unexpected_keys=[])
    with pytest.raises(RuntimeError, match="1 missing"):
        st.check_state_dict_load(bad, "PL-BERT")
    extra = types.SimpleNamespace(missing_keys=[], unexpected_keys=["x", "y"])
    with pytest.raises(RuntimeError, match="2 unexpected"):
        st.check_state_dict_load(extra, "decoder")


# --- settings and construction -----------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"alpha": -0.1}, {"beta": 1.5}, {"diffusion_steps": 0},
    {"diffusion_steps": 2.5}, {"embedding_scale": 0.0}])
def test_out_of_range_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        st.validate_settings(**{**st.DEFAULTS, **kwargs})


def test_upstream_defaults_validate():
    st.validate_settings(**st.DEFAULTS)
    st.validate_settings(alpha=0.0, beta=0.0, diffusion_steps=1,
                         embedding_scale=1.0)


def test_bad_settings_fail_before_anything_is_loaded():
    # Construction checks settings first, so a typo in a sweep costs nothing.
    code = ("import sys\n"
            "from speechforge.backends.styletts2 import StyleTTS2Backend\n"
            "try:\n"
            "    StyleTTS2Backend(temprature=0.5)\n"
            "except ValueError as e:\n"
            "    print('ValueError', 'temprature' in str(e), 'torch' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.stdout.strip() == "ValueError True False", out.stderr


def test_missing_repo_names_the_fix(monkeypatch):
    monkeypatch.delenv("STYLETTS2_REPO", raising=False)
    with pytest.raises(SystemExit, match="yl4579/StyleTTS2"):
        st.StyleTTS2Backend(repo_dir="/nonexistent/styletts2")


def test_describe_always_records_all_four_settings():
    # They are what a diffusion study varies; a number without them is not
    # reproducible, so they appear even at their defaults.
    b = st.StyleTTS2Backend.__new__(st.StyleTTS2Backend)
    b._settings = {**st.DEFAULTS, "alpha": 0.0, "beta": 0.0}
    d = b.describe()
    for part in ("alpha=0.0", "beta=0.0", "diffusion_steps=5",
                 "embedding_scale=1.0", "GPL-3.0", "subprocess"):
        assert part in d


def test_backend_is_registered():
    from speechforge.synth import BACKENDS
    assert BACKENDS["styletts2"] is st.StyleTTS2Backend
