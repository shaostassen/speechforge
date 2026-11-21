"""eval_zeroshot.py stops, rather than blanking a column, when a scorer fails.

A study run once carried on with an empty SIM column because the speaker
encoder's download was blocked: the script printed a warning and the table
rendered anyway. These drive the real ``main()`` with fakes for the corpus,
backend, scorers and synthesis — offline, no weights — and check that a
requested scorer that fails stops the run *before any synthesis*, and that
leaving SIM out is possible only on purpose.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import speechforge.similarity as similarity
from speechforge.evaluate import ScorerUnavailable

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_zeroshot.py"


class _ReachedSynthesis(Exception):
    pass


class _Backend:
    sample_rate = 24000

    def set_threads(self, n):
        pass

    def describe(self):
        return "fake backend"


class _Encoder:
    def describe(self):
        return "fake encoder"


def _fail(exc):
    def load(*a, **k):
        raise exc
    return load


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("eval_zeroshot_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    clip = (np.zeros(16000, dtype=np.float32), 16000, "A TEST SENTENCE")
    monkeypatch.setattr(mod.corpora, "load", lambda *a, **k: {"spk": [clip, clip]})
    monkeypatch.setattr(mod, "load_backend", lambda *a, **k: _Backend())
    monkeypatch.setattr(mod, "whisper_transcriber", lambda **k: (lambda a, sr: ""))
    monkeypatch.setattr(mod, "clone", _fail(_ReachedSynthesis()))

    def run(*flags):
        monkeypatch.setattr(sys, "argv", ["eval_zeroshot.py", "--corpus",
                                          "librispeech_dummy", *flags])
        mod.main()
    return mod, run


def test_a_failed_speaker_encoder_stops_the_run_before_synthesis(script, monkeypatch):
    _mod, run = script
    monkeypatch.setattr(similarity, "load_encoder",
                        _fail(OSError("blocked by the network")))
    # ScorerUnavailable, not _ReachedSynthesis: nothing was synthesized.
    with pytest.raises(ScorerUnavailable, match="--no-sim") as err:
        run()
    assert "speaker encoder" in str(err.value)
    assert "blocked by the network" in str(err.value)


def test_no_sim_skips_the_encoder_entirely(script, monkeypatch, capsys):
    _mod, run = script
    calls = []
    monkeypatch.setattr(similarity, "load_encoder",
                        lambda: calls.append(1) or _Encoder())
    with pytest.raises(_ReachedSynthesis):   # got past every scorer
        run("--no-sim")
    assert calls == []
    assert "off (--no-sim)" in capsys.readouterr().out


def test_a_working_encoder_still_runs(script, monkeypatch):
    _mod, run = script
    monkeypatch.setattr(similarity, "load_encoder", lambda: _Encoder())
    with pytest.raises(_ReachedSynthesis):
        run()


@pytest.mark.parametrize("flag, loader, name", [
    ("--second-asr", "ctc_transcriber", "second ASR"),
    ("--mos", "mos_predictor", "MOS predictor"),
])
def test_a_requested_optional_scorer_that_fails_stops_the_run(
        script, monkeypatch, flag, loader, name):
    mod, run = script
    monkeypatch.setattr(similarity, "load_encoder", lambda: _Encoder())
    monkeypatch.setattr(mod, loader, _fail(ImportError("missing extra")))
    with pytest.raises(ScorerUnavailable, match=f"drop {flag}") as err:
        run(flag)
    assert name in str(err.value)
