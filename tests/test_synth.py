"""The backend-agnostic layer, exercised with a fake backend.

No torch, no weights, no network — same dependency-injection discipline the ASR
project used to keep its suite runnable anywhere. What is under test here is the
plumbing: that the reference gets cleaned before it reaches a model, that the
report survives, and that an unusable reference fails loudly instead of
producing a silently bad clone.
"""
import numpy as np
import pytest

from speechforge.synth import clone, load_backend


class FakeBackend:
    """Records what it was handed, returns a fixed tone."""
    sample_rate = 24000

    def __init__(self):
        self.calls = []
        self.threads = None

    def synthesize(self, text, reference_path, seed=None):
        import soundfile as sf
        ref, sr = sf.read(reference_path, dtype="float32")
        self.calls.append({"text": text, "ref_len": len(ref), "ref_sr": sr,
                           "ref_peak": float(np.max(np.abs(ref))),
                           "seed": seed})
        t = np.arange(int(1.0 * self.sample_rate)) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), self.sample_rate

    def set_threads(self, n):
        self.threads = n

    def describe(self):
        return "fake"


def _speech(seconds=6.0, sr=16000, amp=0.4):
    t = np.arange(int(seconds * sr)) / sr
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (amp * env * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def test_reference_is_resampled_to_the_backend_rate():
    # A 16 kHz reference handed to a 24 kHz model must be resampled, not passed
    # through — sample-rate mismatch is a silent quality killer.
    b = FakeBackend()
    clone(b, "hello", _speech(5.0), 16000)
    assert b.calls[0]["ref_sr"] == 24000


def test_reference_is_normalized_before_the_model_sees_it():
    b = FakeBackend()
    clone(b, "hello", _speech(5.0, amp=0.02), 16000)
    assert b.calls[0]["ref_peak"] == pytest.approx(0.95, abs=0.02)


def test_long_reference_is_capped_before_synthesis():
    b = FakeBackend()
    _wav, _sr, report = clone(b, "hello", _speech(40.0), 16000)
    assert b.calls[0]["ref_len"] / 24000 == pytest.approx(10.0, abs=0.3)
    assert any("truncated" in w for w in report["warnings"])


def test_report_comes_back_with_the_audio():
    # A batch job needs to know *why* a clip cloned badly, at the time.
    b = FakeBackend()
    _wav, _sr, report = clone(b, "hello", _speech(1.0), 16000)
    assert "warnings" in report and report["duration_s"] > 0
    assert any("under-determined" in w for w in report["warnings"])


def test_unusable_reference_raises_instead_of_cloning():
    b = FakeBackend()
    with pytest.raises(ValueError, match="unusable reference"):
        clone(b, "hello", np.array([], np.float32), 16000)
    assert b.calls == []


def test_output_carries_the_backend_sample_rate():
    b = FakeBackend()
    wav, sr, _report = clone(b, "hello", _speech(5.0), 16000)
    assert sr == 24000
    assert len(wav) == pytest.approx(sr, rel=0.01)


def test_writes_a_file_when_asked(tmp_path):
    import soundfile as sf
    out = tmp_path / "out.wav"
    b = FakeBackend()
    clone(b, "hello", _speech(5.0), 16000, out_path=str(out))
    assert out.is_file()
    back, sr = sf.read(str(out), dtype="float32")
    assert sr == 24000 and len(back) > 0


def test_unknown_backend_names_the_available_ones():
    with pytest.raises(SystemExit, match="chatterbox"):
        load_backend("does-not-exist")


def test_importing_synth_does_not_import_torch():
    # Backends pin conflicting torch/transformers versions, so the shared layer
    # must stay import-light or the eval-only install breaks.
    import subprocess
    import sys
    code = ("import sys, speechforge.synth; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.stdout.strip() == "False", out.stderr


# --- reproducibility --------------------------------------------------------

class SeedAwareBackend(FakeBackend):
    """Records the seed it was handed, and varies output when unseeded."""
    def __init__(self):
        super().__init__()
        self.seeds = []
        self._n = 0

    def synthesize(self, text, reference_path, seed=None):
        self.seeds.append(seed)
        self._n += 1
        import numpy as np
        # Unseeded output changes between calls, like a sampling TTS model.
        f = 220.0 if seed is not None else 220.0 + self._n
        t = np.arange(int(1.0 * self.sample_rate)) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * f * t)).astype(np.float32), self.sample_rate


def test_seed_is_passed_through_to_the_backend():
    # Chatterbox samples at temperature 0.8, so an unseeded run is not
    # reproducible. Two identical evaluation runs differed enough to flip the
    # sign of the effect being measured.
    b = SeedAwareBackend()
    clone(b, "hello", _speech(5.0), 16000, seed=1234)
    assert b.seeds == [1234]


def test_no_seed_means_none_not_a_silent_default():
    # A silent default seed would make runs look reproducible while hiding the
    # model's true spread.
    b = SeedAwareBackend()
    clone(b, "hello", _speech(5.0), 16000)
    assert b.seeds == [None]


def test_same_seed_reproduces_the_same_audio():
    b = SeedAwareBackend()
    a1, _sr, _ = clone(b, "hello", _speech(5.0), 16000, seed=7)
    a2, _sr2, _ = clone(b, "hello", _speech(5.0), 16000, seed=7)
    import numpy as np
    assert np.array_equal(a1, a2)


# --- generation settings ----------------------------------------------------

class GenKwargsBackend(FakeBackend):
    """Mimics a backend that forwards generation kwargs to its model."""
    def __init__(self, **gen_kwargs):
        super().__init__()
        self._gen_kwargs = dict(gen_kwargs)

    def describe(self):
        if self._gen_kwargs:
            args = ", ".join(f"{k}={v}" for k, v in sorted(self._gen_kwargs.items()))
            return f"fake [{args}]"
        return "fake [defaults]"


def test_describe_records_the_decode_settings():
    # A WER without its decode configuration is not reproducible even with a
    # seed, so the settings have to reach the provenance banner.
    b = GenKwargsBackend(temperature=0.3, cfg_weight=0.5)
    assert "temperature=0.3" in b.describe() and "cfg_weight=0.5" in b.describe()


def test_describe_says_defaults_when_nothing_was_overridden():
    assert "defaults" in GenKwargsBackend().describe()


def test_load_backend_forwards_generation_kwargs():
    import pytest as _pytest
    from speechforge.synth import BACKENDS, load_backend
    seen = {}

    class Recorder:
        sample_rate = 24000
        align_ms = None

        def __init__(self, device="cpu", **kw):
            seen.update(kw)

        def synthesize(self, text, reference_path, seed=None):
            return np.zeros(10, np.float32), self.sample_rate

        def set_threads(self, n): pass

        def describe(self): return "recorder"

    BACKENDS["_recorder"] = Recorder
    try:
        load_backend("_recorder", temperature=0.25)
        assert seen == {"temperature": 0.25}
    finally:
        del BACKENDS["_recorder"]


# --- reference cap ----------------------------------------------------------

def test_ref_seconds_overrides_the_default_cap():
    # The 10 s default is borrowed from other models' documentation, so it has
    # to be testable rather than baked in.
    b = FakeBackend()
    clone(b, "hello", _speech(25.0), 16000, ref_seconds=20.0)
    assert b.calls[0]["ref_len"] / 24000 == pytest.approx(20.0, abs=0.3)


def test_default_cap_still_applies_when_unset():
    b = FakeBackend()
    clone(b, "hello", _speech(25.0), 16000)
    assert b.calls[0]["ref_len"] / 24000 == pytest.approx(10.0, abs=0.3)


def test_ref_seconds_does_not_pad_a_short_reference():
    # Raising the cap must not invent audio that is not there.
    b = FakeBackend()
    clone(b, "hello", _speech(6.0), 16000, ref_seconds=30.0)
    assert b.calls[0]["ref_len"] / 24000 == pytest.approx(6.0, abs=0.3)
