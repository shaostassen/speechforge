"""Speaker-encoder plumbing, tested with a fake encoder.

The real WavLM model needs torch and a download, so the actual embedding quality
is measured in scripts/ and recorded in docs/VALIDATION.md. What is pinned here
is the logic around it: resampling before encoding, and the self-similarity
baseline that gives the similarity column its ceiling.
"""
import numpy as np
import pytest

from speechforge.similarity import (DEFAULT_MODEL, MODEL_SR, SpeakerEncoder,
                                    self_similarity_baseline)


def _tone(hz, seconds=1.0, sr=16000, amp=0.4):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class FakeEncoder:
    """Embeds dominant frequency plus the sample count it was handed.

    Carrying the length lets a test detect whether resampling happened before
    encoding, which is the thing most likely to be silently wrong.
    """

    def __init__(self):
        self.seen = []

    def __call__(self, audio, sr):
        spec = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
        hz = float(np.fft.rfftfreq(len(audio), 1 / sr)[int(np.argmax(spec))])
        self.seen.append({"n": len(audio), "sr": sr})
        return np.array([hz, 1.0])


# --- self-similarity baseline ----------------------------------------------

def test_baseline_is_high_for_recordings_of_one_speaker():
    enc = FakeEncoder()
    utts = [_tone(200.0), _tone(200.0, 1.3), _tone(200.0, 0.8)]
    assert self_similarity_baseline(enc, utts, 16000) > 0.99


def test_baseline_is_lower_when_the_recordings_differ():
    enc = FakeEncoder()
    same = self_similarity_baseline(enc, [_tone(200.0), _tone(200.0, 1.3)], 16000)
    mixed = self_similarity_baseline(enc, [_tone(200.0), _tone(900.0, 1.3)], 16000)
    assert mixed < same


def test_baseline_needs_at_least_two_utterances():
    # One recording gives no pair, so there is no ceiling to report. None is
    # the honest answer; a 1.0 would be read as a perfect ceiling.
    assert self_similarity_baseline(FakeEncoder(), [_tone(200.0)], 16000) is None
    assert self_similarity_baseline(FakeEncoder(), [], 16000) is None


def test_baseline_averages_every_pair():
    enc = FakeEncoder()
    self_similarity_baseline(enc, [_tone(200.0), _tone(300.0), _tone(400.0)],
                             16000)
    assert len(enc.seen) == 3        # 3 embeddings, 3 pairwise comparisons


# --- resampling before encoding --------------------------------------------

class _FakeExtractor:
    """Captures what the real __call__ hands to the feature extractor."""

    def __init__(self):
        self.seen = None

    def __call__(self, y, sampling_rate=None, return_tensors=None,
                 padding=None):
        self.seen = {"n": len(y), "sampling_rate": sampling_rate}
        # `.to(device)` is called on every value by the real __call__, so the
        # fake has to offer it — the alternative is a fake that diverges from
        # the interface and a test that stops covering the real path.
        return {"input_values": _Movable(y)}


class _Movable:
    def __init__(self, value):
        self.value = value

    def to(self, _device):
        return self


class _FakeModel:
    def __call__(self, **kwargs):
        class Out:
            embeddings = [_FakeTensor()]
        return Out()

    def eval(self):
        return self

    def to(self, device):
        return self


class _FakeTensor:
    def cpu(self):
        return self

    def numpy(self):
        return np.array([1.0, 0.0])


class _FakeTorch:
    @staticmethod
    def no_grad():
        import contextlib
        return contextlib.nullcontext()


def _wired_encoder():
    """A SpeakerEncoder with fakes injected, so the REAL __call__ body runs.

    Injecting at the model boundary rather than monkeypatching __call__: an
    earlier version of this test replaced __call__ itself, which meant it
    exercised none of the resampling logic it claimed to check and passed
    vacuously.
    """
    enc = SpeakerEncoder()
    enc._extractor = _FakeExtractor()
    enc._model = _FakeModel()
    enc._torch = _FakeTorch()
    return enc


def test_encoder_resamples_to_its_training_rate():
    # A 44.1 kHz clip fed to a 16 kHz encoder shifts every formant, and the
    # "speaker difference" it then reports is really a sample-rate difference.
    enc = _wired_encoder()
    enc(_tone(200.0, seconds=1.0, sr=44100), 44100)
    seen = enc._extractor.seen
    assert seen["sampling_rate"] == MODEL_SR
    # 1 s of audio must arrive as ~16000 samples, not 44100.
    assert seen["n"] == pytest.approx(MODEL_SR, rel=0.01)


def test_encoder_leaves_onrate_audio_alone():
    enc = _wired_encoder()
    y = _tone(200.0, seconds=1.0, sr=MODEL_SR)
    enc(y, MODEL_SR)
    assert enc._extractor.seen["n"] == len(y)


def test_encoder_does_not_download_when_already_wired():
    # Guards the `if self._model is None: self.load()` branch: a wired encoder
    # must not reach for the network mid-evaluation.
    enc = _wired_encoder()
    enc.load = lambda: (_ for _ in ()).throw(
        AssertionError("load() called despite an injected model"))
    enc(_tone(200.0), MODEL_SR)


def test_resample_path_is_exercised_for_offrate_audio():
    # Directly: the helper the encoder uses must change the length.
    from speechforge.reference import resample
    y = _tone(200.0, seconds=1.0, sr=44100)
    out = resample(y, 44100, MODEL_SR)
    assert len(out) == pytest.approx(MODEL_SR, rel=0.01)


# --- provenance -------------------------------------------------------------

def test_encoder_reports_which_model_produced_the_numbers():
    # SIM scores are not comparable across encoders — the encoder identity has
    # to travel with the number.
    d = SpeakerEncoder().describe()
    assert DEFAULT_MODEL in d and str(MODEL_SR) in d


def test_importing_similarity_does_not_import_torch():
    import subprocess
    import sys
    code = "import sys, speechforge.similarity; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.stdout.strip() == "False", out.stderr
