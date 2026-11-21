#!/usr/bin/env python3
"""Checks that the StyleTTS 2 backend computes what upstream computes.

Three deviations separate ``StyleTTS2Backend`` from upstream's own inference
code, and each is checked here rather than argued:

  ``phonemizer``  The backend runs phonemizer's ``phonemize`` CLI as a subprocess
                  instead of importing the GPL-3.0 library. Checked byte-for-byte
                  against upstream's library call on every LibriSpeech test-clean
                  transcript, in both text presentations. Also reports what
                  sentence case changes in what a phoneme-input model reads.

  ``memory``      Resident memory over many varied utterances with oneDNN on and
                  off, each in a clean process. The reason the backend switches
                  oneDNN off: with it on, memory grows without bound while live
                  tensors stay constant.

  ``parity``      Bit-exact audio against upstream's inference function — the code
                  of Demo/Inference_LibriTTS.ipynb, run through upstream's own
                  loaders — at three diffusion settings. The backend runs with
                  oneDNN off (it leaks per input shape; see USE_MKLDNN), so parity
                  is checked with upstream under the same flag, and upstream with
                  oneDNN on is compared too, to show what that fix itself costs.

**This script imports phonemizer (GPL-3.0) on purpose**, because comparing
against the library is its job. It is a development check, not part of the
installed package, and nothing in ``speechforge/`` imports it.

    python scripts/styletts2_checks.py phonemizer
    python scripts/styletts2_checks.py parity --reference some_libri.flac
    python scripts/styletts2_checks.py memory --utterances 24
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import shlex
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIBRI = Path.home() / ".cache" / "speechforge" / "LibriSpeech" / "test-clean"


def hardware_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.machine()


def transcripts() -> list:
    out = []
    for trans in sorted(glob.glob(str(LIBRI / "*" / "*" / "*.trans.txt"))):
        out += [line.split(" ", 1)[1].strip() for line in open(trans)]
    if not out:
        raise SystemExit(f"no LibriSpeech transcripts under {LIBRI}; run any "
                         "librispeech study once to download test-clean")
    return out


# --- phonemizer ---------------------------------------------------------------

def check_phonemizer(args) -> None:
    import subprocess

    import phonemizer
    from phonemizer.backend import EspeakBackend

    from speechforge.backends.styletts2 import PHONEMIZE_ARGS, phonemize
    from speechforge.text import apply_case

    espeak = subprocess.run(["espeak-ng", "--version"], capture_output=True,
                            text=True).stdout.strip()
    print(f"phonemizer {phonemizer.__version__}  |  {espeak}")
    print("library call (upstream): EspeakBackend(language='en-us', "
          "preserve_punctuation=True, with_stress=True).phonemize([text])")
    print(f"CLI call (speechforge):  phonemize {' '.join(PHONEMIZE_ARGS)}\n")

    texts = transcripts()
    lib = EspeakBackend(language="en-us", preserve_punctuation=True,
                        with_stress=True)
    out = {}
    for mode in ("raw", "sentence"):
        batch = [apply_case(t, mode) for t in texts]
        cli = phonemize(batch)
        ref = [lib.phonemize([t])[0] for t in batch]   # one at a time, as upstream
        bad = [(t, a, b) for t, a, b in zip(batch, ref, cli) if a != b]
        print(f"{mode:>8}: {len(batch)} transcripts, {len(bad)} mismatches")
        for t, a, b in bad[:5]:
            print(f"    {t[:60]!r}\n      lib {a[:80]!r}\n      cli {b[:80]!r}")
        out[mode] = cli

    # What does sentence case change in what a phoneme-input model reads?
    def strip_stop(p):
        p = p.rstrip()
        return p[:-1].rstrip() if p.endswith(".") else p

    same = [a.rstrip() == strip_stop(b) for a, b in zip(out["raw"], out["sentence"])]
    print(f"\nraw vs sentence case: phonemes identical apart from the final stop "
          f"in {sum(same)}/{len(texts)} ({sum(same) / len(texts):.1%}); "
          f"different in {len(texts) - sum(same)}")
    from collections import Counter
    pairs = Counter()
    for a, b, s in zip(out["raw"], out["sentence"], same):
        if not s:
            wa, wb = a.split(), strip_stop(b).split()
            if len(wa) == len(wb):
                pairs.update((x, y) for x, y in zip(wa, wb) if x != y)
    print("most frequent word-level differences (raw -> sentence), "
          "same-length transcripts only:")
    for (x, y), n in pairs.most_common(10):
        print(f"  {n:5d}  {x:>12} -> {y}")


# --- memory ---------------------------------------------------------------------

def check_memory(args) -> None:
    import json
    import subprocess

    print("Each setting runs in a fresh process: RSS is per-process state, and a "
          "shared process\nwould carry the first run's growth into the second.\n")
    results = {}
    for setting in ("on", "off"):
        r = subprocess.run([sys.executable, __file__, "_memory_worker", setting,
                            str(args.utterances), str(args.threads)],
                           capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if r.returncode != 0 or not line:
            raise SystemExit(f"worker ({setting}) failed:\n{r.stderr[-1500:]}")
        results[setting] = json.loads(line[-1])
    n = args.utterances
    print(f"{'oneDNN':>8} {'after load':>11} " + " ".join(
        f"{f'@{k}':>7}" for k in results['on']['marks']) + f" {'live tensors':>16} {'RTF':>6}")
    for setting, res in results.items():
        print(f"{setting:>8} {res['load_mb']:>9.0f}MB " + " ".join(
            f"{v:>7.0f}" for v in res["rss_mb"]) +
              f" {res['tensors'][0]:>7} -> {res['tensors'][1]:<6} {res['rtf']:>6.3f}")
    on, off = results["on"], results["off"]
    print(f"\ngrowth over {n} utterances: oneDNN on {on['rss_mb'][-1] - on['load_mb']:+.0f} MB, "
          f"off {off['rss_mb'][-1] - off['load_mb']:+.0f} MB (RSS in MB at each mark)")


def _memory_worker(setting: str, n: int, threads: int) -> None:
    import gc
    import json
    import time

    import torch

    from speechforge import corpora
    from speechforge.backends import styletts2
    from speechforge.synth import clone, load_backend

    styletts2.USE_MKLDNN = setting == "on"

    def rss_mb():
        return int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * 4096 / 2**20

    def live_tensors():
        return sum(1 for o in gc.get_objects() if torch.is_tensor(o))

    speakers = corpora.load("librispeech", 12, 5)
    utts = [(u[0], x) for u in speakers.values() for x in u[1:]][:n]
    b = load_backend("styletts2")
    b.set_threads(threads)
    gc.collect()
    load, t0_tensors = rss_mb(), live_tensors()
    marks, rss, wall, audio = [], [], 0.0, 0.0
    step = max(1, n // 6)
    for i, ((ref, sr, _), (_ho, _hsr, text)) in enumerate(utts, 1):
        t0 = time.time()
        wav, osr, _ = clone(b, text, ref, sr, seed=1234)
        wall += time.time() - t0
        audio += len(wav) / osr
        gc.collect()
        if i % step == 0 or i == n:
            marks.append(i)
            rss.append(rss_mb())
    print(json.dumps({"load_mb": load, "marks": marks, "rss_mb": rss,
                      "tensors": [t0_tensors, live_tensors()],
                      "rtf": wall / audio}))


# --- parity ---------------------------------------------------------------------

def check_parity(args) -> None:
    import numpy as np
    import torch

    repo = args.repo or os.environ.get("STYLETTS2_REPO")
    if not repo:
        raise SystemExit("pass --repo or set STYLETTS2_REPO")
    reference = args.reference or str(
        LIBRI / "1221" / "135767" / "1221-135767-0014.flac")
    torch.set_num_threads(args.threads)
    up = _Upstream(repo)

    from speechforge.backends.styletts2 import StyleTTS2Backend

    texts = [
        "StyleTTS 2 is a text to speech model that leverages style diffusion "
        "and adversarial training.",
        "HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND "
        "BRUISED POTATOES",
        'He said "quite so," and left.',
    ]
    configs = [dict(alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0),
               dict(alpha=0.0, beta=0.0, diffusion_steps=5, embedding_scale=1.0),
               dict(alpha=1.0, beta=1.0, diffusion_steps=10, embedding_scale=1.0)]
    print(f"reference: {reference}\n")
    print(f"{'setting':>18} {'text':>26}   {'shipped vs upstream (off)':>26}"
          f"   {'upstream on vs off':>22}")
    worst = 0.0
    for cfg in configs:
        mine_backend = StyleTTS2Backend(repo_dir=repo, **cfg)
        mine_backend.set_threads(args.threads)
        tag = f"a={cfg['alpha']} b={cfg['beta']} n={cfg['diffusion_steps']}"
        for text in texts:
            with torch.backends.mkldnn.flags(enabled=False):
                ref_off = up.run(text, reference, cfg, seed=args.seed)
            with torch.backends.mkldnn.flags(enabled=True):
                ref_on = up.run(text, reference, cfg, seed=args.seed)
            mine, _sr = mine_backend.synthesize(text, reference, seed=args.seed)

            def cmp(a, b):
                if len(a) != len(b):
                    return f"LENGTH {len(a)} vs {len(b)}", float("inf")
                d = float(np.max(np.abs(a - b)))
                return f"max|diff| {d:.3g}", d

            s1, d1 = cmp(mine, ref_off)
            s2, _ = cmp(ref_on, ref_off)
            worst = max(worst, d1)
            print(f"{tag:>18} {text[:24]!r:>26}   {s1:>26}   {s2:>22}")
        del mine_backend
    print(f"\nshipped backend vs upstream, same oneDNN setting: worst max|diff| = "
          f"{worst:.3g}" + ("  (bit-exact)" if worst == 0 else "  <- NOT bit-exact"))


class _Upstream:
    """Upstream's loaders and inference function, as close to verbatim as runs."""

    def __init__(self, repo: str):
        import phonemizer
        import torch
        import torchaudio
        import yaml
        from munch import Munch

        cwd = os.getcwd()
        os.chdir(repo)                 # upstream's config paths are repo-relative
        sys.path.insert(0, repo)
        try:
            import models as up_models
            from Modules.diffusion.sampler import (ADPM2Sampler, DiffusionSampler,
                                                   KarrasSchedule)
            from text_utils import TextCleaner
            from Utils.PLBERT.util import load_plbert

            def recursive_munch(d):
                if isinstance(d, dict):
                    return Munch((k, recursive_munch(v)) for k, v in d.items())
                if isinstance(d, list):
                    return [recursive_munch(v) for v in d]
                return d

            # The one deviation: upstream got weights_only=False by default on
            # torch < 2.6. The file's globals were disassembled before trusting it.
            orig = torch.load
            up_models.torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
            config = yaml.safe_load(open("Models/LibriTTS/config.yml"))
            aligner = up_models.load_ASR_models(config["ASR_path"], config["ASR_config"])
            up_models.torch.load = orig
            pitch = up_models.load_F0_models(config["F0_path"])
            bert = load_plbert(config["PLBERT_dir"])
            self.mp = recursive_munch(config["model_params"])
            model = up_models.build_model(self.mp, aligner, pitch, bert)
            params = torch.load("Models/LibriTTS/epochs_2nd_00020.pth",
                                map_location="cpu")["net"]
            for key in model:
                if key in params:
                    try:
                        model[key].load_state_dict(params[key])
                    except Exception:
                        from collections import OrderedDict
                        model[key].load_state_dict(OrderedDict(
                            (k[7:], v) for k, v in params[key].items()), strict=False)
            _ = [model[key].eval() for key in model]
        finally:
            os.chdir(cwd)
        self.model = model
        self.torch = torch
        self.cleaner = TextCleaner()
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True)
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion, sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    def compute_style(self, path):
        import librosa
        torch, model = self.torch, self.model
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        mel = self.to_mel(torch.from_numpy(audio).float())
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) - (-4)) / 4
        with torch.no_grad():
            ref_s = model.style_encoder(mel.unsqueeze(1))
            ref_p = model.predictor_encoder(mel.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def run(self, text, reference, cfg, seed):
        self.torch.manual_seed(seed)
        return self.inference(text, self.compute_style(reference), **cfg)

    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5,
                  embedding_scale=1):
        from nltk.tokenize import word_tokenize
        torch, model, device = self.torch, self.model, "cpu"

        def length_to_mask(lengths):
            mask = torch.arange(lengths.max()).unsqueeze(0).expand(
                lengths.shape[0], -1).type_as(lengths)
            return torch.gt(mask + 1, lengths.unsqueeze(1))

        text = text.strip()
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                                  embedding=bert_dur, embedding_scale=embedding_scale,
                                  features=ref_s, num_steps=diffusion_steps).squeeze(1)
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if self.mp.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if self.mp.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new
            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_memory_worker":
        _memory_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        return
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="check", required=True)
    sub.add_parser("phonemizer", help="CLI vs library, every test-clean transcript")
    pp = sub.add_parser("parity", help="bit-exact audio vs upstream inference")
    pp.add_argument("--repo", default=None, help="StyleTTS 2 clone (or STYLETTS2_REPO)")
    pp.add_argument("--reference", default=None, help="reference clip; default "
                    "is LibriSpeech 1221-135767-0014, one of upstream's demo clips")
    pp.add_argument("--seed", type=int, default=1234)
    pp.add_argument("--threads", type=int, default=4)
    pm = sub.add_parser("memory", help="RSS growth with oneDNN on vs off")
    pm.add_argument("--utterances", type=int, default=24)
    pm.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    print(f"study=styletts2_checks:{args.check}  hardware={hardware_label()}  "
          f"date={date.today().isoformat()}")
    print(f"cmd: python {shlex.join(sys.argv)}\n")
    {"phonemizer": check_phonemizer, "parity": check_parity,
     "memory": check_memory}[args.check](args)


if __name__ == "__main__":
    main()
