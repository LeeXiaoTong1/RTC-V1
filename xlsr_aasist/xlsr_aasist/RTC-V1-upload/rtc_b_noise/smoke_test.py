"""CPU checks for noise mixing, deterministic validation and source patching."""
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.io import wavfile
from env_noise import NoiseAugment, active_mask, mix_at_snr
from evaluate_b import binary_f1, utterance_seed
from install_b import patched_text


def main():
    sr = 16000
    rng = np.random.RandomState(42)
    x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(64600) / sr)).astype(np.float32)
    x[:8000] = 0
    n = rng.normal(0, 0.2, len(x)).astype(np.float32)
    for snr in (5, 10, 20, 30):
        mixed, info = mix_at_snr(x, n, sr, snr)
        mask = active_mask(x, sr) & active_mask(n, sr)
        residual = mixed.astype(np.float64) / info["gain"] - x
        measured = 10 * np.log10(np.mean(x[mask] ** 2) / np.mean(residual[mask] ** 2))
        assert abs(measured - snr) < 1e-3, (measured, snr)
        assert mixed.shape == x.shape and mixed.dtype == np.float32
        assert np.max(np.abs(mixed)) <= 0.990001
    loud, info = mix_at_snr(x * 5, n, sr, 10)
    assert info["gain"] < 1 and np.max(np.abs(loud)) <= 0.990001
    silent, info = mix_at_snr(np.zeros_like(x), n, sr, 10)
    assert not info["applied"] and not silent.any()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        wavfile.write(root / "long.wav", sr, (n * 30000).astype(np.int16))
        wavfile.write(root / "short.wav", sr, (n[:2400] * 30000).astype(np.int16))
        manifest = root / "dev.jsonl"
        manifest.write_text("".join(json.dumps({"path": str(root / name), "split": "dev"}) + "\n" for name in ("long.wav", "short.wav")))
        off = NoiseAugment(manifest, probability=0)
        assert off(x, sr) is x
        aug = NoiseAugment(manifest, probability=1)
        r1 = aug(x, sr, rng=np.random.RandomState(111), snr_db=20)
        r2 = aug(x, sr, rng=np.random.RandomState(111), snr_db=20)
        np.testing.assert_array_equal(r1, r2)
        assert not np.array_equal(x, r1)
        for seed in range(20):
            output = aug(x, sr, rng=np.random.RandomState(seed))
            assert output.shape == x.shape and np.isfinite(output).all()
    assert binary_f1([0, 0, 1, 1], [0, 0, 1, 1])["macro_f1"] == 100
    assert binary_f1([0, 0, 1, 1], [0, 1, 0, 1])["macro_f1"] == 50
    assert utterance_seed(1, "a") == utterance_seed(1, "a")
    assert utterance_seed(1, "a") != utterance_seed(1, "b")
    source = "class Dataset:\n    def __init__(self):\n        self.cut = 64600\n    def get(self):\n        audio = Tensor(pad_audio(audio, self.cut))\n"
    once = patched_text(source)
    assert patched_text(once) == once
    try:
        patched_text("unsupported = True\n")
    except ValueError:
        pass
    else:
        raise AssertionError("Installer must reject an unknown source layout")
    print("PASS: SNR, clipping prevention, silence, short events, deterministic noise, F1 and patch guards.")
    print("These checks do not train or validate XLS-R + AASIST on competition data.")


if __name__ == "__main__":
    main()
