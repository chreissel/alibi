"""Offline smoke test for the witness-channel toy dataset generator.

Creates a synthetic strain background (so no GWOSC download is needed), runs the
injection pipeline for all three classes, and checks:
  * output shape (batch, 2, T) and finiteness,
  * the witness asymmetry: for glitches the witness is correlated with the strain
    transient, while for signals it is not.

Run with:  python test_smoke.py
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from utils import load_config
from injections import injection


def _make_fake_background(bg_dir, sample_rate, seconds=240):
    bg_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    # White noise is fine for a smoke test, but it must sit at the real strain scale
    # (~1e-21): CBC signals are generated at physical amplitude, so a unit-variance
    # background would mismatch them by ~21 orders of magnitude and make the SNR
    # integrand |h|^2 / PSD underflow float32 inside ml4gw's compute_ifo_snr.
    data = (np.random.randn(n) * 1e-21).astype(np.float64)
    with h5py.File(bg_dir / "background-0-240.hdf5", "w") as f:
        f.create_dataset("H1", data=data)


def _transient_corr(data, sample_rate, right_pad):
    """Correlation between strain and witness around the coalescence location."""
    strain = data[:, 0]
    witness = data[:, 1]
    t = strain.shape[-1]
    center = t - int(right_pad * sample_rate)
    half = int(0.5 * sample_rate)
    lo, hi = max(0, center - half), min(t, center + half)
    s = strain[:, lo:hi]
    w = witness[:, lo:hi]
    s = (s - s.mean(1, keepdims=True))
    w = (w - w.mean(1, keepdims=True))
    num = (s * w).sum(1)
    den = np.sqrt((s**2).sum(1) * (w**2).sum(1)) + 1e-12
    return np.abs(num / den).mean()


def main():
    device = "cpu"
    config = load_config("configs/config_H1.yaml")
    # Shrink for a fast test.
    config.general.num_per_class = 8
    config.general.batch_size = 8

    with tempfile.TemporaryDirectory() as tmp:
        bg_dir = Path(tmp) / "background_data"
        _make_fake_background(bg_dir, config.general.sample_rate)

        results = {}
        for mode in ("background", "signal", "glitch", "signal_glitch"):
            data, params = injection(config, data_dir=bg_dir, device=device, mode=mode)
            arr = data.detach().cpu().numpy()
            assert arr.ndim == 3 and arr.shape[1] == 2, f"{mode}: bad shape {arr.shape}"
            assert np.isfinite(arr).all(), f"{mode}: non-finite values"
            results[mode] = _transient_corr(
                arr, config.general.sample_rate, config.general.right_pad
            )
            print(f"{mode:<14} shape={arr.shape}  |strain-witness corr|={results[mode]:.3f}  "
                  f"params={sorted(params.keys())}")

        # Asymmetry: glitches couple into the witness, signals do not. The combined
        # signal+glitch class also couples (the blip is present), unlike pure signals.
        assert results["glitch"] > results["signal"], (
            f"expected glitch corr ({results['glitch']:.3f}) > "
            f"signal corr ({results['signal']:.3f})"
        )
        assert results["signal_glitch"] > results["signal"], (
            f"expected signal_glitch corr ({results['signal_glitch']:.3f}) > "
            f"signal corr ({results['signal']:.3f})"
        )
        print("\nOK: witness is correlated for glitches (incl. signal+glitch) and not "
              "for signals.")


if __name__ == "__main__":
    main()
