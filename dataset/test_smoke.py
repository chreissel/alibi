"""Offline smoke test for the witness-channel toy dataset generator.

Creates a synthetic strain background and a synthetic glitch bank (so no GWOSC or
GravitySpy download is needed), runs the injection pipeline for all three classes,
and checks:
  * output shape (batch, 2, T) and finiteness,
  * the witness asymmetry: for glitches the witness (synthesised from the strain
    glitch) is correlated with the strain transient, while for signals it is not.

Run with:  python test_smoke.py
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from utils import load_config
from injections import injection
from glitches import make_synthetic_glitch_bank


def _make_fake_background(bg_dir, sample_rate, seconds=240):
    bg_dir.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    # Unit-variance coloured-ish noise is unnecessary; white is fine for a smoke test.
    data = np.random.randn(n).astype(np.float64)
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

        # Offline synthetic glitch bank so the default 'gravityspy' source works
        # without network access.
        bank = make_synthetic_glitch_bank(
            Path(tmp) / "glitch_bank.h5",
            sample_rate=config.general.sample_rate,
            duration=config.general.waveform_duration,
            right_pad=config.general.right_pad,
            classes=config.glitch.gravityspy.classes,
            n_per_class=8,
        )
        config.glitch.gravityspy.bank_path = str(bank)

        results = {}
        for mode in ("background", "signal", "glitch"):
            data, params = injection(config, data_dir=bg_dir, device=device, mode=mode)
            arr = data.detach().cpu().numpy()
            assert arr.ndim == 3 and arr.shape[1] == 2, f"{mode}: bad shape {arr.shape}"
            assert np.isfinite(arr).all(), f"{mode}: non-finite values"
            results[mode] = _transient_corr(
                arr, config.general.sample_rate, config.general.right_pad
            )
            print(f"{mode:<10} shape={arr.shape}  |strain-witness corr|={results[mode]:.3f}  "
                  f"params={sorted(params.keys())}")

        # Asymmetry: glitches couple into the witness, signals do not.
        assert results["glitch"] > results["signal"], (
            f"expected glitch corr ({results['glitch']:.3f}) > "
            f"signal corr ({results['signal']:.3f})"
        )
        print("\nOK: witness is correlated for glitches and not for signals.")


if __name__ == "__main__":
    main()
