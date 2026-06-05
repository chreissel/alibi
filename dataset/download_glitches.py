"""Build the O3a glitch bank (Scattered_Light / Whistle / Power_Line) from GravitySpy + GWOSC.

Run this on a machine with network access to the GravitySpy catalogue and GWOSC
open data. It downloads real glitch morphologies into the HDF5 bank that
``config.glitch.gravityspy.bank_path`` points at; dataset generation
(``main.py``) then samples the strain glitch from that bank and synthesises the
witness from it (``witness.py``).

Examples
--------
    # fetch the GravitySpy catalogue over the network
    python download_glitches.py --config configs/config_H1.yaml

    # or use a locally downloaded GravitySpy catalogue (CSV/HDF5)
    python download_glitches.py --config configs/config_H1.yaml \
        --catalog ./gravityspy_O3a.csv

If you only need something to develop against offline, build a synthetic bank:
    python download_glitches.py --config configs/config_H1.yaml --synthetic
"""

import argparse
from pathlib import Path

from utils import load_config
from glitches import build_real_glitch_bank, make_synthetic_glitch_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/config_H1.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output bank path (overrides config)")
    parser.add_argument("--catalog", type=str, default=None, help="Local GravitySpy catalogue (CSV/HDF5)")
    parser.add_argument("--synthetic", action="store_true", help="Build an offline synthetic bank instead")
    args = parser.parse_args()

    config = load_config(config_path=args.config)
    gs = config.glitch.gravityspy
    out = Path(args.out or gs.bank_path)
    sample_rate = config.general.sample_rate
    duration = config.general.waveform_duration
    right_pad = config.general.right_pad

    if args.synthetic:
        path = make_synthetic_glitch_bank(
            out, sample_rate=sample_rate, duration=duration, right_pad=right_pad,
            classes=gs.classes,
        )
        print(f"Wrote synthetic glitch bank -> {path}")
        return

    catalog_path = args.catalog or getattr(gs, "catalog_path", None)
    print(f"Building real O3a glitch bank {out} (classes={list(gs.classes)}) ...")
    path = build_real_glitch_bank(
        out_path=out,
        ifo=gs.ifo,
        run=gs.run,
        classes=list(gs.classes),
        min_confidence=gs.min_confidence,
        snr_range=gs.snr_range,
        sample_rate=sample_rate,
        duration=duration,
        right_pad=right_pad,
        f_min=config.general.f_min,
        f_max=config.general.f_max,
        max_per_class=gs.max_per_class,
        catalog_path=catalog_path,
    )
    print(f"Wrote real glitch bank -> {path}")


if __name__ == "__main__":
    main()
