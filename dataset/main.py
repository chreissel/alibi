"""Generate the witness-channel toy dataset.

Produces one HDF5 file per class (background.h5, signal.h5, glitch.h5,
signal_glitch.h5). Each file contains:
  * ``data``  : (N, 2, T) float array, channel order [strain, witness] (whitened)
  * ``label`` : (N,) int   (0=background, 1=signal, 2=glitch, 3=signal_glitch)
  * one dataset per sampled parameter (e.g. ``snr``, ``chirp_mass``, glitch params)
"""

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils import load_config
from injections import injection, LABELS
from set_seed import set_seed


def generate_class(config, data_dir, out_dir, device, mode):
    num_target = config.general.num_per_class
    batch_size = config.general.batch_size

    data_chunks = []
    param_chunks = defaultdict(list)

    total = 0
    with tqdm(total=num_target, desc=f"{mode:<10}", unit="ex") as pbar:
        while total < num_target:
            data, params = injection(config, data_dir=data_dir, device=device, mode=mode)
            data_chunks.append(data.detach().cpu().numpy())
            for k, v in params.items():
                arr = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                param_chunks[k].append(np.atleast_1d(arr))

            del data, params
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            total += batch_size
            pbar.update(batch_size)

    data = np.concatenate(data_chunks, axis=0)[:num_target]
    with h5py.File(out_dir / f"{mode}.h5", "w") as h5f:
        h5f.create_dataset("data", data=data)
        for k, chunks in param_chunks.items():
            merged = np.concatenate(chunks, axis=0)[:num_target]
            h5f.create_dataset(k, data=merged)
        h5f.attrs["label"] = LABELS[mode]
        h5f.attrs["channels"] = np.array(["strain", "witness"], dtype="S")


def main(config_path: str, data_dir: str, output_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path)

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(getattr(config.general, "seed", 42))

    for mode in LABELS:
        generate_class(config, data_dir, out_dir, device, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Witness-channel toy dataset generation for GW data"
    )
    parser.add_argument("--config", type=str, default="configs/config_H1.yaml")
    parser.add_argument("--data", type=str, help="Folder with background_data/")
    parser.add_argument("--out", type=str, help="Output folder for .h5 files")
    args = parser.parse_args()

    main(config_path=args.config, data_dir=args.data, output_dir=args.out)
