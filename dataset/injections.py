"""Build whitened, labelled 2-channel examples: [strain, witness].

For each class the strain channel uses real interferometer background; the witness
channel is synthesised. Signals are injected into strain only; glitches are coupled
into both strain and witness (see ``witness.py``); background is noise only.

Adapted from chreissel/GWDatasetGeneration, extended with a witness channel and a
glitch class.
"""

import importlib
from pathlib import Path

import torch
import torch.nn.functional as F

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.gw import compute_network_snr, reweight_snrs

from utils import load_config
from waveforms import generate_signals, generate_glitch_sources
from witness import make_witness_noise, couple_glitch
from glitches import GLITCH_CLASSES

LABELS = {"background": 0, "signal": 1, "glitch": 2}


def _sample_target_snr(snr_cfg, batch_size, device):
    func_path = snr_cfg.func
    module_name, func_name = func_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    args = getattr(snr_cfg, "args", [])
    return func(*args).sample((batch_size,)).to(device)


def _interp_psd(psd, num_freqs):
    if psd.shape[-1] != num_freqs:
        while psd.ndim < 3:
            psd = psd[None]
        psd = F.interpolate(psd, size=(num_freqs,), mode="linear")
    return psd


def _inject_center(kernel, response, kernel_size, pad):
    """Add ``response`` (batch, 1, time) into the central region of ``kernel``."""
    injected = kernel.detach().clone()
    injected[:, :, pad:-pad] += response[..., -kernel_size:]
    return injected


def _reweight_to_snr(responses, target_snrs, psd, sample_rate, highpass):
    """Reweight ``responses`` to ``target_snrs`` and return ``(responses, achieved)``.

    ``ml4gw.gw.reweight_snrs`` can miss the requested network SNR by a few percent
    for waveforms with power near the highpass (e.g. long CBC inspirals). One
    corrective rescale -- SNR is linear in amplitude -- makes the achieved SNR
    match the target exactly, so the stored label is trustworthy and signal vs
    glitch examples are strictly comparable at equal SNR.
    """
    responses = reweight_snrs(
        responses=responses, target_snrs=target_snrs, psd=psd,
        sample_rate=sample_rate, highpass=highpass,
    )
    achieved = compute_network_snr(responses, psd, sample_rate, highpass=highpass)
    responses = responses * (target_snrs / achieved).view(-1, 1, 1)
    achieved = compute_network_snr(responses, psd, sample_rate, highpass=highpass)
    return responses, achieved


def injection(config, data_dir: str, device: str, mode: str):
    """Generate one batch of whitened 2-channel data for the given class.

    Returns ``(data, params)`` where ``data`` has shape (batch, 2, T) with channel
    order [strain, witness], and ``params`` is a dict of tensors (or ``None``).
    """
    assert mode in LABELS, f"unknown mode {mode!r}"

    ifos = config.general.ifos
    assert len(ifos) == 1, "This toy dataset focuses on a single strain detector."

    batch_size = config.general.batch_size
    sample_rate = config.general.sample_rate
    f_min = config.general.f_min
    kernel_length = config.general.waveform_duration

    fduration = config.whiten.fduration
    fftlength = config.whiten.fftlength
    psd_length = config.whiten.psd_length
    overlap = config.whiten.overlap
    average = config.whiten.average

    psd_size = int(psd_length * sample_rate)
    kernel_size = int(kernel_length * sample_rate)
    pad = int(fduration / 2 * sample_rate)

    window_length = psd_length + fduration + kernel_length
    window_size = int(window_length * sample_rate)
    num_samples = int(kernel_length * sample_rate)
    num_freqs = num_samples // 2 + 1

    # ---- backgrounds -------------------------------------------------------
    fnames = list(Path(data_dir).iterdir())
    dataloader = Hdf5TimeSeriesDataset(
        fnames=fnames,
        channels=ifos,
        kernel_size=window_size,
        batch_size=batch_size,
        batches_per_epoch=1,
        coincident=False,
    )
    strain_bg = [x for x in dataloader][0].to(device)  # (batch, 1, window_size)
    witness_bg = make_witness_noise(
        batch_size, window_size, sample_rate, device, getattr(config.witness, "noise", None)
    )

    spectral_density = SpectralDensity(
        sample_rate=sample_rate, fftlength=fftlength, overlap=overlap, average=average
    ).to(device)
    whiten = Whiten(fduration=fduration, sample_rate=sample_rate, highpass=f_min).to(device)

    strain_psd = spectral_density(strain_bg[..., :psd_size].double())
    witness_psd = spectral_density(witness_bg[..., :psd_size].double())

    strain_kernel = strain_bg[..., psd_size:]
    witness_kernel = witness_bg[..., psd_size:]

    params = {}

    # ---- class-specific injection -----------------------------------------
    if mode == "signal":
        waveforms, params = generate_signals(config, device)
        target_snrs = _sample_target_snr(config.snr_reweighting, batch_size, device)
        psd_i = _interp_psd(strain_psd, num_freqs)
        waveforms, params["snr"] = _reweight_to_snr(
            waveforms, target_snrs, psd_i, sample_rate, f_min
        )
        strain_kernel = _inject_center(strain_kernel, waveforms, kernel_size, pad)

    elif mode == "glitch":
        src, params = generate_glitch_sources(config, device)
        src_indep, _ = generate_glitch_sources(config, device)
        strain_g, witness_g = couple_glitch(
            src, src_indep, sample_rate, config.witness.coupling,
            glitch_class=params.get("glitch_class"), class_names=GLITCH_CLASSES,
        )
        strain_g = strain_g.unsqueeze(1)
        witness_g = witness_g.unsqueeze(1)

        strain_psd_i = _interp_psd(strain_psd, num_freqs)
        witness_psd_i = _interp_psd(witness_psd, num_freqs)

        strain_snr = _sample_target_snr(config.glitch.snr.strain, batch_size, device)
        witness_snr = _sample_target_snr(config.glitch.snr.witness, batch_size, device)

        strain_g, strain_snr = _reweight_to_snr(
            strain_g, strain_snr, strain_psd_i, sample_rate, f_min
        )
        witness_g, witness_snr = _reweight_to_snr(
            witness_g, witness_snr, witness_psd_i, sample_rate, f_min
        )

        strain_kernel = _inject_center(strain_kernel, strain_g, kernel_size, pad)
        witness_kernel = _inject_center(witness_kernel, witness_g, kernel_size, pad)
        params["strain_snr"] = strain_snr
        params["witness_snr"] = witness_snr

    # mode == "background": inject nothing.

    # ---- whiten & stack ----------------------------------------------------
    strain_w = whiten(strain_kernel, strain_psd)
    witness_w = whiten(witness_kernel, witness_psd)
    data = torch.cat([strain_w, witness_w], dim=1)  # (batch, 2, T)

    params["label"] = torch.full((batch_size,), LABELS[mode], dtype=torch.long)
    return data, params


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path="configs/config_H1.yaml")
    background_dir = Path("./data") / "background_data"
    for m in ("background", "signal", "glitch"):
        d, p = injection(config, data_dir=background_dir, device=device, mode=m)
        print(m, d.shape)
