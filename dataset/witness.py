"""Witness-channel synthesis and glitch coupling.

GWOSC open data only contains the strain channel, so the witness channel is
synthesised here. The physical story this models:

* The **strain** carries the *real* glitch transient g(t) (a blip / koi-fish /
  tomte morphology drawn from the O3a glitch bank, see ``glitches.py``).
* A witness (auxiliary) sensor would record a copy of that disturbance reaching it
  through an imperfect, frequency-dependent path. We model that path as a linear
  time-invariant (LTI) Butterworth filter ``C`` applied **to the strain glitch**:
  ``witness = C(strain_glitch)``. The witness is therefore built *from* the strain
  glitch, not the other way around.
* The coupling is partial: only a fraction ``alpha`` of the witness power is
  coherent with the strain glitch; the remaining ``1 - alpha`` is an independent
  transient. ``alpha`` therefore sets the strain<->witness coherence and is the
  single "how useful is the witness" knob.

Crucially, *astrophysical* signals do not couple to a witness, so for the signal
and background classes the witness carries noise only. Only the glitch class
injects a correlated transient into the witness.
"""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt


def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalise each row of (batch, time) to unit RMS."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-30
    return x / rms


def make_witness_noise(batch_size, length, sample_rate, device, noise_cfg=None):
    """Synthesise a raw witness background of shape (batch, 1, length).

    ``noise_cfg.color`` selects the spectral shape:
      * ``white`` (default): flat ASD.
      * ``powerlaw``: ASD proportional to ``f ** index`` (with a low-frequency
        floor), giving e.g. red noise for a negative index.
    """
    color = getattr(noise_cfg, "color", "white") if noise_cfg is not None else "white"
    white = torch.randn(batch_size, 1, length, device=device)

    if color == "white":
        return white

    index = float(getattr(noise_cfg, "index", -1.0))
    num_freqs = length // 2 + 1
    freqs = torch.fft.rfftfreq(length, d=1.0 / sample_rate).to(device)
    asd = torch.ones(num_freqs, device=device)
    nonzero = freqs > 0
    asd[nonzero] = freqs[nonzero] ** index
    asd[~nonzero] = asd[nonzero][0] if nonzero.any() else 1.0

    spectrum = torch.fft.rfft(white, dim=-1) * asd
    colored = torch.fft.irfft(spectrum, n=length, dim=-1)
    return _rms_normalize(colored.squeeze(1)).unsqueeze(1)


def _butter_filter(x: torch.Tensor, sample_rate, filt_cfg) -> torch.Tensor:
    """Apply a zero-phase Butterworth coupling filter along the time axis."""
    btype = getattr(filt_cfg, "btype", "bandpass")
    order = int(getattr(filt_cfg, "order", 4))
    cutoff = getattr(filt_cfg, "cutoff", [20.0, 400.0])
    if isinstance(cutoff, (list, tuple)):
        wn = [c / (sample_rate / 2) for c in cutoff]
    else:
        wn = cutoff / (sample_rate / 2)

    sos = butter(order, wn, btype=btype, output="sos")
    arr = x.detach().cpu().numpy().astype(np.float64)
    filtered = sosfiltfilt(sos, arr, axis=-1).copy()
    return torch.from_numpy(filtered).to(x.device, x.dtype)


def couple_glitch(strain_glitch, strain_glitch_indep, sample_rate, coupling_cfg):
    """Synthesise the witness *from* the (real) strain glitch via an LTI filter.

    The strain carries the real glitch; the witness is the linearly-coupled copy
    an auxiliary sensor would record, ``witness = C(strain_glitch)``.

    Parameters
    ----------
    strain_glitch : (batch, time) tensor
        The real glitch as it appears in the strain channel.
    strain_glitch_indep : (batch, time) tensor
        An independent glitch realisation used for the witness-only (incoherent)
        component when ``alpha < 1``.
    coupling_cfg : namespace
        ``type`` (only ``lti`` supported), ``filter`` (btype/cutoff/order) and
        ``alpha`` (coherence fraction in [0, 1]; 1 -> witness fully coherent with
        the strain glitch).

    Returns
    -------
    strain_glitch, witness_glitch : (batch, time) tensors
        Unit-RMS-normalised; absolute amplitude is set later via SNR reweighting.
    """
    coupling_type = getattr(coupling_cfg, "type", "lti")
    if coupling_type != "lti":
        raise ValueError(f"Unsupported coupling type: {coupling_type!r} (only 'lti').")

    alpha = float(getattr(coupling_cfg, "alpha", 0.8))
    alpha = min(max(alpha, 0.0), 1.0)

    coupled = _rms_normalize(_butter_filter(strain_glitch, sample_rate, coupling_cfg.filter))
    indep = _rms_normalize(_butter_filter(strain_glitch_indep, sample_rate, coupling_cfg.filter))

    # Power-weighted mix so that the fraction of witness power coherent with the
    # strain glitch is ~alpha.
    witness_glitch = (alpha**0.5) * coupled + ((1.0 - alpha) ** 0.5) * indep

    return _rms_normalize(strain_glitch), _rms_normalize(witness_glitch)
