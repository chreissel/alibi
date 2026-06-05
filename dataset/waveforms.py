"""Waveform generation.

Two kinds of transients are produced here, both with ml4gw:

* ``generate_signals`` -- astrophysical CBC (BBH) signals, projected onto the
  detector(s). Adapted from chreissel/GWDatasetGeneration.
* ``generate_glitch_sources`` -- the glitch transient that goes into the strain.
  By default these are *real* O3a Blip/Koi_Fish/Tomte morphologies sampled from a
  glitch bank (``glitches.py``); a SineGaussian fallback is also available. The
  returned single-channel g(t) is the strain glitch, and the witness is
  synthesised *from* it in ``witness.py``.
"""

import importlib

import torch
from torch.distributions import Uniform

from ml4gw.distributions import Cosine
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from ml4gw.waveforms.adhoc import SineGaussian
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.gw import get_ifo_geometry, compute_observed_strain

from utils import load_config


def _sample_from_block(block, batch_size, device):
    """Sample a dict of parameters from a config block of {name: {func, args}}.

    ``args`` entries that are strings refer to previously sampled parameters
    (matching the behaviour of the reference repository).
    """
    param_dict = {}
    params = {}
    attrs = [x for x in dir(block) if not x.startswith("__")]
    for k in attrs:
        attrs_config = getattr(block, k)
        func_path = getattr(attrs_config, "func")

        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        if "args" in dir(attrs_config):
            args = getattr(attrs_config, "args")
            args = [a if not isinstance(a, str) else params[a] for a in args]
            param_dict[k] = func(*args)
        else:
            param_dict[k] = func()

        if k == "mass_2":
            params[k] = param_dict[k].sample().to(device)
        else:
            params[k] = param_dict[k].sample((batch_size,)).to(device)
    return params


def generate_signals(config, device: str):
    """Generate detector-projected CBC signals of shape (batch, n_ifos, time)."""

    waveform_duration = config.general.waveform_duration
    batch_size = config.general.batch_size
    sample_rate = config.general.sample_rate
    ifos = config.general.ifos
    f_min = config.general.f_min
    f_ref = config.general.f_ref
    right_pad = config.general.right_pad

    params = _sample_from_block(config.waveform, batch_size, device)

    if getattr(config.general, "type", "BBH") == "BNS":
        approximant = TaylorF2().to(device)
        q = params["mass_2"] / params["mass_1"]
        params["chirp_mass"] = (q / (1 + q) ** 2) ** (3 / 5.0) * (
            params["mass_2"] + params["mass_1"]
        )
        params["mass_ratio"] = q
        params["chi1"], params["chi2"] = params["s1z"], params["s2z"]
    else:
        approximant = IMRPhenomD().to(device)
        params["mass_1"], params["mass_2"] = chirp_mass_and_mass_ratio_to_components(
            params["chirp_mass"], params["mass_ratio"]
        )
        params["s1z"], params["s2z"] = params["chi1"], params["chi2"]

    waveform_generator = TimeDomainCBCWaveformGenerator(
        approximant=approximant,
        sample_rate=sample_rate,
        f_min=f_min,
        duration=waveform_duration,
        right_pad=right_pad,
        f_ref=f_ref,
    ).to(device)

    hc, hp = waveform_generator(**params)

    # Extrinsic parameters / sky projection.
    params["dec"] = Cosine().sample((batch_size,)).to(device)
    params["psi"] = Uniform(0, torch.pi).sample((batch_size,)).to(device)
    params["phi"] = Uniform(-torch.pi, torch.pi).sample((batch_size,)).to(device)

    tensors, vertices = get_ifo_geometry(*ifos)

    waveforms = compute_observed_strain(
        dec=params["dec"],
        psi=params["psi"],
        phi=params["phi"],
        detector_tensors=tensors.to(device),
        detector_vertices=vertices.to(device),
        sample_rate=sample_rate,
        cross=hc,
        plus=hp,
    )
    return waveforms, params


def generate_glitch_sources(config, device: str):
    """Generate single-channel strain-domain glitch sources g(t) of shape (batch, time).

    Selected by ``config.glitch.source``:

    * ``gravityspy`` (default) -- real O3a Blip/Koi_Fish/Tomte morphologies sampled
      from the glitch bank at ``config.glitch.gravityspy.bank_path`` (built by
      ``download_glitches.py`` or, offline, by ``glitches.make_synthetic_glitch_bank``).
    * ``sine_gaussian`` -- ad-hoc SineGaussian bursts from ``config.glitch.prior``.

    The witness is later synthesised *from* this strain glitch (see ``witness.py``).
    """
    source_type = getattr(config.glitch, "source", "sine_gaussian")
    if source_type == "gravityspy":
        from glitches import sample_glitch_sources

        return sample_glitch_sources(
            config.glitch.gravityspy.bank_path, config.general.batch_size, device
        )
    if source_type == "sine_gaussian":
        return _generate_sine_gaussian_sources(config, device)
    raise ValueError(
        f"Unknown glitch source {source_type!r} (use 'gravityspy' or 'sine_gaussian')."
    )


def _generate_sine_gaussian_sources(config, device: str):
    """Ad-hoc SineGaussian glitch sources g(t) of shape (batch, time).

    The waveform is intentionally returned un-normalised (hrss is sampled but the
    final amplitude is set by SNR reweighting in the injection step).
    """

    waveform_duration = config.general.waveform_duration
    batch_size = config.general.batch_size
    sample_rate = config.general.sample_rate
    right_pad = config.general.right_pad

    params = _sample_from_block(config.glitch.prior, batch_size, device)

    sine_gaussian = SineGaussian(
        sample_rate=sample_rate, duration=waveform_duration
    ).to(device)

    cross, plus = sine_gaussian(
        quality=params["quality"],
        frequency=params["frequency"],
        hrss=params["hrss"],
        phase=params["phase"],
        eccentricity=params["eccentricity"],
    )
    # Use the plus polarisation as the scalar glitch source; a glitch is a
    # terrestrial transient, so there is no physical second polarisation.
    glitch_source = plus

    # SineGaussian is centred at duration/2. Shift its peak to ``right_pad`` from
    # the right edge so it lands at the SAME time location as the CBC coalescence
    # point. This removes a timing confound: signal vs glitch must be told apart by
    # the witness, not by where the transient sits in the window.
    shift = int((waveform_duration / 2 - right_pad) * sample_rate)
    glitch_source = torch.roll(glitch_source, shifts=shift, dims=-1)
    return glitch_source, params


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path="configs/config_H1.yaml")
    generate_signals(config, device=device)
    generate_glitch_sources(config, device=device)
