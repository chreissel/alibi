"""Diagnose whether the *achieved* strain SNR of a glitch matches its target.

Self-contained (no GWOSC): the strain background is replaced by synthetic
Gaussian noise. The achieved-vs-target SNR question is independent of whether the
background is real, so this faithfully exercises the glitch injection path
(``generate_glitch_sources`` -> ``bandlimit`` -> ``derive_witness`` ->
``reweight_snrs`` -> ``_inject_center`` -> ``whiten``) and compares it to the
signal path.

For each class we report:
  (a) reweight check  -- network SNR of the reweighted response BEFORE injection
                         (should equal the target; tests ``reweight_snrs`` itself)
  (b) end-to-end check -- inject into a ZERO kernel, whiten, and measure the
                         whitened root-sum-square. Whitened noise is ~unit
                         variance, so RSS ~ matched-filter SNR. This isolates the
                         transient's survival through inject -> whiten.
  (c) display check    -- gwpy Q-scan peak normalised energy (what the eye sees).
"""

import torch

from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_network_snr, reweight_snrs

from utils import load_config
from waveforms import generate_signals, generate_glitch_sources
from witness import derive_witness, bandlimit
from injections import _interp_psd, _inject_center

TARGET = 13.0


def _reweight(responses, target_snrs, psd, sample_rate, highpass):
    """Reweight to the target SNR and report the achieved network SNR."""
    responses = reweight_snrs(
        responses=responses, target_snrs=target_snrs, psd=psd,
        sample_rate=sample_rate, highpass=highpass,
    )
    achieved = compute_network_snr(responses, psd, sample_rate, highpass=highpass)
    return responses, achieved


def _summary(name, x):
    x = x.detach().cpu()
    return f"{name:28s} mean={x.mean():7.3f}  std={x.std():6.3f}  min={x.min():7.3f}  max={x.max():7.3f}"


def main():
    device = "cpu"
    config = load_config(config_path="configs/config_H1.yaml")

    sample_rate = config.general.sample_rate
    f_min = config.general.f_min
    f_max = config.general.f_max
    kernel_length = config.general.waveform_duration
    batch_size = config.general.batch_size

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
    num_freqs = kernel_size // 2 + 1

    torch.manual_seed(config.general.seed)

    # --- synthetic strain background (replaces GWOSC) -----------------------
    # At the real strain scale (~1e-21): CBC signals are generated at physical
    # amplitude, so a unit-variance background would underflow the float32 SNR
    # integrand in ml4gw's compute_ifo_snr.
    strain_bg = torch.randn(batch_size, 1, window_size, device=device) * 1e-21

    spectral_density = SpectralDensity(
        sample_rate=sample_rate, fftlength=fftlength, overlap=overlap, average=average
    ).to(device)
    whiten = Whiten(fduration=fduration, sample_rate=sample_rate, highpass=f_min).to(device)

    strain_psd = spectral_density(strain_bg[..., :psd_size].double())
    strain_kernel = strain_bg[..., psd_size:]
    psd_i = _interp_psd(strain_psd, num_freqs)

    target = torch.full((batch_size,), TARGET, device=device)

    def end_to_end_snr(response):
        """RSS of the whitened, noiseless (zeros-background) injection."""
        zero_kernel = torch.zeros_like(strain_kernel)
        injected = _inject_center(zero_kernel, response, kernel_size, pad)
        w = whiten(injected, strain_psd)
        return torch.sqrt((w**2).sum(dim=-1)).squeeze(1)

    print(f"target SNR = {TARGET}\n")

    # --- signal path (control) ---------------------------------------------
    sig, _ = generate_signals(config, device)
    sig, sig_a = _reweight(sig, target, psd_i, sample_rate, f_min)
    sig_b = end_to_end_snr(sig)
    print(_summary("signal (a) reweight SNR", sig_a))
    print(_summary("signal (b) whitened RSS", sig_b))

    # --- glitch path -------------------------------------------------------
    blip, gparams = generate_glitch_sources(config, device)
    blip_indep, _ = generate_glitch_sources(config, device)
    blip = bandlimit(blip, sample_rate, f_min, f_max)
    strain_g, _ = derive_witness(blip, blip_indep, sample_rate, config.witness.coupling)
    strain_g = strain_g.unsqueeze(1)
    # Match injections._inject_glitch: rescale the unit-RMS glitch to the background
    # scale so the float32 SNR integrand stays well conditioned.
    strain_g = strain_g * strain_kernel.std(dim=-1, keepdim=True)
    strain_g, gl_a = _reweight(strain_g, target, psd_i, sample_rate, f_min)
    gl_b = end_to_end_snr(strain_g)
    print(_summary("glitch (a) reweight SNR", gl_a))
    print(_summary("glitch (b) whitened RSS", gl_b))

    print(f"\n(b) ratio glitch/signal = {(gl_b.mean() / sig_b.mean()).item():.3f}")
    print(_summary("glitch quality", gparams["quality"]))
    print(_summary("glitch frequency", gparams["frequency"]))

    # --- (c) display: Q-scan peak for one matched example -------------------
    try:
        from gwpy.timeseries import TimeSeries

        def qpeak(response, idx):
            injected = _inject_center(strain_kernel.clone(), response, kernel_size, pad)
            w = whiten(injected, strain_psd)[idx, 0].detach().cpu().numpy()
            ts = TimeSeries(w, sample_rate=sample_rate)
            q = ts.q_transform(whiten=False, frange=(20, sample_rate / 2),
                               qrange=(4, 64), logf=True)
            return float(q.value.max())

        idx = int(torch.argmin((gl_b - TARGET).abs()))
        print(f"\n(c) Q-scan peak (idx={idx}):  signal={qpeak(sig, idx):8.1f}   "
              f"glitch={qpeak(strain_g, idx):8.1f}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(c) skipped: {e}")


if __name__ == "__main__":
    main()
