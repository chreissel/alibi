"""Real glitch bank: O3a Scattered_Light / Whistle / Power_Line morphologies.

These classes are chosen because they are *genuinely witnessed* by auxiliary
channels in real interferometers -- scattered light by seismic/length-sensing
channels, whistles by RF/PSL channels, power-line glitches by mains-voltage and
magnetometer monitors -- so synthesising a witness for them is physically
motivated (unlike blips/koi-fish/tomtes, which have no reliable aux witness).

In this dataset the **strain** carries the *real* glitch transient and the
**witness** is synthesised *from* the strain glitch by an LTI coupling
(see ``witness.py``). This module builds and samples a "glitch bank" of
strain-domain glitch morphologies, one waveform per real GravitySpy trigger.

Two entry points:

* :func:`build_real_glitch_bank` -- download real glitches from GWOSC using
  GravitySpy trigger times (needs network access to the GravitySpy catalogue and
  GWOSC). Run it on a data-enabled machine via ``download_glitches.py``.
* :func:`make_synthetic_glitch_bank` -- crude offline stand-ins (scattered-light/
  whistle/power-line-like) so the loader -> witness -> injection path is testable
  without any network access.

Either way the bank is an HDF5 file with:
  ``source``          (N, T) float  -- RMS-normalised strain-domain glitch g(t),
                                       peak aligned to ``right_pad`` (same time
                                       location as a CBC coalescence),
  ``glitch_class``    (N,)   int    -- 0=Scattered_Light, 1=Whistle, 2=Power_Line,
  ``gravityspy_snr``  (N,)   float  -- catalogue SNR (nan for synthetic),
  ``peak_frequency``  (N,)   float  -- catalogue peak frequency [Hz],
  ``gps``             (N,)   float  -- trigger GPS time (0 for synthetic).
"""

from pathlib import Path

import h5py
import numpy as np
import torch

GLITCH_CLASSES = ["Scattered_Light", "Whistle", "Power_Line"]
CLASS_TO_ID = {c: i for i, c in enumerate(GLITCH_CLASSES)}

_BANK_CACHE: dict = {}


def _rms_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.sqrt(np.mean(x**2)) + 1e-30)


def _align_peak(wave: np.ndarray, sample_rate: int, right_pad: float) -> np.ndarray:
    """Roll ``wave`` so its |peak| sits ``right_pad`` seconds from the right edge."""
    target = wave.shape[-1] - int(right_pad * sample_rate)
    peak = int(np.argmax(np.abs(wave)))
    return np.roll(wave, target - peak)


# ---------------------------------------------------------------------------
# Synthetic bank (offline; for testing without GWOSC/GravitySpy access)
# ---------------------------------------------------------------------------
def _synthetic_morphology(cls: str, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A crude scattered-light / whistle / power-line-like waveform centred at t=0.

    These are *not* physically faithful -- they only need distinct, plausible
    morphologies so the pipeline can be exercised offline. The real bank built by
    :func:`build_real_glitch_bank` replaces them with genuine O3a glitches.
    """
    if cls == "Scattered_Light":  # long, low-frequency stacked arches (seismic-driven)
        env = np.exp(-(t**2) / (2 * 0.12**2))
        wave = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
                   for f in (18.0, 32.0, 46.0))
        return env * wave
    if cls == "Whistle":  # high-frequency swept track (RF beat note)
        env = np.exp(-(t**2) / (2 * 0.05**2))
        f0, k = 700.0, 3000.0
        return env * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t) + rng.uniform(0, 2 * np.pi))
    if cls == "Power_Line":  # narrowband 60 Hz mains burst (+harmonic)
        env = np.exp(-(t**2) / (2 * 0.08**2))
        ph = rng.uniform(0, 2 * np.pi)
        return env * (np.sin(2 * np.pi * 60.0 * t + ph) + 0.3 * np.sin(2 * np.pi * 120.0 * t))
    raise ValueError(f"unknown synthetic class {cls!r}")


def make_synthetic_glitch_bank(
    out_path, sample_rate, duration, right_pad, classes=None, n_per_class=64, seed=0
):
    """Write a small offline glitch bank of synthetic scattered-light/whistle/power-line shapes."""
    classes = list(classes) if classes is not None else list(GLITCH_CLASSES)
    rng = np.random.default_rng(seed)
    size = int(duration * sample_rate)
    t = (np.arange(size) - size / 2) / sample_rate

    sources, labels, snrs, freqs, gps = [], [], [], [], []
    for cls in classes:
        for _ in range(n_per_class):
            wave = _synthetic_morphology(cls, t, rng)
            wave = _align_peak(_rms_normalize(wave.astype(np.float64)), sample_rate, right_pad)
            sources.append(wave)
            labels.append(CLASS_TO_ID[cls])
            snrs.append(np.nan)
            freqs.append(np.nan)
            gps.append(0.0)

    _write_bank(out_path, sources, labels, snrs, freqs, gps)
    return Path(out_path)


def _write_bank(out_path, sources, labels, snrs, freqs, gps):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("source", data=np.asarray(sources, dtype=np.float32))
        f.create_dataset("glitch_class", data=np.asarray(labels, dtype=np.int64))
        f.create_dataset("gravityspy_snr", data=np.asarray(snrs, dtype=np.float32))
        f.create_dataset("peak_frequency", data=np.asarray(freqs, dtype=np.float32))
        f.create_dataset("gps", data=np.asarray(gps, dtype=np.float64))
        f.attrs["classes"] = np.array(GLITCH_CLASSES, dtype="S")


# ---------------------------------------------------------------------------
# Real bank (online; GravitySpy triggers + GWOSC strain)
# ---------------------------------------------------------------------------
def _load_gravityspy_catalog(ifo, run, classes, min_confidence, snr_range, catalog_path):
    """Return rows of (gps, label, snr, peak_frequency) for the requested glitches.

    ``catalog_path`` (CSV/HDF5 with columns ``ifo,label,ml_confidence,snr,
    peak_frequency,event_time``/``GPStime``) is preferred and is what the offline
    tests exercise. If it is ``None`` we fall back to fetching the GravitySpy
    table over the network via ``gwpy`` (only works on a data-enabled host).
    """
    from astropy.table import Table

    if catalog_path is not None:
        tab = Table.read(catalog_path)
    else:  # network path -- not reachable in restricted environments
        from gwpy.table import GravitySpyTable

        tab = GravitySpyTable.fetch(
            "gravityspy", "glitches_v2d0",
            selection=[f"ifo={ifo}", f"ml_confidence>={min_confidence}"],
        )

    def _col(row, *names, default=None):
        for n in names:
            if n in row.colnames:
                return row[n]
        return default

    rows = []
    for row in tab:
        label = str(_col(row, "label", "ml_label", default=""))
        if label not in classes:
            continue
        if str(_col(row, "ifo", default=ifo)) != ifo:
            continue
        conf = float(_col(row, "ml_confidence", "confidence", default=1.0))
        snr = float(_col(row, "snr", default=np.nan))
        if conf < min_confidence:
            continue
        if not (snr_range[0] <= snr <= snr_range[1]):
            continue
        gps = float(_col(row, "event_time", "GPStime", "peak_time", default=np.nan))
        if not np.isfinite(gps):
            continue
        rows.append((gps, label, snr, float(_col(row, "peak_frequency", default=np.nan))))
    return rows


def build_real_glitch_bank(
    out_path, ifo, run, classes, min_confidence, snr_range, sample_rate, duration,
    right_pad, f_min, f_max, max_per_class, catalog_path=None, fetch_pad=16.0,
):
    """Download real glitch morphologies from GWOSC into a glitch bank.

    For each selected GravitySpy trigger we fetch a ``2*fetch_pad`` s strain
    segment from GWOSC, whiten it, band-pass to ``[f_min, f_max]``, crop a
    ``duration`` s window centred on the trigger, align the peak to ``right_pad``
    and RMS-normalise. Requires network access to GravitySpy + GWOSC.
    """
    from gwpy.timeseries import TimeSeries

    rows = _load_gravityspy_catalog(
        ifo, run, classes, min_confidence, snr_range, catalog_path
    )

    per_class_count = {c: 0 for c in classes}
    sources, labels, snrs, freqs, gps_out = [], [], [], [], []
    half = duration / 2.0

    for gps, label, snr, peak_freq in sorted(rows, key=lambda r: r[0]):
        if per_class_count[label] >= max_per_class:
            continue
        try:
            ts = TimeSeries.fetch_open_data(
                ifo, gps - fetch_pad, gps + fetch_pad, sample_rate=sample_rate,
                cache=True,
            )
            w = ts.whiten().bandpass(f_min, f_max)
            seg = w.crop(gps - half, gps + half).value.astype(np.float64)
        except Exception as exc:  # noqa: BLE001 -- skip unavailable/short segments
            print(f"  skip {label} @ {gps:.3f}: {type(exc).__name__}: {exc}")
            continue
        size = int(duration * sample_rate)
        if seg.shape[-1] < size:
            continue
        seg = _align_peak(_rms_normalize(seg[:size]), sample_rate, right_pad)
        sources.append(seg)
        labels.append(CLASS_TO_ID[label])
        snrs.append(snr)
        freqs.append(peak_freq)
        gps_out.append(gps)
        per_class_count[label] += 1
        if all(per_class_count[c] >= max_per_class for c in classes):
            break

    if not sources:
        raise RuntimeError(
            "No glitches were collected. Check catalogue access, the class names, "
            "and the confidence/SNR cuts."
        )
    print(f"  collected: {per_class_count}")
    _write_bank(out_path, sources, labels, snrs, freqs, gps_out)
    return Path(out_path)


# ---------------------------------------------------------------------------
# Loading / sampling at generation time
# ---------------------------------------------------------------------------
def load_glitch_bank(path):
    """Load a glitch bank HDF5 into a dict of tensors (cached by path)."""
    path = str(path)
    if path not in _BANK_CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Glitch bank {path!r} not found. Build it first with "
                "`python download_glitches.py` (real O3a glitches) or "
                "`glitches.make_synthetic_glitch_bank` (offline stand-ins)."
            )
        with h5py.File(path, "r") as f:
            _BANK_CACHE[path] = {
                "source": torch.from_numpy(f["source"][:]).float(),
                "glitch_class": torch.from_numpy(f["glitch_class"][:]).long(),
                "gravityspy_snr": torch.from_numpy(f["gravityspy_snr"][:]).float(),
                "peak_frequency": torch.from_numpy(f["peak_frequency"][:]).float(),
            }
    return _BANK_CACHE[path]


def sample_glitch_sources(path, batch_size, device):
    """Sample ``batch_size`` strain-domain glitch morphologies from the bank.

    Returns ``(sources, params)`` with ``sources`` of shape (batch, time) and
    ``params`` carrying the per-example glitch class / catalogue metadata.
    """
    bank = load_glitch_bank(path)
    n = bank["source"].shape[0]
    idx = torch.randint(0, n, (batch_size,))
    sources = bank["source"][idx].to(device)
    params = {
        "glitch_class": bank["glitch_class"][idx].to(device),
        "gravityspy_snr": bank["gravityspy_snr"][idx].to(device),
        "peak_frequency": bank["peak_frequency"][idx].to(device),
    }
    return sources, params
