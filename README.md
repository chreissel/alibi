# alibi — Embedding witness channel information for gravitational-wave data analysis

This repository provides the guideline to answer the question:

> **Does embedding witness channel information alongside the signal-carrying
> strain channel help a model tell real astrophysical signals apart from instrumental
> glitches?**

We approach this questions by generating a labelled **toy dataset**. It combines real interferometer background with simulated
transients, and adds a **synthetic witness channel** that is coupled to glitches but blind to astrophysical signals.

## The idea

In a real interferometer, a witness (auxiliary) sensor records instrumental/environmental
disturbances but does **not** respond to gravitational waves. That asymmetry is the
physical basis of glitch vetoing, and it is what this dataset encodes across three classes:

| Class          | Strain channel (H1)                    | Witness channel                         |
|----------------|----------------------------------------|-----------------------------------------|
| **Signal**     | real noise + injected gravitational wave (GW) signal        | noise only (a GW does not couple here)  |
| **Glitch**     | real noise + injected **real glitch** (O3a scattered-light/whistle/power-line) | noise + LTI-coupled copy synthesised *from* the strain glitch |
| **Background** | real noise only                        | noise only                              |

The signal and glitch transients are placed at the same time location, so the witness, not the arrival time, is the discriminator.

## What is simulated, and how

* **Signals**: ml4gw CBC waveforms (`IMRPhenomD`), projected onto the detector and rescaled
  to a target network SNR (`ml4gw.gw.reweight_snrs`). We start by considering shorter signals from binary black hole merging.
* **Glitches**: *real* O3a glitch morphologies — **Scattered_Light**, **Whistle**,
  **Power_Line** — selected from the GravitySpy catalogue and cropped from GWOSC strain into
  a *glitch bank* (`download_glitches.py` → `glitches.py`). These classes are chosen because
  they are *genuinely witnessed* by auxiliary channels in real interferometers (scattered
  light by seismic/length channels, whistles by RF/PSL channels, power-line glitches by
  mains/magnetometer monitors), so the synthetic witness is physically motivated. The real
  glitch goes into the **strain**, and the **witness is synthesised from it** by an *LTI
  Butterworth filter* (`witness = C(strain glitch)`) — modelled per class on that class's real
  monitor (see `witness.coupling.per_class`). Only a fraction `alpha` of the witness power is
  coherent with the strain glitch, so `alpha` sets how informative the witness is. (An ad-hoc
  `SineGaussian` source is also available via `glitch.source: sine_gaussian`.)
* **Background**: real H1 strain from GWOSC O3a; the witness background is synthesised
  Gaussian noise.
* **Whitening**: per-channel PSD estimation + `ml4gw.transforms.Whiten`.

The design loosely follows [`chreissel/GWDatasetGeneration`](https://github.com/chreissel/GWDatasetGeneration),
extended with the witness channel and the glitch class. WE consider only a one-detector setup (H1) for now.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Download real background** (needs network access to GWOSC):

   ```bash
   cd dataset
   python load_data.py --config configs/config_H1.yaml --data ./data
   ```
   This writes `./data/background_data/background-*.hdf5`. 
   It can be omitted when working on the FASRC cluster since the O3a dataset is already downloaded and available following the path: `/n/holystore01/LABS/iaifi_lab/Lab/creissel/SparseBank/background_data/`.

2. **Build the glitch bank** (real O3a Scattered_Light/Whistle/Power_Line; needs GravitySpy + GWOSC access):

   ```bash
   python download_glitches.py --config configs/config_H1.yaml
   # or, with a locally downloaded GravitySpy catalogue:
   python download_glitches.py --config configs/config_H1.yaml --catalog ./gravityspy_O3a.csv
   ```
   This writes the bank to `glitch.gravityspy.bank_path` (default `data/glitch_bank.h5`).
   To develop without any data access, build offline stand-ins instead:
   `python download_glitches.py --config configs/config_H1.yaml --synthetic`
   (or set `glitch.source: sine_gaussian` to skip the bank entirely).

3. **Generate the dataset**:

   ```bash
   python main.py --config configs/config_H1.yaml --data ./data/background_data --out ./out
   ```
   This writes `background.h5`, `signal.h5`, `glitch.h5` into `./out`.

## Output format

One HDF5 file per class, each containing:

* `data`  — `(N, 2, T)` float array, channel order **`[strain, witness]`** (whitened),
  with `T = waveform_duration * sample_rate`.
* `label` — `(N,)` int: `0=background, 1=signal, 2=glitch`.
* one dataset per sampled parameter (e.g. `snr`, `chirp_mass` for signals;
  `glitch_class` (0=Scattered_Light, 1=Whistle, 2=Power_Line), `gravityspy_snr`, `peak_frequency`,
  `strain_snr`, `witness_snr` for glitches).
* attrs: `label` (the class id) and `channels` (`[strain, witness]`); `glitch.h5` also
  has `glitch_classes` (the `glitch_class` id→name map).

## Key config knobs (`dataset/configs/config_H1.yaml`)

* `general` — detector, sample rate, window duration, counts, GWOSC run.
* `waveform` / `snr_reweighting` — CBC prior and signal SNR distribution.
* `glitch.source` — `gravityspy` (real O3a glitches) or `sine_gaussian` (ad-hoc).
* `glitch.gravityspy` — run/ifo, glitch `classes`, GravitySpy confidence + SNR cuts,
  `max_per_class`, and `bank_path` for the downloaded glitch bank.
* `glitch.snr` — per-channel (strain/witness) target SNR distributions.
* `witness.coupling.alpha` — strain↔witness coherence (the main "how useful is the witness"
  knob); `witness.coupling.filter` — the Butterworth coupling band (`witness = C(strain glitch)`).
* `witness.coupling.per_class` — optional per-class `alpha`/`filter` overrides giving
  Scattered_Light/Whistle/Power_Line distinguishable witness signatures.
