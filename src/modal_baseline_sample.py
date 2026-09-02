#!/usr/bin/env python3
"""Modal-synthesis baseline with mode parameters from SAMPLE.

Mode frequencies and decay rates are estimated with the SAMPLE package
(github.com/LIMUNIMI/SAMPLE, pip `lim-sample`, MIT; Tiraboschi, Avanzini &
Ntalampiras, "Spectral Analysis for Modal Parameters Linear Estimate",
SMC 2020). The estimator-independent parts (gap clustering, force-driven
least-squares gains, quadrature synthesis, material pooling) are imported
from modal_baseline.py.

What SAMPLE provides and what we use
  SAMPLE.fit(x) runs a sinusoidal-model analysis (STFT peak picking +
  ModalTracker partial tracking) and, per surviving track, a hinge
  regression of the dB-magnitude trajectory over time; it exposes
    freqs_  : mean frequency of each track (Hz)
    decays_ : decay parameter d_S with envelope e^{-2 t / d_S}
              (additive_synth: dec = exp(x @ (-2/decays)));
              d_S = -40*log10(e)/k where k is the hinge slope in dB/s
    amps_   : initial amplitude from the hinge intercept (linear)
  Conversion to our convention (modal_baseline: envelope e^{-d t}):
      d_ours = 2 / d_S   [s^-1]      (decay time tau = 1/d_ours = d_S/2)
  We use freqs_ and decays_ (excitation-invariant) and discard amps_:
  SAMPLE's amps_ are fitted to each recording's own excitation amplitude,
  so reusing them would double-count the force amplitude. Gains are
  instead re-fitted by least squares against (measured force (*)
  damped-sinusoid kernels) on the object's training strikes
  (modal_baseline.fit_gains).

Fitting input and sample-rate handling
  Clips are the dataset's raw 48 kHz, 200 ms (9600-sample) onset windows
  (force-peak - 5 ms anchor). SAMPLE does no resampling of its own: the
  sampling rate enters only through sinusoidal__tracker__fs (frequency
  mapping ploc*fs/n and the time axis of the regression), so fs=48000 is
  set explicitly. Each clip is peak-normalized to |x|max=1 before
  SAMPLE.fit so that the package's absolute dB thresholds (peak detection
  t, track peak_threshold) mean the same thing for every recording. This
  is an analysis-input scaling only: freqs/decays are amplitude-invariant,
  amps_ are discarded, and the synthesis gains are fitted on the raw
  un-normalized strikes, so no per-strike renormalization enters the model.

Hyperparameters
  SAMPLE was designed for seconds-long glass-impact recordings at
  44.1 kHz; the package GUI defaults (the authors' working values: blackman
  window 4096, fft 4096, hop 1024, max_n_sines 64, min_sine_dur 0.1 s,
  strip_t 0.5 s, peak_threshold -66, reverse True) target that regime. Our
  clips are 200 ms at 48 kHz, so the time-domain parameters are scaled
  down while frequency-domain choices stay close to the package defaults:
  sinusoidal__tracker__fs = 48000      required: our sample rate.
  sinusoidal__w = blackman(2048)       ~42.7 ms window (GUI type blackman
                                       kept; GUI size 4096 = 93 ms would
                                       be half our clip; the class default
                                       2001 Hamming is the same length
                                       scale, so 2048 keeps the GUI window
                                       type at the class-default length).
  sinusoidal__n = 4096                 2x zero-padded FFT, 11.7 Hz bins
                                       (GUI keeps n = window size; the 2x
                                       pad sharpens peak interpolation on
                                       our short window).
  sinusoidal__tracker__h = 128         main short-clip adaptation: hop
                                       2.67 ms -> ~76 frames per clip
                                       (default 500 / GUI 1024 would give
                                       only ~16 / ~8 regression points
                                       for the hinge fit).
  sinusoidal__tracker__min_sine_dur = 0.02
                                       default 0.04 s / GUI 0.1 s are 20 /
                                       50% of our clip; 0.02 s (~7.5
                                       frames at h=128) keeps fast-decaying
                                       modes trackable.
  sinusoidal__tracker__frequency_bounds = (60, 22000)
                                       the analysis band of the baseline;
                                       the default (20, 16000) would
                                       discard everything above 16 kHz at
                                       48 kHz.
  sinusoidal__tracker__peak_threshold = -80
                                       track acceptance floor (dB at t=0,
                                       on the peak-normalized clip).
                                       Between the class default -90 and
                                       the GUI -66; -66 assumes full-scale
                                       material, our informative modes sit
                                       lower; the support/top-K filter
                                       prunes later.
  sinusoidal__t = -90 (default kept)   peak-detection floor.
  sinusoidal__padded = True            centers the first window on sample
                                       0 and analyses the full clip;
                                       unpadded analysis would drop ~21 ms
                                       (>10%) at each end of a 200 ms clip.
  sinusoidal__tracker__reverse = True  GUI default; analyse time-reversed
                                       audio so decaying partials are
                                       tracked from their quiet end
                                       (recommended modal mode; the package
                                       un-reverses the time axis).
  sinusoidal__tracker__strip_t = 0.05  tracks must start within 50 ms of
                                       clip start (onset at ~5 ms): drops
                                       late-born spurious tracks. GUI 0.5 s
                                       scaled to clip length.
  sinusoidal__tracker__max_n_sines = 64   GUI value (default 100 is also
                                       fine; 64 is the authors' working
                                       value).
  sinusoidal__tracker__freq_dev_offset = 20 (default kept)
  sinusoidal__tracker__freq_dev_slope = 0.0025 (GUI value; class default
                                       0.01) frame-to-frame tracking
                                       tolerance 20 Hz + 0.0025*f.
  merge_strategy = "average" (default kept), max_n_modes = None (no
  resynthesis cap; our own top-K=48 rule does the capping).

Per-object aggregation
  Per training recording, SAMPLE yields a set of (freq, d_ours, weight)
  tracks. All tracks of the object's recordings are pooled and clustered
  with the rule of modal_baseline: 1% relative gap clustering
  (modal_baseline._gap_cluster, CLUSTER_TOL), support >= 30% of
  recordings (MIN_SUPPORT_FRAC), top K=48 by summed weight, cluster
  frequency = median, cluster damping = median of member tracks' d_ours
  (each track clamped to [D_MIN, D_MAX] before the median).
  Prominence analog: clusters are ranked by summed per-track weight.
  SAMPLE tracks have no spectral-peak prominence; the weight used is
  (track dB level at t=0 on the peak-normalized clip) - peak_threshold,
  i.e. dB above the acceptance floor, non-negative. It is stored under
  the 'prominence' key so the pooling/ranking code of modal_baseline
  applies unchanged.

Gains, synthesis, material priors (imported from modal_baseline)
  gains: modal_baseline.fit_gains (joint ridge LS, quadrature kernels,
    relative ridge) on the object's raw training strikes;
  synthesis: modal_baseline.synthesize (quadrature path; params carry
    a/b);
  held-out material priors: modal_baseline._pool_material via
    _pool_all_materials (1% frequency-cluster matching, mean a/b across
    objects, median freq/damping, (n_objects, prominence) ranking,
    top K=48).

Fitting uses the training split only; modal_baseline_eval.py scores the
result on the validation split. Cache:
    cache/modal_baseline_params_{within,heldout}.pt

Usage:
    python3 src/modal_baseline_sample.py --split within|heldout
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import scipy.signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import sample as sample_pkg
from sample import SAMPLE

import modal_baseline as MB
from modal_baseline import (SR, N_SAMPLES, CLUSTER_TOL, MIN_SUPPORT_FRAC,
                            K_MODES, D_MIN, D_MAX, _gap_cluster, _prep_force,
                            fit_gains, synthesize, _pool_all_materials)

# SAMPLE hyperparameters (rationale in the module docstring)
PEAK_THRESHOLD_DB = -80.0
SAMPLE_KWARGS = dict(
    sinusoidal__w=scipy.signal.get_window("blackman", 2048, fftbins=False),
    sinusoidal__n=4096,
    sinusoidal__t=-90.0,
    sinusoidal__padded=True,
    sinusoidal__tracker__fs=SR,
    sinusoidal__tracker__h=128,
    sinusoidal__tracker__max_n_sines=64,
    sinusoidal__tracker__min_sine_dur=0.02,
    sinusoidal__tracker__freq_dev_offset=20.0,
    sinusoidal__tracker__freq_dev_slope=0.0025,
    sinusoidal__tracker__frequency_bounds=(60.0, 22000.0),
    sinusoidal__tracker__peak_threshold=PEAK_THRESHOLD_DB,
    sinusoidal__tracker__reverse=True,
    sinusoidal__tracker__strip_t=0.05,
)


def _sample_kwargs_meta() -> dict:
    """JSON-serializable copy of SAMPLE_KWARGS for the result file."""
    out = {}
    for k, v in SAMPLE_KWARGS.items():
        if isinstance(v, np.ndarray):
            out[k] = f"blackman window, {len(v)} samples (sym)"
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


# 1. Per-recording SAMPLE fit
def sample_fit_one(wav: np.ndarray):
    """SAMPLE.fit on one recording (peak-normalized analysis input).

    Returns (freqs (M,), d_ours (M,), weight_db (M,)):
      freqs   : track frequencies in Hz
      d_ours  : amplitude decay rates in our e^{-d t} convention,
                d_ours = 2 / decays_SAMPLE, clamped to [D_MIN, D_MAX]
                (per-track clamp before aggregation)
      weight_db: prominence analog = track dB level at t=0 (peak-
                normalized clip) minus the acceptance floor; >= 0."""
    x = wav.astype(np.float64)
    peak = np.abs(x).max()
    if peak <= 0:
        return np.empty(0), np.empty(0), np.empty(0)
    x = x / peak
    m = SAMPLE(**SAMPLE_KWARGS)
    m.fit(x)
    f = np.asarray(m.freqs_, dtype=np.float64)
    d_s = np.asarray(m.decays_, dtype=np.float64)
    a = np.asarray(m.amps_, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(d_s) & np.isfinite(a) & (d_s > 0) & (a > 0)
    f, d_s, a = f[ok], d_s[ok], a[ok]
    d_ours = np.clip(2.0 / d_s, D_MIN, D_MAX)
    weight = np.maximum(20.0 * np.log10(a + 1e-12) - PEAK_THRESHOLD_DB, 0.0)
    return f, d_ours, weight


# 2. Cross-recording aggregation (clustering rule of modal_baseline)
def aggregate_modes(per_rec: list):
    """Aggregate per-recording SAMPLE tracks into object modes:
    _gap_cluster at CLUSTER_TOL (1%), support >= MIN_SUPPORT_FRAC (30%) of
    recordings, top K_MODES=48 by summed weight, median frequency, median
    of the member tracks' decay rates.

    per_rec: list of (freqs, d_ours, weight_db) tuples, one per recording.
    Returns (freqs (M,), dampings (M,), prominence (M,), support (M,))."""
    all_f, all_d, all_w, all_rec = [], [], [], []
    n_rec = 0
    for ri, (f, d, w) in enumerate(per_rec):
        if len(f) == 0:
            continue
        n_rec += 1
        all_f.append(f); all_d.append(d); all_w.append(w)
        all_rec.append(np.full(len(f), ri))
    if n_rec == 0:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, int))
    all_f = np.concatenate(all_f); all_d = np.concatenate(all_d)
    all_w = np.concatenate(all_w); all_rec = np.concatenate(all_rec)

    min_support = max(1, int(np.ceil(MIN_SUPPORT_FRAC * n_rec)))
    freqs, damps, proms, supports = [], [], [], []
    for cl in _gap_cluster(all_f):                # 1% relative tolerance
        support = len(np.unique(all_rec[cl]))
        if support < min_support:
            continue
        freqs.append(np.median(all_f[cl]))
        damps.append(float(np.clip(np.median(all_d[cl]), D_MIN, D_MAX)))
        proms.append(float(all_w[cl].sum()))
        supports.append(support)
    if not freqs:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, int))
    freqs = np.array(freqs); damps = np.array(damps)
    proms = np.array(proms); supports = np.array(supports, int)
    order = np.argsort(proms)[::-1][:K_MODES]
    return freqs[order], damps[order], proms[order], supports[order]


# 3. Per-object fit (SAMPLE modes + least-squares gains)
def fit_object(ds, train_idx_of_object) -> dict:
    """One object: SAMPLE.fit per training recording, aggregation, then
    joint ridge LS gains on the raw strikes. Returns a params dict with
    a/b quadrature gains, as consumed by modal_baseline.synthesize."""
    samples = [ds[i] for i in train_idx_of_object]
    waves = [s["waveform"].numpy().astype(np.float64) for s in samples]
    forces = [_prep_force(s["force_waveform"]) for s in samples]
    per_rec = [sample_fit_one(w) for w in waves]
    freqs, dampings, proms, supports = aggregate_modes(per_rec)
    n_tracks = int(sum(len(f) for f, _, _ in per_rec))
    if len(freqs) == 0:                           # degenerate: emit silence
        return dict(freqs=np.empty(0), dampings=np.empty(0),
                    a=np.empty(0), b=np.empty(0), prominence=np.empty(0),
                    support=np.empty(0, int), n_train=len(samples),
                    n_sample_tracks=n_tracks,
                    obj_id_raw=samples[0]["obj_id_raw"],
                    material_idx=samples[0]["material_idx"])
    a, b = fit_gains(freqs, dampings, forces, waves)   # LS on raw strikes
    return dict(freqs=freqs, dampings=dampings, a=a, b=b,
                prominence=proms, support=supports, n_train=len(samples),
                n_sample_tracks=n_tracks,
                obj_id_raw=samples[0]["obj_id_raw"],
                material_idx=samples[0]["material_idx"])


def _fit_all_objects(ds, train_idx, verbose: bool = True) -> dict:
    """{obj_id_raw: params} for every object present in train_idx."""
    by_obj = {}
    for i in train_idx:
        by_obj.setdefault(ds[i]["obj_id_raw"], []).append(i)
    out = {}
    for n, (o, idxs) in enumerate(sorted(by_obj.items())):
        out[o] = fit_object(ds, idxs)
        if verbose and (n + 1) % 10 == 0:
            print(f"  fitted {n + 1}/{len(by_obj)} objects", flush=True)
    return out


# 4. Split-level fit + cache (material pooling imported)
def fit_split(ds, split: str, cache_dir: Path = None, verbose: bool = True):
    """Fit all training objects of a split with SAMPLE + LS gains, pool the
    material priors, cache to cache/modal_baseline_params_{split}.pt."""
    from dataset import load_split_indices
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    train_idx, _, _ = load_split_indices(
        ds, str(ROOT / f"splits/splits_{split}.json"))
    t0 = time.time()
    objects = _fit_all_objects(ds, train_idx, verbose=verbose)
    t_fit = time.time() - t0
    materials = _pool_all_materials(objects)
    meta = dict(K=K_MODES, cluster_tol=CLUSTER_TOL,
                min_support_frac=MIN_SUPPORT_FRAC,
                ridge_rel=MB.RIDGE_REL, d_clamp=(D_MIN, D_MAX),
                n_train_recordings=len(train_idx),
                fitted_on="train split only",
                fit_mode="sample",
                sample_version=sample_pkg.__version__,
                sample_package="lim-sample (github.com/LIMUNIMI/SAMPLE, "
                               "MIT; Tiraboschi, Avanzini & Ntalampiras, "
                               "SMC 2020)",
                sample_kwargs=_sample_kwargs_meta(),
                decay_convention="SAMPLE decays_ d_S has envelope "
                                 "e^{-2t/d_S}; converted d_ours = 2/d_S, "
                                 "per-track clamp to [D_MIN, D_MAX] "
                                 "before the cluster median",
                analysis_input="each clip peak-normalized to |x|max=1 "
                               "before SAMPLE.fit (dB thresholds "
                               "comparable across recordings); freqs/"
                               "decays amplitude-invariant; amps_ "
                               "discarded (would double-count force "
                               "amplitude); gains re-fitted with joint "
                               "ridge LS on the raw train strikes",
                prominence_analog="per-track weight = dB level at t=0 "
                                  "(peak-normalized clip) minus the "
                                  f"{PEAK_THRESHOLD_DB:.0f} dB acceptance "
                                  "floor; used as the summed-prominence "
                                  "ranking weight",
                aggregation="_gap_cluster 1% tol, support >= 30%, top "
                            "K=48 by summed weight, median freq; decay = "
                            "median of member tracks; gains fit_gains; "
                            "material priors _pool_material")
    payload = dict(split=split, objects=objects, materials=materials,
                   meta=meta)
    out = cache_dir / f"modal_baseline_params_{split}.pt"
    torch.save(payload, out)
    if verbose:
        print(f"[modal_baseline_sample] {split}: {len(objects)} objects, "
              f"{len(materials)} materials, {time.time() - t0:.1f}s "
              f"(SAMPLE fits {t_fit:.1f}s over {len(train_idx)} recordings) "
              f"-> {out}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["within", "heldout"], required=True)
    args = ap.parse_args()
    from dataset import ObjectFolderRealDataset
    ds = ObjectFolderRealDataset(return_waveform=True)
    fit_split(ds, args.split)


if __name__ == "__main__":
    main()
