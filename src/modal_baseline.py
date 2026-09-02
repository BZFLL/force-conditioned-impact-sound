#!/usr/bin/env python3
"""Shared machinery of the modal-synthesis baseline.

Linear damped-sinusoid modal model driven by the measured contact force
(van den Doel & Pai 1998; van den Doel, Kry & Pai 2001, FoleyAutomatic;
parameter estimation from recordings as in Ren, Yeh & Lin 2013, whose
nonlinear/transient residual term is omitted). Per object:

    synth(t) = sum_i [ a_i * (force (*) e^{-d_i t} cos 2 pi f_i t)
                     + b_i * (force (*) e^{-d_i t} sin 2 pi f_i t) ]

The amplitude scales with the measured force; there is no per-strike
renormalisation, so synth(2 force) = 2 synth(force).

Mode frequencies and decay rates are estimated elsewhere
(modal_baseline_sample.py, using the SAMPLE package); this module holds the
pieces that are independent of the estimator:

  * gap clustering of per-recording mode estimates: modes sorted by
    frequency and split where the relative gap exceeds CLUSTER_TOL (1%),
    single linkage;
  * gains: joint ridge least squares over all of the object's training
    strikes. Quadrature responses are obtained by FFT convolution of the
    strike's raw force window with the unit kernels, truncated to 9600
    samples; the normal equations are accumulated over strikes; the ridge
    term is relative, lambda = RIDGE_REL * mean(diag(X^T X)), because raw
    force amplitudes are of arbitrary scale;
  * synthesis: the same quadrature kernels convolved with a force window;
  * material priors for unseen objects: the fitted modes of all training
    objects of a material are pooled, sorted by frequency and gap-clustered
    at the same 1% tolerance; within a cluster an object's own modes are
    averaged first, then a_i and b_i are the mean across contributing
    objects and frequency/damping the median across member modes; clusters
    are ranked by (number of contributing objects, summed prominence) and
    the top K_MODES are kept. a_i, b_i carry per-object phase (microphone
    distance/polarity), so cross-object averaging can partially cancel; the
    per-cluster mean gain magnitude is stored alongside as `gain_mag_mean`
    for diagnosis. Synthesis uses the mean a, b.

Fitting uses the training split only. Parameter dicts carry the keys
freqs, dampings, a, b, prominence, support (plus bookkeeping fields).

API:
    fit_gains(freqs, dampings, forces, targets) -> (a (K,), b (K,))
    synthesize(params, force_waveform)          -> torch.FloatTensor (9600,)
    _pool_all_materials(obj_params)              -> {material_name: params}
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SR = 48000
N_SAMPLES = 9600            # 200 ms window, dataset convention

# Mode clustering
CLUSTER_TOL = 0.01          # 1% relative frequency tolerance
MIN_SUPPORT_FRAC = 0.30     # mode must appear in >= 30% of recordings
K_MODES = 48

# Damping clamp
TAU_MIN_S, TAU_MAX_S = 0.003, 2.0   # decay-time clamp 3 ms .. 2 s
D_MIN, D_MAX = 1.0 / TAU_MAX_S, 1.0 / TAU_MIN_S

# Gains
RIDGE_REL = 1e-6            # relative ridge (x mean diag of Gram matrix)
CONV_NFFT = 32768           # >= 2*9600-1 for linear FFT convolution


# Mode clustering
def _gap_cluster(freqs: np.ndarray, tol: float = CLUSTER_TOL):
    """Sort freqs; break clusters where the relative gap exceeds tol.
    Returns list of index arrays into the original freqs array. Single
    linkage, so a chain can exceed tol end-to-end."""
    order = np.argsort(freqs)
    fs = freqs[order]
    clusters, start = [], 0
    for i in range(1, len(fs) + 1):
        if i == len(fs) or (fs[i] - fs[i - 1]) > tol * fs[i - 1]:
            clusters.append(order[start:i])
            start = i
    return clusters


# Gains (joint ridge LS over all train strikes of the object)
def _kernels(freqs: np.ndarray, dampings: np.ndarray) -> np.ndarray:
    """(2K, 9600) quadrature kernels: [cos block; sin block]."""
    t = np.arange(N_SAMPLES) / SR
    decay = np.exp(-dampings[:, None] * t[None, :])
    ph = 2 * np.pi * freqs[:, None] * t[None, :]
    return np.concatenate([decay * np.cos(ph), decay * np.sin(ph)], axis=0)


def _responses(kernels_f: np.ndarray, force: np.ndarray) -> np.ndarray:
    """(2K, 9600) responses = force (*) kernel, truncated. kernels_f is the
    pre-computed rFFT (2K, CONV_NFFT//2+1) of the kernels."""
    F = np.fft.rfft(force, n=CONV_NFFT)
    return np.fft.irfft(kernels_f * F[None, :], n=CONV_NFFT, axis=1)[:, :N_SAMPLES]


def fit_gains(freqs, dampings, forces: list, targets: list):
    """Joint ridge LS over all strikes: min sum_j ||y_j - X_j beta||^2 + lambda ||beta||^2.
    Returns (a (K,), b (K,))."""
    K = len(freqs)
    kern = _kernels(freqs, dampings)
    kern_f = np.fft.rfft(kern, n=CONV_NFFT, axis=1)
    G = np.zeros((2 * K, 2 * K))
    c = np.zeros(2 * K)
    for f, y in zip(forces, targets):
        X = _responses(kern_f, f)                 # (2K, 9600)
        G += X @ X.T
        c += X @ y
    lam = RIDGE_REL * float(np.trace(G)) / (2 * K)
    beta = np.linalg.solve(G + lam * np.eye(2 * K), c)
    return beta[:K], beta[K:]


# Synthesis
def _prep_force(force_waveform) -> np.ndarray:
    f = force_waveform.detach().cpu().numpy() if torch.is_tensor(force_waveform) \
        else np.asarray(force_waveform)
    f = f.astype(np.float64).ravel()
    if len(f) < N_SAMPLES:
        f = np.pad(f, (0, N_SAMPLES - len(f)))
    return f[:N_SAMPLES]


def synthesize(params: dict, force_waveform) -> torch.Tensor:
    """Force-driven quadrature synthesis from a params dict (freqs, dampings,
    a, b)."""
    if len(params["freqs"]) == 0:
        return torch.zeros(N_SAMPLES)
    f = _prep_force(force_waveform)
    kern = _kernels(params["freqs"], params["dampings"])
    beta = np.concatenate([params["a"], params["b"]])
    kern_f = np.fft.rfft(kern, n=CONV_NFFT, axis=1)
    X = _responses(kern_f, f)
    return torch.tensor(beta @ X, dtype=torch.float32)


# Material priors
def _pool_all_materials(obj_params: dict) -> dict:
    """{material_name: params} from a {obj_id_raw: params} dict of fitted
    training objects, pooled per material with _pool_material."""
    from dataset import MATERIALS
    by_mat = {}
    for p in obj_params.values():
        by_mat.setdefault(MATERIALS[p["material_idx"]], []).append(p)
    return {m: _pool_material(ps) for m, ps in sorted(by_mat.items())}


def _pool_material(obj_params_list: list) -> dict:
    """Merge per-object mode sets into one material prior (rule in the
    module docstring)."""
    rows = []            # (freq, damping, a, b, prominence, obj_tag)
    for oi, p in enumerate(obj_params_list):
        for k in range(len(p["freqs"])):
            rows.append((p["freqs"][k], p["dampings"][k], p["a"][k], p["b"][k],
                         p["prominence"][k], oi))
    if not rows:
        return dict(freqs=np.empty(0), dampings=np.empty(0), a=np.empty(0),
                    b=np.empty(0), prominence=np.empty(0),
                    support=np.empty(0, int), n_objects=len(obj_params_list),
                    gain_mag_mean=np.empty(0))
    rows = np.array(rows)
    out = []
    for cl in _gap_cluster(rows[:, 0]):
        sub = rows[cl]
        objs = np.unique(sub[:, 5])
        # per-object average first (an object may put 2 modes in one cluster),
        # then mean across objects
        a_o = [sub[sub[:, 5] == o, 2].mean() for o in objs]
        b_o = [sub[sub[:, 5] == o, 3].mean() for o in objs]
        mag_o = [np.hypot(sub[sub[:, 5] == o, 2], sub[sub[:, 5] == o, 3]).mean()
                 for o in objs]
        out.append((np.median(sub[:, 0]), np.median(sub[:, 1]),
                    float(np.mean(a_o)), float(np.mean(b_o)),
                    float(sub[:, 4].sum()), len(objs), float(np.mean(mag_o))))
    out.sort(key=lambda r: (-r[5], -r[4]))        # n_objects, then prominence
    out = out[:K_MODES]
    arr = np.array(out)
    return dict(freqs=arr[:, 0], dampings=arr[:, 1], a=arr[:, 2], b=arr[:, 3],
                prominence=arr[:, 4], support=arr[:, 5].astype(int),
                n_objects=len(obj_params_list), gain_mag_mean=arr[:, 6])
