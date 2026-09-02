#!/usr/bin/env python3
"""Signal-level metric definitions (descriptive; no significance tests).

Two per-clip paired metrics against ground truth (the energy-envelope
metric of the main evaluation is evaluate.energy_l1):

  LSD  -- log-spectral distance, dB. STFT (n_fft=1024, hop=256, Hann); both
          magnitude spectra clamped at an 80 dB relative floor (ref = max of
          the pair), then sqrt(mean over freq of
          (20log10 S_gt - 20log10 S_gen)^2), mean over frames.
  ENV  -- raw RMS-envelope L1 (win=240, hop=120, linear amplitude);
          level-sensitive.

Domain: the DDSP model emits waveforms in the normalised training domain
(target = waveform / wav_scale); generated clips are multiplied by the
split's training wav_scale std (stats/wav_scale_<split>.pt, written by
train.py) before any metric, so gen and gt are compared in the raw domain.
The functions here are used by modal_baseline_eval.py.
"""
import torch

SR = 48000

def stft_mag(x, n_fft=1024, hop=256):
    w = torch.hann_window(n_fft)
    S = torch.stft(x, n_fft, hop, window=w, return_complex=True, center=True)
    return S.abs()

def lsd(gt, gen):
    mg, mn = stft_mag(gt), stft_mag(gen)
    floor = max(float(mg.max()), float(mn.max())) * 10 ** (-80 / 20)
    a = 20 * torch.log10(mg.clamp(min=floor))
    b = 20 * torch.log10(mn.clamp(min=floor))
    return float(torch.sqrt(((a - b) ** 2).mean(dim=0)).mean())

def rms_env(x, win=240, hop=120):
    pad = (win - hop) // 2
    x2 = torch.nn.functional.pad(x ** 2, (pad, pad))
    fr = x2.unfold(0, win, hop)
    return torch.sqrt(fr.mean(dim=1) + 1e-12)

def env_l1(gt, gen):
    return float((rms_env(gt) - rms_env(gen)).abs().mean())
