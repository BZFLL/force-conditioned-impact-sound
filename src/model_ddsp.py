"""Filterbank-DDSP hybrid backbone.

A NoiseBandNet-style pre-baked noise-band synthesiser driven by an MLP+GRU
controller, with an affine (adaLN/FiLM) injection of the object embedding
and an optional damped-sinusoid modal branch ("NBN+modal" variant) for the
resonant impact tail.

Synthesis end, a compact re-implementation of the NoiseBandNet design
(Barahona-Ríos & Collins, IEEE/ACM TASLP 32:1573-1585, 2024) rather than the
released NoiseBandNet code: it runs at 48 kHz on 200 ms windows, takes the
force curve frame-wise as conditioning and adds a modal branch, none of which
the reference implementation provides.
  * N_BANDS loopable noise bands are pre-baked once and cached: a per-band
    triangular magnitude response over rfft bins × random phase (fixed
    seed), IFFT'd to a loopable 1-second band at 48 kHz. Half the band
    centres are linear on [0, Fs/8], half log on [Fs/8, Fs/2] (NBN design).
    Cache: cache/noise_bands_<Fs>_<N>.pt.
  * The network outputs non-negative per-band amplitudes at frame rate
    W=32 samples (1500 Hz frame rate, ~0.67 ms/frame at 48 kHz; NBN's
    transient-preserving resolution), via a modified sigmoid
    (NBN Eq. 7.4 style). Amplitudes are linearly upsampled ×W, multiplied
    with the noise bands and summed into the waveform.

Conditioning (same level semantics as model.py's _get_conditioning):
  L1: object embedding (16) only           (static)
  L2: + scalar peak force (feature 0)      (static)
  L3: + full 6-D force descriptor          (static)
  L4: + 6-D descriptor (static) and the raw 200 ms force-waveform window
      turned into a frame-rate force curve [B, T_frames, 2] concatenated
      frame-wise into the GRU input, in place of the CVAE's global 1D-CNN
      vector. The force window is 200 ms = 9600 samples = 300 frames at
      W=32, the DDSP output frame count. Per frame we pool [RMS, max-abs]
      over the W samples rather than the signed mean, which at W=32 cancels
      about half of the peak-frame amplitude. Each channel is z-scored inside
      the model with
      train-split statistics set by train.py (set_force_curve_stats).
  At every level the object embedding additionally modulates the GRU
  output through an adaLN/FiLM affine layer (gamma, beta).

Loss: multi-scale MRSTFT (linear + log magnitude L1) at FFT sizes
[2048, 1024, 512, 256, 128, 64], sized for the ~9600-sample onset window.

model.py holds the CVAE backbone; both share the same conditioning
contract and the same training script (train.py --backbone {cvae,ddsp}).
"""

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
SR = 48000                    # sample rate (dataset is 48 kHz throughout)
N_BANDS = 2048                # default noise-band count (quick check: --n_bands 256)
W = 32                        # amplitude frame hop in samples → 1500 Hz frame rate
K_MODAL = 32                  # damped-sinusoid modes in the modal branch
MRSTFT_SCALES = [2048, 1024, 512, 256, 128, 64]   # FFT sizes for the loss
NOISE_BAND_SECONDS = 1.0      # baked loopable band length (≥ output window)
BAKE_SEED = 42                # fixed seed → deterministic bands
BAKE_CHUNK = 128              # bands baked per chunk (fixed → deterministic RNG order)
SYNTH_BAND_CHUNK = 256        # bands multiplied per chunk at synthesis (memory bound)
MODAL_FREQ_RANGE = (20.0, 0.45 * SR)   # Hz (0.45·Fs keeps modes below Nyquist)
MODAL_DAMP_RANGE = (5.0, 2000.0)       # 1/s — impact tails
HIDDEN_SIZE = 256             # controller width (GRU size)
OBJ_EMBED_DIM = 16            # same as model.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"


def modified_sigmoid(x):
    """Non-negative amplitude nonlinearity, NBN Eq. 7.4 / DDSP style:
    2·sigmoid(x)^log(10) + 1e-7."""
    return 2.0 * torch.sigmoid(x) ** math.log(10.0) + 1e-7


def safe_log(x):
    return torch.log(x + 1e-7)


# Noise-band baking (frequency-domain simplified implementation,
# design from NoiseBandNet, Barahona-Ríos & Collins 2024)

def _band_centers(sr: int, n_bands: int) -> torch.Tensor:
    """NBN band layout: half the centres linear on [0, Fs/8], half
    logarithmic on [Fs/8, Fs/2]."""
    n_lin = n_bands // 2
    n_log = n_bands - n_lin
    lin = torch.linspace(0.0, sr / 8.0, n_lin + 1)[:n_lin]          # excl. Fs/8
    log = torch.logspace(math.log10(sr / 8.0), math.log10(sr / 2.0), n_log)
    return torch.cat([lin, log])                                     # (n_bands,)


def bake_noise_bands(sr: int = SR, n_bands: int = N_BANDS,
                     seconds: float = NOISE_BAND_SECONDS,
                     seed: int = BAKE_SEED,
                     cache_dir: Path = CACHE_DIR) -> torch.Tensor:
    """Bake (or load from cache) N loopable unit-RMS noise bands.

    Frequency-domain simplified implementation of the NoiseBandNet design
    (Barahona-Ríos & Collins, IEEE/ACM TASLP 32:1573-1585, 2024): per band, a
    triangular magnitude response spanning its neighbouring centres ×
    uniform random phase (fixed-seed generator), then IFFT. Each band is
    synthesised as one exact DFT period, so it loops seamlessly. The same
    seed and BAKE_CHUNK give bit-identical bands. Returns (n_bands, L)
    float32.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"noise_bands_{sr}_{n_bands}.pt"
    if path.exists():
        blob = torch.load(path)
        if (blob.get('seed') == seed and blob.get('seconds') == seconds
                and blob['bands'].shape == (n_bands, int(sr * seconds))):
            print(f"Loaded baked noise bands: {path}")
            return blob['bands']
        print(f"Stale noise-band cache (seed/shape mismatch); re-baking: {path}")

    L = int(sr * seconds)
    n_bins = L // 2 + 1
    freqs = torch.linspace(0.0, sr / 2.0, n_bins)                    # (n_bins,)
    centers = _band_centers(sr, n_bands)
    # Triangle edges: band k rises from c_{k-1} to c_k, falls to c_{k+1}
    left = torch.cat([torch.tensor([0.0]), centers[:-1]])
    right = torch.cat([centers[1:], torch.tensor([sr / 2.0])])

    gen = torch.Generator().manual_seed(seed)
    bands = torch.empty(n_bands, L, dtype=torch.float32)
    for s in range(0, n_bands, BAKE_CHUNK):
        e = min(s + BAKE_CHUNK, n_bands)
        c_ = centers[s:e].unsqueeze(1)   # (chunk, 1)
        l_ = left[s:e].unsqueeze(1)
        r_ = right[s:e].unsqueeze(1)
        f_ = freqs.unsqueeze(0)          # (1, n_bins)
        rise = (f_ - l_) / (c_ - l_).clamp(min=1e-3)
        fall = (r_ - f_) / (r_ - c_).clamp(min=1e-3)
        mag = torch.minimum(rise, fall).clamp(min=0.0, max=1.0)      # (chunk, n_bins)
        # Random phase, fixed seed. DC & Nyquist bins must stay real.
        phase = torch.rand(e - s, n_bins, generator=gen) * 2.0 * math.pi
        phase[:, 0] = 0.0
        phase[:, -1] = 0.0
        spec = mag * torch.exp(1j * phase)
        wav = torch.fft.irfft(spec, n=L)                             # (chunk, L)
        rms = wav.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        bands[s:e] = (wav / rms).to(torch.float32)                   # unit RMS

    torch.save({'bands': bands, 'sr': sr, 'n_bands': n_bands,
                'seed': seed, 'seconds': seconds}, path)
    print(f"Baked {n_bands} noise bands ({sr} Hz, {seconds:.1f}s) -> {path}")
    return bands


class NoiseBandSynth(nn.Module):
    """Amplitude-over-pre-baked-noise-bands synthesiser (NBN style).

    forward(amps) takes non-negative frame-rate amplitudes (B, T_frames,
    n_bands), upsamples them linearly ×W, multiplies each band with its
    loop-sliced baked band and sums over bands, giving (B, T_frames·W).
    """

    def __init__(self, bands: torch.Tensor, w: int = W):
        super().__init__()
        self.register_buffer('bands', bands, persistent=False)  # (N, L)
        self.w = w

    def forward(self, amps: torch.Tensor, loop_offsets: torch.Tensor = None):
        B, T_frames, n_bands = amps.shape
        assert n_bands == self.bands.shape[0]
        T_out = T_frames * self.w
        L = self.bands.shape[1]
        assert T_out <= L, "baked bands must be at least one output window long"

        if loop_offsets is None:
            if self.training:
                # Random loop start per item (global RNG — seeded in train.py)
                loop_offsets = torch.randint(0, L, (B,), device=amps.device)
            else:
                loop_offsets = torch.zeros(B, dtype=torch.long, device=amps.device)
        t_idx = (loop_offsets.unsqueeze(1)
                 + torch.arange(T_out, device=amps.device)) % L      # (B, T_out)

        out = amps.new_zeros(B, T_out)
        # Chunk over bands to bound peak memory (B × chunk × T_out floats)
        for s in range(0, n_bands, SYNTH_BAND_CHUNK):
            e = min(s + SYNTH_BAND_CHUNK, n_bands)
            a = F.interpolate(amps[:, :, s:e].permute(0, 2, 1), size=T_out,
                              mode='linear', align_corners=True)     # (B, c, T)
            nb = self.bands[s:e][:, t_idx].permute(1, 0, 2)          # (B, c, T)
            out = out + (a * nb).sum(dim=1)
        return out


class DampedSinusoids(nn.Module):
    """Damped-sinusoid modal bank:
    y(t) = Σ_k a_k · exp(-d_k t) · sin(2π f_k t). The resonant-tail branch of
    the "NBN+modal" variant, a modal resonator bank as in the Sounding Object
    / differentiable-modal-resonator line of work."""

    def __init__(self, n_modes: int = K_MODAL, sr: int = SR):
        super().__init__()
        self.n_modes = n_modes
        self.sr = sr

    def forward(self, amplitudes, frequencies, dampings, n_samples):
        # amplitudes/frequencies/dampings: (B, n_modes)
        t = torch.arange(n_samples, device=amplitudes.device,
                         dtype=torch.float32) / self.sr
        t = t.unsqueeze(0).unsqueeze(0)                              # (1, 1, T)
        a = amplitudes.unsqueeze(-1)
        f = frequencies.unsqueeze(-1)
        d = dampings.unsqueeze(-1)
        signal = a * torch.exp(-d * t) * torch.sin(2 * math.pi * f * t)
        return signal.sum(dim=1)                                     # (B, T)


# Controller (MLP+GRU) + adaLN/FiLM object modulation

class MLP(nn.Module):
    """Linear / LayerNorm / LeakyReLU stack."""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                  nn.LeakyReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.LeakyReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AdaLNFiLM(nn.Module):
    """adaLN-style FiLM: LayerNorm (no affine) then per-sample (γ, β) from
    the object embedding.
    γ is parameterised as 1+Δγ with a zero-init projection so training
    starts at identity."""

    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: (B, T, H), cond: (B, cond_dim)
        dg, b = self.proj(cond).chunk(2, dim=-1)
        return self.ln(x) * (1.0 + dg.unsqueeze(1)) + b.unsqueeze(1)


class ForceConditionedFilterbankDDSP(nn.Module):
    """Filterbank-DDSP hybrid: NBN-style noise-band synthesiser + optional
    modal branch, driven by an MLP+GRU controller, with the same 4-level
    conditioning contract as model.py's CVAE.

    forward(obj_id, force_features, force_waveform) returns (wav, aux).
    wav is (B, signal_samples) at 48 kHz, the onset window waveform in the
    globally-scaled target domain (train.py divides targets by the
    train-split waveform std, stats/wav_scale_<split>.pt).
    """

    def __init__(
        self,
        n_objects: int = 100,
        ablation_level: int = 3,
        n_bands: int = N_BANDS,
        w: int = W,
        k_modal: int = K_MODAL,
        use_modal: bool = True,
        sr: int = SR,
        signal_samples: int = 9600,        # 200 ms onset window @48 kHz
        force_waveform_samples: int = 9600,  # 200 ms force window @48 kHz
        hidden_size: int = HIDDEN_SIZE,
        obj_embed_dim: int = OBJ_EMBED_DIM,
        cache_dir: Path = CACHE_DIR,
        bake_seed: int = BAKE_SEED,
        identity: str = "obj",             # 'obj' | 'material'
        n_materials: int = 7,
        force_feature_dim: int = 6,        # L3 descriptor dim (v1=6)
    ):
        super().__init__()
        assert signal_samples % w == 0, "signal length must divide by W"
        assert force_waveform_samples % w == 0, "force window must divide by W"
        assert identity in ("obj", "material"), identity

        self.sr = sr
        self.w = w
        self.signal_samples = signal_samples
        self.n_frames = signal_samples // w              # 300 @ 200 ms, W=32
        self.force_frames = force_waveform_samples // w  # 300 @ 200 ms, W=32
        self.n_bands = n_bands
        self.use_modal = use_modal
        self.ablation_level = ablation_level
        self.identity = identity

        # Identity embedding: 'obj' has one row per object, 'material' one
        # row per material (n_materials=7), same embedding dim either way.
        n_identity = n_objects if identity == "obj" else n_materials
        self.obj_embed = nn.Embedding(n_identity, obj_embed_dim)

        # Static conditioning dim, mirroring model.py._get_conditioning.
        if ablation_level == 1:
            static_dim = obj_embed_dim                   # 16
        elif ablation_level == 2:
            static_dim = obj_embed_dim + 1               # 17
        elif ablation_level in (3, 4):
            static_dim = obj_embed_dim + force_feature_dim   # 22 when dim=6
        else:
            raise ValueError(f"Unknown ablation_level: {ablation_level}")

        self.cond_mlp = MLP(static_dim, hidden_size, hidden_size, n_layers=3)
        # L4 force curve has 2 channels ([RMS, max-abs]); GRU input +2.
        gru_in = hidden_size + (2 if ablation_level == 4 else 0)
        self.gru = nn.GRU(gru_in, hidden_size, batch_first=True)
        self.film = AdaLNFiLM(obj_embed_dim, hidden_size)

        # Per-channel z-score stats for the [RMS, max-abs] force curve, set
        # from the train split by train.py via set_force_curve_stats().
        # Buffers, so they move with the model and land in the checkpoint.
        # Identity until set.
        self.register_buffer('force_curve_mean', torch.zeros(2))
        self.register_buffer('force_curve_std', torch.ones(2))

        # Heads
        self.band_head = nn.Linear(hidden_size, n_bands)
        self.gain_head = nn.Linear(hidden_size, 1)       # per-frame global gain
        if use_modal:
            self.modal_amp_head = nn.Linear(hidden_size, k_modal)
            self.modal_freq_head = nn.Linear(hidden_size, k_modal)
            self.modal_damp_head = nn.Linear(hidden_size, k_modal)
            self.modal_synth = DampedSinusoids(n_modes=k_modal, sr=sr)

        # Synthesiser with pre-baked loopable bands (deterministic, cached)
        bands = bake_noise_bands(sr=sr, n_bands=n_bands, seed=bake_seed,
                                 cache_dir=cache_dir)
        self.synth = NoiseBandSynth(bands, w=w)

    # conditioning
    def _static_cond(self, obj_emb, force_features):
        """Static conditioning vector (same level semantics as model.py)."""
        if self.ablation_level == 1:
            return obj_emb
        if force_features is None:
            raise ValueError(f"ablation level {self.ablation_level} needs force_features")
        if self.ablation_level == 2:
            return torch.cat([obj_emb, force_features[:, 0:1]], dim=1)
        return torch.cat([obj_emb, force_features], dim=1)   # levels 3 & 4

    def set_force_curve_stats(self, mean, std):
        """Set per-channel z-score stats (each shape (2,)) for the
        [RMS, max-abs] force curve, computed on the train split by train.py."""
        self.force_curve_mean.copy_(torch.as_tensor(mean).view(2))
        self.force_curve_std.copy_(torch.as_tensor(std).clamp(min=1e-8).view(2))

    @staticmethod
    def pool_force_curve(force_waveform, force_frames, w):
        """Pool the raw force window into a per-frame [RMS, max-abs] curve
        (B, force_frames, 2). Both channels are non-negative; a signed mean
        cancels about half of the peak frame's amplitude. force_waveform arrives
        raw, not pre-z-scored, in the DDSP path."""
        B = force_waveform.shape[0]
        frames = force_waveform.view(B, force_frames, w)      # (B, F, W)
        rms = frames.pow(2).mean(dim=-1).clamp(min=0).sqrt()  # (B, F)
        maxabs = frames.abs().amax(dim=-1)                    # (B, F)
        return torch.stack([rms, maxabs], dim=-1)             # (B, F, 2)

    def _force_curve(self, force_waveform):
        """L4 conditioning: pool the raw 200 ms force window into a per-frame
        [RMS, max-abs] curve (B, n_frames, 2) and z-score each channel with
        the train-split stats. At 200 ms and W=32, force_frames == n_frames,
        so nothing is padded."""
        curve = self.pool_force_curve(force_waveform, self.force_frames, self.w)
        curve = (curve - self.force_curve_mean) / self.force_curve_std
        pad = self.n_frames - self.force_frames
        if pad > 0:
            # reached only when the force window is shorter than the signal
            curve = F.pad(curve.transpose(1, 2), (0, pad)).transpose(1, 2)
        return curve                                          # (B, n_frames, 2)

    # forward
    def forward(self, obj_id, force_features=None, force_waveform=None,
                loop_offsets=None):
        obj_emb = self.obj_embed(obj_id)                      # (B, 16)
        cond = self.cond_mlp(self._static_cond(obj_emb, force_features))
        cond_seq = cond.unsqueeze(1).expand(-1, self.n_frames, -1)

        if self.ablation_level == 4:
            if force_waveform is None:
                raise ValueError("ablation level 4 needs force_waveform")
            cond_seq = torch.cat([cond_seq, self._force_curve(force_waveform)],
                                 dim=-1)                      # (B, T, H+2)

        gru_out, _ = self.gru(cond_seq)                       # (B, T, H)
        h = self.film(gru_out, obj_emb)                       # adaLN(γ,β) by obj

        # Noise-band amplitudes (modified sigmoid, NBN Eq. 7.4 style),
        # scaled by a per-frame global gain envelope
        band_amps = modified_sigmoid(self.band_head(h))       # (B, T, N)
        gain = modified_sigmoid(self.gain_head(h))            # (B, T, 1)
        band_amps = band_amps * gain
        wav = self.synth(band_amps, loop_offsets=loop_offsets)

        aux = {'band_amps': band_amps, 'gain': gain.squeeze(-1)}

        if self.use_modal:
            pooled = h.mean(dim=1)                            # (B, H) global
            m_amp = modified_sigmoid(self.modal_amp_head(pooled))
            f_lo, f_hi = MODAL_FREQ_RANGE
            m_freq = f_lo + (f_hi - f_lo) * torch.sigmoid(self.modal_freq_head(pooled))
            d_lo, d_hi = MODAL_DAMP_RANGE
            m_damp = d_lo + (d_hi - d_lo) * torch.sigmoid(self.modal_damp_head(pooled))
            wav = wav + self.modal_synth(m_amp, m_freq, m_damp, self.signal_samples)
            aux.update({'modal_amp': m_amp, 'modal_freq': m_freq,
                        'modal_damp': m_damp})

        return wav, aux

    @torch.no_grad()
    def generate(self, obj_id, force_features=None, force_waveform=None):
        self.eval()
        wav, _ = self.forward(obj_id, force_features, force_waveform)
        return wav


# Loss: multi-scale MRSTFT (sized for the ~9600-sample onset window)
# with automatic MPS->CPU fallback

class MRSTFTLoss(nn.Module):
    """Linear + log magnitude L1 over multiple STFT resolutions.

    If torch.stft fails on the current device, the STFTs permanently fall
    back to CPU from that point on; gradients flow through .to('cpu').
    """

    def __init__(self, scales=None):
        super().__init__()
        self.scales = list(scales) if scales is not None else list(MRSTFT_SCALES)
        self.cpu_fallback = False
        self._warned = False

    def _spec(self, x, scale):
        window = torch.hann_window(scale, device=x.device)
        return torch.stft(x, n_fft=scale, hop_length=scale // 4, window=window,
                          return_complex=True).abs()

    def forward(self, pred, target):
        if self.cpu_fallback and pred.device.type != 'cpu':
            pred, target = pred.cpu(), target.cpu()
        try:
            return self._loss(pred, target)
        except (RuntimeError, NotImplementedError) as err:
            if pred.device.type == 'cpu':
                raise
            if not self._warned:
                print(f"[MRSTFTLoss] torch.stft failed on {pred.device} "
                      f"({err}); falling back to CPU for the loss "
                      f"(gradients still flow).")
                self._warned = True
            self.cpu_fallback = True
            return self._loss(pred.cpu(), target.cpu())

    def _loss(self, pred, target):
        loss = pred.new_zeros(())
        for scale in self.scales:
            S_p = self._spec(pred, scale)
            S_t = self._spec(target, scale)
            loss = loss + F.l1_loss(S_p, S_t)                    # linear L1
            loss = loss + F.l1_loss(safe_log(S_p), safe_log(S_t))  # log L1
        return loss / len(self.scales)


# Factory

def build_model(n_objects: int, ablation_level: int, n_bands: int = N_BANDS,
                use_modal: bool = True, signal_samples: int = 9600,
                force_waveform_samples: int = 9600,
                cache_dir: Path = CACHE_DIR,
                identity: str = "obj",
                n_materials: int = 7,
                force_feature_dim: int = 6) -> ForceConditionedFilterbankDDSP:
    """Factory used by train.py's --backbone ddsp branch."""
    return ForceConditionedFilterbankDDSP(
        n_objects=n_objects,
        ablation_level=ablation_level,
        n_bands=n_bands,
        use_modal=use_modal,
        signal_samples=signal_samples,
        force_waveform_samples=force_waveform_samples,
        cache_dir=cache_dir,
        identity=identity,
        n_materials=n_materials,
        force_feature_dim=force_feature_dim,
    )


if __name__ == "__main__":
    # Quick check: forward/backward at all four ablation levels, small bands
    torch.manual_seed(42)
    B, T = 4, 9600
    loss_fn = MRSTFTLoss()

    for level in [1, 2, 3, 4]:
        model = build_model(n_objects=100, ablation_level=level, n_bands=256)
        n_params = sum(p.numel() for p in model.parameters())
        obj_id = torch.randint(0, 100, (B,))
        feats = torch.randn(B, 6)
        fwav = torch.randn(B, 9600)      # 200 ms force window @48 kHz
        wav, aux = model(obj_id, feats if level >= 2 else None,
                         fwav if level == 4 else None)
        target = torch.randn(B, T) * 0.1
        loss = loss_fn(wav, target)
        loss.backward()
        grads = sum(1 for p in model.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
        total = sum(1 for _ in model.parameters())
        print(f"L{level}: params={n_params:,} wav={tuple(wav.shape)} "
              f"loss={loss.item():.4f} grads={grads}/{total}")
