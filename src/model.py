"""Force-conditioned mel-spectrogram CVAE.

Four force-conditioning levels:
  1. identity embedding only (baseline)
  2. + scalar peak force
  3. + 6-D force descriptor [peak, fwhm, bounce_count, max_bounce, tau, asymmetry]
  4. + raw force waveform via a 1D CNN encoder

The identity embedding is a conditioning input at all four levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForceWaveformEncoder(nn.Module):
    """1-D CNN encoder for the raw force window (conditioning level 4).

    input_samples sizes no layer: the global average pool makes the module
    length-agnostic. It documents the window used, 200 ms at 48 kHz = 9,600
    samples."""

    def __init__(self, input_samples: int = 9600, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, samples) → (B, 1, samples)
        return self.net(x.unsqueeze(1))


class ConditionalMelVAE(nn.Module):
    """
    Conditional VAE over mel spectrograms; ablation_level (1-4) selects the
    force conditioning that is concatenated to the identity embedding.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_frames: int = 38,       # ~200ms onset at hop=256, sr=48000
        latent_dim: int = 32,
        n_objects: int = 7,
        obj_embed_dim: int = 16,
        ablation_level: int = 3,
        force_waveform_samples: int = 9600,  # 200 ms at 48 kHz
        identity: str = "obj",               # 'obj' | 'material'
        n_materials: int = 7,
        force_feature_dim: int = 6,          # L3 descriptor dim (v1=6)
    ):
        super().__init__()

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        self.ablation_level = ablation_level
        assert identity in ("obj", "material"), identity
        self.identity = identity

        # Identity embedding: 'obj' has n_objects rows, 'material' has
        # n_materials=7, both at obj_embed_dim, so the four-level ablation
        # stays architecturally identical. forward() receives the matching
        # id (obj_idx or material_idx) via `obj_id`.
        n_identity = n_objects if identity == "obj" else n_materials
        self.obj_embed = nn.Embedding(n_identity, obj_embed_dim)

        # Force conditioning dimension based on ablation level
        if ablation_level == 1:
            self.cond_dim = obj_embed_dim                    # 16
        elif ablation_level == 2:
            self.cond_dim = obj_embed_dim + 1                # 17
        elif ablation_level == 3:
            self.cond_dim = obj_embed_dim + force_feature_dim   # 22 when dim=6
        elif ablation_level == 4:
            self.force_wav_encoder = ForceWaveformEncoder(
                input_samples=force_waveform_samples, out_dim=32
            )
            self.cond_dim = obj_embed_dim + force_feature_dim + 32  # 54 when dim=6
        else:
            raise ValueError(f"Unknown ablation_level: {ablation_level}")

        # Encoder
        # Input: mel spectrogram (1, n_mels, n_frames) + conditioning
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # → (32, 64, 19)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (64, 32, 10)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → (128, 16, 5)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# → (128, 8, 3)
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Compute flattened conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            conv_out = self.enc_conv(dummy)
            self.conv_out_shape = conv_out.shape[1:]  # (C, H, W)
            conv_flat_dim = conv_out.numel()

        # Encoder FC: conv_flat + cond → μ, log_var
        self.enc_fc = nn.Sequential(
            nn.Linear(conv_flat_dim + self.cond_dim, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        # Input: z + conditioning → mel spectrogram
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + self.cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, conv_flat_dim),
            nn.ReLU(),
        )

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            # Linear output, no sigmoid: the targets are globally z-scored.
        )

        # Adaptive crop to exact output size
        self.target_h = n_mels
        self.target_w = n_frames

    def _get_conditioning(self, obj_id, force_features=None, force_waveform=None):
        """Build conditioning vector based on ablation level."""
        cond = self.obj_embed(obj_id)  # (B, obj_embed_dim)

        if self.ablation_level >= 2 and force_features is not None:
            if self.ablation_level == 2:
                # Scalar peak only
                cond = torch.cat([cond, force_features[:, 0:1]], dim=1)
            elif self.ablation_level >= 3:
                # Full 6-dim features
                cond = torch.cat([cond, force_features], dim=1)

        if self.ablation_level == 4 and force_waveform is not None:
            wav_feat = self.force_wav_encoder(force_waveform)
            cond = torch.cat([cond, wav_feat], dim=1)

        return cond

    def encode(self, mel, cond):
        """Encode mel spectrogram + conditioning → μ, log_var."""
        # mel: (B, n_mels, n_frames) → (B, 1, n_mels, n_frames)
        x = mel.unsqueeze(1)
        x = self.enc_conv(x)
        x = x.flatten(1)  # (B, conv_flat_dim)
        x = torch.cat([x, cond], dim=1)
        h = self.enc_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z, cond):
        """Decode latent + conditioning → mel spectrogram."""
        x = torch.cat([z, cond], dim=1)
        x = self.dec_fc(x)
        x = x.view(-1, *self.conv_out_shape)
        x = self.dec_conv(x)
        # Crop to exact target size
        x = x[:, :, :self.target_h, :self.target_w]
        return x.squeeze(1)  # (B, n_mels, n_frames)

    def forward(self, mel, obj_id, force_features=None, force_waveform=None):
        """
        Forward pass.

        Args:
            mel: (B, n_mels, n_frames) — target mel spectrogram
            obj_id: (B,) — object indices
            force_features: (B, 6) — [peak, fwhm, bounce_count, max_bounce, tau, asymmetry]
            force_waveform: (B, samples) — raw force waveform window
        """
        cond = self._get_conditioning(obj_id, force_features, force_waveform)
        mu, logvar = self.encode(mel, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def generate(self, obj_id, force_features=None, force_waveform=None, n_samples=1):
        """Generate mel spectrogram from conditioning (no encoder needed)."""
        self.eval()
        with torch.no_grad():
            cond = self._get_conditioning(obj_id, force_features, force_waveform)
            z = torch.randn(n_samples, self.latent_dim, device=cond.device)
            return self.decode(z, cond)


def cvae_loss(recon, target, mu, logvar, beta=1.0, free_bits=0.25):
    """
    CVAE loss = reconstruction + β * KL divergence with free bits.

    Free bits (Kingma et al., 2016): per-dimension KL is clamped to at least
    λ, so the model gains nothing by driving a dimension's KL to zero.

    Args:
        recon: reconstructed mel spectrogram
        target: ground truth mel spectrogram
        mu, logvar: latent distribution parameters
        beta: KL weight (for β-VAE / cyclical annealing)
        free_bits: minimum KL per latent dimension (λ). 0 = no free bits.
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # Per-dimension KL: (B, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.mean()

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


if __name__ == "__main__":
    # Quick test all ablation levels
    B = 4
    n_mels, n_frames = 128, 38

    for level in [1, 2, 3, 4]:
        print(f"\n--- Ablation Level {level} ---")
        model = ConditionalMelVAE(
            n_mels=n_mels, n_frames=n_frames,
            ablation_level=level, n_objects=100
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # z-scored-like targets (unbounded)
        mel = torch.randn(B, n_mels, n_frames)
        obj_id = torch.randint(0, 100, (B,))
        force_feat = torch.randn(B, 6)
        force_wav = torch.randn(B, 9600)   # 200 ms force window @48 kHz

        recon, mu, logvar = model(mel, obj_id, force_feat, force_wav)
        loss, rl, kl = cvae_loss(recon, mel, mu, logvar)

        print(f"  Recon shape: {recon.shape}")
        print(f"  Loss: {loss.item():.4f} (recon={rl.item():.4f}, kl={kl.item():.4f})")
