"""Shape and parameter-budget checks for the two generators (CPU, no data)."""

import pytest
import torch

from model import ConditionalMelVAE
from model_ddsp import ForceConditionedFilterbankDDSP

LEVELS = (1, 2, 3, 4)
BATCH = 2
N_OBJECTS = 100
N_MELS, N_FRAMES = 128, 38

# Tiny DDSP settings: a short window and few bands keep the test fast.
DDSP_SMALL = dict(n_bands=16, signal_samples=320, force_waveform_samples=320)

# Parameter growth must stay well under this fraction of the level-1 budget.
MAX_GROWTH = 0.03


def cvae_inputs():
    return dict(
        mel=torch.randn(BATCH, N_MELS, N_FRAMES),
        obj_id=torch.randint(0, N_OBJECTS, (BATCH,)),
        force_features=torch.randn(BATCH, 6),
        force_waveform=torch.randn(BATCH, 9600),
    )


@pytest.mark.parametrize("level", LEVELS)
def test_cvae_forward_and_generate_produce_the_target_mel_shape(level):
    model = ConditionalMelVAE(
        n_mels=N_MELS, n_frames=N_FRAMES, n_objects=N_OBJECTS, ablation_level=level
    )
    args = cvae_inputs()
    recon, mu, logvar = model(**args)
    assert recon.shape == (BATCH, N_MELS, N_FRAMES)
    assert mu.shape == logvar.shape == (BATCH, model.latent_dim)

    args.pop("mel")
    gen = model.generate(n_samples=BATCH, **args)
    assert gen.shape == (BATCH, N_MELS, N_FRAMES)
    assert not model.training


@pytest.mark.parametrize("level", LEVELS)
def test_ddsp_forward_returns_a_waveform_of_the_requested_length(level, tmp_path):
    model = ForceConditionedFilterbankDDSP(
        n_objects=N_OBJECTS, ablation_level=level, cache_dir=tmp_path, **DDSP_SMALL
    )
    wav, aux = model(
        obj_id=torch.randint(0, N_OBJECTS, (BATCH,)),
        force_features=torch.randn(BATCH, 6),
        force_waveform=torch.randn(BATCH, DDSP_SMALL["force_waveform_samples"]),
    )
    assert wav.shape == (BATCH, DDSP_SMALL["signal_samples"])
    assert torch.isfinite(wav).all()
    assert aux["band_amps"].shape == (BATCH, model.n_frames, DDSP_SMALL["n_bands"])


def test_conditioning_adds_less_than_three_percent_of_parameters(tmp_path):
    def count(model):
        return sum(p.numel() for p in model.parameters())

    # sr only sizes the baked noise buffer, never a parameter tensor, so a
    # short buffer gives the same counts as the 48 kHz configuration.
    budgets = {
        "cvae": {
            lvl: count(
                ConditionalMelVAE(
                    n_mels=N_MELS,
                    n_frames=N_FRAMES,
                    n_objects=N_OBJECTS,
                    ablation_level=lvl,
                )
            )
            for lvl in LEVELS
        },
        "ddsp": {
            lvl: count(
                ForceConditionedFilterbankDDSP(
                    n_objects=N_OBJECTS,
                    ablation_level=lvl,
                    sr=1600,
                    signal_samples=320,
                    force_waveform_samples=320,
                    cache_dir=tmp_path,
                )
            )
            for lvl in LEVELS
        },
    }

    for backbone, n in budgets.items():
        growth = (n[4] - n[1]) / n[1]
        assert 0.0 < growth < MAX_GROWTH, f"{backbone}: {growth:.4%}"
        assert n[1] < n[2] < n[3] < n[4], f"{backbone}: {n}"
