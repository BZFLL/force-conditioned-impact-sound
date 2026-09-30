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


def test_cvae_level5_is_identity_plus_frames_without_summary():
    """Level 5 on the CVAE: identity only on the static route, L4's global
    32-D code of the force window on the force route, and no descriptor —
    which must hold even when train.py passes force_features anyway."""
    m5 = ConditionalMelVAE(n_mels=N_MELS, n_frames=N_FRAMES,
                           n_objects=N_OBJECTS, ablation_level=5)
    assert m5.cond_dim == 16 + 32
    args = cvae_inputs()
    obj_id, fw = args["obj_id"], args["force_waveform"]
    c_with = m5._get_conditioning(obj_id, args["force_features"], fw)
    c_without = m5._get_conditioning(obj_id, None, fw)
    assert c_with.shape == (BATCH, 48)
    assert torch.equal(c_with, c_without)     # the descriptor must not leak in
    with pytest.raises(ValueError):
        m5._get_conditioning(obj_id, None, None)


def test_ddsp_level5_is_identity_plus_frames_without_summary(tmp_path):
    """Level 5 (post-hoc frames-only control): the static route is L1's
    (identity only), the per-frame route is L4's; it runs without a
    descriptor and its parameter count sits strictly between L1 and L4."""
    def count(model):
        return sum(p.numel() for p in model.parameters())

    m5 = ForceConditionedFilterbankDDSP(
        n_objects=N_OBJECTS, ablation_level=5, cache_dir=tmp_path, **DDSP_SMALL
    )
    wav, _ = m5(
        obj_id=torch.randint(0, N_OBJECTS, (BATCH,)),
        force_features=None,
        force_waveform=torch.randn(BATCH, DDSP_SMALL["force_waveform_samples"]),
    )
    assert wav.shape == (BATCH, DDSP_SMALL["signal_samples"])
    assert torch.isfinite(wav).all()
    with pytest.raises(ValueError):
        m5(obj_id=torch.randint(0, N_OBJECTS, (BATCH,)), force_waveform=None)
    m1 = ForceConditionedFilterbankDDSP(
        n_objects=N_OBJECTS, ablation_level=1, cache_dir=tmp_path, **DDSP_SMALL
    )
    m4 = ForceConditionedFilterbankDDSP(
        n_objects=N_OBJECTS, ablation_level=4, cache_dir=tmp_path, **DDSP_SMALL
    )
    assert count(m1) < count(m5) < count(m4)


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
