"""Checks for the training-side variants of src/train.py (envelope-L1
auxiliary loss, init_from weight loading) and for the synthetic pretraining
corpus (synthetic_data.py). No dataset, checkpoints or IR simulation needed."""

import numpy as np
import pytest
import torch

import train
from model_ddsp import ForceConditionedFilterbankDDSP, MRSTFTLoss
from signal_metrics import rms_env
from synthetic_data import SyntheticImpactDataset, sample_burst

DDSP_SMALL = dict(n_bands=16, signal_samples=320, force_waveform_samples=320)


def _tiny_batch(batch=2, samples=320, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "obj_id": torch.randint(0, 100, (batch,), generator=g),
        "material_idx": torch.randint(0, 7, (batch,), generator=g),
        "force_features": torch.randn(batch, 6, generator=g),
        "force_waveform": torch.randn(batch, samples, generator=g),
        "waveform": torch.randn(batch, samples, generator=g),
    }


def test_rms_env_batched_matches_the_scalar_definition():
    x = torch.randn(3, 9600)
    for i in range(3):
        ref = rms_env(x[i])                      # signal_metrics definition
        got = train._rms_env_batched(x[i:i + 1], 240, 120)[0]
        assert torch.allclose(ref, got, atol=1e-6)


def test_env_l1_is_zero_for_identical_signals_and_positive_apart():
    x = torch.randn(2, 4800)
    assert train._env_l1(x, x.clone(), 240, 120).item() == pytest.approx(0.0)
    assert train._env_l1(x, 2 * x, 240, 120).item() > 0


def test_ddsp_batch_loss_adds_the_envelope_term(tmp_path):
    torch.manual_seed(0)
    model = ForceConditionedFilterbankDDSP(n_objects=100, ablation_level=4,
                                           cache_dir=tmp_path, **DDSP_SMALL)
    model.eval()
    mrstft = MRSTFTLoss(scales=[64])   # small FFT: the batch is only 320 samples
    batch = _tiny_batch()
    kwargs = dict(feat_mean=torch.zeros(6), feat_std=torch.ones(6),
                  wav_scale=torch.tensor(1.0), id_key="obj_id")
    base = train._ddsp_batch_loss(model, mrstft, batch, "cpu", **kwargs,
                                  env_lambda=0.0)
    with torch.no_grad():
        obj = batch["obj_id"]
        pred, _ = model(obj, batch["force_features"], batch["force_waveform"])
        env = train._env_l1(pred, batch["waveform"], 240, 120)
    total = train._ddsp_batch_loss(model, mrstft, batch, "cpu", **kwargs,
                                   env_lambda=8.8)
    assert total.item() == pytest.approx(base.item() + 8.8 * env.item(),
                                         rel=1e-5)


def test_init_from_skips_mismatched_and_statistics_tensors(tmp_path):
    torch.manual_seed(0)
    src_model = ForceConditionedFilterbankDDSP(
        n_objects=100, ablation_level=4, identity="material",
        cache_dir=tmp_path / "a", **DDSP_SMALL)
    src_model.set_force_curve_stats(torch.tensor([3.0, 4.0]),
                                    torch.tensor([5.0, 6.0]))
    ckpt = tmp_path / "run_a"
    ckpt.mkdir()
    torch.save({"epoch": 7, "model_state_dict": src_model.state_dict()},
               ckpt / "best_model.pt")

    torch.manual_seed(0)
    dst = ForceConditionedFilterbankDDSP(n_objects=100, ablation_level=4,
                                         identity="obj",  # 100 rows vs 7
                                         cache_dir=tmp_path / "b",
                                         **DDSP_SMALL)
    ref = dst.state_dict()
    skipped = train.load_init_weights(dst, ckpt)   # directory form accepted
    got = dst.state_dict()

    assert any(k.startswith("obj_embed.weight") for k in skipped)
    assert any(k.startswith("force_curve_") for k in skipped)
    # the identity embedding and the stats buffers keep the destination values
    assert torch.equal(got["obj_embed.weight"], ref["obj_embed.weight"])
    assert torch.equal(got["force_curve_mean"], ref["force_curve_mean"])
    # every other tensor comes from the checkpoint
    for k in ref:
        if k.startswith(("obj_embed.weight", "force_curve_")):
            continue
        assert torch.equal(got[k], src_model.state_dict()[k]), k


def test_synthetic_corpus_shapes_and_epoch_redraw():
    pool = np.zeros((2, 9600), dtype=np.float64)
    pool[:, 0] = 1.0                              # delta IRs: target = burst
    ds = SyntheticImpactDataset(pool, base_seed=7, n_train=8, n_val=2,
                                n_test=2)
    assert len(ds) == 12
    s = ds[0]
    assert s["mel"].shape == (128, 38)
    assert s["force_waveform"].shape == (9600,)
    assert s["waveform"].shape == (9600,)
    assert s["force_features"].shape == (6,)

    again = ds[0]
    assert torch.equal(s["waveform"], again["waveform"])   # cached in epoch
    ds.set_epoch(1)
    redrawn = ds[0]
    assert not torch.equal(s["waveform"], redrawn["waveform"])  # train redrawn
    val_before = ds[8]["waveform"]                    # first val index
    ds.set_epoch(2)
    assert torch.equal(val_before, ds[8]["waveform"])       # val fixed


def test_sample_burst_onset_and_peak():
    rng = np.random.default_rng(0)
    b = sample_burst(rng)
    assert b.shape == (9600,)
    assert np.abs(b[:240]).max() == 0.0              # silence before 5 ms
    assert 0.05 <= np.abs(b).max() <= 0.5            # recorded peak range
