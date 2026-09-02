"""Closed-form checks on the statistics in src/evaluate.py (no encoder, no data)."""

import math

import numpy as np
import pytest

import evaluate


def test_frechet_distance_matches_the_closed_form():
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(32, 4))
    mu, sig = evaluate.gaussian_stats(embs)
    assert evaluate.frechet_distance(mu, sig, mu, sig) == pytest.approx(0.0, abs=1e-6)

    # Diagonal covariances, so the distance is the per-dimension sum of
    # |mu1 - mu2| ** 2 + (s1 - s2) ** 2. Both sets have exact moments:
    # mean 0 and 3, variance 2/3 and 8/3 in each of the two dimensions.
    a = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    b = 2.0 * a + np.array([3.0, 0.0])
    mu1, sig1 = evaluate.gaussian_stats(a)
    mu2, sig2 = evaluate.gaussian_stats(b)
    s1, s2 = math.sqrt(2.0 / 3.0), math.sqrt(8.0 / 3.0)
    expected = 3.0 ** 2 + 2.0 * (s1 - s2) ** 2
    assert evaluate.frechet_distance(mu1, sig1, mu2, sig2) == pytest.approx(
        expected, abs=1e-4
    )


def test_per_object_fad_returns_one_entry_per_object():
    obj_ids = [o for o in range(3) for _ in range(4)]
    base = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5], [0.5, -1.0]])
    embs = np.concatenate([base + 100.0 * o for o in range(3)])
    fad = evaluate.per_object_fad(embs, embs.copy(), obj_ids)
    assert set(fad) == {0, 1, 2}
    # Per object the two sets are identical, so the well-separated object
    # offsets must not leak into the scores.
    assert all(v == pytest.approx(0.0, abs=1e-6) for v in fad.values())


def test_holm_correction_reproduces_the_textbook_result():
    adjusted = evaluate.holm_correction({2: 0.01, 3: 0.04, 4: 0.03})
    assert adjusted[2] == pytest.approx(0.03)   # 3 * 0.01
    assert adjusted[4] == pytest.approx(0.06)   # 2 * 0.03
    assert adjusted[3] == pytest.approx(0.06)   # 1 * 0.04, raised to stay monotone
    assert evaluate.holm_correction({2: 0.5, 3: 0.9})[3] == pytest.approx(1.0)


def test_cohens_d_is_the_mean_over_the_sample_sd():
    assert evaluate.cohens_d([-1.0, 1.0, 3.0]) == pytest.approx(0.5)  # mean 1, sd 2
    vals = [0.4, -0.1, 0.9, 0.2, 0.7]
    assert evaluate.cohens_d(vals) == pytest.approx(
        np.mean(vals) / np.std(vals, ddof=1)
    )
