"""Consistency checks over the result files in results/: stored verdicts follow the decision rule documented in src/evaluate.py, and the ladder headlines equal the main-grid levels."""

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULT_FILES = [
    f"results_{backbone}_{split}_test.json"
    for backbone in ("ddsp", "cvae")
    for split in ("within", "heldout")
]
REQUIRED_KEYS = {
    "delta_Li_minus_L1_mean",
    "ci95",
    "cohens_d",
    "cliffs_delta",
    "p_raw",
    "p_holm",
    "significant_holm_0.05",
    "interpretation",
    "verdict",
}
D_MIN = 0.2
ALPHA = 0.05


def load_comparisons(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)["primary_comparisons"]


def expected_verdict(c):
    """The decision rule as documented in src/evaluate.py."""
    d_ok = abs(c["cohens_d"]) >= D_MIN
    p_ok = c["significant_holm_0.05"]
    lo, hi = c["ci95"]
    ci_excludes_0 = (hi < 0) or (lo > 0)
    delta = c["delta_Li_minus_L1_mean"]
    if d_ok and p_ok and delta < 0:
        return "FORCE_HELPS"
    if d_ok and p_ok and delta > 0:
        return "FORCE_HARMS"
    if (not ci_excludes_0) and abs(c["cohens_d"]) < D_MIN:
        return "NO_DETECTED_DIFFERENCE"
    return "INCONCLUSIVE"


@pytest.mark.parametrize("filename", RESULT_FILES)
def test_every_comparison_has_the_expected_fields(filename):
    comparisons = load_comparisons(filename)
    assert set(comparisons) == {"2", "3", "4"}
    for level, c in comparisons.items():
        assert REQUIRED_KEYS <= set(c), f"L{level} missing {REQUIRED_KEYS - set(c)}"
        assert len(c["ci95"]) == 2


@pytest.mark.parametrize("filename", RESULT_FILES)
def test_stored_verdict_matches_the_decision_rule(filename):
    for level, c in load_comparisons(filename).items():
        assert c["verdict"] == expected_verdict(c), f"L{level} in {filename}"


@pytest.mark.parametrize("filename", RESULT_FILES)
def test_verdict_does_not_contradict_the_interpretation(filename):
    for level, c in load_comparisons(filename).items():
        if "worse than L1" in c["interpretation"]:
            assert "HELPS" not in c["verdict"], f"L{level} in {filename}"
        assert c["significant_holm_0.05"] is (c["p_holm"] < ALPHA), (
            f"L{level} in {filename}"
        )


LADDER_MAP = {"L1": "1", "L2a": "2", "L3a": "3", "L4": "4"}


@pytest.mark.parametrize("split", ["within", "heldout"])
def test_ladder_headline_matches_main_grid_levels(split):
    ladder = json.load(open(RESULTS_DIR / f"ladder_ddsp_{split}_test.json"))[
        "headline_object_first_fad"]
    grid = json.load(open(RESULTS_DIR / f"results_ddsp_{split}_test.json"))["levels"]
    for level, grid_key in LADDER_MAP.items():
        assert ladder[level] == grid[grid_key]["object_first_fad_mean"], (
            f"{level} in ladder_ddsp_{split}_test.json"
        )
