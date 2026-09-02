"""Structural checks on the shipped split files (no dataset needed)."""

import json
from pathlib import Path

import pytest

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
PARTITIONS = ("train", "val", "test")

# Expected partition sizes of the split files in splits/.
DOCUMENTED = {
    "within": {"train": 2572, "val": 488, "test": 488},
    "heldout": {"train": 2936, "val": 304, "test": 308},
}
TOTAL_STRIKES = 3548


def load_splits(mode):
    with open(SPLITS_DIR / f"splits_{mode}.json") as f:
        return json.load(f)["splits"]


def strike_keys(partition):
    """All (object, strike) pairs of one partition."""
    return [(obj, sid) for obj, sids in partition.items() for sid in sids]


@pytest.mark.parametrize("mode", ["within", "heldout"])
def test_every_strike_belongs_to_exactly_one_partition(mode):
    splits = load_splits(mode)
    keys = {p: set(strike_keys(splits[p])) for p in PARTITIONS}
    for p in PARTITIONS:
        assert len(keys[p]) == len(strike_keys(splits[p])), f"duplicate strike in {p}"
    assert keys["train"] & keys["val"] == set()
    assert keys["train"] & keys["test"] == set()
    assert keys["val"] & keys["test"] == set()
    assert len(keys["train"] | keys["val"] | keys["test"]) == sum(
        len(k) for k in keys.values()
    )


@pytest.mark.parametrize("mode", ["within", "heldout"])
def test_partition_sizes_match_the_documented_counts(mode):
    splits = load_splits(mode)
    sizes = {p: len(strike_keys(splits[p])) for p in PARTITIONS}
    assert sizes == DOCUMENTED[mode]
    assert sum(sizes.values()) == TOTAL_STRIKES


def test_both_split_files_cover_the_same_strikes():
    within = load_splits("within")
    heldout = load_splits("heldout")
    within_keys = {k for p in PARTITIONS for k in strike_keys(within[p])}
    heldout_keys = {k for p in PARTITIONS for k in strike_keys(heldout[p])}
    assert len(within_keys) == len(heldout_keys) == TOTAL_STRIKES
    assert within_keys == heldout_keys


def test_object_overlap_matches_the_split_semantics():
    heldout = load_splits("heldout")
    h_train = set(heldout["train"])
    assert h_train & set(heldout["val"]) == set()
    assert h_train & set(heldout["test"]) == set()

    within = load_splits("within")
    w_train = set(within["train"])
    assert set(within["val"]) <= w_train
    assert set(within["test"]) <= w_train
