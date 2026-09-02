"""
Generate the train/val/test split files (both split modes).

Outputs (written to <project>/splits/, shared by all ablation levels):

  splits_heldout.json  (unseen objects, material-stratified).
      Per material, ~15% of objects (at least 1; Glass has only 2 objects,
      so 1 is held out) are held out entirely from train. Each held-out
      object's impacts are split 50/50 between val and test, so every
      held-out material is covered in both val and test. Objects in train
      never appear in val/test.

  splits_within.json   (seen objects, within-object).
      Each object's impacts are split 70/15/15 into train/val/test
      (shuffle per object; first 15% test, next 15% val, rest train).

Both files are generated with a fixed seed (42) and store explicit
obj_id -> [impact_id] lists, so every ablation level trains/evaluates on
identical splits.

Usage:
    python3 make_splits.py
"""

import os
import sys
import csv
import glob
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import DATA_ROOT, OBJECTS_CSV  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "splits")

SEED = 42
HELDOUT_OBJ_RATIO = 0.15   # fraction of objects per material held out (>=1)
WITHIN_VAL_RATIO = 0.15
WITHIN_TEST_RATIO = 0.15


def scan_dataset(data_root):
    """Return {obj_id(int): [impact_id(str), ...]} for impacts that have both wavs."""
    impacts_by_obj = {}
    for obj_dir in sorted(glob.glob(os.path.join(data_root, "[0-9]*"))):
        if not os.path.isdir(obj_dir):
            continue
        obj_id = int(os.path.basename(obj_dir))
        impacts = []
        for impact_dir in sorted(glob.glob(os.path.join(obj_dir, "audio", "*"))):
            # Skip deprecated takes (e.g. 24/audio/13_deprecated); must mirror
            # the same filter in dataset.py so split ids stay valid.
            if not os.path.basename(impact_dir).isdigit():
                continue
            if (os.path.exists(os.path.join(impact_dir, "Force.wav"))
                    and os.path.exists(os.path.join(impact_dir, "mic.wav"))):
                impacts.append(os.path.basename(impact_dir))
        if impacts:
            impacts_by_obj[obj_id] = impacts
    return impacts_by_obj


def load_materials(csv_path, obj_ids):
    """Material labels: col0 = obj_id, col3 = material."""
    materials = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip().isdigit():
                continue
            materials[int(row[0])] = row[3].strip()
    out = {}
    for obj_id in obj_ids:
        if obj_id not in materials:
            print(f"WARNING: obj {obj_id} not in objects.csv, material='Unknown'")
            out[obj_id] = "Unknown"
        else:
            out[obj_id] = materials[obj_id]
    return out


def make_heldout(impacts_by_obj, materials, rng):
    """Object-held-out split, stratified by material."""
    by_material = {}
    for obj_id in sorted(impacts_by_obj):
        by_material.setdefault(materials[obj_id], []).append(obj_id)

    heldout_by_material = {}
    train, val, test = {}, {}, {}

    for material in sorted(by_material):
        objs = sorted(by_material[material])
        shuffled = list(objs)
        rng.shuffle(shuffled)
        n_hold = max(1, int(round(HELDOUT_OBJ_RATIO * len(objs))))
        heldout = sorted(shuffled[:n_hold])
        heldout_by_material[material] = heldout

        for obj_id in objs:
            impacts = list(impacts_by_obj[obj_id])
            if obj_id in heldout:
                # Entire object excluded from train; impacts 50/50 val/test.
                rng.shuffle(impacts)
                n_val = len(impacts) // 2
                val[str(obj_id)] = sorted(impacts[:n_val])
                test[str(obj_id)] = sorted(impacts[n_val:])
            else:
                train[str(obj_id)] = sorted(impacts)

    return {
        "mode": "heldout",
        "seed": SEED,
        "heldout_obj_ratio": HELDOUT_OBJ_RATIO,
        "heldout_objects_by_material": heldout_by_material,
        "splits": {"train": train, "val": val, "test": test},
    }


def make_within(impacts_by_obj, rng):
    """Within-object split: per object 70/15/15 (test cut first, then val)."""
    train, val, test = {}, {}, {}
    for obj_id in sorted(impacts_by_obj):
        impacts = list(impacts_by_obj[obj_id])
        rng.shuffle(impacts)
        n = len(impacts)
        n_test = max(1, int(n * WITHIN_TEST_RATIO))
        n_val = max(1, int(n * WITHIN_VAL_RATIO))
        test[str(obj_id)] = sorted(impacts[:n_test])
        val[str(obj_id)] = sorted(impacts[n_test:n_test + n_val])
        train[str(obj_id)] = sorted(impacts[n_test + n_val:])
    return {
        "mode": "within",
        "seed": SEED,
        "val_ratio": WITHIN_VAL_RATIO,
        "test_ratio": WITHIN_TEST_RATIO,
        "splits": {"train": train, "val": val, "test": test},
    }


def summarize(spec, materials):
    print(f"\n=== {spec['mode']} (seed={spec['seed']}) ===")
    for split in ("train", "val", "test"):
        part = spec["splits"][split]
        n_obj = len(part)
        n_imp = sum(len(v) for v in part.values())
        mats = sorted(set(materials[int(o)] for o in part))
        print(f"  {split:5s}: {n_obj:3d} objects, {n_imp:4d} impacts, "
              f"materials={mats}")
    if spec["mode"] == "heldout":
        print("  held-out objects by material:")
        for m, objs in sorted(spec["heldout_objects_by_material"].items()):
            print(f"    {m:14s}: {objs}")


def main():
    impacts_by_obj = scan_dataset(DATA_ROOT)
    n_obj = len(impacts_by_obj)
    n_imp = sum(len(v) for v in impacts_by_obj.values())
    print(f"Scanned {DATA_ROOT}: {n_obj} objects, {n_imp} impacts")

    materials = load_materials(OBJECTS_CSV, sorted(impacts_by_obj))
    mat_counts = {}
    for m in materials.values():
        mat_counts[m] = mat_counts.get(m, 0) + 1
    print(f"Material distribution: {dict(sorted(mat_counts.items()))}")

    os.makedirs(OUT_DIR, exist_ok=True)

    heldout = make_heldout(impacts_by_obj, materials, np.random.RandomState(SEED))
    within = make_within(impacts_by_obj, np.random.RandomState(SEED))

    for spec, name in [(heldout, "splits_heldout.json"), (within, "splits_within.json")]:
        path = os.path.abspath(os.path.join(OUT_DIR, name))
        with open(path, "w") as f:
            json.dump(spec, f, indent=1)
        summarize(spec, materials)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
