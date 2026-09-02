#!/usr/bin/env python3
"""Build the pool of synthetic room impulse responses used for pretraining.

Hybrid image-source (order 3) + ray-tracing simulation with pyroomacoustics.
All parameters are recorded in
experiments/pretrain/pretrain_synthetic/pretrain_config.json (ir_pool_meta):

  fs 48000 Hz, ir_len 9600 samples (200 ms), max_order 3,
  absorption ~ U[0.05, 0.4], scattering ~ U[0.1, 0.5],
  room side lengths ~ U[3, 8] m, source/mic at least `margin` 0.5 m from
  walls and `min_dist` 0.5 m apart, n_rays 10000, pool of 2000 IRs from
  generator seed 4242.

Convention: the direct arrival is shifted to sample 0 and normalised to
gain 1; the pool keeps dense tails (no drop gate); the reflection/direct
energy ratio is stored per IR as refl_direct_db. Rooms whose RIR has no
usable direct arrival are dropped and redrawn (n_dropped_degenerate is
reported).

Output: cache/synthetic_ir_pool.pt  {"irs": (2000, 9600) float32,
"refl_direct_db": (2000,) float32, "meta": {...}}

Usage:
    python3 src/make_synthetic_ir_pool.py [--n 2000] [--out cache/synthetic_ir_pool.pt]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ir_pool_meta (see pretrain_config.json)
FS = 48000
IR_LEN = 9600                 # 200 ms
MAX_ORDER = 3
ABSORPTION_RANGE = (0.05, 0.4)
SCATTERING_RANGE = (0.1, 0.5)
ROOM_RANGE = (3.0, 8.0)
MARGIN = 0.5                  # min distance of source/mic from any wall (m)
MIN_DIST = 0.5                # min source-mic distance (m)
N_RAYS = 10000
POOL_SEED = 4242
N_IR = 2000
C = 343.0                     # speed of sound (m/s), for the direct-tap window


def _random_room(rng):
    """One randomised shoebox room; returns (room, src, mic)."""
    dims = rng.uniform(*ROOM_RANGE, size=3)
    absorption = rng.uniform(*ABSORPTION_RANGE)
    scattering = rng.uniform(*SCATTERING_RANGE)
    import pyroomacoustics as pra
    room = pra.ShoeBox(
        dims, fs=FS, max_order=MAX_ORDER,
        materials=pra.Material(absorption, scattering),
        ray_tracing=True, use_rand_ism=True,
    )
    room.set_ray_tracing(n_rays=N_RAYS)
    lo = np.array([MARGIN] * 3)
    hi = dims - MARGIN
    src = rng.uniform(lo, hi)
    while True:
        mic = rng.uniform(lo, hi)
        if np.linalg.norm(mic - src) >= MIN_DIST:
            break
    room.add_source(src)
    room.add_microphone(mic)
    return room


def _direct_window(ir, fs=FS):
    """Index of the direct arrival: first sample above 1% of the peak
    (robust to the fractional delay of the simulated direct tap)."""
    peak = np.abs(ir).max()
    if peak <= 0:
        return None
    above = np.flatnonzero(np.abs(ir) >= 0.01 * peak)
    return int(above[0]) if len(above) else None


def make_pool(n=N_IR, seed=POOL_SEED, verbose=True):
    """Simulate n IRs. Returns (irs (n, IR_LEN) float32, refl_direct_db (n,),
    n_dropped)."""
    rng = np.random.default_rng(seed)
    irs, ratios = [], []
    n_dropped = 0
    while len(irs) < n:
        room = _random_room(rng)
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float64)
        d = _direct_window(h)
        if d is None or not np.isfinite(h).all():
            n_dropped += 1
            continue
        h = h[d:]                                   # direct tap at t = 0
        if len(h) < IR_LEN:
            h = np.pad(h, (0, IR_LEN - len(h)))
        h = h[:IR_LEN]
        if h[0] == 0:                               # degenerate direct tap
            n_dropped += 1
            continue
        h = h / h[0]                                # direct gain = 1
        # reflection/direct energy ratio: direct = first propagation-time
        # window of the minimum source-mic distance, reflections = the rest
        direct_w = max(1, int(0.001 * FS))
        e_dir = float((h[:direct_w] ** 2).sum())
        e_refl = float((h[direct_w:] ** 2).sum())
        ratios.append(10.0 * np.log10((e_refl + 1e-15) / (e_dir + 1e-15)))
        irs.append(h.astype(np.float32))
        if verbose and len(irs) % 200 == 0:
            print(f"  {len(irs)}/{n} IRs (dropped {n_dropped})", flush=True)
    return np.stack(irs), np.array(ratios, dtype=np.float32), n_dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=N_IR)
    ap.add_argument("--seed", type=int, default=POOL_SEED)
    ap.add_argument("--out", type=str, default=str(ROOT / "cache" / "synthetic_ir_pool.pt"))
    args = ap.parse_args()

    irs, ratios, n_dropped = make_pool(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "irs": torch.from_numpy(irs),
        "refl_direct_db": torch.from_numpy(ratios),
        "meta": dict(fs=FS, ir_len=IR_LEN, max_order=MAX_ORDER,
                     absorption_range=list(ABSORPTION_RANGE),
                     scattering_range=list(SCATTERING_RANGE),
                     room_range=list(ROOM_RANGE), margin=MARGIN,
                     min_dist=MIN_DIST, n_rays=N_RAYS, seed=args.seed,
                     n_requested=args.n, n_dropped_degenerate=n_dropped),
    }, out)
    print(f"wrote {out}  ({len(irs)} IRs, {n_dropped} dropped)")


if __name__ == "__main__":
    main()
