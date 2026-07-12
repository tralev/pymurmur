#!/usr/bin/env python3
"""P0.1 — Golden trajectory regeneration script.

Generates deterministic golden-trajectory .npz files for all active simulation
modes. Run this after any deliberate physics change to re-pin the regression
baselines.  CI fails if goldens are stale (guarded by test/test_golden.py).

Usage:
    python test/regenerate_golden.py                  # All 5 modes
    python test/regenerate_golden.py --mode projection # Single mode
    python test/regenerate_golden.py --mode spatial --seed 42 --frames 60 --birds 20
    python test/regenerate_golden.py --dry-run          # Print what would happen

Generated files:
    test/data/golden_projection.npz   (30, 15, 3) float32 pos + vel
    test/data/golden_spatial.npz
    test/data/golden_field.npz
    test/data/golden_vicsek.npz
    test/data/golden_influencer.npz

Determinism contract (P0.4): Same seed + config → bit-identical after N frames.
Currently projection, spatial, and field are deterministic; vicsek and influencer
use module-level np.random.* calls and will become deterministic after P0.4
(flock.rng replaces all np.random.* call sites).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure pymurmur is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── Defaults (matching P0.1 spec) ─────────────────────────────────

GOLDEN_MODES = [
    "projection",
    "spatial",
    "field",
    "vicsek",
    "influencer",
    # TODO P5: add "angle" when angle mode ships
    # TODO P12: add "marl" when MARL mode ships
]
DEFAULT_SEED = 77
DEFAULT_BIRDS = 15
DEFAULT_FRAMES = 30
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DT = 1.0 / 60.0

# Modes known to be non-deterministic until P0.4 (module-level np.random.*)
NONDETERMINISTIC = {"vicsek", "influencer"}


def generate_golden(
    mode: str,
    seed: int = DEFAULT_SEED,
    birds: int = DEFAULT_BIRDS,
    frames: int = DEFAULT_FRAMES,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate a single golden .npz file for *mode*.

    Args:
        mode: One of 'projection', 'spatial', 'field', 'vicsek', 'influencer'.
        seed: RNG seed (default 77, matching P0.1 spec).
        birds: Number of birds (default 15).
        frames: Number of simulation steps (default 30).
        output_dir: Directory for the .npz file.

    Returns:
        Path to the generated file.

    Raises:
        ValueError: If *mode* is not in GOLDEN_MODES.
    """
    if mode not in GOLDEN_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: {', '.join(GOLDEN_MODES)}"
        )

    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = birds
    cfg.seed = seed
    cfg.use_numba = False  # Deterministic: no JIT

    engine = SimulationEngine(cfg)

    positions = []
    velocities = []

    for _ in range(frames):
        engine.step(DT)
        positions.append(engine.flock.positions.copy())
        velocities.append(engine.flock.velocities.copy())

    pos_stack = np.stack(positions).astype(np.float32)  # (frames, N, 3)
    vel_stack = np.stack(velocities).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    path = output_dir / f"golden_{mode}.npz"
    np.savez(path, pos=pos_stack, vel=vel_stack)

    return path


def verify_golden(path: Path, expected_shape: tuple = (DEFAULT_FRAMES, DEFAULT_BIRDS, 3)):
    """Verify a generated .npz file has the expected structure."""
    data = np.load(path)
    assert data["pos"].shape == expected_shape, (
        f"{path.name}: pos shape {data['pos'].shape}, expected {expected_shape}"
    )
    assert data["vel"].shape == expected_shape
    assert data["pos"].dtype == np.float32
    assert data["vel"].dtype == np.float32

    # Check no NaN/Inf
    assert np.isfinite(data["pos"]).all(), f"{path.name}: positions contain NaN/Inf"
    assert np.isfinite(data["vel"]).all(), f"{path.name}: velocities contain NaN/Inf"

    # Position range check: all values must be finite and within domain.
    # Toroidal wrapping means positions may be at either boundary; we just
    # verify they are not wildly out of range.
    pos_min = data["pos"].min()
    pos_max = data["pos"].max()
    assert pos_min >= -1000.0, f"{path.name}: positions out of range (min={pos_min})"
    assert pos_max <= 2000.0, f"{path.name}: positions out of range (max={pos_max})"


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate golden trajectory .npz files for pymurmur.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=GOLDEN_MODES + ["all"],
        default="all",
        help="Mode to regenerate (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--birds",
        type=int,
        default=DEFAULT_BIRDS,
        help=f"Number of birds (default: {DEFAULT_BIRDS}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of simulation steps (default: {DEFAULT_FRAMES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without generating files.",
    )
    args = parser.parse_args()

    modes = GOLDEN_MODES if args.mode == "all" else [args.mode]

    print(f"Golden trajectory regeneration")
    print(f"  Modes:   {', '.join(modes)}")
    print(f"  Seed:    {args.seed}")
    print(f"  Birds:   {args.birds}")
    print(f"  Frames:  {args.frames}")
    print(f"  Output:  {args.output_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would generate:")
        for mode in modes:
            path = args.output_dir / f"golden_{mode}.npz"
            nd = " ⚠️  non-deterministic (P0.4)" if mode in NONDETERMINISTIC else ""
            print(f"  {path}{nd}")
        return 0

    generated = []
    errors = []

    for mode in modes:
        try:
            path = generate_golden(
                mode,
                seed=args.seed,
                birds=args.birds,
                frames=args.frames,
                output_dir=args.output_dir,
            )
            verify_golden(path, (args.frames, args.birds, 3))
            size_kb = os.path.getsize(path) / 1024
            nd = " ⚠️  non-deterministic — will be fixed in P0.4" if mode in NONDETERMINISTIC else ""
            print(f"  ✓ {mode:15s} → {path.name:30s}  {size_kb:6.1f} KB{nd}")
            generated.append(mode)
        except Exception as e:
            print(f"  ✗ {mode:15s} FAILED: {e}")
            errors.append(mode)

    print()
    if generated:
        print(f"✓ {len(generated)} golden files generated")
    if errors:
        print(f"✗ {len(errors)} modes failed: {', '.join(errors)}")
        return 1

    if any(m in NONDETERMINISTIC for m in generated):
        print()
        print("⚠️  Note: vicsek and influencer modes use module-level np.random.*")
        print("    calls and are currently NON-DETERMINISTIC (structural gap #5).")
        print("    After P0.4 (flock.rng), re-run this script to re-pin them.")
        print("    Until then, test_golden.py marks them as expected failures (xfail).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
