#!/usr/bin/env python3
"""Regenerate the curated metrics examples in examples/.

Produces a small, committed set of example metrics captures
demonstrating what pymurmur's full FlockMetrics schema actually looks
like across a few contrasting configs — unlike output/ (gitignored,
whatever a user last ran), these are checked into the repo so anyone
can see real output without running the simulator themselves.

Mirrors test/regenerate_golden.py's structure. Reuses
pymurmur.capture.recorder.Recorder for the actual metrics export
(save_metrics_csv/save_metrics_json) rather than hand-rolling it —
Recorder.on_frame only touches the GPU/Visualizer when
capture_with_viz is true, so this script stays fully headless by
setting it False.

Usage:
    python scripts/generate_examples.py                # all 3 targets
    python scripts/generate_examples.py --name showcase # single target
    python scripts/generate_examples.py --dry-run       # print what would happen

Generated files:
    examples/metrics_projection.json / .csv   (default SimConfig() baseline)
    examples/metrics_influencer.json / .csv   (conf/murmuration_influencer.yaml —
                                                the only config where
                                                target_dist_min/max populate)
    examples/metrics_showcase.json / .csv     (conf/murmuration_showcase.yaml —
                                                extensions/kernels/sphere_soft/
                                                hash_grid all exercised)

Determinism: every target is re-seeded to DEFAULT_SEED regardless of
what the source preset specifies, so regeneration is reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure pymurmur is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymurmur.capture.recorder import Recorder
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

DEFAULT_SEED = 42
DEFAULT_FRAMES = 180
DEFAULT_METRICS_INTERVAL = 20
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples"
DT = 1.0 / 60.0

# (name, source preset path or None for SimConfig() defaults)
EXAMPLE_TARGETS: list[tuple[str, str | None]] = [
    ("projection", None),
    ("influencer", "conf/murmuration_influencer.yaml"),
    ("showcase", "conf/murmuration_showcase.yaml"),
]


def generate_example(
    name: str,
    source: str | None,
    seed: int = DEFAULT_SEED,
    frames: int = DEFAULT_FRAMES,
    metrics_interval: int = DEFAULT_METRICS_INTERVAL,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Generate one example's metrics CSV + JSON.

    Args:
        name: example name, used for the output filenames.
        source: path to a conf/*.yaml preset, or None for SimConfig() defaults.
        seed: RNG seed (overrides whatever the source preset specifies).
        frames: number of simulation steps to run.
        metrics_interval: frames between expensive/gated metric computations.
        output_dir: directory for the .csv/.json files.

    Returns:
        (csv_path, json_path)
    """
    cfg = SimConfig.from_file(source) if source else SimConfig()
    cfg.seed = seed
    cfg.metrics_detail_level = 2
    cfg.metrics_interval = metrics_interval
    cfg.capture_with_viz = False  # headless: never touches GPU/Visualizer
    cfg.capture_frames = frames
    cfg.capture_prewarm = 0
    if cfg.visual_range <= 0:
        # conf/murmuration_influencer.yaml ships visual_range=0.0 (influencer
        # mode never queries neighbors, so it's a documented no-op value) --
        # but SimConfig.validate() rejects visual_range<=0 unconditionally.
        # This is a pre-existing bug in the shipped preset/validation rule,
        # out of scope here; work around it since the override has zero
        # physics effect on influencer mode specifically.
        cfg.visual_range = 1.0
    cfg.validate()

    csv_path = output_dir / f"metrics_{name}.csv"
    json_path = output_dir / f"metrics_{name}.json"
    cfg.capture_metrics_csv = str(csv_path)
    cfg.capture_metrics_json = str(json_path)

    sim = SimulationEngine(cfg)
    rec = Recorder(sim, cfg)
    sim.run_headless(steps=frames, callback=rec.on_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_csv = rec.save_metrics_csv()
    saved_json = rec.save_metrics_json()
    if saved_csv is None or saved_json is None:
        raise RuntimeError(f"{name}: no metrics were captured")

    return Path(saved_csv), Path(saved_json)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate curated example metrics captures for pymurmur.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--name",
        choices=[n for n, _ in EXAMPLE_TARGETS] + ["all"],
        default="all",
        help="Example to regenerate (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--metrics-interval", type=int, default=DEFAULT_METRICS_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without generating files.",
    )
    args = parser.parse_args()

    targets = EXAMPLE_TARGETS if args.name == "all" else [
        (n, s) for n, s in EXAMPLE_TARGETS if n == args.name
    ]

    print("Example metrics regeneration")
    print(f"  Seed:             {args.seed}")
    print(f"  Frames:           {args.frames}")
    print(f"  Metrics interval: {args.metrics_interval}")
    print(f"  Output:           {args.output_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would generate:")
        for name, source in targets:
            print(f"  {args.output_dir / f'metrics_{name}.csv'}  (source: {source or 'defaults'})")
            print(f"  {args.output_dir / f'metrics_{name}.json'}")
        return 0

    generated = []
    errors = []
    for name, source in targets:
        try:
            csv_path, json_path = generate_example(
                name, source,
                seed=args.seed, frames=args.frames,
                metrics_interval=args.metrics_interval,
                output_dir=args.output_dir,
            )
            print(f"  ✓ {name:12s} → {csv_path.name}, {json_path.name}")
            generated.append(name)
        except Exception as e:
            print(f"  ✗ {name:12s} FAILED: {e}")
            errors.append(name)

    print()
    if generated:
        print(f"✓ {len(generated)} example(s) generated")
    if errors:
        print(f"✗ {len(errors)} example(s) failed: {', '.join(errors)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
