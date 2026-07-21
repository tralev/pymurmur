#!/usr/bin/env python3
"""S8.4 — 24-hour release-gate soak.

Manual/release-lane companion to the nightly T6.3 soak
(`test/crosscutting/perf/test_performance.py::TestSoak`), which only
runs 20,000 frames because pytest CI lanes have a wall-clock budget.
This script runs headless for a wall-clock duration (default 24h) and
checks, throughout:

  - no NaN in positions/velocities
  - positions stay in-bounds
  - the speed contract holds (v <= v0 * 1.5)
  - Recorder ring-buffer caps are respected (D19)
  - no monotone memory growth after warm-up (tracemalloc, linear-fit
    slope on periodic samples)

Not wired into CI (a 24h job has no place in a PR/nightly lane) — run
by hand before a release:

    python scripts/soak_release_gate.py --hours 24

Use --hours with a small value (e.g. 0.01) for a smoke test of the
harness itself.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="S8.4 release-gate soak")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="Wall-clock duration in hours (default: 24)")
    parser.add_argument("--num-boids", type=int, default=500,
                        help="Flock size (default: 500, matches T6.3)")
    parser.add_argument("--mode", type=str, default="spatial",
                        help="Force mode (default: spatial)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-frames", type=int, default=500,
                        help="Frames per checked chunk (default: 500)")
    parser.add_argument("--frame-cap", type=int, default=1000,
                        help="Recorder ring-buffer cap (D19, default: 1000)")
    parser.add_argument("--warmup-chunks", type=int, default=2,
                        help="Chunks excluded from the memory-growth fit "
                             "while the ring buffers fill (default: 2). "
                             "Keep warmup_chunks * chunk_frames >= frame_cap "
                             "or the fit window will still be inside the "
                             "one-time ring-buffer fill-up and false-positive.")
    parser.add_argument("--max-growth", type=float, default=0.05,
                        help="Max allowed fractional memory growth across "
                             "the fitted (post-warmup) window (default: 0.05)")
    args = parser.parse_args()

    from pymurmur.capture.recorder import Recorder
    from pymurmur.core.config import SimConfig
    from pymurmur.simulation.engine import SimulationEngine

    cfg = SimConfig()
    cfg.mode = args.mode
    cfg.num_boids = args.num_boids
    cfg.seed = args.seed
    cfg.metrics_detail_level = 1
    cfg.capture_frame_cap = args.frame_cap
    cfg.history_cap = args.frame_cap  # D19: keep metrics history in lockstep
    cfg.capture_with_viz = False

    sim = SimulationEngine(cfg)
    rec = Recorder(sim, cfg)

    domain = np.array([cfg.width, cfg.height, cfg.depth], dtype=np.float32)
    max_speed_allowed = cfg.v0 * 1.5

    tracemalloc.start()
    samples: list[tuple[int, float]] = []  # (frame, traced_bytes)

    deadline = time.monotonic() + args.hours * 3600.0
    chunk = 0
    frames_done = 0
    t_start = time.monotonic()
    exit_code = 0

    try:
        while time.monotonic() < deadline:
            sim.run_headless(steps=args.chunk_frames, callback=rec.on_frame)
            frames_done += args.chunk_frames
            chunk += 1

            if np.any(np.isnan(sim.flock.positions)) or np.any(np.isnan(sim.flock.velocities)):
                raise AssertionError(f"NaN detected after {frames_done} frames")

            pos = sim.flock.positions
            if not (np.all(pos >= 0.0) and np.all(pos <= domain)):
                raise AssertionError(
                    f"Positions out of bounds [0, {domain}] after {frames_done} frames"
                )

            speeds = np.linalg.norm(sim.flock.velocities, axis=1)
            if np.any(speeds > max_speed_allowed):
                raise AssertionError(
                    f"Speed contract violated after {frames_done} frames: "
                    f"max={speeds.max():.2f} > {max_speed_allowed:.2f}"
                )

            if len(rec.metrics_history) > args.frame_cap or len(rec.frames) > args.frame_cap:
                raise AssertionError(
                    f"Ring-buffer cap ({args.frame_cap}) exceeded after "
                    f"{frames_done} frames: metrics={len(rec.metrics_history)}, "
                    f"frames={len(rec.frames)}"
                )

            current_bytes, _peak = tracemalloc.get_traced_memory()
            samples.append((frames_done, float(current_bytes)))

            elapsed = time.monotonic() - t_start
            print(
                f"[{elapsed / 3600.0:6.2f}h] frame={frames_done:>10,} "
                f"traced_mem={current_bytes / (1024 * 1024):8.2f} MB "
                f"max_speed={speeds.max():.2f}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nInterrupted — running final checks on partial run.", flush=True)
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", flush=True)
        exit_code = 1

    if exit_code == 0:
        fit_samples = samples[args.warmup_chunks:]
        if len(fit_samples) >= 2:
            xs = np.array([f for f, _ in fit_samples], dtype=np.float64)
            ys = np.array([b for _, b in fit_samples], dtype=np.float64)
            slope, intercept = np.polyfit(xs, ys, 1)
            fitted_start = intercept + slope * xs[0]
            fitted_end = intercept + slope * xs[-1]
            growth = (fitted_end - fitted_start) / fitted_start if fitted_start > 0 else 0.0
            print(
                f"\nMemory growth fit (post-warmup, {len(fit_samples)} samples): "
                f"{growth * 100:.2f}% (limit {args.max_growth * 100:.0f}%)"
            )
            if growth > args.max_growth:
                print(
                    f"FAIL: monotone memory growth {growth * 100:.2f}% exceeds "
                    f"{args.max_growth * 100:.0f}% budget"
                )
                exit_code = 1
        else:
            print("\nToo few samples for a memory-growth fit (run was too short).")

    tracemalloc.stop()

    total_hours = (time.monotonic() - t_start) / 3600.0
    print(
        f"\n{'PASS' if exit_code == 0 else 'FAIL'}: "
        f"{frames_done:,} frames over {total_hours:.2f}h "
        f"(mode={args.mode}, N={args.num_boids}, seed={args.seed})"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
