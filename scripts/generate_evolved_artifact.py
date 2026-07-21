#!/usr/bin/env python3
"""P0.16 CI prerequisite — generate output/evolved.yaml for guard-rail-evolved.

`output/` is gitignored, so a fresh checkout (any CI runner, or a fresh
Docker image build) never has `output/evolved.yaml` — but
`test/l3_modules/analysis/test_evolved_yaml.py` hard-asserts it exists.
Without this step, `guard-rail-evolved` (part of the merge-blocking
`guard-rails-summary`) fails on every fresh checkout, not just a
missing-artifact edge case.

Runs a small, fast EvoFlock GA session (~2-3s) purely to produce a
schema-valid artifact for the guard to validate — not a real
optimization run. For an actual evolved-parameter search, use a full
`EvoConfig` (see `scripts/run_evoflock_small.py` for a larger example).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymurmur.analysis.evoflock import EvoConfig, EvoFlock  # noqa: E402
from pymurmur.core.config import SimConfig  # noqa: E402


def main() -> None:
    cfg = SimConfig()
    cfg.num_boids = 20
    cfg.seed = 42

    ga_cfg = EvoConfig(
        population_size=10,
        max_steps=20,
        n_islands=1,
        eval_steps=20,
    )

    evo = EvoFlock(cfg, ga_cfg)
    evo.run(n_runs=1, save_path="output/evolved.yaml")
    print("Generated output/evolved.yaml")


if __name__ == "__main__":
    main()
