"""P1 — Per-mode step-time budget table.

Parametrised over :data:`MODE_REGISTRY` so a newly-registered mode without
a budget entry fails collection.  The ``@slow`` test benchmarks each mode
at N=2,000 and asserts mean step time ≤ budget × headroom (×3).

Non-``@slow`` smoke test: verifies every registered mode has a budget entry.
"""

from __future__ import annotations

import time

import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.forces import MODE_REGISTRY
from pymurmur.simulation.engine import SimulationEngine

# ── Budget registry ─────────────────────────────────────────────
#
# (mode_name, budget_ms_at_2000)
# Budget is the maximum acceptable mean step time at N=2,000 in milliseconds.
# New modes MUST add an entry here or test_budget_registry_covers_all_modes
# will fail at collection time (P1 contract).

# Budgets calibrated 2026-07-20 on Apple Silicon M-series.
# Each budget is the EXPECTED mean ms/step at N=2,000 (target, not CI limit).
# The assertion uses budget × HEADROOM so CI variance (×3) is absorbed.
#
# Vicsek and angle are inherently O(N²) at this N — the budgets
# document that reality; field and influencer are O(N) and fast.
STEP_BUDGET_2000: dict[str, float] = {
    "projection": 130.0,
    "spatial":      20.0,
    "field":         8.0,
    "vicsek":     6500.0,
    "influencer":    8.0,
    "angle":       200.0,
    "marl":        150.0,
}

HEADROOM = 3.0   # budget × headroom is the assertion threshold
STEPS = 50        # benchmarking steps per mode
N_BOIDS = 2_000   # P1 target N


def _benchmark_mode(mode: str) -> float:
    """Return mean step time in milliseconds for *mode* at N=2,000."""
    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = N_BOIDS
    cfg.seed = 7
    sim = SimulationEngine(cfg)
    # Warm-up: 2 steps
    sim.run_headless(steps=2)
    # Timed run
    t0 = time.perf_counter()
    sim.run_headless(steps=STEPS)
    elapsed = (time.perf_counter() - t0) / STEPS * 1000.0  # ms per step
    return elapsed


# ── Smoke tests (fast) ───────────────────────────────────────────


def test_budget_registry_covers_all_modes():
    """P1: Every registered mode has a budget entry (fails collection
    if a new mode is added without a budget)."""
    for mode in sorted(MODE_REGISTRY):
        assert mode in STEP_BUDGET_2000, (
            f"Mode '{mode}' registered in MODE_REGISTRY but missing from "
            f"STEP_BUDGET_2000.  Add a budget entry in "
            f"test/l4_crosscutting/perf/test_budgets.py."
        )


def test_budget_registry_no_stale_entries():
    """P1: Every budget entry maps to a registered mode.

    Reverse check — prevents stale budget entries from accumulating
    when a mode is removed from MODE_REGISTRY.
    """
    for mode in STEP_BUDGET_2000:
        assert mode in MODE_REGISTRY, (
            f"Budget entry for '{mode}' has no matching mode in "
            f"MODE_REGISTRY.  Remove the stale entry from "
            f"STEP_BUDGET_2000."
        )


def test_budget_o1_cheaper_than_on2():
    """P1: O(N) modes have lower budgets than O(N²) modes at N=2,000.

    Design-contract sanity: field, influencer, and marl are O(N)
    and should be cheaper than vicsek (O(N²) dense pairwise) at
    this scale.  If this test fails, either a budget was wildly
    mis-calibrated or the algorithm regressed.
    """
    on2_max = max(STEP_BUDGET_2000[m] for m in ("vicsek",))
    # The three O(N) modes should all be cheaper than vicsek's O(N²) budget
    on1_modes = ["field", "influencer", "marl"]
    for m in on1_modes:
        assert STEP_BUDGET_2000[m] < on2_max, (
            f"{m} (O(N)) budget {STEP_BUDGET_2000[m]} ms >= "
            f"vicsek (O(N²)) budget {on2_max} ms — design contract violated"
        )


# ── Benchmark (slow) ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("mode", sorted(MODE_REGISTRY))
def test_mode_step_time_within_budget(mode: str):
    """P1 (@slow): Mean step time at N=2,000 ≤ budget × headroom."""
    budget = STEP_BUDGET_2000[mode]
    elapsed = _benchmark_mode(mode)
    threshold = budget * HEADROOM
    assert elapsed <= threshold, (
        f"{mode}: {elapsed:.1f} ms/step exceeds budget {budget:.0f} ms "
        f"× headroom {HEADROOM:.0f} = {threshold:.0f} ms"
    )


# ── Budget table sanity ──────────────────────────────────────────


def test_budget_table_headroom_check():
    """P1: Budgets are positive and headroom is valid."""
    for mode, ms in STEP_BUDGET_2000.items():
        assert ms > 0, f"{mode}: budget must be positive, got {ms}"
        assert ms * HEADROOM > 0, f"{mode}: headroom product must be positive"
