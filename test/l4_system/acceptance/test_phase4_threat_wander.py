"""Phase 4 acceptance tests — predator threat pass-through and wander
in-domain fuzz.

Split out of test_phase4.py (file-size split) — golden trajectories,
presets, architecture edges, physical metrics, and performance tests
stay in the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.extensions._base import StepContext
from pymurmur.physics.extensions.wander import Wander, bounded_unit_path
from pymurmur.simulation.engine import SimulationEngine

pytestmark = [pytest.mark.acceptance, pytest.mark.guard, pytest.mark.phase4]


# ══════════════════════════════════════════════════════════════════════
# 4. Threat pass-through — predator passes through flock and exits
# ══════════════════════════════════════════════════════════════════════

class TestThreatPassThrough:
    """Predator approaches flock centre, passes through, and exits.

    Uses a compact domain so the predator can traverse capture/clear
    distances within a reasonable number of frames.
    """

    # Compact domain: U = 0.4 * 80 = 32, capture_dist = 8.64 * 32 = 276
    THREAT_DOMAIN = 80.0

    def _make_threat_engine(self, n_birds: int = 30):
        """Build an engine with a small domain for rapid predator traversal."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = n_birds
        cfg.width = cfg.height = cfg.depth = self.THREAT_DOMAIN
        cfg.predator_enabled = True
        cfg.roosting_enabled = False
        cfg.noise_scale = 0.0
        cfg.v0 = 8.0  # faster predator
        return SimulationEngine(cfg)

    def test_predator_approaches_flock_center(self):
        """Predator in approach phase moves toward the flock centre."""
        engine = self._make_threat_engine()
        pred = getattr(engine.extensions, '_predator', None)
        assert pred is not None, "Predator extension not initialized"

        # Place predator far enough to be outside capture_dist (~276)
        pred._pos = np.array([400.0, 40.0, 40.0], dtype=np.float32)
        pred._phase = "approach"

        # Point the predator's heading toward the flock centre so it
        # starts closing distance immediately (no turn-around delay).
        active = engine.flock.active
        center0 = engine.flock.positions[active].mean(axis=0)
        to_flock = center0 - pred._pos
        pred._dir = to_flock / max(np.linalg.norm(to_flock), 1e-6)

        # Record initial distance to flock centre
        dist0 = float(np.linalg.norm(pred._pos - center0))

        # Run enough frames for predator to get measurably closer
        for _ in range(120):
            engine.step(1.0 / 60.0)

        center1 = engine.flock.positions[engine.flock.active].mean(axis=0)
        dist1 = float(np.linalg.norm(pred._pos - center1))

        assert dist1 < dist0 * 0.95, (
            f"Predator should move toward flock: dist {dist0:.0f} -> {dist1:.0f}"
        )
        assert "approach" in {pred._phase}, (
            f"Predator should still be in approach phase, got {pred._phase}"
        )

    def test_predator_enters_egress_near_flock(self):
        """Predator near flock centre enters egress phase."""
        engine = self._make_threat_engine()
        pred = getattr(engine.extensions, '_predator', None)
        assert pred is not None, "Predator extension not initialized"

        # Place predator at flock centre in approach phase
        engine.flock.update_center()
        center = engine.flock.center
        pred._pos = center.copy()
        pred._phase = "approach"

        # One step should trigger approach->egress transition
        engine.step(1.0 / 60.0)
        assert pred._phase == "egress", (
            f"Predator at flock centre should enter egress, got {pred._phase}"
        )

    def test_predator_produces_threat_force(self):
        """Birds near predator receive non-zero threat force within a few frames."""
        engine = self._make_threat_engine(n_birds=50)
        pred = getattr(engine.extensions, '_predator', None)
        assert pred is not None, "Predator extension not initialized"

        # Place predator near flock centre
        active = engine.flock.active
        center = engine.flock.positions[active].mean(axis=0)
        pred._pos = center.copy() + np.array([5.0, 0.0, 0.0], dtype=np.float32)
        pred._phase = "approach"

        # Run a few steps - nearby birds should receive threat forces.
        # Note: accelerations are zeroed at end of integrate(), so check
        # last_accelerations (stashed before zeroing).
        has_force = False
        for _ in range(10):
            engine.step(1.0 / 60.0)
            acc_mags = np.linalg.norm(
                engine.flock.last_accelerations[engine.flock.active], axis=1
            )
            if acc_mags.max() > 0.01:
                has_force = True
                break

        assert has_force, (
            "Predator near flock should produce non-zero forces within 10 frames"
        )

    def test_flash_expansion_visible_within_30_frames(self):
        """P4 acceptance: flock scatters measurably when predator appears.

        Uses a very tight domain (40³) so birds start close together — the
        1/d² escape force is strong across most of the flock and the P10
        "hollow core" signal is unambiguous within 30 frames.

        Measures expansion via two complementary metrics:
          1. P10 distance — inner birds near predator scatter outward.
          2. P50 (median) distance — bulk expansion of the flock centre.
        """
        # Tight domain: 40³, 80 birds → compact cluster, strong 1/d² forces
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 80
        cfg.width = cfg.height = cfg.depth = 40.0
        cfg.predator_enabled = True
        cfg.roosting_enabled = False
        cfg.wander_enabled = False
        cfg.ripple_enabled = False
        cfg.noise_scale = 0.0
        cfg.v0 = 8.0
        cfg.seed = 42
        cfg.max_force = 10.0
        cfg.metrics_detail_level = 0

        engine = SimulationEngine(cfg)

        # Park predator far away during settling
        pred = getattr(engine.extensions, '_predator', None)
        assert pred is not None, "Predator extension must be initialised"
        pred._phase = "egress"
        pred._pos = np.array([-200.0, -200.0, -200.0], dtype=np.float32)

        # Settle the flock into a compact cluster
        for _ in range(40):
            engine.step(1.0 / 60.0)

        # Measure baseline
        active = engine.flock.active
        center = engine.flock.positions[active].mean(axis=0)
        dists_before = np.linalg.norm(
            engine.flock.positions[active] - center, axis=1
        )
        p10_before = float(np.percentile(dists_before, 10))
        p50_before = float(np.percentile(dists_before, 50))

        # Inject predator at flock centre
        pred._pos = center.copy()
        pred._phase = "approach"
        to_flock = center - pred._pos
        dist = np.linalg.norm(to_flock)
        if dist > 1e-6:
            pred._dir = to_flock / dist

        # Run 30 frames — flock should scatter away from predator
        for _ in range(30):
            engine.step(1.0 / 60.0)

        # Measure expansion
        center_after = engine.flock.positions[active].mean(axis=0)
        dists_after = np.linalg.norm(
            engine.flock.positions[active] - center_after, axis=1
        )
        p10_after = float(np.percentile(dists_after, 10))
        p50_after = float(np.percentile(dists_after, 50))

        # 1. Hollow-core: inner birds pushed out by strong 1/d² escape force
        assert p10_after > p10_before * 1.3, (
            f"Hollow core not visible: p10 {p10_before:.1f} → {p10_after:.1f} "
            f"(need > {p10_before * 1.3:.1f})"
        )

        # 2. Bulk expansion: median distance shifts outward
        assert p50_after > p50_before, (
            f"No bulk expansion: p50 {p50_before:.1f} → {p50_after:.1f}"
        )


# ══════════════════════════════════════════════════════════════════════
# 5. Wander path stays in-domain over 10⁴ frames
# ══════════════════════════════════════════════════════════════════════

class TestWanderInDomain10k:
    """Wander centre stays within domain bounds over 10⁴ simulation steps."""

    def test_wander_center_10k_frames(self):
        """10⁴ wander steps — centre stays within expanded domain bounds."""
        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "spatial"
        cfg.width, cfg.height, cfg.depth = 1000.0, 700.0, 400.0

        from pymurmur.physics.flock import PhysicsFlock
        flock = PhysicsFlock(cfg)

        ctx = StepContext(frame=0, dt=1.0 / 60.0,
                          rng=flock.rng, center=flock.center, config=cfg)

        w = Wander()
        for _ in range(10_000):
            w.apply(flock, ctx)
            ctx.frame += 1
            wc = flock.wander_center
            radius = getattr(cfg, 'wander_attractor_radius', 300.0)
            assert -radius <= wc[0] <= cfg.width + radius, (
                f"wander x={wc[0]:.1f} out of domain"
            )
            assert -radius <= wc[1] <= cfg.height + radius, (
                f"wander y={wc[1]:.1f} out of domain"
            )
            assert -radius <= wc[2] <= cfg.depth + radius, (
                f"wander z={wc[2]:.1f} out of domain"
            )

    def test_path_bound_1e6_fuzzed(self):
        """10⁶ random t ∈ [0, 10000] → all ‖path(t)‖ ≤ 1 (bounded_unit_path)."""
        rng = np.random.default_rng(42)
        t_values = rng.uniform(0, 10_000, size=1_000_000).astype(np.float32)
        paths = bounded_unit_path(t_values)
        norms = np.linalg.norm(paths, axis=1)
        max_norm = float(norms.max())
        assert max_norm <= 1.0 + 1e-6, f"max ‖path(t)‖ = {max_norm:.6f} > 1"
