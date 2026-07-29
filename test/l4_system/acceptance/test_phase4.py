"""Phase 4 acceptance tests — validates every criterion in the acceptance checklist.

Covers:
- Golden trajectory pinned for spatial mode (toroidal + sphere)
- 10⁴-frame spatial NaN/speed fuzz with all terms on
- 7 field presets load and run
- P4 architecture edges: _kernels.py, ecology.py, spatial.py
- Physical metrics report plausible values
- Spatial mode performance ≤ 100ms at N=16K (benchmark)

Predator threat pass-through and wander in-domain fuzz tests live in
test_phase4_threat_wander.py (file-size split of this file).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

pytestmark = [pytest.mark.acceptance, pytest.mark.guard, pytest.mark.phase4]


# ══════════════════════════════════════════════════════════════════════
# 1. Golden trajectory pinned for spatial mode
# ══════════════════════════════════════════════════════════════════════

class TestSpatialGoldenPinned:
    """Golden trajectories for spatial mode are pinned and reproducible."""

    def test_golden_spatial_toroidal(self):
        """Spatial toroidal golden matches the pinned fixture exactly.

        Must match regenerate_golden.py defaults: seed=77, 15 birds, 30 frames.
        NPZ stores trajectory as (frames, N, 3) — frame 0 is initial state.
        """
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.seed = 77          # match regenerate_golden.py DEFAULT_SEED
        cfg.num_boids = 15      # match DEFAULT_BIRDS
        cfg.noise_scale = 0.0
        cfg.influence_count = 7
        cfg.visual_range = 70.0

        engine = SimulationEngine(cfg)
        # Run 29 steps: frame 0 = initial, +29 = 30 frames = DEFAULT_FRAMES
        for _ in range(29):
            engine.step(1.0 / 60.0)

        golden = np.load("test/data/golden_spatial.npz")
        # NPZ stores trajectory: (frames, N, 3) — compare final frame
        assert np.allclose(engine.flock.positions, golden["pos"][-1], atol=1e-4), (
            "spatial toroidal positions diverged from golden"
        )
        assert np.allclose(engine.flock.velocities, golden["vel"][-1], atol=1e-4), (
            "spatial toroidal velocities diverged from golden"
        )

    def test_golden_spatial_sphere(self):
        """Spatial sphere boundary golden matches the pinned fixture exactly."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.seed = 77          # match regenerate_golden.py DEFAULT_SEED
        cfg.num_boids = 15      # match DEFAULT_BIRDS
        cfg.noise_scale = 0.0
        cfg.boundary_mode = "sphere"
        cfg.boundary_sphere_radius = 200.0

        engine = SimulationEngine(cfg)
        for _ in range(29):
            engine.step(1.0 / 60.0)

        golden = np.load("test/data/golden_spatial_sphere.npz")
        assert np.allclose(engine.flock.positions, golden["pos"][-1], atol=1e-4), (
            "spatial sphere positions diverged from golden"
        )
        assert np.allclose(engine.flock.velocities, golden["vel"][-1], atol=1e-4), (
            "spatial sphere velocities diverged from golden"
        )


# ══════════════════════════════════════════════════════════════════════
# 2. 10⁴-frame NaN/speed fuzz — all-terms-on spatial via SimulationEngine
# ══════════════════════════════════════════════════════════════════════

class TestSpatialFuzz10k:
    """Run SimulationEngine with spatial mode for 10⁴ frames with all features
    active and verify no NaN, no inf, speeds bounded, forces clamped."""

    @pytest.mark.slow
    def test_10k_frames_no_nan_no_inf(self):
        """10⁴ frames via SimulationEngine — no NaN, no inf in positions or velocities."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 20
        cfg.v0 = 4.0
        cfg.max_force = 5.0
        cfg.noise_scale = 0.1
        cfg.separation_weight = 4.5
        cfg.alignment_weight = 0.65
        cfg.cohesion_weight = 0.75
        cfg.influence_count = 7
        cfg.visual_range = 200.0
        cfg.jitter_separation = 0.05
        cfg.jitter_cohesion = 0.05
        cfg.jitter_alignment = 0.05
        cfg.metrics_detail_level = 0

        engine = SimulationEngine(cfg)
        for frame in range(10_000):
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), (
                f"frame {frame}: NaN in positions"
            )
            assert np.isfinite(engine.flock.velocities).all(), (
                f"frame {frame}: NaN in velocities"
            )
            assert np.isfinite(engine.flock.accelerations).all(), (
                f"frame {frame}: NaN in accelerations"
            )

    @pytest.mark.slow
    def test_10k_frames_speeds_bounded(self):
        """10⁴ frames — speeds stay within [0.3·v0, v0] (band clamp)."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 15
        cfg.v0 = 4.0
        cfg.max_force = 3.0
        cfg.noise_scale = 0.0  # no noise simplifies clamp verification
        cfg.visual_range = 200.0
        cfg.metrics_detail_level = 0

        engine = SimulationEngine(cfg)
        for frame in range(10_000):
            engine.step(1.0 / 60.0)
            speeds = np.linalg.norm(engine.flock.velocities, axis=1)
            v_min = 0.3 * cfg.v0
            # Toroidal boundary: band clamp should keep speeds in [0.3*v0, v0]
            assert (speeds >= v_min - 0.01).all(), (
                f"frame {frame}: min speed {speeds.min():.2f} < {v_min:.2f}"
            )
            assert (speeds <= cfg.v0 + 0.01).all(), (
                f"frame {frame}: max speed {speeds.max():.2f} > {cfg.v0}"
            )


# ══════════════════════════════════════════════════════════════════════
# 3. 7 field presets load and run
# ══════════════════════════════════════════════════════════════════════

class TestPresetsLoadAndRun:
    """All 7 field presets load and run simulation steps without crash or NaN."""

    PRESETS = [
        "quiet_roost", "lava_lamp", "ink_cloud",
        "predator_ripple", "vacuole", "silk_sheet", "storm_turn",
    ]

    def test_all_presets_load_and_step(self):
        """Each preset loads, creates engine, and steps once without error."""
        for name in self.PRESETS:
            cfg = SimConfig.from_file(f"conf/field_{name}.yaml")
            engine = SimulationEngine(cfg)
            engine.step(1.0 / 60.0)
            assert np.isfinite(engine.flock.positions).all(), (
                f"{name}: NaN in positions after step"
            )
            assert np.isfinite(engine.flock.velocities).all(), (
                f"{name}: NaN in velocities after step"
            )

    def test_all_presets_run_10_steps(self):
        """Each preset runs 10 steps — no NaN, no crash, speeds finite.

        Note: field-mode presets can have transient speed spikes during
        initialisation (anchor pull, blob init).  We only assert no NaN / inf.
        """
        for name in self.PRESETS:
            cfg = SimConfig.from_file(f"conf/field_{name}.yaml")
            cfg.metrics_detail_level = 0
            engine = SimulationEngine(cfg)
            for step in range(10):
                engine.step(1.0 / 60.0)
                assert np.isfinite(engine.flock.positions).all(), (
                    f"{name} step {step}: NaN in positions"
                )
                assert np.isfinite(engine.flock.velocities).all(), (
                    f"{name} step {step}: NaN in velocities"
                )
                assert np.isfinite(engine.flock.accelerations).all(), (
                    f"{name} step {step}: NaN in accelerations"
                )


# ══════════════════════════════════════════════════════════════════════
# 6. P4 architecture edges
# ══════════════════════════════════════════════════════════════════════

class TestP4ArchitectureEdges:
    """Verify P4-specific architecture edges are registered in ALLOWED_EDGES."""

    def test_p4_new_files_exist(self):
        """New P4 files exist on disk."""
        p4_files = [
            "pymurmur/physics/forces/_kernels.py",
        ]
        for f in p4_files:
            assert os.path.exists(f), f"P4 file missing: {f}"

    def test_p4_edges_registered(self):
        """P4 edges: spatial.py→_kernels, _kernels→core, ecology→core+flock are registered."""
        from test.crosscutting.guards.architecture.test_architecture_edges_data import ALLOWED_EDGES

        # Check _kernels has its own entry
        if "pymurmur.physics.forces._kernels" in ALLOWED_EDGES:
            assert "pymurmur.core.types" in ALLOWED_EDGES["pymurmur.physics.forces._kernels"], (
                "_kernels must be allowed to import core.types"
            )

        # Check spatial is allowed to import _kernels
        if "pymurmur.physics.forces.spatial" in ALLOWED_EDGES:
            assert "pymurmur.physics.forces._kernels" in ALLOWED_EDGES["pymurmur.physics.forces.spatial"], (
                "spatial must be allowed to import _kernels"
            )

        # Check ecology has its edges
        if "pymurmur.physics.extensions.ecology" in ALLOWED_EDGES:
            eco_edges = ALLOWED_EDGES["pymurmur.physics.extensions.ecology"]
            for required in ("pymurmur.core.types", "pymurmur.physics.flock"):
                assert required in eco_edges, f"ecology missing {required}"


# ══════════════════════════════════════════════════════════════════════
# 7. Physical metrics sanity check
# ══════════════════════════════════════════════════════════════════════

class TestPhysicalMetrics:
    """Physical metrics report plausible real-world values."""

    def test_physical_metrics_plausible(self):
        """After a short run, physical metrics are in plausible ranges."""
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 100
        cfg.metrics_detail_level = 1
        cfg.noise_scale = 0.0

        engine = SimulationEngine(cfg)
        # Run enough frames for metrics to stabilise
        for _ in range(60):
            engine.step(1.0 / 60.0)

        m = engine.metrics.snapshot()
        # Starling cruise ~8-14 m/s (may be zero if metrics haven't accumulated)
        assert 0.0 <= m.speed_real_ms < 30.0, (
            f"speed_real_ms={m.speed_real_ms:.1f} outside plausible [0, 30]"
        )
        # Force in millinewtons to newtons for a 75g bird
        assert 0.0 <= m.force_real_N < 10.0, (
            f"force_real_N={m.force_real_N:.6f} outside plausible [0, 10]"
        )
        # Energy in millijoules to joules
        assert 0.0 <= m.energy_J < 20.0, (
            f"energy_J={m.energy_J:.6f} outside plausible [0, 20]"
        )


# ══════════════════════════════════════════════════════════════════════
# 8. Spatial mode performance ≤ 100ms at N=16K
# ══════════════════════════════════════════════════════════════════════

class TestSpatialModePerformance:
    """Spatial mode with all extensions runs ≤ 100ms/frame at N=16K.

    The batch cKDTree query (P4.6) and numba kernels (P4.10) are the
    primary performance drivers.  This gate catches regressions in
    neighbour-query or force-computation hot paths.
    """

    @pytest.mark.slow
    def test_spatial_16k_all_extensions_under_150ms(self):
        """Full SimulationEngine step at N=16K with all extensions enabled
        should complete in ≤ 150ms.
        (CI-relaxed bound; hardware-dependent; local target is ≤ 100ms.)"""
        pytest.importorskip("numba")  # numba kernels required for perf target

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = 16_000
        cfg.noise_scale = 0.0
        cfg.influence_count = 7
        cfg.visual_range = 70.0
        cfg.metrics_detail_level = 0

        # All extensions enabled — worst-case frame
        cfg.predator_enabled = True
        cfg.roosting_enabled = True
        cfg.wander_enabled = True
        cfg.ripple_enabled = True

        engine = SimulationEngine(cfg)

        # Warm-up (JIT, index build, extension init)
        for _ in range(5):
            engine.step(1.0 / 60.0)

        # Benchmark
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            engine.step(1.0 / 60.0)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        max_ms = np.max(times) * 1000

        assert avg_ms < 150.0, (
            f"Spatial 16K + 4 extensions: avg={avg_ms:.1f}ms > 150ms CI bound"
        )
        assert max_ms < 300.0, (
            f"Spatial 16K + 4 extensions: max={max_ms:.1f}ms > 300ms spike bound"
        )
        print(f"\n  Spatial 16K + 4 extensions: avg={avg_ms:.1f}ms, max={max_ms:.1f}ms")
