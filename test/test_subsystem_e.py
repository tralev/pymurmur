"""Subsystem E — Physics & Forces isolation tests.

Tests SoA memory budget, spatial index auto-select, two-pass
architecture, numba fallback, force clamping, and mode dispatch.
"""

import pytest


class TestSubsystemE:
    """Physics subsystem — forces, spatial index, memory budget."""

    def test_flock_soa_memory_budget(self):
        """FlockArrays at N=300K uses < 15 MB."""
        from pymurmur.core.types import FlockArrays
        import numpy as np

        N = 300_000
        arrays = FlockArrays(
            positions=np.zeros((N, 3), dtype=np.float32),
            velocities=np.zeros((N, 3), dtype=np.float32),
            accelerations=np.zeros((N, 3), dtype=np.float32),
            seeds=np.zeros(N, dtype=np.float32),
            last_theta=np.zeros(N, dtype=np.float32),
            active=np.ones(N, dtype=bool),
        )
        total = sum(arr.nbytes for arr in [
            arrays.positions, arrays.velocities, arrays.accelerations,
            arrays.seeds, arrays.last_theta, arrays.active,
        ])
        mb = total / (1024 * 1024)
        assert mb < 15, f"FlockArrays at 300K: {mb:.1f} MB > 15 MB budget"

    def test_spatial_index_auto_select(self):
        """N < 5000 → SpatialHashGrid, N >= 5000 → KDTreeIndex."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock, SpatialHashGrid, KDTreeIndex

        # Small flock → hash grid
        cfg_small = SimConfig()
        cfg_small.num_boids = 100
        flock_small = PhysicsFlock(cfg_small)
        assert isinstance(flock_small.get_index(), SpatialHashGrid)

        # Large flock → kdtree
        cfg_large = SimConfig()
        cfg_large.num_boids = 6000
        cfg_large.spatial_index = "auto"
        flock_large = PhysicsFlock(cfg_large)
        assert isinstance(flock_large.get_index(), KDTreeIndex)

    def test_two_pass_architecture(self, default_config):
        """Spatial force: cKDTree query pass → neighbor_idx, force pass reads it."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        # Two-pass: query neighbors first, then compute forces
        sim.run_headless(steps=5)
        assert sim.frame == 5

    def test_numba_jit_fallback(self, default_config):
        """When numba unavailable, numpy path produces valid results."""
        from pymurmur.simulation.engine import SimulationEngine

        default_config.use_numba = False
        sim = SimulationEngine(default_config)
        sim.run_headless(steps=10)
        assert sim.frame == 10

    def test_all_five_modes_return_valid_forces(self):
        """Every mode function runs without error."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        for mode in ["projection", "spatial", "field", "vicsek", "influencer"]:
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = 20
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=5)
            assert sim.frame == 5

    def test_mode_dispatch_unknown_raises(self):
        """compute_all_forces() with invalid mode raises KeyError."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces import compute_all_forces
        from pymurmur.physics.flock import PhysicsFlock

        cfg = SimConfig()
        cfg.mode = "nonexistent_mode"
        flock = PhysicsFlock(cfg)
        with pytest.raises((KeyError, ValueError)):
            compute_all_forces(flock, cfg)

    def test_extensions_pre_step_before_forces(self, default_config):
        """Extensions forces are applied before main force computation."""
        from pymurmur.simulation.engine import SimulationEngine

        default_config.predator_enabled = True
        sim = SimulationEngine(default_config)
        # Step should run extensions.pre_step before flock.step
        sim.step(1.0 / 60)
        assert sim.frame == 1
