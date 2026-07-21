"""Mode-switch guard-rail test — cycle all 7 force modes mid-simulation.

P14 guard: no mode transition shall crash, leak state, or stall the frame
counter. Each mode runs 10 steps; the full cycle covers 7×10 = 70 steps.

Marked with @pytest.mark.guard so `pytest -m guard` selects it.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.guard, pytest.mark.e2e]

ALL_MODES = ["projection", "spatial", "field", "vicsek", "influencer", "angle", "marl"]

# All modes are now registered — no future modes pending
FUTURE_MODES: list[str] = []


class TestModeSwitchNoCrash:
    """Cycle through every registered mode in a single engine run."""

    def test_all_seven_modes_cycle_no_crash(self, default_config):
        """Step 10 frames in each of 7 modes, never resetting the engine.

        marl is marked xfail — it ships in P12.
        """
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 20
        cfg.seed = 77
        sim = SimulationEngine(cfg)
        assert sim.frame == 0

        all_modes = ALL_MODES + FUTURE_MODES
        for mode in all_modes:
            sim.config.mode = mode
            if mode in FUTURE_MODES:
                # angle/marl not yet registered — clean error expected
                with pytest.raises(ValueError, match=f"Unknown force mode.*{mode}"):
                    sim.run_headless(steps=1)
                continue
            step_before = sim.frame
            sim.run_headless(steps=10)
            # Core assertion: frame counter advanced by exactly 10
            assert sim.frame == step_before + 10, (
                f"Mode '{mode}' stalled: frame {step_before} → {sim.frame}"
            )
            # No NaN in positions after this mode's run
            assert np.isfinite(sim.flock.positions).all(), (
                f"Mode '{mode}' produced NaN positions"
            )

        # 7 active modes × 10 steps = 70 total
        assert sim.frame == 70

    def test_mode_cycle_roundtrip(self, default_config):
        """Switch to spatial and back to projection — positions diverge from
        a same-seed pure-projection control, proving the switch ran."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 20
        cfg.seed = 77

        # Run: 5 proj → 10 spatial → 5 proj
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=5)
        sim.config.mode = "spatial"
        sim.run_headless(steps=10)
        assert sim.frame == 15
        assert np.isfinite(sim.flock.positions).all()

        sim.config.mode = "projection"
        sim.run_headless(steps=5)
        assert sim.frame == 20
        pos_switched = sim.flock.positions.copy()

        # Control: 20 proj-only with same seed
        cfg2 = default_config
        cfg2.num_boids = 20
        cfg2.seed = 77
        sim2 = SimulationEngine(cfg2)
        sim2.run_headless(steps=20)
        pos_control = sim2.flock.positions.copy()

        # Positions MUST differ — spatial mode runs different dynamics
        assert not np.allclose(pos_switched, pos_control, atol=1e-3), (
            "Mode switch had no effect — positions match pure-projection control"
        )

    def test_rapid_mode_toggling_no_crash(self, default_config):
        """Toggle mode every single frame for 28 frames — no crash."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 20
        cfg.seed = 77
        sim = SimulationEngine(cfg)

        # 28 rapid toggles (cycling through 5 active modes)
        for i in range(28):
            sim.config.mode = ALL_MODES[i % len(ALL_MODES)]
            sim.step(1.0 / 60.0)
            assert np.isfinite(sim.flock.positions).all(), (
                f"NaN at frame {sim.frame} after switching to '{ALL_MODES[i % len(ALL_MODES)]}'"
            )

        assert sim.frame == 28

    def test_mode_switch_preserves_active_mask(self, default_config):
        """Active mask survives mode transitions — no birds lost."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 30
        cfg.seed = 77
        sim = SimulationEngine(cfg)
        n_total = sim.flock.N_active

        for mode in ALL_MODES:
            sim.config.mode = mode
            sim.run_headless(steps=5)
            assert sim.flock.N_active == n_total, (
                f"Mode '{mode}' changed active count: {sim.flock.N_active} ≠ {n_total}"
            )

    def test_mode_switch_after_add_remove(self, default_config):
        """Mode switch after add/remove mid-run doesn't crash."""
        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.num_boids = 30
        cfg.seed = 77
        sim = SimulationEngine(cfg)

        # Run, add birds, switch mode, remove birds
        sim.run_headless(steps=5)
        sim.flock.add_boids(10, cfg)
        sim.config.mode = "vicsek"
        sim.run_headless(steps=5)
        assert np.isfinite(sim.flock.positions).all()

        sim.flock.remove_boids(5)
        sim.config.mode = "field"
        sim.run_headless(steps=5)
        assert np.isfinite(sim.flock.positions).all()
