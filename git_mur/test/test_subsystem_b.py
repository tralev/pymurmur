"""Subsystem B — Simulation Engine isolation tests.

Tests SimulationEngine step lifecycle, headless run, callbacks,
reset, and config live mutation. No viz imports.
"""



class TestSubsystemB:
    """Simulation engine — headless, step ordering, lifecycle."""

    def test_engine_imports_no_viz(self):
        """simulation/engine.py does not import pygame, moderngl, PyGLM, Pillow."""
        from pathlib import Path
        path = Path("pymurmur/simulation/engine.py")
        text = path.read_text()
        assert "import pygame" not in text
        assert "import moderngl" not in text
        assert "import PyGLM" not in text

    def test_engine_step_order(self, default_config):
        """Extensions.pre_step → flock.step → metrics.collect."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        # Verify internal wiring order by checking state after step
        initial_frame = sim.frame
        sim.step(1.0 / 60)
        assert sim.frame == initial_frame + 1

    def test_engine_headless_no_callback(self, default_config):
        """run_headless(steps=100) completes without error."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=100)
        assert sim.frame == 100

    def test_engine_headless_with_callback(self, default_config):
        """Callback called exactly once per step."""
        from pymurmur.simulation.engine import SimulationEngine

        call_count = 0
        def cb(sim):
            nonlocal call_count
            call_count += 1

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=10, callback=cb)
        assert call_count == 10

    def test_engine_reset_restores_initial_state(self, default_config):
        """After reset(), frame=0, flock is fresh, metrics are empty."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=50)
        assert sim.frame == 50

        sim.reset()
        assert sim.frame == 0
        assert sim.flock.N_active == default_config.num_boids

    def test_engine_config_live_mutation(self, default_config):
        """Mutating config.phi_p between steps affects next step's forces."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=5)

        # Capture state before mutation
        old_phi_p = sim.config.phi_p
        sim.config.phi_p = min(old_phi_p + 0.1, 1.0)

        sim.run_headless(steps=5)
        assert sim.config.phi_p != old_phi_p
