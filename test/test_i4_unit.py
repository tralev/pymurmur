"""I4 Phase — Missing unit tests for independent entities.

Covers:
  I4.1 — headless_frame/frame pure-render contract (CPU tests): M1, M2
  I4.2 — flock↔forces cycle break: M3, M4
  I4.4 — Architecture waiver removal: M11, M12
"""

from unittest.mock import MagicMock, patch

import pytest

from pymurmur.simulation.engine import SimulationEngine


# ── I4.1: Hoist step() out of Visualizer ──────────────────────────

class TestVisualizerRenderContractCPU:
    """Verify headless_frame() and frame() never step the simulation (CPU-only)."""

    @staticmethod
    def _make_mock_renderer():
        """Create a mock Renderer3D that fakes begin/end/capture without GPU."""
        mock = MagicMock()
        mock.capture_frame.return_value = MagicMock()  # fake PIL Image
        return mock

    def test_headless_frame_does_not_call_sim_step(self, default_config, monkeypatch):
        """I4.1 M1: headless_frame() is pure render — never calls sim.step()."""
        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)

        # Spy on sim.step to detect any calls from headless_frame
        sim.step = MagicMock(wraps=sim.step)

        mock_renderer = self._make_mock_renderer()
        monkeypatch.setattr(
            "pymurmur.viz.visualizer.Renderer3D",
            lambda **kw: mock_renderer,
        )
        monkeypatch.setattr(
            "pymurmur.viz.visualizer.OrbitCamera",
            lambda **kw: MagicMock(),
        )

        from pymurmur.viz.visualizer import Visualizer

        viz = Visualizer(sim, cfg, headless=True)
        img = viz.headless_frame()

        # Contract: rendering must NOT advance simulation
        sim.step.assert_not_called()

        # Renderer was used correctly
        mock_renderer.begin_frame.assert_called_once()
        mock_renderer.draw_birds.assert_called_once_with(sim.flock)
        mock_renderer.end_frame.assert_called_once()
        mock_renderer.capture_frame.assert_called_once()

    def test_frame_does_not_call_sim_step(self, default_config, monkeypatch):
        """I4.1 M2: frame() is pure render — never calls sim.step()."""
        cfg = default_config
        cfg.num_boids = 10
        sim = SimulationEngine(cfg)

        sim.step = MagicMock(wraps=sim.step)

        mock_renderer = self._make_mock_renderer()
        monkeypatch.setattr(
            "pymurmur.viz.visualizer.Renderer3D",
            lambda **kw: mock_renderer,
        )
        monkeypatch.setattr(
            "pymurmur.viz.visualizer.OrbitCamera",
            lambda **kw: MagicMock(),
        )

        from pymurmur.viz.visualizer import Visualizer

        viz = Visualizer(sim, cfg, headless=False)
        viz.frame()

        # Contract: rendering must NOT advance simulation
        sim.step.assert_not_called()

        # Renderer was used correctly (frame() doesn't call capture_frame)
        mock_renderer.begin_frame.assert_called_once()
        mock_renderer.draw_birds.assert_called_once_with(sim.flock)
        mock_renderer.end_frame.assert_called_once()


# ── I4.2: Break flock↔forces cycle ────────────────────────────────

class TestFlockForcesCycleBreak:
    """Verify the flock↔forces cycle is broken at the import level."""

    def test_engine_is_only_module_importing_both_flock_and_forces(self):
        """I4.2 M3: Only simulation.engine may import both flock AND forces.

        A second dual-importer would become a de facto orchestrator.
        Forces sub-modules that import forces._base + flock are internal
        to the forces package and are excluded from this check.
        """
        from test.test_architecture import ALLOWED_EDGES

        dual_importers = []
        for mod, targets in ALLOWED_EDGES.items():
            # Exclude forces-package-internal modules
            if mod.startswith("pymurmur.physics.forces"):
                continue
            has_flock = any(
                t == "pymurmur.physics.flock"
                or t.startswith("pymurmur.physics.flock.")
                for t in targets
            )
            has_forces = any(
                t == "pymurmur.physics.forces"
                or t.startswith("pymurmur.physics.forces.")
                for t in targets
            )
            if has_flock and has_forces:
                dual_importers.append(mod)

        assert dual_importers == ["pymurmur.simulation.engine"], (
            f"Only simulation.engine may import both flock and forces. "
            f"Found {len(dual_importers)} dual-importers: {dual_importers}"
        )

    def test_step_pipeline_all_6_stages_in_order(self, default_config):
        """I4.2 M4: step() runs all 6 stages in exact order.

        Sequence: drain → extensions → index → forces → integrate → metrics.
        Existing test_engine_step_order only spies 3 of 6 stages.
        """
        cfg = default_config
        cfg.num_boids = 10
        cfg.mode = "spatial"  # spatial mode uses index rebuild
        engine = SimulationEngine(cfg)

        order_log = []

        # Save originals
        orig_drain = engine.drain_commands
        orig_ext_pre = engine.extensions.pre_step
        orig_integrate = engine.flock.integrate
        orig_collect = engine.metrics.collect
        if engine.flock._index is not None:
            orig_rebuild = engine.flock._index.rebuild

        def spy_drain():
            order_log.append("drain")
            return orig_drain()

        def spy_ext(flock, ctx):
            order_log.append("extensions")
            return orig_ext_pre(flock, ctx)

        def spy_rebuild(positions, active):
            order_log.append("index")
            return orig_rebuild(positions, active)

        def spy_integrate(config, dt):
            order_log.append("integrate")
            return orig_integrate(config, dt)

        def spy_collect(flock, frame):
            order_log.append("metrics")
            return orig_collect(flock, frame)

        engine.drain_commands = spy_drain
        engine.extensions.pre_step = spy_ext
        engine.flock.integrate = spy_integrate
        engine.metrics.collect = spy_collect

        if engine.flock._index is not None:
            engine.flock._index.rebuild = spy_rebuild

        # Patch compute_all_forces in the engine module
        with patch(
            "pymurmur.simulation.engine.compute_all_forces"
        ) as mock_forces:
            mock_forces.side_effect = lambda flock, cfg: order_log.append("forces")
            engine.step()

        # All 6 stages must appear
        assert "drain" in order_log
        assert "extensions" in order_log
        assert "index" in order_log
        assert "forces" in order_log
        assert "integrate" in order_log
        assert "metrics" in order_log

        # Verify exact ordering
        drain_idx = order_log.index("drain")
        ext_idx = order_log.index("extensions")
        index_idx = order_log.index("index")
        forces_idx = order_log.index("forces")
        integrate_idx = order_log.index("integrate")
        metrics_idx = order_log.index("metrics")

        assert drain_idx < ext_idx < index_idx < forces_idx < integrate_idx < metrics_idx, (
            f"Pipeline order wrong: {order_log}"
        )


# ── I4.4: Architecture waiver removal ─────────────────────────────

class TestArchitectureWaiversRemoved:
    """Verify that flock→forces is NOT in known violations or phase removals."""

    def test_flock_to_forces_not_in_known_violations(self):
        """I4.4 M11: flock→forces must not be in KNOWN_VIOLATIONS."""
        from test.test_architecture import KNOWN_VIOLATIONS

        for src, tgt, phase in KNOWN_VIOLATIONS:
            is_flock = src == "pymurmur.physics.flock" or src.startswith(
                "pymurmur.physics.flock."
            )
            is_forces = tgt == "pymurmur.physics.forces" or tgt.startswith(
                "pymurmur.physics.forces."
            )
            assert not (is_flock and is_forces), (
                f"flock→forces is in KNOWN_VIOLATIONS as ({src}, {tgt}, {phase}). "
                f"It was resolved in I4.2 — remove this entry."
            )

    def test_flock_to_forces_not_in_phase_violation_removals(self):
        """I4.4 M12: flock→forces must not be in PHASE_VIOLATION_REMOVALS."""
        from test.test_architecture import PHASE_VIOLATION_REMOVALS

        for ph_key, removals in PHASE_VIOLATION_REMOVALS.items():
            for src, tgt in removals:
                is_flock = src == "pymurmur.physics.flock" or src.startswith(
                    "pymurmur.physics.flock."
                )
                is_forces = tgt == "pymurmur.physics.forces" or tgt.startswith(
                    "pymurmur.physics.forces."
                )
                assert not (is_flock and is_forces), (
                    f"flock→forces is in PHASE_VIOLATION_REMOVALS[{ph_key}] "
                    f"as ({src}, {tgt}). It was resolved in I4.2 — remove this entry."
                )
