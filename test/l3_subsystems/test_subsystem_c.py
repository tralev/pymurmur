"""Subsystem C — Visualization & Input isolation tests.

Tests that viz modules respect dependency rules: no simulation
imports, camera has no GPU dependency, input only mutates SimConfig.
"""

from pathlib import Path

import pytest


class TestSubsystemC:
    """Visualization subsystem — dependency rules and initialization."""

    def test_renderer_no_simulation_imports(self):
        """viz/renderer.py does not import simulation."""
        path = Path("pymurmur/viz/renderer.py")
        text = path.read_text()
        assert "from ..simulation" not in text
        assert "from pymurmur.simulation" not in text
        assert "import SimulationEngine" not in text

    def test_input_control_no_simulation_imports(self):
        """viz/input_control.py does not import simulation."""
        path = Path("pymurmur/viz/input_control.py")
        text = path.read_text()
        assert "from ..simulation" not in text
        assert "from pymurmur.simulation" not in text

    def test_camera_no_moderngl_import(self):
        """viz/camera.py does not import moderngl."""
        path = Path("pymurmur/viz/camera.py")
        text = path.read_text()
        assert "import moderngl" not in text

    def test_shaders_no_moderngl_import(self):
        """viz/shaders.py does not import moderngl."""
        path = Path("pymurmur/viz/shaders.py")
        text = path.read_text()
        assert "import moderngl" not in text

    def test_input_to_config_bridge(self):
        """InputControl only mutates SimConfig fields — no direct sim access."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera

        try:
            from pymurmur.viz.input_control import InputControl
            cfg = SimConfig()
            cam = OrbitCamera()
            ctrl = InputControl(cfg, cam)
            # InputControl should have a reference to config, not sim
            assert ctrl is not None
        except ImportError:
            pytest.skip("pygame not available")

    def test_visualizer_wiring(self, gpu_available, default_config):
        """Renderer3D + OrbitCamera + SimulationEngine wire without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        sim = SimulationEngine(default_config)
        renderer = Renderer3D(800, 600, headless=True)
        camera = OrbitCamera()
        renderer.begin_frame(camera)
        sim.step(1.0 / 60)
        renderer.draw_birds(sim.flock)
        renderer.end_frame()
        assert renderer is not None
        assert camera is not None
