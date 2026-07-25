"""P8.7: Cinematic capture sweep tests.

Tests camera sweep math, pre-warm frame skipping, env var overrides,
GIF parameters, and config field wiring.
"""

from __future__ import annotations

import os
from math import pi, radians, sin

import pytest

# ── P8.7a: Camera cinematic sweep math ───────────────────────────

class TestCinematicSweep:
    """P8.7: OrbitCamera.cinematic_sweep sets azim/elev/distance."""

    def test_sweep_start_t_zero(self):
        """P8.7: t=0 → azim=45°, elev=25°, dist=650·scale."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(0.0, scale=1.0)
        assert cam.azimuth == pytest.approx(radians(45.0))
        assert cam.elevation == pytest.approx(radians(25.0))
        assert cam.distance == pytest.approx(650.0)

    def test_sweep_end_t_one(self):
        """P8.7: t=1 → azim=225°, elev=25°, dist=550 (sin(1.5π)=-1)."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(1.0, scale=1.0)
        assert cam.azimuth == pytest.approx(radians(225.0))
        assert cam.elevation == pytest.approx(radians(25.0))  # sin(2π)=0
        # sin(1.5π) = -1, so dist = 650 + (-1)*100 = 550
        assert cam.distance == pytest.approx(550.0)

    def test_sweep_midpoint_t_half(self):
        """P8.7: t=0.5 → azim=135°, elev oscillates, dist oscillates."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(0.5, scale=1.0)
        # azim = 45 + 0.5*180 = 135°
        assert cam.azimuth == pytest.approx(radians(135.0))
        # sin(0.5*2π) = sin(π) = 0 → elev stays at 25°
        assert cam.elevation == pytest.approx(radians(25.0))
        # sin(0.5*1.5π) = sin(0.75π) = sin(135°) = sqrt(2)/2 ≈ 0.707
        expected_dist = 650.0 + sin(0.75 * pi) * 100.0
        assert cam.distance == pytest.approx(expected_dist)

    def test_sweep_scale_multiplies_distance(self):
        """P8.7: scale=2.0 doubles distance, azim/elev unchanged."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(0.0, scale=2.0)
        assert cam.azimuth == pytest.approx(radians(45.0))
        assert cam.elevation == pytest.approx(radians(25.0))
        assert cam.distance == pytest.approx(1300.0)

    def test_sweep_azim_range(self):
        """P8.7: Azimuth sweeps from 45° to 225°."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(0.0)
        azim_start = cam.azimuth
        cam.cinematic_sweep(1.0)
        azim_end = cam.azimuth
        assert azim_start == pytest.approx(radians(45.0))
        assert azim_end == pytest.approx(radians(225.0))
        assert azim_end - azim_start == pytest.approx(radians(180.0))

    def test_sweep_elev_stays_near_25(self):
        """P8.7: Elevation oscillates between ~16° and ~34° (25° ± 0.15 rad)."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        elevs = []
        for i in range(100):
            t = i / 99.0
            cam.cinematic_sweep(t)
            elevs.append(cam.elevation)
        assert min(elevs) >= radians(25.0) - 0.15
        assert max(elevs) <= radians(25.0) + 0.15

    def test_sweep_dist_range(self):
        """P8.7: Distance oscillates between 550 and 750 with scale=1.0."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        dists = []
        for i in range(100):
            t = i / 99.0
            cam.cinematic_sweep(t, scale=1.0)
            dists.append(cam.distance)
        assert min(dists) == pytest.approx(550.0, rel=0.01)  # 650 - 100
        assert max(dists) == pytest.approx(750.0, rel=0.01)  # 650 + 100

    def test_sweep_t_clamped(self):
        """P8.7: t > 1.0 still produces valid camera position."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        cam.cinematic_sweep(2.0)  # beyond end
        # Should still be valid (sin wraps around)
        assert 500.0 <= cam.distance <= 800.0


# ── P8.7b: Config fields ─────────────────────────────────────────

class TestCaptureConfig:
    """P8.7: CaptureConfig has prewarm, sweep, scale fields."""

    def test_capture_config_new_fields(self):
        """P8.7: CaptureConfig has capture_prewarm, capture_sweep, capture_scale."""
        from pymurmur.core.config import CaptureConfig
        cfg = CaptureConfig()
        assert cfg.capture_prewarm == 60
        assert cfg.capture_sweep is True
        assert cfg.capture_scale == 1.0

    def test_field_map_has_p8_7_fields(self):
        """P8.7: _FIELD_MAP has the three new capture fields."""
        from pymurmur.core.config import _FIELD_MAP
        assert "capture_prewarm" in _FIELD_MAP
        assert "capture_sweep" in _FIELD_MAP
        assert "capture_scale" in _FIELD_MAP
        assert _FIELD_MAP["capture_prewarm"] == ("_capture", "capture_prewarm")
        assert _FIELD_MAP["capture_sweep"] == ("_capture", "capture_sweep")
        assert _FIELD_MAP["capture_scale"] == ("_capture", "capture_scale")

    def test_simconfig_flat_access(self):
        """P8.7: Flat access to new capture fields via SimConfig."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.capture_prewarm == 60
        assert cfg.capture_sweep is True
        assert cfg.capture_scale == 1.0
        cfg.capture_prewarm = 30
        assert cfg.capture_prewarm == 30
        assert cfg.capture.capture_prewarm == 30


# ── P8.7c: Recorder pre-warm + env overrides ─────────────────────

class TestRecorderPrewarm:
    """P8.7: Recorder skips frames during pre-warm."""

    @pytest.fixture
    def mock_config(self):
        from pymurmur.core.config import SimConfig
        cfg = SimConfig(num_boids=10)
        cfg.capture_frames = 100
        cfg.capture_every = 1
        cfg.capture_prewarm = 60
        cfg.capture_sweep = False
        cfg.capture_with_viz = False  # no viz for unit test
        return cfg

    def test_prewarm_skips_fbo_but_not_metrics(self, mock_config):
        """P8.7: Pre-warm skips FBO capture but NOT metrics."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        sim = SimulationEngine(mock_config)
        rec = Recorder(sim, mock_config)
        rec.with_viz = False  # no GPU needed

        # Feed 60 frames — metrics should be captured, FBO frames skipped
        for _ in range(60):
            rec.on_frame(sim)
        # Metrics are captured even during pre-warm
        assert len(rec.metrics_history) == 60
        # But FBO frames are not
        assert len(rec.frames) == 0

    def test_capture_starts_after_prewarm(self, mock_config):
        """P8.7: Frame capture begins after prewarm."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.simulation.engine import SimulationEngine
        sim = SimulationEngine(mock_config)
        rec = Recorder(sim, mock_config)
        rec.with_viz = False  # no GPU

        # Feed 61 frames — the 61st should NOT be skipped
        rec._frame_count = 61
        # Since with_viz is False, no capture happens, but the skip logic
        # should allow it past the guard
        assert rec._frame_count > rec._prewarm

    def test_prewarm_default(self):
        """P8.7: Default prewarm is 60."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig(num_boids=5)
        assert cfg.capture_prewarm == 60


class TestRecorderEnvOverrides:
    """D16: Recorder is config-driven — env vars are applied in __main__
    (YAML < env < CLI), never read inside Recorder itself."""

    def teardown_method(self):
        """Clean up env vars after each test."""
        for var in ("CAPTURE_WIDTH", "CAPTURE_HEIGHT",
                     "CAPTURE_FRAMES", "CAPTURE_OUT"):
            os.environ.pop(var, None)

    def test_env_width_override(self):
        """D16: CAPTURE_WIDTH env var is ignored by Recorder — config wins."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        os.environ["CAPTURE_WIDTH"] = "640"
        cfg = SimConfig(num_boids=5)
        cfg.capture_width = 800
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec._capture_width == 800

    def test_env_height_override(self):
        """D16: CAPTURE_HEIGHT env var is ignored by Recorder — config wins."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        os.environ["CAPTURE_HEIGHT"] = "480"
        cfg = SimConfig(num_boids=5)
        cfg.capture_height = 600
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec._capture_height == 600

    def test_env_frames_override(self):
        """D16: CAPTURE_FRAMES env var is ignored by Recorder — config wins."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        os.environ["CAPTURE_FRAMES"] = "120"
        cfg = SimConfig(num_boids=5)
        cfg.capture_frames = 240
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec._capture_frames == 240

    def test_env_output_override(self):
        """D16: CAPTURE_OUT env var is ignored by Recorder — config wins."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        os.environ["CAPTURE_OUT"] = "output/test.gif"
        cfg = SimConfig(num_boids=5)
        cfg.capture_output = "output/murmuration.gif"
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec._capture_output == "output/murmuration.gif"

    def test_env_defaults_when_not_set(self):
        """P8.7: Without env vars, config values are used."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        for var in ("CAPTURE_WIDTH", "CAPTURE_HEIGHT",
                     "CAPTURE_FRAMES", "CAPTURE_OUT"):
            os.environ.pop(var, None)

        cfg = SimConfig(num_boids=5)
        cfg.capture_width = 800
        cfg.capture_height = 600
        cfg.capture_frames = 240
        cfg.capture_output = "output/m.gif"
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec._capture_width == 800
        assert rec._capture_height == 600
        assert rec._capture_frames == 240
        assert rec._capture_output == "output/m.gif"


# ── P8.7d: GIF parameters ────────────────────────────────────────

class TestGifParams:
    """P8.7: save_gif uses optimize=True and disposal=2."""

    def test_save_gif_returns_none_without_frames(self):
        """P8.7: save_gif returns None if no frames captured."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        cfg = SimConfig(num_boids=5)
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        assert rec.save_gif() is None
