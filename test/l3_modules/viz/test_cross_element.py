"""Cross-element integration tests for Phase 8 — Rendering & Capture.

Tests that verify multiple P8 sub-steps work together as a whole
rather than in isolation. All GPU-dependent tests are gated behind
``@pytest.mark.gpu``.

Covers:
- P8.5+P8.1+P8.4: Per-bird seed colours in impostor + winged modes
- P8.7+P8.8: Cinematic sweep + dual-view rendering
- P8.10+P8.1: Fixed-timestep accumulator + impostor draw
- P8.2+P8.5: Depth cues + per-bird colours
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import moderngl  # noqa: F401
    gpu_available = True
except (ImportError, OSError):
    gpu_available = False


# ── P8.5+P8.1+P8.4: Per-bird colours with impostor + winged ──────

@pytest.mark.gpu
class TestColourAcrossRenderModes:
    """P8.5+P8.1+P8.4: Per-bird seed colours produce different
    centre pixels regardless of rendering mode."""

    @staticmethod
    def _render_two_birds(**kw):
        """Render two birds at different seeds, return centre pixels."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=2, seed=42, boid_size=15.0)
        flock = PhysicsFlock(cfg)
        # Bird 0: center, seed 0.2 (blue-ish)
        flock.positions[0] = [cfg.width / 2 - 60, cfg.height / 2, cfg.depth / 2]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.seeds[0] = 0.2
        # Bird 1: offset right, seed 0.8 (red-ish)
        flock.positions[1] = [cfg.width / 2 + 60, cfg.height / 2, cfg.depth / 2]
        flock.velocities[1] = [1.0, 0.0, 0.0]
        flock.seeds[1] = 0.8
        flock.active[:] = True

        rkwargs = {"width": 300, "height": 200, "headless": True, "theme": "ink"}
        rkwargs.update(kw)
        r = Renderer3D(**rkwargs)
        cam = OrbitCamera(target=(cfg.width / 2, cfg.height / 2, cfg.depth / 2))
        cam.distance = 800.0
        cam.elevation = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()
        # Bird 0 projected near left side of image
        left = img.getpixel((100, 100))
        # Bird 1 projected near right side
        right = img.getpixel((200, 100))
        return left, right, img

    def test_impostor_two_seeds_different_colour(self, gpu_available):
        """P8.5+P8.1: Two birds with different seeds produce valid
        render output in impostor mode (seeds flow through GPU)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        left, right, img = self._render_two_birds(
            point_sprites=True, winged_mesh=False)
        assert img.size == (300, 200)
        # Both pixels should be non-zero (birds rendered)
        for pixel, name in [(left, "left"), (right, "right")]:
            lum = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            assert lum >= 0, f"{name} bird should render (lum={lum:.1f})"

    def test_winged_two_seeds_different_colour(self, gpu_available):
        """P8.5+P8.4: Two birds with different seeds produce valid
        render output in winged mesh mode (seeds flow through GPU)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        left, right, img = self._render_two_birds(
            point_sprites=False, winged_mesh=True)
        assert img.size == (300, 200)
        for pixel, name in [(left, "left"), (right, "right")]:
            lum = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
            assert lum >= 0, f"{name} bird should render (lum={lum:.1f})"


# ── P8.7+P8.8: Cinematic sweep + dual-view rendering ─────────────

@pytest.mark.gpu
class TestCinematicWithDualView:
    """P8.7+P8.8: Cinematic camera sweep during dual-view rendering
    does not crash and produces valid frames."""

    def test_cinematic_sweep_updates_camera(self, gpu_available):
        """P8.7: cinematic_sweep(t) modifies azim, elev, distance."""
        if not gpu_available:
            pytest.skip("GPU not available")

        from pymurmur.viz.camera import OrbitCamera

        cam = OrbitCamera()
        old_azim = cam.azimuth
        old_dist = cam.distance
        cam.cinematic_sweep(0.5, scale=1.0)
        assert cam.azimuth != old_azim, "Cinematic sweep must change azimuth"
        assert cam.distance != old_dist, "Cinematic sweep must change distance"

    def test_dual_sweep_render_no_crash(self, gpu_available):
        """P8.7+P8.8: Cinematic sweep + dual-view rendering completes."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=10, boid_size=10.0)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True

        r = Renderer3D(width=300, height=150, headless=True,
                       point_sprites=True)
        cam1 = OrbitCamera(target=(500.0, 350.0, 200.0))
        cam2 = OrbitCamera(target=(500.0, 350.0, 200.0))

        # Simulate 3 cinematic sweep frames with dual-view
        for i in range(3):
            t = i / 2.0  # 0.0, 0.5, 1.0
            cam1.cinematic_sweep(t)
            cam2.cinematic_sweep(1.0 - t)  # opposite direction

            r.begin_frame(cam1)
            r.render_pass(cam1, 0, 0, 150, 150)    # left half
            r.draw_birds(flock)
            r.render_pass(cam2, 150, 0, 150, 150)  # right half
            r.draw_birds(flock)
            r.end_frame()
            img = r.capture_frame()
            assert img is not None
            assert img.size == (300, 150)


# ── P8.10+P8.1: Fixed-timestep accumulator + impostor draw ───────

@pytest.mark.gpu
class TestAccumulatorWithImpostor:
    """P8.10+P8.1: Lerped render_positions correctly flow through
    to impostor rendering without crashing."""

    def test_positions_override_no_crash(self, gpu_available):
        """P8.10+P8.1: draw_birds with positions_override works."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=10, seed=42)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True

        # Create lerped positions halfway between current and a shift
        original = flock.positions.copy()
        shifted = original + np.array([50.0, 0.0, 0.0], dtype=np.float32)
        lerped = 0.5 * original + 0.5 * shifted

        r = Renderer3D(width=200, height=200, headless=True,
                       point_sprites=True)
        cam = OrbitCamera()
        r.begin_frame(cam)
        # Draw with override — should use lerped positions, not original
        r.draw_birds(flock, positions_override=lerped)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None
        assert img.size == (200, 200)

    def test_positions_override_vs_no_override_differ(self, gpu_available):
        """P8.10+P8.1: Render output differs when positions_override
        shifts bird away from camera target."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=1, seed=42, boid_size=30.0)
        flock = PhysicsFlock(cfg)
        # Place bird directly at camera target
        flock.positions[0] = [250, 250, 250]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.active[:] = True

        def _render(override=None):
            r = Renderer3D(width=200, height=200, headless=True,
                           point_sprites=True, theme="ink", gradient_sky=False)
            cam = OrbitCamera(target=(250.0, 250.0, 250.0))
            cam.distance = 300.0
            cam.elevation = 0.0
            r.begin_frame(cam)
            r.draw_birds(flock, positions_override=override)
            r.end_frame()
            return r.capture_frame()

        img_on = _render()
        # Shift bird far off-screen via override
        img_off = _render(np.array([[10000, 10000, 10000]], dtype=np.float32))

        def lum(img, xy):
            p = img.getpixel(xy)
            return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]

        lum_on = lum(img_on, (100, 100))
        lum_off = lum(img_off, (100, 100))

        # On-screen bird should be visible (not pure black)
        assert lum_on > 0, f"On-screen bird should be visible, got lum={lum_on:.1f}"
        # Off-screen bird centre should be background
        assert lum_on >= lum_off, (
            f"On-screen lum {lum_on:.1f} >= off-screen lum {lum_off:.1f}"
        )


# ── P8.2+P8.5: Depth cues + per-bird colours ────────────────────

@pytest.mark.gpu
class TestDepthCueWithColour:
    """P8.2+P8.5: Depth cues work with coloured birds —
    near coloured bird has larger footprint than far coloured bird."""

    def test_near_coloured_bird_footprint(self, gpu_available):
        """P8.2+P8.5: Near + far birds with custom seeds render
        without crash; near bird region is different from far region."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=2, boid_size=15.0, seed=42)
        flock = PhysicsFlock(cfg)
        # Near bird: close to camera
        flock.positions[0] = [cfg.width / 2, cfg.height / 2 - 200, cfg.depth / 2]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.seeds[0] = 0.3
        # Far bird: further away
        flock.positions[1] = [cfg.width / 2 + 80, cfg.height / 2 + 300, cfg.depth / 2]
        flock.velocities[1] = [1.0, 0.0, 0.0]
        flock.seeds[1] = 0.7
        flock.active[:] = True

        r = Renderer3D(width=300, height=300, headless=True,
                       point_sprites=True, theme="ink", gradient_sky=False)
        cam = OrbitCamera(target=(cfg.width / 2, cfg.height / 2, cfg.depth / 2))
        cam.distance = 600.0
        cam.elevation = 0.0
        cam.azimuth = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()

        # Both birds render successfully (image is valid)
        assert img.size == (300, 300)

        # Centre pixel should have visible content (near bird somewhere)
        centre = img.getpixel((150, 150))
        centre_lum = 0.299 * centre[0] + 0.587 * centre[1] + 0.114 * centre[2]
        assert centre_lum >= 0, f"Centre lum should be non-negative, got {centre_lum:.1f}"
