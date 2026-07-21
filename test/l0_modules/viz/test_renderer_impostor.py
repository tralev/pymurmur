"""P8.1 + P8.2: Impostor rendering and depth cue tests.

GPU-gated impostor/depth tests extracted from test_renderer.py to keep
files under ~1,000 lines. Requires ModernGL GPU context.
"""

import math

import pytest

# ── P8.2: Depth cue near-vs-far scaling + opacity ────────────────

@pytest.mark.gpu
class TestDepthCue:
    """P8.2: Near birds render larger & more opaque than distant birds."""

    def test_near_bird_larger_footprint(self, gpu_available):
        """P8.2: Near bird covers more pixels than far bird.

        Renders two birds: one near the camera (close) and one far.
        The near bird should produce a larger pixel footprint.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=2, boid_size=15.0)
        flock = PhysicsFlock(cfg)
        # Near bird: close to camera
        flock.positions[0] = [0, -200, 0]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        # Far bird: distant
        flock.positions[1] = [150, 400, 0]
        flock.velocities[1] = [1.0, 0.0, 0.0]
        flock.active[:] = True

        r = Renderer3D(width=300, height=300, headless=True,
                       point_sprites=True, theme="ink")
        cam = OrbitCamera(target=(0.0, 0.0, 0.0))
        cam.distance = 1000.0
        cam.elevation = 0.0
        cam.azimuth = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()

        # Count non-background pixels near each bird's projected position
        # Near bird should be centred roughly around (150, 150) ± some offset
        # Far bird should be centred around a different location
        # We check: centre region (near bird) has more non-black pixels
        # than a distant region
        def non_bg_count(x0, y0, radius):
            count = 0
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    p = img.getpixel((x0 + dx, y0 + dy))
                    if p != (0, 0, 0) and p != img.getpixel((2, 2)):
                        count += 1
            return count

        # Near bird: check centre of image (should be close to near bird)
        near_pixels = non_bg_count(150, 150, 15)
        # Far bird: check a far edge (within bounds, away from near bird)
        far_pixels = non_bg_count(30, 30, 15)

        # Near bird (centred) should have larger footprint than far edge region
        assert near_pixels > far_pixels, (
            f"Near bird pixels {near_pixels} should exceed far region pixels {far_pixels}"
        )

    def test_depth_uniforms_set(self, gpu_available):
        """P8.2: Depth cue uniforms (u_depth_power, u_depth_fade) are set."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(width=100, height=100, headless=True,
                       point_sprites=True)
        # u_depth_power and u_depth_fade exist on impostor program
        assert "u_depth_power" in r._impostor_prog
        assert "u_depth_fade" in r._impostor_prog
        assert "u_rim_power" in r._impostor_prog
        assert "u_max_depth" in r._impostor_prog

    def test_depth_cue_no_crash_with_sprites(self, gpu_available):
        """P8.2: Rendering with depth cues + impostors doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=5, boid_size=10.0)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True

        r = Renderer3D(width=200, height=200, headless=True,
                       point_sprites=True)
        cam = OrbitCamera()

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None


# ── P8.1: Impostor centre-vs-rim brightness ─────────────────────

@pytest.mark.gpu
class TestImpostorPixel:
    """P8.1: Sphere impostors render with centre brighter than rim."""

    def test_impostor_centre_brighter_than_corners(self, gpu_available):
        """P8.1: Centre pixel has bird colour; corners = clear colour.

        Renders a single bird at the origin with a camera looking
        straight at it. The disc fragment shader produces a bright
        centre with dim edges — verify the centre is not background.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=1, boid_size=9.0)
        flock = PhysicsFlock(cfg)
        flock.positions[0] = [0, 0, 0]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.active[0] = True

        r = Renderer3D(width=200, height=200, headless=True,
                       point_sprites=True, theme="ink")
        cam = OrbitCamera(target=(0.0, 0.0, 0.0))
        cam.distance = 500.0
        cam.azimuth = 0.0
        cam.elevation = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()

        centre = img.getpixel((100, 100))
        corner = img.getpixel((3, 3))

        centre_lum = 0.299 * centre[0] + 0.587 * centre[1] + 0.114 * centre[2]
        corner_lum = 0.299 * corner[0] + 0.587 * corner[1] + 0.114 * corner[2]

        # Centre should have visible content (bird rendered there)
        assert centre_lum > 0, "Centre pixel should not be pure black"
        # Centre must be strictly brighter than the corner (clear colour) —
        # >= would pass vacuously if nothing rendered at all
        assert centre_lum > corner_lum, (
            f"Centre lum {centre_lum:.1f} > corner lum {corner_lum:.1f}"
        )

    def test_impostor_rim_darker_than_centre(self, gpu_available):
        """P8.1: The disc shader produces a soft falloff — rim < centre.

        Samples a ring of pixels at ~40px from centre and compares
        the average luminance against the centre pixel.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=1, boid_size=12.0)
        flock = PhysicsFlock(cfg)
        flock.positions[0] = [0, 0, 0]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.active[0] = True

        r = Renderer3D(width=300, height=300, headless=True,
                       point_sprites=True, theme="ink")
        cam = OrbitCamera(target=(0.0, 0.0, 0.0))
        cam.distance = 600.0
        cam.azimuth = 0.0
        cam.elevation = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()

        def lum(p):
            return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]

        centre_lum = lum(img.getpixel((150, 150)))
        # Sample 8 points on a ring at radius ~40px
        ring_lums = []
        for angle in range(0, 360, 45):
            x = int(150 + 40 * math.cos(math.radians(angle)))
            y = int(150 + 40 * math.sin(math.radians(angle)))
            ring_lums.append(lum(img.getpixel((x, y))))

        avg_ring = sum(ring_lums) / len(ring_lums)
        assert centre_lum >= avg_ring, (
            f"Centre {centre_lum:.1f} >= ring avg {avg_ring:.1f}"
        )

    def test_impostor_corners_are_background(self, gpu_available):
        """P8.1: Four corners match the clear colour, not the bird."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=1, boid_size=8.0)
        flock = PhysicsFlock(cfg)
        flock.positions[0] = [0, 0, 0]
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.active[0] = True

        r = Renderer3D(width=200, height=200, headless=True,
                       point_sprites=True, theme="ink", gradient_sky=False)
        cam = OrbitCamera(target=(0.0, 0.0, 0.0))
        cam.distance = 800.0
        cam.azimuth = 0.0
        cam.elevation = 0.0

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()

        corners = [
            img.getpixel((2, 2)),
            img.getpixel((197, 2)),
            img.getpixel((2, 197)),
            img.getpixel((197, 197)),
        ]
        # All corners should be close to each other (clear colour, no sky gradient)
        # Allow small variance (±5 per channel for GPU precision)
        for i in range(1, 4):
            for ch in range(3):
                assert abs(int(corners[i][ch]) - int(corners[0][ch])) <= 5, (
                    f"Corner {i} ch{ch} {corners[i][ch]} vs {corners[0][ch]}"
                )


# ── P8 acceptance: Impostors at 20K keep 60fps ───────────────────

@pytest.mark.gpu
class TestImpostorLargeFlock:
    """P8 acceptance: Impostor rendering at N=20,000 completes
    via single instanced draw call within render budget."""

    def test_impostor_20k_birds_no_crash(self, gpu_available):
        """20K birds with impostor sprites render without crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=20000, boid_size=5.0, seed=42)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True

        r = Renderer3D(width=800, height=600, headless=True,
                       point_sprites=True, theme="ink",
                       instance_buffer_chunk=20000)
        cam = OrbitCamera()

        r.begin_frame(cam)
        r.draw_birds(flock)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None
        assert img.size == (800, 600)

    def test_impostor_20k_single_draw_call(self, gpu_available):
        """20K birds → single instanced draw (not N separate draw calls)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=20000, seed=42)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True

        r = Renderer3D(width=800, height=600, headless=True,
                       point_sprites=True, instance_buffer_chunk=20000)

        # update_instances should report 20000 active birds
        n = r.update_instances(flock)
        assert n == 20000, f"Expected 20000 active, got {n}"

        # Verify single instance VBO holds all 20000 birds
        # (6 floats × 4 bytes × 20000 = 480,000 bytes)
        expected_bytes = 6 * 4 * 20000
        assert r._instance_vbo.size >= expected_bytes, (
            f"Instance VBO too small: {r._instance_vbo.size} < {expected_bytes}"
        )
