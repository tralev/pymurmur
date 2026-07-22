"""P8.3 — Trail rendering tests: velocity lines, ring-history sprites,
screen-space accumulation, and CPU ribbon lines.

Covers: TrailRenderer init, mode validation, trail_length, begin_frame,
push_history, draw_velocity, draw_ring, draw_accumulation, draw_lines,
mode toggling, edge cases.
All tests GPU-gated (require ModernGL).
"""

import moderngl
import numpy as np
import pytest

# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def headless_ctx():
    """Standalone ModernGL context for trail tests."""
    return moderngl.create_context(standalone=True, require=330)


@pytest.fixture
def fake_camera():
    """Mock orbit camera with view/projection matrix methods."""
    class FakeCamera:
        def view_matrix(self):
            import glm
            return glm.mat4(1.0)
        def projection_matrix(self, aspect=1.0):
            import glm
            return glm.perspective(glm.radians(45.0), aspect, 0.1, 5000.0)
    return FakeCamera()


@pytest.fixture
def fake_flock():
    """Mock flock: 20 active (slots 0-19), 10 inactive (20-29)."""
    class FakeFlock:
        N_capacity = 30
        N_active = 20
        active = np.array([True]*20 + [False]*10, dtype=bool)
        positions = np.random.default_rng(42).uniform(0, 500, (30, 3)).astype(np.float32)
        velocities = np.random.default_rng(43).uniform(-2, 2, (30, 3)).astype(np.float32)
        seeds = np.random.default_rng(44).uniform(0.0, 1.0, 30).astype(np.float32)
        position_history = None
    return FakeFlock()


@pytest.fixture
def fake_instance_vbo(headless_ctx, fake_flock):
    """Instance buffer with pos + vel per bird."""
    data = np.zeros((fake_flock.N_capacity, 6), dtype=np.float32)
    data[:, :3] = fake_flock.positions
    data[:, 3:] = fake_flock.velocities
    return headless_ctx.buffer(data.tobytes())


# ── Init + basic properties ────────────────────────────────────

@pytest.mark.gpu
class TestTrailRendererInit:
    """TrailRenderer construction and basic properties."""

    def test_init_default_mode_off(self, headless_ctx, gpu_available):
        """Default mode is 'off'."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx)
        assert t.mode == "off"

    def test_init_with_mode(self, headless_ctx, gpu_available):
        """Construct with explicit mode."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        assert t.mode == "velocity"

    def test_init_with_trail_length(self, headless_ctx, gpu_available):
        """trail_length is stored correctly."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, trail_length=15)
        assert t.trail_length == 15

    def test_mode_setter(self, headless_ctx, gpu_available):
        """Mode can be changed at runtime."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx)
        t.mode = "ring"
        assert t.mode == "ring"
        t.mode = "velocity"
        assert t.mode == "velocity"

    def test_mode_setter_invalid_raises(self, headless_ctx, gpu_available):
        """Invalid mode raises ValueError."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx)
        with pytest.raises(ValueError, match="Unknown trail mode"):
            t.mode = "nonexistent"

    def test_trail_length_setter(self, headless_ctx, gpu_available):
        """trail_length can be changed at runtime."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, trail_length=10)
        t.trail_length = 50
        assert t.trail_length == 50

    def test_trail_length_clamped_min_1(self, headless_ctx, gpu_available):
        """trail_length is clamped to minimum 1."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, trail_length=-5)
        assert t.trail_length == 1
        t.trail_length = 0
        assert t.trail_length == 1


# ── begin_frame ─────────────────────────────────────────────────

@pytest.mark.gpu
class TestBeginFrame:
    """begin_frame uploads camera uniforms to shader programs."""

    def test_begin_frame_does_not_crash(self, headless_ctx, fake_camera, gpu_available):
        """begin_frame with valid camera and aspect ratio."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t.begin_frame(fake_camera, aspect=1.5)
        # No crash = pass

    def test_begin_frame_ring_mode(self, headless_ctx, fake_camera, gpu_available):
        """begin_frame works with ring mode too."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring")
        t.begin_frame(fake_camera, aspect=1.0)
        # No crash = pass


# ── push_history ────────────────────────────────────────────────

@pytest.mark.gpu
class TestPushHistory:
    """push_history manages the position_history ring buffer."""

    def test_push_history_initialises_buffer(self, headless_ctx, fake_flock, gpu_available):
        """First push_history call initialises position_history."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=30)
        assert fake_flock.position_history is None

        t.push_history(fake_flock)
        assert fake_flock.position_history is not None
        assert fake_flock.position_history.shape == (fake_flock.N_capacity, 30, 3)

    def test_push_history_records_current_positions(self, headless_ctx, fake_flock, gpu_available):
        """Slot 0 of history matches current positions after push."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=10)
        t.push_history(fake_flock)

        np.testing.assert_array_equal(
            fake_flock.position_history[:, 0, :],
            fake_flock.positions,
        )

    def test_push_history_shifts_old_entries(self, headless_ctx, fake_flock, gpu_available):
        """Second push moves previous data to slot 1."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=10)

        pos_before = fake_flock.positions.copy()
        t.push_history(fake_flock)

        # Change positions
        fake_flock.positions += 1.0
        t.push_history(fake_flock)

        # Slot 0 = new positions, slot 1 = old positions
        np.testing.assert_array_equal(
            fake_flock.position_history[:, 0, :],
            fake_flock.positions,
        )
        np.testing.assert_array_equal(
            fake_flock.position_history[:, 1, :],
            pos_before,
        )

    def test_push_history_noop_when_off(self, headless_ctx, fake_flock, gpu_available):
        """push_history is a no-op when mode is 'off'."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="off")
        t.push_history(fake_flock)
        assert fake_flock.position_history is None

    def test_push_history_noop_when_velocity(self, headless_ctx, fake_flock, gpu_available):
        """push_history is a no-op when mode is 'velocity'."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t.push_history(fake_flock)
        assert fake_flock.position_history is None

    def test_ensure_history_seeds_with_current(self, headless_ctx, fake_flock, gpu_available):
        """ensure_history fills all slots with current positions."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=5)
        t.ensure_history(fake_flock)

        hist = fake_flock.position_history
        for k in range(5):
            np.testing.assert_array_equal(
                hist[:, k, :], fake_flock.positions,
                err_msg=f"Slot {k} should be seeded with current positions",
            )


# ── draw — velocity mode ───────────────────────────────────────

@pytest.mark.gpu
class TestDrawVelocity:
    """Velocity trail rendering."""

    def test_draw_velocity_does_not_crash(self, headless_ctx, fake_flock,
                                          fake_instance_vbo, fake_camera, gpu_available):
        """draw() with velocity mode renders without crash."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

    def test_draw_velocity_zero_instances_noop(self, headless_ctx, fake_flock,
                                                fake_instance_vbo, fake_camera, gpu_available):
        """draw() with 0 instances does nothing."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, 0)  # should not crash

    def test_draw_velocity_no_unnecessary_realloc(self, headless_ctx, fake_flock,
                                                    fake_instance_vbo, fake_camera,
                                                    gpu_available):
        """Velocity buffer doesn't reallocate when within capacity."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        assert t._velocity_capacity == 100000  # default

        # Draw with normal count → no realloc
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t._velocity_capacity == 100000  # unchanged


# ── draw — ring mode ───────────────────────────────────────────

@pytest.mark.gpu
class TestDrawRing:
    """Ring trail rendering."""

    def test_draw_ring_does_not_crash(self, headless_ctx, fake_flock,
                                       fake_instance_vbo, fake_camera, gpu_available):
        """draw() with ring mode renders without crash."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=10)
        t.push_history(fake_flock)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

    def test_draw_ring_no_history_noop(self, headless_ctx, fake_flock,
                                        fake_instance_vbo, fake_camera, gpu_available):
        """draw_ring with no position_history does nothing."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=10)
        # Don't call push_history
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)  # should not crash

    def test_draw_ring_zero_instances_noop(self, headless_ctx, fake_flock,
                                            fake_instance_vbo, fake_camera, gpu_available):
        """draw_ring with 0 instances does nothing."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=10)
        t.push_history(fake_flock)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, 0)  # should not crash


# ── Edge cases ─────────────────────────────────────────────────

@pytest.mark.gpu
class TestDrawAccumulation:
    """Accumulation mode — FBO persistence, decay, and blit.

    NOTE: this class was previously named TestTrailEdgeCases, which was
    silently shadowed by the second TestTrailEdgeCases class below — the
    accumulation tests were never collected.
    """

    def test_draw_accumulation_does_not_crash(self, headless_ctx, fake_flock,
                                                   fake_instance_vbo, fake_camera,
                                                   gpu_available):
        """draw() with accumulation mode renders without crash."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

    def test_draw_accumulation_blit_after_draw(self, headless_ctx, fake_flock,
                                                 fake_instance_vbo, fake_camera,
                                                 gpu_available):
        """blit_accumulation() renders without crash after draw."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        t.blit_accumulation()  # blits persistent FBO to main framebuffer

    def test_draw_accumulation_zero_instances_noop(self, headless_ctx, fake_flock,
                                                     fake_instance_vbo, fake_camera,
                                                     gpu_available):
        """draw() with accumulation + 0 instances does nothing."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, 0)  # should not crash

    def test_accumulation_fbo_created_lazily(self, headless_ctx, fake_flock,
                                               fake_instance_vbo, fake_camera,
                                               gpu_available):
        """Accumulation FBO is None until first draw."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        assert t._accum_fbo is None
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        # After draw, FBO should be created
        assert t._accum_fbo is not None
        assert t._accum_tex is not None


# ── draw — lines (ribbon) mode ─────────────────────────────────

@pytest.mark.gpu
class TestDrawLines:
    """CPU sinusoidal ribbon line rendering."""

    def test_draw_lines_does_not_crash(self, headless_ctx, fake_flock,
                                         fake_instance_vbo, fake_camera, gpu_available):
        """draw() with lines mode renders without crash."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="lines", trail_length=15)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

    def test_draw_lines_zero_instances_noop(self, headless_ctx, fake_flock,
                                              fake_instance_vbo, fake_camera, gpu_available):
        """draw() with lines + 0 instances does nothing."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="lines")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, 0)  # should not crash

    def test_lines_ribbon_vertex_count(self, headless_ctx, fake_flock,
                                         fake_instance_vbo, fake_camera, gpu_available):
        """S4.3: Lines mode generates exactly 10 GL_LINES vertices (5
        disjoint segments) per active bird, regardless of trail_length —
        trail_length only scales trailScale, not the segment count."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="lines", trail_length=20)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t._lines_count == 200  # 20 active x 10 vertices/bird

    def test_lines_stationary_bird_no_crash(self, headless_ctx, fake_instance_vbo,
                                              fake_camera, gpu_available):
        """Stationary bird (zero velocity) doesn't crash lines mode."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="lines", trail_length=10)

        class StationaryFlock:
            N_capacity = 1
            N_active = 1
            active = np.array([True], dtype=bool)
            positions = np.array([[250, 250, 250]], dtype=np.float32)
            velocities = np.array([[0, 0, 0]], dtype=np.float32)
            seeds = np.array([0.5], dtype=np.float32)
            position_history = None

        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(StationaryFlock(), fake_instance_vbo, 1)  # should not crash

    def test_lines_trail_length_scales_reach_not_segment_count(
        self, headless_ctx, fake_flock, fake_instance_vbo, fake_camera, gpu_available,
    ):
        """S4.3: trail_length changes trailScale (how far back the ribbon
        reaches), not the number of segments — always 10 vertices/bird."""
        from pymurmur.viz.trails import TrailRenderer

        t5 = TrailRenderer(headless_ctx, mode="lines", trail_length=5)
        t5.begin_frame(fake_camera, aspect=1.0)
        t5.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t5._lines_count == 200  # 20 x 10, unaffected by trail_length

        t30 = TrailRenderer(headless_ctx, mode="lines", trail_length=30)
        t30.begin_frame(fake_camera, aspect=1.0)
        t30.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t30._lines_count == 200  # still 20 x 10


# ── Edge cases (continued) ─────────────────────────────────────

@pytest.mark.gpu
class TestTrailEdgeCases:
    """Trail rendering edge cases and error handling."""

    def test_draw_off_mode_noop(self, headless_ctx, fake_flock,
                                 fake_instance_vbo, fake_camera, gpu_available):
        """draw() with mode='off' is a no-op."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="off")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)  # should not crash

    def test_empty_flock_no_active(self, headless_ctx, fake_instance_vbo,
                                    fake_camera, gpu_available):
        """Velocity draw with all-inactive flock doesn't crash."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")

        class EmptyFlock:
            N_capacity = 5
            N_active = 0
            active = np.zeros(5, dtype=bool)
            positions = np.zeros((5, 3), dtype=np.float32)
            velocities = np.zeros((5, 3), dtype=np.float32)
            position_history = None

        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(EmptyFlock(), fake_instance_vbo, 0)  # should not crash

    def test_mode_toggle_velocity_to_ring(self, headless_ctx, fake_flock,
                                           fake_instance_vbo, fake_camera, gpu_available):
        """Toggle from velocity to ring mid-session."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

        t.mode = "ring"
        t.push_history(fake_flock)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)  # should not crash

    def test_ring_history_trail_length_respected(self, headless_ctx, fake_flock,
                                                  fake_instance_vbo, fake_camera, gpu_available):
        """K history slots = min(buffer K, trail_length)."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=5)
        t.push_history(fake_flock)

        # Buffer was initialised with trail_length=5
        assert fake_flock.position_history.shape[1] == 5

        # Change trail_length → buffer isn't shrunk, just fewer slots drawn
        t.trail_length = 3
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)  # should not crash


# ── Coverage-gap tests (2026-07-19 audit): buffer growth, degenerate
#    inputs, FBO resize, release, and Renderer3D accumulation wiring ──

@pytest.mark.gpu
class TestTrailBufferGrowth:
    """Reallocation-on-growth paths for all three CPU-side buffers."""

    def test_velocity_buffer_grows_beyond_capacity(self, headless_ctx, fake_flock,
                                                   fake_instance_vbo, fake_camera):
        """draw() with more birds than capacity reallocates and rebuilds VAO."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="velocity")
        t._velocity_capacity = 5  # force the growth branch with 20 birds
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t._velocity_capacity >= fake_flock.N_active

    def test_ring_buffer_grows_beyond_capacity(self, headless_ctx, fake_flock,
                                               fake_instance_vbo, fake_camera):
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="ring", trail_length=5)
        t.push_history(fake_flock)
        t._ring_capacity = 5  # 20 birds × 5 slots = 100 > 5
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t._ring_capacity >= fake_flock.N_active * 5

    def test_lines_buffer_grows_beyond_capacity(self, headless_ctx, fake_flock,
                                                fake_instance_vbo, fake_camera):
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="lines")
        t._lines_capacity = 5
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        assert t._lines_capacity >= t._lines_count


@pytest.mark.gpu
class TestTrailDegenerateInputs:
    """Degenerate velocity vectors must not crash the ribbon generator."""

    def test_lines_vertical_velocity_perp_fallback(self, headless_ctx, fake_flock,
                                                   fake_instance_vbo, fake_camera):
        """Velocity along +Z has a zero XY-perp — fallback axis is used."""
        from pymurmur.viz.trails import TrailRenderer
        fake_flock.velocities[:] = 0.0
        fake_flock.velocities[:, 2] = 2.0
        t = TrailRenderer(headless_ctx, mode="lines")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)

    def test_lines_stationary_bird_repeats_position(self, headless_ctx, fake_flock,
                                                    fake_instance_vbo, fake_camera):
        """Zero velocity → trail collapses to the bird position, no NaNs."""
        from pymurmur.viz.trails import TrailRenderer
        fake_flock.velocities[:] = 0.0
        t = TrailRenderer(headless_ctx, mode="lines")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)


@pytest.mark.gpu
class TestAccumulationFboLifecycle:
    """Accumulation FBO lazy-create, resize-recreate, and release."""

    def test_fbo_recreated_on_size_change(self, headless_ctx):
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t._ensure_accum_fbo(64, 64)
        assert t._accum_tex.size == (64, 64)
        t._ensure_accum_fbo(128, 128)
        assert t._accum_tex.size == (128, 128)

    def test_fbo_reused_when_size_unchanged(self, headless_ctx):
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t._ensure_accum_fbo(64, 64)
        tex = t._accum_tex
        t._ensure_accum_fbo(64, 64)
        assert t._accum_tex is tex

    def test_release_clears_gpu_state(self, headless_ctx, fake_flock,
                                      fake_instance_vbo, fake_camera):
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(fake_flock, fake_instance_vbo, fake_flock.N_active)
        t.release()
        assert t._accum_fbo is None and t._accum_tex is None

    def test_blit_without_draw_is_noop(self, headless_ctx):
        """blit_accumulation() before any draw must be a safe no-op."""
        from pymurmur.viz.trails import TrailRenderer
        t = TrailRenderer(headless_ctx, mode="accumulation")
        t.blit_accumulation()  # no accumulation texture yet


@pytest.mark.gpu
class TestRendererTrailWiring:
    """Renderer3D.draw_trails drives TrailRenderer, incl. the accumulation
    restore-then-blit path (renderer.py lines that TrailRenderer-only
    tests cannot reach)."""

    def _run_mode(self, mode):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig()
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        r = Renderer3D(width=128, height=128, headless=True,
                       trails_mode=mode, trails_length=5)
        cam = OrbitCamera()
        r.begin_frame(cam)
        r.draw_birds(flock)
        r.draw_trails(flock)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None
        assert img.size == (128, 128)

    def test_draw_trails_velocity_through_renderer(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        self._run_mode("velocity")

    def test_draw_trails_accumulation_through_renderer(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        self._run_mode("accumulation")

    def test_draw_trails_lines_through_renderer(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        self._run_mode("lines")


class TestLinesRibbonGeometry:
    """S4.3: verify the exact per-vertex layout of the lines ribbon
    directly on the CPU-side buffer (not just that it renders)."""

    def test_head_vertex_has_zero_wave_displacement(
        self, headless_ctx, fake_camera, gpu_available,
    ):
        """prog=0 (the head, k=0/j=0) must equal the bird's raw position
        exactly — the wave amplitude vanishes as prog^2 at the head."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import moderngl

        from pymurmur.viz.trails import TrailRenderer

        class OneBird:
            N_capacity = 1
            N_active = 1
            active = np.array([True], dtype=bool)
            positions = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
            velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            seeds = np.array([0.3], dtype=np.float32)
            position_history = None

        flock = OneBird()
        t = TrailRenderer(headless_ctx, mode="lines", trail_length=10)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(flock, headless_ctx.buffer(reserve=8 * 4), 1)

        raw = t._lines_vbo.read(10 * 3 * 4)
        verts = np.frombuffer(raw, dtype=np.float32).reshape(10, 3)
        # Vertex 0 is (k=0, j=0) -> prog=0 -> zero wave, zero trailScale offset
        assert np.allclose(verts[0], flock.positions[0], atol=1e-4)

    def test_segments_trace_backward_along_velocity(
        self, headless_ctx, fake_camera, gpu_available,
    ):
        """Later segments (higher prog) are displaced further backward
        along -v_hat than earlier ones (trailScale*prog term)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.trails import TrailRenderer

        class OneBird:
            N_capacity = 1
            N_active = 1
            active = np.array([True], dtype=bool)
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            seeds = np.array([0.0], dtype=np.float32)
            position_history = None

        flock = OneBird()
        t = TrailRenderer(headless_ctx, mode="lines", trail_length=10)
        t.begin_frame(fake_camera, aspect=1.0)
        t.draw(flock, headless_ctx.buffer(reserve=8 * 4), 1)

        raw = t._lines_vbo.read(10 * 3 * 4)
        verts = np.frombuffer(raw, dtype=np.float32).reshape(10, 3)
        # Vertex indices 0,2,4,6,8 are the "j=0" (segment start) endpoints
        # at prog = 0, 1/5, 2/5, 3/5, 4/5 — x should decrease monotonically
        # (moving backward, i.e. -x, since velocity is +x).
        xs = verts[0::2, 0]
        assert all(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)), (
            f"Segments should trace backward along velocity: {xs}"
        )
