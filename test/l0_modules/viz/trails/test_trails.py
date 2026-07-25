"""P8.3 — Trail rendering tests: init/basic properties, begin_frame, push_history, draw velocity/ring/accumulation.

Covers: TrailRenderer init, mode validation, trail_length, begin_frame,
push_history, draw_velocity, draw_ring, draw_accumulation, draw_lines,
mode toggling, edge cases.
All tests GPU-gated (require ModernGL).

Split out of test_trails.py (file-size split).
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


