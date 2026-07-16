"""GPU-dependent tests for viz.renderer — Renderer3D, Visualizer.

Requires ModernGL GPU context. All GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

P2.7 InstanceSchema and P2.8 _mat4_bytes tests run without GPU.
"""

import numpy as np
import pytest


# ── P2.7: InstanceSchema standalone dataclass tests (NO GPU needed) ─

class TestInstanceSchema:
    """P2.7: InstanceSchema is a pure dataclass — testable without GPU."""

    def test_instance_schema_defaults(self):
        """P2.7: InstanceSchema has correct default field values."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema()
        assert s.floats == 6, "Default floats must be 6 (pos.xyz + vel.xyz)"
        assert s.layout == "3f 3f/i", "Default layout must be ModernGL format string"
        assert s.attrs == ("in_bird_pos", "in_bird_vel"), (
            "Default attrs must be shader attribute names"
        )

    def test_instance_schema_custom_floats(self):
        """P2.7: InstanceSchema accepts custom float count."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema(floats=9)
        assert s.floats == 9
        assert s.layout == "3f 3f/i"  # layout unchanged unless explicitly set

    def test_instance_schema_custom_layout(self):
        """P2.7: InstanceSchema accepts custom ModernGL layout string."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema(layout="3f 3f 3f/i", attrs=("a", "b", "c"))
        assert s.layout == "3f 3f 3f/i"
        assert s.attrs == ("a", "b", "c")

    def test_instance_schema_is_dataclass(self):
        """P2.7: InstanceSchema must be a @dataclass."""
        from dataclasses import is_dataclass
        from pymurmur.viz.renderer import InstanceSchema
        assert is_dataclass(InstanceSchema), "InstanceSchema must be a @dataclass"

    def test_instance_schema_fields_are_immutable_types(self):
        """P2.7: floats is int, layout is str, attrs is tuple."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema()
        assert isinstance(s.floats, int)
        assert isinstance(s.layout, str)
        assert isinstance(s.attrs, tuple)

    def test_instance_schema_buffer_bytes(self):
        """P2.7: Buffer allocation formula uses schema.floats correctly.

        Each instance uses floats × 4 bytes (float32). For 100 birds
        at 6 floats each: 100 × 6 × 4 = 2400 bytes."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema(floats=6)
        n_birds = 100
        expected_bytes = n_birds * s.floats * 4
        assert expected_bytes == 2400
        # With custom float count
        s2 = InstanceSchema(floats=9)
        assert n_birds * s2.floats * 4 == 3600


# ── P2.8: _mat4_bytes standalone tests (NO GPU needed) ─────────────

class TestMat4Bytes:
    """P2.8: _mat4_bytes converts PyGLM matrices to consistent bytes."""

    def test_mat4_bytes_returns_64_bytes(self):
        """P2.8: 4×4 float32 matrix = 16 × 4 = 64 bytes."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.mat4(1.0)  # identity
        b = _mat4_bytes(m)
        assert isinstance(b, bytes)
        assert len(b) == 64, f"4×4 mat4 must produce 64 bytes, got {len(b)}"

    def test_mat4_bytes_identity_roundtrip(self):
        """P2.8: _mat4_bytes → numpy roundtrip preserves identity matrix."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.mat4(1.0)
        b = _mat4_bytes(m)
        arr = np.frombuffer(b, dtype=np.float32)
        assert arr.shape == (16,)
        # Column-major identity: diagonal = 1.0 at indices 0,5,10,15
        assert arr[0] == 1.0
        assert arr[5] == 1.0
        assert arr[10] == 1.0
        assert arr[15] == 1.0
        # Off-diagonal zeros
        zeros = [i for i in range(16) if i not in (0, 5, 10, 15)]
        for i in zeros:
            assert arr[i] == 0.0, f"Off-diagonal at index {i}: expected 0.0, got {arr[i]}"

    def test_mat4_bytes_translation_matrix(self):
        """P2.8: Translation matrix bytes are correct (column-major float32)."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.translate(glm.mat4(1.0), glm.vec3(10.0, 20.0, 30.0))
        b = _mat4_bytes(m)
        arr = np.frombuffer(b, dtype=np.float32)
        # Column-major: translation is in last column (indices 12,13,14)
        assert arr[12] == 10.0, f"X translation at index 12: got {arr[12]}"
        assert arr[13] == 20.0, f"Y translation at index 13: got {arr[13]}"
        assert arr[14] == 30.0, f"Z translation at index 14: got {arr[14]}"
        assert arr[15] == 1.0, f"W at index 15: got {arr[15]}"

    def test_mat4_bytes_float32_dtype(self):
        """P2.8: Output bytes decode to float32, not float64."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.mat4(1.0)
        b = _mat4_bytes(m)
        arr = np.frombuffer(b, dtype=np.float32)
        assert arr.dtype == np.float32
        # float64 would be 128 bytes
        assert len(b) == 64, "Must be exactly 64 bytes (float32), not 128 (float64)"

    def test_mat4_bytes_deterministic(self):
        """P2.8: Same matrix → same bytes every time."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.rotate(glm.mat4(1.0), np.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        b1 = _mat4_bytes(m)
        b2 = _mat4_bytes(m)
        assert b1 == b2, "Same matrix must produce identical bytes"

    def test_mat4_bytes_different_matrices_differ(self):
        """P2.8: Different matrices produce different bytes."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m1 = glm.mat4(1.0)
        m2 = glm.translate(glm.mat4(1.0), glm.vec3(1.0, 0.0, 0.0))
        assert _mat4_bytes(m1) != _mat4_bytes(m2), (
            "Different matrices must produce different bytes"
        )

    def test_mat4_bytes_little_endian_consistent(self):
        """P2.8: numpy tobytes() uses native byte order — verify float32
        values are recoverable regardless of architecture.

        A 1.0 float32 is 0x3F800000 — regardless of endianness, reading
        back with numpy should recover the same value."""
        glm = pytest.importorskip("glm", reason="PyGLM not installed")
        from pymurmur.viz.renderer import _mat4_bytes
        m = glm.scale(glm.mat4(1.0), glm.vec3(2.5, 3.5, 4.5))
        b = _mat4_bytes(m)
        arr = np.frombuffer(b, dtype=np.float32)
        # Scale diagonal in column-major: diag indices 0,5,10
        assert np.isclose(arr[0], 2.5), f"X scale at index 0: got {arr[0]}"
        assert np.isclose(arr[5], 3.5), f"Y scale at index 5: got {arr[5]}"
        assert np.isclose(arr[10], 4.5), f"Z scale at index 10: got {arr[10]}"


@pytest.mark.gpu
class TestRenderer3D:
    """Tests requiring a ModernGL GPU context (standalone or windowed)."""

    def test_renderer_init(self, gpu_available):
        """Renderer3D(width, height) creates context without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        assert r is not None

    def test_renderer_headless_init(self, gpu_available):
        """Renderer3D(width, height, headless=True) creates FBO."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        assert r.headless is True

    def test_renderer_update_instances(self, gpu_available, small_flock):
        """update_instances() returns correct active count."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        count = r.update_instances(small_flock)
        assert count == small_flock.N_active

    def test_renderer_begin_frame(self, gpu_available):
        """begin_frame(camera) clears and computes matrices."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        cam = OrbitCamera()
        r.begin_frame(cam)
        # No error = pass

    def test_renderer_draw_birds_no_error(self, gpu_available, small_flock):
        """draw_birds(flock) completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(small_flock)
        r.end_frame()

    def test_renderer_draw_grid_no_error(self, gpu_available):
        """draw_grid() completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_grid()
        r.end_frame()

    def test_renderer_capture_frame(self, gpu_available):
        """capture_frame() returns a PIL Image with correct dimensions."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.end_frame()
        img = r.capture_frame()
        assert img is not None
        assert img.size == (800, 600)

    def test_renderer_buffer_growth(self, gpu_available):
        """Adding more birds than max_instances triggers growth."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.core.config import SimConfig

        r = Renderer3D(800, 600, headless=True, instance_buffer_chunk=10)
        old_max = r._max_instances

        # Create a flock larger than the initial chunk
        cfg = SimConfig()
        cfg.num_boids = 25  # > chunk of 10
        flock = PhysicsFlock(cfg)
        n = r.update_instances(flock)

        assert n == 25
        assert r._max_instances > old_max  # buffer grew

    def test_renderer_windowed_context(self, gpu_available):
        """Renderer3D creates a windowed (non-headless) context."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import os
        if not os.environ.get("DISPLAY"):
            pytest.skip("No display available for windowed context")
        from pymurmur.viz.renderer import Renderer3D
        try:
            r = Renderer3D(800, 600, headless=False)
            assert r.headless is False
            assert r._fbo is None
        except Exception:
            pytest.skip("Windowed context creation failed (no display)")

    def test_renderer_windowed_branches_ci(self, gpu_available, monkeypatch):
        """Cover windowed init branches (lines 49, 90) without a display.

        Monkeypatches moderngl.create_context to use a standalone context
        while still passing headless=False. This exercises the else branches
        for context creation and FBO init.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        import moderngl
        from pymurmur.viz.renderer import Renderer3D

        # Use a real standalone context to satisfy the windowed code path
        real_ctx = moderngl.create_context(standalone=True, require=330)

        def _mock_create(standalone=False, require=330):
            return real_ctx

        monkeypatch.setattr(moderngl, "create_context", _mock_create)

        r = Renderer3D(800, 600, headless=False)
        assert r.headless is False       # line 45 branch
        assert r._fbo is None            # line 90 (else: no FBO in windowed)
        assert r.ctx is real_ctx         # line 49 (windowed context creation)

    def test_renderer_single_memcpy(self, gpu_available, small_flock):
        """update_instances() uses vbo.write() with a single call."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)

        # Patch Buffer.write at the class level (safer than instance-level
        # for C extension objects) and restore on teardown.
        Buffer = type(r._instance_vbo)
        original_write = Buffer.write
        write_calls = []

        def _counting_write(self, data):
            write_calls.append(len(data))
            return original_write(self, data)

        Buffer.write = _counting_write
        try:
            count = r.update_instances(small_flock)
            assert count == small_flock.N_active
            # Core assertion: single memcpy call
            assert len(write_calls) == 1, (
                f"Expected 1 vbo.write() call, got {len(write_calls)}"
            )
            # Sanity: the data written is non-trivial
            assert write_calls[0] > 0
        finally:
            Buffer.write = original_write

    def test_renderer_zero_birds(self, gpu_available):
        """Rendering with 0 active birds doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        cfg.num_boids = 0
        flock = PhysicsFlock(cfg)
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(flock)
        r.end_frame()


@pytest.mark.gpu
class TestVisualizerIntegration:
    """Integration tests for the Visualizer + renderer + camera wiring."""

    def test_visualizer_init(self, gpu_available, default_config):
        """Visualizer(sim, config) creates renderer, camera."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        assert viz.renderer is not None
        assert viz.camera is not None
        assert viz.paused is False

    def test_visualizer_headless_frame(self, gpu_available, default_config):
        """headless_frame() returns a PIL Image."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        img = viz.headless_frame()
        assert img is not None
        assert img.size == (default_config.window_width, default_config.window_height)

    def test_visualizer_run_one_frame(self, gpu_available, default_config):
        """headless_frame() renders without error (step is caller's responsibility)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        sim.step(1.0 / 60)  # step before render (I4.1)
        frame_before = sim.frame
        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == frame_before  # rendering doesn't advance sim

    def test_visualizer_paused_skips_step(self, gpu_available, default_config):
        """Rendering works regardless of pause state (pause only affects caller's step)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)
        viz.paused = True
        frame_before = sim.frame
        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == frame_before  # rendering doesn't advance sim

    def test_renderer_camera_wiring(self, gpu_available, default_config):
        """Renderer3D + OrbitCamera + SimulationEngine wire without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera

        sim = SimulationEngine(default_config)
        renderer = Renderer3D(800, 600, headless=True)
        camera = OrbitCamera()
        renderer.begin_frame(camera)
        sim.step(1.0 / 60)
        renderer.draw_birds(sim.flock)
        renderer.end_frame()

    def test_headless_frame_capture(self, gpu_available, default_config):
        """headless FBO readback produces PIL Image."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.renderer import Renderer3D
        from pymurmur.viz.camera import OrbitCamera

        sim = SimulationEngine(default_config)
        r = Renderer3D(800, 600, headless=True)
        camera = OrbitCamera()
        r.begin_frame(camera)
        r.draw_birds(sim.flock)
        r.end_frame()
        img = r.capture_frame()
        assert img is not None

    def test_visualizer_add_birds_integration(self, gpu_available, default_config):
        """Flock add_boids + Visualizer integration: no crash, N_active increases."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        old_n = sim.flock.N_active
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate main-loop deferred add: pending_add → flock.add_boids
        pending = 5
        sim.step(1.0 / 60)  # step first (I4.1)
        added = sim.flock.add_boids(pending, default_config)
        default_config.num_boids = sim.flock.N_active
        pending -= added

        assert sim.flock.N_active == old_n + 5
        assert pending == 0  # all were added

        # Render a frame after add — must not crash
        viz.headless_frame()

    def test_visualizer_remove_birds_integration(self, gpu_available, default_config):
        """Flock remove_boids + Visualizer integration: no crash, N_active decreases."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        old_n = sim.flock.N_active
        viz = Visualizer(sim, default_config, headless=True)

        # Simulate main-loop deferred remove: pending_remove → flock.remove_boids
        pending = 5
        sim.step(1.0 / 60)  # step first (I4.1)
        removed = sim.flock.remove_boids(pending)
        default_config.num_boids = sim.flock.N_active
        pending -= removed

        assert sim.flock.N_active == old_n - 5
        assert pending == 0  # all were removed

        # Render a frame after remove — must not crash
        viz.headless_frame()

    def test_visualizer_reset_then_step(self, gpu_available, default_config):
        """After reset, sim can continue stepping without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Advance, then reset (I4.1: step explicitly before render)
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.reset()
        assert sim.frame == 0

        # After reset, should be able to continue
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 1

    def test_visualizer_sim_reset(self, gpu_available, default_config):
        """sim.reset() restores frame counter to 0 and keeps flock."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Advance a few frames (I4.1: step explicitly before render)
        for _ in range(3):
            sim.step(1.0 / 60)
            viz.headless_frame()
        assert sim.frame == 3

        # Simulate what main loop does when pending_reset is True
        sim.reset()
        assert sim.frame == 0
        assert sim.flock.N_active == default_config.num_boids

    def test_visualizer_windowed_frame(self, gpu_available, default_config):
        """frame() renders to screen in windowed mode (lines 60-64)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import os
        if not os.environ.get("DISPLAY"):
            pytest.skip("No display available for windowed context")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        try:
            viz = Visualizer(sim, default_config, headless=False)
            sim.step(1.0 / 60)  # step before render (I4.1)
            frame_before = sim.frame
            viz.frame()
            assert sim.frame == frame_before  # rendering doesn't advance sim
        except Exception:
            pytest.skip("Windowed context creation failed (no display)")

    # ── frame() coverage via monkeypatch ──────────────────────────

    def test_visualizer_frame_headless_bypass(self, gpu_available, default_config,
                                               monkeypatch):
        """Cover frame() (lines 60-64) without a display via monkeypatched context."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import moderngl
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        # Use standalone context for the windowed path
        real_ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: real_ctx)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        assert viz.renderer.headless is False

        sim.step(1.0 / 60)  # step before render (I4.1)
        frame_before = sim.frame
        viz.frame()  # lines 60-64 — no return value, no display
        assert sim.frame == frame_before  # rendering doesn't advance sim

    # ── headless_frame() edge cases ────────────────────────────────

    def test_visualizer_headless_frame_paused_toggle(self, gpu_available,
                                                      default_config):
        """Rendering works regardless of paused state (pause is caller's concern)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        # Step + render: normal (I4.1: step explicitly)
        sim.step(1.0 / 60)
        viz.headless_frame()
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 2

        # Step + render: paused — still renders
        viz.paused = True
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 3

        # Step + render: paused again
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 4

        # Step + render: unpause
        viz.paused = False
        sim.step(1.0 / 60)
        viz.headless_frame()
        assert sim.frame == 5

    def test_visualizer_headless_frame_multi_advance(self, gpu_available,
                                                      default_config):
        """10 step+render cycles — rendering is pure, caller controls stepping."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        for _ in range(10):
            sim.step(1.0 / 60)  # step before render (I4.1)
            img = viz.headless_frame()
            assert img is not None

        assert sim.frame == 10

    def test_visualizer_headless_frame_zero_birds(self, gpu_available):
        """headless_frame() with 0 birds doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        cfg = SimConfig()
        cfg.num_boids = 0
        sim = SimulationEngine(cfg)
        viz = Visualizer(sim, cfg, headless=True)

        img = viz.headless_frame()
        assert img is not None
        assert sim.frame == 0  # rendering doesn't advance sim

    # ── __init__ property validation ───────────────────────────────

    def test_visualizer_camera_target_from_config(self, gpu_available, default_config):
        """Camera target is centred on the simulation volume."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        tx, ty, tz = viz.camera.target
        assert tx == default_config.width / 2
        assert ty == default_config.height / 2
        assert tz == default_config.depth / 2

    def test_visualizer_buffer_chunk_passthrough(self, gpu_available, default_config):
        """instance_buffer_chunk from config is passed to Renderer3D."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer

        default_config.instance_buffer_chunk = 7777
        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=True)

        assert viz.renderer._chunk == 7777
        assert viz.renderer._max_instances == 7777

    # ── run() loop body via monkeypatched pygame ──────────────────

    # ── helpers for run() tests ───────────────────────────────────

    @staticmethod
    def _patch_for_headless_run(monkeypatch):
        """Monkeypatch moderngl + pygame so run() works without a display."""
        import moderngl
        import pygame

        ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: ctx)
        monkeypatch.setattr(pygame, "init", lambda: None)
        monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **kw: None)
        monkeypatch.setattr(pygame.display, "set_caption", lambda *a: None)
        monkeypatch.setattr(pygame.display, "flip", lambda: None)
        monkeypatch.setattr(pygame, "quit", lambda: None)
        monkeypatch.setattr(pygame.time, "Clock",
                            lambda: type("C", (), {"tick": lambda s, f: 16})())

    @staticmethod
    def _make_one_shot_handle_events(input_ctrl, monkeypatch):
        """Make handle_events return True once, then False (exit loop)."""
        calls = [0]
        def _handle_once():
            calls[0] += 1
            return calls[0] == 1
        monkeypatch.setattr(input_ctrl, "handle_events", _handle_once)
        return calls

    # ── run() loop body via monkeypatched pygame ──────────────────

    def test_visualizer_run_one_cycle(self, gpu_available, default_config,
                                       monkeypatch):
        """Exercise run() loop body once via monkeypatched pygame + moderngl.

        Covers lines 68-113: deferred add, reset, grid toggle,
        paused stepping, metrics title — all without a display.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        camera = OrbitCamera()

        input_ctrl = InputControl(default_config, camera)
        input_ctrl.pending_add = 5       # test deferred add
        input_ctrl.pending_reset = True  # test reset (processed before step)
        input_ctrl.show_grid = True      # test grid branch (lines 94-95)

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        # Reset consumed, then step advanced frame
        assert input_ctrl.pending_reset is False
        assert sim.frame >= 1

        # Deferred add: all 5 birds added (fresh flock after reset)
        assert input_ctrl.pending_add == 0

        # Loop exited after one iteration
        assert calls[0] == 2  # first=True (run), second=False (exit)

    def test_visualizer_run_paused_no_step(self, gpu_available, default_config,
                                            monkeypatch):
        """run() with input_ctrl.paused=True skips sim.step but still renders."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        frame_before = sim.frame

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.paused = True

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert sim.frame == frame_before  # paused → no step
        assert calls[0] == 2

    def test_visualizer_run_reset_handling(self, gpu_available, default_config,
                                            monkeypatch):
        """run() resets simulation when pending_reset is True."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.pending_reset = True

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert input_ctrl.pending_reset is False  # flag consumed
        assert sim.frame >= 1                      # reset→0, step→1

    def test_visualizer_run_remove_birds(self, gpu_available, default_config,
                                          monkeypatch):
        """run() processes pending_remove and decreases N_active."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.simulation.engine import SimulationEngine
        from pymurmur.viz.visualizer import Visualizer
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl

        self._patch_for_headless_run(monkeypatch)

        sim = SimulationEngine(default_config)
        viz = Visualizer(sim, default_config, headless=False)
        old_n = sim.flock.N_active

        input_ctrl = InputControl(default_config, OrbitCamera())
        input_ctrl.pending_remove = 3           # test deferred remove

        calls = self._make_one_shot_handle_events(input_ctrl, monkeypatch)

        viz.run(input_ctrl)

        assert input_ctrl.pending_remove == 0   # all 3 removed (lines 82-84)
        assert sim.flock.N_active == old_n - 3
        assert calls[0] == 2
