"""GPU-dependent tests for viz.renderer — InstanceSchema (P2.7, no GPU needed), _mat4_bytes (P2.8, no GPU needed), Renderer3D core, draw_layer, Renderer3D release.

Requires ModernGL GPU context. All GPU-dependent tests are gated behind
@pytest.mark.gpu and skipped when gpu_available is False.

Split out of test_renderer.py (file-size split).
"""

import numpy as np
import pytest

# ── P2.7: InstanceSchema standalone dataclass tests (NO GPU needed) ─

class TestInstanceSchema:
    """P2.7: InstanceSchema is a pure dataclass — testable without GPU."""

    def test_instance_schema_defaults(self):
        """D7: InstanceSchema has correct default field values — one
        merged 8-float layout (pos.xyz vel.xyz hue scale), not the old
        6-float pos+vel with a separate colour VBO."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema()
        assert s.floats == 8, (
            "Default floats must be 8 (pos.xyz + vel.xyz + hue + scale)"
        )
        assert s.layout == "3f 3f 1f 1f/i", "Default layout must be ModernGL format string"
        assert s.attrs == (
            "in_bird_pos", "in_bird_vel", "in_bird_hue", "in_bird_scale",
        ), "Default attrs must be shader attribute names"

    def test_instance_schema_pos_vel_only_view(self):
        """D7: the pos+vel-only padded view skips the trailing hue+scale
        floats (8 bytes = 2×float32) for shaders that don't declare
        them, e.g. the impostor VAO."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema()
        assert s.pos_vel_layout == "3f 3f 8x/i"
        assert s.pos_vel_attrs == ("in_bird_pos", "in_bird_vel")

    def test_instance_schema_custom_floats(self):
        """P2.7: InstanceSchema accepts custom float count."""
        from pymurmur.viz.renderer import InstanceSchema
        s = InstanceSchema(floats=9)
        assert s.floats == 9
        assert s.layout == "3f 3f 1f 1f/i"  # layout unchanged unless explicitly set

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
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        cam = OrbitCamera()
        r.begin_frame(cam)
        # No error = pass

    def test_renderer_draw_birds_no_error(self, gpu_available, small_flock):
        """draw_birds(flock) completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(small_flock)
        r.end_frame()

    def test_renderer_draw_grid_no_error(self, gpu_available):
        """draw_grid() completes without GL error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_grid()
        r.end_frame()

    def test_renderer_capture_frame(self, gpu_available):
        """capture_frame() returns a PIL Image with correct dimensions."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
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
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

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
        """D7: update_instances() writes the instance VBO exactly once
        per frame — pos+vel+hue+scale all interleave into one merged
        InstanceSchema buffer now (was 2 writes: instance + a separate
        colour VBO for hue+scale, before the D7 schema merge)."""
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
            assert len(write_calls) == 1, (
                f"Expected 1 vbo.write() call (single merged instance "
                f"buffer), got {len(write_calls)}"
            )
            assert write_calls[0] > 0
        finally:
            Buffer.write = original_write

    def test_renderer_zero_birds(self, gpu_available):
        """Rendering with 0 active birds doesn't crash."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        cfg = SimConfig()
        cfg.num_boids = 0
        flock = PhysicsFlock(cfg)
        r = Renderer3D(800, 600, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_birds(flock)
        r.end_frame()


@pytest.mark.gpu
class TestDrawLayer:
    """D7: draw_layer — single non-instanced marker seam, feeds S2.A8
    (threat marker) and S2.E5 (influencer target marker)."""

    def test_draw_layer_default_mesh_no_crash(self, gpu_available):
        """Default call (ellipsoid mesh) renders without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(200, 150, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_layer((100.0, 50.0, 25.0))
        r.end_frame()

    def test_draw_layer_each_registered_mesh(self, gpu_available):
        """Every S4.4a mesh usable as a marker renders without error."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(200, 150, headless=True)
        r.begin_frame(OrbitCamera())
        for mesh in ("ellipsoid", "cone", "arrow"):
            r.draw_layer((0.0, 0.0, 0.0), mesh=mesh)
        r.end_frame()

    def test_draw_layer_unknown_mesh_falls_back(self, gpu_available):
        """An unrecognised mesh name falls back to ellipsoid instead of
        raising (e.g. "tetra"/"winged"/"impostor" use a different
        shader program and aren't valid draw_layer meshes)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(200, 150, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_layer((0.0, 0.0, 0.0), mesh="tetra")  # not in _mesh_vbos
        r.end_frame()

    def test_draw_layer_does_not_touch_bird_instance_data(self, gpu_available, small_flock):
        """D7 regression guard: draw_layer must not corrupt the shared
        per-bird instance buffer (it uses a separate, non-instanced VAO
        — this is exactly the bug the docstring warns about avoiding)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(200, 150, headless=True)
        r.update_instances(small_flock)
        packed_before = r._packed[: small_flock.N_active].copy()

        r.begin_frame(OrbitCamera())
        r.draw_layer((999.0, -999.0, 500.0), hue=0.9, scale=5.0)
        r.draw_birds(small_flock)
        r.end_frame()

        packed_after = r._packed[: small_flock.N_active]
        assert np.array_equal(packed_before, packed_after), (
            "draw_layer must not mutate the shared instance buffer"
        )

    def test_draw_layer_caches_marker_vao_per_mesh(self, gpu_available):
        """Repeated calls for the same mesh reuse one cached VAO rather
        than rebuilding it every call."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(200, 150, headless=True)
        r.begin_frame(OrbitCamera())
        r.draw_layer((0.0, 0.0, 0.0), mesh="cone")
        vao_first = r._marker_vao_cone
        r.draw_layer((1.0, 2.0, 3.0), mesh="cone")
        r.end_frame()
        assert r._marker_vao_cone is vao_first


@pytest.mark.gpu
class TestRenderer3DRelease:
    """Regression guard for a real, severe bug: without `release()` +
    `__del__`, each `Renderer3D` leaks its entire GL context for the
    process lifetime. Running the full `-m "gl or gpu"` suite under
    software Mesa llvmpipe (Docker) reproduced this directly — RSS grew
    from ~240 MB to ~1.7 GB within seconds and the process was
    OOM-killed partway through, every time, until fixed."""

    def test_release_does_not_raise(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        r.release()  # must not raise

    def test_release_is_idempotent(self, gpu_available):
        """Calling release() twice must not raise (moderngl.Context's
        own release() guards against double-release; Renderer3D relies
        on that, both explicitly here and implicitly via __del__ firing
        after an explicit release() elsewhere)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.viz.renderer import Renderer3D
        r = Renderer3D(800, 600, headless=True)
        r.release()
        r.release()  # must not raise

    def test_del_releases_context_without_manual_call(self, gpu_available):
        """__del__ releases the context even when release() was never
        called explicitly — this is the actual leak fix: many tests
        across the suite construct a Renderer3D and let it go out of
        scope without ever calling release() themselves."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import gc

        from pymurmur.viz.renderer import Renderer3D

        r = Renderer3D(800, 600, headless=True)
        ctx = r.ctx
        del r
        gc.collect()
        # Released contexts replace their internal mglo with an
        # InvalidObject sentinel (see moderngl.Context.release()).
        assert type(ctx.mglo).__name__ == "InvalidObject"

    def test_many_renderers_do_not_accumulate_live_contexts(self, gpu_available):
        """Creating and dropping many Renderer3D instances in a row
        does not exhaust the driver's concurrent-context limit — the
        exact failure mode observed before this fix (`_moderngl.Error:
        cannot create vertex array/buffer` after ~90 contexts under
        llvmpipe, fewer on some other drivers)."""
        if not gpu_available:
            pytest.skip("GPU not available")
        import gc

        from pymurmur.viz.renderer import Renderer3D

        for _ in range(30):
            r = Renderer3D(64, 64, headless=True)
            del r
        gc.collect()
        # If contexts were accumulating, this final allocation would be
        # the one to fail with "cannot create vertex array/buffer".
        r = Renderer3D(64, 64, headless=True)
        r.release()


