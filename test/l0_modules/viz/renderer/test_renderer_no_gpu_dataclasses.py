"""Tests for viz.renderer — InstanceSchema (P2.7) and _mat4_bytes
(P2.8) pure dataclass/helper tests. No GPU context required.

Split out of test_renderer.py (file-size split) — Renderer3D core,
draw_layer, and Renderer3D release tests (all GPU-dependent) stay in
the original.
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

