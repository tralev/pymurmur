"""Tests for viz.shaders — GLSL sources and tetrahedron mesh.

No ModernGL context required — these tests validate shader source
strings and mesh vertex data, not GPU compilation.
"""

from pathlib import Path


class TestShaders:
    """Shader sources are valid strings, mesh data has correct shape."""

    def test_vertex_shader_is_nonempty_string(self):
        """Vertex shader source is a non-empty string."""
        from pymurmur.viz.shaders import VERTEX_SHADER
        assert isinstance(VERTEX_SHADER, str)
        assert len(VERTEX_SHADER) > 0

    def test_fragment_shader_is_nonempty_string(self):
        """Fragment shader source is a non-empty string."""
        from pymurmur.viz.shaders import FRAGMENT_SHADER
        assert isinstance(FRAGMENT_SHADER, str)
        assert len(FRAGMENT_SHADER) > 0

    def test_tetrahedron_vertices_shape(self):
        """Tetrahedron mesh has 4 vertices."""
        from pymurmur.viz.shaders import TETRA_VERTICES
        assert len(TETRA_VERTICES) == 4
        for v in TETRA_VERTICES:
            assert len(v) == 3  # (x, y, z)

    def test_tetrahedron_faces_shape(self):
        """Tetrahedron mesh has 4 faces."""
        from pymurmur.viz.shaders import TETRA_INDICES
        assert len(TETRA_INDICES) == 4
        for f in TETRA_INDICES:
            assert len(f) == 3  # 3 vertex indices

    def test_shaders_no_moderngl_import(self):
        """shaders.py does not import moderngl."""
        path = Path("pymurmur/viz/shaders.py")
        text = path.read_text()
        assert "import moderngl" not in text
        assert "from moderngl" not in text

    # ── Theme palette tests ───────────────────────────────────────

    def test_themes_has_five_palettes(self):
        """THEMES dict contains exactly 5 named palettes (S4.6 added "heading")."""
        from pymurmur.viz.shaders import THEMES
        assert set(THEMES.keys()) == {
            "ink", "inverse", "paper", "graphite", "heading",
        }

    def test_theme_keys_all_present(self):
        """Each theme has slow, fast, spec, clear, trail colour keys."""
        from pymurmur.viz.shaders import THEMES
        required = {"slow", "fast", "spec", "clear", "trail"}
        for name, theme in THEMES.items():
            missing = required - set(theme.keys())
            assert not missing, f"{name}: missing keys {missing}"

    def test_theme_rgb_values_in_range(self):
        """All theme colour channels are floats in [0, 1]."""
        from pymurmur.viz.shaders import THEMES
        for name, theme in THEMES.items():
            for key, colour in theme.items():
                assert len(colour) == 3, f"{name}.{key}: expected 3 channels"
                for c in colour:
                    assert 0.0 <= c <= 1.0, f"{name}.{key}: channel {c} out of [0,1]"

    def test_theme_fallback_to_ink(self):
        """Invalid theme name falls back to 'ink' via THEMES.get()."""
        from pymurmur.viz.shaders import THEMES
        result = THEMES.get("nonexistent_theme", THEMES["ink"])
        assert result is THEMES["ink"]

    def test_themes_consistent_across_accesses(self):
        """Repeated THEMES access returns stable values (not mutated)."""
        from pymurmur.viz.shaders import THEMES
        t1 = THEMES["ink"]
        t2 = THEMES["ink"]
        assert t1["slow"] == t2["slow"]
        assert t1["fast"] == t2["fast"]
        assert t1["spec"] == t2["spec"]

    # ── Trail shader tests ────────────────────────────────────────

    def test_trail_vertex_shader_is_nonempty_string(self):
        """Trail vertex shader source is a non-empty string."""
        from pymurmur.viz.shaders import TRAIL_VERTEX_SHADER
        assert isinstance(TRAIL_VERTEX_SHADER, str)
        assert len(TRAIL_VERTEX_SHADER) > 0

    def test_trail_fragment_shader_is_nonempty_string(self):
        """Trail fragment shader source is a non-empty string."""
        from pymurmur.viz.shaders import TRAIL_FRAGMENT_SHADER
        assert isinstance(TRAIL_FRAGMENT_SHADER, str)
        assert len(TRAIL_FRAGMENT_SHADER) > 0

    def test_trail_vertex_shader_is_glsl(self):
        """Trail vertex shader contains #version 330 core and main()."""
        from pymurmur.viz.shaders import TRAIL_VERTEX_SHADER
        assert "#version 330" in TRAIL_VERTEX_SHADER
        assert "void main" in TRAIL_VERTEX_SHADER

    def test_trail_fragment_shader_is_glsl(self):
        """Trail fragment shader contains #version 330 core and main()."""
        from pymurmur.viz.shaders import TRAIL_FRAGMENT_SHADER
        assert "#version 330" in TRAIL_FRAGMENT_SHADER
        assert "void main" in TRAIL_FRAGMENT_SHADER

    def test_trail_shaders_have_required_uniforms(self):
        """Trail shaders declare expected uniforms."""
        from pymurmur.viz.shaders import TRAIL_FRAGMENT_SHADER, TRAIL_VERTEX_SHADER
        assert "u_view" in TRAIL_VERTEX_SHADER
        assert "u_projection" in TRAIL_VERTEX_SHADER
        assert "u_trail_length" in TRAIL_VERTEX_SHADER
        assert "u_trail_color" in TRAIL_FRAGMENT_SHADER

    # ── Grid data tests ───────────────────────────────────────────

    def test_grid_vertices_is_valid(self):
        """GRID_VERTICES is a non-empty float32 array with shape (N, 3)."""
        from pymurmur.viz.shaders import GRID_VERTICES
        assert GRID_VERTICES is not None
        assert GRID_VERTICES.ndim == 2
        assert GRID_VERTICES.shape[1] == 3
        assert GRID_VERTICES.shape[0] > 0
        assert GRID_VERTICES.dtype.name == "float32"
