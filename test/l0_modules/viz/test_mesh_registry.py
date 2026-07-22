"""S4.4a — Mesh registry tests (mesh data, render-mode recommendation, theme materials)."""

import numpy as np
import pytest

from pymurmur.viz.mesh_registry import (
    MATERIAL_REGISTRY,
    MESH_REGISTRY,
    VALID_MESH_NAMES,
    VALID_THEMES,
    get_mesh,
    get_theme_materials,
    recommend_render_mode,
    resolve_bird_mesh,
)

# ── Mesh data ────────────────────────────────────────────────────

class TestMeshData:
    """Each mesh entry has valid vertex and index data."""

    @pytest.mark.parametrize("name", sorted(VALID_MESH_NAMES))
    def test_mesh_has_vertices_and_indices(self, name):
        entry = MESH_REGISTRY[name]
        assert "vertices" in entry
        assert "indices" in entry
        assert isinstance(entry["vertices"], np.ndarray)
        assert isinstance(entry["indices"], np.ndarray)
        assert entry["vertices"].ndim == 2
        assert entry["indices"].ndim == 2

    @pytest.mark.parametrize("name", sorted(VALID_MESH_NAMES - {"points"}))
    def test_indices_in_range(self, name):
        entry = MESH_REGISTRY[name]
        n_verts = len(entry["vertices"])
        if entry["indices"].size > 0:
            assert entry["indices"].min() >= 0
            assert entry["indices"].max() < n_verts

    @pytest.mark.parametrize("name", sorted(VALID_MESH_NAMES))
    def test_vertex_format_and_attributes(self, name):
        entry = MESH_REGISTRY[name]
        assert "vertex_format" in entry
        assert "attributes" in entry
        assert isinstance(entry["vertex_format"], str)
        assert isinstance(entry["attributes"], tuple)

    def test_ellipsoid_is_elongated_along_z(self):
        """Ellipsoid vertices have larger Z span than XY span."""
        verts = MESH_REGISTRY["ellipsoid"]["vertices"]
        z_span = verts[:, 2].max() - verts[:, 2].min()
        xy_span = max(verts[:, 0].max() - verts[:, 0].min(),
                       verts[:, 1].max() - verts[:, 1].min())
        assert z_span > xy_span * 1.5, (
            f"Ellipsoid should be elongated along Z: z_span={z_span:.1f}, "
            f"xy_span={xy_span:.1f}"
        )

    def test_cone_tip_forward(self):
        """Cone tip is at +Z with narrow base at -Z."""
        verts = MESH_REGISTRY["cone"]["vertices"]
        tip = verts[0]  # first vertex should be the tip
        assert tip[2] > 0.5, f"Cone tip Z should be positive, got {tip[2]}"
        # Base vertices should be at negative Z
        base_z = verts[1:, 2]
        assert np.all(base_z < 0), f"Cone base should be at negative Z, got {base_z}"

    def test_arrow_tip_forward(self):
        """Arrow tip points forward (+Z), shaft extends back."""
        verts = MESH_REGISTRY["arrow"]["vertices"]
        tip = verts[0]
        assert tip[2] > 1.0, f"Arrow tip should be far forward, got {tip[2]}"
        # Some vertex should be far behind
        assert verts[:, 2].min() < -1.0, (
            f"Arrow shaft should extend backward, min Z={verts[:, 2].min()}"
        )

    def test_points_has_single_vertex(self):
        entries = MESH_REGISTRY["points"]
        assert len(entries["vertices"]) == 1
        assert entries["primitive"] == 0  # GL_POINTS

    def test_winged_has_flap_weight_attribute(self):
        attrs = MESH_REGISTRY["winged"]["attributes"]
        assert "in_flap_weight" in attrs

    def test_impostor_is_2d_quad(self):
        entry = MESH_REGISTRY["impostor"]
        assert entry["vertex_format"] == "2f"
        assert entry["attributes"] == ("in_quad_pos",)

    def test_get_mesh_returns_correct_entry(self):
        for name in ("tetra", "winged", "ellipsoid"):
            assert get_mesh(name) is MESH_REGISTRY[name]

    def test_get_mesh_unknown_falls_back_to_tetra(self):
        assert get_mesh("nonexistent") is MESH_REGISTRY["tetra"]


# ── Render-mode recommendation ───────────────────────────────────

class TestRecommendRenderMode:
    def test_below_instanced_threshold_returns_winged(self):
        assert recommend_render_mode(0) == "winged"
        assert recommend_render_mode(1) == "winged"
        assert recommend_render_mode(5_000) == "winged"
        assert recommend_render_mode(10_000) == "winged"

    def test_between_thresholds_returns_impostor(self):
        assert recommend_render_mode(10_001) == "impostor"
        assert recommend_render_mode(30_000) == "impostor"
        assert recommend_render_mode(60_000) == "impostor"

    def test_above_impostor_threshold_returns_points(self):
        assert recommend_render_mode(60_001) == "points"
        assert recommend_render_mode(100_000) == "points"
        assert recommend_render_mode(1_000_000) == "points"

    def test_custom_thresholds(self):
        assert recommend_render_mode(
            5_000, instanced_threshold=2_000, impostor_threshold=10_000,
        ) == "impostor"
        assert recommend_render_mode(
            2_000, instanced_threshold=2_000, impostor_threshold=10_000,
        ) == "winged"
        assert recommend_render_mode(
            15_000, instanced_threshold=2_000, impostor_threshold=10_000,
        ) == "points"


# ── Theme materials ──────────────────────────────────────────────

class TestThemeMaterials:
    def test_all_valid_themes_are_ink_inverse_paper_graphite_heading(self):
        assert VALID_THEMES == frozenset(
            {"ink", "inverse", "paper", "graphite", "heading"}
        )

    @pytest.mark.parametrize(
        "theme", ["ink", "inverse", "paper", "graphite", "heading"]
    )
    def test_theme_has_required_keys(self, theme):
        mat = MATERIAL_REGISTRY[theme]
        required = {"ambient", "diffuse", "spec", "slow", "fast",
                     "clear", "trail", "paper", "ink"}
        for key in required:
            assert key in mat, f"Theme {theme!r} missing key {key!r}"
            val = mat[key]
            assert len(val) == 3
            assert all(0.0 <= c <= 1.0 for c in val), (
                f"Theme {theme!r}[{key!r}] = {val} — values must be in [0, 1]"
            )

    def test_get_theme_materials_returns_correct_theme(self):
        mats = get_theme_materials("graphite")
        assert mats is MATERIAL_REGISTRY["graphite"]

    def test_get_theme_materials_unknown_falls_back_to_ink(self):
        mats = get_theme_materials("neon")
        assert mats is MATERIAL_REGISTRY["ink"]

    def test_ambient_dimmer_than_diffuse_across_all_themes(self):
        """Ambient should always be dimmer or equal to diffuse (lighting model)."""
        for theme in MATERIAL_REGISTRY:
            mat = MATERIAL_REGISTRY[theme]
            ambient_sum = sum(mat["ambient"])
            diffuse_sum = sum(mat["diffuse"])
            assert ambient_sum <= diffuse_sum, (
                f"Theme {theme}: ambient={mat['ambient']} should be dimmer "
                f"than diffuse={mat['diffuse']}"
            )


# ── Resolve bird mesh ────────────────────────────────────────────

class TestResolveBirdMesh:
    def test_auto_delegates_to_recommend(self):
        assert resolve_bird_mesh("auto", 100) == "winged"
        assert resolve_bird_mesh("auto", 30_000) == "impostor"
        assert resolve_bird_mesh("auto", 100_000) == "points"

    def test_sphere_maps_to_impostor(self):
        assert resolve_bird_mesh("sphere", 50) == "impostor"
        assert resolve_bird_mesh("sphere", 100_000) == "impostor"

    def test_explicit_mesh_passthrough(self):
        for name in VALID_MESH_NAMES:
            assert resolve_bird_mesh(name, 50) == name

    def test_unknown_falls_back_to_tetra(self):
        assert resolve_bird_mesh("icosahedron", 50) == "tetra"

    def test_ellipsoid_cone_arrow_passthrough(self):
        for name in ("ellipsoid", "cone", "arrow"):
            assert resolve_bird_mesh(name, 500) == name
