"""P8.4: Winged flapping mesh + gradient sky tests.

Tests winged mesh geometry, flap weight attributes, shader compilation,
gradient sky uniforms, and config field wiring.  GPU-dependent tests
are gated behind ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── GPU availability check ───────────────────────────────────────
try:
    import moderngl  # noqa: F401
    gpu_available = True
except (ImportError, OSError):
    gpu_available = False


# ── P8.4a: Winged mesh geometry ──────────────────────────────────

class TestWingedMeshGeometry:
    """P8.4: Winged mesh is a 7-vertex, 6-triangle bird shape."""

    def test_winged_vertices_shape(self):
        """P8.4: WINGED_VERTICES is (7, 4) — xyz + flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        assert WINGED_VERTICES.shape == (7, 4)
        assert WINGED_VERTICES.dtype == np.float32

    def test_winged_indices_shape(self):
        """P8.4: WINGED_INDICES is (6, 3) — 6 triangles."""
        from pymurmur.viz.shaders import WINGED_INDICES
        assert WINGED_INDICES.shape == (6, 3)
        assert WINGED_INDICES.dtype == np.uint32

    def test_winged_nose_at_front(self):
        """P8.4: Vertex 0 (nose tip) points +Z."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        nose = WINGED_VERTICES[0]
        assert nose[2] > 0.5, "Nose should be at front (+Z)"

    def test_winged_tail_at_back(self):
        """P8.4: Tail vertices (5, 6) are behind the body centre (Z < -0.5)."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        tail_top = WINGED_VERTICES[5]
        tail_bottom = WINGED_VERTICES[6]
        assert tail_top[2] < -0.5, "Tail upper should be behind centre"
        assert tail_bottom[2] < -0.5, "Tail lower should be behind centre"

    def test_wing_tips_have_flap_weight(self):
        """P8.4: Wing tips (vertices 3, 4) have non-zero flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        right_wing = WINGED_VERTICES[3]
        left_wing = WINGED_VERTICES[4]
        assert right_wing[3] != 0.0, "Right wing tip must have flap_weight"
        assert left_wing[3] != 0.0, "Left wing tip must have flap_weight"
        # Opposite signs for anti-symmetric flap
        assert np.sign(right_wing[3]) != np.sign(left_wing[3]), (
            "Wing flap weights must have opposite signs"
        )

    def test_body_vertices_no_flap(self):
        """P8.4: Body vertices (0, 1, 2, 5, 6) have zero flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        for i in (0, 1, 2, 5, 6):
            assert WINGED_VERTICES[i, 3] == 0.0, f"Vertex {i} should have 0 flap_weight"

    def test_winged_triangle_count(self):
        """P8.4: Exactly 6 triangles — 2 body + 2 wing + 2 tail."""
        from pymurmur.viz.shaders import WINGED_INDICES
        assert len(WINGED_INDICES) == 6

    def test_all_indices_in_range(self):
        """P8.4: All indices reference valid vertices (0–6)."""
        from pymurmur.viz.shaders import WINGED_INDICES
        flat = WINGED_INDICES.flatten()
        assert flat.min() >= 0
        assert flat.max() <= 6

    def test_winged_mesh_is_larger_than_tetra(self):
        """P8.4: Winged mesh has more vertices and faces than tetrahedron."""
        from pymurmur.viz.shaders import (
            TETRA_INDICES,
            TETRA_VERTICES,
            WINGED_INDICES,
            WINGED_VERTICES,
        )
        assert len(WINGED_VERTICES) > len(TETRA_VERTICES)
        assert len(WINGED_INDICES) > len(TETRA_INDICES)


# ── P8.4b: Gradient sky mesh ─────────────────────────────────────

class TestSkyQuad:
    """P8.4: Sky is a fullscreen quad covering [-1, 1]² in clip space."""

    def test_sky_quad_shape(self):
        """P8.4: SKY_QUAD has 4 vertices with 2D positions."""
        from pymurmur.viz.shaders import SKY_QUAD
        assert SKY_QUAD.shape == (4, 2)

    def test_sky_quad_indices(self):
        """P8.4: SKY_QUAD_INDICES has 2 triangles."""
        from pymurmur.viz.shaders import SKY_QUAD_INDICES
        assert SKY_QUAD_INDICES.shape == (2, 3)

    def test_sky_quad_covers_fullscreen(self):
        """P8.4: Sky quad corners are at clip-space extents."""
        from pymurmur.viz.shaders import SKY_QUAD
        assert SKY_QUAD.min() == -1.0
        assert SKY_QUAD.max() == 1.0


# ── P8.4c: Shader source existence ───────────────────────────────

class TestWingedShaders:
    """P8.4: Winged and sky shader source strings exist and compile."""

    def test_winged_vertex_shader_exists(self):
        """P8.4: WINGED_VERTEX_SHADER is a non-empty string."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert isinstance(WINGED_VERTEX_SHADER, str)
        assert len(WINGED_VERTEX_SHADER) > 100

    def test_winged_shader_has_flap_attribute(self):
        """P8.4: WINGED_VERTEX_SHADER references in_flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert "in_flap_weight" in WINGED_VERTEX_SHADER

    def test_winged_shader_uses_u_frame(self):
        """P8.4: WINGED_VERTEX_SHADER has a u_frame uniform for animation."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert "u_frame" in WINGED_VERTEX_SHADER

    def test_winged_shader_has_flap_math(self):
        """P8.4: Flap computation uses sin() and in_flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert "sin(" in WINGED_VERTEX_SHADER
        assert "flap" in WINGED_VERTEX_SHADER

    def test_sky_vertex_shader_exists(self):
        """P8.4: SKY_VERTEX_SHADER is non-empty."""
        from pymurmur.viz.shaders import SKY_VERTEX_SHADER
        assert isinstance(SKY_VERTEX_SHADER, str)
        assert len(SKY_VERTEX_SHADER) > 50

    def test_sky_fragment_shader_exists(self):
        """P8.4: SKY_FRAGMENT_SHADER is non-empty."""
        from pymurmur.viz.shaders import SKY_FRAGMENT_SHADER
        assert isinstance(SKY_FRAGMENT_SHADER, str)
        assert len(SKY_FRAGMENT_SHADER) > 50

    def test_sky_shader_uses_gradient(self):
        """P8.4: Sky fragment shader references top/bottom colours."""
        from pymurmur.viz.shaders import SKY_FRAGMENT_SHADER
        assert "u_sky_top" in SKY_FRAGMENT_SHADER
        assert "u_sky_bottom" in SKY_FRAGMENT_SHADER
        assert "mix(" in SKY_FRAGMENT_SHADER


# ── P8.4d: Config field wiring ───────────────────────────────────

class TestWingedConfig:
    """P8.4: winged_mesh and gradient_sky config fields."""

    def test_viz_config_has_winged_mesh(self):
        """P8.4: VizConfig.winged_mesh exists and defaults to True."""
        from pymurmur.core.config import VizConfig
        cfg = VizConfig()
        assert hasattr(cfg, "winged_mesh")
        assert cfg.winged_mesh is True

    def test_viz_config_has_gradient_sky(self):
        """P8.4: VizConfig.gradient_sky exists and defaults to True."""
        from pymurmur.core.config import VizConfig
        cfg = VizConfig()
        assert hasattr(cfg, "gradient_sky")
        assert cfg.gradient_sky is True

    def test_simconfig_flat_access_winged(self):
        """P8.4: SimConfig exposes winged_mesh via flat access."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.winged_mesh is True
        cfg.winged_mesh = False
        assert cfg.winged_mesh is False
        assert cfg.viz.winged_mesh is False

    def test_simconfig_flat_access_gradient_sky(self):
        """P8.4: SimConfig exposes gradient_sky via flat access."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        assert cfg.gradient_sky is True
        cfg.gradient_sky = False
        assert cfg.gradient_sky is False
        assert cfg.viz.gradient_sky is False

    def test_field_map_has_winged_entries(self):
        """P8.4: _FIELD_MAP has winged_mesh and gradient_sky."""
        from pymurmur.core.config import _FIELD_MAP
        assert "winged_mesh" in _FIELD_MAP
        assert "gradient_sky" in _FIELD_MAP
        assert _FIELD_MAP["winged_mesh"] == ("_viz", "winged_mesh")
        assert _FIELD_MAP["gradient_sky"] == ("_viz", "gradient_sky")


# ── P8.4e: Renderer integration (GPU-dependent) ──────────────────

@pytest.mark.gpu
class TestWingedRenderer:
    """P8.4: Renderer3D creates winged VAO and sky VAO."""

    @pytest.fixture
    def headless_renderer(self):
        """Headless Renderer3D with winged mesh and gradient sky enabled."""
        if not gpu_available:
            pytest.skip("ModernGL GPU context not available")
        from pymurmur.viz.renderer import Renderer3D
        return Renderer3D(
            width=200, height=150, headless=True,
            winged_mesh=True, gradient_sky=True,
        )

    def test_winged_prog_compiles(self, headless_renderer):
        """P8.4: _winged_prog is a valid ModernGL program."""
        assert headless_renderer._winged_prog is not None

    def test_winged_vao_built(self, headless_renderer):
        """P8.4: _winged_vao exists after construction."""
        assert headless_renderer._winged_vao is not None

    def test_sky_prog_compiles(self, headless_renderer):
        """P8.4: _sky_prog is a valid ModernGL program."""
        assert headless_renderer._sky_prog is not None

    def test_sky_vao_built(self, headless_renderer):
        """P8.4: _sky_vao exists after construction."""
        assert headless_renderer._sky_vao is not None

    def test_frame_count_initialised(self, headless_renderer):
        """P8.4: _frame_count starts at 0."""
        assert headless_renderer._frame_count == 0

    def test_frame_count_advances(self, headless_renderer):
        """P8.4: begin_frame increments _frame_count."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        assert headless_renderer._frame_count == 0
        headless_renderer.begin_frame(cam)
        assert headless_renderer._frame_count == 1
        headless_renderer.begin_frame(cam)
        assert headless_renderer._frame_count == 2

    def test_winged_mesh_disabled_uses_tetra(self, headless_renderer):
        """P8.4: when winged_mesh=False, draw_birds uses tetra VAO (no crash)."""
        headless_renderer._winged_mesh = False
        headless_renderer._point_sprites = False
        # This should render using _vao without error
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        cfg = SimConfig(num_boids=10)
        flock = PhysicsFlock(cfg)
        cam = OrbitCamera()
        headless_renderer.begin_frame(cam)
        headless_renderer.draw_birds(flock)

    def test_winged_mesh_enabled_uses_winged_vao(self, headless_renderer):
        """P8.4: when winged_mesh=True, draw_birds uses winged VAO (no crash)."""
        headless_renderer._winged_mesh = True
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        cfg = SimConfig(num_boids=10)
        flock = PhysicsFlock(cfg)
        cam = OrbitCamera()
        headless_renderer.begin_frame(cam)
        headless_renderer.draw_birds(flock)

    def test_draw_sky_no_crash(self, headless_renderer):
        """P8.4: _draw_sky renders without error."""
        headless_renderer._draw_sky()

    def test_begin_frame_with_gradient_sky(self, headless_renderer):
        """P8.4: begin_frame calls _draw_sky when gradient_sky=True."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        # Should not crash
        headless_renderer.begin_frame(cam)

    def test_begin_frame_without_gradient_sky(self, headless_renderer):
        """P8.4: begin_frame skips sky when gradient_sky=False."""
        headless_renderer._gradient_sky = False
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        # Should not crash (no sky draw)
        headless_renderer.begin_frame(cam)

    def test_winged_vao_rebuilt_on_buffer_growth(self, headless_renderer):
        """P8.4: _winged_vao is rebuilt when instance buffer grows."""
        old_vao = id(headless_renderer._winged_vao)
        # Force buffer growth by calling update_instances with max+1
        headless_renderer._max_instances = 5
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig(num_boids=10)
        flock = PhysicsFlock(cfg)
        headless_renderer.update_instances(flock)
        new_vao = id(headless_renderer._winged_vao)
        assert new_vao != old_vao, "Winged VAO must be rebuilt on buffer growth"


# ── P8.4f: Flap animation properties ─────────────────────────────

class TestFlapAnimation:
    """P8.4: Flap oscillation properties (non-GPU)."""

    def test_flap_weight_signs_oppose_for_wings(self):
        """P8.4: Right wing flaps up when left flaps down (anti-symmetric)."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        right = WINGED_VERTICES[3, 3]
        left = WINGED_VERTICES[4, 3]
        # One is positive, one negative
        assert right * left < 0

    def test_flap_period_is_configurable(self):
        """C3: sin(u_frame/u_flap_period_frames * 2π) — period from config.flap_period * fps."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert "u_flap_period_frames" in WINGED_VERTEX_SHADER
        assert "/ u_flap_period_frames" in WINGED_VERTEX_SHADER
        assert "6.283185" in WINGED_VERTEX_SHADER  # 2π

    def test_flap_applied_before_rotation(self):
        """P8.4: Flap modifies pos.y before LookAt rotation matrix."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        # Flap calculation should appear before the rotation matrix construction
        flap_idx = WINGED_VERTEX_SHADER.index("flap")
        rotation_idx = WINGED_VERTEX_SHADER.index("rotation")
        assert flap_idx < rotation_idx, (
            "Flap must be applied before LookAt rotation matrix"
        )

    def test_flap_amplitude_is_flap_weight(self):
        """P8.4: Wings oscillate with amplitude proportional to flap_weight."""
        from pymurmur.viz.shaders import WINGED_VERTICES
        # Wing tips have flap_weight = ±0.5
        assert abs(WINGED_VERTICES[3, 3]) == pytest.approx(0.5)
        assert abs(WINGED_VERTICES[4, 3]) == pytest.approx(0.5)


# ── P8.4g: Winged VAO attribute layout ───────────────────────────

@pytest.mark.gpu
class TestWingedVAOAttributes:
    """P8.4: Winged VAO has correct attribute bindings for flap_weight."""

    @pytest.fixture
    def headless_renderer(self):
        """Headless Renderer3D with winged mesh enabled."""
        if not gpu_available:
            pytest.skip("ModernGL GPU context not available")
        from pymurmur.viz.renderer import Renderer3D
        return Renderer3D(
            width=200, height=150, headless=True,
            winged_mesh=True, gradient_sky=True,
        )

    def test_winged_vao_has_in_flap_weight(self, headless_renderer):
        """P8.4: Winged VAO program contains in_flap_weight attribute."""
        prog = headless_renderer._winged_prog
        assert "in_flap_weight" in prog, (
            "Winged shader program must declare in_flap_weight"
        )

    def test_winged_vao_uses_winged_program(self, headless_renderer):
        """P8.4: Winged VAO uses the winged shader program, not the tetra one."""
        vao = headless_renderer._winged_vao
        # The VAO's program should be the winged program
        assert vao.program is headless_renderer._winged_prog, (
            "Winged VAO must use the winged shader program"
        )

    def test_winged_mesh_buffer_has_4_floats(self, headless_renderer):
        """P8.4: Mesh VBO has 4 floats/vertex (xyz + flap_weight)."""
        vbo = headless_renderer._winged_mesh_vbo
        # 7 vertices × 4 floats × 4 bytes = 112 bytes
        assert vbo.size == 7 * 4 * 4, (
            f"Winged mesh VBO should be 112 bytes, got {vbo.size}"
        )

    def test_winged_prog_not_same_as_tetra_prog(self, headless_renderer):
        """P8.4: Winged program is distinct from the tetrahedron program."""
        assert headless_renderer._winged_prog is not headless_renderer._prog, (
            "Winged and tetrahedron programs must be separate"
        )
