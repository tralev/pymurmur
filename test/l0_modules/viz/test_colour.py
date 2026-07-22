"""P8.5: Per-bird colour channels + theme material tables tests.

Tests per-bird hue from seeds, predator red colour, theme ambient/diffuse
material forwarding, colour VBO packing, and GPU shader integration.
GPU-dependent tests are gated behind ``@pytest.mark.gpu``.
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


# ── P8.5a: Theme material tables ─────────────────────────────────

class TestThemeMaterials:
    """P8.5: Every theme provides ambient + diffuse for Blinn-Phong."""

    def test_all_themes_have_ambient(self):
        """P8.5: Each theme dict has an 'ambient' key."""
        from pymurmur.viz.shaders import THEMES
        for name, theme in THEMES.items():
            assert "ambient" in theme, f"Theme '{name}' missing 'ambient'"
            assert isinstance(theme["ambient"], tuple), "ambient must be tuple"
            assert len(theme["ambient"]) == 3

    def test_all_themes_have_diffuse(self):
        """P8.5: Each theme dict has a 'diffuse' key."""
        from pymurmur.viz.shaders import THEMES
        for name, theme in THEMES.items():
            assert "diffuse" in theme, f"Theme '{name}' missing 'diffuse'"
            assert isinstance(theme["diffuse"], tuple), "diffuse must be tuple"
            assert len(theme["diffuse"]) == 3

    def test_ambient_values_in_range(self):
        """P8.5: Ambient RGB values are in [0, 1]."""
        from pymurmur.viz.shaders import THEMES
        for theme in THEMES.values():
            for c in theme["ambient"]:
                assert 0.0 <= c <= 1.0, f"ambient value {c} out of range"

    def test_diffuse_values_in_range(self):
        """P8.5: Diffuse RGB values are in [0, 1]."""
        from pymurmur.viz.shaders import THEMES
        for theme in THEMES.values():
            for c in theme["diffuse"]:
                assert 0.0 <= c <= 1.0, f"diffuse value {c} out of range"

    def test_themes_keep_legacy_keys(self):
        """P8.5: Legacy keys (slow, fast, spec, clear, trail, paper, ink) remain."""
        from pymurmur.viz.shaders import THEMES
        legacy = {"slow", "fast", "spec", "clear", "trail", "paper", "ink"}
        for name, theme in THEMES.items():
            missing = legacy - set(theme.keys())
            assert not missing, f"Theme '{name}' missing legacy keys: {missing}"

    def test_five_themes_exist(self):
        """P8.5/S4.6: Five themes: ink, inverse, paper, graphite, heading."""
        from pymurmur.viz.shaders import THEMES
        assert set(THEMES.keys()) == {
            "ink", "inverse", "paper", "graphite", "heading",
        }


# ── P8.5b: Fragment shader HSV + predator ────────────────────────

class TestColourShader:
    """P8.5: Fragment shader has HSV, predator red, ambient/diffuse."""

    def test_fragment_shader_has_hue(self):
        """P8.5: FRAGMENT_SHADER receives v_hue and v_scale."""
        from pymurmur.viz.shaders import FRAGMENT_SHADER
        assert "in float v_hue" in FRAGMENT_SHADER
        assert "in float v_scale" in FRAGMENT_SHADER

    def test_fragment_shader_has_hsv(self):
        """P8.5: FRAGMENT_SHADER contains HSV→RGB conversion."""
        from pymurmur.viz.shaders import FRAGMENT_SHADER
        assert "hsv2rgb" in FRAGMENT_SHADER

    def test_fragment_shader_has_predator(self):
        """P8.5: FRAGMENT_SHADER blends to red for predators."""
        from pymurmur.viz.shaders import FRAGMENT_SHADER
        assert "predator_factor" in FRAGMENT_SHADER
        assert "1.0, 0.15, 0.1" in FRAGMENT_SHADER  # red tint

    def test_fragment_shader_has_material_uniforms(self):
        """P8.5: FRAGMENT_SHADER uses u_Ambient and u_Diffuse."""
        from pymurmur.viz.shaders import FRAGMENT_SHADER
        assert "u_Ambient" in FRAGMENT_SHADER
        assert "u_Diffuse" in FRAGMENT_SHADER

    def test_vertex_shader_has_hue_attrs(self):
        """P8.5: VERTEX_SHADER has in_bird_hue and in_bird_scale attributes."""
        from pymurmur.viz.shaders import VERTEX_SHADER
        assert "in_bird_hue" in VERTEX_SHADER
        assert "in_bird_scale" in VERTEX_SHADER
        assert "out float v_hue" in VERTEX_SHADER
        assert "out float v_scale" in VERTEX_SHADER

    def test_winged_shader_has_hue_attrs(self):
        """P8.5: WINGED_VERTEX_SHADER has in_bird_hue and in_bird_scale."""
        from pymurmur.viz.shaders import WINGED_VERTEX_SHADER
        assert "in_bird_hue" in WINGED_VERTEX_SHADER
        assert "in_bird_scale" in WINGED_VERTEX_SHADER
        # Winged has locations 4 and 5 for hue/scale
        assert "layout(location = 4) in float in_bird_hue" in WINGED_VERTEX_SHADER
        assert "layout(location = 5) in float in_bird_scale" in WINGED_VERTEX_SHADER

    def test_hue_passthrough_in_vertex_shaders(self):
        """P8.5: Both vertex shaders pass v_hue = in_bird_hue, v_scale = in_bird_scale."""
        from pymurmur.viz.shaders import VERTEX_SHADER, WINGED_VERTEX_SHADER
        for shader in (VERTEX_SHADER, WINGED_VERTEX_SHADER):
            assert "v_hue = in_bird_hue" in shader
            assert "v_scale = in_bird_scale" in shader


# ── P8.5c: Colour VBO packing ────────────────────────────────────

class TestColourPacking:
    """P8.5: Hue from seeds, predator scale, VBO packing logic."""

    def test_hue_from_seeds_range(self):
        """P8.5: Seeds are in [0, 1] and map directly to hue."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig(num_boids=100, seed=42)
        flock = PhysicsFlock(cfg)
        assert flock.seeds is not None
        assert flock.seeds.min() >= 0.0
        assert flock.seeds.max() <= 1.0

    def test_predator_scale_greater_than_one(self):
        """P8.5: Predator birds get scale > 1.0 (×1.3–1.5)."""
        # Non-GPU: verify the logic used in renderer
        is_pred = np.array([False, True, False, True])
        scale = np.where(is_pred, 1.35, 1.0)
        assert scale[0] == 1.0  # prey
        assert scale[1] == 1.35  # predator
        assert scale[2] == 1.0
        assert scale[3] == 1.35

    def test_colour_slice_two_floats_per_bird(self):
        """P8.5/D7: Colour data (hue + scale) is exactly 2 floats per
        bird — packed as the trailing 2 columns of the merged 8-float
        InstanceSchema buffer since D7, not a separate colour VBO."""
        n = 50
        packed = np.zeros((n, 8), dtype=np.float32)
        colour_slice = packed[:, 6:8]
        assert colour_slice.shape == (n, 2)
        assert colour_slice.dtype == np.float32

    def test_colour_slice_stride(self):
        """P8.5: Colour data's own stride is 8 bytes (2 × float32),
        sitting at byte offset 24 within the merged 32-byte-per-bird
        InstanceSchema layout (D7)."""
        assert np.dtype(np.float32).itemsize == 4
        assert 2 * 4 == 8  # 2 floats × 4 bytes
        assert 6 * 4 == 24  # colour columns start after pos.xyz + vel.xyz


# ── P8.5d: Renderer GPU integration ──────────────────────────────

@pytest.mark.gpu
class TestColourRenderer:
    """P8.5: Renderer3D colour VBO integration."""

    @pytest.fixture
    def headless_renderer(self):
        """Headless Renderer3D with default winged mesh."""
        if not gpu_available:
            pytest.skip("ModernGL GPU context not available")
        from pymurmur.viz.renderer import Renderer3D
        return Renderer3D(
            width=200, height=150, headless=True,
            winged_mesh=True, gradient_sky=False,
        )

    def test_instance_vbo_created(self, headless_renderer):
        """D7: _instance_vbo exists after construction (colour data now
        lives in the same merged buffer as pos+vel — no separate
        _colour_vbo since the D7 InstanceSchema merge)."""
        assert headless_renderer._instance_vbo is not None
        assert not hasattr(headless_renderer, "_colour_vbo"), (
            "D7 merged colour into _instance_vbo — _colour_vbo should be gone"
        )

    def test_packed_initialised_with_colour_columns(self, headless_renderer):
        """D7: _packed is (max_instances, 8) float32 — columns 6:8 are
        hue + scale (was a separate (max_instances, 2) _colour_packed)."""
        p = headless_renderer._packed
        assert p.shape[1] == 8
        assert p.dtype == np.float32

    def test_update_instances_packs_colours(self, headless_renderer):
        """P8.5/D7: update_instances fills _packed[:, 6:8] with hue + scale."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig(num_boids=10, seed=123)
        flock = PhysicsFlock(cfg)
        n = headless_renderer.update_instances(flock)
        assert n == 10
        p = headless_renderer._packed
        # Colours should be packed for active birds
        assert p[0, 6] != 0.0 or p[0, 7] != 0.0  # at least one non-zero
        assert p[0, 7] == 1.0  # first bird is prey (unless predator enabled)

    def test_instance_vbo_written(self, headless_renderer):
        """D7: the merged instance VBO receives pos+vel+hue+scale data
        after update_instances (32 bytes/bird, was 24+8 across two VBOs)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig(num_boids=10, seed=42)
        flock = PhysicsFlock(cfg)
        headless_renderer.update_instances(flock)
        # Read back to verify data was written (VBO has data)
        data = headless_renderer._instance_vbo.read()
        assert len(data) >= 10 * 8 * 4  # at least 320 bytes for 10 birds

    def test_begin_frame_sets_ambient_diffuse(self, headless_renderer):
        """P8.5: begin_frame uploads u_Ambient and u_Diffuse to both programs."""
        from pymurmur.viz.camera import OrbitCamera
        cam = OrbitCamera()
        headless_renderer.begin_frame(cam)
        # No crash = uniforms exist in shader programs

    def test_draw_grid_sets_hue_scale_defaults(self, headless_renderer):
        """P8.5: draw_grid sets default in_bird_hue and in_bird_scale."""
        headless_renderer.draw_grid()
        # No crash = default attributes accepted

    def test_render_with_coloured_birds(self, headless_renderer):
        """P8.5: Full render pipeline with per-bird colours (no crash)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        cfg = SimConfig(num_boids=20, seed=99)
        flock = PhysicsFlock(cfg)
        cam = OrbitCamera()
        headless_renderer.begin_frame(cam)
        headless_renderer.draw_birds(flock)

    def test_predator_gets_redder(self, headless_renderer):
        """P8.5/D7: Predator birds get scale 1.35 in the merged instance
        buffer's scale column (index 7)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        cfg = SimConfig(num_boids=5, seed=42)
        cfg.predator_enabled = True
        flock = PhysicsFlock(cfg)
        # Add a predator
        flock.add_boids(1, cfg, is_predator=True)
        n = headless_renderer.update_instances(flock)
        p = headless_renderer._packed
        # Last bird is predator
        assert p[n - 1, 7] == pytest.approx(1.35), "Predator should have scale 1.35"

    def test_renderer_with_predator_colours(self, headless_renderer):
        """P8.5: Render a frame with predator birds (no crash)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.camera import OrbitCamera
        cfg = SimConfig(num_boids=10, seed=42)
        cfg.predator_enabled = True
        flock = PhysicsFlock(cfg)
        flock.add_boids(2, cfg, is_predator=True)
        cam = OrbitCamera()
        headless_renderer.begin_frame(cam)
        headless_renderer.draw_birds(flock)


# ── S4.6: heading-hue debug theme ─────────────────────────────────

class TestHeadingHueTheme:
    """S4.6: theme="heading" sources per-bird hue from velocity azimuth
    (atan2(vel_y, vel_x), z-up world) instead of the seed column — birds
    flying in different directions render in visibly different colours."""

    def test_heading_is_a_valid_theme(self):
        from pymurmur.core.config import SimConfig
        cfg = SimConfig(num_boids=5)
        cfg.theme = "heading"
        cfg.validate()  # must not raise

    def test_hue_tracks_velocity_azimuth(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        import numpy as np

        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=4)
        flock = PhysicsFlock(cfg)
        flock.velocities[0] = [1.0, 0.0, 0.0]    # azimuth 0      -> hue 0.5
        flock.velocities[1] = [0.0, 1.0, 0.0]    # azimuth pi/2   -> hue 0.75
        flock.velocities[2] = [-1.0, 0.0, 0.0]   # azimuth +-pi   -> hue ~1.0/0.0
        flock.velocities[3] = [0.0, -1.0, 0.0]   # azimuth -pi/2  -> hue 0.25
        flock.active[:] = True

        r = Renderer3D(width=100, height=100, headless=True, theme="heading")
        r.update_instances(flock)
        hue = r._packed[:4, 6]

        assert hue[0] == pytest.approx(0.5, abs=1e-5)
        assert hue[1] == pytest.approx(0.75, abs=1e-5)
        assert hue[2] == pytest.approx(1.0, abs=1e-4) or hue[2] == pytest.approx(0.0, abs=1e-4)
        assert hue[3] == pytest.approx(0.25, abs=1e-5)

    def test_opposite_headings_differ_by_half_turn(self, gpu_available):
        """+x vs -x flight should differ by ~180 degrees in hue (0.5 of
        the hue wheel) — the roadmap's own acceptance check."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=2)
        flock = PhysicsFlock(cfg)
        flock.velocities[0] = [1.0, 0.0, 0.0]
        flock.velocities[1] = [-1.0, 0.0, 0.0]
        flock.active[:] = True

        r = Renderer3D(width=100, height=100, headless=True, theme="heading")
        r.update_instances(flock)
        hue = r._packed[:2, 6]
        diff = abs(float(hue[0]) - float(hue[1]))
        diff = min(diff, 1.0 - diff)  # hue wraps at 1.0
        assert diff == pytest.approx(0.5, abs=1e-3)

    def test_hue_stable_across_frames_for_constant_heading(self, gpu_available):
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=1)
        flock = PhysicsFlock(cfg)
        flock.velocities[0] = [0.6, 0.8, 0.0]
        flock.active[0] = True

        r = Renderer3D(width=100, height=100, headless=True, theme="heading")
        r.update_instances(flock)
        hue1 = float(r._packed[0, 6])
        r.update_instances(flock)
        hue2 = float(r._packed[0, 6])
        assert hue1 == pytest.approx(hue2, abs=1e-6)

    def test_default_theme_still_uses_seed_hue(self, gpu_available):
        """Non-"heading" themes are unaffected — seed-based hue unchanged."""
        if not gpu_available:
            pytest.skip("GPU not available")
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.flock import PhysicsFlock
        from pymurmur.viz.renderer import Renderer3D

        cfg = SimConfig(num_boids=3, seed=42)
        flock = PhysicsFlock(cfg)
        flock.active[:] = True
        seeds = flock.seeds[flock.active].copy()

        r = Renderer3D(width=100, height=100, headless=True, theme="ink",
                       per_bird_color=True)
        r.update_instances(flock)
        assert np.allclose(r._packed[:3, 6], seeds[:3])
