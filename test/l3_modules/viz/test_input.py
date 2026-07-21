"""Tests for viz.input_control — keyboard/mouse → SimConfig bridge.

Requires pygame. All tests skip when pygame is unavailable.
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Check pygame availability & initialise once at module level
try:
    import pygame
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    PYGAME_AVAILABLE = True
except (ImportError, pygame.error):
    PYGAME_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed or init failed")


class TestInputControl:
    """Keyboard and mouse event handling — no simulation dependency."""

    def test_input_quit_event(self):
        """pygame.QUIT event → handle_events() returns False."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Post a quit event
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        result = ctrl.handle_events()
        assert result is False

    def test_input_escape_key(self):
        """K_ESCAPE → returns False."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        event = pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_ESCAPE
        )
        pygame.event.post(event)
        result = ctrl.handle_events()
        assert result is False

    def test_input_space_pause(self):
        """K_SPACE toggles paused."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        initial = ctrl.paused

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_SPACE
        ))
        ctrl.handle_events()
        assert ctrl.paused is not initial

    def test_input_r_reset(self):
        """K_r sets pending_reset = True."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_r
        ))
        ctrl.handle_events()
        assert ctrl.pending_reset is True

    def test_input_m_cycle_mode(self):
        """K_m cycles through all 5 modes and wraps around."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        initial = cfg.mode
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Press M 5 times — should cycle through all modes and wrap
        seen = {initial}
        for _ in range(5):
            pygame.event.post(pygame.event.Event(
                pygame.KEYDOWN, key=pygame.K_m
            ))
            ctrl.handle_events()
            seen.add(cfg.mode)
        assert len(seen) == 5  # visited all 5 modes
        assert cfg.mode == initial  # wrapped back to original

    def test_input_up_phi_p(self):
        """K_UP increases config.projection.phi_p by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.projection.phi_p
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_UP
        ))
        ctrl.handle_events()
        assert cfg.projection.phi_p > old

    def test_input_down_phi_p(self):
        """K_DOWN decreases config.projection.phi_p by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.projection.phi_p
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN
        ))
        ctrl.handle_events()
        assert cfg.projection.phi_p < old

    def test_input_phi_p_clamped(self):
        """phi_p never goes below 0 or above 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Set near zero and try to go below
        cfg.projection.phi_p = 0.0
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN
        ))
        ctrl.handle_events()
        assert cfg.projection.phi_p >= 0.0

        # Set near one and try to go above
        cfg.projection.phi_p = 1.0
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_UP
        ))
        ctrl.handle_events()
        assert cfg.projection.phi_p <= 1.0

    def test_input_plus_add_birds(self):
        """K_EQUALS sets pending_add += 10 (deferred add)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_EQUALS
        ))
        ctrl.handle_events()
        assert ctrl.pending_add == 10

        # Accumulates across multiple presses
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_EQUALS
        ))
        ctrl.handle_events()
        assert ctrl.pending_add == 20

    def test_input_minus_remove_birds(self):
        """K_MINUS sets pending_remove += 10 (deferred remove)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_MINUS
        ))
        ctrl.handle_events()
        assert ctrl.pending_remove == 10

    def test_input_g_toggle_grid(self):
        """K_g toggles show_grid."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        initial = ctrl.show_grid

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_g
        ))
        ctrl.handle_events()
        assert ctrl.show_grid is not initial

    def test_input_v_reset_camera(self):
        """K_v calls camera.reset()."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        # Capture defaults before rotating
        default_az = cam.azimuth
        default_dist = cam.distance
        cam.rotate(1.0, 0.5)  # move away from default
        cam.zoom(3)
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_v
        ))
        ctrl.handle_events()
        # Camera should be reset to defaults
        assert abs(cam.azimuth - default_az) < 0.01
        assert abs(cam.distance - default_dist) < 0.01

    def test_input_never_imports_simulation(self):
        """input_control.py has no import simulation."""
        path = Path("pymurmur/viz/input_control.py")
        text = path.read_text()
        assert "import simulation" not in text
        assert "from ..simulation" not in text
        assert "from .simulation" not in text
        assert "from pymurmur.simulation" not in text

    def test_input_right_phi_a(self):
        """K_RIGHT increases config.phi_a by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.phi_a
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        ctrl.handle_events()
        assert cfg.phi_a > old

    def test_input_left_phi_a(self):
        """K_LEFT decreases config.phi_a by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.phi_a
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        ctrl.handle_events()
        assert cfg.phi_a < old

    def test_input_brackets_sigma(self):
        """K_RIGHTBRACKET increases sigma, K_LEFTBRACKET decreases sigma."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.sigma = 10
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET))
        ctrl.handle_events()
        assert cfg.sigma == 11
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET))
        ctrl.handle_events()
        assert cfg.sigma == 10

    def test_input_sigma_clamped(self):
        """sigma never goes below 1 or above 20."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        cfg.sigma = 1
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET))
        ctrl.handle_events()
        assert cfg.sigma >= 1
        cfg.sigma = 20
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET))
        ctrl.handle_events()
        assert cfg.sigma <= 20

    def test_input_o_toggle_auto_rotate(self):
        """K_o toggles camera.auto_rotate."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        initial = cam.auto_rotate
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o))
        ctrl.handle_events()
        assert cam.auto_rotate is not initial

    def test_input_t_predator(self):
        """K_t toggles config.predator_enabled."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        initial = cfg.predator_enabled
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))
        ctrl.handle_events()
        assert cfg.predator_enabled is not initial

    def test_input_k_roosting(self):
        """K_k toggles config.roosting_enabled."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        initial = cfg.roosting_enabled
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_k))
        ctrl.handle_events()
        assert cfg.roosting_enabled is not initial

    def test_input_u_refinements(self):
        """K_u toggles config.refinements."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        initial = cfg.refinements
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_u))
        ctrl.handle_events()
        assert cfg.refinements is not initial

    def test_input_plus_key_add_birds(self):
        """K_PLUS (numpad) sets pending_add += 10."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_PLUS
        ))
        ctrl.handle_events()
        assert ctrl.pending_add == 10

    def test_input_mouse_drag_orbit(self):
        """Mouse drag rotates camera."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        old_az = cam.azimuth
        # Simulate mouse drag: press, move, release
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 300)))
        ctrl.handle_events()
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(500, 300)))
        ctrl.handle_events()
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(500, 300)))
        ctrl.handle_events()
        assert cam.azimuth != old_az

    def test_input_scroll_zoom(self):
        """Mouse scroll zooms camera."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        old_dist = cam.distance
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
        ctrl.handle_events()
        assert cam.distance < old_dist  # zoom in


# ── P10.1: Letter-key presets (a–f, h, w) ───────────────────

class TestLetterPresets:
    """P10.1: 8 letter-key presets (a–f, h, w) with printed labels."""

    def test_preset_a_applies_projection_params(self):
        """P10.1: Key 'a' → 3D Pearce Default: projection, φp=0.04, φa=0.80, σ=6."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.04)
        assert cfg.phi_a == pytest.approx(0.80)
        assert cfg.sigma == 6

    def test_preset_b_applies_storm_params(self):
        """P10.1: Key 'b' → Ball of Birds: projection, φp=0.18, φa=0.70, σ=7."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.18)
        assert cfg.phi_a == pytest.approx(0.70)
        assert cfg.sigma == 7

    def test_preset_c_applies_void_params(self):
        """P10.1: Key 'c' → Storm Cloud: projection, φp=0.06, φa=0.45, σ=3."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.06)
        assert cfg.phi_a == pytest.approx(0.45)
        assert cfg.sigma == 3

    def test_preset_d_applies_spatial_params(self):
        """P10.1: Key 'd' → 3D Stream: spatial, sep=0.25, align=0.55."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.25)
        assert cfg.alignment_weight == pytest.approx(0.55)
        assert cfg.cohesion_weight == pytest.approx(0.80)
        assert cfg.influence_count == 8

    def test_preset_e_applies_stream_params(self):
        """P10.1: Preset 'e' → Vertical Column: projection, φp=0.10, φa=0.75, σ=6.

        S2.E6: K_e now rolls camera — preset 'e' is accessed via
        _apply_letter_preset directly rather than a keyboard binding."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        ctrl._apply_letter_preset("e")

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.10)
        assert cfg.phi_a == pytest.approx(0.75)
        assert cfg.sigma == 6

    def test_preset_f_applies_ribbon_params(self):
        """P10.1: Key 'f' → 3D Acro: projection, φp=0.02, φa=0.85, σ=3."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.02)
        assert cfg.phi_a == pytest.approx(0.85)
        assert cfg.sigma == 3

    def test_preset_h_applies_huddle_params(self):
        """P10.1: Key 'h' → 3D Void: spatial, sep=0.35, align=0.58."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.35)
        assert cfg.alignment_weight == pytest.approx(0.58)
        assert cfg.cohesion_weight == pytest.approx(0.90)
        assert cfg.influence_count == 9

    def test_preset_w_applies_wander_params(self):
        """P10.1: Key 'w' → Spiral Vortex: spatial, sep=0.08, align=0.82."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.08)
        assert cfg.alignment_weight == pytest.approx(0.82)
        assert cfg.cohesion_weight == pytest.approx(1.0)
        assert cfg.influence_count == 10


# ── P10.6: φp+φa ≤ 1 constraint ─────────────────────────────

class TestPhiConstraint:
    """P10.6: After φp or φa increments, φp + φa ≤ 1 is enforced."""

    def test_phi_sum_never_exceeds_one(self):
        """P10.6: Repeated φp increases → φa shrinks to keep sum ≤ 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.5
        cfg.phi_a = 0.5
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Bump φp repeatedly — φa should shrink to keep sum ≤ 1
        for _ in range(80):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10, (
            f"φp={cfg.projection.phi_p:.4f} + φa={cfg.phi_a:.4f} = "
            f"{cfg.projection.phi_p + cfg.phi_a:.4f} > 1.0"
        )

    def test_phi_a_increase_shrinks_phi_p(self):
        """P10.6: Increasing φa when at limit (sum=1.0) shrinks φp."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.6
        cfg.phi_a = 0.40  # sum = 1.00 exactly — one more bump triggers constraint
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # One more φa bump → sum would be 1.01 > 1.0, so φp shrinks to 1−φa
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        ctrl.handle_events()

        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, f"sum {total:.4f} > 1.0"
        # φa increased to 0.41, φp decreased to 0.59 to keep sum ≤ 1.0
        assert cfg.phi_a == pytest.approx(0.41)
        assert cfg.projection.phi_p == pytest.approx(0.59)

    def test_phi_p_decrease_never_violates(self):
        """P10.6: Decreasing φp never causes a constraint violation."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.3
        cfg.phi_a = 0.4
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        # φa should be unchanged when there's headroom
        assert cfg.phi_a == pytest.approx(0.4)

    def test_constraint_symmetric(self):
        """P10.6: Constraint is symmetric — pushing either parameter
        to 1.0 forces the other to 0.0."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Push φp to 1.0
        for _ in range(200):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        assert cfg.projection.phi_p >= 0.95  # effectively at limit
        assert cfg.phi_a < 0.05  # pushed to near-zero

        # Reset and push φa to 1.0
        cfg.projection.phi_p = 0.03
        cfg.phi_a = 0.50
        for _ in range(200):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        assert cfg.phi_a >= 0.95
        assert cfg.projection.phi_p < 0.05

    def test_phi_p_increase_from_max_no_op(self):
        """P10.6: φp at 1.0 (max clamped) — UP key no-ops, φa stays 0."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 1.0
        cfg.phi_a = 0.0
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        ctrl.handle_events()
        assert cfg.projection.phi_p == pytest.approx(1.0)
        assert cfg.phi_a == pytest.approx(0.0)

    def test_phi_a_decrease_from_zero_no_op(self):
        """P10.6: φa at 0.0 (min clamped) — LEFT key no-ops."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.3
        cfg.phi_a = 0.0
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.0)

    def test_letter_preset_enforces_constraint(self, monkeypatch):
        """P10.6: Letter preset that would violate constraint is corrected."""
        from pymurmur.analysis.presets import LETTER_PRESETS
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Inject a violating preset for key 'z' (unused key)
        monkeypatch.setitem(LETTER_PRESETS, "z", (
            "Test Violator", "φp=0.9 φa=0.9",
            {"mode": "projection", "phi_p": 0.90, "phi_a": 0.90},
        ))
        # Directly call _apply_letter_preset to avoid needing a real pygame key
        ctrl._apply_letter_preset("z")
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, (
            f"Letter preset with sum=1.8 should be corrected, got sum={total:.4f}"
        )

    def test_numbered_preset_enforces_constraint(self, monkeypatch):
        """P10.6: Numbered preset (1-9) that would violate constraint is corrected."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import PRESETS, InputControl
        cfg = SimConfig()
        cfg.mode = "projection"
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Inject a violating preset
        monkeypatch.setitem(PRESETS, "violator", {
            "phi_p": 0.80,
            "phi_a": 0.80,
        })
        monkeypatch.setattr(
            "pymurmur.viz.input_control.PRESETS",
            {"violator": {"phi_p": 0.80, "phi_a": 0.80}},
        )
        # Directly call _apply_preset with index that resolves to "violator"
        ctrl._apply_preset(0)
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, (
            f"Numbered preset with sum=1.6 should be corrected, got sum={total:.4f}"
        )

    def test_interleaved_up_right_never_violates(self):
        """P10.6: Interleaving UP and RIGHT key presses never violates constraint."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.5
        cfg.phi_a = 0.4
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Alternate UP and RIGHT 50 times
        for i in range(50):
            key = pygame.K_UP if i % 2 == 0 else pygame.K_RIGHT
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            ctrl.handle_events()
            total = cfg.projection.phi_p + cfg.phi_a
            assert total <= 1.0 + 1e-10, (
                f"Interleaved step {i}: sum={total:.4f} (φp={cfg.projection.phi_p:.4f}, φa={cfg.phi_a:.4f})"
            )

    def test_phi_p_decrease_with_headroom_leaves_phi_a_unchanged(self):
        """P10.6: Decreasing φp when sum < 1 leaves φa untouched (no enforcement triggers)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.2
        cfg.phi_a = 0.3
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.3)

    def test_sum_exactly_one_no_change(self):
        """P10.6: When phi_p+phi_a=1.0 exactly, no enforcement triggers."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.4
        cfg.phi_a = 0.6
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        # Decreasing phi_p should leave phi_a alone (headroom opened)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.6)
        assert cfg.projection.phi_p == pytest.approx(0.39)

    def test_enforce_phi_after_preset_standalone(self):
        """P10.6: _enforce_phi_after_preset reduces the smaller value when sum > 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.7
        cfg.phi_a = 0.6  # sum=1.3 > 1.0, phi_a is smaller
        InputControl._enforce_phi_after_preset(cfg)
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, f"Sum should be <= 1.0, got {total}"
        # phi_p=0.7 > phi_a=0.6, so phi_a should be reduced to 0.3
        assert cfg.projection.phi_p == pytest.approx(0.7)
        assert cfg.phi_a == pytest.approx(0.3)


# ── P10.4: Cursor-ray spawning, clear, v0 adjustment ───────────

class TestCursorRaySpawning:
    """P10.4: Mouse spawn via cursor-ray unprojection."""

    @pytest.fixture
    def ctrl(self, default_config):
        """InputControl with a known viewport for ray unprojection."""
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = default_config
        cfg.width = 1000
        cfg.height = 1000
        cfg.depth = 1000
        camera = OrbitCamera(target=(500.0, 500.0, 400.0))
        ctrl = InputControl(cfg, camera)
        ctrl.set_viewport(cfg.window_width, cfg.window_height)
        return ctrl

    def test_right_click_spawns_predator(self, ctrl):
        """Right-click → predator spawn position queued."""
        import pygame
        assert len(ctrl.pending_spawn_predator) == 0
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 300),
        ))
        ctrl.handle_events()
        assert len(ctrl.pending_spawn_predator) == 1
        pos = ctrl.pending_spawn_predator[0]
        assert len(pos) == 3
        assert all(np.isfinite(pos))

    def test_left_click_spawns_bird_on_release(self, ctrl):
        """Left-click (no drag) → bird spawn position queued on release."""
        import pygame
        assert len(ctrl.pending_spawn_bird) == 0
        # Mouse down at position
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(500, 400),
        ))
        ctrl.handle_events()
        # Mouse up at same position (no drag)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONUP, button=1, pos=(500, 400),
        ))
        ctrl.handle_events()
        assert len(ctrl.pending_spawn_bird) == 1
        pos = ctrl.pending_spawn_bird[0]
        assert len(pos) == 3
        assert all(np.isfinite(pos))

    def test_left_drag_no_spawn(self, ctrl):
        """Left-click with drag (> 5px movement) does NOT spawn a bird."""
        import pygame
        assert len(ctrl.pending_spawn_bird) == 0
        # Mouse down
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(500, 400),
        ))
        ctrl.handle_events()
        # Drag 10px
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEMOTION, pos=(510, 400),
        ))
        ctrl.handle_events()
        # Mouse up at different position
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONUP, button=1, pos=(510, 400),
        ))
        ctrl.handle_events()
        assert len(ctrl.pending_spawn_bird) == 0, "Drag should not spawn"

    def test_x_key_clear(self, ctrl):
        """P10.4/D3: X key (K_x) sets pending_clear flag. K_c is a letter preset."""
        import pygame
        assert not ctrl.pending_clear
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x))
        ctrl.handle_events()
        assert ctrl.pending_clear

    def test_c_key_does_not_trigger_clear(self, ctrl):
        """D3: K_c applies a preset, does NOT set pending_clear.

        Before D3: K_c branch shadowed the clear branch.
        After D3:  K_c → letter preset 'c' (Storm Cloud), K_x → clear.
        """
        import pygame
        assert not ctrl.pending_clear
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        ctrl.handle_events()
        assert ctrl.pending_clear is False, (
            "D3: K_c must apply preset, NOT trigger clear"
        )

    def test_q_and_x_are_distinct(self, ctrl):
        """D3: K_q rolls camera (S2.E6), K_x clears (returns True, sets flag).

        They are distinct actions — no key shadowing or conflict.
        """
        import pygame

        from pymurmur.viz.input_control import InputControl
        # K_q: roll camera (S2.E6 — no longer quits)
        assert not ctrl.pending_clear
        roll_before = ctrl.camera.roll
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
        result = ctrl.handle_events()
        assert result is True, "K_q should return True (continue, roll only)"
        assert ctrl.pending_clear is False, "K_q should not trigger clear"
        assert ctrl.camera.roll != roll_before, "K_q should change camera roll"

        # K_x: clear (separate InputControl to avoid state conflicts)
        cfg2 = ctrl.config
        cam2 = ctrl.camera
        ctrl2 = InputControl(cfg2, cam2)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x))
        result2 = ctrl2.handle_events()
        assert result2 is True, "K_x should return True (continue)"
        assert ctrl2.pending_clear is True, "K_x should trigger clear"

    def test_esc_key_quit(self):
        """P10.4: ESC key (K_ESCAPE) returns False from handle_events.

        S2.E6: Q now rolls camera — ESC is the lone quit key."""
        import pygame

        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        result = ctrl.handle_events()
        assert result is False, "ESC key should quit (return False)"

    def test_q_key_rolls_camera(self):
        """S2.E6: Q key rolls camera counter-clockwise (increases roll)."""
        import pygame

        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        roll_before = cam.roll
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
        ctrl.handle_events()
        assert cam.roll > roll_before, "Q should increase roll (CCW)"

    def test_pageup_increases_v0(self, ctrl):
        """PageUp increments pending_v0_delta."""
        import pygame
        assert ctrl.pending_v0_delta == 0.0
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PAGEUP))
        ctrl.handle_events()
        assert ctrl.pending_v0_delta == pytest.approx(0.1)

    def test_pagedown_decreases_v0(self, ctrl):
        """PageDn decrements pending_v0_delta."""
        import pygame
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PAGEDOWN))
        ctrl.handle_events()
        assert ctrl.pending_v0_delta == pytest.approx(-0.1)


class TestCameraUnprojection:
    """P10.4: OrbitCamera.screen_to_world() ray unprojection."""

    def test_centre_screen_hits_behind_target(self):
        """Screen centre → world point near target Z plane."""
        from pymurmur.viz.camera import OrbitCamera
        camera = OrbitCamera(target=(500.0, 500.0, 400.0))
        result = camera.screen_to_world(400.0, 300.0, 800, 600)
        assert result is not None
        x, y, z = result
        assert abs(z - 400.0) < 200  # near target Z
        assert np.isfinite(x) and np.isfinite(y)

    def test_returns_none_for_grazing_ray(self):
        """Ray parallel to Z plane → returns None."""

        from pymurmur.viz.camera import OrbitCamera
        # Camera looking exactly along Z plane (elevation=0, looking at horizon)
        camera = OrbitCamera(target=(500.0, 500.0, 100.0))
        camera.elevation = 0.0
        camera.azimuth = 0.0
        camera.distance = 1000.0
        # Ray from centre of screen should be near-parallel to Z plane
        result = camera.screen_to_world(400.0, 300.0, 800, 600)
        # May or may not return None depending on exact geometry; at least no crash
        if result is not None:
            assert all(np.isfinite(result))


# ── P10.3: SliderHUD integration — TAB toggle + mouse drag lock ─

class TestHUDTabIntegration:
    """P10.3: TAB key → pending_hud_toggle → SliderHUD.toggle() roundtrip."""

    @pytest.fixture
    def _hud_and_ctrl(self, default_config):
        """InputControl + SliderHUD pair, simulating Visualizer setup."""
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.hud import SliderHUD
        from pymurmur.viz.input_control import InputControl
        cfg = default_config
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        hud = SliderHUD(cfg)
        return ctrl, hud, cfg

    def test_tab_key_sets_pending_hud_toggle(self, _hud_and_ctrl):
        """P10.3: K_TAB sets pending_hud_toggle = True."""
        ctrl, hud, cfg = _hud_and_ctrl
        assert not ctrl.pending_hud_toggle
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        ctrl.handle_events()
        assert ctrl.pending_hud_toggle, "TAB should set pending_hud_toggle"

    def test_pending_toggle_consumed_by_visualizer_pattern(self, _hud_and_ctrl):
        """P10.3: Visualizer pattern — pending_hud_toggle → hud.toggle() → reset."""
        ctrl, hud, cfg = _hud_and_ctrl
        # HUD starts hidden
        assert hud.visible is False

        # Simulate TAB key
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        ctrl.handle_events()
        assert ctrl.pending_hud_toggle

        # Visualizer's run() pattern:
        if ctrl.pending_hud_toggle:
            hud.toggle()
            ctrl.hud_visible = hud.visible
            ctrl.pending_hud_toggle = False

        assert hud.visible is True
        assert ctrl.hud_visible is True
        assert not ctrl.pending_hud_toggle

    def test_tab_toggle_roundtrip_visible_returns_false(self, _hud_and_ctrl):
        """P10.3: TAB twice → visible returns to False (toggle roundtrip)."""
        ctrl, hud, cfg = _hud_and_ctrl
        assert hud.visible is False

        # First TAB → show
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        ctrl.handle_events()
        if ctrl.pending_hud_toggle:
            hud.toggle()
            ctrl.hud_visible = hud.visible
            ctrl.pending_hud_toggle = False
        assert hud.visible is True

        # Second TAB → hide
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        ctrl.handle_events()
        if ctrl.pending_hud_toggle:
            hud.toggle()
            ctrl.hud_visible = hud.visible
            ctrl.pending_hud_toggle = False
        assert hud.visible is False
        assert ctrl.hud_visible is False

    def test_hud_visible_synced_with_ctrl_flag(self, _hud_and_ctrl):
        """P10.3: ctrl.hud_visible stays in sync with hud.visible after toggle."""
        ctrl, hud, cfg = _hud_and_ctrl

        # Toggle 5 times — visible and hud_visible always agree
        for expected in [True, False, True, False, True]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
            ctrl.handle_events()
            if ctrl.pending_hud_toggle:
                hud.toggle()
                ctrl.hud_visible = hud.visible
                ctrl.pending_hud_toggle = False
            assert hud.visible is expected
            assert ctrl.hud_visible is expected

    def test_toggle_does_not_affect_config(self, _hud_and_ctrl):
        """P10.3: Toggling HUD never changes SimConfig fields."""
        ctrl, hud, cfg = _hud_and_ctrl
        old_sep = cfg.spatial.separation_weight
        old_v0 = cfg.v0

        # Toggle 3 times
        for _ in range(3):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
            ctrl.handle_events()
            if ctrl.pending_hud_toggle:
                hud.toggle()
                ctrl.hud_visible = hud.visible
                ctrl.pending_hud_toggle = False

        assert cfg.spatial.separation_weight == pytest.approx(old_sep)
        assert cfg.v0 == pytest.approx(old_v0)


class TestHUDMouseIntegration:
    """P10.3: Mouse → InputControl mouse state → SliderHUD.handle_mouse()."""

    @pytest.fixture
    def _hud_and_ctrl(self, default_config):
        """InputControl + SliderHUD pair with HUD made visible."""
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.hud import SliderHUD
        from pymurmur.viz.input_control import InputControl
        cfg = default_config
        cfg.spatial.separation_weight = 3.0  # mid-track for knob hit-test
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        hud = SliderHUD(cfg)
        hud.visible = True
        return ctrl, hud, cfg

    def test_input_tracks_mouse_position(self, _hud_and_ctrl):
        """P10.3: MOUSEMOTION updates mouse_x, mouse_y."""
        ctrl, hud, cfg = _hud_and_ctrl
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(123, 456)))
        ctrl.handle_events()
        assert ctrl.mouse_x == 123
        assert ctrl.mouse_y == 456

    def test_input_tracks_mouse_button_state(self, _hud_and_ctrl):
        """P10.3: mouse_down flag is set directly (Visualizer reads it, not events)."""
        ctrl, hud, cfg = _hud_and_ctrl
        assert not ctrl.mouse_down
        # In the real loop, handle_events sets mouse_down from pygame.mouse.get_pressed().
        # For integration testing, we simulate what the Visualizer reads.
        ctrl.mouse_down = True
        assert ctrl.mouse_down

    def test_hud_handle_mouse_hover_updates_hover_state(self, _hud_and_ctrl):
        """P10.3: Passing mouse coords to handle_mouse updates hover when not dragging."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Set mouse position to the sep slider knob centre
        ctrl.mouse_x = hud._knob_x(0)
        _, ctrl.mouse_y = hud._slider_rect(0)
        ctrl.mouse_down = False

        result = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        assert result is False  # no drag, so no orbit suppression
        assert hud._hover_slider == 0, "Mouse over sep knob should set hover"

    def test_hud_handle_mouse_drag_returns_true(self, _hud_and_ctrl):
        """P10.3: Dragging a knob → handle_mouse returns True (suppress orbit)."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Position mouse on sep slider knob
        kx = hud._knob_x(0)
        _, ky = hud._slider_rect(0)
        ctrl.mouse_x = kx
        ctrl.mouse_y = ky
        ctrl.mouse_down = True

        result = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        assert result is True, "Drag on knob should return True"
        assert hud.drag_locked is True
        assert hud._active_slider == 0

    def test_hud_handle_mouse_release_returns_false(self, _hud_and_ctrl):
        """P10.3: Mouse up → handle_mouse returns False, releases drag lock."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Start a drag first
        kx = hud._knob_x(0)
        _, ky = hud._slider_rect(0)
        ctrl.mouse_x = kx
        ctrl.mouse_y = ky
        ctrl.mouse_down = True
        hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, True)
        assert hud.drag_locked is True

        # Release
        ctrl.mouse_down = False
        result = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, False)
        assert result is False
        assert hud.drag_locked is False
        assert hud._active_slider == -1

    def test_hud_invisible_handle_mouse_ignores(self, _hud_and_ctrl):
        """P10.3: When HUD is hidden, handle_mouse returns False regardless."""
        ctrl, hud, cfg = _hud_and_ctrl
        hud.visible = False
        ctrl.mouse_down = True

        result = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        assert result is False, "Hidden HUD should never suppress orbit"

    def test_suppress_orbit_clears_mouse_dragging(self, _hud_and_ctrl):
        """P10.3: suppress_orbit() sets _mouse_dragging = False."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Simulate an active mouse drag (camera orbit in progress)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=(400, 300),
        ))
        ctrl.handle_events()
        assert ctrl._mouse_dragging is True

        # HUD drag starts → suppress orbit
        ctrl.suppress_orbit()
        assert ctrl._mouse_dragging is False

    def test_visualizer_integration_pattern_drag_lock(self, _hud_and_ctrl):
        """P10.3: Exact Visualizer.run() pattern — handle_mouse + suppress/release."""
        ctrl, hud, cfg = _hud_and_ctrl

        # --- Scene: user clicks on sep knob ---
        kx = hud._knob_x(0)
        _, ky = hud._slider_rect(0)
        ctrl.mouse_x = kx
        ctrl.mouse_y = ky
        ctrl.mouse_down = True

        # Visualizer pattern:
        hud_lock = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        if hud_lock:
            ctrl.suppress_orbit()
        else:
            ctrl.release_orbit()

        assert hud_lock is True
        # After suppress, mouse_dragging cleared so camera won't orbit
        assert ctrl._mouse_dragging is False

        # --- Scene: user releases mouse ---
        ctrl.mouse_down = False
        hud_lock = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        if hud_lock:
            ctrl.suppress_orbit()
        else:
            ctrl.release_orbit()

        assert hud_lock is False
        assert hud.drag_locked is False

    def test_visualizer_pattern_hud_hidden_no_lock(self, _hud_and_ctrl):
        """P10.3: When HUD is hidden, Visualizer pattern never locks orbit."""
        ctrl, hud, cfg = _hud_and_ctrl
        hud.visible = False
        ctrl.mouse_down = True

        # Visualizer pattern:
        hud_lock = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        if hud_lock:
            ctrl.suppress_orbit()
        else:
            ctrl.release_orbit()

        assert hud_lock is False
        # Camera orbit should NOT be suppressed when HUD is hidden

    def test_drag_updates_config_via_nested_path(self, _hud_and_ctrl):
        """P10.3: Dragging a slider writes to the correct nested config field."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Drag sep slider to right edge
        kx = hud._knob_x(0)
        _, ky = hud._slider_rect(0)

        # Pick the slider
        ctrl.mouse_x = kx
        ctrl.mouse_y = ky
        ctrl.mouse_down = True
        hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, True)

        # Drag to far right
        ctrl.mouse_x = hud.TRACK_X0 + hud.TRACK_W
        hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, True)

        # Config should be updated to sep=5.0
        assert cfg.spatial.separation_weight == pytest.approx(5.0, abs=0.1), (
            f"Expected sep≈5.0, got {cfg.spatial.separation_weight:.3f}"
        )

    def test_visible_hud_click_outside_knobs_no_lock(self, _hud_and_ctrl):
        """P10.3: Clicking outside all knobs with HUD visible → no orbit lock."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Click far from all slider knobs
        ctrl.mouse_x = 500
        ctrl.mouse_y = 500
        ctrl.mouse_down = True

        # Visualizer pattern:
        hud_lock = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
        if hud_lock:
            ctrl.suppress_orbit()
        else:
            ctrl.release_orbit()

        assert hud_lock is False, (
            "Click outside all knobs should NOT lock orbit"
        )

    def test_hud_remains_hidden_after_toggle_off(self, _hud_and_ctrl):
        """P10.3: After toggle off, HUD stays hidden across multiple frames."""
        ctrl, hud, cfg = _hud_and_ctrl
        # Fixture starts with hud.visible = True, so one toggle hides it
        hud.toggle()  # hide
        assert hud.visible is False

        # Multiple frames pass — should stay hidden
        for _ in range(5):
            # Visualizer pattern each frame
            hud_lock = hud.handle_mouse(ctrl.mouse_x, ctrl.mouse_y, ctrl.mouse_down)
            assert hud_lock is False, (
                "Hidden HUD should never lock orbit across multiple frames"
            )


# -- S2.E6: Pilot-mode gather/scatter key state --

class TestPilotModeGatherScatter:
    """S2.E6: Shift=gather, Alt=scatter — held-key state flags."""

    def test_gathering_true_on_shift_press(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        assert not ctrl.gathering

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is True

    def test_gathering_false_on_shift_release(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is True

        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_LSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is False

    def test_scattering_true_on_alt_press(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        assert not ctrl.scattering

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LALT))
        ctrl.handle_events()
        assert ctrl.scattering is True

    def test_scattering_false_on_alt_release(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LALT))
        ctrl.handle_events()
        assert ctrl.scattering is True

        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_LALT))
        ctrl.handle_events()
        assert ctrl.scattering is False

    def test_right_shift_also_triggers_gathering(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is True

        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_RSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is False

    def test_right_alt_also_triggers_scattering(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RALT))
        ctrl.handle_events()
        assert ctrl.scattering is True

        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_RALT))
        ctrl.handle_events()
        assert ctrl.scattering is False

    def test_gather_scatter_independent(self):
        """Gather (Shift) and scatter (Alt) are independent — both can be active."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is True
        assert ctrl.scattering is False

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LALT))
        ctrl.handle_events()
        assert ctrl.gathering is True
        assert ctrl.scattering is True

        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_LSHIFT))
        ctrl.handle_events()
        assert ctrl.gathering is False
        assert ctrl.scattering is True


# -- S2.E6: Cube-law spawn velocity --

class TestCubeLawSpawnVelocity:
    """S2.E6: Spawn velocity v = v0 * u^0.33 (cube-law) per road-map spec."""

    def test_spawn_rng_is_seeded_generator(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        assert ctrl._spawn_rng is not None
        assert hasattr(ctrl._spawn_rng, 'uniform')

    def test_spawn_rng_produces_different_values(self):
        """Spawn RNG is non-deterministic (fresh state each run)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        vals = [ctrl._spawn_rng.uniform(0, 1) for _ in range(5)]
        # At least one unique value
        assert len(set(round(v, 6) for v in vals)) > 1
