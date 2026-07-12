"""Tests for viz.input_control — keyboard/mouse → SimConfig bridge.

Requires pygame. All tests skip when pygame is unavailable.
"""

import os
from pathlib import Path

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
        """K_UP increases config.phi_p by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.phi_p
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_UP
        ))
        ctrl.handle_events()
        assert cfg.phi_p > old

    def test_input_down_phi_p(self):
        """K_DOWN decreases config.phi_p by 0.01."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        old = cfg.phi_p
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN
        ))
        ctrl.handle_events()
        assert cfg.phi_p < old

    def test_input_phi_p_clamped(self):
        """phi_p never goes below 0 or above 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Set near zero and try to go below
        cfg.phi_p = 0.0
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_DOWN
        ))
        ctrl.handle_events()
        assert cfg.phi_p >= 0.0

        # Set near one and try to go above
        cfg.phi_p = 1.0
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_UP
        ))
        ctrl.handle_events()
        assert cfg.phi_p <= 1.0

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
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1))
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
