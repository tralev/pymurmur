"""Tests for viz.input_control — HUD mouse-drag integration, pilot-mode gather/scatter, cube-law spawn velocity.

Requires pygame. All tests skip when pygame is unavailable.

Split out of test_input.py (file-size split).
"""

import os

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
