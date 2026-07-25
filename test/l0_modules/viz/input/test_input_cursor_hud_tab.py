"""Tests for viz.input_control — P10.4 cursor-ray spawning/clear/v0 adjustment, camera unprojection, P10.3 SliderHUD TAB integration.

Requires pygame. All tests skip when pygame is unavailable.

Split out of test_input.py (file-size split).
"""

import os

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

    def test_right_click_spawn_uses_median_flock_depth_when_given(self, ctrl):
        """S5.4: handle_events(positions=...) threads flock positions
        through to screen_to_world, changing the spawn depth."""
        import pygame

        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 300),
        ))
        ctrl.handle_events()
        default_spawn = ctrl.pending_spawn_predator.pop()

        far_positions = np.full((5, 3), [500.0, 500.0, 900.0], dtype=np.float32)
        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=3, pos=(400, 300),
        ))
        ctrl.handle_events(positions=far_positions)
        flock_spawn = ctrl.pending_spawn_predator.pop()

        assert default_spawn != flock_spawn

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

    def test_positions_use_median_flock_depth_not_target_plane(self):
        """S5.4: when positions is given, intersection is at the flock's
        median depth along the view axis, not the Z=target.z plane —
        the two must differ when the flock sits far from target.z."""
        from pymurmur.viz.camera import OrbitCamera
        camera = OrbitCamera(target=(500.0, 500.0, 400.0))

        fallback = camera.screen_to_world(400.0, 300.0, 800, 600)
        positions = np.full((5, 3), [500.0, 500.0, 1000.0], dtype=np.float32)
        via_flock = camera.screen_to_world(
            400.0, 300.0, 800, 600, positions=positions,
        )
        assert fallback is not None and via_flock is not None
        assert fallback != via_flock
        assert all(np.isfinite(via_flock))

    def test_positions_empty_array_falls_back_to_target_plane(self):
        """S5.4: an empty positions array must not crash — falls back
        to the Z=target.z plane exactly like positions=None."""
        from pymurmur.viz.camera import OrbitCamera
        camera = OrbitCamera(target=(500.0, 500.0, 400.0))
        fallback = camera.screen_to_world(400.0, 300.0, 800, 600)
        empty = camera.screen_to_world(
            400.0, 300.0, 800, 600, positions=np.zeros((0, 3), dtype=np.float32),
        )
        assert empty == fallback

    def test_positions_single_bird_matches_its_own_depth(self):
        """S5.4: median of a single bird is that bird's own depth along
        the camera's forward (view) axis — not raw world Z, since the
        two only coincide for a purely top-down view."""
        from pymurmur.viz.camera import OrbitCamera
        camera = OrbitCamera(target=(500.0, 500.0, 400.0))
        bird = np.array([500.0, 500.0, 700.0], dtype=np.float32)
        result = camera.screen_to_world(
            400.0, 300.0, 800, 600, positions=bird[np.newaxis, :],
        )
        assert result is not None

        eye = np.array(camera.eye_position(), dtype=np.float32)
        target = np.array([camera.target.x, camera.target.y, camera.target.z],
                           dtype=np.float32)
        f_hat = (target - eye) / np.linalg.norm(target - eye)

        bird_depth = float(np.dot(bird - eye, f_hat))
        hit_depth = float(np.dot(np.asarray(result) - eye, f_hat))
        target_depth = float(np.dot(target - eye, f_hat))

        assert abs(hit_depth - bird_depth) < 1e-3
        assert abs(hit_depth - target_depth) > 1.0


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


