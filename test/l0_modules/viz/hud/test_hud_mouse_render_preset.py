"""Unit tests for viz.hud.SliderHUD — mouse handling edge cases, draw_hud_rect render calls, panel hit-test spawn suppression, preset HUD integration.

Pure unit tests — no GPU or pygame dependency needed.
SliderHUD communicates solely through SimConfig.

Split out of test_hud.py (file-size split).
"""

from unittest.mock import MagicMock

import pytest

from pymurmur.core.config import SimConfig
from pymurmur.viz.hud import SLIDERS, SliderHUD


class TestHandleMouse:
    """P10.3: handle_mouse — drag, hover, orbit suppression."""

    def test_handle_mouse_ignores_when_hidden(self):
        """handle_mouse returns False when HUD is not visible."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        assert hud.visible is False
        result = hud.handle_mouse(100, 100, mouse_down=True)
        assert result is False

    def test_hover_updates_on_mouse_move(self):
        """Moving mouse over a knob sets _hover_slider when not dragging."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        # Hover over slider 0 knob (mouse not down)
        hud.handle_mouse(kx0, cy0, mouse_down=False)
        assert hud._hover_slider == 0

    def test_hover_clears_when_away(self):
        """Moving mouse away clears _hover_slider."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        hud._hover_slider = 0
        # Move far away from all sliders
        hud.handle_mouse(-100, -100, mouse_down=False)
        assert hud._hover_slider == -1

    def test_drag_picks_slider(self):
        """Mouse down on knob → picks that slider, sets drag_locked."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        result = hud.handle_mouse(kx0, cy0, mouse_down=True)
        assert result is True, "Drag on knob should return True (suppress orbit)"
        assert hud._active_slider == 0
        assert hud.drag_locked is True

    def test_drag_updates_value(self):
        """Dragging knob to different positions updates the config field."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # default mid
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        # Pick slider
        hud.handle_mouse(kx0, cy0, mouse_down=True)

        # Drag to far right
        hud.handle_mouse(hud.TRACK_X0 + hud.TRACK_W, cy0, mouse_down=True)
        assert cfg.spatial.separation_weight == pytest.approx(5.0, abs=0.1)

    def test_drag_returns_true_while_active(self):
        """Continued drag on active slider returns True."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        # Pick slider
        hud.handle_mouse(kx0, cy0, mouse_down=True)
        # Drag left
        result = hud.handle_mouse(hud.TRACK_X0, cy0, mouse_down=True)
        assert result is True

    def test_mouse_up_releases_slider(self):
        """Mouse up releases active slider and drag lock."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        hud._active_slider = 1
        hud.drag_locked = True

        hud.handle_mouse(100, 100, mouse_down=False)
        assert hud._active_slider == -1
        assert hud.drag_locked is False

    def test_click_outside_no_pick(self):
        """Clicking outside all knobs doesn't pick any slider."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        # Click far from all sliders
        result = hud.handle_mouse(-100, -100, mouse_down=True)
        assert result is False
        assert hud._active_slider == -1
        assert hud.drag_locked is False

    def test_drag_continues_same_slider(self):
        """Once a slider is picked, dragging stays on that slider even if mouse moves away."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        # Pick slider 0
        hud.handle_mouse(kx0, cy0, mouse_down=True)
        assert hud._active_slider == 0

        # Move to slider 1's row but at a DIFFERENT x-position (still on slider 0)
        _, cy1 = hud._slider_rect(1)
        hud.handle_mouse(hud.TRACK_X0 + hud.TRACK_W, cy1, mouse_down=True)
        assert hud._active_slider == 0, "Should stay on slider 0 even at slider 1's row"
        # Value should be updated to slider 0's high endpoint (sep=5.0)
        assert cfg.spatial.separation_weight == pytest.approx(5.0, abs=0.1), (
            f"Value should be updated via slider 0's range, got {cfg.spatial.separation_weight:.3f}"
        )


# ── Render: draw_hud_rect calls — coordinates, colours, visibility ─

class TestRender:
    """P10.3: SliderHUD.render() — verify renderer.draw_hud_rect() calls."""

    @pytest.fixture
    def _hud_and_mock(self) -> tuple[SliderHUD, MagicMock]:
        """SliderHUD with mock renderer, visible, sep at midpoint."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track for sep knob
        hud = SliderHUD(cfg)
        hud.visible = True
        mock = MagicMock()
        return hud, mock

    # ── Visibility gating ──────────────────────────────────

    def test_render_hidden_does_nothing(self, _hud_and_mock):
        """When visible=False, render() makes no draw_hud_rect calls."""
        hud, mock = _hud_and_mock
        hud.visible = False
        hud.render(mock, 0, 0)
        mock.draw_hud_rect.assert_not_called()

    # ── Call count ─────────────────────────────────────────

    def test_render_fifteen_rects_total(self, _hud_and_mock):
        """5 sliders × 3 rects (track + knob + label) = 15 calls."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        assert mock.draw_hud_rect.call_count == 15, (
            f"Expected 15 calls (5×3), got {mock.draw_hud_rect.call_count}"
        )

    # ── Track bar rects ────────────────────────────────────

    def test_track_bar_coordinates(self, _hud_and_mock):
        """Each track bar rect has correct (x, y, w, h)."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        # Track bars are calls 0, 3, 6, 9, 12 (first of each slider's 3 rects)
        for i in range(5):
            call_args = mock.draw_hud_rect.call_args_list[i * 3][0]
            expected_x = hud.TRACK_X0
            expected_y = hud.Y0 + i * hud.ROW_H - 2
            expected_w = hud.TRACK_W
            expected_h = 4
            assert call_args == (
                expected_x, expected_y, expected_w, expected_h, hud.TRACK_COLOUR,
            ), f"Slider {i} track: expected ({expected_x},{expected_y},{expected_w},{expected_h}), got {call_args[:4]}"

    def test_track_bar_colour(self, _hud_and_mock):
        """All track bars use TRACK_COLOUR."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        for i in range(5):
            colour = mock.draw_hud_rect.call_args_list[i * 3][0][4]
            assert colour == hud.TRACK_COLOUR, (
                f"Slider {i} track: expected {hud.TRACK_COLOUR}, got {colour}"
            )

    # ── Knob rects ─────────────────────────────────────────

    def test_knob_coordinates(self, _hud_and_mock):
        """Each knob rect is centered at the correct position."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        # Knobs are calls 1, 4, 7, 10, 13 (second of each slider's 3 rects)
        for i in range(5):
            call_args = mock.draw_hud_rect.call_args_list[i * 3 + 1][0]
            kx = hud._knob_x(i)
            _, cy = hud._slider_rect(i)
            expected_x = kx - hud.KNOB_R
            expected_y = cy - hud.KNOB_R
            expected_w = hud.KNOB_R * 2
            expected_h = hud.KNOB_R * 2
            assert call_args[:4] == (
                expected_x, expected_y, expected_w, expected_h,
            ), f"Slider {i} knob: expected ({expected_x},{expected_y},{expected_w},{expected_h}), got {call_args[:4]}"

    def test_knob_default_colour_cold(self, _hud_and_mock):
        """Default knob uses KNOB_COLOUR when not hovered or active."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        # No slider is hovered or active → all knobs use KNOB_COLOUR
        for i in range(5):
            colour = mock.draw_hud_rect.call_args_list[i * 3 + 1][0][4]
            assert colour == hud.KNOB_COLOUR, (
                f"Slider {i} knob: expected KNOB_COLOUR, got {colour}"
            )

    def test_knob_hot_colour_when_hovered(self, _hud_and_mock):
        """Hovered slider knob uses KNOB_HOT_COLOUR."""
        hud, mock = _hud_and_mock
        hud._hover_slider = 2  # hover over "align" slider
        hud.render(mock, 0, 0)

        # Slider 2 (align) knob should be hot
        colour_2 = mock.draw_hud_rect.call_args_list[2 * 3 + 1][0][4]
        assert colour_2 == hud.KNOB_HOT_COLOUR, (
            f"Hovered slider: expected KNOB_HOT_COLOUR, got {colour_2}"
        )
        # Slider 0 (sep) knob should be cold
        colour_0 = mock.draw_hud_rect.call_args_list[0 * 3 + 1][0][4]
        assert colour_0 == hud.KNOB_COLOUR, (
            f"Non-hovered slider: expected KNOB_COLOUR, got {colour_0}"
        )

    def test_knob_hot_colour_when_active(self, _hud_and_mock):
        """Active (dragging) slider knob uses KNOB_HOT_COLOUR."""
        hud, mock = _hud_and_mock
        hud._active_slider = 3  # dragging "avoid" slider
        hud.render(mock, 0, 0)

        # Slider 3 (avoid) knob should be hot
        colour_3 = mock.draw_hud_rect.call_args_list[3 * 3 + 1][0][4]
        assert colour_3 == hud.KNOB_HOT_COLOUR, (
            f"Active slider: expected KNOB_HOT_COLOUR, got {colour_3}"
        )

    # ── Label ticks ────────────────────────────────────────

    def test_label_tick_coordinates(self, _hud_and_mock):
        """Each label tick is a small 8×2 rect at the left edge."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        # Labels are calls 2, 5, 8, 11, 14 (third of each slider's 3 rects)
        for i in range(5):
            call_args = mock.draw_hud_rect.call_args_list[i * 3 + 2][0]
            _, cy = hud._slider_rect(i)
            expected_x = hud.X0
            expected_y = cy - 1
            expected_w = 8
            expected_h = 2
            assert call_args == (
                expected_x, expected_y, expected_w, expected_h, hud.LABEL_COLOUR,
            ), f"Slider {i} label: expected ({expected_x},{expected_y},{expected_w},{expected_h}), got {call_args[:4]}"

    def test_label_tick_colour(self, _hud_and_mock):
        """All label ticks use LABEL_COLOUR."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        for i in range(5):
            colour = mock.draw_hud_rect.call_args_list[i * 3 + 2][0][4]
            assert colour == hud.LABEL_COLOUR, (
                f"Slider {i} label: expected LABEL_COLOUR, got {colour}"
            )

    # ── Row layout ─────────────────────────────────────────

    def test_slider_rows_increasing_y(self, _hud_and_mock):
        """Slider Y positions increment by ROW_H each row."""
        hud, mock = _hud_and_mock
        hud.render(mock, 0, 0)
        y_positions = []
        for i in range(5):
            # Track bar Y (first call per slider)
            y = mock.draw_hud_rect.call_args_list[i * 3][0][1]
            y_positions.append(y)
        # Each subsequent slider should be ROW_H pixels lower
        for i in range(1, 5):
            assert y_positions[i] == y_positions[0] + i * hud.ROW_H, (
                f"Slider {i} Y={y_positions[i]}, expected {y_positions[0] + i * hud.ROW_H}"
            )

    def test_hot_colour_takes_priority_active_over_hover(self, _hud_and_mock):
        """When both active and hover are set, KNOB_HOT_COLOUR is used (no conflict)."""
        hud, mock = _hud_and_mock
        # Set active on slider 0 and hover on slider 0 simultaneously
        hud._active_slider = 0
        hud._hover_slider = 0
        hud.render(mock, 0, 0)
        # Slider 0 knob should be hot (both flags set → hot is True)
        colour = mock.draw_hud_rect.call_args_list[0 * 3 + 1][0][4]
        assert colour == hud.KNOB_HOT_COLOUR, (
            "Active + hover: knob should be hot"
        )

    def test_different_active_and_hover_both_hot(self, _hud_and_mock):
        """When active and hover are on different sliders, both are hot."""
        hud, mock = _hud_and_mock
        hud._active_slider = 0  # dragging sep
        hud._hover_slider = 2  # hovering over align
        hud.render(mock, 0, 0)
        # Both slider 0 and slider 2 knobs should be hot
        colour_0 = mock.draw_hud_rect.call_args_list[0 * 3 + 1][0][4]
        colour_2 = mock.draw_hud_rect.call_args_list[2 * 3 + 1][0][4]
        assert colour_0 == hud.KNOB_HOT_COLOUR, "Active slider 0 should be hot"
        assert colour_2 == hud.KNOB_HOT_COLOUR, "Hovered slider 2 should be hot"
        # Slider 1 (neither) should be cold
        colour_1 = mock.draw_hud_rect.call_args_list[1 * 3 + 1][0][4]
        assert colour_1 == hud.KNOB_COLOUR, "Slider 1 should be cold"

    def test_render_passes_mouse_coords_through(self, _hud_and_mock):
        """render() accepts mx, my but draw_hud_rect doesn't use them."""
        hud, mock = _hud_and_mock
        # mx, my are passed to render but currently unused (hover is pre-computed)
        hud.render(mock, 999, 888)
        # Should still render all 15 rects
        assert mock.draw_hud_rect.call_count == 15


# ── Panel hit-test: spawn suppression bounding box ─────────────

class TestHitTestAny:
    """P10.3: hit_test_any — full HUD panel bounding box for spawn suppression."""

    @pytest.fixture
    def _hud(self) -> SliderHUD:
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track
        hud = SliderHUD(cfg)
        hud.visible = True
        return hud

    def test_hidden_returns_false(self, _hud):
        """When HUD is hidden, hit_test_any always returns False."""
        _hud.visible = False
        # The panel centre should be inside bounds when visible
        cx = _hud.TRACK_X0 + _hud.TRACK_W // 2
        cy = _hud.Y0 + 2 * _hud.ROW_H
        assert _hud.hit_test_any(cx, cy) is False

    def test_visible_inside_panel_returns_true(self, _hud):
        """Mouse inside the full panel bounding box → True."""
        # Centre of the panel
        cx = _hud.TRACK_X0 + _hud.TRACK_W // 2
        cy = _hud.Y0 + 2 * _hud.ROW_H
        assert _hud.hit_test_any(cx, cy) is True

    def test_visible_near_label_area_returns_true(self, _hud):
        """Mouse near the left label area (X0=16, with padding) → True."""
        # X=16 is inside the padded panel (left=X0-6=10)
        assert _hud.hit_test_any(16, _hud.Y0) is True

    def test_visible_outside_left_returns_false(self, _hud):
        """Mouse beyond the left padding boundary → False."""
        # left = X0 - 6 = 10, so X=4 should be outside
        left = _hud.X0 - 7
        assert _hud.hit_test_any(left, _hud.Y0) is False

    def test_visible_outside_right_returns_false(self, _hud):
        """Mouse beyond the right padding boundary → False."""
        # right = TRACK_X0 + TRACK_W + KNOB_R + 6
        right = _hud.TRACK_X0 + _hud.TRACK_W + _hud.KNOB_R + 7
        assert _hud.hit_test_any(right, _hud.Y0) is False

    def test_visible_above_panel_returns_false(self, _hud):
        """Mouse above the padded top boundary → False."""
        # top = Y0 - KNOB_R - 6
        above = _hud.Y0 - _hud.KNOB_R - 7
        assert _hud.hit_test_any(_hud.TRACK_X0, above) is False

    def test_visible_below_panel_returns_false(self, _hud):
        """Mouse below the padded bottom boundary → False."""
        n = len(SLIDERS)
        # bottom = Y0 + (n-1)*ROW_H + KNOB_R + 6
        below = _hud.Y0 + (n - 1) * _hud.ROW_H + _hud.KNOB_R + 7
        assert _hud.hit_test_any(_hud.TRACK_X0, below) is False

    def test_visible_corner_inside_bounds(self, _hud):
        """Mouse at the top-left corner of the panel → True."""
        # Panel left edge with padding
        left = _hud.X0 - 5  # inside left=10
        top = _hud.Y0 - _hud.KNOB_R - 5  # inside top
        assert _hud.hit_test_any(left, top) is True

    def test_bounding_box_is_attribute_independent(self, _hud):
        """hit_test_any uses layout constants, not config-dependent knob positions."""
        # Change config value — knob_x changes, but panel bounds shouldn't
        _hud._config.spatial.separation_weight = 1.0  # knob moves to left
        # Panel centre should still be inside
        cx = _hud.TRACK_X0 + _hud.TRACK_W // 2
        cy = _hud.Y0 + 2 * _hud.ROW_H
        assert _hud.hit_test_any(cx, cy) is True

    def test_all_visible_knobs_within_bounds(self, _hud):
        """Every slider knob centre is inside the panel bounding box."""
        for i in range(len(SLIDERS)):
            kx = _hud._knob_x(i)
            _, cy = _hud._slider_rect(i)
            assert _hud.hit_test_any(kx, cy) is True, (
                f"Slider {i} knob at ({kx}, {cy}) should be inside panel"
            )


# Cross-cutting: P10.1 + P10.3 — preset changes reflected in HUD knobs

class TestPresetHUDIntegration:
    """P10.1 + P10.3 cross-cutting: applying a letter preset updates HUD knobs."""

    def test_preset_changes_config_knob_moves(self):
        """P10.1->P10.3: Applying a spatial preset changes config value;
        HUD knob position reflects the new value."""
        from pymurmur.analysis.presets import apply_preset
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track
        hud = SliderHUD(cfg)

        # Before preset: sep at mid
        mid_before = hud._knob_x(0)
        assert mid_before == hud.TRACK_X0 + int(0.5 * hud.TRACK_W)

        # Apply preset 'h' (3D Void) — sets separation_weight to 0.35
        apply_preset(cfg, "h")
        # 0.35 is below slider low=1.0, so clamped to left edge
        assert hud._knob_x(0) == hud.TRACK_X0
        assert hud._knob_x(0) != mid_before, "Knob moved from mid to left after preset"

    def test_hud_knobs_update_after_config_mutation(self):
        """P10.3: _knob_x reads config live; changing config moves the knob."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track
        hud = SliderHUD(cfg)
        mid = hud._knob_x(0)

        # Change config externally (simulating a preset or CLI --set)
        cfg.spatial.separation_weight = 5.0
        assert hud._knob_x(0) == hud.TRACK_X0 + hud.TRACK_W
        assert hud._knob_x(0) != mid

    def test_clear_birds_hud_still_renders(self):
        """P10.4 + P10.3: After clearing all birds, HUD still renders without crash."""
        cfg = SimConfig()
        cfg.num_boids = 10
        hud = SliderHUD(cfg)
        hud.visible = True
        from unittest.mock import MagicMock
        mock = MagicMock()

        # Simulate clear — num_boids goes to 0 via config mutation
        cfg.num_boids = 0

        # HUD should still render normally (no crash, 15 rects)
        hud.render(mock, 0, 0)
        assert mock.draw_hud_rect.call_count == 15
