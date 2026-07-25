"""Unit tests for viz.hud.SliderHUD — knob_x all-sliders parametrized, knob_x float precision, knob hit-test boundary/endpoints, toggle config-preservation, nested config-path writes.

Pure unit tests — no GPU or pygame dependency needed.
SliderHUD communicates solely through SimConfig.

Split out of test_hud.py (file-size split).
"""


import pytest

from pymurmur.core.config import SimConfig
from pymurmur.viz.hud import SLIDERS, SliderDef, SliderHUD


class TestKnobXAllSliders:
    """P10.3: Parametrized _knob_x endpoints and midpoint for every slider."""

    # (idx, label, low, high, default, section, field)
    _SLIDER_PARAMS: list[tuple[int, str, float, float, float, str, str]] = [
        (0, "sep", 1.0, 5.0, 3.0, "spatial", "separation_weight"),
        (1, "coh", 0.0, 2.0, 0.2, "spatial", "cohesion_weight"),
        (2, "align", 0.0, 0.5, 0.02, "spatial", "alignment_weight"),
        (3, "avoid", 0.0, 1.0, 0.05, "boundary", "boundary_avoidance_factor"),
        (4, "noise", 0.0, 0.5, 0.05, "spatial", "noise_scale"),
    ]

    @pytest.mark.parametrize("idx,label,low,high,default,section,field", _SLIDER_PARAMS)
    def test_low_endpoint(self, idx, label, low, high, default, section, field):
        """Every slider at its low value → knob at left edge (TRACK_X0)."""
        cfg = SimConfig()
        sec = getattr(cfg, section)
        setattr(sec, field, low)
        hud = SliderHUD(cfg)
        assert hud._knob_x(idx) == hud.TRACK_X0, (
            f"{label} at low={low}: expected {hud.TRACK_X0}, got {hud._knob_x(idx)}"
        )

    @pytest.mark.parametrize("idx,label,low,high,default,section,field", _SLIDER_PARAMS)
    def test_high_endpoint(self, idx, label, low, high, default, section, field):
        """Every slider at its high value → knob at right edge."""
        cfg = SimConfig()
        sec = getattr(cfg, section)
        setattr(sec, field, high)
        hud = SliderHUD(cfg)
        expected = hud.TRACK_X0 + hud.TRACK_W
        assert hud._knob_x(idx) == expected, (
            f"{label} at high={high}: expected {expected}, got {hud._knob_x(idx)}"
        )

    @pytest.mark.parametrize("idx,label,low,high,default,section,field", _SLIDER_PARAMS)
    def test_midpoint(self, idx, label, low, high, default, section, field):
        """Every slider at its midpoint → knob at track centre."""
        cfg = SimConfig()
        sec = getattr(cfg, section)
        mid = (low + high) / 2.0
        setattr(sec, field, mid)
        hud = SliderHUD(cfg)
        expected = hud.TRACK_X0 + int(0.5 * hud.TRACK_W)
        assert hud._knob_x(idx) == expected, (
            f"{label} at midpoint={mid}: expected {expected}, got {hud._knob_x(idx)}"
        )

    def test_zero_range_clamps_to_left(self, monkeypatch):
        """Degenerate slider (low==high) → t=0.0 → left edge."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        # Build a copy of SLIDERS with a degenerate slider at index 0
        modified = list(SLIDERS)
        modified[0] = SliderDef("deg", 5.0, 5.0, 5.0, ("spatial", "separation_weight"))
        monkeypatch.setattr("pymurmur.viz.hud.SLIDERS", modified)
        cfg.spatial.separation_weight = 5.0
        assert hud._knob_x(0) == hud.TRACK_X0, (
            "Zero-range slider should always return left edge"
        )


# ── Value mapping: _knob_x float → int precision ──────────────

class TestKnobXPrecision:
    """P10.3: _knob_x precision — float t clamped, then truncated to int."""

    def test_one_third_position(self):
        """sep at 1/3 of range [1.0, 5.0] → t≈0.333 → int truncation."""
        cfg = SimConfig()
        # value = low + (1/3)*(high-low) = 1.0 + 1.333... = 2.333...
        cfg.spatial.separation_weight = 1.0 + (1.0 / 3.0) * 4.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        # t = 1/3, so knob at TRACK_X0 + int(TRACK_W/3)
        expected = hud.TRACK_X0 + int(hud.TRACK_W / 3.0)
        assert kx == expected, (
            f"1/3 position: expected ~{expected}, got {kx}"
        )

    def test_near_zero_but_positive(self):
        """Very small t → int truncates to 0 fractional → still at TRACK_X0."""
        cfg = SimConfig()
        # coh: [0.0, 2.0], set to 0.001 → t=0.0005 → int=0
        cfg.spatial.cohesion_weight = 0.001
        hud = SliderHUD(cfg)
        assert hud._knob_x(1) == hud.TRACK_X0, (
            f"tiny coh: expected left edge, got {hud._knob_x(1)}"
        )

    def test_near_one_but_not_quite(self):
        """t just under 1.0 → int truncates to TRACK_W-1 fractional."""
        cfg = SimConfig()
        # coh: [0.0, 2.0], set to 1.999 → t=0.9995 → still < 1.0
        cfg.spatial.cohesion_weight = 1.999
        hud = SliderHUD(cfg)
        kx = hud._knob_x(1)
        # Should NOT reach the full right edge (t < 1.0)
        assert kx < hud.TRACK_X0 + hud.TRACK_W, (
            f"t<1.0 should not reach right edge, got {kx}"
        )


# ── Knob hit-test: boundary precision ─────────────────────────

class TestKnobHitBoundary:
    """P10.3: _knob_hit — exact hit-rect boundary at KNOB_R+4."""

    @pytest.fixture
    def _hud_at_mid(self) -> tuple[SliderHUD, int, int]:
        """HUD with sep slider at midpoint (knob centred)."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        return hud, kx, cy

    def test_exact_boundary_x(self, _hud_at_mid):
        """Pixel at exactly KNOB_R+4 on x-axis → hit (boundary inclusive)."""
        hud, kx, cy = _hud_at_mid
        r = hud.KNOB_R + 4  # = 10
        assert hud._knob_hit(0, kx + r, cy) is True, (
            f"KNOB_R+4={r} should be within hit rect (inclusive boundary)"
        )
        assert hud._knob_hit(0, kx - r, cy) is True

    def test_exact_boundary_y(self, _hud_at_mid):
        """Pixel at exactly KNOB_R+4 on y-axis → hit (boundary inclusive)."""
        hud, kx, cy = _hud_at_mid
        r = hud.KNOB_R + 4
        assert hud._knob_hit(0, kx, cy + r) is True
        assert hud._knob_hit(0, kx, cy - r) is True

    def test_one_pixel_beyond_boundary_x(self, _hud_at_mid):
        """Pixel at KNOB_R+5 on x-axis → miss."""
        hud, kx, cy = _hud_at_mid
        r = hud.KNOB_R + 5  # = 11
        assert hud._knob_hit(0, kx + r, cy) is False
        assert hud._knob_hit(0, kx - r, cy) is False

    def test_diagonal_corner_within_both_axes(self, _hud_at_mid):
        """Diagonal at (KNOB_R, KNOB_R) is within both axes → hit."""
        hud, kx, cy = _hud_at_mid
        assert hud._knob_hit(0, kx + hud.KNOB_R, cy + hud.KNOB_R) is True

    def test_diagonal_corner_beyond_x_axis(self, _hud_at_mid):
        """Diagonal at (KNOB_R+5, KNOB_R): x beyond, y within → miss (axis-aligned)."""
        hud, kx, cy = _hud_at_mid
        # x is beyond but y is within — still a miss because hit rect is axis-aligned
        assert hud._knob_hit(0, kx + hud.KNOB_R + 5, cy + hud.KNOB_R) is False, (
            "Axis-aligned hit rect: x beyond → miss even if y within"
        )


# ── Knob hit-test: endpoint knob positions ────────────────────

class TestKnobHitAtEndpoints:
    """P10.3: _knob_hit works when knob is at left/right edge of track."""

    def test_hit_at_leftmost_knob(self):
        """Knob at left edge (sep=1.0): click at TRACK_X0 should hit."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 1.0  # left edge
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)  # should be TRACK_X0
        _, cy = hud._slider_rect(0)
        assert kx == hud.TRACK_X0
        assert hud._knob_hit(0, kx, cy) is True

    def test_hit_at_rightmost_knob(self):
        """Knob at right edge (sep=5.0): click at right edge should hit."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 5.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)  # should be TRACK_X0 + TRACK_W
        _, cy = hud._slider_rect(0)
        assert kx == hud.TRACK_X0 + hud.TRACK_W
        assert hud._knob_hit(0, kx, cy) is True

    def test_leftmost_knob_extends_hit_rect(self):
        """Left-edge knob: hit rect extends KNOB_R+4 pixels left of TRACK_X0."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 1.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        # The hit rect extends left of the track start
        left_bound = kx - hud.KNOB_R - 4
        assert hud._knob_hit(0, left_bound, cy) is True, (
            f"Hit rect should extend to {left_bound} (left of TRACK_X0={hud.TRACK_X0})"
        )


# ── TAB toggle: config value preservation ────────────────────

class TestToggleExtended:
    """P10.3: TAB toggle — config integrity, rapid toggles, drag abort."""

    def test_toggle_preserves_config_values(self):
        """Toggling visibility does not change any config fields."""
        cfg = SimConfig()
        # Set known values
        cfg.spatial.separation_weight = 2.5
        cfg.boundary.boundary_avoidance_factor = 0.3
        hud = SliderHUD(cfg)

        hud.toggle()  # show
        hud.toggle()  # hide
        hud.toggle()  # show
        hud.toggle()  # hide

        assert cfg.spatial.separation_weight == pytest.approx(2.5)
        assert cfg.boundary.boundary_avoidance_factor == pytest.approx(0.3)

    def test_rapid_toggle_sequence(self):
        """Multiple rapid toggles: state stays consistent (not stuck)."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)

        # Toggle 10 times rapidly
        for _i in range(10):
            hud.toggle()

        # After even count, should be hidden
        assert hud.visible is False
        assert hud.drag_locked is False
        assert hud._active_slider == -1

        # One more toggle → visible
        hud.toggle()
        assert hud.visible is True

    def test_toggle_while_dragging_aborts_drag(self):
        """Toggling off while mid-drag resets drag state and active slider."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0
        hud = SliderHUD(cfg)
        hud.visible = True
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)

        # Start a drag
        hud.handle_mouse(kx0, cy0, mouse_down=True)
        assert hud._active_slider == 0
        assert hud.drag_locked is True

        # Toggle off mid-drag
        hud.toggle()
        assert hud.visible is False
        assert hud.drag_locked is False
        assert hud._active_slider == -1

        # Value should not be corrupted from the interrupted drag
        # (drag didn't move far, so value should still be near 3.0)
        assert cfg.spatial.separation_weight == pytest.approx(3.0, abs=0.1)

    def test_toggle_while_dragging_mouse_ignored_when_hidden(self):
        """After toggle-off while dragging, further mouse events ignored."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        hud._active_slider = 0
        hud.drag_locked = True

        # Toggle off
        hud.toggle()

        # Mouse still down but widget is hidden — should be ignored
        result = hud.handle_mouse(200, 200, mouse_down=True)
        assert result is False


# ── Config field writes: nested path verification ──────────────

class TestSetValueNestedPath:
    """P10.3: _set_value writes through the nested config path correctly."""

    @pytest.fixture
    def _hud(self) -> SliderHUD:
        cfg = SimConfig()
        return SliderHUD(cfg)

    def test_write_visible_via_section_accessor(self, _hud):
        """After _set_value on sep slider, cfg.spatial.separation_weight reflects it."""
        cfg = _hud._config
        mid_px = _hud.TRACK_X0 + _hud.TRACK_W // 2
        _hud._set_value(0, mid_px)  # sep: [1.0, 5.0] → ~3.0
        # Read via nested sub-config accessor
        assert cfg.spatial.separation_weight == pytest.approx(3.0, abs=0.1)

    def test_write_visible_via_flat_access(self, _hud):
        """After _set_value, flat attribute access returns same value."""
        cfg = _hud._config
        _hud._set_value(0, _hud.TRACK_X0)  # sep low
        # Flat access reads through __getattr__ delegation
        assert cfg.separation_weight == pytest.approx(1.0)

    def test_slider_isolation(self, _hud):
        """Writing one slider does not affect other sliders' config fields."""
        cfg = _hud._config

        # Record initial values for all 5 sliders
        def _snap():
            return [
                cfg.spatial.separation_weight,
                cfg.spatial.cohesion_weight,
                cfg.spatial.alignment_weight,
                cfg.boundary.boundary_avoidance_factor,
                cfg.spatial.noise_scale,
            ]

        before = _snap()

        # Write sep slider (idx 0) to high endpoint
        _hud._set_value(0, _hud.TRACK_X0 + _hud.TRACK_W)

        after = _snap()
        # Only sep (index 0) should change
        assert after[0] != before[0], "sep should have changed"
        for i in (1, 2, 3, 4):
            assert after[i] == pytest.approx(before[i]), (
                f"Slider {i} value changed when only slider 0 was written"
            )

    def test_boundary_section_independent(self, _hud):
        """Writing avoid slider (boundary section) doesn't touch spatial section."""
        cfg = _hud._config
        old_sep = cfg.spatial.separation_weight

        _hud._set_value(3, _hud.TRACK_X0 + _hud.TRACK_W)  # avoid to high

        assert cfg.boundary.boundary_avoidance_factor == pytest.approx(1.0)
        assert cfg.spatial.separation_weight == pytest.approx(old_sep), (
            "Writing avoid slider should not affect spatial.separation_weight"
        )

    def test_all_sliders_write_to_distinct_fields(self, _hud):
        """Each slider's _set_value writes to the correct config field."""
        cfg = _hud._config
        written_fields: dict[str, float] = {}

        for i in range(5):
            sd = SLIDERS[i]
            section_name, field_name = sd.config_path
            section = getattr(cfg, section_name)

            # Write to a position that maps to a known, distinct value
            # Use left edge so value = low (well-defined and no roundtrip ambiguity)
            _hud._set_value(i, _hud.TRACK_X0)
            written_fields[sd.label] = getattr(section, field_name)

        # Each slider should have written to its low endpoint
        assert written_fields["sep"] == pytest.approx(1.0), (
            f"sep should be 1.0, got {written_fields['sep']}"
        )
        assert written_fields["coh"] == pytest.approx(0.0)
        assert written_fields["align"] == pytest.approx(0.0)
        assert written_fields["avoid"] == pytest.approx(0.0)
        assert written_fields["noise"] == pytest.approx(0.0)


