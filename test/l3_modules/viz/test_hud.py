"""Unit tests for viz.hud.SliderHUD — value mapping, hit-test, toggle, config writes.

Pure unit tests — no GPU or pygame dependency needed.
SliderHUD communicates solely through SimConfig.
"""

from unittest.mock import MagicMock

import pytest

from pymurmur.core.config import SimConfig
from pymurmur.viz.hud import SLIDERS, SliderDef, SliderHUD


class TestSliderDef:
    """SliderDef dataclass and SLIDERS table."""

    def test_all_five_sliders_defined(self):
        """P10.3: Exactly 5 sliders: sep, coh, align, avoid, noise."""
        assert len(SLIDERS) == 5
        labels = [s.label for s in SLIDERS]
        assert labels == ["sep", "coh", "align", "avoid", "noise"]

    def test_sliders_have_valid_ranges(self):
        """All sliders have low < high and default within range."""
        for sd in SLIDERS:
            assert sd.low < sd.high, f"{sd.label}: low={sd.low} >= high={sd.high}"
            assert sd.low <= sd.default <= sd.high, (
                f"{sd.label}: default={sd.default} not in [{sd.low}, {sd.high}]"
            )

    def test_slider_config_paths_exist(self):
        """Every slider's config_path (section, field) exists on SimConfig."""
        cfg = SimConfig()
        for sd in SLIDERS:
            section_name, field_name = sd.config_path
            section = getattr(cfg, section_name)
            assert hasattr(section, field_name), (
                f"Slider '{sd.label}': {section_name}.{field_name} not found"
            )


class TestSliderHUDInit:
    """SliderHUD initialisation and defaults."""

    def test_init_visible_false(self):
        """HUD starts hidden."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        assert hud.visible is False

    def test_init_drag_not_locked(self):
        """HUD starts with drag_locked=False."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        assert hud.drag_locked is False

    def test_init_no_active_or_hover(self):
        """No slider is active or hovered at init."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        assert hud._active_slider == -1
        assert hud._hover_slider == -1


# ── Value mapping: _knob_x (config value → pixel X) ────────────

class TestKnobX:
    """P10.3: _knob_x maps config value to pixel position on track."""

    def test_separation_default_midpoint(self):
        """sep default=3.0 in [1.0, 5.0] → t=(3−1)/(5−1)=0.5 → mid-track."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0
        hud = SliderHUD(cfg)
        # idx 0 = "sep"
        kx = hud._knob_x(0)
        expected = hud.TRACK_X0 + int(0.5 * hud.TRACK_W)
        assert kx == expected, f"sep default=3.0 should be mid-track, got {kx}"

    def test_separation_low_endpoint(self):
        """sep=1.0 (low) → knob at left edge of track."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 1.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        assert kx == hud.TRACK_X0, (
            f"sep=1.0 (low) should be at TRACK_X0={hud.TRACK_X0}, got {kx}"
        )

    def test_separation_high_endpoint(self):
        """sep=5.0 (high) → knob at right edge of track."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 5.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        expected = hud.TRACK_X0 + hud.TRACK_W
        assert kx == expected, (
            f"sep=5.0 (high) should be at right edge {expected}, got {kx}"
        )

    def test_below_low_clamped(self):
        """Value below low → clamped to low (left edge)."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = -10.0  # below low=1.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        assert kx == hud.TRACK_X0, f"below-low should clamp to left, got {kx}"

    def test_above_high_clamped(self):
        """Value above high → clamped to high (right edge)."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 99.0  # above high=5.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        expected = hud.TRACK_X0 + hud.TRACK_W
        assert kx == expected, f"above-high should clamp to right, got {kx}"

    def test_cohesion_zero_endpoint(self):
        """coh=0.0 (low) → left edge for slider 1 (coh)."""
        cfg = SimConfig()
        cfg.spatial.cohesion_weight = 0.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(1)  # idx 1 = "coh"
        assert kx == hud.TRACK_X0, f"coh=0.0 should be at left edge, got {kx}"

    def test_cohesion_high_endpoint(self):
        """coh=2.0 (high) → right edge."""
        cfg = SimConfig()
        cfg.spatial.cohesion_weight = 2.0
        hud = SliderHUD(cfg)
        kx = hud._knob_x(1)
        expected = hud.TRACK_X0 + hud.TRACK_W
        assert kx == expected

    def test_avoidance_slider_boundary_config(self):
        """avoid slider (idx 3) reads from boundary.avoidance_factor."""
        cfg = SimConfig()
        cfg.boundary.boundary_avoidance_factor = 0.0
        hud = SliderHUD(cfg)
        kx_low = hud._knob_x(3)  # idx 3 = "avoid"
        assert kx_low == hud.TRACK_X0

        cfg.boundary.boundary_avoidance_factor = 1.0
        kx_high = hud._knob_x(3)
        expected = hud.TRACK_X0 + hud.TRACK_W
        assert kx_high == expected

    def test_all_sliders_different_rows(self):
        """Each slider has a different Y centre (row spacing)."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        y_positions = set()
        for i in range(len(SLIDERS)):
            _, cy = hud._slider_rect(i)
            y_positions.add(cy)
        assert len(y_positions) == 5, "All 5 sliders should have unique Y positions"


# ── Value mapping: _set_value (pixel X → config value) ─────────

class TestSetValue:
    """P10.3: _set_value writes from pixel position to the correct config field."""

    def test_set_midpoint_writes_correct_value(self):
        """Middle of track → (low+high)/2."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        mid_px = hud.TRACK_X0 + hud.TRACK_W // 2
        hud._set_value(0, mid_px)  # sep slider: [1.0, 5.0]
        # Midpoint ≈ 3.0
        assert 2.5 <= cfg.spatial.separation_weight <= 3.5, (
            f"Midpoint should be near 3.0, got {cfg.spatial.separation_weight:.3f}"
        )

    def test_set_left_edge_writes_low(self):
        """Left edge → low value."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0)  # sep: low=1.0
        assert cfg.spatial.separation_weight == pytest.approx(1.0)

    def test_set_right_edge_writes_high(self):
        """Right edge → high value."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W)  # sep: high=5.0
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

    def test_set_beyond_left_clamped(self):
        """Pixel beyond left edge → clamped to low."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0 - 100)
        assert cfg.spatial.separation_weight == pytest.approx(1.0)

    def test_set_beyond_right_clamped(self):
        """Pixel beyond right edge → clamped to high."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud._set_value(0, hud.TRACK_X0 + hud.TRACK_W + 100)
        assert cfg.spatial.separation_weight == pytest.approx(5.0)

    def test_set_writes_to_correct_config_field(self):
        """Each slider writes to its declared config_path."""
        cfg = SimConfig()

        # Set all to their low endpoint
        hud = SliderHUD(cfg)
        for i in range(len(SLIDERS)):
            hud._set_value(i, hud.TRACK_X0)

        # Verify each field was written
        assert cfg.spatial.separation_weight == pytest.approx(1.0)   # sep: [1.0, 5.0]
        assert cfg.spatial.cohesion_weight == pytest.approx(0.0)     # coh: [0.0, 2.0]
        assert cfg.spatial.alignment_weight == pytest.approx(0.0)    # align: [0.0, 0.5]
        assert cfg.boundary.boundary_avoidance_factor == pytest.approx(0.0)  # avoid: [0.0, 1.0]
        assert cfg.spatial.noise_scale == pytest.approx(0.0)         # noise: [0.0, 0.5]

        # Set all to their high endpoint
        for i in range(len(SLIDERS)):
            hud._set_value(i, hud.TRACK_X0 + hud.TRACK_W)

        assert cfg.spatial.separation_weight == pytest.approx(5.0)
        assert cfg.spatial.cohesion_weight == pytest.approx(2.0)
        assert cfg.spatial.alignment_weight == pytest.approx(0.5)
        assert cfg.boundary.boundary_avoidance_factor == pytest.approx(1.0)
        assert cfg.spatial.noise_scale == pytest.approx(0.5)

    def test_set_value_roundtrip(self):
        """_knob_x(_set_value(x)) → x for various positions on track."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        # Test at several positions across the track for sep slider
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            px = hud.TRACK_X0 + int(fraction * hud.TRACK_W)
            hud._set_value(0, px)
            kx = hud._knob_x(0)
            # Allow ±1 pixel due to integer truncation
            assert abs(kx - px) <= 1, (
                f"Roundtrip at fraction={fraction}: set({px}) → knob_x={kx}"
            )


# ── Knob hit-test precision ────────────────────────────────────

class TestKnobHit:
    """P10.3: _knob_hit — hit-rect ±(KNOB_R+4) px from knob centre."""

    def test_hit_at_knob_centre(self):
        """Pixel exactly at knob centre → hit."""
        cfg = SimConfig()
        cfg.spatial.separation_weight = 3.0  # mid-track
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        assert hud._knob_hit(0, kx, cy) is True

    def test_hit_within_knob_bounds(self):
        """Pixel within KNOB_R pixels of centre → hit."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        # Just inside the hit rect
        assert hud._knob_hit(0, kx + hud.KNOB_R, cy) is True
        assert hud._knob_hit(0, kx, cy + hud.KNOB_R) is True

    def test_hit_at_hit_rect_boundary(self):
        """Pixel at KNOB_R+3 (within the +4 padding) → hit."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        # +4 padding means KNOB_R+4 is the boundary
        assert hud._knob_hit(0, kx + hud.KNOB_R + 3, cy) is True, (
            "KNOB_R+3 should be within hit rect (padding=+4)"
        )

    def test_miss_outside_hit_rect(self):
        """Pixel beyond KNOB_R+4 → miss."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        # Outside the hit rect (KNOB_R+4 is 10, so +11 is a miss)
        assert hud._knob_hit(0, kx + hud.KNOB_R + 5, cy) is False, (
            "KNOB_R+5 should be outside hit rect"
        )
        assert hud._knob_hit(0, kx, cy + hud.KNOB_R + 5) is False

    def test_hit_corners(self):
        """Diagonal corner test: pixel at knob corner."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        kx = hud._knob_x(0)
        _, cy = hud._slider_rect(0)
        # Corner at (KNOB_R, KNOB_R) within the 6+4=10 hit box
        assert hud._knob_hit(0, kx + hud.KNOB_R, cy + hud.KNOB_R) is True

    def test_hit_different_slider_id(self):
        """Hit test respects slider index — click on slider 0 does not hit slider 1."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        # Hover over slider 0's knob
        kx0 = hud._knob_x(0)
        _, cy0 = hud._slider_rect(0)
        # Should NOT hit slider 1 at the same coordinates
        assert hud._knob_hit(0, kx0, cy0) is True
        assert hud._knob_hit(1, kx0, cy0) is False, (
            "Slider 0 click should not hit slider 1"
        )


# ── TAB toggle visibility ──────────────────────────────────────

class TestToggle:
    """P10.3: TAB toggles visible state and resets drag/active."""

    def test_toggle_visible_on(self):
        """toggle() flips visible from False → True."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        assert hud.visible is False
        hud.toggle()
        assert hud.visible is True

    def test_toggle_visible_off(self):
        """toggle() flips visible from True → False."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.toggle()  # on
        hud.toggle()  # off
        assert hud.visible is False

    def test_toggle_off_resets_drag(self):
        """Toggling off resets drag_locked and active slider."""
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.toggle()  # show
        # Simulate drag state
        hud._active_slider = 2
        hud.drag_locked = True
        hud.toggle()  # hide
        assert hud.drag_locked is False
        assert hud._active_slider == -1

    def test_toggle_off_resets_active_and_drag_only(self):
        """Toggling off resets _active_slider and drag_locked but NOT _hover_slider.

        _hover_slider is managed by handle_mouse(), not toggle().
        """
        cfg = SimConfig()
        hud = SliderHUD(cfg)
        hud.visible = True
        hud._active_slider = 2
        hud.drag_locked = True
        hud._hover_slider = 3
        hud.toggle()  # off
        assert hud.drag_locked is False
        assert hud._active_slider == -1
        assert hud._hover_slider == 3, (
            "_hover_slider is NOT cleared by toggle — handle_mouse manages it"
        )


# ── Mouse handling ─────────────────────────────────────────────

# ── Value mapping: all 5 sliders parametrized ────────────────

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


# ── Mouse handling: edge cases ────────────────────────────────

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
