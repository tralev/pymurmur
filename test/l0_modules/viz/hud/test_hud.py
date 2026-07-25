"""Unit tests for viz.hud.SliderHUD — SliderDef/SliderHUD init, _knob_x value mapping, _set_value, knob hit-test, TAB toggle.

Pure unit tests — no GPU or pygame dependency needed.
SliderHUD communicates solely through SimConfig.

Split out of test_hud.py (file-size split).
"""


import pytest

from pymurmur.core.config import SimConfig
from pymurmur.viz.hud import SLIDERS, SliderHUD


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


