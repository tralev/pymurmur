"""Slider HUD — orthographic overlays for live parameter tuning.

P10.3: 5 sliders rendered as ortho-pass track + knob quads.
TAB toggles panel visibility; drag locks suppress camera orbit.

Level 2 — never imports simulation. Communicates solely through SimConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig


@dataclass
class SliderDef:
    """Definition of a single slider knob."""
    label: str
    low: float
    high: float
    default: float
    config_path: tuple[str, str]  # (section_name, field_name)


# P10.3: 5 sliders with endpoints and defaults per spec table
SLIDERS: list[SliderDef] = [
    SliderDef("sep", 1.0, 5.0, 3.0, ("spatial", "separation_weight")),
    SliderDef("coh", 0.0, 2.0, 0.2, ("spatial", "cohesion_weight")),
    SliderDef("align", 0.0, 0.5, 0.02, ("spatial", "alignment_weight")),
    SliderDef("avoid", 0.0, 1.0, 0.05, ("boundary", "boundary_avoidance_factor")),
    SliderDef("noise", 0.0, 0.5, 0.05, ("spatial", "noise_scale")),
]


class SliderHUD:
    """P10.3: Ortho-pass slider panel for live parameter tuning.

    Renders 5 horizontal sliders as a track bar (full-width rectangle)
    + knob (small filled rectangle) at the current value position.

    Mouse drag on a knob updates the corresponding config field.
    Drag on any slider suppresses the orbit camera (via _drag_locked).
    TAB toggles panel visibility.

    Usage in the Visualizer main loop:
        hud = SliderHUD(config)
        ...
        if hud.visible:
            hud.render(renderer, mx, my, mouse_down)
            if hud.drag_locked:
                input_ctrl._mouse_dragging = False  # suppress orbit
    """

    # Layout constants (in pixel coordinates, orthographic)
    X0: int = 16          # left edge of track labels
    TRACK_X0: int = 70    # left edge of track bar
    TRACK_W: int = 140    # track bar width
    KNOB_R: int = 6       # half-width of the knob rectangle
    ROW_H: int = 22       # vertical spacing between sliders
    Y0: int = 20          # top of first slider
    LABEL_COLOUR: tuple[float, float, float] = (0.9, 0.9, 0.9)
    TRACK_COLOUR: tuple[float, float, float] = (0.25, 0.25, 0.28)
    KNOB_COLOUR: tuple[float, float, float] = (0.6, 0.75, 0.95)
    KNOB_HOT_COLOUR: tuple[float, float, float] = (0.85, 0.85, 1.0)

    def __init__(self, config: SimConfig) -> None:
        self._config = config
        self.visible: bool = False
        self.drag_locked: bool = False
        self._active_slider: int = -1  # index into SLIDERS, or -1
        self._hover_slider: int = -1

    def _slider_rect(self, idx: int) -> tuple[int, int]:
        """Return the screen-Y centre of the slider row."""
        return self.TRACK_X0, self.Y0 + idx * self.ROW_H

    def _knob_x(self, idx: int) -> int:
        """Compute knob pixel-X from the current config value."""
        sd = SLIDERS[idx]
        section_name, field_name = sd.config_path
        section = getattr(self._config, section_name)
        value = getattr(section, field_name)
        t = max(0.0, min(1.0, (value - sd.low) / (sd.high - sd.low))) if sd.high > sd.low else 0.0
        return int(self.TRACK_X0 + t * self.TRACK_W)

    def _set_value(self, idx: int, pixel_x: int) -> None:
        """Set the config field from a pixel-X position on the track."""
        sd = SLIDERS[idx]
        t = max(0.0, min(1.0, (pixel_x - self.TRACK_X0) / self.TRACK_W))
        value = sd.low + t * (sd.high - sd.low)
        section_name, field_name = sd.config_path
        section = getattr(self._config, section_name)
        setattr(section, field_name, value)

    def _knob_hit(self, idx: int, mx: int, my: int) -> bool:
        """Check if pixel (mx, my) is within the knob rectangle."""
        _, cy = self._slider_rect(idx)
        kx = self._knob_x(idx)
        return (abs(mx - kx) <= self.KNOB_R + 4 and abs(my - cy) <= self.KNOB_R + 4)

    def handle_mouse(self, mx: int, my: int, mouse_down: bool) -> bool:
        """Process mouse input. Returns True if orbiting should be suppressed.

        Args:
            mx, my: mouse pixel coords (viewport space, y=0 top).
            mouse_down: True while left mouse button is held.
        """
        if not self.visible:
            return False

        if not mouse_down:
            self._active_slider = -1
            self.drag_locked = False
            # Update hover state
            self._hover_slider = -1
            for i in range(len(SLIDERS)):
                if self._knob_hit(i, mx, my):
                    self._hover_slider = i
                    break
            return False

        # Mouse is down — check for active drag or new pick
        if self._active_slider >= 0:
            self._set_value(self._active_slider, mx)
            self.drag_locked = True
            return True

        # Try to pick a slider
        for i in range(len(SLIDERS)):
            if self._knob_hit(i, mx, my):
                self._active_slider = i
                self._set_value(i, mx)
                self.drag_locked = True
                return True

        return False

    def toggle(self) -> None:
        """TAB key: toggle panel visibility."""
        self.visible = not self.visible
        if not self.visible:
            self._active_slider = -1
            self.drag_locked = False

    def hit_test_any(self, mx: int, my: int) -> bool:
        """P10.3: Check if (mx, my) is within the HUD panel bounds.

        Used by the Visualizer to suppress cursor-ray spawning
        when the user is clicking on the HUD panel.  Uses the
        full bounding box (labels through track bar + padding)
        rather than per-knob hit-tests — any click within the
        panel area suppresses the spawn.
        """
        if not self.visible:
            return False
        n = len(SLIDERS)
        # Panel bounding box with 6 px padding on all sides
        left = self.X0 - 6
        right = self.TRACK_X0 + self.TRACK_W + self.KNOB_R + 6
        top = self.Y0 - self.KNOB_R - 6
        bottom = self.Y0 + (n - 1) * self.ROW_H + self.KNOB_R + 6
        return left <= mx <= right and top <= my <= bottom

    def render(self, renderer, mx: int, my: int) -> None:
        """Draw all slider tracks and knobs via the renderer's ortho pass.

        The caller must already have set up an orthographic projection
        matching the viewport pixel dimensions (viewport_w × viewport_h).
        """
        if not self.visible:
            return

        for i, _sd in enumerate(SLIDERS):
            tx, cy = self._slider_rect(i)
            kx = self._knob_x(i)
            hot = (i == self._hover_slider) or (i == self._active_slider)

            # Track bar — thin horizontal rect
            renderer.draw_hud_rect(
                tx, cy - 2, self.TRACK_W, 4,
                self.TRACK_COLOUR,
            )

            # Knob — small filled square
            knob_col = self.KNOB_HOT_COLOUR if hot else self.KNOB_COLOUR
            renderer.draw_hud_rect(
                kx - self.KNOB_R, cy - self.KNOB_R,
                self.KNOB_R * 2, self.KNOB_R * 2,
                knob_col,
            )

            # Label text — rendered as simple colour indicator (no font)
            # For now, just a small left-edge tick
            renderer.draw_hud_rect(
                self.X0, cy - 1, 8, 2, self.LABEL_COLOUR,
            )
