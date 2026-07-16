"""Keyboard/mouse input handler — translates pygame events to SimConfig mutations.

Level 2 — NEVER imports simulation. Communicates solely through SimConfig.
"""

from __future__ import annotations

import pygame
from typing import TYPE_CHECKING

from ..analysis.presets import PRESETS

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from .camera import OrbitCamera


# Mode cycle order
_MODES = ["projection", "spatial", "field", "vicsek", "influencer"]


class InputControl:
    """Translates pygame events into SimConfig mutations.

    Never imports simulation — all communication goes through the
    shared SimConfig reference and camera object.
    """

    def __init__(self, config: SimConfig, camera: OrbitCamera) -> None:
        self.config = config
        self.camera = camera
        self.paused = False
        self.pending_reset = False
        self.pending_add: int = 0
        self.pending_remove: int = 0
        self.show_grid = False
        self._mouse_dragging = False
        self._last_mouse_pos = (0, 0)

    def handle_events(self) -> bool:
        """Process one frame of pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if not self._handle_keydown(event):
                    return False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left
                    self._mouse_dragging = True
                    self._last_mouse_pos = event.pos

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._mouse_dragging = False

            if event.type == pygame.MOUSEMOTION and self._mouse_dragging:
                dx = event.pos[0] - self._last_mouse_pos[0]
                dy = event.pos[1] - self._last_mouse_pos[1]
                self.camera.rotate(dx, -dy)
                self._last_mouse_pos = event.pos

            if event.type == pygame.MOUSEWHEEL:
                self.camera.zoom(event.y)

        return True

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """Handle key press. Returns False for quit keys."""
        cfg = self.config
        key = event.key

        if key == pygame.K_ESCAPE:
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.pending_reset = True
        elif key == pygame.K_m:
            idx = _MODES.index(cfg.mode)
            cfg.mode = _MODES[(idx + 1) % len(_MODES)]
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif key == pygame.K_v:
            self.camera.reset()
        elif key == pygame.K_o:
            self.camera.auto_rotate = not self.camera.auto_rotate
        elif key == pygame.K_UP:
            cfg.phi_p = min(cfg.phi_p + 0.01, 1.0)
        elif key == pygame.K_DOWN:
            cfg.phi_p = max(cfg.phi_p - 0.01, 0.0)
        elif key == pygame.K_RIGHT:
            cfg.phi_a = min(cfg.phi_a + 0.01, 1.0)
        elif key == pygame.K_LEFT:
            cfg.phi_a = max(cfg.phi_a - 0.01, 0.0)
        elif key == pygame.K_RIGHTBRACKET:
            cfg.sigma = min(cfg.sigma + 1, 20)
        elif key == pygame.K_LEFTBRACKET:
            cfg.sigma = max(cfg.sigma - 1, 1)
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            self.pending_add += 10
        elif key == pygame.K_MINUS:
            self.pending_remove += 10
        elif key == pygame.K_t:
            cfg.predator_enabled = not cfg.predator_enabled
        elif key == pygame.K_k:
            cfg.roosting_enabled = not cfg.roosting_enabled
        elif key == pygame.K_u:
            cfg.refinements = not cfg.refinements
        elif pygame.K_1 <= key <= pygame.K_9:
            self._apply_preset(key - pygame.K_1)

        return True

    def _apply_preset(self, index: int) -> None:
        """Apply named preset by index (0-based).

        Updates active mode's weights without changing the mode itself.
        """
        preset_names = list(PRESETS.keys())
        if index >= len(preset_names):
            return

        name = preset_names[index]
        preset = PRESETS[name]

        # Apply all fields except mode (preserve current mode)
        for field, value in preset.items():
            if field == "mode":
                continue
            if hasattr(self.config, field):
                setattr(self.config, field, value)

        print(f"[preset] {name}")
