"""Keyboard/mouse input handler — translates pygame events to SimConfig mutations.

Level 2 — NEVER imports simulation. Communicates solely through SimConfig.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from ..analysis.presets import LETTER_PRESETS, PRESETS, apply_preset
from ..core.logging import cli_out

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from .camera import OrbitCamera


# Mode cycle order
_MODES = ["projection", "spatial", "field", "vicsek", "influencer"]

# S2.E6: Roll sensitivity (radians per frame at 60fps)
_ROLL_SENSITIVITY: float = 0.05


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
        # P10.4: Cursor-ray spawning
        self.pending_spawn_bird: list[tuple[float, float, float]] = []
        self.pending_spawn_predator: list[tuple[float, float, float]] = []
        self.pending_clear: bool = False
        self.pending_v0_delta: float = 0.0
        self.show_grid = False
        self._mouse_dragging = False
        self._last_mouse_pos = (0, 0)
        self._mouse_down_pos: tuple[int, int] | None = None  # P10.4: click vs drag
        self._viewport_w: int = 800
        self._viewport_h: int = 600
        # P10.3: Mouse position for HUD interaction
        self.mouse_x: int = 0
        self.mouse_y: int = 0
        self.mouse_down: bool = False
        self.hud_visible: bool = False  # P10.3: TAB toggle state
        self.pending_hud_toggle: bool = False
        # S2.E6: Pilot-mode gather/scatter (held keys)
        self.gathering: bool = False
        self.scattering: bool = False
        # S2.E6: Cube-law spawn velocities
        self._spawn_rng: np.random.Generator = np.random.default_rng()

    def suppress_orbit(self) -> None:
        """P10.3: Suppress camera orbit — called by visualizer when HUD is dragging."""
        self._mouse_dragging = False

    def release_orbit(self) -> None:
        """P10.3: Release orbit suppression (no-op; orbit releases on mouseup)."""
        pass

    def set_viewport(self, width: int, height: int) -> None:
        """P10.4: Store viewport dimensions for ray unprojection."""
        self._viewport_w = width
        self._viewport_h = height

    def handle_events(self, positions: np.ndarray | None = None) -> bool:
        """Process one frame of pygame events. Returns False to quit.

        Args:
            positions: (N, 3) active flock positions, for S5.4's
                median-flock-depth spawn-ray intersection. None falls
                back to the Z=target.z plane (see
                ``OrbitCamera.screen_to_world``).
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if not self._handle_keydown(event):
                    return False

            if event.type == pygame.KEYUP:
                self._handle_keyup(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left — start drag
                    self._mouse_dragging = True
                    self._last_mouse_pos = event.pos
                    self._mouse_down_pos = event.pos
                elif event.button == 3:  # P10.4: right-click → spawn predator
                    world = self._unproject_mouse(event.pos, positions)
                    if world is not None:
                        self.pending_spawn_predator.append(world)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # left release
                    self._mouse_dragging = False
                    # P10.4: If no drag (< 5px movement), treat as click → spawn bird
                    if self._mouse_down_pos is not None:
                        dx = event.pos[0] - self._mouse_down_pos[0]
                        dy = event.pos[1] - self._mouse_down_pos[1]
                        if dx * dx + dy * dy < 25:  # < 5px movement
                            world = self._unproject_mouse(event.pos, positions)
                            if world is not None:
                                self.pending_spawn_bird.append(world)
                    self._mouse_down_pos = None

            if event.type == pygame.MOUSEMOTION and self._mouse_dragging:
                dx = event.pos[0] - self._last_mouse_pos[0]
                dy = event.pos[1] - self._last_mouse_pos[1]
                self.camera.rotate(dx, -dy)
                self._last_mouse_pos = event.pos

            if event.type == pygame.MOUSEMOTION:
                # P10.3: Track mouse position for HUD
                self.mouse_x, self.mouse_y = event.pos

            # P10.3: Track mouse button state for HUD
            self.mouse_down = bool(pygame.mouse.get_pressed()[0])

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
        elif key == pygame.K_TAB:
            self.pending_hud_toggle = True
        elif key == pygame.K_v:
            self.camera.reset()
        elif key == pygame.K_o:
            self.camera.auto_rotate = not self.camera.auto_rotate
        elif key == pygame.K_UP:
            cfg.projection.phi_p = min(cfg.projection.phi_p + 0.01, 1.0)
            self._enforce_phi_constraint(cfg, changed="phi_p")
        elif key == pygame.K_DOWN:
            cfg.projection.phi_p = max(cfg.projection.phi_p - 0.01, 0.0)
            self._enforce_phi_constraint(cfg, changed="phi_p")
        elif key == pygame.K_RIGHT:
            cfg.phi_a = min(cfg.phi_a + 0.01, 1.0)
            self._enforce_phi_constraint(cfg, changed="phi_a")
        elif key == pygame.K_LEFT:
            cfg.phi_a = max(cfg.phi_a - 0.01, 0.0)
            self._enforce_phi_constraint(cfg, changed="phi_a")
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
        # P8.8: Ortho/perspective camera presets (keys 7/8/9)
        elif key == pygame.K_7:
            self.camera.set_ortho_top(max(cfg.width, cfg.height, cfg.depth))
        elif key == pygame.K_8:
            self.camera.set_ortho_side(max(cfg.width, cfg.height, cfg.depth))
        elif key == pygame.K_9:
            self.camera.set_perspective()
        # P10.1: Letter-key presets a–f, h, w (g reserved for grid toggle)
        elif key == pygame.K_a:
            self._apply_letter_preset("a")
        elif key == pygame.K_b:
            self._apply_letter_preset("b")
        elif key == pygame.K_c:
            self._apply_letter_preset("c")
        elif key == pygame.K_d:
            self._apply_letter_preset("d")
        elif key == pygame.K_f:
            self._apply_letter_preset("f")
        elif key == pygame.K_h:
            self._apply_letter_preset("h")
        elif key == pygame.K_w:
            self._apply_letter_preset("w")
        elif key == pygame.K_LSHIFT or key == pygame.K_RSHIFT:
            self.gathering = True
        elif key == pygame.K_LALT or key == pygame.K_RALT:
            self.scattering = True
        elif key == pygame.K_x:
            self.pending_clear = True
        # S2.E6: Q/E camera roll (quit via ESC only)
        elif key == pygame.K_q:
            self.camera.roll_camera(_ROLL_SENSITIVITY)
        elif key == pygame.K_e:
            self.camera.roll_camera(-_ROLL_SENSITIVITY)
        elif key == pygame.K_PAGEUP:
            self.pending_v0_delta += 0.1
        elif key == pygame.K_PAGEDOWN:
            self.pending_v0_delta -= 0.1
        elif pygame.K_1 <= key <= pygame.K_9:
            self._apply_preset(key - pygame.K_1)

        return True

    def _apply_letter_preset(self, key: str) -> None:
        """P10.1: Apply a letter-key preset with printed label + description."""
        entry = LETTER_PRESETS.get(key)
        if entry is None:
            return
        label = apply_preset(self.config, key)
        if label is not None:
            _desc = entry[1]
            cli_out(f"[preset] {key} — {label}: {_desc}")
        # P10.6: Enforce φp + φa ≤ 1 after preset applies both values at once.
        self._enforce_phi_after_preset(self.config)

    @staticmethod
    def _enforce_phi_constraint(cfg, *, changed: str) -> None:
        """P10.6: After φp or φa changes, enforce φp + φa ≤ 1.

        If the sum exceeds 1.0, the other parameter is reduced to
        make room: φ_other = 1.0 − φ_changed.
        """
        total = cfg.projection.phi_p + cfg.phi_a
        if total > 1.0:
            if changed == "phi_p":
                cfg.phi_a = 1.0 - cfg.projection.phi_p
            else:
                cfg.projection.phi_p = 1.0 - cfg.phi_a

    @staticmethod
    def _enforce_phi_after_preset(cfg) -> None:
        """P10.6: After a preset applies both φp and φa, enforce φp + φa ≤ 1.

        Unlike the arrow-key path, we don't know which value the user
        intended to change.  When the sum exceeds 1.0, the smaller value
        is reduced (matching the CLI path).  If both are equal, φa is
        reduced — consistent with _enforce_phi_constraint's φp≥φa branch.
        """
        total = cfg.projection.phi_p + cfg.phi_a
        if total > 1.0:
            if cfg.projection.phi_p >= cfg.phi_a:
                cfg.phi_a = max(0.0, 1.0 - cfg.projection.phi_p)
            else:
                cfg.projection.phi_p = max(0.0, 1.0 - cfg.phi_a)

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
            if field == "phi_p":  # nested-only (flat shim retired)
                self.config.projection.phi_p = value
            else:
                setattr(self.config, field, value)

        cli_out(f"[preset] {name}")
        # P10.6: Enforce φp + φa ≤ 1 after preset applies both values.
        self._enforce_phi_after_preset(self.config)

    def _handle_keyup(self, event: pygame.event.Event) -> None:
        """S2.E6: Handle key release — gather/scatter release."""
        key = event.key
        if key == pygame.K_LSHIFT or key == pygame.K_RSHIFT:
            self.gathering = False
        elif key == pygame.K_LALT or key == pygame.K_RALT:
            self.scattering = False

    # ── P10.4: Cursor-ray unprojection helper ──────────────────

    def _unproject_mouse(
        self, pos: tuple[int, int], positions: np.ndarray | None = None
    ) -> tuple[float, float, float] | None:
        """P10.4/S5.4: Convert screen coords to world position via camera ray."""
        return self.camera.screen_to_world(
            float(pos[0]), float(pos[1]),
            self._viewport_w, self._viewport_h,
            positions=positions,
        )
