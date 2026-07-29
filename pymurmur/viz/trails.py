"""P8.3 — Trail rendering: 4 modes for motion-afterimage effects.

Level 2 — owns GPU state for trail rendering. Receives the ModernGL
context and theme from Renderer3D, reads position/velocity data from
PhysicsFlock.

Modes:
  "off"          — no trails (default)
  "velocity"     — line segments stretched along velocity, fade at tips
  "ring"         — K past positions from flock.position_history as fading dots
  "accumulation" — screen-space FBO persistence (additive blend, slow decay)
  "lines"        — CPU sinusoidal ribbon polylines traced backward

Config: viz.trails, viz.trail_length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .trails_modes import _TrailExtraModesMixin

if TYPE_CHECKING:
    from ..physics.flock import PhysicsFlock
    from .camera import OrbitCamera

# ── Accumulation shaders (fullscreen-quad pass-through + alpha) ──

_ACCUM_VERTEX_SHADER = """#version 330 core

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_ACCUM_FRAGMENT_SHADER = """#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_accum_tex;
uniform float u_decay;  // 0.96–0.99: blend previous frame down

void main() {
    vec4 prev = texture(u_accum_tex, v_uv);
    frag_color = prev * u_decay;
}
"""

# Fullscreen quad with UVs for accumulation pass
_ACCUM_QUAD = np.array([
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 1.0,
], dtype=np.float32)

_ACCUM_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


class TrailRenderer(_TrailExtraModesMixin):
    """GPU trail rendering — velocity lines, ring-history sprites,
    screen-space accumulation, and CPU ribbon lines.

    Created by Renderer3D during __init__.  The caller is responsible
    for calling :meth:`begin_frame` (once per frame, after camera uniforms
    are set) and :meth:`draw` (after bird rendering).

    Accumulation/lines mode methods (_ensure_accum_fbo/_draw_accumulation/
    blit_accumulation/_draw_lines) are provided by _TrailExtraModesMixin
    (file-size split — see .trails_modes).
    """

    def __init__(
        self,
        ctx,
        mode: str = "off",
        trail_length: int = 30,
        theme: dict | None = None,
    ) -> None:
        self._ctx = ctx
        self._mode = mode
        self._trail_length = max(1, trail_length)
        self._theme = theme or {}

        from .shaders import (
            RING_VERTEX_SHADER,
            TRAIL_FRAGMENT_SHADER,
            TRAIL_VERTEX_SHADER,
        )
        # Velocity trail program (instanced: pos + vel per bird)
        self._prog = ctx.program(
            vertex_shader=TRAIL_VERTEX_SHADER,
            fragment_shader=TRAIL_FRAGMENT_SHADER,
        )
        # Ring trail program (pass-through: world-space positions, no instance data)
        self._ring_prog = ctx.program(
            vertex_shader=RING_VERTEX_SHADER,
            fragment_shader=TRAIL_FRAGMENT_SHADER,
        )

        # Velocity trail buffer — 2 vertices per line segment for
        # up to 100K trail segments (reallocated on demand).
        self._velocity_capacity = 100000
        self._velocity_vbo = ctx.buffer(
            reserve=self._velocity_capacity * 2 * 3 * 4  # 2 verts × 3f × 4 bytes
        )
        self._velocity_vao: Any = None
        self._velocity_count: int = 0

        # Ring trail buffer — reallocated on demand
        self._ring_capacity = 50000
        self._ring_vbo = ctx.buffer(
            reserve=self._ring_capacity * 3 * 4  # 3f per point × 4 bytes
        )
        self._ring_vao: Any = None
        self._ring_count: int = 0

        # ── Accumulation mode — persistent FBO + fullscreen quad ──
        self._accum_prog = ctx.program(
            vertex_shader=_ACCUM_VERTEX_SHADER,
            fragment_shader=_ACCUM_FRAGMENT_SHADER,
        )
        accum_vbo = ctx.buffer(_ACCUM_QUAD.tobytes())
        accum_ibo = ctx.buffer(_ACCUM_INDICES.tobytes())
        self._accum_vao = ctx.vertex_array(
            self._accum_prog,
            [(accum_vbo, "2f 2f", "in_position", "in_uv")],
            accum_ibo,
        )
        self._accum_fbo: Any = None     # created on first use (lazy, needs size)
        self._accum_tex: Any = None
        self._accum_decay: float = 0.97            # blend factor per frame

        # ── Lines mode — CPU ribbon buffer ───────────────────────
        self._lines_capacity = 100000   # vertices
        self._lines_vbo = ctx.buffer(
            reserve=self._lines_capacity * 3 * 4
        )
        self._lines_vao: Any = None
        self._lines_count: int = 0
        # Lines vertex shader is shared with ring (pass-through world-space positions)
        # but rendered as LINE_STRIP instead of POINTS.

    # ── Mode toggle ────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("off", "velocity", "ring", "accumulation", "lines"):
            raise ValueError(f"Unknown trail mode: {value}")
        self._mode = value

    @property
    def trail_length(self) -> int:
        return self._trail_length

    @trail_length.setter
    def trail_length(self, value: int) -> None:
        self._trail_length = max(1, value)

    # ── Frame lifecycle ────────────────────────────────────────

    def begin_frame(
        self,
        camera: OrbitCamera,
        aspect: float,
    ) -> None:
        """Set camera uniforms on both trail shader programs."""
        from .renderer import _mat4_bytes

        trail_color = self._theme.get("trail", (0.5, 0.5, 0.5))
        view_bytes = _mat4_bytes(camera.view_matrix())
        proj_bytes = _mat4_bytes(camera.projection_matrix(aspect))
        color_bytes = np.array(trail_color, dtype=np.float32).tobytes()

        # Velocity program
        self._prog["u_view"].write(view_bytes)
        self._prog["u_projection"].write(proj_bytes)
        self._prog["u_trail_length"] = float(self._trail_length)
        self._prog["u_trail_color"].write(color_bytes)

        # Ring program (also used by lines mode for projection)
        self._ring_prog["u_view"].write(view_bytes)
        self._ring_prog["u_projection"].write(proj_bytes)
        self._ring_prog["u_trail_color"].write(color_bytes)

    def push_history(self, flock: PhysicsFlock) -> None:
        """Record current positions into the position_history ring buffer.

        Automatically initialises ``flock.position_history`` if needed.
        Must be called once per frame BEFORE draw() so that ring trails
        reflect the current frame.
        """
        if self._mode != "ring":
            return

        # Lazy-init history buffer on first call
        if flock.position_history is None:
            self.ensure_history(flock)
        assert flock.position_history is not None  # narrow for mypy after ensure_history

        # Roll buffer: shift all entries back, write current to slot 0
        flock.position_history[:, 1:, :] = flock.position_history[:, :-1, :]
        flock.position_history[:, 0, :] = flock.positions

    def ensure_history(self, flock: PhysicsFlock) -> None:
        """Initialise position_history on the flock if not already set.

        Safe to call every frame — no-op if already initialised with the
        correct shape.
        """
        N = flock.N_capacity
        if flock.position_history is None:
            flock.position_history = np.zeros(
                (N, self._trail_length, 3), dtype=np.float32,
            )
            # Seed with current positions so first frame shows something
            flock.position_history[:, :, :] = flock.positions[:, np.newaxis, :]

    def draw(
        self,
        flock: PhysicsFlock,
        instance_vbo,
        instance_count: int,
    ) -> None:
        """Render trails for the current mode.

        Args:
            flock: PhysicsFlock with position/velocity data.
            instance_vbo: Renderer3D's shared 8-float InstanceSchema VBO
                (pos.xyz vel.xyz hue scale per bird, D7) — velocity mode
                reads pos+vel from it with a padded format string;
                accumulation mode doesn't touch it (reads flock.positions
                directly instead).
            instance_count: Number of active instances.
        """
        if self._mode == "off" or instance_count == 0:
            return

        if self._mode == "velocity":
            self._draw_velocity(flock, instance_vbo, instance_count)
        elif self._mode == "ring":
            self._draw_ring(flock, instance_count)
        elif self._mode == "accumulation":
            self._draw_accumulation(flock, instance_vbo, instance_count)
        elif self._mode == "lines":
            self._draw_lines(flock, instance_count)

    # ── Velocity trail — line segments along velocity ──────────

    def _draw_velocity(
        self,
        flock: PhysicsFlock,
        instance_vbo,
        instance_count: int,
    ) -> None:
        """Render velocity-stretched line segments.

        Each active bird gets a 2-vertex line: head at current position,
        tail offset backward along velocity by trail_length * 0.12.
        """
        import moderngl

        active_idx = np.where(flock.active)[0][:instance_count]
        n = len(active_idx)
        if n == 0:
            return

        if n > self._velocity_capacity:
            self._velocity_capacity = n + 50000
            self._velocity_vbo = self._ctx.buffer(
                reserve=self._velocity_capacity * 2 * 3 * 4
            )
            self._velocity_vao = None

        # Build 2 vertices per bird: head (x=0) and tail (x=-1)
        verts = np.zeros((n * 2, 3), dtype=np.float32)
        verts[0::2, 0] = 0.0    # head: no stretch
        verts[1::2, 0] = -1.0   # tail: full negative stretch

        self._velocity_vbo.write(verts.tobytes())

        if self._velocity_vao is None:
            self._velocity_vao = self._ctx.vertex_array(
                self._prog,
                [
                    (self._velocity_vbo, "3f", "in_position"),
                    # D7: instance_vbo is Renderer3D's shared 8-float
                    # InstanceSchema buffer (pos.xyz vel.xyz hue scale),
                    # not a dedicated 6-float pos+vel buffer — "3f 3f/i"
                    # alone would compute a 24-byte stride against a
                    # true 32-byte one, misaligning every instance after
                    # the first. "8x" pads/skips the trailing hue+scale
                    # floats this shader doesn't use.
                    (instance_vbo, "3f 3f 8x/i", "in_bird_pos", "in_bird_vel"),
                ],
            )

        # Draw line pairs: 2 vertices × N birds
        self._velocity_vao.render(moderngl.LINES, vertices=n * 2, instances=n)

    # ── Ring trail — past positions as fading dots ─────────────

    def _draw_ring(
        self,
        flock: PhysicsFlock,
        instance_count: int,
    ) -> None:
        """Render past positions from position_history as fading point sprites.

        Each history slot gets a progressively smaller alpha, creating a
        comet-tail effect behind each bird.
        """
        import moderngl

        if flock.position_history is None:
            return

        active_idx = np.where(flock.active)[0][:instance_count]
        n = len(active_idx)
        if n == 0:
            return

        K = min(flock.position_history.shape[1], self._trail_length)
        total = n * K

        if total > self._ring_capacity:
            self._ring_capacity = total + 50000
            self._ring_vbo = self._ctx.buffer(
                reserve=self._ring_capacity * 3 * 4
            )
            self._ring_vao = None

        # Flatten (N, K, 3) → (N*K, 3), newest (slot 0) first
        history = flock.position_history[active_idx, :K, :]
        flat = history.reshape(-1, 3).astype(np.float32)
        self._ring_vbo.write(flat.tobytes())

        if self._ring_vao is None:
            self._ring_vao = self._ctx.vertex_array(
                self._ring_prog,
                [(self._ring_vbo, "3f", "in_position")],
            )

        self._ring_vao.render(moderngl.POINTS, vertices=total)

    # ── Cleanup ────────────────────────────────────────────────

    def release(self) -> None:
        """Release GPU resources."""
        if self._accum_tex is not None:
            self._accum_tex.release()
        if self._accum_fbo is not None:
            self._accum_fbo.release()
        if self._velocity_vbo is not None:
            self._velocity_vbo.release()
        if self._ring_vbo is not None:
            self._ring_vbo.release()
        if self._lines_vbo is not None:
            self._lines_vbo.release()
        self._accum_fbo = None
        self._accum_tex = None
