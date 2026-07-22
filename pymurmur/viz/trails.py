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


class TrailRenderer:
    """GPU trail rendering — velocity lines, ring-history sprites,
    screen-space accumulation, and CPU ribbon lines.

    Created by Renderer3D during __init__.  The caller is responsible
    for calling :meth:`begin_frame` (once per frame, after camera uniforms
    are set) and :meth:`draw` (after bird rendering).
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

    # ── Accumulation — screen-space FBO persistence ────────────

    def _ensure_accum_fbo(self, width: int, height: int) -> None:
        """Lazy-create the accumulation FBO + texture at the given size."""
        import moderngl

        if self._accum_fbo is not None and self._accum_tex is not None:
            # Check if size changed; if so, recreate
            if self._accum_tex.size == (width, height):
                return

        # Release old resources if they exist
        if self._accum_tex is not None:
            self._accum_tex.release()
        if self._accum_fbo is not None:
            self._accum_fbo.release()

        self._accum_tex = self._ctx.texture((width, height), 4)
        self._accum_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._accum_fbo = self._ctx.framebuffer(
            color_attachments=[self._accum_tex]
        )
        # Clear to transparent black on first creation
        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def _draw_accumulation(
        self,
        flock: PhysicsFlock,
        instance_vbo,
        instance_count: int,
    ) -> None:
        """Screen-space accumulation trail.

        Each frame:
        1. Apply decay to the persistent accumulation FBO (fade toward black)
        2. Draw current bird positions as point sprites into the FBO (additive)
        3. Blend the accumulation FBO back into the main framebuffer
        """
        import moderngl

        active_idx = np.where(flock.active)[0][:instance_count]
        n = len(active_idx)
        if n == 0:
            return

        # Get viewport size from context; a zero-sized viewport (no
        # framebuffer bound yet) cannot back a valid FBO — fall back.
        vp = self._ctx.viewport
        _, _, vp_w, vp_h = vp if len(vp) == 4 else (0, 0, 800, 600)
        if vp_w <= 0 or vp_h <= 0:
            vp_w, vp_h = 800, 600
        self._ensure_accum_fbo(vp_w, vp_h)
        assert self._accum_fbo is not None and self._accum_tex is not None

        # Step 1: Decay previous frame — render accumulation texture
        # onto itself with u_decay blending
        self._accum_fbo.use()
        self._accum_tex.use(location=0)
        self._accum_prog["u_decay"] = self._accum_decay
        self._accum_prog["u_accum_tex"] = 0
        self._accum_vao.render(moderngl.TRIANGLES)

        # Step 2: Draw current bird positions as point sprites
        # (additive blending into the accumulation buffer)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        # Reuse the ring VBO approach: upload world-space positions as points
        positions = flock.positions[active_idx]
        pts = positions.astype(np.float32)
        if n > self._ring_capacity:
            self._ring_capacity = n + 50000
            self._ring_vbo = self._ctx.buffer(
                reserve=self._ring_capacity * 3 * 4
            )
            self._ring_vao = None
        self._ring_vbo.write(pts.tobytes())
        if self._ring_vao is None:
            self._ring_vao = self._ctx.vertex_array(
                self._ring_prog,
                [(self._ring_vbo, "3f", "in_position")],
            )
        self._ring_vao.render(moderngl.POINTS, vertices=n)

        self._ctx.disable(moderngl.BLEND)

        # Step 3: The accumulation FBO now contains the persistent trail.
        # The caller (Renderer3D.draw_trails) will handle blending it
        # back into the main framebuffer. We store the texture for later.
        # Renderer3D.draw_trails is responsible for calling a final
        # blit pass after accumulation mode finishes.

    def blit_accumulation(self) -> None:
        """Blit the accumulation texture into the current framebuffer.

        Called by Renderer3D after draw_trails() in accumulation mode.
        Blends the persistent accumulation over the main scene using
        alpha blending.
        """
        import moderngl

        if self._accum_tex is None:
            return

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._ctx.disable(moderngl.DEPTH_TEST)

        self._accum_tex.use(location=0)
        self._accum_prog["u_decay"] = 1.0  # no decay on blit
        self._accum_prog["u_accum_tex"] = 0
        self._accum_vao.render(moderngl.TRIANGLES)

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.BLEND)

    # ── Lines — CPU sinusoidal ribbon polylines ────────────────

    def _draw_lines(
        self,
        flock: PhysicsFlock,
        instance_count: int,
    ) -> None:
        """S4.3: Render CPU-generated sinusoidal ribbon lines.

        Spec layout: 5 segments per bird traced backward along velocity
        (vertex k at ``p - v_hat * trailScale * prog``,
        ``trailScale = 0.1 * trail_length``, ``prog = k/5``); a ribbon
        wave displaces vertices along the camera-plane (XY, z-up)
        perpendicular ``(-v_y, v_x, 0)/sqrt(v_x^2+v_y^2)`` (falling back
        to ``(1,0,0)`` when v_x = v_y = 0) by
        ``sin(prog*2*pi*2.6 + seed) * waveScale * prog^2`` — amplitude
        vanishing at the head (prog=0). One GL_LINES draw of
        ``2*5 = 10`` disjoint vertices per bird (not LINE_STRIP — segment
        k's own pair of endpoints, so adjacent segments don't need to
        share a vertex and every bird's ribbon draws in one call).
        """
        import moderngl

        active_idx = np.where(flock.active)[0][:instance_count]
        n = len(active_idx)
        if n == 0:
            return

        segments = 5
        verts_per_bird = segments * 2  # GL_LINES: 2 endpoints/segment
        total_verts = n * verts_per_bird
        trail_scale = 0.1 * self._trail_length
        wave_scale = 2.0  # world-unit wave amplitude at prog=1 (not spec-pinned)

        if total_verts > self._lines_capacity:
            self._lines_capacity = total_verts + 50000
            self._lines_vbo = self._ctx.buffer(
                reserve=self._lines_capacity * 3 * 4
            )
            self._lines_vao = None

        positions = flock.positions[active_idx].astype(np.float64)
        velocities = flock.velocities[active_idx].astype(np.float64)
        seeds = flock.seeds[active_idx].astype(np.float64)

        speed = np.linalg.norm(velocities, axis=1)
        forward = np.zeros_like(velocities)
        moving = speed > 1e-9
        forward[moving] = velocities[moving] / speed[moving, np.newaxis]
        # Stationary birds (speed ~= 0): forward stays zero -> every
        # segment collapses to `positions` (finite, degenerate ribbon).

        vx, vy = velocities[:, 0], velocities[:, 1]
        speed_xy = np.sqrt(vx * vx + vy * vy)
        perp = np.zeros((n, 3), dtype=np.float64)
        has_xy = speed_xy > 1e-9
        perp[has_xy, 0] = -vy[has_xy] / speed_xy[has_xy]
        perp[has_xy, 1] = vx[has_xy] / speed_xy[has_xy]
        perp[~has_xy, 0] = 1.0  # degenerate vertical-v fallback

        verts = np.zeros((n, verts_per_bird, 3), dtype=np.float64)
        for k in range(segments):
            for j, prog in enumerate((k / segments, (k + 1) / segments)):
                base = positions - forward * (trail_scale * prog)
                wave_amount = np.sin(prog * 2.0 * np.pi * 2.6 + seeds) * wave_scale * (prog ** 2)
                verts[:, 2 * k + j] = base + perp * wave_amount[:, np.newaxis]

        flat = verts.reshape(-1, 3).astype(np.float32)
        self._lines_vbo.write(flat.tobytes())
        self._lines_count = total_verts

        if self._lines_vao is None:
            self._lines_vao = self._ctx.vertex_array(
                self._ring_prog,
                [(self._lines_vbo, "3f", "in_position")],
            )

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._lines_vao.render(moderngl.LINES, vertices=total_verts)
        self._ctx.enable(moderngl.DEPTH_TEST)

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
