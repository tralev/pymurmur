"""TrailRenderer's accumulation and lines trail-mode mixin.

Extracted from trails.py (file-size split). _TrailExtraModesMixin holds
the two heavier trail modes' drawing methods -- kept as a mixin (not
free functions) since every method reads shared GL-context/FBO/buffer
state set up by TrailRenderer.__init__, mirroring the existing
Renderer3D(_RendererVAOMixin, _RendererDrawMixin) split in viz/renderer.py.
velocity/ring trail modes (simpler, smaller) stay on TrailRenderer itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..physics.flock import PhysicsFlock


class _TrailExtraModesMixin:
    """Accumulation and lines trail-mode methods, mixed into TrailRenderer."""

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

