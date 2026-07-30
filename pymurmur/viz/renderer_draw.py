"""Renderer3D drawing mixin.

Extracted from renderer.py (file-size split). _RendererDrawMixin holds
the draw_* methods -- kept as a mixin (not free functions) since every
method reads shared GL-context/instance-buffer state set up by
Renderer3D.__init__.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .mesh_registry import MESH_REGISTRY, resolve_bird_mesh

if TYPE_CHECKING:
    from ..physics.flock import PhysicsFlock
    from .trails import TrailRenderer


class _RendererDrawMixin:
    """Drawing methods, mixed into Renderer3D."""

    # Set by Renderer3D.__init__ (renderer.py); declared here for mypy
    # (mirrors _RendererVAOMixin's identical pattern in renderer_vao.py
    # — see that file for the "GPU handles -> Any" rationale).
    ctx: Any
    _prog: Any
    _fbo: Any
    _grid_vao: Any
    _hud_prog: Any
    _hud_vao: Any
    _impostor_vao: Any
    _instance_vbo: Any
    _vao: Any
    _winged_vao: Any
    _mesh_vaos: dict[str, object]
    _mesh_vbos: dict[str, object]
    _mesh_ibos: dict[str, object]
    _bird_mesh: str
    _density_mode: bool
    _point_sprites: bool
    _trails_mode: str
    _winged_mesh: bool
    _trails: "TrailRenderer | None"

    if TYPE_CHECKING:
        # Provided by Renderer3D itself (renderer.py:424), not this mixin.
        def update_instances(self, flock: PhysicsFlock, positions_override=None) -> int: ...

    def draw_birds(self, flock: PhysicsFlock, positions_override=None) -> None:
        """Single instanced draw call — winged (P8.4) / impostor (P8.1) / tetra.

        P8.10: Accepts optional *positions_override* for lerped render positions.
        P8.11: When density_mode is active with impostors, enables alpha
        blending and disables depth-write so overlapping sprites accumulate
        — dense regions appear darker (murmuratR aesthetic).

        S4.4a: Uses :func:`resolve_bird_mesh` to dynamically select the
        best mesh based on ``bird_mesh`` config + active bird count.
        """
        n = self.update_instances(flock, positions_override=positions_override)
        if n == 0:
            return

        # S4.4a: Resolve which mesh to render
        mesh_name = resolve_bird_mesh(self._bird_mesh, n)

        # P8.11: Density mode — alpha-blend on, depth off for impostors
        _density = self._density_mode and mesh_name == "impostor"
        if _density:
            import moderngl as _mgl
            self.ctx.enable(_mgl.BLEND)
            self.ctx.blend_func = (_mgl.SRC_ALPHA, _mgl.ONE_MINUS_SRC_ALPHA)
            self.ctx.disable(_mgl.DEPTH_TEST)
            self.ctx.depth_mask = False

        # S4.4a: Route to the appropriate VAO
        try:
            if mesh_name == "winged" and self._winged_mesh:
                self._winged_vao.render(instances=n)
            elif mesh_name == "impostor" and self._point_sprites:
                self._impostor_vao.render(instances=n)
            elif mesh_name == "points":
                # TODO(S4.4a): True GL_POINTS rendering with a dedicated
                # point sprite shader for >60K birds.  Currently falls
                # through to impostor quads (closest visual match).
                self._impostor_vao.render(instances=n)
            elif mesh_name in self._mesh_vaos:
                self._mesh_vaos[mesh_name].render(instances=n)  # type: ignore[attr-defined]
            else:
                # Fallback: default tetrahedron VAO
                self._vao.render(instances=n)
        finally:
            # P8.11: Restore state after density-mode render
            if _density:
                import moderngl as _mgl
                self.ctx.depth_mask = True
                self.ctx.enable(_mgl.DEPTH_TEST)
                self.ctx.disable(_mgl.BLEND)

    def draw_trails(self, flock: PhysicsFlock) -> None:
        """P8.3: Render trails (velocity lines / ring dots / accumulation / ribbon).

        Automatically records current positions into the history buffer
        for ring mode before drawing. For accumulation mode, blits the
        persistent FBO back into the main framebuffer after drawing.
        """
        if self._trails is None:
            return
        n = flock.N_active
        if n > 0:
            self._trails.push_history(flock)
            self._trails.draw(flock, self._instance_vbo, n)
            # P8.3: Accumulation mode — restore main FBO then blit persistent FBO
            if self._trails_mode == "accumulation":
                if self._fbo is not None:
                    self._fbo.use()
                self._trails.blit_accumulation()

    def draw_grid(self) -> None:
        """Reference grid on the XY plane (Z=0)."""
        # Set default attribute values for non-instanced rendering
        self._prog["in_bird_pos"] = (0.0, 0.0, 0.0)
        self._prog["in_bird_vel"] = (1.0, 0.0, 0.0)
        self._prog["in_bird_hue"] = 0.0   # P8.5: default hue
        self._prog["in_bird_scale"] = 1.0  # P8.5: default scale
        import moderngl
        self._grid_vao.render(moderngl.LINES)

    def draw_layer(
        self,
        position: tuple[float, float, float],
        hue: float = 0.0,
        scale: float = 1.0,
        mesh: str = "ellipsoid",
    ) -> None:
        """D7: Draw a single non-instanced marker at a world position.

        Feeds S2.A8 (threat marker) and S2.E5 (influencer target
        marker) — both currently invisible because no seam existed to
        render a one-off overlay outside the per-bird instanced draw
        call. Reuses the tetra shader program (self._prog) with default
        (non-instanced) attribute values — the same pattern draw_grid()
        already uses for the reference grid.

        Deliberately does NOT reuse self._mesh_vaos[mesh] (the S4.4a
        per-bird VAOs) — those bind in_bird_pos/vel/hue/scale to
        self._instance_vbo with a per-instance divisor, so a plain
        render() on them would draw at whatever bird #0's data
        currently is, silently ignoring the position/hue/scale
        arguments here. Instead builds (and caches) a dedicated VAO
        binding only the mesh's own static vertex/index buffers, so
        in_bird_pos/vel/hue/scale fall back to the per-draw-call default
        values set below, exactly like draw_grid()'s in_position VAO.
        These marker VAOs never need rebuilding on instance-buffer
        growth since they don't reference self._instance_vbo at all.

        Args:
            position: world-space (x, y, z) marker centre.
            hue: 0..1, matches the per-bird hue convention (P8.5).
            scale: NOT a geometric size multiplier — FRAGMENT_SHADER
                only reads in_bird_scale for the predator-highlight
                colour blend (>1.0 tints toward red and brightens,
                shaders.py's `predator_factor`); mesh vertex positions
                use a fixed size constant regardless of this value. Kept
                named "scale" for consistency with the per-bird
                attribute it sets, not because it resizes the marker.
            mesh: one of the S4.4a mesh-registry names sharing self._prog
                ("ellipsoid", "cone", "arrow"); falls back to "ellipsoid"
                if given an unknown name (e.g. "tetra"/"winged"/"impostor",
                which use a different shader program or aren't in this
                registry).
        """
        if mesh not in self._mesh_vbos:
            mesh = "ellipsoid"

        cache_attr = f"_marker_vao_{mesh}"
        vao = getattr(self, cache_attr, None)
        if vao is None:
            entry = MESH_REGISTRY[mesh]
            vao = self.ctx.vertex_array(
                self._prog,
                [(self._mesh_vbos[mesh], entry["vertex_format"], *entry["attributes"])],
                self._mesh_ibos[mesh],
            )
            setattr(self, cache_attr, vao)

        self._prog["in_bird_pos"] = tuple(position)
        self._prog["in_bird_vel"] = (0.0, 0.0, 1.0)  # arbitrary facing
        self._prog["in_bird_hue"] = hue
        self._prog["in_bird_scale"] = scale
        import moderngl
        vao.render(moderngl.TRIANGLES)

    def draw_hud_rect(
        self,
        x: int, y: int, w: int, h: int,
        colour: tuple[float, float, float],
    ) -> None:
        """P10.3: Draw a filled 2D rectangle using the HUD shader program.

        Args:
            x, y: top-left pixel position (y goes down).
            w, h: width and height in pixels.
            colour: (r, g, b) in [0, 1].
        """
        import moderngl
        c = np.array(colour, dtype=np.float32)
        self._hud_prog["u_hud_offset"].write(
            np.array([float(x), float(y)], dtype=np.float32).tobytes()
        )
        self._hud_prog["u_hud_size"].write(
            np.array([float(w), float(h)], dtype=np.float32).tobytes()
        )
        self._hud_prog["u_hud_colour"].write(c.tobytes())
        self._hud_vao.render(moderngl.TRIANGLES)
