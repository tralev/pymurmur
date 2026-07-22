"""Instanced 3D renderer using ModernGL.

Level 2 - optional GPU layer. Single instanced draw call for all birds.
Pre-allocated instance buffer grows in configurable chunks.
Supports 4 theme palettes and velocity trail rendering.

P2.7: InstanceSchema dataclass centralises GPU buffer layout so
buffer allocation, packing, and VAO creation share one source of truth.

P2.8: _mat4_bytes() helper avoids PyGLM memory-layout variance —
converts matrices to numpy float32 arrays before uploading to GPU.

P8.1: Sphere impostors — camera-facing billboard quads with a disc
fragment shader and speed-stretched ellipsoids. Toggled via
point_sprites config flag. A separate shader program + quad VAO is
built alongside the tetrahedron mesh path.

P8.3: Trail rendering — handled by :class:`TrailRenderer` which owns
the trail shader program and VAOs.  Renderer3D creates the
TrailRenderer and delegates draw_trails() to it.

P8.4: Winged flapping mesh (7-vertex bird, 6 triangles) + gradient sky.

P8.5: Per-bird colour channels — hue from flock.seeds, predator red,
theme material tables (ambient + diffuse) forwarded to shaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .camera import OrbitCamera
from .mesh_registry import (
    MESH_REGISTRY,
    get_theme_materials,
    resolve_bird_mesh,
)
from .shaders import (
    FRAGMENT_SHADER,
    GRID_VERTICES,
    HUD_FRAGMENT_SHADER,
    HUD_QUAD,
    HUD_VERTEX_SHADER,
    IMPOSTOR_FRAGMENT_SHADER,
    IMPOSTOR_QUAD,
    IMPOSTOR_QUAD_INDICES,
    IMPOSTOR_VERTEX_SHADER,
    SKY_FRAGMENT_SHADER,
    SKY_QUAD,
    SKY_QUAD_INDICES,
    SKY_VERTEX_SHADER,
    TETRA_INDICES,
    TETRA_VERTICES,
    THEMES,
    VERTEX_SHADER,
    WINGED_INDICES,
    WINGED_VERTEX_SHADER,
    WINGED_VERTICES,
)
from .trails import TrailRenderer

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from ..physics.flock import PhysicsFlock


@dataclass
class InstanceSchema:
    """GPU instance buffer layout descriptor (P2.7).

    Captures the per-bird float count, ModernGL vertex-array format
    string, and shader attribute names so that buffer allocation,
    CPU-side packing, and VAO creation all agree on the same layout.

    Changing *floats* or *layout* here propagates to every buffer
    allocation and VAO binding site in Renderer3D.

    D7: single 8-float schema (was 6 floats here + 2 in a separate
    colour VBO) — one packed array, one instance buffer, one
    ``vbo.write()`` per frame instead of two. The impostor VAO doesn't
    consume hue/scale (its shader has no such inputs — impostors aren't
    per-bird coloured), so it binds this same buffer with a padded
    format string (POS_VEL_LAYOUT below) that skips the trailing two
    floats instead of needing its own separate buffer.
    """

    floats: int = 8
    """Per-bird float count: pos.xyz (3) + vel.xyz (3) + hue (1) + scale (1)."""

    layout: str = "3f 3f 1f 1f/i"
    """ModernGL vertex-array format string."""

    attrs: tuple[str, str, str, str] = (
        "in_bird_pos", "in_bird_vel", "in_bird_hue", "in_bird_scale",
    )
    """Shader attribute names matching the layout components."""

    # D7: pos+vel-only view of the same buffer (8 bytes = 2 trailing
    # floats padded/skipped) for shaders that don't declare hue/scale.
    pos_vel_layout: str = "3f 3f 8x/i"
    pos_vel_attrs: tuple[str, str] = ("in_bird_pos", "in_bird_vel")


def _mat4_bytes(m) -> bytes:
    """Convert a PyGLM 4×4 matrix to bytes safely (P2.8).

    PyGLM builds differ in internal memory layout (column-major vs
    row-major, padding, alignment).  Converting via numpy float32
    guarantees a consistent 64-byte layout on every platform.
    """
    return np.array(m.to_list(), dtype=np.float32).tobytes()


class Renderer3D:
    """ModernGL instanced rendering - all birds in one draw call.

    Mesh: 4-vertex tetrahedron instanced per bird (default).
    P8.1: Sphere impostor mode — camera-facing quads with disc fragments,
    toggled via ``point_sprites=True``.
    Instance buffer layout defined by InstanceSchema (P2.7/D7):
    8 floats/bird (pos.xyz + vel.xyz + hue + scale), one shared buffer.
    Theme: ink | inverse | paper | graphite.
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        headless: bool = False,
        instance_buffer_chunk: int = 50000,
        theme: str = "ink",
        point_sprites: bool = False,
        winged_mesh: bool = True,
        gradient_sky: bool = True,
        trails_mode: str = "off",
        trails_length: int = 30,
        density_mode: bool = False,
        density_alpha: float = 0.2,
        bird_mesh: str = "auto",
        per_bird_color: bool = False,
        background_top: tuple[float, float, float] | None = None,
        background_bottom: tuple[float, float, float] | None = None,
        flap_period: float = 0.35,
        fps: int = 60,
    ) -> None:
        import moderngl  # deferred GPU context

        self.width = width
        self.height = height
        self.headless = headless
        self._chunk = instance_buffer_chunk
        self._max_instances = instance_buffer_chunk
        # S4.4a: MATERIAL_REGISTRY is the single source of truth for theme
        # materials.  self._theme (from shaders.py THEMES) is kept for
        # backward compatibility during migration and will be retired.
        self._theme = THEMES.get(theme, THEMES["ink"])
        self._materials = get_theme_materials(theme)  # S4.4a: per-theme material set
        self._point_sprites = point_sprites
        self._winged_mesh = winged_mesh
        self._bird_mesh = bird_mesh          # config.viz.bird_mesh
        self._gradient_sky = gradient_sky
        self._trails_mode = trails_mode
        self._trails_length = trails_length
        self._frame_count = 0  # P8.4: frame counter for flap animation
        # C3: flap_period (seconds) * fps — frames per full flap cycle.
        self._flap_period_frames: float = max(1e-3, flap_period * fps)
        self._render_scale: float = 1.0  # P8.6: adaptive quality
        # C3: background_top/background_bottom — None keeps the existing
        # theme-derived sky gradient (see _draw_sky).
        self._background_top = background_top
        self._background_bottom = background_bottom
        # P8.11: Alpha-accumulation density mode
        self._density_mode = density_mode
        self._density_alpha = density_alpha
        self._per_bird_color = per_bird_color  # config.viz.per_bird_color

        # P2.7: Single source of truth for instance buffer layout
        self._schema = InstanceSchema()

        if headless:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        else:
            self.ctx = moderngl.create_context(require=330)

        self.ctx.enable(moderngl.DEPTH_TEST)

        # Compile shader programs — tetra + winged (P8.4) + impostor (P8.1)
        self._prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self._winged_prog = self.ctx.program(
            vertex_shader=WINGED_VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,  # reuse Blinn-Phong fragment
        )
        self._impostor_prog = self.ctx.program(
            vertex_shader=IMPOSTOR_VERTEX_SHADER,
            fragment_shader=IMPOSTOR_FRAGMENT_SHADER,
        )
        # P8.11: Set density alpha uniform once (config-level constant)
        self._impostor_prog["u_density_alpha"] = self._density_alpha if self._density_mode else 1.0

        # Tetrahedron mesh VAO — store mesh buffers for VAO rebuild (I6.4)
        self._mesh_vbo = self.ctx.buffer(TETRA_VERTICES.astype(np.float32).tobytes())
        self._mesh_ibo = self.ctx.buffer(TETRA_INDICES.astype(np.uint32).tobytes())

        # P8.4: Winged mesh buffers (4f per vertex: xyz + flap_weight)
        self._winged_mesh_vbo = self.ctx.buffer(
            WINGED_VERTICES.astype(np.float32).tobytes()
        )
        self._winged_mesh_ibo = self.ctx.buffer(
            WINGED_INDICES.astype(np.uint32).tobytes()
        )

        # P8.4: Gradient sky VAO
        self._sky_prog = self.ctx.program(
            vertex_shader=SKY_VERTEX_SHADER,
            fragment_shader=SKY_FRAGMENT_SHADER,
        )
        sky_vbo = self.ctx.buffer(SKY_QUAD.astype(np.float32).tobytes())
        sky_ibo = self.ctx.buffer(SKY_QUAD_INDICES.astype(np.uint32).tobytes())
        self._sky_vao = self.ctx.vertex_array(
            self._sky_prog,
            [(sky_vbo, "2f", "in_position")],
            sky_ibo,
        )

        # P8.1: Impostor quad mesh buffers
        self._impostor_mesh_vbo = self.ctx.buffer(
            IMPOSTOR_QUAD.astype(np.float32).tobytes()
        )
        self._impostor_mesh_ibo = self.ctx.buffer(
            IMPOSTOR_QUAD_INDICES.astype(np.uint32).tobytes()
        )

        # Instance VBO: size from schema (P2.7/D7 — single 8-float buffer,
        # pos+vel+hue+scale all interleaved; one write()/frame).
        nf = self._schema.floats
        self._instance_vbo = self.ctx.buffer(
            reserve=self._max_instances * nf * 4  # 4 bytes per float32
        )
        self._packed = np.zeros((self._max_instances, nf), dtype=np.float32)

        self._vao = self._build_vao()
        self._winged_vao = self._build_winged_vao()  # P8.4
        self._impostor_vao = self._build_impostor_vao()

        # S4.4a: Build VAOs for additional mesh types (ellipsoid, cone, arrow).
        # Typed as 'Any' because moderngl.VertexArray is not available at
        # import time for type-checking.
        self._mesh_vaos: dict[str, object] = {}
        self._mesh_vbos: dict[str, object] = {}
        self._mesh_ibos: dict[str, object] = {}
        for name in ("ellipsoid", "cone", "arrow"):
            entry = MESH_REGISTRY[name]
            vbo = self.ctx.buffer(entry["vertices"].astype(np.float32).tobytes())
            ibo = self.ctx.buffer(entry["indices"].astype(np.uint32).tobytes())
            self._mesh_vbos[name] = vbo
            self._mesh_ibos[name] = ibo
            self._mesh_vaos[name] = self._build_mesh_vao(name, vbo, ibo)

        # Store VBOs/IBOs for VAO rebuild after instance buffer reallocation
        self._ellipsoid_vbo = self._mesh_vbos["ellipsoid"]
        self._ellipsoid_ibo = self._mesh_ibos["ellipsoid"]
        self._cone_vbo = self._mesh_vbos["cone"]
        self._cone_ibo = self._mesh_ibos["cone"]
        self._arrow_vbo = self._mesh_vbos["arrow"]
        self._arrow_ibo = self._mesh_ibos["arrow"]

        # Grid VAO
        grid_vbo = self.ctx.buffer(GRID_VERTICES.astype(np.float32).tobytes())
        self._grid_vao = self.ctx.vertex_array(
            self._prog, [(grid_vbo, "3f", "in_position")]
        )

        # P10.3: HUD shader program + quad VAO for 2D overlays
        self._hud_prog = self.ctx.program(
            vertex_shader=HUD_VERTEX_SHADER,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        hud_vbo = self.ctx.buffer(HUD_QUAD.astype(np.float32).tobytes())
        self._hud_vao = self.ctx.vertex_array(
            self._hud_prog,
            [(hud_vbo, "2f", "in_position")],
        )
        self._hud_prog["u_viewport"].write(  # type: ignore[union-attr]
            np.array([float(width), float(height)], dtype=np.float32).tobytes()
        )

        # Headless FBO for frame capture (D17: depth attachment added)
        if headless:
            self._depth_rb = self.ctx.depth_renderbuffer((width, height))
            self._fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 3)],
                depth_attachment=self._depth_rb,
            )
        else:
            self._fbo = None  # type: ignore[assignment]

        # P8.3: Trail renderer — owns trail GPU state
        self._trails: TrailRenderer | None = None
        if trails_mode != "off":
            self._trails = TrailRenderer(
                self.ctx,
                mode=trails_mode,
                trail_length=trails_length,
                theme=self._theme,
            )

    def release(self) -> None:
        """Release the underlying GL context and all its resources
        (buffers, programs, FBOs).

        Without this, each `Renderer3D` instance leaks its entire GL
        context for the process lifetime: `moderngl.Context` does define
        its own `__del__`, but relying on it alone was verified
        insufficient (the leak reproduced identically with an explicit
        `gc.collect()` forced after every test). Confirmed via a real
        run of the full `-m "gl or gpu"` suite under software Mesa
        llvmpipe: RSS grew from ~240 MB to ~1.7 GB within seconds and
        the process was OOM-killed partway through — this method plus
        `__del__` below fix that.  Safe to call more than once
        (`moderngl.Context.release()` is idempotent).
        """
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            try:
                ctx.release()
            except Exception:
                pass  # best-effort — already released, or context is gone

    def __del__(self) -> None:
        # Best-effort: interpreter shutdown can leave partially-torn-down
        # state, and this must never raise from a destructor.
        try:
            self.release()
        except Exception:
            pass

    # ── Public toggle ──────────────────────────────────────────

    @property
    def gl_lost(self) -> bool:
        """G6: Whether the GL context has been lost.

        Set to True when a moderngl.Error is caught mid-render.
        Once lost, rendering is skipped until the context is recreated.
        """
        return getattr(self, "_gl_lost", False)

    @gl_lost.setter
    def gl_lost(self, value: bool) -> None:
        self._gl_lost = value

    def simulate_gl_loss(self) -> None:
        """G6: Simulate GL context loss for testing.

        Sets gl_lost=True and releases the current GL context so
        subsequent GL calls raise moderngl.Error.
        """
        self._gl_lost = True
        try:
            if hasattr(self.ctx, "release"):
                self.ctx.release()
        except Exception:
            pass  # best-effort release

    @property
    def point_sprites(self) -> bool:
        """P8.1: Whether sphere impostors are active."""
        return self._point_sprites

    @point_sprites.setter
    def point_sprites(self, value: bool) -> None:
        """P8.1: Toggle sphere impostor rendering at runtime."""
        self._point_sprites = value

    # ── P8.6: Adaptive quality API ─────────────────────────────

    @property
    def render_scale(self) -> float:
        """P8.6: Render resolution scale (1.0 = full, 0.75 = min)."""
        return self._render_scale

    @render_scale.setter
    def render_scale(self, value: float) -> None:
        """P8.6: Set render resolution scale, clamped to [0.75, 1.0]."""
        self._render_scale = max(0.75, min(1.0, value))

    @property
    def effective_width(self) -> int:
        """P8.6: Width after render_scale applied."""
        return max(1, int(self.width * self._render_scale))

    @property
    def effective_height(self) -> int:
        """P8.6: Height after render_scale applied."""
        return max(1, int(self.height * self._render_scale))

    def disable_trails(self) -> None:
        """P8.6: Disable trail rendering for performance."""
        self._trails = None

    def enable_trails(self, mode: str, trail_length: int) -> None:
        """P8.6: Re-enable trail rendering after recovery."""
        if mode != "off":
            self._trails = TrailRenderer(
                self.ctx,
                mode=mode,
                trail_length=trail_length,
                theme=self._theme,
            )

    # ── Instance buffer ────────────────────────────────────────

    def update_instances(self, flock: PhysicsFlock, positions_override=None) -> int:
        """Pack SoA arrays into the GPU instance buffer — one write/frame.

        D7: pos+vel+hue+scale all interleave into one InstanceSchema-sized
        buffer (was two separate VBOs / two writes).
        Rebuilds the VAO when the instance buffer is reallocated (I6.4).
        P8.10: Accepts optional *positions_override* for lerped render positions.
        """
        n = flock.N_active
        if n == 0:
            return 0

        # P8.10: Use override positions (lerped) when provided
        pos_source = positions_override if positions_override is not None else flock.positions

        # Grow buffer if needed
        reallocated = False
        while n > self._max_instances:
            self._max_instances += self._chunk
            reallocated = True

        if reallocated:
            nf = self._schema.floats
            self._instance_vbo = self.ctx.buffer(
                reserve=self._max_instances * nf * 4
            )
            self._packed = np.zeros((self._max_instances, nf), dtype=np.float32)
            # Rebuild VAO against the new instance buffer (I6.4)
            self._vao = self._build_vao()
            self._winged_vao = self._build_winged_vao()  # P8.4
            self._impostor_vao = self._build_impostor_vao()
            self._rebuild_mesh_vaos()  # S4.4a

        active_pos = pos_source[flock.active]
        active_vel = flock.velocities[flock.active]
        active_seeds = flock.seeds[flock.active]
        active_pred = flock.is_predator[flock.active]

        self._packed[:n, 0:3] = active_pos[:n]
        self._packed[:n, 3:6] = active_vel[:n]
        # P8.5: per-bird hue (seed, if enabled) or flat gold
        self._packed[:n, 6] = (
            active_seeds[:n] if self._per_bird_color else np.full(n, 0.33)
        )
        self._packed[:n, 7] = np.where(active_pred[:n], 1.35, 1.0)  # predator scale
        self._instance_vbo.write(self._packed[:n].tobytes())
        return n

    def _build_vao(self):  # returns moderngl.VertexArray
        """Build a tetrahedron VAO from the current mesh + instance buffers (P2.7).

        Uses InstanceSchema.layout and InstanceSchema.attrs so that
        buffer layout changes propagate to VAO creation automatically.
        Called during __init__ and after every buffer reallocation.
        """
        s = self._schema
        return self.ctx.vertex_array(
            self._prog,
            [
                (self._mesh_vbo, "3f", "in_position"),
                (self._instance_vbo, s.layout, *s.attrs),  # D7: pos+vel+hue+scale
            ],
            self._mesh_ibo,
        )

    def _build_winged_vao(self):  # returns moderngl.VertexArray
        """P8.4: Build winged VAO — 3f 1f mesh (xyz + flap_weight) + instance.

        The winged vertex shader expects ``in_flap_weight`` at location 1
        alongside ``in_position`` at location 0.
        """
        s = self._schema
        return self.ctx.vertex_array(
            self._winged_prog,
            [
                (self._winged_mesh_vbo, "3f 1f", "in_position", "in_flap_weight"),
                (self._instance_vbo, s.layout, *s.attrs),  # D7: pos+vel+hue+scale
            ],
            self._winged_mesh_ibo,
        )

    def _build_impostor_vao(self):  # returns moderngl.VertexArray
        """P8.1: Build impostor quad VAO — 2f mesh + 3f/3f instance layout.

        The impostor vertex shader expects ``in_quad_pos`` (vec2) from
        the mesh buffer instead of ``in_position`` (vec3) used by the
        tetrahedron path. It has no in_bird_hue/in_bird_scale inputs
        (impostors aren't per-bird coloured) — D7: reads the same shared
        instance buffer as every other VAO, but with the pos+vel-only
        padded format that skips the trailing hue+scale floats instead
        of needing its own separate buffer.
        """
        s = self._schema
        return self.ctx.vertex_array(
            self._impostor_prog,
            [
                (self._impostor_mesh_vbo, "2f", "in_quad_pos"),
                (self._instance_vbo, s.pos_vel_layout, *s.pos_vel_attrs),
            ],
            self._impostor_mesh_ibo,
        )

    def _build_mesh_vao(self, name: str, vbo, ibo):  # returns moderngl.VertexArray
        """S4.4a: Build a VAO for a named mesh entry using the tetra
        shader program.

        Reuses the same program + instance layout as the tetra path;
        only the mesh geometry VBO/IBO differs.
        """
        s = self._schema
        entry = MESH_REGISTRY[name]
        return self.ctx.vertex_array(
            self._prog,
            [
                (vbo, entry["vertex_format"], *entry["attributes"]),
                (self._instance_vbo, s.layout, *s.attrs),  # D7: pos+vel+hue+scale
            ],
            ibo,
        )

    def _rebuild_mesh_vaos(self) -> None:
        """S4.4a: Rebuild mesh VAOs after instance buffer reallocation."""
        for name in self._mesh_vaos:
            self._mesh_vaos[name] = self._build_mesh_vao(
                name, self._mesh_vbos[name], self._mesh_ibos[name],
            )

    # ── Frame lifecycle ────────────────────────────────────────

    def begin_frame(self, camera: OrbitCamera) -> None:
        """Clear buffers, draw sky, and begin a new frame.

        P8.8: When dual-view is active, call begin_frame once, then
        use render_pass() for each camera/viewport pair.
        """
        if self._fbo is not None:
            self._fbo.use()

        # P8.6: Apply render_scale to viewport — reduces GPU workload
        self.ctx.viewport = (0, 0, self.effective_width, self.effective_height)

        mats = self._materials
        clear = mats["clear"]
        self.ctx.clear(*clear)

        # P8.4: Gradient sky — render before birds (depth test off)
        if self._gradient_sky:
            self._draw_sky()

        self._frame_count += 1  # P8.4: advance frame counter for flap

        # Upload camera + theme uniforms for all programs
        self._upload_camera_uniforms(camera, 0, 0, self.width, self.height)

    def render_pass(
        self, camera: OrbitCamera, vx: int, vy: int, vw: int, vh: int
    ) -> None:
        """P8.8: Render one camera pass into a viewport sub-rectangle.

        Sets viewport, uploads camera matrices for the given camera,
        and draws birds + trails + grid.  Does NOT clear or draw sky —
        call begin_frame() first.

        Used for dual-view rendering where two cameras render into
        left/right viewport halves.
        """
        self.ctx.viewport = (vx, vy, vw, vh)
        self._upload_camera_uniforms(camera, vx, vy, vw, vh)

    def _upload_camera_uniforms(
        self, camera: OrbitCamera, vx: int, vy: int, vw: int, vh: int
    ) -> None:
        """P8.8: Upload view/projection + theme uniforms for a camera pass.

        Extracted from begin_frame so both single-view and dual-view
        paths share the same uniform setup logic.
        """
        aspect = vw / vh if vh > 0 else 1.0
        eye = camera.eye_position()

        # Tetrahedron program uniforms
        self._prog["u_view"].write(_mat4_bytes(camera.view_matrix()))  # type: ignore[union-attr]
        self._prog["u_projection"].write(  # type: ignore[union-attr]
            _mat4_bytes(camera.projection_matrix(aspect))
        )
        self._prog["u_light_dir"].write(np.array([0.5, 0.5, 1.0], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._prog["u_camera_pos"].write(np.array(eye, dtype=np.float32).tobytes())  # type: ignore[union-attr]
        # P8.5 + S4.4a: Theme material tables from MATERIAL_REGISTRY
        mats = self._materials
        self._prog["u_Ambient"].write(np.array(mats["ambient"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._prog["u_Diffuse"].write(np.array(mats["diffuse"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        # Legacy theme colours
        self._prog["u_theme_slow"].write(np.array(mats["slow"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._prog["u_theme_spec"].write(np.array(mats["spec"], dtype=np.float32).tobytes())  # type: ignore[union-attr]

        # P8.4: Winged program uniforms
        self._winged_prog["u_view"].write(_mat4_bytes(camera.view_matrix()))  # type: ignore[union-attr]
        self._winged_prog["u_projection"].write(  # type: ignore[union-attr]
            _mat4_bytes(camera.projection_matrix(aspect))
        )
        self._winged_prog["u_frame"] = float(self._frame_count)
        self._winged_prog["u_flap_period_frames"] = self._flap_period_frames
        self._winged_prog["u_light_dir"].write(np.array([0.5, 0.5, 1.0], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._winged_prog["u_camera_pos"].write(np.array(eye, dtype=np.float32).tobytes())  # type: ignore[union-attr]
        # P8.5 + S4.4a: Theme materials from MATERIAL_REGISTRY
        mats = self._materials
        self._winged_prog["u_Ambient"].write(np.array(mats["ambient"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._winged_prog["u_Diffuse"].write(np.array(mats["diffuse"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._winged_prog["u_theme_slow"].write(np.array(mats["slow"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._winged_prog["u_theme_spec"].write(np.array(mats["spec"], dtype=np.float32).tobytes())  # type: ignore[union-attr]

        # P8.1+P8.2: Impostor program uniforms
        self._impostor_prog["u_view"].write(_mat4_bytes(camera.view_matrix()))  # type: ignore[union-attr]
        self._impostor_prog["u_projection"].write(  # type: ignore[union-attr]
            _mat4_bytes(camera.projection_matrix(aspect))
        )
        self._impostor_prog["u_bird_scale"] = 9.0
        self._impostor_prog["u_camera_pos"].write(np.array(eye, dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._impostor_prog["u_depth_power"] = 0.3
        self._impostor_prog["u_depth_fade"] = 0.5
        self._impostor_prog["u_rim_power"] = 3.0
        self._impostor_prog["u_max_depth"] = 5000.0
        self._impostor_prog["u_Paper"].write(np.array(mats["paper"], dtype=np.float32).tobytes())  # type: ignore[union-attr]
        self._impostor_prog["u_Ink"].write(np.array(mats["ink"], dtype=np.float32).tobytes())  # type: ignore[union-attr]

        # P8.3: Trail renderer camera uniforms
        if self._trails is not None:
            self._trails.begin_frame(camera, aspect)

    def _draw_sky(self) -> None:
        """P8.4: Render gradient sky as fullscreen quad (depth test off).

        C3: background_top/background_bottom override the theme-derived
        gradient when explicitly configured (config.viz defaults are
        non-None, so production runs use them; direct Renderer3D
        construction without them falls back to the theme gradient).
        """
        import moderngl
        self.ctx.disable(moderngl.DEPTH_TEST)
        mats = self._materials
        clear = mats["clear"]
        top = self._background_top if self._background_top is not None else (
            tuple(min(c * 1.3, 1.0) for c in clear)
        )
        bottom = self._background_bottom if self._background_bottom is not None else clear
        self._sky_prog["u_sky_top"].write(  # type: ignore[union-attr]
            np.array(top, dtype=np.float32).tobytes()
        )
        self._sky_prog["u_sky_bottom"].write(  # type: ignore[union-attr]
            np.array(bottom, dtype=np.float32).tobytes()
        )
        self._sky_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

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
            self.ctx.depth_mask = False  # type: ignore[attr-defined]

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
                self.ctx.depth_mask = True  # type: ignore[attr-defined]
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
        vao.render(moderngl.TRIANGLES)  # type: ignore[attr-defined]

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
        self._hud_prog["u_hud_offset"].write(  # type: ignore[union-attr]
            np.array([float(x), float(y)], dtype=np.float32).tobytes()
        )
        self._hud_prog["u_hud_size"].write(  # type: ignore[union-attr]
            np.array([float(w), float(h)], dtype=np.float32).tobytes()
        )
        self._hud_prog["u_hud_colour"].write(c.tobytes())  # type: ignore[union-attr]
        self._hud_vao.render(moderngl.TRIANGLES)

    def hud_begin(self) -> None:
        """P10.3: Set up OpenGL state for HUD overlay rendering.

        Disables depth test and resets viewport to full window so
        HUD coords are correct regardless of render_scale.
        """
        import moderngl
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.viewport = (0, 0, self.width, self.height)

    def hud_end(self) -> None:
        """P10.3: Restore OpenGL state after HUD pass."""
        import moderngl
        self.ctx.enable(moderngl.DEPTH_TEST)

    def end_frame(self) -> None:
        """Finish frame. In headless mode, ensure GPU work is complete."""
        if self._fbo is not None:
            self.ctx.finish()

    def capture_frame(self) -> PILImage:
        """FBO readback → PIL Image (headless mode only)."""
        from PIL import Image
        fbo = self._fbo if self._fbo is not None else self.ctx.screen
        data = fbo.read(components=3)
        return Image.frombytes("RGB", (self.width, self.height), data)
