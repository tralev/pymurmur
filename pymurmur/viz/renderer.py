"""Instanced 3D renderer using ModernGL.

Level 2 - optional GPU layer. Single instanced draw call for all birds.
Pre-allocated instance buffer grows in configurable chunks.
Supports 4 theme palettes and velocity trail rendering.

P2.7: InstanceSchema dataclass centralises GPU buffer layout so
buffer allocation, packing, and VAO creation share one source of truth.

P2.8: _mat4_bytes() helper avoids PyGLM memory-layout variance —
converts matrices to numpy float32 arrays before uploading to GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING

from .camera import OrbitCamera
from .shaders import (
    TETRA_VERTICES, TETRA_INDICES,
    VERTEX_SHADER, FRAGMENT_SHADER,
    TRAIL_VERTEX_SHADER, TRAIL_FRAGMENT_SHADER,
    GRID_VERTICES, THEMES,
)

if TYPE_CHECKING:
    from ..physics.flock import PhysicsFlock
    from PIL.Image import Image as PILImage


@dataclass
class InstanceSchema:
    """GPU instance buffer layout descriptor (P2.7).

    Captures the per-bird float count, ModernGL vertex-array format
    string, and shader attribute names so that buffer allocation,
    CPU-side packing, and VAO creation all agree on the same layout.

    Changing *floats* or *layout* here propagates to every buffer
    allocation and VAO binding site in Renderer3D.
    """

    floats: int = 6
    """Per-bird float count: pos.xyz (3) + vel.xyz (3)."""

    layout: str = "3f 3f/i"
    """ModernGL vertex-array format string."""

    attrs: tuple[str, str] = ("in_bird_pos", "in_bird_vel")
    """Shader attribute names matching the layout components."""


def _mat4_bytes(m) -> bytes:
    """Convert a PyGLM 4×4 matrix to bytes safely (P2.8).

    PyGLM builds differ in internal memory layout (column-major vs
    row-major, padding, alignment).  Converting via numpy float32
    guarantees a consistent 64-byte layout on every platform.
    """
    return np.array(m.to_list(), dtype=np.float32).tobytes()


class Renderer3D:
    """ModernGL instanced rendering - all birds in one draw call.

    Mesh: 4-vertex tetrahedron instanced per bird.
    Instance buffer layout defined by InstanceSchema (P2.7):
    6 floats/bird (pos.xyz + vel.xyz).
    Theme: ink | inverse | paper | graphite.
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        headless: bool = False,
        instance_buffer_chunk: int = 50000,
        theme: str = "ink",
    ) -> None:
        import moderngl  # deferred GPU context

        self.width = width
        self.height = height
        self.headless = headless
        self._chunk = instance_buffer_chunk
        self._max_instances = instance_buffer_chunk
        self._theme = THEMES.get(theme, THEMES["ink"])

        # P2.7: Single source of truth for instance buffer layout
        self._schema = InstanceSchema()

        if headless:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        else:
            self.ctx = moderngl.create_context(require=330)

        self.ctx.enable(moderngl.DEPTH_TEST)

        # Compile shader program
        self._prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        # Tetrahedron mesh VAO — store mesh buffers for VAO rebuild (I6.4)
        self._mesh_vbo = self.ctx.buffer(TETRA_VERTICES.astype(np.float32).tobytes())
        self._mesh_ibo = self.ctx.buffer(TETRA_INDICES.astype(np.uint32).tobytes())

        # Instance VBO: size from schema (P2.7)
        nf = self._schema.floats
        self._instance_vbo = self.ctx.buffer(
            reserve=self._max_instances * nf * 4  # 4 bytes per float32
        )
        self._packed = np.zeros((self._max_instances, nf), dtype=np.float32)

        self._vao = self._build_vao()

        # Grid VAO
        grid_vbo = self.ctx.buffer(GRID_VERTICES.astype(np.float32).tobytes())
        self._grid_vao = self.ctx.vertex_array(
            self._prog, [(grid_vbo, "3f", "in_position")]
        )

        # Headless FBO for frame capture
        if headless:
            self._fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 3)]
            )
        else:
            self._fbo = None

    def update_instances(self, flock: PhysicsFlock) -> int:
        """Pack SoA arrays into GPU buffer — single memcpy.

        Rebuilds the VAO when the instance buffer is reallocated (I6.4).
        """
        n = flock.N_active
        if n == 0:
            return 0

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

        active_pos = flock.positions[flock.active]
        active_vel = flock.velocities[flock.active]
        self._packed[:n, :3] = active_pos[:n]
        self._packed[:n, 3:] = active_vel[:n]
        self._instance_vbo.write(self._packed[:n].tobytes())
        return n

    def _build_vao(self) -> object:
        """Build a VAO from the current mesh + instance buffers (P2.7).

        Uses InstanceSchema.layout and InstanceSchema.attrs so that
        buffer layout changes propagate to VAO creation automatically.
        Called during __init__ and after every buffer reallocation.
        """
        s = self._schema
        return self.ctx.vertex_array(
            self._prog,
            [
                (self._mesh_vbo, "3f", "in_position"),
                (self._instance_vbo, s.layout, *s.attrs),
            ],
            self._mesh_ibo,
        )

    def begin_frame(self, camera: OrbitCamera) -> None:
        """Clear buffers and set camera uniforms."""
        if self._fbo is not None:
            self._fbo.use()
        clear = self._theme["clear"]
        self.ctx.clear(*clear)
        self._prog["u_view"].write(_mat4_bytes(camera.view_matrix()))
        self._prog["u_projection"].write(
            _mat4_bytes(camera.projection_matrix(self.width / self.height))
        )
        self._prog["u_light_dir"].write(np.array([0.5, 0.5, 1.0], dtype=np.float32).tobytes())
        eye = camera.eye_position()
        self._prog["u_camera_pos"].write(np.array(eye, dtype=np.float32).tobytes())
        # Theme colours
        self._prog["u_theme_slow"].write(np.array(self._theme["slow"], dtype=np.float32).tobytes())
        self._prog["u_theme_fast"].write(np.array(self._theme["fast"], dtype=np.float32).tobytes())
        self._prog["u_theme_spec"].write(np.array(self._theme["spec"], dtype=np.float32).tobytes())

    def draw_birds(self, flock: PhysicsFlock) -> None:
        """Single instanced draw call."""
        n = self.update_instances(flock)
        if n > 0:
            self._vao.render(instances=n)

    def draw_grid(self) -> None:
        """Reference grid on the XY plane (Z=0)."""
        # Set default attribute values for non-instanced rendering
        self._prog["in_bird_pos"] = (0.0, 0.0, 0.0)
        self._prog["in_bird_vel"] = (1.0, 0.0, 0.0)
        import moderngl
        self._grid_vao.render(moderngl.LINES)

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
