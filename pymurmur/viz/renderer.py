"""Instanced 3D renderer using ModernGL.

Level 2 - optional GPU layer. Single instanced draw call for all birds.
Pre-allocated instance buffer grows in configurable chunks.
Supports 4 theme palettes and velocity trail rendering.
"""

from __future__ import annotations

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


class Renderer3D:
    """ModernGL instanced rendering - all birds in one draw call.

    Mesh: 4-vertex tetrahedron instanced per bird.
    Instance buffer: 6 floats/bird (pos.xyz + vel.xyz).
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

        # Tetrahedron mesh VAO
        vbo = self.ctx.buffer(TETRA_VERTICES.astype(np.float32).tobytes())
        ibo = self.ctx.buffer(TETRA_INDICES.astype(np.uint32).tobytes())

        # Instance VBO: 6 floats per bird (pos.xyz + vel.xyz)
        self._instance_vbo = self.ctx.buffer(
            reserve=self._max_instances * 6 * 4  # 4 bytes per float32
        )
        self._packed = np.zeros((self._max_instances, 6), dtype=np.float32)

        self._vao = self.ctx.vertex_array(
            self._prog,
            [
                (vbo, "3f", "in_position"),
                (self._instance_vbo, "3f 3f/i", "in_bird_pos", "in_bird_vel"),
            ],
            ibo,
        )

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
        """Pack SoA arrays into GPU buffer — single memcpy."""
        n = flock.N_active
        if n == 0:
            return 0

        # Grow buffer if needed
        while n > self._max_instances:
            self._max_instances += self._chunk
            self._instance_vbo = self.ctx.buffer(
                reserve=self._max_instances * 6 * 4
            )
            self._packed = np.zeros((self._max_instances, 6), dtype=np.float32)

        active_pos = flock.positions[flock.active]
        active_vel = flock.velocities[flock.active]
        self._packed[:n, :3] = active_pos[:n]
        self._packed[:n, 3:] = active_vel[:n]
        self._instance_vbo.write(self._packed[:n].tobytes())
        return n

    def begin_frame(self, camera: OrbitCamera) -> None:
        """Clear buffers and set camera uniforms."""
        if self._fbo is not None:
            self._fbo.use()
        clear = self._theme["clear"]
        self.ctx.clear(*clear)
        self._prog["u_view"].write(camera.view_matrix().to_bytes())
        self._prog["u_projection"].write(
            camera.projection_matrix(self.width / self.height).to_bytes()
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
