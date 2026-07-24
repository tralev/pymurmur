"""Renderer3D VAO-building mixin.

Extracted from renderer.py (file-size split). _RendererVAOMixin holds
the _build_*_vao methods -- kept as a mixin (not free functions) since
every method reads shared GL-context state (self.ctx, self._schema,
self._prog, instance/mesh buffers) set up by Renderer3D.__init__.
"""
from __future__ import annotations

from .mesh_registry import MESH_REGISTRY


class _RendererVAOMixin:
    """VAO construction methods, mixed into Renderer3D."""

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
