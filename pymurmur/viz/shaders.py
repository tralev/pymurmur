"""GLSL shaders and tetrahedron mesh data for 3D rendering.

Level 2 — static data, no project imports. Only numpy for mesh arrays.
"""

from __future__ import annotations

import numpy as np

# ── Tetrahedron mesh (4 vertices, 4 triangular faces) ────────────
# Asymmetric: front tip at +Z for visible orientation.
TETRA_VERTICES = np.array([
    [ 0.0,  0.0,  1.0],  # front tip
    [ 0.0,  0.943, -0.333],  # top
    [-0.816, -0.471, -0.333],  # bottom-left
    [ 0.816, -0.471, -0.333],  # bottom-right
], dtype=np.float32)

TETRA_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 1],
    [1, 3, 2],
], dtype=np.uint32)

# ── Vertex shader — instanced, per-bird LookAt rotation ──────────
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;   // tetrahedron vertex
layout(location = 1) in vec3 in_bird_pos;    // instanced: bird position
layout(location = 2) in vec3 in_bird_vel;    // instanced: bird velocity

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_world_pos;
out float v_speed;

void main() {
    // LookAt rotation: bird faces its velocity direction (Z-up world)
    vec3 forward = normalize(in_bird_vel);
    vec3 up = vec3(0.0, 0.0, 1.0);

    // Gimbal-lock guard: if forward is near vertical, use alt up
    if (abs(dot(forward, up)) > 0.999) {
        up = vec3(1.0, 0.0, 0.0);
    }

    vec3 right = normalize(cross(up, forward));
    vec3 cam_up = cross(forward, right);

    mat3 rotation = mat3(right, forward, cam_up);

    vec3 world_pos = in_bird_pos + rotation * in_position * 5.0;  // scale bird
    v_world_pos = world_pos;
    v_normal = rotation * normalize(in_position);
    v_speed = length(in_bird_vel);

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

# ── Fragment shader — Blinn-Phong + speed-based colour tint ──────
FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_world_pos;
in float v_speed;

out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
uniform float u_time;
uniform vec3 u_theme_slow;   // colour at low speed
uniform vec3 u_theme_fast;   // colour at high speed
uniform vec3 u_theme_spec;   // specular highlight tint

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float ambient = 0.15;
    float diffuse = max(dot(N, L), 0.0) * 0.6;
    float specular = pow(max(dot(N, H), 0.0), 32.0) * 0.25;

    float t = clamp(v_speed / 6.0, 0.0, 1.0);
    vec3 base_color = mix(u_theme_slow, u_theme_fast, t);

    vec3 lit = base_color * (ambient + diffuse) + specular * u_theme_spec;
    frag_color = vec4(lit, 1.0);
}
"""

# ── Trail shader — velocity-stretched line segments ──────────────
TRAIL_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_bird_pos;
layout(location = 2) in vec3 in_bird_vel;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_trail_length;

out float v_alpha;

void main() {
    // Stretch vertex along velocity: in_position.x controls stretch amount
    vec3 trail_pos = in_bird_pos + in_bird_vel * in_position.x * u_trail_length * 0.3;
    // in_position.yz are perpendicular offset (unused for simple line)
    v_alpha = 1.0 - abs(in_position.x);  // fade at tips
    gl_Position = u_projection * u_view * vec4(trail_pos, 1.0);
}
"""

TRAIL_FRAGMENT_SHADER = """
#version 330 core

in float v_alpha;
out vec4 frag_color;

uniform vec3 u_trail_color;

void main() {
    frag_color = vec4(u_trail_color, v_alpha * 0.3);
}
"""

# ── Theme palettes (4 monochrome) ─────────────────────────────────
THEMES: dict[str, dict[str, tuple[float, float, float]]] = {
    "ink": {
        "slow": (0.1, 0.2, 0.5),
        "fast": (0.4, 0.8, 1.0),
        "spec": (1.0, 1.0, 1.0),
        "clear": (0.05, 0.05, 0.1),
        "trail": (0.3, 0.6, 0.9),
    },
    "inverse": {
        "slow": (0.8, 0.75, 0.7),
        "fast": (0.2, 0.15, 0.1),
        "spec": (0.1, 0.1, 0.1),
        "clear": (0.9, 0.88, 0.85),
        "trail": (0.4, 0.35, 0.3),
    },
    "paper": {
        "slow": (0.25, 0.2, 0.15),
        "fast": (0.45, 0.35, 0.2),
        "spec": (1.0, 0.95, 0.8),
        "clear": (0.95, 0.92, 0.85),
        "trail": (0.5, 0.4, 0.25),
    },
    "graphite": {
        "slow": (0.15, 0.15, 0.15),
        "fast": (0.85, 0.85, 0.85),
        "spec": (0.3, 0.3, 0.3),
        "clear": (0.1, 0.1, 0.1),
        "trail": (0.5, 0.5, 0.5),
    },
}

# ── Grid line vertices (XY plane, centered on origin) ────────────
GRID_VERTICES = np.array([
    # X-axis lines
    [-1000, -1000, 0], [-1000, 1000, 0],
    [-750, -1000, 0], [-750, 1000, 0],
    [-500, -1000, 0], [-500, 1000, 0],
    [-250, -1000, 0], [-250, 1000, 0],
    [0, -1000, 0], [0, 1000, 0],
    [250, -1000, 0], [250, 1000, 0],
    [500, -1000, 0], [500, 1000, 0],
    [750, -1000, 0], [750, 1000, 0],
    [1000, -1000, 0], [1000, 1000, 0],
    # Y-axis lines
    [-1000, -1000, 0], [1000, -1000, 0],
    [-1000, -750, 0], [1000, -750, 0],
    [-1000, -500, 0], [1000, -500, 0],
    [-1000, -250, 0], [1000, -250, 0],
    [-1000, 0, 0], [1000, 0, 0],
    [-1000, 250, 0], [1000, 250, 0],
    [-1000, 500, 0], [1000, 500, 0],
    [-1000, 750, 0], [1000, 750, 0],
    [-1000, 1000, 0], [1000, 1000, 0],
], dtype=np.float32)
