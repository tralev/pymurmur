"""GLSL shaders and tetrahedron mesh data for 3D rendering.

Level 2 — static data, no project imports. Only numpy for mesh arrays.

P8.1: Sphere impostors — camera-facing billboard quads with a disc fragment
shader, speed-stretched ellipsoids, and paper/ink theme colours.

P8.2: Depth cues + Fresnel rim — depth-based quad scaling, per-fragment
alpha fading (depth/speed/rim), and Fresnel rim highlight on impostors.

P8.4: Winged flapping mesh (7-vertex bird, 6 triangles) + gradient sky.

P8.5: Per-bird colour channels — hue from seeds, predator red,
theme ambient/diffuse material tables forwarded to shaders.
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

# ── P8.4: Winged mesh — 7 vertices, 6 triangles (body + wings + tail) ──
# Each vertex is (x, y, z, flap_weight).  flap_weight = 0 for body/tail,
# ±0.5 for wing tips — the shader uses this as the oscillation amplitude.
WINGED_VERTICES = np.array([
    # 0: nose tip
    [ 0.0,  0.0,  1.0,  0.0],
    # 1: body top
    [ 0.0,  0.25, -0.2,  0.0],
    # 2: body bottom
    [ 0.0, -0.15, -0.2,  0.0],
    # 3: right wing tip (flaps up/down)
    [ 0.65,  0.05,  0.0,  0.5],
    # 4: left wing tip (flaps opposite)
    [-0.65,  0.05,  0.0, -0.5],
    # 5: tail upper
    [ 0.0,  0.1, -0.7,  0.0],
    # 6: tail lower
    [ 0.0, -0.05, -0.7,  0.0],
], dtype=np.float32)

WINGED_INDICES = np.array([
    # Body (2 triangles)
    [0, 1, 2],   # right body panel
    [0, 2, 1],   # left body panel (opposite winding)
    # Wings (2 triangles)
    [1, 3, 2],   # right wing
    [1, 2, 4],   # left wing
    # Tail (2 triangles)
    [2, 1, 5],   # tail upper
    [1, 2, 6],   # tail lower
], dtype=np.uint32)

# ── Fullscreen quad for gradient sky (2 triangles, clip-space) ────
SKY_QUAD = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
], dtype=np.float32)

SKY_QUAD_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.uint32)

# ── Vertex shader — instanced, per-bird LookAt rotation ──────────
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;   // tetrahedron vertex
layout(location = 1) in vec3 in_bird_pos;    // instanced: bird position
layout(location = 2) in vec3 in_bird_vel;    // instanced: bird velocity
layout(location = 3) in float in_bird_hue;   // P8.5: per-bird hue (seed·360)
layout(location = 4) in float in_bird_scale; // P8.5: per-bird scale (predator >1)

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_normal;
out vec3 v_world_pos;
out float v_speed;
out float v_hue;       // P8.5
out float v_scale;     // P8.5

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
    v_hue = in_bird_hue;
    v_scale = in_bird_scale;

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

# ── Fragment shader — Blinn-Phong + per-bird hue + theme materials ──
FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_world_pos;
in float v_speed;
in float v_hue;      // P8.5: per-bird hue (0..1)
in float v_scale;    // P8.5: per-bird scale (>1 for predators)

out vec4 frag_color;

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
uniform vec3 u_Ambient;   // P8.5: theme ambient material
uniform vec3 u_Diffuse;   // P8.5: theme diffuse material
uniform vec3 u_theme_slow;   // colour at low speed (for hue base)
uniform vec3 u_theme_spec;   // specular highlight tint
uniform float u_rim_power;   // S4.2: mesh Fresnel rim exponent (k ~= 2-3)

vec3 hsv2rgb(vec3 c) {
    // c.x = hue [0,1], c.y = sat, c.z = val
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 32.0) * 0.25;

    // S4.2: Fresnel rim — view-angle silhouette highlight, the 3D
    // generalisation of a 1-px outline. Distinct from the impostor
    // disc's r^2-based rim (IMPOSTOR_FRAGMENT_SHADER) since meshes have
    // real normals to take a N.V angle from.
    float rim = pow(1.0 - max(dot(N, V), 0.0), u_rim_power);

    // P8.5: Per-bird hue — rotate the base colour by seed-derived hue
    float t = clamp(v_speed / 6.0, 0.0, 1.0);
    vec3 base_color = mix(u_theme_slow, vec3(1.0), t);  // brighten at speed
    vec3 hsv = vec3(v_hue, 0.7, 1.0);  // hue from seed, fixed saturation/value
    vec3 tint = hsv2rgb(hsv);
    base_color = base_color * tint;  // apply per-bird colour

    // P8.5: Predator → redder + larger (scale > 1 appears brighter)
    float predator_factor = clamp((v_scale - 1.0) / 0.5, 0.0, 1.0);  // 1.0→0, 1.5→1
    base_color = mix(base_color, vec3(1.0, 0.15, 0.1), predator_factor * 0.85);
    float bright = 1.0 + predator_factor * 0.35;  // predators glow brighter

    // P8.5: Theme ambient + diffuse instead of hardcoded constants
    vec3 lit = u_Ambient + u_Diffuse * diff * bright;
    lit += specular * u_theme_spec;
    lit += rim * u_theme_spec;  // S4.2: rim-light tinted by the theme's specular colour
    lit *= base_color;

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

# ── P8.3: Ring pass-through shader — world-space history points ──
# Unlike TRAIL_VERTEX_SHADER, this does NOT apply instance data
# (in_bird_pos / in_bird_vel) — the ring VBO already contains
# world-space positions from position_history.
RING_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;

uniform mat4 u_view;
uniform mat4 u_projection;

out float v_alpha;

void main() {
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
    v_alpha = 0.5;  // uniform fade for all ring points
}
"""

# ── P8.4: Winged vertex shader — instanced, flap before LookAt ────
# Same as VERTEX_SHADER but accepts in_flap_weight at location 1
# and applies vertex.y += flap_weight * sin(frame/100 * 2π)
# before the LookAt rotation (local-up flap).
WINGED_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;     // mesh vertex xyz
layout(location = 1) in float in_flap_weight; // wing-tip oscillation amplitude
layout(location = 2) in vec3 in_bird_pos;     // instance: bird position
layout(location = 3) in vec3 in_bird_vel;     // instance: bird velocity
layout(location = 4) in float in_bird_hue;    // P8.5: per-bird hue
layout(location = 5) in float in_bird_scale;  // P8.5: per-bird scale

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_frame;    // frame counter for flap animation
uniform float u_flap_period_frames;  // C3: flap_period (seconds) * fps

out vec3 v_normal;
out vec3 v_world_pos;
out float v_speed;
out float v_hue;       // P8.5
out float v_scale;     // P8.5

void main() {
    // P8.4: Wing flap — oscillate vertex y before rotation
    vec3 pos = in_position;
    float flap = in_flap_weight * sin(u_frame / u_flap_period_frames * 6.283185);
    pos.y += flap;

    // LookAt rotation: bird faces its velocity direction (Z-up world)
    vec3 forward = normalize(in_bird_vel);
    vec3 up = vec3(0.0, 0.0, 1.0);

    if (abs(dot(forward, up)) > 0.999) {
        up = vec3(1.0, 0.0, 0.0);
    }

    vec3 right = normalize(cross(up, forward));
    vec3 cam_up = cross(forward, right);

    mat3 rotation = mat3(right, forward, cam_up);

    vec3 world_pos = in_bird_pos + rotation * pos * 5.0;  // scale bird
    v_world_pos = world_pos;
    v_normal = rotation * normalize(pos);
    v_speed = length(in_bird_vel);
    v_hue = in_bird_hue;
    v_scale = in_bird_scale;

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

# ── P8.4: Gradient sky shaders ───────────────────────────────────
SKY_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 in_position;

out vec2 v_uv;

void main() {
    v_uv = in_position * 0.5 + 0.5;  // map [-1,1] to [0,1]
    gl_Position = vec4(in_position, 0.999999, 1.0);  // far plane so birds render in front
}
"""

SKY_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;

out vec4 frag_color;

uniform vec3 u_sky_top;     // colour at top of screen
uniform vec3 u_sky_bottom;  // colour at bottom of screen

void main() {
    frag_color = vec4(mix(u_sky_bottom, u_sky_top, v_uv.y), 1.0);
}
"""

# ── P8.1: Sphere impostor quad mesh ──────────────────────────────
# Camera-facing unit quad, centred at origin, 2 triangles.
IMPOSTOR_QUAD = np.array([
    [-0.5, -0.5],   # bottom-left
    [ 0.5, -0.5],   # bottom-right
    [ 0.5,  0.5],   # top-right
    [-0.5,  0.5],   # top-left
], dtype=np.float32)

IMPOSTOR_QUAD_INDICES = np.array([
    [0, 1, 2],
    [0, 2, 3],
], dtype=np.uint32)

# ── P8.1+P8.2: Impostor vertex shader — billboard + speed stretch + depth ──
IMPOSTOR_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 in_quad_pos;    // quad corner (-0.5..0.5)
layout(location = 1) in vec3 in_bird_pos;    // instance: bird world position
layout(location = 2) in vec3 in_bird_vel;    // instance: bird velocity

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_camera_pos;
uniform float u_bird_scale;
uniform float u_depth_power;    // P8.2: depth scaling exponent

out vec2 v_uv;
out float v_depth;              // P8.2: distance from camera
out float v_speed;              // P8.2: speed for alpha fading
out vec3 v_world_pos;           // P8.2: world position for rim lighting

void main() {
    // Billboard: extract camera right/up from view matrix
    vec3 cam_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 cam_up    = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    // Speed stretch factor: ellipsoid along projected velocity
    float speed = length(in_bird_vel);
    float stretch = 1.0 + (speed / 6.0) * 0.3;

    // Project velocity direction to screen plane
    vec3 vel_n = in_bird_vel / max(speed, 0.001);
    vec3 cam_fwd = cross(cam_right, cam_up);
    vec3 stretch_dir = vel_n - dot(vel_n, cam_fwd) * cam_fwd;
    float stretch_len = length(stretch_dir);
    if (stretch_len > 0.001) stretch_dir /= stretch_len;

    // Build quad offset: billboard axes + stretch along velocity
    vec3 offset = cam_right * in_quad_pos.x + cam_up * in_quad_pos.y;
    float stretch_contrib = dot(offset, stretch_dir) * (stretch - 1.0);
    offset += stretch_dir * stretch_contrib;

    vec3 world_pos = in_bird_pos + offset * u_bird_scale;
    v_uv = in_quad_pos + 0.5;  // map [-0.5,0.5] to [0,1]

    // P8.2: Depth cue — scale quad by 1/depth^k for perspective size
    v_depth = length(u_camera_pos - world_pos);
    float depth_scale = 1.0 / pow(v_depth / 1000.0, u_depth_power);
    vec3 scaled_world_pos = in_bird_pos + offset * u_bird_scale * depth_scale;

    v_speed = speed;
    v_world_pos = scaled_world_pos;

    gl_Position = u_projection * u_view * vec4(scaled_world_pos, 1.0);
}
"""

# ── P8.1+P8.2: Impostor fragment shader — disc + depth fade + Fresnel rim ──
IMPOSTOR_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;
in float v_depth;
in float v_speed;
in vec3 v_world_pos;

out vec4 frag_color;

uniform vec3 u_Paper;          // background / rim colour
uniform vec3 u_Ink;            // centre / bird colour
uniform vec3 u_camera_pos;     // P8.2: for Fresnel rim
uniform float u_depth_fade;    // P8.2: depth → alpha fade strength
uniform float u_rim_power;     // P8.2: Fresnel rim exponent
uniform float u_max_depth;     // P8.2: normalisation distance
uniform float u_density_alpha; // P8.11: sprite alpha in density mode (1.0 = off)

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float z = sqrt(1.0 - r2);

    // P8.1: Spherical shading with edge rim
    float edge = smoothstep(1.0, 0.72, r2);
    float shade = 0.55 + 0.45 * z;
    vec3 color = mix(u_Paper, u_Ink, shade * (1.0 - edge * 0.22));

    // P8.2: Fresnel rim highlight — implicit sphere normal
    vec3 N = vec3(p, z);  // normalised implicitly since r²+z²=1
    vec3 V = normalize(u_camera_pos - v_world_pos);
    float rim = pow(1.0 - abs(dot(N, V)) + 0.001, u_rim_power);
    // P8.11: Dampen rim glow in density mode so accumulated sprites darken
    color += u_Paper * rim * (0.35 * u_density_alpha);
    // P8.2: Alpha compositing — depth fade, speed fade, rim fade
    float depth01 = clamp(v_depth / u_max_depth, 0.0, 1.0);
    float speed01 = clamp(v_speed / 6.0, 0.0, 1.0);
    float alpha = 1.0;
    alpha *= mix(1.0, 1.0 - depth01, u_depth_fade);           // far → more transparent
    alpha *= mix(0.65, 1.0, speed01);                         // slow → slightly transparent
    alpha *= mix(1.0, 0.76, smoothstep(0.72, 1.0, r2));       // edge → slightly transparent

    frag_color = vec4(color, alpha * u_density_alpha);
}
"""

# ── Theme palettes (4 monochrome) + P8.5 material tables ──────────────────
THEMES: dict[str, dict[str, tuple[float, float, float]]] = {
    "ink": {
        "ambient": (0.02, 0.04, 0.10),
        "diffuse": (0.06, 0.12, 0.40),
        "slow": (0.1, 0.2, 0.5),
        "fast": (0.4, 0.8, 1.0),
        "spec": (1.0, 1.0, 1.0),
        "clear": (0.05, 0.05, 0.1),
        "trail": (0.3, 0.6, 0.9),
        "paper": (0.15, 0.25, 0.55),
        "ink": (0.02, 0.04, 0.12),
    },
    "inverse": {
        "ambient": (0.25, 0.23, 0.20),
        "diffuse": (0.30, 0.28, 0.22),
        "slow": (0.8, 0.75, 0.7),
        "fast": (0.2, 0.15, 0.1),
        "spec": (0.1, 0.1, 0.1),
        "clear": (0.9, 0.88, 0.85),
        "trail": (0.4, 0.35, 0.3),
        "paper": (0.85, 0.82, 0.78),
        "ink": (0.05, 0.05, 0.06),
    },
    "paper": {
        "ambient": (0.18, 0.16, 0.12),
        "diffuse": (0.35, 0.28, 0.16),
        "slow": (0.25, 0.2, 0.15),
        "fast": (0.45, 0.35, 0.2),
        "spec": (1.0, 0.95, 0.8),
        "clear": (0.95, 0.92, 0.85),
        "trail": (0.5, 0.4, 0.25),
        "paper": (0.92, 0.88, 0.82),
        "ink": (0.15, 0.12, 0.10),
    },
    "graphite": {
        "ambient": (0.08, 0.08, 0.08),
        "diffuse": (0.45, 0.45, 0.45),
        "slow": (0.15, 0.15, 0.15),
        "fast": (0.85, 0.85, 0.85),
        "spec": (0.3, 0.3, 0.3),
        "clear": (0.1, 0.1, 0.1),
        "trail": (0.5, 0.5, 0.5),
        "paper": (0.25, 0.25, 0.25),
        "ink": (0.05, 0.05, 0.05),
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

# ── P10.3: HUD shaders — 2D orthographic rendering ───────────────

# Simple unit quad (0,0)→(1,1) in 2D for HUD rect rendering
HUD_QUAD = np.array([
    0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  # tri 1
    0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  # tri 2
], dtype=np.float32)

HUD_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 in_position;

uniform vec2 u_hud_offset;   // (x, y) top-left in pixels
uniform vec2 u_hud_size;     // (w, h) in pixels
uniform vec2 u_viewport;     // (viewport_w, viewport_h)

void main() {
    // Map pixel coords to clip space [-1, 1]
    vec2 pixel_pos = u_hud_offset + in_position * u_hud_size;
    vec2 ndc = pixel_pos / u_viewport * 2.0 - 1.0;
    ndc.y = -ndc.y;  // flip Y: pixel y=0 at top → NDC y=+1 at top
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

HUD_FRAGMENT_SHADER = """
#version 330 core

uniform vec3 u_hud_colour;

out vec4 frag_color;

void main() {
    frag_color = vec4(u_hud_colour, 1.0);
}
"""
