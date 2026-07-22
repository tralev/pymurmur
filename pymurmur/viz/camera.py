"""3D orbit camera — pure PyGLM math, no GPU dependency.

Level 2 — no project imports. Mouse drag to rotate, scroll to zoom.
Z-up world coordinate system.

P8.7: cinematic_sweep(t) — sets azim/elev/distance for a smooth
camera sweep during capture.

P8.8: Orthographic presets — set_ortho_top / set_ortho_side /
set_perspective, projection_mode switches between persp and ortho.
"""

from __future__ import annotations

from math import pi, radians, sin

import glm
import numpy as np


class OrbitCamera:
    """Spherical orbit camera around a 3D target point."""

    AUTO_ROTATE_SPEED: float = 0.45   # rad/s
    ROTATE_SENSITIVITY: float = 0.005  # rad/pixel
    ZOOM_SENSITIVITY: float = 50.0
    MIN_DISTANCE: float = 200.0
    MAX_DISTANCE: float = 4000.0
    MIN_ELEVATION: float = radians(-89)
    MAX_ELEVATION: float = radians(89)

    def __init__(
        self,
        target: tuple[float, float, float] = (500.0, 350.0, 200.0),
    ) -> None:
        self.target = glm.vec3(*target)
        self.azimuth: float = radians(45.0)
        self.elevation: float = radians(30.0)
        self.distance: float = 1200.0
        self.roll: float = 0.0  # S2.E6: pilot-mode roll (radians)
        self.auto_rotate: bool = False
        self.projection_mode: str = "perspective"  # P8.8: perspective|ortho_top|ortho_side

    def rotate(self, d_azimuth: float, d_elevation: float = 0.0) -> None:
        """Rotate camera by mouse delta (radians)."""
        self.azimuth += d_azimuth * self.ROTATE_SENSITIVITY
        self.elevation = np.clip(
            self.elevation + d_elevation * self.ROTATE_SENSITIVITY,
            self.MIN_ELEVATION,
            self.MAX_ELEVATION,
        )

    def zoom(self, delta: float) -> None:
        """Zoom in (positive delta) or out (negative delta)."""
        self.distance = np.clip(
            self.distance - delta * self.ZOOM_SENSITIVITY,
            self.MIN_DISTANCE,
            self.MAX_DISTANCE,
        )

    def roll_camera(self, d_roll: float) -> None:
        """S2.E6: Roll camera around the view axis (radians).

        Positive d_roll = counter-clockwise (Q), negative = clockwise (E).
        Clamped to ±π/2 to avoid gimbal-lock confusion."""
        self.roll = np.clip(
            self.roll + d_roll,
            -pi / 2,
            pi / 2,
        )

    def step_auto_rotate(self, dt: float) -> None:
        """Advance azimuth for automatic camera rotation."""
        if self.auto_rotate:
            self.azimuth += self.AUTO_ROTATE_SPEED * dt

    def cinematic_sweep(self, t: float, scale: float = 1.0) -> None:
        """P8.7: Set camera for cinematic capture sweep.

        Parameters
        ----------
        t: Normalised time [0, 1] where 0 = start, 1 = end of sweep.
        scale: Distance multiplier (config.capture_scale).

        Math (P8.7 spec)::

            azim = 45° + t·180°
            elev = 25° + sin(t·2π)·0.15
            dist = (650 + sin(t·1.5π)·100) · scale
        """
        self.azimuth = radians(45.0) + t * radians(180.0)
        self.elevation = radians(25.0) + sin(t * 2.0 * pi) * 0.15
        self.distance = (650.0 + sin(t * 1.5 * pi) * 100.0) * scale

    def reset(self) -> None:
        """Reset to default camera position."""
        self.azimuth = radians(45.0)
        self.elevation = radians(30.0)
        self.distance = 1200.0
        self.roll = 0.0  # S2.E6
        self.projection_mode = "perspective"  # P8.8

    # ── P8.8: Orthographic presets ──────────────────────────────

    def set_ortho_top(self, domain_size: float = 1200.0) -> None:
        """P8.8: Orthographic top-down view (looking from +Z)."""
        self.projection_mode = "ortho_top"
        self._ortho_size = domain_size
        # Position camera above
        self.azimuth = radians(0.0)
        self.elevation = radians(89.0)
        self.distance = domain_size * 0.6

    def set_ortho_side(self, domain_size: float = 1200.0) -> None:
        """P8.8: Orthographic side view (looking from +Y)."""
        self.projection_mode = "ortho_side"
        self._ortho_size = domain_size
        self.azimuth = radians(90.0)
        self.elevation = radians(0.0)
        self.distance = domain_size * 0.6

    def set_perspective(self) -> None:
        """P8.8: Default perspective projection."""
        self.projection_mode = "perspective"

    def eye_position(self) -> tuple[float, float, float]:
        """Return the camera's world-space eye position."""
        ex = self.target.x + self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        ey = self.target.y + self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        ez = self.target.z + self.distance * np.sin(self.elevation)
        return (ex, ey, ez)

    def view_matrix(self) -> glm.mat4:
        """Compute view matrix from spherical coordinates.

        S2.E6: Roll rotates the up vector around the view direction.
        The default up is (0, 0, 1) for Z-up world; with roll, the
        up vector is rotated by self.roll radians around the view axis."""
        eye = glm.vec3(*self.eye_position())
        view_dir = glm.normalize(self.target - eye)

        # S2.E6: Roll — rotate default up (0,0,1) around view_dir by roll
        default_up = glm.vec3(0.0, 0.0, 1.0)
        if abs(self.roll) > 1e-6:
            # Rodrigues rotation: rotate default_up around view_dir
            cos_r = np.cos(self.roll)
            sin_r = np.sin(self.roll)
            dot_vu = glm.dot(view_dir, default_up)
            up_rot = (
                default_up * cos_r
                + glm.cross(view_dir, default_up) * sin_r
                + view_dir * dot_vu * (1.0 - cos_r)
            )
            return glm.lookAt(eye, self.target, glm.normalize(up_rot))
        return glm.lookAt(eye, self.target, default_up)

    def screen_to_world(
        self, screen_x: float, screen_y: float,
        viewport_width: int, viewport_height: int,
        positions: np.ndarray | None = None,
    ) -> tuple[float, float, float] | None:
        """S5.4/P10.4: Unproject screen coords to a world spawn position.

        Casts a ray from the camera through the screen point and
        intersects it with a plane perpendicular to the camera's view
        axis, at the flock's median depth along that axis
        (`depth = median((p_i - o)*f_hat)`, spec S5.4) when *positions*
        is given (S5.4: `spawn = o + r_hat*depth/(r_hat*f_hat)`).
        Falls back to the `Z = target.z` plane when no flock context is
        available (positions is None/empty) — matches the pre-S5.4
        behaviour used by callers that don't have live flock state.

        Returns None if the ray is parallel to the intersection plane
        or the intersection is behind the camera.
        """
        view = self.view_matrix()
        aspect = viewport_width / max(viewport_height, 1)
        proj = self.projection_matrix(aspect)
        viewport = glm.vec4(0, 0, viewport_width, viewport_height)

        # Flip Y: screen (top-left origin) → GL (bottom-left origin)
        gl_y = viewport_height - screen_y

        # Near and far points in world space
        near = glm.unProject(glm.vec3(screen_x, gl_y, 0.0), view, proj, viewport)
        far = glm.unProject(glm.vec3(screen_x, gl_y, 1.0), view, proj, viewport)

        direction = glm.normalize(far - near)

        if positions is not None and len(positions) > 0:
            eye = glm.vec3(*self.eye_position())
            forward = glm.normalize(self.target - eye)
            o = np.array([eye.x, eye.y, eye.z])
            f_hat = np.array([forward.x, forward.y, forward.z])
            r_hat = np.array([direction.x, direction.y, direction.z])

            denom = float(np.dot(r_hat, f_hat))
            if abs(denom) < 1e-10:
                return None
            depth = float(np.median(np.asarray(positions) @ f_hat - np.dot(o, f_hat)))
            t = depth / denom
            if t < 0:
                return None
            hit = o + r_hat * t
            return (float(hit[0]), float(hit[1]), float(hit[2]))

        # Fallback: intersect with Z = target.z plane
        if abs(direction.z) < 1e-10:
            return None
        t = (self.target.z - near.z) / direction.z
        if t < 0:
            return None
        hit = near + direction * t
        return (float(hit.x), float(hit.y), float(hit.z))

    def projection_matrix(self, aspect: float) -> glm.mat4:
        """Projection matrix — perspective or orthographic (P8.8)."""
        if self.projection_mode == "ortho_top":
            half = getattr(self, '_ortho_size', 1200.0) / 2.0
            return glm.ortho(-half * aspect, half * aspect, -half, half, 1.0, 10000.0)
        elif self.projection_mode == "ortho_side":
            half = getattr(self, '_ortho_size', 1200.0) / 2.0
            return glm.ortho(-half * aspect, half * aspect, -half, half, 1.0, 10000.0)
        else:
            return glm.perspective(glm.radians(45.0), aspect, 1.0, 10000.0)
