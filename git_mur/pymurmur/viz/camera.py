"""3D orbit camera — pure PyGLM math, no GPU dependency.

Level 2 — no project imports. Mouse drag to rotate, scroll to zoom.
Z-up world coordinate system.
"""

from __future__ import annotations

import glm
import numpy as np
from math import radians


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
        self.auto_rotate: bool = False

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

    def step_auto_rotate(self, dt: float) -> None:
        """Advance azimuth for automatic camera rotation."""
        if self.auto_rotate:
            self.azimuth += self.AUTO_ROTATE_SPEED * dt

    def reset(self) -> None:
        """Reset to default camera position."""
        self.azimuth = radians(45.0)
        self.elevation = radians(30.0)
        self.distance = 1200.0

    def eye_position(self) -> tuple[float, float, float]:
        """Return the camera's world-space eye position."""
        ex = self.target.x + self.distance * np.cos(self.elevation) * np.cos(self.azimuth)
        ey = self.target.y + self.distance * np.cos(self.elevation) * np.sin(self.azimuth)
        ez = self.target.z + self.distance * np.sin(self.elevation)
        return (ex, ey, ez)

    def view_matrix(self) -> glm.mat4:
        """Compute view matrix from spherical coordinates."""
        eye = glm.vec3(*self.eye_position())
        return glm.lookAt(eye, self.target, glm.vec3(0.0, 0.0, 1.0))

    def projection_matrix(self, aspect: float) -> glm.mat4:
        """Perspective projection matrix."""
        return glm.perspective(glm.radians(45.0), aspect, 1.0, 10000.0)
