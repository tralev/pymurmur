"""Unit tests for viz.camera — OrbitCamera."""

from pathlib import Path

from math import radians

import glm

from pymurmur.viz.camera import OrbitCamera


def test_camera_default_position():
    """Default camera position is reasonable."""
    cam = OrbitCamera()
    assert cam.distance == 1200.0
    assert abs(cam.azimuth - radians(45)) < 0.01
    assert abs(cam.elevation - radians(30)) < 0.01


def test_camera_rotate():
    """rotate() changes azimuth."""
    cam = OrbitCamera()
    old_az = cam.azimuth
    cam.rotate(radians(45), 0)
    assert cam.azimuth != old_az


def test_camera_rotate_elevation():
    """rotate() changes elevation."""
    cam = OrbitCamera()
    old_el = cam.elevation
    cam.rotate(0, radians(10))
    assert cam.elevation != old_el


def test_camera_elevation_clamped():
    """Elevation never exceeds bounds."""
    cam = OrbitCamera()
    cam.elevation = radians(80)
    cam.rotate(0, radians(50))
    assert cam.elevation <= cam.MAX_ELEVATION


def test_camera_zoom():
    """zoom() changes distance."""
    cam = OrbitCamera()
    old_dist = cam.distance
    cam.zoom(1)
    assert cam.distance < old_dist


def test_camera_zoom_out():
    """zoom(-1) increases distance."""
    cam = OrbitCamera()
    old_dist = cam.distance
    cam.zoom(-1)
    assert cam.distance > old_dist


def test_camera_distance_clamped():
    """Distance clamped to [min_distance, max_distance]."""
    cam = OrbitCamera()
    # Try to zoom out beyond max
    cam.distance = cam.MAX_DISTANCE - 10
    cam.zoom(-10)  # would go beyond max if unclamped
    assert cam.distance <= cam.MAX_DISTANCE
    # Try to zoom in beyond min
    cam.distance = cam.MIN_DISTANCE + 10
    cam.zoom(10)  # would go below min if unclamped
    assert cam.distance >= cam.MIN_DISTANCE


def test_camera_auto_rotate():
    """step_auto_rotate() advances azimuth when enabled."""
    cam = OrbitCamera()
    cam.auto_rotate = True
    old_az = cam.azimuth
    cam.step_auto_rotate(1.0)  # 1 second
    # After 1s, azimuth should advance by AUTO_ROTATE_SPEED (0.45 rad/s)
    expected = old_az + cam.AUTO_ROTATE_SPEED
    assert abs(cam.azimuth - expected) < 0.01


def test_camera_auto_rotate_off():
    """step_auto_rotate() does nothing when disabled."""
    cam = OrbitCamera()
    cam.auto_rotate = False
    old_az = cam.azimuth
    cam.step_auto_rotate(5.0)  # 5 seconds
    assert cam.azimuth == old_az


def test_camera_reset():
    """reset() restores defaults."""
    cam = OrbitCamera()
    cam.rotate(radians(90), 0)
    cam.zoom(5)
    cam.reset()
    assert abs(cam.azimuth - radians(45)) < 0.01
    assert cam.distance == 1200.0


def test_camera_view_matrix():
    """view_matrix() returns a glm.mat4."""
    cam = OrbitCamera()
    mat = cam.view_matrix()
    assert mat is not None
    assert isinstance(mat, type(glm.mat4()))


def test_camera_projection_matrix():
    """projection_matrix() returns a glm.mat4."""
    cam = OrbitCamera()
    mat = cam.projection_matrix(1.5)
    assert mat is not None
    assert isinstance(mat, type(glm.mat4()))


def test_camera_eye_position():
    """eye_position() returns a 3-tuple distinct from target."""
    cam = OrbitCamera()
    eye = cam.eye_position()
    assert len(eye) == 3
    assert isinstance(eye[0], float)
    # Eye should not be at the target (camera is at distance 1200)
    assert eye[0] != cam.target.x or eye[1] != cam.target.y


def test_camera_no_moderngl_import():
    """camera.py does not import moderngl."""
    path = Path("pymurmur/viz/camera.py")
    text = path.read_text()
    assert "import moderngl" not in text
    assert "from moderngl" not in text
