"""P0.14 — SDF primitives and collision helpers for physics/obstacles.py."""

import numpy as np
import pytest

from pymurmur.physics.obstacles import (
    collision_detected,
    kinematic_correction,
    sdf_box,
    sdf_cylinder,
    sdf_gradient,
    sdf_sphere,
    sdf_subtract,
    sdf_union,
)

# ── SDF primitives ────────────────────────────────────────────────


class TestSdfSphere:
    def test_point_on_surface(self):
        """Point exactly on sphere surface → SDF = 0."""
        p = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        d = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        assert d[0] == pytest.approx(0.0, abs=1e-5)

    def test_point_inside(self):
        """Point inside sphere → SDF < 0."""
        p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        d = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        assert d[0] == pytest.approx(-5.0, abs=1e-5)

    def test_point_outside(self):
        """Point outside sphere → SDF > 0."""
        p = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        d = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        assert d[0] == pytest.approx(5.0, abs=1e-5)

    def test_offset_center(self):
        """Sphere at non-origin centre."""
        p = np.array([[7.0, 3.0, 3.0]], dtype=np.float32)
        center = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        d = sdf_sphere(p, center, 4.0)
        assert d[0] == pytest.approx(0.0, abs=1e-5)


class TestSdfBox:
    def test_point_at_center(self):
        """Point at box centre → negative (inside)."""
        p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        d = sdf_box(p, np.zeros(3, dtype=np.float32), b)
        assert d[0] < 0

    def test_point_on_face(self):
        """Point exactly on +X face → SDF = 0."""
        p = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        d = sdf_box(p, np.zeros(3, dtype=np.float32), b)
        assert d[0] == pytest.approx(0.0, abs=1e-5)

    def test_point_outside(self):
        """Point well outside box → SDF > 0."""
        p = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        d = sdf_box(p, np.zeros(3, dtype=np.float32), b)
        assert d[0] == pytest.approx(4.0, abs=1e-5)

    def test_point_outside_corner(self):
        """Point outside box corner → SDF = distance to corner."""
        p = np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
        b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        d = sdf_box(p, np.zeros(3, dtype=np.float32), b)
        expected = np.sqrt(3.0)
        assert d[0] == pytest.approx(expected, abs=1e-5)

    def test_point_inside(self):
        """Point deep inside box → SDF < 0 with correct magnitude."""
        p = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        d = sdf_box(p, np.zeros(3, dtype=np.float32), b)
        # Closest face is at x=2.0, so distance = 0.5 - 2.0 = -1.5
        assert d[0] == pytest.approx(-1.5, abs=1e-5)


class TestSdfCylinder:
    def test_point_on_surface_radial(self):
        """Point on cylinder radial surface → SDF = 0."""
        p = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 3.0, 10.0)
        assert d[0] == pytest.approx(0.0, abs=1e-5)

    def test_point_on_top_cap(self):
        """Point on cylinder top cap → SDF = 0."""
        p = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 5.0, 10.0)
        assert d[0] == pytest.approx(0.0, abs=1e-5)

    def test_point_inside(self):
        """Point inside cylinder → SDF < 0."""
        p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 5.0, 10.0)
        assert d[0] == pytest.approx(-5.0, abs=1e-5)

    def test_point_outside_radial(self):
        """Point outside cylinder radially → SDF > 0."""
        p = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 3.0, 10.0)
        assert d[0] == pytest.approx(7.0, abs=1e-5)

    def test_point_outside_vertical(self):
        """Point above cylinder → SDF > 0."""
        p = np.array([[0.0, 15.0, 0.0]], dtype=np.float32)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 5.0, 10.0)
        assert d[0] == pytest.approx(5.0, abs=1e-5)

    def test_offset_center(self):
        """Cylinder at non-origin centre."""
        p = np.array([[10.0, 10.0, 0.0]], dtype=np.float32)
        center = np.array([10.0, 10.0, 0.0], dtype=np.float32)
        d = sdf_cylinder(p, center, 3.0, 1.0)
        # radial=-3, height=-1 → closest is caps at 1, SDF = -1
        assert d[0] == pytest.approx(-1.0, abs=1e-5)


# ── CSG operations ────────────────────────────────────────────────


class TestSdfUnion:
    def test_min_of_two(self):
        """Union is the minimum of both SDFs."""
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([0.5, -1.0, 4.0], dtype=np.float32)
        result = sdf_union(a, b)
        np.testing.assert_allclose(result, [0.5, -2.0, 3.0])


class TestSdfSubtract:
    def test_max_of_a_and_neg_b(self):
        """Subtraction is max(a, -b)."""
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([0.5, -1.0, 4.0], dtype=np.float32)
        result = sdf_subtract(a, b)
        np.testing.assert_allclose(result, [1.0, 1.0, 3.0])


# ── Collision detection ───────────────────────────────────────────


class TestCollisionDetection:
    def test_no_collision_outside(self):
        """Both outside → no collision."""
        old = np.array([2.0, 3.0], dtype=np.float32)
        new = np.array([2.5, 1.0], dtype=np.float32)
        result = collision_detected(old, new)
        assert not result.any()

    def test_no_collision_inside(self):
        """Both inside → no collision (already penetrated)."""
        old = np.array([-1.0, -2.0], dtype=np.float32)
        new = np.array([-0.5, -1.0], dtype=np.float32)
        result = collision_detected(old, new)
        assert not result.any()

    def test_collision_entered(self):
        """Was outside, now inside → collision."""
        old = np.array([1.0, 0.5], dtype=np.float32)
        new = np.array([-0.1, -1.0], dtype=np.float32)
        result = collision_detected(old, new)
        assert result[0]
        assert result[1]

    def test_exit_not_collision(self):
        """Was inside, now outside → NOT a collision (exit)."""
        old = np.array([-1.0, 0.5], dtype=np.float32)
        new = np.array([1.0, 1.0], dtype=np.float32)
        result = collision_detected(old, new)
        assert not result[0]  # exit, not collision
        assert not result[1]  # stayed outside

    def test_collision_on_zero(self):
        """Touching surface → no collision (sign is 0)."""
        old = np.array([1.0], dtype=np.float32)
        new = np.array([0.0], dtype=np.float32)
        result = collision_detected(old, new)
        assert not result.any()


# ── Kinematic correction ──────────────────────────────────────────


class TestKinematicCorrection:
    def test_inside_sphere_pushed_to_surface(self):
        """Bird inside sphere → corrected to surface."""
        p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        def scene_fn(pts):
            return sdf_sphere(pts, np.zeros(3, dtype=np.float32), 5.0)

        corrected = kinematic_correction(p, scene_fn)
        # Should be pushed out to approximately radius 5.0
        d = np.linalg.norm(corrected[0])
        assert d == pytest.approx(5.0, abs=0.2)

    def test_outside_unchanged(self):
        """Bird outside sphere → position unchanged."""
        p = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)

        def scene_fn(pts):
            return sdf_sphere(pts, np.zeros(3, dtype=np.float32), 5.0)

        corrected = kinematic_correction(p, scene_fn)
        np.testing.assert_allclose(corrected[0], p[0], atol=1e-5)

    def test_on_surface_unchanged(self):
        """Bird on surface → position unchanged."""
        p = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)

        def scene_fn(pts):
            return sdf_sphere(pts, np.zeros(3, dtype=np.float32), 5.0)

        corrected = kinematic_correction(p, scene_fn)
        np.testing.assert_allclose(corrected[0], p[0], atol=1e-5)

    def test_batched_correction(self):
        """Multiple birds, some inside, some outside."""
        p = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        def scene_fn(pts):
            return sdf_sphere(pts, np.zeros(3, dtype=np.float32), 5.0)

        corrected = kinematic_correction(p, scene_fn)

        # Bird 0 was inside → pushed out
        d0 = np.linalg.norm(corrected[0])
        assert d0 == pytest.approx(5.0, abs=0.2)

        # Bird 1 was outside → unchanged
        np.testing.assert_allclose(corrected[1], p[1], atol=1e-5)

        # Bird 2 was on surface → unchanged
        np.testing.assert_allclose(corrected[2], p[2], atol=1e-5)


# ── SDF gradient ──────────────────────────────────────────────────


class TestSdfGradient:
    def test_sphere_gradient_points_radial(self):
        """Gradient of sphere SDF points radially outward."""
        p = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)

        def scene_fn(pts):
            return sdf_sphere(pts, np.zeros(3, dtype=np.float32), 5.0)

        grad = sdf_gradient(scene_fn, p)
        # Gradient should point away from center (positive x direction)
        assert grad[0, 0] > 0
        # y, z components should be near zero
        assert abs(grad[0, 1]) < 0.01
        assert abs(grad[0, 2]) < 0.01


# ── SDF composition (CSG scene) ───────────────────────────────────


class TestSdfComposition:
    def test_union_scene(self):
        """Union of two spheres creates a compound shape."""
        p = np.array([[5.0, 0.0, 0.0], [0.0, 11.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float32)
        s1 = sdf_sphere(p, np.array([0.0, 0.0, 0.0], dtype=np.float32), 5.0)
        s2 = sdf_sphere(p, np.array([0.0, 8.0, 0.0], dtype=np.float32), 3.0)
        scene = sdf_union(s1, s2)

        # Point on surface of sphere 1 (radius 5) → SDF ≈ 0
        assert abs(scene[0]) < 0.01
        # Point on surface of sphere 2 → SDF ≈ 0
        assert abs(scene[1]) < 0.01
        # Point outside both → SDF > 0
        assert scene[2] > 0

    def test_subtract_scene(self):
        """Subtract creates a cavity."""
        # Box at [-1,0,0] with half [6,2,2] spans [-7,5] in X
        # Sphere cavity at [1,0,0] with radius 2 spans [-1,3] in X
        # Point at [-4,0,0]: inside box (-1→-7), outside cavity (>1 left)
        # Point at [0,0,0]: inside box, inside cavity (dist 1 < 2)
        p = np.array([[0.0, 0.0, 0.0], [-4.0, 0.0, 0.0]], dtype=np.float32)
        box = sdf_box(p, np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                      np.array([6.0, 2.0, 2.0], dtype=np.float32))
        cavity = sdf_sphere(p, np.array([1.0, 0.0, 0.0], dtype=np.float32), 2.0)
        scene = sdf_subtract(box, cavity)

        # Point [0,0,0]: inside box + inside cavity → subtracted (positive)
        assert scene[0] > 0
        # Point [-4,0,0]: inside box + outside cavity → still inside box (negative)
        assert scene[1] < 0


# ── P11.4: ObstacleScene ──────────────────────────────────────────


class TestObstacleScene:
    """CSG scene tree — composition, collision, correction."""

    def test_empty_scene_infinite_sdf(self):
        """Empty scene → +inf everywhere (no obstacles)."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene()
        p = np.zeros((3, 3), dtype=np.float32)
        assert np.all(np.isinf(scene.sdf(p)))
        assert scene.n_shapes == 0

    def test_union_composition(self):
        """Two spheres union: SDF is min of both."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = (
            ObstacleScene()
            .add_sphere([0.0, 0.0, 0.0], 5.0)
            .add_sphere([20.0, 0.0, 0.0], 5.0)
        )
        p = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        assert scene.sdf(p)[0] == pytest.approx(5.0)

    def test_subtract_composition(self):
        """Shell: outer sphere minus inner sphere."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = (
            ObstacleScene()
            .add_sphere([0.0, 0.0, 0.0], 10.0)
            .add_sphere([0.0, 0.0, 0.0], 8.0, op="subtract")
        )
        p = np.array(
            [[9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        d = scene.sdf(p)
        assert d[0] < 0    # inside the shell wall
        assert d[1] > 0    # in the cavity → outside
        assert d[2] > 0    # beyond the outer surface → outside

    def test_cylinder_and_box_shapes(self):
        """All three primitive builders compose."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = (
            ObstacleScene()
            .add_box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
            .add_cylinder([10.0, 0.0, 0.0], 1.0, 5.0)
        )
        assert scene.n_shapes == 2
        p = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        assert np.all(scene.sdf(p) < 0)

    def test_invalid_op_raises(self):
        """Unknown CSG op and leading subtract both raise."""
        from pymurmur.physics.obstacles import ObstacleScene
        with pytest.raises(ValueError):
            ObstacleScene().add_sphere([0, 0, 0], 1.0, op="intersect")
        with pytest.raises(ValueError):
            ObstacleScene().add_sphere([0, 0, 0], 1.0, op="subtract")

    def test_resolve_detects_sign_flip(self):
        """Collision when sign(SDF(p_old)) ≠ sign(SDF(p_new))."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        p_old = np.array([[7.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=np.float32)
        p_new = np.array([[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]], dtype=np.float32)
        corrected, collided = scene.resolve(p_old, p_new)
        assert collided.tolist() == [True, False]
        assert scene.collision_count == 1
        # Colliding bird pushed back to the surface
        assert scene.sdf(corrected[:1])[0] == pytest.approx(0.0, abs=1e-2)
        # Non-colliding bird untouched
        assert corrected[1] == pytest.approx(p_new[1])

    def test_resolve_empty_scene_noop(self):
        """Empty scene: no collisions, positions unchanged."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene()
        p = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
        corrected, collided = scene.resolve(p, p)
        assert not collided.any()
        assert corrected == pytest.approx(p)

    def test_from_spec_builds_scene(self):
        """YAML-friendly spec list builds the same scene."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene.from_spec([
            {"shape": "sphere", "center": [0, 0, 0], "radius": 5.0},
            {"shape": "box", "center": [20, 0, 0], "half_extents": [2, 2, 2]},
            {"shape": "cylinder", "center": [-20, 0, 0], "radius": 1.0,
             "half_height": 4.0},
        ])
        assert scene.n_shapes == 3
        p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        assert scene.sdf(p)[0] == pytest.approx(-5.0)

    def test_from_spec_unknown_shape_raises(self):
        from pymurmur.physics.obstacles import ObstacleScene
        with pytest.raises(ValueError):
            ObstacleScene.from_spec([{"shape": "torus", "center": [0, 0, 0]}])

    def test_avoidance_accel_points_away(self):
        """P11.5: static + predictive avoidance push away from the surface."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)  # approaching
        accel = scene.avoidance_accel(
            pos, vel, static_weight=1.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        assert accel[0, 0] > 0.0, "Avoidance must push away (+x)"

    def test_avoidance_accel_zero_weights_noop(self):
        """Zero weights → zero acceleration."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        accel = scene.avoidance_accel(pos, vel)
        assert np.all(accel == 0.0)

    def test_avoidance_accel_static_only(self):
        """P11.4: Static-only avoidance pushes away from surface;
        predictive component is zero."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        # Bird near surface approaching
        pos = np.array([[6.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        accel = scene.avoidance_accel(
            pos, vel, static_weight=1.0, predictive_weight=0.0,
            fly_away_max_dist=5.0,
        )
        # Static avoidance pushes away from surface
        assert accel[0, 0] > 0.0, "Static avoidance must push away (+x)"

    def test_avoidance_accel_predictive_only(self):
        """P11.4: Predictive-only avoidance pushes away when
        approaching the surface with non-zero velocity."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[7.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        accel = scene.avoidance_accel(
            pos, vel, static_weight=0.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        # Predictive avoidance pushes away from surface when approaching
        assert accel[0, 0] > 0.0, (
            "Predictive avoidance must push away (+x) when approaching"
        )

    def test_avoidance_accel_far_from_surface(self):
        """P11.4: Bird far from any surface gets zero avoidance
        acceleration regardless of weights."""
        from pymurmur.physics.obstacles import ObstacleScene
        scene = ObstacleScene().add_sphere([0.0, 0.0, 0.0], 5.0)
        pos = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
        vel = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        accel = scene.avoidance_accel(
            pos, vel, static_weight=1.0, predictive_weight=1.0,
            fly_away_max_dist=5.0, min_time_to_collide=5.0,
        )
        # Far from any surface → no avoidance needed
        assert np.all(accel == 0.0), (
            f"Far from surface should get zero avoidance, got {accel}"
        )
