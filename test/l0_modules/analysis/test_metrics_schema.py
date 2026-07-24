"""I6.6 — JSON round-trip tests for FlockMetrics.to_dict()."""

import json

import numpy as np
import pytest

from pymurmur.analysis.metrics import FlockMetrics


class TestFlockMetricsToDict:
    """FlockMetrics.to_dict() produces JSON-safe output."""

    def test_json_roundtrip_all_fields(self):
        """to_dict() → json.dumps() → json.loads() round-trips cleanly."""
        m = FlockMetrics()
        m.alpha = 0.85
        m.theta = 0.3
        m.theta_prime = 0.12
        m.angular_momentum = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        m.dispersion = 45.0
        m.speed_avg = 3.2
        m.force_avg = 0.15
        m.power_avg = 0.08
        m.local_spacing = 12.3
        m.h2 = 1.23
        m.aspect_ratio = 3.5
        m.thickness_ratio = 0.28
        m.gyration_radius = 55.0
        m.optimal_m = 6

        d = m.to_dict()
        serialized = json.dumps(d)
        loaded = json.loads(serialized)

        # Scalar fields round-trip
        assert loaded["alpha"] == 0.85
        assert loaded["theta"] == 0.3
        assert loaded["h2"] == 1.23

        # ndarray → list
        assert loaded["angular_momentum"] == [1.0, -2.0, 0.5]

        # None fields are null
        assert loaded["msd"] is None
        assert loaded["tau_rho"] is None

    def test_nan_becomes_null(self):
        """NaN values serialize as null (JSON-safe)."""
        m = FlockMetrics()
        m.theta = float("nan")
        m.alpha = 0.5

        d = m.to_dict()
        assert d["theta"] is None
        assert d["alpha"] == 0.5

        serialized = json.dumps(d)
        assert "null" in serialized

    def test_none_fields_serialize_as_null(self):
        """None fields (e.g., uncomputed expensive metrics) → null."""
        m = FlockMetrics()
        m.h2 = None
        m.msd = None

        d = m.to_dict()
        serialized = json.dumps(d)
        loaded = json.loads(serialized)

        assert loaded["h2"] is None
        assert loaded["msd"] is None

    def test_numpy_scalar_becomes_python_scalar(self):
        """numpy.float32(0.85) → 0.85 Python float."""
        m = FlockMetrics()
        m.alpha = np.float32(0.85)
        m.dispersion = np.float64(123.45)

        d = m.to_dict()
        assert isinstance(d["alpha"], float)
        assert d["alpha"] == pytest.approx(0.85)
        assert isinstance(d["dispersion"], float)
        assert d["dispersion"] == pytest.approx(123.45)

    def test_all_keys_present(self):
        """All 15 expected metric keys are present in the output."""
        m = FlockMetrics()
        d = m.to_dict()

        expected_keys = {
            "alpha", "nematic_S", "theta", "theta_prime", "silhouette_2d",
            "angular_momentum", "normalized_angular_momentum",
            "dispersion", "speed_avg", "force_avg", "power_avg",
            "local_spacing", "velocity_deviation", "boundary_overshoot",
            "altitude_deviation", "collisions_this_step",
            "h2", "r_nodal", "tau_rho", "msd",
            "gyration_radius", "aspect_ratio", "thickness_ratio",
            "optimal_m", "suggested_m", "eta_m",
            # A10: convergence speed λ₂(L)
            "convergence_speed",
            # B3: max pairwise 3D distance (fragmentation)
            "r_max",
            # B9: accel-vs-opacity cross-correlation
            "theta_accel_correlation", "theta_accel_peak_lag",
            # B14: steering-saturation "jammed" states diagnostic
            "jamming_index",
            # B5/B6: mean-field Psky + marginal-opacity critical density
            "psky_meanfield", "marginal_opacity_density",
            # P9.2: MSD curve fields
            "msd_slope", "msd_crossover", "msd_curve",
            # P9.3: Hull-volume density
            "hull_volume", "density_rho",
            # P4.4: Physical metrics
            "speed_real_ms", "accel_real_ms2", "force_real_N",
            "power_real_W", "energy_J",
            # S2.E5: influencer-mode target-distance diagnostics
            "target_dist_min", "target_dist_max",
        }
        assert set(d.keys()) == expected_keys

    def test_no_numpy_types_in_json_output(self):
        """Serialized JSON contains no numpy type strings (ndarray, float32, etc.)."""
        m = FlockMetrics()
        m.angular_momentum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        m.alpha = np.float64(0.5)

        serialized = json.dumps(m.to_dict())
        # Should not contain numpy type repr strings
        assert "float32" not in serialized
        assert "float64" not in serialized
        assert "ndarray" not in serialized
        assert "dtype" not in serialized

    def test_empty_metrics_defaults(self):
        """FlockMetrics() with all defaults produces valid JSON."""
        m = FlockMetrics()
        serialized = json.dumps(m.to_dict())
        loaded = json.loads(serialized)
        assert loaded["alpha"] == 0.0
        assert loaded["angular_momentum"] == [0.0, 0.0, 0.0]


# ── P9.10: inf serialization + full pipeline ────────────────────

def test_inf_becomes_null():
    """P9.10: float('inf') serializes as null (JSON-safe)."""
    m = FlockMetrics()
    m.eta_m = float('inf')
    m.alpha = 0.5

    d = m.to_dict()
    assert d["eta_m"] is None, f"inf should become null, got {d['eta_m']}"
    assert d["alpha"] == 0.5

    serialized = json.dumps(d)
    assert "null" in serialized


def test_neg_inf_becomes_null():
    """P9.10: float('-inf') serializes as null."""
    m = FlockMetrics()
    m.eta_m = float('-inf')

    d = m.to_dict()
    assert d["eta_m"] is None

    serialized = json.dumps(d)
    loaded = json.loads(serialized)
    assert loaded["eta_m"] is None


def test_full_collector_pipeline_json_roundtrip():
    """P9.10: Full collector → snapshot → to_dict → JSON roundtrip
    includes all P9 fields."""
    from pymurmur.analysis.metrics import MetricsCollector
    from pymurmur.core.config import SimConfig
    from pymurmur.physics.flock import PhysicsFlock

    cfg = SimConfig(num_boids=20, seed=42,
                    metrics_detail_level=2, metrics_interval=2)
    flock = PhysicsFlock(cfg)
    collector = MetricsCollector(cfg)

    for frame in range(8):
        collector.collect(flock, frame)

    snap = collector.snapshot()
    d = snap.to_dict()

    # Must produce valid JSON (no inf, no NaN, no ndarray)
    serialized = json.dumps(d)
    loaded = json.loads(serialized)

    # Verify P9 fast fields are present and valid
    assert isinstance(loaded["nematic_S"], (int, float))
    assert isinstance(loaded["silhouette_2d"], (int, float))
    assert isinstance(loaded["velocity_deviation"], (int, float))
    assert isinstance(loaded["normalized_angular_momentum"], (int, float))

    # Expensive fields: either null or valid numeric
    for field in ["suggested_m", "eta_m", "msd_slope", "hull_volume"]:
        val = loaded[field]
        assert val is None or isinstance(val, (int, float)), (
            f"{field} should be null or numeric, got {type(val).__name__}"
        )
