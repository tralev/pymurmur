"""I7 — Config validation integration, copy isolation through Recorder.

Split out of test_config_contract.py (file-size split).
"""

from copy import copy

import numpy as np
import pytest


class TestConfigValidationIntegration:
    """IT2: config.validate() catches multi-rule cross-field errors.

    Tests that the aggregated error message contains all violations
    simultaneously, not just the first one encountered.
    """

    def test_validation_predator_enabled_threat_radius_zero(self):
        """IT2a: predator_enabled=True with predator_threat_radius=0 raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.predator_enabled = True
        cfg.predator_threat_radius = 0.0
        cfg.predator_strength = -0.1  # must be ≤ 0 to trigger the violation

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        # Should mention both predator cross-field violations
        assert "predator_enabled=True" in msg.lower() or "predator_threat_radius" in msg.lower(), (
            f"Expected predator validation error, got: {msg}"
        )
        assert "predator_strength" in msg.lower(), (
            f"Expected predator_strength error too, got: {msg}"
        )

    def test_validation_roosting_enabled_roost_outside_domain(self):
        """IT2b: roosting_enabled=True with roost outside domain bounds raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.roosting_enabled = True
        # Roost at (2000, 100, 50) — x=2000 exceeds domain width=1000
        cfg.ecology_roost = (2000.0, 100.0, 50.0)

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "roosting_enabled=True" in msg.lower() or "ecology_roost" in msg.lower(), (
            f"Expected roost-outside-domain error, got: {msg}"
        )
        assert "outside domain" in msg.lower(), (
            f"Expected 'outside domain' in error, got: {msg}"
        )

    def test_validation_vicsek_radius_ordering_violation(self):
        """IT2c: vicsek_radius_influence <= vicsek_radius_avoid raises."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.vicsek_radius_influence = 1.0
        cfg.vicsek_radius_avoid = 5.0  # influence must be > avoid

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "vicsek_radius_influence" in msg, (
            f"Expected vicsek radius ordering error, got: {msg}"
        )
        assert "vicsek_radius_avoid" in msg, (
            f"Expected mention of vicsek_radius_avoid, got: {msg}"
        )

    def test_validation_invalid_mode_rejected(self):
        """IT2d: invalid mode string raises with helpful message."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.mode = "quantum_flocking"  # not a valid mode

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "mode" in msg.lower(), (
            f"Expected mode validation error, got: {msg}"
        )
        assert "quantum_flocking" in msg, (
            f"Expected invalid mode name in error, got: {msg}"
        )

    def test_validation_multi_rule_errors_aggregated(self):
        """IT2e: Multiple violations produce an aggregated error message
        listing ALL issues, not just the first one encountered."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Violation 1: predator_enabled but threat_radius <= 0
        cfg.predator_enabled = True
        cfg.predator_threat_radius = -1.0
        cfg.predator_strength = -0.5
        # Violation 2: roosting but roost outside domain
        cfg.roosting_enabled = True
        cfg.ecology_roost = (9999.0, 9999.0, 9999.0)
        # Violation 3: invalid mode
        cfg.mode = "not_a_mode"
        # Violation 4: negative domain depth
        cfg.depth = -100.0
        # Violation 5: vicsek radius ordering (vicsek mode but invalid params)
        # Note: vicsek check only applies when mode=="vicsek", so skipped here
        # Violation 6: boundary mode invalid
        cfg.boundary_mode = "wormhole"

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)

        # The error message should aggregate multiple issues
        error_count = msg.count("  - ")
        assert error_count >= 5, (
            f"Expected at least 5 aggregated issues, got {error_count}.\n"
            f"Message:\n{msg}"
        )

        # Each specific violation should appear
        assert "predator_enabled=True" in msg.lower() or "predator_threat_radius" in msg.lower(), (
            f"Missing predator violation in aggregated message:\n{msg}"
        )
        assert "roosting_enabled=True" in msg.lower() or "ecology_roost" in msg.lower(), (
            f"Missing roost violation in aggregated message:\n{msg}"
        )
        assert "not_a_mode" in msg, (
            f"Missing mode violation in aggregated message:\n{msg}"
        )
        assert "-100.0" in msg or "depth" in msg.lower(), (
            f"Missing domain.depth violation in aggregated message:\n{msg}"
        )
        assert "wormhole" in msg, (
            f"Missing boundary_mode violation in aggregated message:\n{msg}"
        )

        # Verify the header mentions the count
        assert "SimConfig validation failed" in msg, (
            f"Missing validation header in aggregated message:\n{msg}"
        )

    def test_validation_valid_config_passes_silently(self):
        """IT2f: A default config passes validate() without error."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Should not raise
        try:
            cfg.validate()
        except ValueError as e:
            pytest.fail(f"Default SimConfig should validate cleanly, got: {e}")

    def test_validation_non_numeric_field_caught_by_type_guard(self):
        """IT2g: Type guard catches non-numeric values before comparisons."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        # Inject a non-numeric value that would crash a comparison
        object.__setattr__(cfg._domain, "width", "huge")

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "width" in msg.lower(), (
            f"Expected width type error, got: {msg}"
        )
        assert "numeric" in msg.lower(), (
            f"Expected 'numeric' in type error, got: {msg}"
        )


# ═══════════════════════════════════════════════════════════════════
# I7 High-Priority Integration Tests — cross I7.1 + I7.7 + I6.5
# ═══════════════════════════════════════════════════════════════════


class TestCopyIsolationThroughRecorder:
    """IT3: copy(config) isolation propagates through engine→recorder→metrics.

    Two engines with copy(config) but different v0 values, each with its
    own Recorder. After headless runs, metrics must differ — proving that
    sub-config isolation survives the full pipeline.
    """

    def test_copy_config_preserves_original_after_mutation(self):
        """IT3a: Mutating copy(config).v0 does NOT mutate original config.v0."""
        from pymurmur import SimConfig

        cfg = SimConfig()
        cfg.num_boids = 100
        cfg.v0 = 4.0
        cfg.seed = 42

        cfg2 = copy(cfg)
        cfg2.v0 = 8.0

        assert cfg.v0 == 4.0, (
            f"copy(config).v0 = 8.0 mutated original to {cfg.v0}"
        )
        assert cfg2.v0 == 8.0, (
            f"Copied config should have v0=8.0, got {cfg2.v0}"
        )

    def test_two_engines_with_copy_produce_different_metrics(self):
        """IT3b: Two engines with copy(config)+different v0 produce different metrics.

        Key assertion: speed_avg or alpha from Recorder differ between the two runs.
        This verifies isolation propagates through the full engine→recorder→metrics
        pipeline, not just positions.
        """
        from pymurmur import Recorder, SimConfig, SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 50
        cfg.v0 = 4.0
        cfg.seed = 42
        cfg.metrics_detail_level = 1

        # Engine 1: original config, slow speed
        e1 = SimulationEngine(cfg)
        r1 = Recorder(e1, cfg)

        # Engine 2: copy with faster speed
        cfg2 = copy(cfg)
        cfg2.v0 = 8.0
        e2 = SimulationEngine(cfg2)
        r2 = Recorder(e2, cfg2)

        # Verify original config untouched
        assert cfg.v0 == 4.0, (
            f"copy(config).v0 = 8.0 leaked to original: {cfg.v0}"
        )

        # Run both headless for 30 steps
        e1.run_headless(steps=30, callback=r1.on_frame)
        e2.run_headless(steps=30, callback=r2.on_frame)

        # Both recorders must have captured metrics
        assert len(r1.metrics_history) == 30, (
            f"Recorder 1: expected 30 metrics, got {len(r1.metrics_history)}"
        )
        assert len(r2.metrics_history) == 30, (
            f"Recorder 2: expected 30 metrics, got {len(r2.metrics_history)}"
        )

        # Average speed should differ (different v0)
        avg_speed_1 = np.mean([m["speed_avg"] for m in r1.metrics_history])
        avg_speed_2 = np.mean([m["speed_avg"] for m in r2.metrics_history])

        assert avg_speed_1 != avg_speed_2, (
            f"Engines with v0=4.0 and v0=8.0 produced same average speed "
            f"({avg_speed_1:.3f}). Metrics isolation is broken."
        )
        # Higher v0 should produce higher speed
        assert avg_speed_2 > avg_speed_1, (
            f"v0=8.0 engine should be faster than v0=4.0, but got "
            f"{avg_speed_2:.3f} ≤ {avg_speed_1:.3f}"
        )

    def test_copy_config_metrics_diverge_over_time(self):
        """IT3c: copy(config) isolation causes metric trajectories to diverge.

        Per-frame alpha values should differ between the two runs, not just
        aggregate stats. This catches shallow isolation where only the
        initial state differs but the config leak causes reconvergence.
        """
        from pymurmur import Recorder, SimConfig, SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 30
        cfg.v0 = 3.0
        cfg.seed = 100
        cfg.metrics_detail_level = 1

        cfg2 = copy(cfg)
        cfg2.v0 = 7.0

        e1 = SimulationEngine(cfg)
        r1 = Recorder(e1, cfg)
        e2 = SimulationEngine(cfg2)
        r2 = Recorder(e2, cfg2)

        e1.run_headless(steps=50, callback=r1.on_frame)
        e2.run_headless(steps=50, callback=r2.on_frame)

        # Per-frame alpha should differ for at least half the frames
        alpha1 = np.array([m["alpha"] for m in r1.metrics_history])
        alpha2 = np.array([m["alpha"] for m in r2.metrics_history])

        # They should not be identical on every frame
        differences = np.abs(alpha1 - alpha2)
        divergent_frames = np.sum(differences > 1e-10)

        assert divergent_frames > 0, (
            "All 50 frames have identical alpha — copy(config) isolation is broken"
        )
        # At least a few frames should differ meaningfully
        assert divergent_frames >= 5, (
            f"Only {divergent_frames}/50 frames differ in alpha — "
            f"isolation may be shallow or config leaked"
        )

    def test_three_way_copy_chain_isolation(self):
        """IT3d: A→copy→B, A→copy→C — three-way copy chain isolation."""
        from pymurmur import SimConfig

        base = SimConfig()
        base.num_boids = 100
        base.v0 = 5.0
        base.seed = 42
        base.separation_weight = 4.5

        cfg_a = copy(base)
        cfg_a.separation_weight = 1.0

        cfg_b = copy(base)
        cfg_b.separation_weight = 10.0

        # Base must be unchanged
        assert base.v0 == 5.0
        assert base.separation_weight == 4.5

        # Each copy must have its own value
        assert cfg_a.separation_weight == 1.0
        assert cfg_b.separation_weight == 10.0

        # All three must be different objects (not shared sub-configs)
        assert cfg_a.separation_weight != cfg_b.separation_weight
        assert cfg_a.separation_weight != base.separation_weight
        assert cfg_b.separation_weight != base.separation_weight

        # Verify sub-config objects are distinct (not shared references)
        assert cfg_a._spatial is not base._spatial, (
            "copy(config) shared _spatial sub-config — deep copy broken"
        )
        assert cfg_b._spatial is not base._spatial, (
            "copy(config) shared _spatial sub-config — deep copy broken"
        )
        assert cfg_a._spatial is not cfg_b._spatial, (
            "Two copies share same _spatial object — deep copy broken"
        )


