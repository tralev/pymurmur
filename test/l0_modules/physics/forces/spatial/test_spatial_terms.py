"""Modularity pass 9 — SPATIAL_TERMS / composeForces tests.

Split out of test_spatial_variants.py (file-size split, 613 -> under
600 lines) — mirrors test_force_terms.py's role for the generic
ForceTerm/composeForces mechanism, but scoped to spatial mode's actual
term list and SpatialTermContext.
"""

import numpy as np


class TestSpatialTermsComposition:
    def test_spatial_terms_registered_in_order(self):
        from pymurmur.physics.forces.spatial import SPATIAL_TERMS

        assert [t.name for t in SPATIAL_TERMS] == [
            "separation", "alignment", "cohesion", "flow",
            "predator_escape", "forward_thrust",
        ]

    def test_spatial_terms_compose_matches_manual_sum(self):
        """composeForces(ctx, SPATIAL_TERMS, n) must equal the manual
        weighted-sum expression it replaced — direct unit-level check
        alongside golden-trajectory coverage (which proves this for real
        simulation runs). forward_thrust is exercised separately below
        since it depends on velocities/w_fwd, not just the context's
        precomputed arrays."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces._base import composeForces
        from pymurmur.physics.forces.spatial import (
            SPATIAL_TERMS,
            SpatialTermContext,
        )

        n = 4
        rng = np.random.default_rng(1)
        positions = rng.normal(size=(n, 3)).astype(np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)  # w_fwd term stays zero
        active = np.ones(n, dtype=bool)
        sep = rng.normal(size=(n, 3)).astype(np.float32)
        align = rng.normal(size=(n, 3)).astype(np.float32)
        coh = rng.normal(size=(n, 3)).astype(np.float32)
        flow_contrib = rng.normal(size=(n, 3)).astype(np.float32)
        escape_force = rng.normal(size=(n, 3)).astype(np.float32)

        cfg = SimConfig()
        sep_w, align_w, coh_w = cfg.separation_weight, cfg.alignment_weight, cfg.cohesion_weight
        sep_j, align_j, coh_j = 1.2, 0.8, 1.5

        fx = SpatialTermContext(
            config=cfg, positions=positions, velocities=velocities, active=active,
            sep=sep, align=align, coh=coh,
            sep_jitter=sep_j, align_jitter=align_j, coh_jitter=coh_j,
            flow_contrib=flow_contrib, escape_force=escape_force,
        )
        result = composeForces(fx, SPATIAL_TERMS, n=n)

        expected = (
            sep * (sep_w * sep_j) + align * (align_w * align_j)
            + coh * (coh_w * coh_j) + flow_contrib + escape_force
            # forward_thrust: velocities are all zero -> moving mask empty -> 0
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_spatial_terms_escape_force_none_contributes_zero(self):
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces._base import composeForces
        from pymurmur.physics.forces.spatial import (
            SPATIAL_TERMS,
            SpatialTermContext,
        )

        n = 3
        positions = np.zeros((n, 3), dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        active = np.ones(n, dtype=bool)
        zeros = np.zeros((n, 3), dtype=np.float32)
        cfg = SimConfig()

        fx = SpatialTermContext(
            config=cfg, positions=positions, velocities=velocities, active=active,
            sep=zeros, align=zeros, coh=zeros,
            sep_jitter=1.0, align_jitter=1.0, coh_jitter=1.0,
            flow_contrib=zeros, escape_force=None,
        )
        result = composeForces(fx, SPATIAL_TERMS, n=n)
        np.testing.assert_allclose(result, zeros)

    def test_forward_thrust_term_pulls_toward_cruise_speed(self):
        """F_fwd = w_fwd * (v0 - |v|) * v_hat -- a slow-moving bird gets
        a positive thrust along its own heading."""
        from pymurmur.core.config import SimConfig
        from pymurmur.physics.forces.spatial import SpatialTermContext, _term_forward_thrust

        n = 1
        positions = np.zeros((n, 3), dtype=np.float32)
        velocities = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # slow, v0 default is faster
        active = np.ones(n, dtype=bool)
        zeros = np.zeros((n, 3), dtype=np.float32)
        cfg = SimConfig()
        cfg.spatial.w_fwd = 0.5

        fx = SpatialTermContext(
            config=cfg, positions=positions, velocities=velocities, active=active,
            sep=zeros, align=zeros, coh=zeros,
            sep_jitter=1.0, align_jitter=1.0, coh_jitter=1.0,
            flow_contrib=zeros, escape_force=None,
        )
        result = _term_forward_thrust(fx)
        expected_mag = 0.5 * (cfg.v0 - 1.0)
        np.testing.assert_allclose(result[0], [expected_mag, 0.0, 0.0], atol=1e-6)
