"""Field/blob mode tests — S2.A5 field-term composition, S2.A9 preset table verification.

O(N) scaling, shell force, alignment, noise, slot repulsion,
tangential orbital, buoyancy, curl flow, fold noise, drag, drift,
floating boundary, edge cases.

Split out of test_field.py (file-size split). FieldConfigWiring moved
out to test_field_config_wiring.py (further file-size split).
"""

import warnings

import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock
from pymurmur.physics.forces.field import (
    _compute_slot_repulsion,
    field_forces,
)
from test.helpers import _call_force


class TestFieldTermComposition:
    """S2.A5: FIELD_TERMS/composeForces composition contract."""

    def test_target_pull_formula(self):
        """S2.A5: F_target_pull = (T-p)/U * coh * target_pull."""
        from pymurmur.physics.forces.field import _compute_target_pull

        positions_active = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        targets = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        U = 10.0
        cohesion = 2.0
        target_pull = 0.5

        result = _compute_target_pull(positions_active, targets, U, cohesion, target_pull)
        expected = (targets - positions_active) / U * cohesion * target_pull
        np.testing.assert_allclose(result, expected)

    def test_target_pull_zero_gain_returns_zero(self):
        from pymurmur.physics.forces.field import _compute_target_pull

        positions_active = np.zeros((5, 3), dtype=np.float32)
        targets = np.ones((5, 3), dtype=np.float32)
        result = _compute_target_pull(positions_active, targets, 10.0, 1.0, 0.0)
        assert np.allclose(result, 0.0)

    def test_target_pull_no_longer_dead_config_field(self):
        """S2.A5: field_target_pull (Part III C3 deferral) now drives real force."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 20
        cfg.field_cohesion = 1.0
        cfg.field_target_pull = 0.5
        cfg.field_noise = 0.0
        cfg.disabled_terms = [
            "shell", "slot_repulsion", "tangential", "buoyancy",
            "curl_flow", "fold_noise", "viscous_drag", "drift_alignment",
            "floating_boundary",
        ]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        _call_force(field_forces, flock, cfg)
        assert not np.allclose(flock.accelerations[flock.active], 0.0)

    def test_slot_repulsion_mod_wraps_first_and_last(self):
        """S2.A5: birds at opposite ends of the active-index ordering now
        interact via offset=1's mod-wrap (previously they never paired,
        an artefact of index ordering, not physical distance)."""

        n = 10
        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            positions[i] = [i * 1000.0, 0.0, 0.0]
        # Bird 9 (last) placed right next to bird 0 (first) — only the
        # offset=1 wrap pairing (9 -> 0) can explain a force between them.
        positions[9] = positions[0] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        active = np.ones(n, dtype=bool)

        F = _compute_slot_repulsion(positions, active, n, U=100.0, separation=1.0, chase_strength=0.0)

        assert not np.allclose(F[0], 0.0), "Mod-wrapped pairing (9->0) should produce force on bird 0"
        assert not np.allclose(F[9], 0.0), "Mod-wrapped pairing (9->0) should produce force on bird 9"
        np.testing.assert_allclose(F[0], -F[9], atol=1e-4)  # action-reaction

    def test_disabled_terms_unknown_name_warns(self):
        """S2.A5: an unrecognized disabled_terms entry warns instead of silently no-op'ing."""
        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        cfg.disabled_terms = ["not_a_real_term"]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        with pytest.warns(UserWarning, match="unknown term name"):
            _call_force(field_forces, flock, cfg)

    def test_disabled_terms_known_names_do_not_warn(self):
        """Sanity: real term names never trigger the unknown-name warning."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        cfg.disabled_terms = [term.name for term in FIELD_TERMS]

        flock = PhysicsFlock(cfg)
        flock.accelerations[:] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _call_force(field_forces, flock, cfg)

    def test_full_step_equals_sum_of_individually_isolated_terms(self):
        """S2.A5: Σ (each term run alone via disabled_terms) == full step
        with every term enabled — proves FIELD_TERMS/composeForces sums
        linearly, as the contract requires."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 25
        cfg.seed = 3
        cfg.max_force = 1e6  # disable clamping so the sum stays exact
        cfg.field_target_pull = 0.3
        all_names = [term.name for term in FIELD_TERMS]

        def run(disabled: list[str]) -> np.ndarray:
            cfg.disabled_terms = disabled
            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0
            _call_force(field_forces, flock, cfg)
            return flock.accelerations[flock.active].copy()

        full = run([])
        total = np.zeros_like(full)
        for name in all_names:
            isolate_disabled = [n for n in all_names if n != name]
            total += run(isolate_disabled)

        np.testing.assert_allclose(full, total, atol=1e-3)

    def test_disabling_one_term_changes_sum_by_exactly_its_contribution(self):
        """S2.A5: full − (full with X disabled) == X run in isolation."""
        from pymurmur.physics.forces.field import FIELD_TERMS

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 25
        cfg.seed = 3
        cfg.max_force = 1e6
        cfg.field_target_pull = 0.3
        all_names = [term.name for term in FIELD_TERMS]

        def run(disabled: list[str]) -> np.ndarray:
            cfg.disabled_terms = disabled
            flock = PhysicsFlock(cfg)
            flock.accelerations[:] = 0.0
            _call_force(field_forces, flock, cfg)
            return flock.accelerations[flock.active].copy()

        full = run([])
        for name in all_names:
            without = run([name])
            alone = run([n for n in all_names if n != name])
            np.testing.assert_allclose(
                full - without, alone, atol=1e-3,
                err_msg=f"term {name!r}: (full - without) != alone",
            )


# ── S2.A9: field_*.yaml preset table verification ──────────────────

class TestFieldPresets:
    """S2.A9: the seven conf/field_*.yaml presets load with the tabled
    (N, v0, sep, align, coh, chase, trail, threat) values from the
    roadmap2.md P3.10 preset table."""

    _CONF_DIR = None

    @staticmethod
    def _load(name: str) -> SimConfig:
        from pathlib import Path
        conf_dir = Path(__file__).resolve().parents[5] / "conf"
        return SimConfig.from_file(conf_dir / name)

    def test_quiet_roost(self):
        cfg = self._load("field_quiet_roost.yaml")
        assert cfg.num_boids == 3000
        assert cfg.v0 == pytest.approx(0.48)
        assert cfg.field_separation == pytest.approx(0.85)
        assert cfg.field_alignment == pytest.approx(0.65)
        assert cfg.field_cohesion == pytest.approx(1.85)
        assert cfg.field.field_chase_strength == pytest.approx(0.72)
        assert cfg.viz.trails == "velocity"
        assert cfg.predator_enabled is False

    def test_lava_lamp_pure_blob(self):
        cfg = self._load("field_lava_lamp.yaml")
        assert cfg.num_boids == 16000
        assert cfg.field_separation == 0.0
        assert cfg.field_alignment == 0.0
        assert cfg.field_cohesion == 0.0
        assert cfg.field.field_chase_strength == 0.0

    def test_ink_cloud(self):
        cfg = self._load("field_ink_cloud.yaml")
        assert cfg.num_boids == 18000
        assert cfg.v0 == pytest.approx(0.62)
        assert cfg.field_separation == pytest.approx(0.92)
        assert cfg.field_alignment == pytest.approx(0.90)
        assert cfg.field_cohesion == pytest.approx(1.80)
        assert cfg.field.field_chase_strength == pytest.approx(0.82)
        assert cfg.viz.trails == "accumulation"
        assert cfg.predator_enabled is True
        assert cfg.predator.predator_mode == "autonomous"

    def test_predator_ripple(self):
        cfg = self._load("field_predator_ripple.yaml")
        assert cfg.num_boids == 12000
        assert cfg.v0 == pytest.approx(0.78)
        assert cfg.field_separation == pytest.approx(1.05)
        assert cfg.field_alignment == pytest.approx(1.05)
        assert cfg.field_cohesion == pytest.approx(1.15)
        assert cfg.field.field_chase_strength == pytest.approx(0.64)
        assert cfg.viz.trails == "velocity"
        assert cfg.predator_enabled is True
        assert cfg.predator.predator_mode == "orbit"

    def test_vacuole(self):
        cfg = self._load("field_vacuole.yaml")
        assert cfg.num_boids == 10000
        assert cfg.v0 == pytest.approx(0.68)
        assert cfg.field_separation == pytest.approx(1.12)
        assert cfg.field_alignment == pytest.approx(0.92)
        assert cfg.field_cohesion == pytest.approx(1.25)
        assert cfg.field.field_chase_strength == pytest.approx(0.76)
        assert cfg.viz.trails == "accumulation"
        assert cfg.predator_enabled is True
        assert cfg.predator.predator_vacuole_strength == pytest.approx(0.9)

    def test_silk_sheet(self):
        cfg = self._load("field_silk_sheet.yaml")
        assert cfg.num_boids == 14000
        assert cfg.v0 == pytest.approx(0.46)
        assert cfg.field_separation == pytest.approx(0.92)
        assert cfg.field_alignment == pytest.approx(1.10)
        assert cfg.field_cohesion == pytest.approx(1.10)
        assert cfg.field.field_chase_strength == pytest.approx(0.68)
        assert cfg.viz.trails == "velocity"
        assert cfg.predator_enabled is False

    def test_storm_turn(self):
        cfg = self._load("field_storm_turn.yaml")
        assert cfg.num_boids == 16000
        assert cfg.v0 == pytest.approx(0.90)
        assert cfg.field_separation == pytest.approx(1.10)
        assert cfg.field_alignment == pytest.approx(1.15)
        assert cfg.field_cohesion == pytest.approx(1.25)
        assert cfg.field.field_chase_strength == pytest.approx(0.42)
        assert cfg.viz.trails == "velocity"
        assert cfg.predator_enabled is True
        assert cfg.predator.predator_mode == "autonomous"

    @pytest.mark.parametrize("name", [
        "field_quiet_roost.yaml", "field_lava_lamp.yaml", "field_ink_cloud.yaml",
        "field_predator_ripple.yaml", "field_vacuole.yaml", "field_silk_sheet.yaml",
        "field_storm_turn.yaml",
    ])
    def test_all_presets_settle_without_nan(self, name):
        """Loading + a short headless run stays finite for every preset."""
        from pymurmur.simulation.engine import SimulationEngine
        cfg = self._load(name)
        cfg.num_boids = 40  # override for a fast smoke run
        engine = SimulationEngine(cfg)
        engine.run_headless(steps=10)
        assert np.isfinite(engine.flock.positions).all()
        assert np.isfinite(engine.flock.velocities).all()
