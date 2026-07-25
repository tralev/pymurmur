"""D11 — Mode flags honoured by engine; D12 — field_inertia wired through engine.

Split out of test_mode_contract.py (file-size split).
"""


from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock

# ── D11: Mode flags honoured by engine ────────────────────────────


class TestD11ModeFlags:
    """D11: Engine wire-up — speed_mode and owns_positions flags
    are honoured by SimulationEngine._step_physics."""

    def test_influencer_mode_has_owns_positions(self):
        """D11: InfluencerMode.owns_positions = True."""
        from pymurmur.physics.forces.influencer import InfluencerMode
        assert hasattr(InfluencerMode, 'owns_positions')
        assert InfluencerMode.owns_positions is True

    def test_non_influencer_modes_dont_own_positions(self):
        """D11: Non-influencer modes do NOT claim owns_positions."""
        from pymurmur.physics.forces import MODE_REGISTRY
        for name, cls in MODE_REGISTRY.items():
            if name == "influencer":
                continue
            owns = getattr(cls, 'owns_positions', False)
            assert not owns, (
                f"{name} mode must not claim owns_positions=True"
            )

    def test_engine_passes_move_false_for_influencer(self, default_config):
        """D11: For 'influencer' mode, integrate() gets move=False."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.mode = "influencer"
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as mock_integrate:
            eng._step_physics(1.0 / 60.0)

        assert mock_integrate.call_args[1].get('move') is False, (
            "influencer mode should pass move=False"
        )

    def test_engine_passes_move_true_for_spatial(self, default_config):
        """D11: For 'spatial' mode, integrate() gets move=True."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.mode = "spatial"
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as mock_integrate:
            eng._step_physics(1.0 / 60.0)

        assert mock_integrate.call_args[1].get('move', True) is True, (
            "spatial mode should pass move=True"
        )

    def test_speed_mode_wired_from_config(self, default_config):
        """D11: speed_mode is wired from config.spatial.speed_mode."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = default_config
        cfg.spatial.speed_mode = "ceiling"
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as mock_integrate:
            eng._step_physics(1.0 / 60.0)

        assert mock_integrate.call_args[1].get('speed_mode') == "ceiling"

    def test_all_registered_modes_have_needs_index(self):
        """D11: Every registered mode declares needs_index (bool)."""
        from pymurmur.physics.forces import MODE_REGISTRY
        for name, cls in MODE_REGISTRY.items():
            assert hasattr(cls, 'needs_index'), f"{name}: missing needs_index"
            assert isinstance(cls.needs_index, bool), f"{name}: not bool"


# ── D12: field_inertia wired through engine ───────────────────────


class TestD12FieldInertia:
    """D12: field_inertia from config reaches integrate() via
    PhysicsFlock.integrate() with the correct inertia value."""

    def test_inertia_wired_from_config(self):
        """D12: config.field_inertia is passed as inertia to integrate()
        in field mode (the mode the parameter belongs to)."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.field_inertia = 0.42
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as m:
            eng._step_physics(1.0 / 60.0)

        assert m.call_args[1].get('inertia') == 0.42

    def test_inertia_default_value_reaches_integrate(self):
        """D12: Default field_inertia (0.82) reaches integrate() in field mode."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "field"
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as m:
            eng._step_physics(1.0 / 60.0)

        assert m.call_args[1].get('inertia') == 0.82

    def test_inertia_zero_outside_field_mode(self):
        """D12: non-field modes get inertia=0.0 — the raw/clamped lerp
        would soften the hard speed-band contract (P4 acceptance)."""
        from unittest.mock import patch

        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.field_inertia = 0.42
        cfg.num_boids = 10
        eng = SimulationEngine(cfg)

        with patch.object(eng.flock, 'integrate',
                          wraps=eng.flock.integrate) as m:
            eng._step_physics(1.0 / 60.0)

        assert m.call_args[1].get('inertia') == 0.0

    def test_inertia_flows_from_flock_to_boid_integrate(self):
        """D12: inertia parameter flows from PhysicsFlock.integrate()
        to boid.integrate() via the kwarg pass-through."""
        from unittest.mock import patch


        cfg = SimConfig()
        cfg.num_boids = 5
        flock = PhysicsFlock(cfg)

        # Patch boid.integrate to capture the inertia kwarg
        with patch('pymurmur.physics.flock.integrate') as mock_boid:
            flock.integrate(cfg, 1.0/60.0, inertia=0.37)

        assert mock_boid.call_args[1].get('inertia') == 0.37, (
            f"inertia should reach boid.integrate, got "
            f"{mock_boid.call_args[1].get('inertia')}"
        )
