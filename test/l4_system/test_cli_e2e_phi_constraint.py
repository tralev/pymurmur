"""CLI end-to-end tests — P10.6: φp+φa ≤ 1 enforcement after --set.

Split out of test_cli_e2e.py (file-size split) — --set/--print-config/
--fullscreen/edge-case tests stay in the original.
"""

import sys

import pytest

pytestmark = [pytest.mark.guard, pytest.mark.e2e]


# ──────────────────────────────────────────────────────────────────────
# P10.6: φp+φa ≤ 1 enforcement after --set overrides
# ──────────────────────────────────────────────────────────────────────

class TestPhiConstraintCLI:
    """P10.6: After --set, φp + φa ≤ 1 is enforced by _enforce_phi_cli."""

    def test_both_at_limit_sum_does_not_exceed_one(self, monkeypatch):
        """--set phi_p=0.8 --set phi_a=0.8 → constraint enforced, sum ≤ 1."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.80",
            "--set", "phi_a=0.80",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, (
            f"φp={cfg.projection.phi_p:.4f} + φa={cfg.phi_a:.4f} = {total:.4f}"
        )

    def test_one_at_one_other_zero(self, monkeypatch):
        """--set phi_p=1.0 forces phi_a to 0.0."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=1.00",
            "--set", "phi_a=0.50",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.projection.phi_p == pytest.approx(1.0)
        assert cfg.phi_a == pytest.approx(0.0)

    def test_sum_within_limit_no_change(self, monkeypatch):
        """--set with sum ≤ 1.0 leaves both values unchanged."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.30",
            "--set", "phi_a=0.40",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.projection.phi_p == pytest.approx(0.30)
        assert cfg.phi_a == pytest.approx(0.40)

    def test_equal_values_reduces_phi_a(self, monkeypatch):
        """--set both to 0.9 (equal) → phi_p ≥ phi_a so phi_a reduced."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.90",
            "--set", "phi_a=0.90",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10
        # phi_p ≥ phi_a → phi_a is reduced
        assert cfg.projection.phi_p == pytest.approx(0.90)
        assert cfg.phi_a == pytest.approx(0.10)

    def test_phi_a_larger_reduces_phi_p(self, monkeypatch):
        """--set phi_p=0.3 phi_a=0.8 → φa > φp so φp reduced to 0.2."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.30",
            "--set", "phi_a=0.80",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10
        # φa=0.8 > φp=0.3 → φp should be reduced to 0.2
        assert cfg.phi_a == pytest.approx(0.80)
        assert cfg.projection.phi_p == pytest.approx(0.20)

    def test_near_limit_values_enforced(self, monkeypatch):
        """--set phi_p=0.99 phi_a=0.90 → sum=1.89 → φa clipped to 0.01."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.99",
            "--set", "phi_a=0.90",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10
        assert cfg.projection.phi_p == pytest.approx(0.99)
        assert cfg.phi_a == pytest.approx(0.01)

    def test_reverse_order_same_result(self, monkeypatch):
        """--set order doesn't matter — enforcement runs after all overrides."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "phi_a=0.80",
            "--set", "projection.phi_p=0.30",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10
        # Same result as test_phi_a_larger_reduces_phi_p — φa > φp so φp reduced
        assert cfg.phi_a == pytest.approx(0.80)
        assert cfg.projection.phi_p == pytest.approx(0.20)

    def test_sum_exactly_one_no_change(self, monkeypatch):
        """P10.6: --set with phi_p+phi_a=1.0 exactly leaves both unchanged."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless", _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.phi_p=0.40",
            "--set", "phi_a=0.60",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.projection.phi_p == pytest.approx(0.40)
        assert cfg.phi_a == pytest.approx(0.60)

