"""CLI end-to-end tests — `--set`, `--print-config`, `--fullscreen` flags.

These flags are roadmap P10.5 features (Phase 10 — UX & Tooling).
Tests are marked xfail with clear reasons until each flag ships.

P14 guard: `pytest -m guard` selects this file so missing flags are
visible in CI as expected failures.
"""

import sys
import pytest

pytestmark = [pytest.mark.guard, pytest.mark.e2e]


# ──────────────────────────────────────────────────────────────────────
# --set key.subkey=value
# ──────────────────────────────────────────────────────────────────────

class TestSetMalformed:
    """Malformed `--set` input — should produce clear errors, not stack traces.

    These are marked `skip` (not `xfail`) because they currently pass for the
    wrong reason: argparse rejects `--set` as an unrecognized flag. Once
    P10.5 adds `--set` to argparse, unskip these and they should exercise the
    actual malformed-input validation path.
    """

    @pytest.mark.skip(
        reason="P10.5: --set flag not yet in argparse. Currently passing because "
               "argparse rejects unknown flag; unskip when --set ships."
    )
    def test_set_missing_equals_exits(self, monkeypatch):
        """`--set badformat` (missing =) prints usage and exits."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--set", "badformat", "--no-viz",
        ])
        with pytest.raises(SystemExit):
            main()

    @pytest.mark.skip(
        reason="P10.5: --set flag not yet in argparse. Currently passing because "
               "argparse rejects unknown flag; unskip when --set ships."
    )
    def test_set_empty_value_handled(self, monkeypatch):
        """`--set key=` (empty value) is handled gracefully."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--set", "spatial.separation_weight=", "--no-viz",
        ])
        # Should either error cleanly or treat as default
        # (exact behavior TBD when flag is implemented)
        try:
            main()
        except SystemExit:
            pass  # either outcome is valid — test just proves no crash


class TestSetFlag:
    """`--set key.subkey=value` — nested config override from CLI."""

    @pytest.mark.xfail(
        reason="P10.5: --set flag not yet implemented (adds argparse + nested key assignment). "
               "NOTE: uses nested config (P2.1) — if P10.5 ships first, switch to flat cfg.accessor."
    )
    def test_set_overrides_config_field(self, monkeypatch):
        """`--set spatial.separation_weight=6` is reflected in config."""
        from pymurmur.__main__ import main

        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "spatial.separation_weight=6",
            "--set", "flock.num_boids=500",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.spatial.separation_weight == 6
        assert cfg.flock.num_boids == 500

    @pytest.mark.xfail(
        reason="P10.5: --set flag not yet implemented"
    )
    def test_set_unknown_key_exits_gracefully(self, monkeypatch):
        """`--set nonexistent.field=99` prints error and exits with code 1."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "nonexistent.field=99",
            "--no-viz",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    @pytest.mark.xfail(
        reason="P10.5: --set flag not yet implemented. "
               "NOTE: uses nested config (P2.1) — if P10.5 ships first, switch to flat cfg.accessor."
    )
    def test_set_repeated_flags_accumulate(self, monkeypatch):
        """Multiple --set flags stack without overwriting each other."""
        from pymurmur.__main__ import main

        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "spatial.separation_weight=3.0",
            "--set", "spatial.alignment_weight=0.9",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.spatial.separation_weight == 3.0
        assert cfg.spatial.alignment_weight == 0.9

    @pytest.mark.xfail(
        reason="P10.5: --set flag not yet implemented. "
               "NOTE: uses nested config (P2.1) — if P10.5 ships first, switch to flat cfg.accessor."
    )
    def test_set_int_field_type_coerced(self, monkeypatch):
        """`--set projection.sigma=8` assigns int, not str."""
        from pymurmur.__main__ import main

        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        configs_seen = []
        def _capture_config(self, steps=None, callback=None):
            configs_seen.append(self.config)
            return original(self, steps=1, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            _capture_config,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "projection.sigma=8",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert isinstance(cfg.projection.sigma, int)
        assert cfg.projection.sigma == 8


# ──────────────────────────────────────────────────────────────────────
# --print-config
# ──────────────────────────────────────────────────────────────────────

class TestPrintConfig:
    """`--print-config` — dump resolved config and exit."""

    @pytest.mark.xfail(
        reason="P10.5: --print-config flag not yet implemented"
    )
    def test_print_config_outputs_yaml(self, monkeypatch, capsys):
        """`--print-config` prints valid YAML to stdout and exits."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--print-config", "--no-viz",
        ])
        main()
        output = capsys.readouterr().out
        # Should contain mode and some config keys
        assert "mode:" in output
        assert "num_boids:" in output or "flock:" in output

    @pytest.mark.xfail(
        reason="P10.5: --print-config flag not yet implemented"
    )
    def test_print_config_reflects_set_overrides(self, monkeypatch, capsys):
        """`--set` + `--print-config` shows the mutated config."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "flock.num_boids=777",
            "--print-config",
            "--no-viz",
        ])
        main()
        output = capsys.readouterr().out
        assert "777" in output

    @pytest.mark.xfail(
        reason="P10.5: --print-config flag not yet implemented"
    )
    def test_print_config_with_defaults(self, monkeypatch, capsys):
        """`--print-config` without --set shows default values."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--print-config", "--no-viz",
        ])
        main()
        output = capsys.readouterr().out
        assert len(output) > 0

    @pytest.mark.xfail(
        reason="P10.5: --print-config flag not yet implemented"
    )
    def test_print_config_exits_before_engine_init(self, monkeypatch):
        """`--print-config` prints and exits without starting the simulation."""
        from pymurmur.__main__ import main
        import pymurmur.simulation.engine

        # Prove engine was never touched — if it is, fail hard
        monkeypatch.setattr(
            pymurmur.simulation.engine, "SimulationEngine",
            lambda *a, **kw: pytest.fail(
                "Engine created — print-config should exit first"
            ),
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--print-config", "--no-viz",
        ])
        main()


# ──────────────────────────────────────────────────────────────────────
# --fullscreen
# ──────────────────────────────────────────────────────────────────────

class TestFullscreen:
    """`--fullscreen` — launch in fullscreen mode."""

    @pytest.mark.xfail(
        reason="P10.5: --fullscreen flag not yet implemented"
    )
    def test_fullscreen_flag_parsed(self, monkeypatch):
        """`--fullscreen` is recognized by parse_args."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--fullscreen",
        ])
        args = parse_args()
        assert args.fullscreen is True

    @pytest.mark.xfail(
        reason="P10.5: --fullscreen flag not yet implemented"
    )
    def test_fullscreen_defaults_false(self, monkeypatch):
        """`--fullscreen` absent → defaults to False."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", ["pymurmur"])
        args = parse_args()
        assert args.fullscreen is False
