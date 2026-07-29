"""CLI end-to-end tests — `--set`, `--print-config`, `--fullscreen` flags.

P10.5 features (Phase 10 — UX & Tooling).

P10.6 φp+φa constraint-enforcement tests live in
test_cli_e2e_phi_constraint.py (file-size split of this file).
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

    def test_set_missing_equals_exits(self, monkeypatch):
        """`--set badformat` (missing =) prints usage and exits."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--set", "badformat", "--no-viz",
        ])
        with pytest.raises(SystemExit):
            main()

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

    def test_set_overrides_config_field(self, monkeypatch):
        """`--set spatial.separation_weight=6` is reflected in config."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
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

    def test_set_repeated_flags_accumulate(self, monkeypatch):
        """Multiple --set flags stack without overwriting each other."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
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

    def test_set_int_field_type_coerced(self, monkeypatch):
        """`--set projection.sigma=8` assigns int, not str."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
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

    def test_print_config_with_defaults(self, monkeypatch, capsys):
        """`--print-config` without --set shows default values."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--print-config", "--no-viz",
        ])
        main()
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_print_config_exits_before_engine_init(self, monkeypatch):
        """`--print-config` prints and exits without starting the simulation."""
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main

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
    """D4: `--fullscreen` — parsed AND applied to pygame window."""

    def test_fullscreen_flag_parsed(self, monkeypatch):
        """`--fullscreen` sets args.fullscreen = True."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--fullscreen",
        ])
        args = parse_args()
        assert args.fullscreen is True

    def test_fullscreen_defaults_false(self, monkeypatch):
        """`--fullscreen` absent → defaults to False."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", ["pymurmur"])
        args = parse_args()
        assert args.fullscreen is False

    def test_fullscreen_code_path_exists(self):
        """D4: __main__.py contains the fullscreen→FULLSCREEN wiring.

        The 2-line code path at lines 367-374:
            if args.fullscreen: flags |= pygame.FULLSCREEN
        is trivially verifiable by static check.
        """
        import pymurmur.__main__ as main_module
        source = __import__("inspect").getsource(main_module.main)
        assert "pygame.FULLSCREEN" in source, (
            "D4: main() must reference pygame.FULLSCREEN"
        )
        assert "args.fullscreen" in source, (
            "D4: main() must check args.fullscreen"
        )

# P10.5: Additional CLI flag edge cases

class TestSetEdgeCases:
    """P10.5: Edge cases for --set flag parsing."""

    def test_set_flat_key(self, monkeypatch):
        """P10.5: --set with a flat key (no dot) accesses top-level config."""
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
            "--set", "num_boids=77",
            "--no-viz",
        ])
        main()
        cfg = configs_seen[0]
        assert cfg.num_boids == 77

    def test_set_too_many_dots_exits(self, monkeypatch):
        """P10.5: --set a.b.c=99 (3 parts) exits with an error."""
        from pymurmur.__main__ import main

        monkeypatch.setattr(sys, "argv", [
            "pymurmur",
            "--set", "a.b.c=99",
            "--no-viz",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_coerce_value_str_fallback(self):
        """P10.5: _coerce_value falls back to str for non-numeric values."""
        from pymurmur.__main__ import _coerce_value
        assert _coerce_value("hello") == "hello"
        assert _coerce_value("true") == "true"

    def test_coerce_value_int_and_float(self):
        """P10.5: _coerce_value parses int and float correctly."""
        from pymurmur.__main__ import _coerce_value
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)
        assert _coerce_value("3.14") == pytest.approx(3.14)
        assert isinstance(_coerce_value("3.14"), float)


class TestLightScheme:
    """P10.5: --light-scheme flag."""

    def test_light_scheme_flag_parsed(self, monkeypatch):
        """P10.5: --light-scheme is recognized by parse_args."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", ["pymurmur", "--light-scheme"])
        args = parse_args()
        assert args.light_scheme is True

    def test_light_scheme_defaults_false(self, monkeypatch):
        """P10.5: --light-scheme absent -> defaults to False."""
        from pymurmur.__main__ import parse_args

        monkeypatch.setattr(sys, "argv", ["pymurmur"])
        args = parse_args()
        assert args.light_scheme is False
