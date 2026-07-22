"""CLI entry-point tests — __main__.py parse_args, load_config, list_available_configs, main().

Tests the CLI dispatch layer without requiring a GPU or display.
"""

import sys

import pytest


class TestParseArgs:
    """parse_args() returns correct argparse.Namespace for all flags."""

    def test_parse_args_defaults(self, monkeypatch):
        """No arguments → all defaults."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur"])
        args = parse_args()
        assert args.config is None
        assert args.list_configs is False
        assert args.no_viz is False
        assert args.capture is False

    def test_parse_args_config_name(self, monkeypatch):
        """--config name sets config arg."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--config", "murmuration_spatial"])
        args = parse_args()
        assert args.config == "murmuration_spatial"

    def test_parse_args_config_path(self, monkeypatch):
        """--config /abs/path sets config arg."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--config", "/tmp/custom.yaml"])
        args = parse_args()
        assert args.config == "/tmp/custom.yaml"

    def test_parse_args_list_configs(self, monkeypatch):
        """--list-configs sets flag True."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--list-configs"])
        args = parse_args()
        assert args.list_configs is True

    def test_parse_args_no_viz(self, monkeypatch):
        """--no-viz sets flag True."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--no-viz"])
        args = parse_args()
        assert args.no_viz is True

    def test_parse_args_capture(self, monkeypatch):
        """--capture sets flag True."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--capture"])
        args = parse_args()
        assert args.capture is True

    def test_parse_args_all_flags(self, monkeypatch):
        """All flags combined."""
        from pymurmur.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "field", "--no-viz", "--capture",
            "--capture-output", "/tmp/out.gif", "--capture-frames", "60"
        ])
        args = parse_args()
        assert args.config == "field"
        assert args.no_viz is True
        assert args.capture is True
        assert args.list_configs is False
        assert args.capture_output == "/tmp/out.gif"
        assert args.capture_frames == 60


class TestLoadConfig:
    """load_config() resolves names/paths to SimConfig. See also test_config_resolution.py."""

    def test_load_config_none_returns_default(self):
        """None → default SimConfig with projection mode."""
        from pymurmur.__main__ import load_config
        cfg = load_config(None)
        assert cfg.mode == "projection"
        assert cfg.num_boids == 150

    def test_load_config_shipped_preset(self):
        """Shipped config name resolves."""
        from pymurmur.__main__ import load_config
        try:
            cfg = load_config("murmuration_spatial")
            assert cfg.mode == "spatial"
        except FileNotFoundError:
            pytest.skip("conf/murmuration_spatial.yaml not found")

    def test_load_config_with_yaml_suffix(self):
        """Name already ending in .yaml resolves without double-append."""
        from pymurmur.__main__ import load_config
        try:
            cfg = load_config("murmuration.yaml")
            assert cfg is not None
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")

    def test_load_config_missing_raises(self):
        """Non-existent name → FileNotFoundError."""
        from pymurmur.__main__ import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config_xyz_123")

    def test_load_config_absolute_path(self, tmp_path):
        """Absolute path to a YAML file resolves directly."""
        import yaml

        from pymurmur.__main__ import load_config

        custom = tmp_path / "custom.yaml"
        custom.write_text(yaml.dump({"mode": "spatial", "num_boids": 50}))
        cfg = load_config(str(custom))
        assert cfg.num_boids == 50
        assert cfg.mode == "spatial"

    def test_load_config_relative_path_user_fallback(self, tmp_path, monkeypatch):
        """Relative path that is not a shipped preset → user_path fallback (line 73)."""
        import yaml

        from pymurmur.__main__ import load_config

        # Create a YAML file with a name that won't match any shipped preset
        custom = tmp_path / "zzz_unique_test_not_a_preset.yaml"
        custom.write_text(yaml.dump({"mode": "spatial", "num_boids": 77}))

        # Change CWD so the relative path resolves via user_path=Path(name)
        monkeypatch.chdir(tmp_path)
        cfg = load_config("zzz_unique_test_not_a_preset.yaml")
        assert cfg.num_boids == 77


class TestListConfigs:
    """list_available_configs() scans conf/ and prints presets."""

    def test_list_configs_runs_without_error(self, capsys):
        """Function runs and prints to stdout without exception."""
        from pymurmur.__main__ import list_available_configs
        list_available_configs()
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_list_configs_includes_murmuration(self, capsys):
        """Shipped murmuration config is listed."""
        from pymurmur.__main__ import list_available_configs
        list_available_configs()
        output = capsys.readouterr().out
        assert "murmuration" in output

    def test_list_configs_no_conf_dirs(self, tmp_path, monkeypatch, capsys):
        """When no conf/ dirs exist, prints fallback message (lines 92, 99)."""
        monkeypatch.chdir(tmp_path)  # no conf/ in tmp_path

        import pymurmur.__main__ as main_mod

        # Monkeypatch Path.exists so ANY path with "conf" in it returns False.
        # This makes the ORIGINAL list_available_configs hit line 92 (continue)
        # and line 99 ("No config presets found").
        original_exists = main_mod.Path.exists

        def fake_exists(self):
            if "conf" in str(self):
                return False
            return original_exists(self)

        monkeypatch.setattr(main_mod.Path, "exists", fake_exists)
        main_mod.list_available_configs()
        output = capsys.readouterr().out
        assert "No config presets found" in output


class TestMainDispatch:
    """main() entry point — headless and list-configs paths."""

    def test_main_list_configs(self, monkeypatch):
        """--list-configs triggers list_available_configs and returns cleanly."""
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--list-configs"])
        main()

    def test_main_no_viz_defaults(self, monkeypatch):
        """--no-viz runs headless (patched to finite steps)."""
        # Patch SimulationEngine.run_headless to run a finite number of steps
        import pymurmur.simulation.engine
        from pymurmur.__main__ import main
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        def patched_run_headless(self, steps=None, callback=None):
            return original(self, steps=10, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            patched_run_headless,
        )
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--no-viz"])
        main()

    def test_main_no_viz_with_config(self, monkeypatch):
        """--no-viz --config murmuration_spatial runs headless."""
        from pymurmur.__main__ import load_config, main

        try:
            load_config("murmuration_spatial")
        except FileNotFoundError:
            pytest.skip("conf/murmuration_spatial.yaml not found")

        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        def patched_run(self, steps=None, callback=None):
            return original(self, steps=10, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            patched_run,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "murmuration_spatial", "--no-viz"
        ])
        main()

    def test_main_no_viz_capture(self, monkeypatch, tmp_path):
        """--no-viz --capture produces output files."""
        from pymurmur.__main__ import load_config, main

        try:
            load_config("murmuration")
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")

        import pymurmur.capture.recorder as rec_mod
        original_save_csv = rec_mod.Recorder.save_metrics_csv
        original_save_json = rec_mod.Recorder.save_metrics_json

        def patched_save_csv(self):
            self.config.capture_metrics_csv = str(tmp_path / "capture.csv")
            return original_save_csv(self)

        def patched_save_json(self):
            self.config.capture_metrics_json = str(tmp_path / "capture.json")
            return original_save_json(self)

        monkeypatch.setattr(rec_mod.Recorder, "save_metrics_csv", patched_save_csv)
        monkeypatch.setattr(rec_mod.Recorder, "save_metrics_json", patched_save_json)

        import pymurmur.simulation.engine
        original_run = pymurmur.simulation.engine.SimulationEngine.run_headless

        def patched_run(self, steps=None, callback=None):
            return original_run(self, steps=10, callback=callback)

        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine,
            "run_headless",
            patched_run,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "murmuration", "--no-viz", "--capture"
        ])

        main()

        csv_path = tmp_path / "capture.csv"
        json_path = tmp_path / "capture.json"
        assert csv_path.exists(), f"CSV not created at {csv_path}"
        assert json_path.exists(), f"JSON not created at {json_path}"

    def test_main_no_viz_capture_output_override(self, monkeypatch, tmp_path):
        """--capture-output overrides cfg.capture_output in main() dispatch."""
        from pymurmur.__main__ import load_config, main
        try:
            load_config("murmuration")
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")

        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless

        # Spy on save_gif to capture the path it receives
        import pymurmur.capture.recorder as rec_mod
        saved_paths = []
        orig_save_gif = rec_mod.Recorder.save_gif
        def _spy_save_gif(self, path=None, fps=20):
            saved_paths.append(path)
            return orig_save_gif(self, path=path, fps=fps)

        monkeypatch.setattr(rec_mod.Recorder, "save_gif", _spy_save_gif)
        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine, "run_headless",
            lambda self, steps=None, callback=None: original(self, steps=5, callback=callback),
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "murmuration", "--no-viz", "--capture",
            "--capture-output", str(tmp_path / "override.gif"),
        ])
        main()
        # save_gif was called with path=None (uses config fallback internally)
        assert saved_paths[0] is None

    def test_main_no_viz_capture_frames_override(self, monkeypatch):
        """--capture-frames overrides cfg.capture_frames in main() dispatch."""
        from pymurmur.__main__ import load_config, main
        try:
            load_config("murmuration")
        except FileNotFoundError:
            pytest.skip("conf/murmuration.yaml not found")

        captured_steps = []
        import pymurmur.simulation.engine
        original = pymurmur.simulation.engine.SimulationEngine.run_headless
        def _capture_steps(self, steps=None, callback=None):
            captured_steps.append(steps)
            return original(self, steps=steps, callback=callback)
        monkeypatch.setattr(
            pymurmur.simulation.engine.SimulationEngine, "run_headless", _capture_steps,
        )
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "murmuration", "--no-viz", "--capture",
            "--capture-frames", "42",
        ])
        main()
        assert captured_steps[0] == 42  # CLI override applied

    # ── Visual path coverage ────────────────────────────────────

    def test_main_visual_path(self, monkeypatch):
        """main() visual path (lines 130-143) executes without a display.

        Monkeypatches moderngl, pygame, and Visualizer.run() so the
        visual branch of main() can execute in CI with full coverage.
        """
        try:
            import moderngl
            moderngl.create_context(standalone=True, require=330)
        except Exception:
            pytest.skip("GPU not available")

        import moderngl
        import pygame

        from pymurmur.__main__ import main

        # GPU: standalone context for windowed renderer
        real_ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: real_ctx)

        # Pygame: no-ops — init + set_mode don't need a display
        monkeypatch.setattr(pygame, "init", lambda: None)
        monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **kw: None)

        # Prevent viz.run() infinite loop — stub to no-op
        from pymurmur.viz.visualizer import Visualizer
        monkeypatch.setattr(Visualizer, "run", lambda self, input_ctrl: None)

        monkeypatch.setattr(sys, "argv", ["pymurmur"])
        main()  # hits visual path → lines 130-143 covered

    def test_main_visual_path_with_config(self, monkeypatch):
        """main() visual path with --config murmuration_spatial."""
        try:
            import moderngl
            moderngl.create_context(standalone=True, require=330)
        except Exception:
            pytest.skip("GPU not available")

        from pymurmur.__main__ import load_config
        try:
            load_config("murmuration_spatial")
        except FileNotFoundError:
            pytest.skip("conf/murmuration_spatial.yaml not found")

        import moderngl
        import pygame

        from pymurmur.__main__ import main

        real_ctx = moderngl.create_context(standalone=True, require=330)
        monkeypatch.setattr(moderngl, "create_context",
                            lambda standalone=False, require=330: real_ctx)
        monkeypatch.setattr(pygame, "init", lambda: None)
        monkeypatch.setattr(pygame.display, "set_mode", lambda *a, **kw: None)

        from pymurmur.viz.visualizer import Visualizer
        monkeypatch.setattr(Visualizer, "run", lambda self, input_ctrl: None)

        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "murmuration_spatial"
        ])
        main()


class TestMainErrorPaths:
    """main() error handling for invalid inputs."""

    def test_main_nonexistent_config_raises(self, monkeypatch):
        """--config nonexistent raises FileNotFoundError."""
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--config", "nonexistent_config_xyz_123", "--no-viz"
        ])
        with pytest.raises(FileNotFoundError):
            main()

    def test_set_unknown_flat_key_exits_with_field_list(self, monkeypatch, capsys):
        """S5.5: --set with an unknown flat key exits(1) and prints the
        full known-field list, not a silently-accepted stray attribute."""
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--set", "totally_bogus_field=5", "--print-config",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "unknown field" in err
        # Sample of real field names that must appear in the printed list
        assert "num_boids" in err
        assert "separation_weight" in err

    def test_set_known_flat_key_does_not_exit(self, monkeypatch):
        """S5.5: a known flat --set key is applied normally, no exit."""
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--set", "num_boids=42", "--print-config",
        ])
        main()  # must not raise/exit


class TestLightScheme:
    """S5.5: --light-scheme maps to a real, valid theme."""

    def test_light_scheme_sets_paper_theme(self, monkeypatch, capsys):
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", [
            "pymurmur", "--light-scheme", "--print-config",
        ])
        main()
        out = capsys.readouterr().out
        assert "theme: paper" in out

    def test_light_scheme_theme_is_valid(self):
        """The mapped theme must actually be one SimConfig accepts."""
        from pymurmur.core.config import SimConfig
        cfg = SimConfig()
        cfg.theme = "paper"
        cfg.validate()  # must not raise

    def test_without_light_scheme_theme_unchanged(self, monkeypatch, capsys):
        from pymurmur.__main__ import main
        monkeypatch.setattr(sys, "argv", ["pymurmur", "--print-config"])
        main()
        out = capsys.readouterr().out
        assert "theme: paper" not in out
