"""CLI entry point for the pymurmur 3D murmuration simulation.

Usage:
    python -m pymurmur                                    # defaults (projection, N=150)
    python -m pymurmur --config murmuration_spatial       # conf/murmuration_spatial.yaml
    python -m pymurmur --config /path/to/custom.yaml       # custom config
    python -m pymurmur --config field --no-viz             # headless simulation
    python -m pymurmur --config field --no-viz --capture   # headless + capture
    python -m pymurmur --list-configs                      # show available presets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.config import SimConfig
from .core.logging import (
    cli_err,
    cli_out,
    log_run_footer,
    log_run_header,
    setup_run_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="pymurmur",
        description="3D murmuration simulation and visualisation",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config name (resolves to conf/NAME.yaml) or absolute path",
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List available config presets in conf/",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Probe available capabilities (GPU, numba, etc.) and exit",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Run headlessly without visualisation",
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Capture output (GIF, CSV, JSON) during headless run",
    )
    parser.add_argument(
        "--capture-output", type=str, default=None,
        help="Override GIF output path (default: config capture_output)",
    )
    parser.add_argument(
        "--capture-frames", type=int, default=None,
        help="Override number of frames to capture (default: config capture_frames)",
    )
    # P10.5: CLI flags — --set, --print-config, --fullscreen
    parser.add_argument(
        "--set", action="append", default=[], dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override config fields (repeatable). "
             "Example: --set spatial.separation_weight=6 --set flock.num_boids=500",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print resolved config as YAML and exit (no simulation)",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Launch in fullscreen mode",
    )
    parser.add_argument(
        "--light-scheme", action="store_true",
        help="Use light colour theme instead of dark",
    )
    # S5.6: Run logging
    parser.add_argument(
        "--log-level", type=str, default="warning",
        choices=["debug", "info", "warning"],
        help="Log level for structured run logging (default: warning)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="output",
        help="Directory for log files (default: output)",
    )
    return parser.parse_args()


def load_config(name: str | None) -> SimConfig:
    """Resolve a config name or path to a SimConfig.

    Search order: conf/{name}.yaml → {name} as path → error.
    If name already ends with .yaml, it is used as-is (no double-append).
    """
    if name is None:
        return SimConfig()

    # Strip .yaml if already present (prevent double-append)
    config_name = name
    if name.endswith(".yaml"):
        config_name = name[:-5]

    # Try shipped preset first
    # Use strict=False: shipped presets may carry legacy keys from
    # config refactoring (G5) — the unknown-key guard is for API users.
    package_dir = Path(__file__).parent
    shipped = package_dir / "conf" / f"{config_name}.yaml"
    if shipped.exists():
        return SimConfig.from_file(shipped, strict=False)

    # Also try project-root conf/
    project_conf = Path("conf") / f"{config_name}.yaml"
    if project_conf.exists():
        return SimConfig.from_file(project_conf, strict=False)

    # Try as absolute/relative path (use original name for paths)
    # User-supplied paths use strict=True to surface misconfigurations.
    user_path = Path(name)
    if user_path.exists():
        return SimConfig.from_file(user_path)

    raise FileNotFoundError(
        f"Config '{name}' not found in conf/ or as path.\n"
        f"  Searched: {shipped}, {project_conf}, {user_path}"
    )


def probe_capabilities() -> dict[str, str | None]:
    """Probe optional dependencies and return capability dict.

    Returns a dict mapping capability name to version string
    or None if not available. Used by --probe CLI flag and
    by CI guard-rails to detect environment.
    """
    caps: dict[str, str | None] = {}

    # GPU rendering
    try:
        import moderngl
        caps["moderngl"] = moderngl.__version__  # type: ignore[attr-defined]
    except ImportError:
        caps["moderngl"] = None

    # JIT compilation
    try:
        import numba
        caps["numba"] = numba.__version__
    except ImportError:
        caps["numba"] = None

    # Windowing / input
    try:
        import pygame
        caps["pygame"] = pygame.version.ver
    except ImportError:
        caps["pygame"] = None

    # Spatial indexing
    try:
        import scipy
        caps["scipy"] = scipy.__version__
    except ImportError:
        caps["scipy"] = None

    # MARL bridge
    try:
        import gymnasium
        caps["gymnasium"] = gymnasium.__version__
    except ImportError:
        caps["gymnasium"] = None

    # GPU-free fallback rendering
    try:
        import matplotlib
        caps["matplotlib"] = matplotlib.__version__
    except ImportError:
        caps["matplotlib"] = None

    # Camera math (GPU path)
    try:
        import glm
        caps["PyGLM"] = glm.__version__
    except ImportError:
        caps["PyGLM"] = None

    return caps


def list_available_configs() -> None:
    """Scan conf/ and print all .yaml files with descriptions."""
    # Try project-root conf/ first
    conf_dirs = [Path("conf")]
    package_conf = Path(__file__).parent / "conf"
    if package_conf.exists():
        conf_dirs.append(package_conf)

    found = False
    for conf_dir in conf_dirs:
        if not conf_dir.exists():
            continue
        for f in sorted(conf_dir.glob("*.yaml")):
            first_line = f.read_text().split("\n")[0].lstrip("# ")
            cli_out(f"  {f.stem:30s} — {first_line}")
            found = True

    if not found:
        cli_out("  No config presets found in conf/")


def _apply_set_overrides(cfg: SimConfig, overrides: list[str]) -> None:
    """P10.5: Apply --set KEY=VALUE overrides to a SimConfig.

    Parses dotted keys like 'spatial.separation_weight=6' and
    routes them to the correct sub-config. Exits with error on
    malformed or unknown keys.
    """

    for override in overrides:
        if "=" not in override:
            cli_err(f"Error: --set '{override}' missing '=' separator. "
                    f"Use --set key=value")
            sys.exit(1)
        key, _, value_str = override.partition("=")
        if not key or not value_str:
            cli_err(f"Error: --set '{override}' has empty key or value. "
                    f"Use --set key=value")
            sys.exit(1)

        # Parse dotted key: section.field → setattr(cfg.section, field, value)
        parts = key.split(".")
        if len(parts) == 2:
            section_name, field_name = parts
            if not hasattr(cfg, section_name):
                cli_err(f"Error: --set '{key}': unknown section '{section_name}'. "
                        f"Available sections: "
                        f"{', '.join(_list_sections(cfg))}")
                sys.exit(1)
            section = getattr(cfg, section_name)
            if not hasattr(section, field_name):
                cli_err(f"Error: --set '{key}': unknown field '{field_name}' "
                        f"in section '{section_name}'. "
                        f"Available fields: {', '.join(_list_fields(section))}")
                sys.exit(1)
            # Type coercion: try int, float, then str
            typed_value = _coerce_value(value_str)
            setattr(section, field_name, typed_value)
        elif len(parts) == 1:
            # Flat key: route via flat accessor (handles _SETTER_ONLY too)
            field_name = key
            value = _coerce_value(value_str)
            setattr(cfg, field_name, value)
        else:
            cli_err(f"Error: --set '{key}': too many dots. "
                    f"Use section.field or flat_key")
            sys.exit(1)


def _coerce_value(raw: str) -> int | float | str:
    """Try to coerce a string to int or float; fall back to str."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _list_sections(cfg) -> list[str]:
    """P10.5: List all nested config section names.

    SimConfig.__getattr__ delegates flat access and intercepts
    __dataclass_fields__, so dataclasses.fields() fails.  Walk
    the instance __dict__ instead; every sub-config is a
    dataclass instance stored directly on SimConfig.
    """
    import dataclasses
    sections: list[str] = []
    for name, value in vars(cfg).items():
        if dataclasses.is_dataclass(value):
            sections.append(name)
    return sorted(sections)


def _enforce_phi_cli(cfg: SimConfig) -> None:
    """P10.6: After --set overrides, enforce φp + φa ≤ 1.

    Called after all --set overrides have been applied.  If the
    sum exceeds 1.0, the smaller of the two values is reduced
    to make room.  This matches the interactive behaviour in
    InputControl._enforce_phi_constraint.
    """
    total = cfg.projection.phi_p + cfg.phi_a
    if total > 1.0:
        # Check which was likely set last by looking at which
        # is closer to the full 1.0 — reduce the other.
        if cfg.projection.phi_p >= cfg.phi_a:
            cfg.phi_a = max(0.0, 1.0 - cfg.projection.phi_p)
        else:
            cfg.projection.phi_p = max(0.0, 1.0 - cfg.phi_a)


def _list_fields(section) -> list[str]:
    """P10.5: List all field names on a sub-config section."""
    import dataclasses
    try:
        return sorted(f.name for f in dataclasses.fields(section))
    except TypeError:
        return []


def main() -> None:
    """Main entry point."""
    import time
    args = parse_args()

    if args.probe:
        caps = probe_capabilities()
        cli_out("pymurmur capability probe:")
        for name, version in caps.items():
            status = version if version else "NOT FOUND"
            cli_out(f"  {name:12s}  {status}")
        return

    if args.list_configs:
        list_available_configs()
        return

    cfg = load_config(args.config)

    # S5.6: Setup structured run logging
    _log = setup_run_logging(log_dir=args.log_dir, level=args.log_level)
    seed_str = str(cfg.seed) if cfg.seed is not None else "random"
    config_label = args.config or "defaults"
    log_run_header(config_label, seed_str, cfg.mode, cfg.num_boids)

    # D16: Apply env var overrides BEFORE CLI overrides so CLI wins.
    # Contract: YAML < env < CLI.
    import os
    for _env_key, _cfg_attr in [
        ("CAPTURE_WIDTH", "capture_width"),
        ("CAPTURE_HEIGHT", "capture_height"),
        ("CAPTURE_FRAMES", "capture_frames"),
        ("CAPTURE_OUT", "capture_output"),
    ]:
        _val = os.environ.get(_env_key)
        if _val is not None:
            try:
                setattr(cfg, _cfg_attr, int(_val))
            except ValueError:
                setattr(cfg, _cfg_attr, _val)

    # P10.5: Apply --set overrides before anything else
    if args.set_overrides:
        _apply_set_overrides(cfg, args.set_overrides)
        # P10.6: Enforce φp + φa ≤ 1 after CLI overrides
        _enforce_phi_cli(cfg)

    # P10.5: --print-config dumps resolved YAML and exits
    if args.print_config:
        # Dump as nested YAML using the to_file() format
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            tmp = Path(f.name)
        try:
            cfg.to_file(tmp)
            import sys as _sys
            _sys.stdout.write(tmp.read_text())
        finally:
            tmp.unlink()
        return

    # ── Build simulation engine ──────────────────────────────────
    from .simulation.engine import SimulationEngine
    sim = SimulationEngine(cfg)

    # S5.6: Log lifecycle — engine created
    _log.info("Lifecycle | engine_created mode=%s N=%d", cfg.mode, cfg.num_boids)

    # ── Headless path ────────────────────────────────────────────
    if args.no_viz:
        t0 = time.time()
        if args.capture:
            # Apply CLI overrides before building recorder (CLI > env > YAML)
            if args.capture_output:
                cfg.capture_output = args.capture_output
            if args.capture_frames is not None:
                cfg.capture_frames = args.capture_frames

            from .capture.recorder import Recorder
            _log.info("Lifecycle | capture_started frames=%d", cfg.capture_frames)
            rec = Recorder(sim, cfg)
            sim.run_headless(steps=cfg.capture_frames, callback=rec.on_frame)
            rec.save_gif()
            rec.save_metrics_csv()
            rec.save_metrics_json()
            _log.info("Lifecycle | capture_complete")
        else:
            sim.run_headless()
        wall_s = time.time() - t0
        log_run_footer(sim.frame, wall_s)
        return

    # ── Visual path ──────────────────────────────────────────────
    import pygame

    from .viz.input_control import InputControl
    from .viz.visualizer import Visualizer

    pygame.init()

    # Request OpenGL 3.3 context — required everywhere for moderngl.
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    # macOS (especially Apple Silicon) requires a Core Profile with
    # forward compatibility; these hints are harmless on Linux but
    # absolutely necessary on Darwin — without them SDL2 cannot
    # allocate a GL 3.3 context and moderngl sees version 0.
    if sys.platform == "darwin":
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1
        )

    # P10.5: --fullscreen flag
    flags = pygame.DOUBLEBUF | pygame.OPENGL
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    _screen = pygame.display.set_mode(  # needed for OpenGL context
        (cfg.window_width, cfg.window_height),
        flags,
    )

    # P10.5: --light-scheme flag
    if args.light_scheme:
        cfg.theme = "light"

    t0 = time.time()
    _log.info("Lifecycle | viz_started")
    viz = Visualizer(sim, cfg)
    input_ctrl = InputControl(cfg, viz.camera)
    viz.run(input_ctrl)
    wall_s = time.time() - t0
    log_run_footer(sim.frame, wall_s)
    # pygame.quit() handled inside viz.run()


if __name__ == "__main__":
    main()
