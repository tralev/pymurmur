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
from pathlib import Path

from .core.config import SimConfig


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
    package_dir = Path(__file__).parent
    shipped = package_dir / "conf" / f"{config_name}.yaml"
    if shipped.exists():
        return SimConfig.from_file(shipped)

    # Also try project-root conf/
    project_conf = Path("conf") / f"{config_name}.yaml"
    if project_conf.exists():
        return SimConfig.from_file(project_conf)

    # Try as absolute/relative path (use original name for paths)
    user_path = Path(name)
    if user_path.exists():
        return SimConfig.from_file(user_path)

    raise FileNotFoundError(
        f"Config '{name}' not found in conf/ or as path.\n"
        f"  Searched: {shipped}, {project_conf}, {user_path}"
    )


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
            print(f"  {f.stem:30s} — {first_line}")
            found = True

    if not found:
        print("  No config presets found in conf/")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.list_configs:
        list_available_configs()
        return

    cfg = load_config(args.config)

    # ── Build simulation engine ──────────────────────────────────
    from .simulation.engine import SimulationEngine
    sim = SimulationEngine(cfg)

    # ── Headless path ────────────────────────────────────────────
    if args.no_viz:
        if args.capture:
            # Apply CLI overrides before building recorder
            if args.capture_output:
                cfg.capture_output = args.capture_output
            capture_steps = args.capture_frames or cfg.capture_frames

            from .capture.recorder import Recorder
            rec = Recorder(sim, cfg)
            sim.run_headless(steps=capture_steps, callback=rec.on_frame)
            rec.save_gif()
            rec.save_metrics_csv()
            rec.save_metrics_json()
        else:
            sim.run_headless()
        return

    # ── Visual path ──────────────────────────────────────────────
    import pygame

    from .viz.visualizer import Visualizer
    from .viz.input_control import InputControl

    pygame.init()
    _screen = pygame.display.set_mode(  # needed for OpenGL context
        (cfg.window_width, cfg.window_height),
        pygame.DOUBLEBUF | pygame.OPENGL,
    )

    viz = Visualizer(sim, cfg)
    input_ctrl = InputControl(cfg, viz.camera)
    viz.run(input_ctrl)
    # pygame.quit() handled inside viz.run()


if __name__ == "__main__":
    main()
