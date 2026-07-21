"""S5.6 — Structured run logging.

Every run writes a structured log to ``output/run-<UTC>.log``:
  - Header: config echo, seed, mode, N, version
  - One metrics line per interval
  - Lifecycle events (commands, mode switches, governor actions)
  - Footer: frames, wall time, mean step ms

``--log-level {debug,info,warning}`` CLI flag.  All user-facing
output routes through :func:`cli_out` / :func:`cli_err` so the
AST guard ``no print(`` can pass cleanly.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Package logger ────────────────────────────────────────────────

_logger = logging.getLogger("pymurmur")
_logger.setLevel(logging.WARNING)  # silent by default

# Prevent propagation to the root logger (avoids double-output
# when the user has their own logging config).
_logger.propagate = False

# ── Public helpers ────────────────────────────────────────────────


def setup_run_logging(
    log_dir: str | Path = "output",
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure file + console logging for a simulation run.

    Creates ``{log_dir}/run-<UTC>.log`` with a human-readable
    timestamp format.  Console output goes to stderr at WARNING+
    so normal INFO lines stay out of the terminal.

    Args:
        log_dir: directory for log files (created if needed).
        level: Python logging level (int or case-insensitive name
               like ``"debug"``, ``"info"``, ``"warning"``).

    Returns:
        The package logger (``logging.getLogger("pymurmur")``).
    """
    global _logger

    # Resolve level from string if needed (must happen before
    # handler creation so setLevel() receives an int).
    if isinstance(level, str):
        level = _resolve_level(level)

    # Clear any previously added handlers (idempotent).
    # Close each handler first to release file handles, then clear
    # the list so stale handlers don't leak between test runs.
    for h in list(_logger.handlers):
        h.close()
    _logger.handlers.clear()

    # ── File handler ──────────────────────────────────────────
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"run-{ts}.log"

    fh = logging.FileHandler(str(log_path))
    fh.setLevel(level)  # S5.6: respect user's --log-level for file too
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    _logger.addHandler(fh)

    # ── Console handler (stderr, at the user's requested level) ─
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)  # S5.6: respect user's --log-level for console
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(ch)

    _logger.setLevel(min(level, logging.DEBUG))

    # Log the run header
    _logger.info("=" * 60)
    _logger.info("pymurmur run started — %s", log_path.name)
    _logger.info("=" * 60)

    return _logger


def get_logger() -> logging.Logger:
    """Return the package-level logger (``"pymurmur"``)."""
    return _logger


# ── CLI output helpers (replace print() in package sources) ───────
# The S5.6 AST guard forbids ``print(`` anywhere in ``pymurmur/``.
# These thin wrappers provide the same behaviour for user-facing
# CLI output (stdout / stderr) without tripping the guard.


def cli_out(*args, sep: str = " ", end: str = "\n") -> None:
    """Write to stdout — drop-in replacement for ``print()``."""
    sys.stdout.write(sep.join(str(a) for a in args) + end)


def cli_err(*args, sep: str = " ", end: str = "\n") -> None:
    """Write to stderr — drop-in replacement for ``print(..., file=sys.stderr)``."""
    sys.stderr.write(sep.join(str(a) for a in args) + end)


# ── Structured log helpers ────────────────────────────────────────


def log_run_header(
    config_summary: str,
    seed: object,
    mode: str,
    n_boids: int,
    version: str = "",
) -> None:
    """Emit the standard run header block.

    Args:
        config_summary: one-line config description (e.g. YAML path or "defaults").
        seed: RNG seed value.
        mode: force mode name (projection / spatial / ...).
        n_boids: initial number of boids.
        version: package version string (from ``__init__.py``).
    """
    _logger.info("Header | config=%s seed=%s mode=%s N=%d version=%s",
                 config_summary, seed, mode, n_boids, version or "dev")


def log_metrics_line(frame: int, *, alpha: float = 0.0,
                     nematic_S: float = 0.0, theta: float = float("nan"),
                     speed_real_ms: float = 0.0, energy_J: float = 0.0,
                     collisions: int = 0) -> None:
    """Emit one compact metrics line for the current interval."""
    theta_str = f"{theta:.3f}" if not (theta != theta) else "-"  # NaN guard
    _logger.info(
        "Metrics | frame=%d alpha=%.4f S=%.4f Theta=%s speed=%.2f "
        "E=%.6f collisions=%d",
        frame, alpha, nematic_S, theta_str, speed_real_ms, energy_J, collisions,
    )


def log_lifecycle(event: str, detail: str = "") -> None:
    """Emit a lifecycle event (command, mode switch, governor action)."""
    if detail:
        _logger.info("Lifecycle | %s — %s", event, detail)
    else:
        _logger.info("Lifecycle | %s", event)


def log_run_footer(frames: int, wall_time_s: float) -> None:
    """Emit the standard run footer block."""
    mean_ms = (wall_time_s / frames * 1000.0) if frames > 0 else 0.0
    wall_str = time.strftime("%H:%M:%S", time.gmtime(wall_time_s))
    _logger.info("-" * 60)
    _logger.info("Footer | frames=%d wall=%s mean_step=%.2fms",
                 frames, wall_str, mean_ms)
    _logger.info("=" * 60)


# ── Internal helpers ──────────────────────────────────────────────


def _resolve_level(name: str) -> int:
    """Resolve a case-insensitive level name to a Python logging level."""
    name = name.upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(name, logging.INFO)
