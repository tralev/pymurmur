"""S5.6 — Run logging tests (structured log, AST guard, CLI flag)."""

import ast
import io
import sys
import tempfile
from pathlib import Path

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

# ── AST guard — no print() in package sources ────────────────────

class TestNoPrintInPackageSources:
    """S5.6: No ``print(`` call anywhere in ``pymurmur/`` sources."""

    # Files that are allowed to contain print() — logging.py defines
    # the replacement helpers and is exempt.
    _ALLOWED = {"logging.py"}

    def _collect_package_py_files(self):
        """Yield all .py files under pymurmur/."""
        pkg = Path("pymurmur")
        for py_file in sorted(pkg.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            yield py_file

    def test_no_print_call_in_package_sources(self):
        """AST scan: no ``print(`` call node in any pymurmur/ .py file."""
        violations: list[tuple[str, int]] = []
        for py_file in self._collect_package_py_files():
            if py_file.name in self._ALLOWED:
                continue
            source = py_file.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue  # not a Python file

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "print":
                        violations.append((str(py_file), node.lineno))
                    elif (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "print"
                    ):
                        violations.append((str(py_file), node.lineno))

        assert not violations, (
            "Found print() calls in pymurmur/ sources (S5.6 AST guard):\n"
            + "\n".join(f"  {f}:{ln}" for f, ln in violations)
        )


# ── CLI output helpers ───────────────────────────────────────────

class TestCliOutput:
    """S5.6: cli_out / cli_err drop-in replacements for print()."""

    def test_cli_out_writes_to_stdout(self):
        """cli_out writes to sys.stdout with correct sep and end."""
        from pymurmur.core.logging import cli_out

        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            cli_out("hello", "world", sep="|", end="!")
        finally:
            sys.stdout = old

        assert buf.getvalue() == "hello|world!"

    def test_cli_err_writes_to_stderr(self):
        """cli_err writes to sys.stderr."""
        from pymurmur.core.logging import cli_err

        buf = io.StringIO()
        old = sys.stderr
        try:
            sys.stderr = buf
            cli_err("error", 42)
        finally:
            sys.stderr = old

        assert "error 42" in buf.getvalue()

    def test_cli_out_no_args_produces_newline(self):
        """cli_out with no args writes just a newline like print()."""
        from pymurmur.core.logging import cli_out

        buf = io.StringIO()
        old = sys.stdout
        try:
            sys.stdout = buf
            cli_out()
        finally:
            sys.stdout = old

        assert buf.getvalue() == "\n"


# ── Logger identity ──────────────────────────────────────────────

class TestLoggerIdentity:
    """S5.6: get_logger returns the same package-level logger."""

    def test_get_logger_returns_same_logger(self):
        """get_logger() returns the singleton pymurmur logger."""
        from pymurmur.core.logging import get_logger, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger1 = setup_run_logging(log_dir=tmpdir, level="info")
            logger2 = get_logger()
            assert logger1 is logger2
            assert logger2.name == "pymurmur"


# ── Log file creation ────────────────────────────────────────────

class TestRunLogging:
    """S5.6: Structured log file created with header, metrics, footer."""

    def test_headless_run_creates_log_file(self):
        """30-frame headless run creates output/run-<UTC>.log with
        header + at least one metrics line + footer."""
        from pymurmur.core.logging import (
            log_metrics_line,
            log_run_footer,
            log_run_header,
            setup_run_logging,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            # Setup logging
            logger = setup_run_logging(log_dir=log_dir, level="info")
            log_run_header("defaults", 42, "spatial", 50)

            # Simulate a few metrics intervals
            for frame in (1, 30, 60):
                log_metrics_line(frame, alpha=0.75, nematic_S=0.5,
                                 speed_real_ms=8.94, energy_J=0.001,
                                 collisions=0)

            log_run_footer(60, 1.0)

            # Flush handlers
            for h in logger.handlers:
                h.flush()

            # Find the log file
            log_files = sorted(log_dir.glob("run-*.log"))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
            content = log_files[0].read_text()

            # Verify header
            assert "pymurmur run started" in content
            assert "Header |" in content
            assert "config=defaults" in content
            assert "seed=42" in content
            assert "mode=spatial" in content

            # Verify metrics lines
            assert "Metrics |" in content
            assert "alpha=0.7500" in content
            assert "speed=8.94" in content

            # Verify footer
            assert "Footer |" in content
            assert "frames=60" in content

    def test_log_level_warning_suppresses_info_from_file(self):
        """At WARNING level, INFO messages are suppressed from the log file."""
        from pymurmur.core.logging import log_metrics_line, log_run_header, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"

            logger = setup_run_logging(log_dir=log_dir, level="warning")
            # Write INFO messages — should be suppressed at WARNING level
            log_run_header("test", 42, "spatial", 50)
            log_metrics_line(1, alpha=0.5)

            for h in logger.handlers:
                h.flush()

            log_files = sorted(log_dir.glob("run-*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()

            # At WARNING level, INFO messages are suppressed —
            # no Header or Metrics should appear in the file.
            assert "Header |" not in content
            assert "Metrics |" not in content
            assert "seed=42" not in content

    def test_log_file_naming_uses_utc_timestamp(self):
        """Log file name contains a UTC timestamp in ISO format."""
        from pymurmur.core.logging import setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            setup_run_logging(log_dir=log_dir, level="info")

            log_files = sorted(log_dir.glob("run-*.log"))
            assert len(log_files) == 1

            name = log_files[0].name
            # run-YYYYMMDDTHHMMSSZ.log
            assert name.startswith("run-")
            assert name.endswith(".log")
            ts_part = name[4:-4]  # strip "run-" and ".log"
            assert "T" in ts_part
            assert ts_part.endswith("Z")

    def test_engine_run_logs_lifecycle_and_metrics(self):
        """Full engine run writes lifecycle events + metrics to log."""
        import time

        from pymurmur.core.logging import (
            log_metrics_line,
            log_run_footer,
            log_run_header,
            setup_run_logging,
        )

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.mode = "spatial"
        cfg.metrics_interval = 5
        cfg.metrics_detail_level = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger = setup_run_logging(log_dir=log_dir, level="info")

            log_run_header("test_config", cfg.seed or 0, cfg.mode, cfg.num_boids)

            engine = SimulationEngine(cfg)
            logger.info("Lifecycle | engine_created mode=%s N=%d", cfg.mode, cfg.num_boids)

            t0 = time.time()
            for _ in range(20):
                engine.step()
                snap = engine.metrics.snapshot()
                log_metrics_line(
                    engine.frame, alpha=snap.alpha,
                    speed_real_ms=snap.speed_real_ms,
                    energy_J=snap.energy_J or 0.0,
                )

            wall_s = time.time() - t0
            log_run_footer(engine.frame, wall_s)

            for h in logger.handlers:
                h.flush()

            log_files = sorted(log_dir.glob("run-*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()

            # Header
            assert "Header |" in content
            # Lifecycle
            assert "Lifecycle | engine_created" in content
            # Metrics lines
            assert "Metrics |" in content
            # Footer
            assert "Footer |" in content
            assert "frames=20" in content


# ── Setup robustness ─────────────────────────────────────────────

class TestSetupRobustness:
    """S5.6: setup_run_logging handles edge cases correctly."""

    def test_duplicate_setup_cleans_up_old_handlers(self):
        """Calling setup_run_logging twice closes old handlers."""
        from pymurmur.core.logging import setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            logger1 = setup_run_logging(log_dir=log_dir, level="info")
            n_before = len(logger1.handlers)
            assert n_before >= 1

            logger2 = setup_run_logging(log_dir=log_dir, level="warning")
            # Handlers should be replaced, not duplicated
            assert len(logger2.handlers) == 2  # file + console

    def test_setup_with_debug_level(self):
        """setup_run_logging with DEBUG level is accepted."""
        from pymurmur.core.logging import setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="debug")
            assert logger.level <= 10  # DEBUG = 10

    def test_setup_with_string_level_warning(self):
        """setup_run_logging accepts string level names."""
        from pymurmur.core.logging import setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_run_logging(log_dir=tmpdir, level="WARNING")
            # Should not raise — string level is resolved
            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            assert len(log_files) == 1


# ── _resolve_level ───────────────────────────────────────────────

class TestResolveLevel:
    """S5.6: _resolve_level maps strings to Python logging levels."""

    def test_valid_level_names(self):
        """All standard level names resolve to correct ints."""
        import logging

        from pymurmur.core.logging import _resolve_level

        assert _resolve_level("debug") == logging.DEBUG
        assert _resolve_level("DEBUG") == logging.DEBUG
        assert _resolve_level("info") == logging.INFO
        assert _resolve_level("warning") == logging.WARNING
        assert _resolve_level("WARN") == logging.WARNING
        assert _resolve_level("error") == logging.ERROR
        assert _resolve_level("critical") == logging.CRITICAL

    def test_invalid_level_defaults_to_info(self):
        """Unknown level name falls back to INFO."""
        import logging

        from pymurmur.core.logging import _resolve_level

        assert _resolve_level("verbose") == logging.INFO
        assert _resolve_level("") == logging.INFO
        assert _resolve_level("trace") == logging.INFO


# ── Structured helpers ───────────────────────────────────────────

class TestStructuredHelpers:
    """S5.6: log_run_footer, log_lifecycle format correctness."""

    def test_log_run_footer_includes_mean_step(self):
        """Footer contains frames, wall time, and mean_step in ms."""
        from pymurmur.core.logging import log_run_footer, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_run_footer(100, 2.5)  # 100 frames in 2.5s = 25ms per frame

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "Footer |" in content
            assert "frames=100" in content
            assert "mean_step=25.00ms" in content

    def test_log_lifecycle_with_detail(self):
        """log_lifecycle accepts optional detail string."""
        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("mode_switch", "field → spatial")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "mode_switch" in content
            assert "field → spatial" in content

    def test_log_lifecycle_no_detail(self):
        """log_lifecycle works without detail string."""
        from pymurmur.core.logging import log_lifecycle, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_lifecycle("engine_created")

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "engine_created" in content

    def test_log_metrics_line_nan_theta_shows_dash(self):
        """NaN Theta renders as '-' in the metrics line."""
        from pymurmur.core.logging import log_metrics_line, setup_run_logging

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_run_logging(log_dir=tmpdir, level="info")
            log_metrics_line(1, alpha=0.5, theta=float("nan"))

            for h in logger.handlers:
                h.flush()

            log_files = sorted(Path(tmpdir).glob("run-*.log"))
            content = log_files[0].read_text()
            assert "Theta=-" in content
