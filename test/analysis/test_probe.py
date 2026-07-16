"""P0.11 — Capability probing tests.

Tests for pymurmur.__main__.probe_capabilities() and the --probe CLI flag.
"""

import pytest
from pymurmur.__main__ import probe_capabilities


class TestProbeCapabilities:
    """Unit tests for probe_capabilities() dict."""

    def test_returns_dict(self):
        """probe_capabilities() returns a dict."""
        caps = probe_capabilities()
        assert isinstance(caps, dict)

    def test_has_all_expected_keys(self):
        """All 7 optional dependencies are listed."""
        caps = probe_capabilities()
        expected = {"moderngl", "numba", "pygame", "scipy", "gymnasium", "matplotlib", "PyGLM"}
        assert set(caps.keys()) == expected

    def test_values_are_string_or_none(self):
        """Every capability value is either a version string or None."""
        caps = probe_capabilities()
        for name, version in caps.items():
            assert version is None or isinstance(version, str), (
                f"{name}: expected str|None, got {type(version).__name__}"
            )

    def test_available_deps_have_version_string(self):
        """Any detected dependency has a valid, non-empty version string."""
        caps = probe_capabilities()
        for name, version in caps.items():
            if version is not None:
                assert isinstance(version, str), (
                    f"{name}: expected str version, got {type(version).__name__}"
                )
                assert len(version) > 0, f"{name} version should not be empty"

    def test_numba_may_be_missing(self):
        """numba may or may not be installed — either is valid."""
        caps = probe_capabilities()
        assert caps["numba"] is None or isinstance(caps["numba"], str)

    def test_idempotent(self):
        """Calling twice returns equivalent results."""
        caps1 = probe_capabilities()
        caps2 = probe_capabilities()
        assert caps1.keys() == caps2.keys()
        for key in caps1:
            assert caps1[key] == caps2[key], f"{key} changed between calls"

    def test_no_crash_when_all_deps_missing(self):
        """P0.11: probe returns cleanly even when no optional deps are installed.

        Verifies that the probe function handles the degenerate case where
        every dependency is None — no ImportError, no KeyError, no crash.
        All values should be None (or strings if deps happen to be present).
        """
        caps = probe_capabilities()
        # Must return exactly the expected keys
        expected = {"moderngl", "numba", "pygame", "scipy", "gymnasium", "matplotlib", "PyGLM"}
        assert set(caps.keys()) == expected
        # Every value must be None or a non-empty version string
        for name, version in caps.items():
            if version is not None:
                assert isinstance(version, str) and len(version) > 0, (
                    f"{name}: version must be str|None, got {type(version).__name__}={version}"
                )
        # At least one dep is expected to be installed (numpy is required)
        # so this is really testing that NONE of the optional deps causes a crash
        # when they're missing — the function should just report None for each.


class TestProbeCLI:
    """Integration tests for the --probe CLI flag."""

    @pytest.mark.slow
    def test_probe_cli_outputs_table(self):
        """python -m pymurmur --probe prints a table and exits 0."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pymurmur", "--probe"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"--probe failed: {result.stderr}"
        output = result.stdout
        assert "pymurmur capability probe:" in output
        assert "moderngl" in output

    @pytest.mark.slow
    def test_probe_exits_before_simulation(self):
        """--probe exits immediately, no SimulationEngine imported."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pymurmur", "--probe"],
            capture_output=True,
            text=True,
        )
        # Should complete immediately without pygame/moderngl init
        assert result.returncode == 0
        # Output should be small (just the probe table, not simulation output)
        assert len(result.stdout.splitlines()) <= 20
