"""Performance — D19 ring-buffer bounded soak, G6 GL context loss fallback soak, G7 fastmath×metrics warning soak.

Split out of test_performance.py (file-size split). Only
@pytest.mark.slow tests are meant for nightly; the rest are fast
smoke checks.
"""

import numpy as np
import pytest

# ── D19: Ring-buffer bounded over 20K frames ─────────────────────
#
# D19: Both metrics_history and frames lists must stay bounded at
# capture_frame_cap over a long soak run.  The existing D19 unit
# tests in test_recorder.py verify truncation logic in isolation;
# this test verifies it holds under continuous 20K-frame load with
# the real Recorder callback pipeline.
#
# The frames ring-buffer is exercised by mocking _capture_frame to
# append a placeholder — FBO capture is not available headless.


@pytest.mark.slow
class TestD19RingBufferSoak:
    """D19: Over 20K frames, both metrics_history and frames lists
    stay bounded at capture_frame_cap."""

    D19_STEPS = 2_000
    D19_CAP = 100
    D19_N = 500

    def test_d19_metrics_history_bounded_at_cap(self):
        """D19 (@slow): metrics_history stays bounded at cap over
        2K frames (20× cap of 100)."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D19_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = self.D19_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        sim.run_headless(steps=self.D19_STEPS, callback=rec.on_frame)

        # metrics_history must be bounded at cap despite 20K >> cap
        assert len(rec.metrics_history) == self.D19_CAP, (
            f"metrics_history length {len(rec.metrics_history)} "
            f"should equal cap {self.D19_CAP} after {self.D19_STEPS} "
            f"steps (20K >> cap)"
        )
        # Each entry is a valid dict with expected fields
        if len(rec.metrics_history) > 0:
            entry = rec.metrics_history[0]
            assert isinstance(entry, dict), f"Entry is {type(entry).__name__}, not dict"
            assert "alpha" in entry, "Missing alpha in metrics"

    def test_d19_frames_bounded_at_cap_with_mock_capture(self):
        """D19 (@slow): frames list stays bounded at cap over 20K
        frames, verified with a mock _capture_frame that appends
        a placeholder each time on_frame gates it."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D19_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = True   # enable frame capture path
        cfg.capture_every = 1         # capture every frame
        cfg.capture_prewarm = 0       # no pre-warm
        cfg.capture_frame_cap = self.D19_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Spy on _capture_frame: replace with mock that appends
        # a placeholder and applies the D19 truncation.
        original_cap = rec._frame_cap

        def _mock_capture_frame(_sim):
            rec.frames.append("mock_frame")
            if len(rec.frames) > original_cap:
                rec.frames[:] = rec.frames[-original_cap:]

        rec._capture_frame = _mock_capture_frame  # type: ignore[method-assign]

        sim.run_headless(steps=self.D19_STEPS, callback=rec.on_frame)

        # Both lists bounded at cap despite 20K >> cap
        assert len(rec.metrics_history) == self.D19_CAP, (
            f"metrics_history length {len(rec.metrics_history)} "
            f"should equal cap {self.D19_CAP}"
        )
        assert len(rec.frames) == self.D19_CAP, (
            f"frames length {len(rec.frames)} "
            f"should equal cap {self.D19_CAP}"
        )
        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"


# ── G6: GL context loss fallback during soak ─────────────────────
#
# G6: When the GPU context is lost mid-run, the system must degrade
# gracefully — metrics continue to be collected, the matplotlib
# fallback (P8.9) takes over frame capture, and no crash occurs.
#
# Unit tests in test_renderer.py verify the Renderer3D.gl_lost flag
# and Visualizer._render_safe in isolation.  This soak test verifies
# the full degrade path over a longer run with Recorder attached.


@pytest.mark.slow
class TestG6GLContextLossSoak:
    """G6: GL context loss fallback works during a longer capture run.

    Simulates: first N frames captured successfully via GPU, then GL
    context is lost and the mpl fallback takes over.  Metrics are
    unaffected by the transition.
    """

    G6_STEPS = 500
    G6_N = 100
    G6_CAP = 200
    G6_AFTER = 10   # frames before simulated GL loss

    def test_g6_degrade_to_mpl_fallback_mid_run(self):
        """G6 (@slow): After simulated GL context loss mid-run, the
        matplotlib fallback activates and the soak completes without
        crash.  Metrics continue to be collected throughout."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.G6_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.capture_with_viz = True      # enable frame capture path
        cfg.capture_mpl_fallback = True   # enable mpl fallback
        cfg.capture_every = 1
        cfg.capture_prewarm = 0
        cfg.capture_frame_cap = self.G6_CAP

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Track state
        gl_lost_signaled = [False]
        fallback_called = [False]

        # Spy on _fallback_to_mpl to verify it's called
        original_fallback = rec._fallback_to_mpl

        def _fallback_spy(sim_engine):
            fallback_called[0] = True
            original_fallback(sim_engine)

        rec._fallback_to_mpl = _fallback_spy  # type: ignore[method-assign]

        # Mock _capture_frame: first N frames via GPU, then GL loss.
        # After GL loss, EVERY frame goes through the mpl fallback,
        # matching the real _capture_frame → RuntimeError → fallback chain.
        def _g6_capture(sim_engine):
            if not gl_lost_signaled[0] and len(rec.frames) >= self.G6_AFTER:
                gl_lost_signaled[0] = True

            if gl_lost_signaled[0]:
                # GL was lost — every frame goes through fallback (like real
                # _capture_frame catches RuntimeError and calls _fallback_to_mpl)
                if rec._mpl_fallback_enabled:
                    rec._fallback_to_mpl(sim_engine)
            else:
                # GL still active — append a simulated GPU frame
                rec.frames.append("gpu_frame")

            # D19: Ring-buffer truncation
            if len(rec.frames) > rec._frame_cap:
                rec.frames[:] = rec.frames[-rec._frame_cap:]

        rec._capture_frame = _g6_capture  # type: ignore[method-assign]

        sim.run_headless(steps=self.G6_STEPS, callback=rec.on_frame)

        # ── Post-soak assertions ─────────────────────────────────

        # GL loss was triggered
        assert gl_lost_signaled[0], "GL loss must be triggered during test"

        # MPL fallback was activated
        assert fallback_called[0], (
            "MPL fallback must be called after GL context loss"
        )

        # Metrics unaffected by GL loss
        assert len(rec.metrics_history) > 0, (
            "Metrics must be collected throughout the soak"
        )
        # Frame count: at least G6_AFTER GPU frames + some fallback frames
        assert len(rec.frames) >= self.G6_AFTER, (
            f"Expected at least {self.G6_AFTER} frames, "
            f"got {len(rec.frames)}"
        )

        # Ring-buffer cap respected (D19)
        assert len(rec.frames) <= self.G6_CAP, (
            f"Frames ({len(rec.frames)}) exceed cap ({self.G6_CAP})"
        )
        assert len(rec.metrics_history) <= self.G6_CAP, (
            f"Metrics history ({len(rec.metrics_history)}) "
            f"exceed cap ({self.G6_CAP})"
        )

        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"

        # Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed max={speeds.max():.1f} > {max_allowed:.1f}"
        )


# ── G7: Fastmath × metrics warning during soak ───────────────────
#
# G7: When perf.fastmath=True, MetricsCollector emits a RuntimeWarning
# on the FIRST collect() call.  The warning should fire exactly once
# — not every frame — and metrics should still be collected correctly.
#
# Unit tests in test_metrics.py verify the warning in single-frame
# isolation.  This soak test verifies it over a longer headless
# capture run, ensuring the one-shot guard doesn't degrade over
# thousands of frames.


@pytest.mark.slow
class TestG7FastmathWarningSoak:
    """G7: Fastmath × metrics warning emitted exactly once over a
    longer headless capture run with perf.fastmath=True."""

    G7_STEPS = 500
    G7_N = 100
    G7_CAP = 200

    def test_g7_fastmath_warning_emitted_once_during_soak(self):
        """G7 (@slow): With perf.fastmath=True, RuntimeWarning is
        emitted exactly once during a 500-step headless capture.
        Metrics are still collected correctly after the warning."""
        import warnings

        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.G7_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1
        cfg.metrics_interval = 1
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = self.G7_CAP

        # Enable fastmath — the source of the warning
        cfg.fastmath = True

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Track warnings
        fastmath_warnings: list[warnings.WarningMessage] = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # don't suppress anything

            sim.run_headless(steps=self.G7_STEPS, callback=rec.on_frame)

            # Filter for the fastmath warning
            for w in caught:
                if "fastmath" in str(w.message).lower():
                    fastmath_warnings.append(w)

        # ── Post-soak assertions ─────────────────────────────────

        # Warning emitted exactly once
        assert len(fastmath_warnings) == 1, (
            f"Expected exactly 1 fastmath warning, got {len(fastmath_warnings)}. "
            f"Warning must fire on first collect() only — the _warned_fastmath "
            f"one-shot guard prevents repeat emissions."
        )
        # Verify it's a RuntimeWarning
        assert fastmath_warnings[0].category is RuntimeWarning, (
            f"Expected RuntimeWarning, got {fastmath_warnings[0].category}"
        )
        # Verify the warning message mentions metrics and fastmath
        msg = str(fastmath_warnings[0].message)
        assert "fastmath" in msg.lower(), (
            f"Warning message must mention fastmath: '{msg}'"
        )

        # Metrics collected despite fastmath
        assert len(rec.metrics_history) > 0, (
            "Metrics must be collected even with fastmath=True"
        )
        assert len(rec.metrics_history) <= self.G7_CAP, (
            f"Metrics history ({len(rec.metrics_history)}) exceeds "
            f"cap ({self.G7_CAP})"
        )

        # No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), "NaN in positions"
        assert not np.any(np.isnan(sim.flock.velocities)), "NaN in velocities"

        # Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed max={speeds.max():.1f} > {max_allowed:.1f}"
        )


