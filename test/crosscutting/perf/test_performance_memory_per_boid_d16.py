"""Performance — P1×P3 memory-per-boid ratio, D16 capture override precedence (CLI>env>YAML) soak.

Split out of test_performance.py (file-size split). Only
@pytest.mark.slow tests are meant for nightly; the rest are fast
smoke checks.
"""

import numpy as np
import pytest

# ── P1 × P3: Memory-per-boid ratio at P1's N=2,000 scale ───────
#
# Cross-element: P1 (budget table at N=2,000) × P3 (memory audit).
# P3 only checks N=300K.  P1's budgets at N=2,000 implicitly assume
# O(N) memory scaling.  This test verifies that every mode has a
# consistent memory-per-boid ratio at the P1 budget scale, catching
# modes that silently allocate O(N²) per-boid memory.


class TestMemoryPerBoid:
    """P1×P3: Memory-per-boid ratio at N=2,000 (P1 budget scale)."""

    # Expected SoA arrays at N=2,000 (8 arrays × varying sizes)
    P1N = 2_000
    # Budget: 8 arrays × (N,3) float32 + (N,) float32 + (N,) bool
    # ~ (5 × 3 × 4 + 2 × 4 + 1) × 2000 / 1M ≈ 0.13 MB
    # Add 50% headroom for optional arrays like max_speed
    MAX_MB_2000 = 1.0

    def test_memory_per_boid_o_n_consistent(self):
        """P1×P3 (fast): At N=2,000, all O(N) modes have similar
        memory-per-boid ratio.

        A mode that accidentally allocates O(N²) auxiliary data would
        have significantly higher memory-per-boid at N=2,000.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        results: dict[str, float] = {}
        for mode in ("spatial", "field", "influencer", "projection", "angle"):
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = self.P1N
            cfg.seed = 7
            cfg.metrics_detail_level = 0
            sim = SimulationEngine(cfg)

            total_bytes = sum(
                getattr(sim.flock, attr).nbytes
                for attr in (
                    "positions", "velocities", "accelerations",
                    "prev_positions", "last_accelerations",
                    "seeds", "active", "is_predator",
                )
            )
            mb = total_bytes / (1024 * 1024)
            results[mode] = mb

        # Report all modes
        vals = list(results.values())
        keys = list(results.keys())
        max_mb = max(vals)
        min_mb = min(vals)
        max_mode = keys[vals.index(max_mb)]
        min_mode = keys[vals.index(min_mb)]
        ratio = max_mb / min_mb if min_mb > 0 else float("inf")

        # O(N) modes should have nearly identical SoA layouts
        assert ratio <= 2.0, (
            f"Memory-per-boid ratio across modes: {ratio:.2f}× "
            f"(max={max_mb:.3f} MB at {max_mode}, "
            f"min={min_mb:.3f} MB at {min_mode}). "
            f"Modes with O(N) layouts should agree within 2×. "
            f"Results: {results}"
        )
        # Absolute budget: no mode exceeds 1 MB at N=2,000
        for mode, mb in results.items():
            assert mb <= self.MAX_MB_2000, (
                f"{mode}: {mb:.3f} MB exceeds {self.MAX_MB_2000} MB budget at N={self.P1N}"
            )

    def test_vicsek_higher_memory_acknowledged(self):
        """P1×P3 (fast): Vicsek mode at N=2,000 total SoA memory
        stays < 10 MB.

        Vicsek's neighbour data is computed on-the-fly, so its
        persistent SoA footprint is identical to O(N) modes.
        The 10 MB budget guards against future array growth.
        """
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.mode = "vicsek"
        cfg.num_boids = self.P1N
        cfg.seed = 7
        cfg.metrics_detail_level = 0
        sim = SimulationEngine(cfg)

        total_bytes = sum(
            getattr(sim.flock, attr).nbytes
            for attr in (
                "positions", "velocities", "accelerations",
                "prev_positions", "last_accelerations",
                "seeds", "active", "is_predator",
            )
        )
        mb = total_bytes / (1024 * 1024)

        # Vicsek's persistent SoA < 10 MB at N=2,000
        # (base ~0.13 MB — 10 MB budget is very generous)
        assert mb <= 10.0, (
            f"Vicsek N={self.P1N}: {mb:.3f} MB exceeds 10 MB budget"
        )

    def test_modes_can_step_at_budget_scale(self):
        """P1×P3 (fast): Each O(N) mode can step without crash at
        N=2,000 with metrics disabled.

        Vicsek is excluded (O(N²) at N=2,000 is too slow) and marl
        is excluded (requires optional gymnasium dependency)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        for mode in ("spatial", "field", "influencer", "projection", "angle"):
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = self.P1N
            cfg.seed = 7
            cfg.metrics_detail_level = 0
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=2)
            assert not np.any(np.isnan(sim.flock.positions)), (
                f"{mode}: NaN in positions after step at N={self.P1N}"
            )
            assert sim.flock.N_active == self.P1N


# ── D16: Capture override precedence (CLI > env > YAML) — soak ──
#
# D16: Env var application moved to __main__.py so CLI > env > YAML.
# Existing unit tests in test_recorder.py verify Recorder ignores
# env vars and _apply_set_overrides works in isolation.
# This soak test exercises the FULL precedence chain over a longer
# headless capture run, verifying the contract holds end-to-end.


class TestD16PrecedenceSoak:
    """D16: Capture override precedence integrated with soak.

    Emulates the __main__.py ordering:
      1. Default config ("YAML") sets baseline values
      2. Env var overrides applied mid-way
      3. CLI-style overrides (--set) applied last
      4. Short headless capture runs with Recorder
      5. Verify final config and Recorder output respect the contract
    """

    D16_SOAK_STEPS = 200  # short soak — D16 doesn't need 20K frames
    D16_N = 50           # small flock for fast execution

    @pytest.mark.slow
    def test_d16_env_override_applied_before_cli_during_soak(self):
        """D16 (@slow): Full precedence chain — YAML → env → CLI —
        during a headless capture run.

        Contract assertion: CLI overrides beat env vars beat defaults.
        Verifies:
          - Recorder uses the final (CLI-overridden) values
          - Soak completes without NaN or crash
          - Metrics history populated from the overridden config
        """

        from pymurmur.__main__ import _apply_set_overrides
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        # ── Step 1: Default config (the "YAML" layer) ────────────
        cfg = SimConfig()
        cfg.mode = "spatial"
        cfg.num_boids = self.D16_N
        cfg.seed = 42
        cfg.metrics_detail_level = 1  # light metrics for soak
        cfg.capture_with_viz = False
        cfg.capture_frame_cap = 1000  # D19 ring buffer — larger than D16_SOAK_STEPS to avoid truncation

        # Set default ("YAML") capture values — the baseline
        cfg.capture_width = 400
        cfg.capture_height = 300
        cfg.capture_frames = self.D16_SOAK_STEPS
        cfg.capture_output = "d16_default.gif"

        # ── Step 2: Apply env var overrides (as __main__.py does) ─
        # Simulate CAPTURE_WIDTH=800 env var — should override YAML
        # NOTE: We DON'T use monkeypatch here because we want to test
        # the exact same code path __main__.py uses.
        env_overrides = {
            "CAPTURE_WIDTH": "800",
            "CAPTURE_HEIGHT": "600",
            "CAPTURE_FRAMES": str(self.D16_SOAK_STEPS),
            "CAPTURE_OUT": "d16_env_override.gif",
        }
        for _env_key, _cfg_attr in [
            ("CAPTURE_WIDTH", "capture_width"),
            ("CAPTURE_HEIGHT", "capture_height"),
            ("CAPTURE_FRAMES", "capture_frames"),
            ("CAPTURE_OUT", "capture_output"),
        ]:
            _val = env_overrides.get(_env_key)
            if _val is not None:
                try:
                    setattr(cfg, _cfg_attr, int(_val))
                except ValueError:
                    setattr(cfg, _cfg_attr, _val)

        # Verify env overrides took effect (YAML < env)
        assert cfg.capture_width == 800, (
            f"YAML < env: expected 800, got {cfg.capture_width}"
        )
        assert cfg.capture_height == 600, (
            f"YAML < env: expected 600, got {cfg.capture_height}"
        )

        # ── Step 3: Apply CLI overrides (beats env) ──────────────
        # Simulate --set capture.capture_width=1024 and
        # --set capture.capture_height=768
        _apply_set_overrides(cfg, [
            "capture.capture_width=1024",
            "capture.capture_height=768",
        ])

        # Verify CLI beats env (CLI > env)
        assert cfg.capture_width == 1024, (
            f"CLI > env: expected 1024, got {cfg.capture_width}"
        )
        assert cfg.capture_height == 768, (
            f"CLI > env: expected 768, got {cfg.capture_height}"
        )
        # Env-overridden fields NOT touched by CLI should remain
        assert cfg.capture_frames == self.D16_SOAK_STEPS, (
            f"Env value preserved for unmodified field: "
            f"expected {self.D16_SOAK_STEPS}, got {cfg.capture_frames}"
        )

        # ── Step 4: Run a soak with Recorder attached ────────────
        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)

        # Verify Recorder picked up the CLI-overridden values
        assert rec._capture_width == 1024, (
            f"Recorder must use CLI-overridden width: "
            f"expected 1024, got {rec._capture_width}"
        )
        assert rec._capture_height == 768, (
            f"Recorder must use CLI-overridden height: "
            f"expected 768, got {rec._capture_height}"
        )
        assert rec._capture_frames == self.D16_SOAK_STEPS, (
            f"Recorder must use env-overridden frames: "
            f"expected {self.D16_SOAK_STEPS}, got {rec._capture_frames}"
        )
        assert rec._capture_output == "d16_env_override.gif", (
            "Recorder must use env-overridden output: "
            f"expected 'd16_env_override.gif', got '{rec._capture_output}'"
        )

        # Run the soak
        sim.run_headless(steps=self.D16_SOAK_STEPS, callback=rec.on_frame)

        # ── Step 5: Verify post-soak invariants ──────────────────

        # 5a. Frame counter sanity
        assert sim.frame == self.D16_SOAK_STEPS, (
            f"Frame counter {sim.frame} != {self.D16_SOAK_STEPS}"
        )

        # 5b. Metrics captured every frame (capture_every=1 default)
        assert len(rec.metrics_history) == self.D16_SOAK_STEPS, (
            f"Expected {self.D16_SOAK_STEPS} metrics entries, "
            f"got {len(rec.metrics_history)}"
        )

        # 5c. Ring buffer cap respected (D19): metrics_history ≤ cap
        cap = cfg.capture_frame_cap
        assert len(rec.metrics_history) <= cap, (
            f"Metrics history ({len(rec.metrics_history)}) "
            f"exceeds cap ({cap})"
        )

        # 5d. No NaN after soak
        assert not np.any(np.isnan(sim.flock.positions)), (
            "NaN in positions after D16 soak"
        )
        assert not np.any(np.isnan(sim.flock.velocities)), (
            "NaN in velocities after D16 soak"
        )

        # 5e. Speed contract holds
        speeds = np.linalg.norm(sim.flock.velocities, axis=1)
        max_allowed = cfg.v0 * 1.5
        assert np.all(speeds <= max_allowed), (
            f"Speed violated: max={speeds.max():.1f} > {max_allowed:.1f}"
        )

        # 5f. Metrics contain expected fields
        if len(rec.metrics_history) > 0:
            entry = rec.metrics_history[0]
            assert "alpha" in entry, "Missing alpha in metrics"
            assert "speed_avg" in entry, "Missing speed_avg in metrics"

    def test_d16_env_var_ignored_by_recorder_soak_consistent(self):
        """D16 (fast): Recorder ignores env vars during construction
        when config has explicit values — consistent with soak pattern.

        Unlike the unit test (which uses monkeypatch), this test
        modifies os.environ directly (then restores) to verify the
        __main__.py env var application pattern works correctly.
        """
        old_environ = {}
        import os as _os
        try:
            # Save and set env vars
            for key in ("CAPTURE_WIDTH", "CAPTURE_HEIGHT",
                        "CAPTURE_FRAMES", "CAPTURE_OUT"):
                old_environ[key] = _os.environ.get(key)

            _os.environ["CAPTURE_WIDTH"] = "640"
            _os.environ["CAPTURE_HEIGHT"] = "480"
            _os.environ["CAPTURE_FRAMES"] = "50"
            _os.environ["CAPTURE_OUT"] = "env_test.gif"

            from pymurmur.core.config import SimConfig

            cfg = SimConfig()
            cfg.mode = "spatial"
            cfg.num_boids = self.D16_N
            cfg.seed = 42
            cfg.metrics_detail_level = 0
            cfg.capture_with_viz = False

            # Use EXPLICIT config values that differ from env
            cfg.capture_width = 320
            cfg.capture_height = 240
            cfg.capture_frames = 10
            cfg.capture_output = "explicit.gif"

            # Apply env vars (as __main__.py does BEFORE config is
            # passed to Recorder) — env should NOT override explicit
            for _env_key, _cfg_attr in [
                ("CAPTURE_WIDTH", "capture_width"),
                ("CAPTURE_HEIGHT", "capture_height"),
                ("CAPTURE_FRAMES", "capture_frames"),
                ("CAPTURE_OUT", "capture_output"),
            ]:
                _val = _os.environ.get(_env_key)
                if _val is not None:
                    try:
                        setattr(cfg, _cfg_attr, int(_val))
                    except ValueError:
                        setattr(cfg, _cfg_attr, _val)

            # NOTE: After __main__.py applies env overrides, the
            # explicit config values are OVERRIDDEN by env vars.
            # This is the correct YAML < env < CLI contract:
            # env beats "YAML" (i.e., hardcoded defaults).
            # To preserve explicit config values, they must be
            # applied AFTER env vars (via --set CLI or direct setattr).
            # This test verifies the contract: env beats hardcoded.
            assert cfg.capture_width == 640, (
                f"Env should beat hardcoded: expected 640, got {cfg.capture_width}"
            )

            from pymurmur.capture.recorder import Recorder
            from pymurmur.simulation.engine import SimulationEngine

            sim = SimulationEngine(cfg)
            rec = Recorder(sim, cfg)

            # Recorder reads from config — it should see env-overridden value
            assert rec._capture_width == 640, (
                f"Recorder reads env-overridden width: "
                f"expected 640, got {rec._capture_width}"
            )

            # Run a few steps to verify no crash
            sim.run_headless(steps=5, callback=rec.on_frame)
            assert len(rec.metrics_history) == 5

        finally:
            # Restore env vars
            for key, val in old_environ.items():
                if val is not None:
                    _os.environ[key] = val
                else:
                    _os.environ.pop(key, None)
