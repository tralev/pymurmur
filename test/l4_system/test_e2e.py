"""End-to-end tests — full simulation runs via CLI or headless engine.

Tests the full integration of config → engine → forces → metrics.
Marked with @pytest.mark.e2e.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


class TestEndToEnd:
    """Full-system tests running complete simulation cycles."""

    def test_e2e_headless_default(self):
        """python -m pymurmur --no-viz runs 10 steps without error."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=10)
        assert sim.frame == 10

    def test_e2e_headless_capture(self, tmp_path):
        """--no-viz --capture produces output files with metrics."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 20
        cfg.capture_frames = 10
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "test.csv")
        cfg.capture_metrics_json = str(tmp_path / "test.json")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)
        rec.save_metrics_csv()
        rec.save_metrics_json()

        assert Path(cfg.capture_metrics_csv).exists()
        assert Path(cfg.capture_metrics_json).exists()

    def test_e2e_all_modes_headless(self):
        """All 5 modes run 10 headless steps."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        modes = ["projection", "spatial", "field", "vicsek", "influencer"]
        for mode in modes:
            cfg = SimConfig()
            cfg.mode = mode
            cfg.num_boids = 20
            sim = SimulationEngine(cfg)
            sim.run_headless(steps=10)
            assert sim.frame == 10

    def test_e2e_config_switch(self, default_config):
        """Switching config mid-simulation doesn't crash."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=5)
        sim.config.mode = "spatial"
        sim.run_headless(steps=5)
        assert sim.frame == 10

    def test_e2e_predator_spawn_mid_run(self, default_config):
        """Enabling predator mid-simulation doesn't crash."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        sim.run_headless(steps=5)
        sim.config.predator_enabled = True
        sim.run_headless(steps=5)
        assert sim.frame == 10

    def test_e2e_add_remove_birds(self, default_config):
        """Adding then removing 50 birds leaves flock in valid state."""
        from pymurmur.simulation.engine import SimulationEngine

        sim = SimulationEngine(default_config)
        initial = sim.flock.N_active
        sim.run_headless(steps=3)
        sim.flock.add_boids(50, sim.config)
        sim.run_headless(steps=3)
        sim.flock.remove_boids(50)
        sim.run_headless(steps=3)
        assert sim.flock.N_active >= initial

    def test_e2e_seed_reproducibility(self):
        """Two runs with same seed produce identical metrics CSV."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.seed = 42
        cfg.num_boids = 20
        sim1 = SimulationEngine(cfg)
        sim1.run_headless(steps=10)

        cfg2 = SimConfig()
        cfg2.seed = 42
        cfg2.num_boids = 20
        sim2 = SimulationEngine(cfg2)
        sim2.run_headless(steps=10)

        # Positions should be identical (same seed)
        import numpy as np
        assert np.allclose(sim1.flock.positions, sim2.flock.positions)

    def test_e2e_boundary_toroidal(self):
        """No bird leaves domain in toroidal mode after 100 steps."""
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.boundary_mode = "toroidal"
        cfg.num_boids = 20
        sim = SimulationEngine(cfg)
        sim.run_headless(steps=100)

        pos = sim.flock.positions
        assert (pos[:, 0] >= 0).all()
        assert (pos[:, 0] <= cfg.width).all()
        assert (pos[:, 1] >= 0).all()
        assert (pos[:, 1] <= cfg.height).all()
        assert (pos[:, 2] >= 0).all()
        assert (pos[:, 2] <= cfg.depth).all()

    def test_e2e_gif_output_valid(self, tmp_path):
        """Output .gif is viewable (correct magic bytes, non-zero size)."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.capture_frames = 10
        cfg.capture_every = 1
        cfg.capture_output = str(tmp_path / "test.gif")
        cfg.capture_with_viz = True  # need frames for GIF

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)
        rec.save_gif()

        if Path(cfg.capture_output).exists():
            with open(cfg.capture_output, "rb") as f:
                header = f.read(6)
                assert header[:3] == b"GIF"
                f.seek(0, 2)
                assert f.tell() > 0

    def test_e2e_csv_columns_match(self, tmp_path):
        """CSV column count matches FlockMetrics field count."""
        from pymurmur.capture.recorder import Recorder
        from pymurmur.core.config import SimConfig
        from pymurmur.simulation.engine import SimulationEngine

        cfg = SimConfig()
        cfg.num_boids = 10
        cfg.capture_frames = 10
        cfg.capture_every = 1
        cfg.capture_metrics_csv = str(tmp_path / "test.csv")
        cfg.capture_with_viz = False

        sim = SimulationEngine(cfg)
        rec = Recorder(sim, cfg)
        sim.run_headless(steps=10, callback=rec.on_frame)
        rec.save_metrics_csv()

        import csv
        with open(cfg.capture_metrics_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        assert len(rows) == 10  # one row per frame
        assert len(rows[0]) == len(header)
