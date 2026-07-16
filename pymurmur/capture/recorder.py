"""Frame capture and metrics export for headless simulation runs.

Level 2 — optional recording layer. Called via SimulationEngine callback.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..simulation.engine import SimulationEngine


class Recorder:
    """Captures rendered frames and metric time-series during headless runs.

    Attach to SimulationEngine.run_headless() via callback:
        rec = Recorder(sim, config)
        sim.run_headless(callback=rec.on_frame)
        rec.save_gif()
        rec.save_metrics_csv()
    """

    def __init__(
        self,
        simulation: SimulationEngine,
        capture_config: SimConfig,
    ) -> None:
        self.sim = simulation
        self.config = capture_config
        self.with_viz = capture_config.capture_with_viz
        self.every: int = capture_config.capture_every
        self.frames: list[object] = []       # PIL Images
        self.metrics_history: list[dict] = []
        self._frame_count: int = 0
        self._renderer: object | None = None  # cached headless Visualizer

    def on_frame(self, sim: SimulationEngine) -> None:
        """Called every frame by run_headless() callback.

        Captures metric snapshots every frame (via to_dict) and
        FBO renders every capture_every frames when with_viz is enabled.
        """
        self._frame_count += 1

        # Always capture metrics (I6.5: use to_dict for JSON-safe serialization)
        if sim.metrics:
            self.metrics_history.append(sim.metrics.snapshot().to_dict())

        # Capture FBO frame every capture_every frames
        if self.with_viz and self._frame_count % self.every == 0:
            self._capture_frame(sim)

    def _capture_frame(self, sim: SimulationEngine) -> None:
        """Compose Visualizer for headless frame capture (I6.1)."""
        try:
            from ..viz.visualizer import Visualizer
        except ImportError:
            return  # viz not available, skip frame capture

        try:
            if self._renderer is None:
                # Use capture dimensions, not window dimensions (I6.2)
                self._renderer = Visualizer(
                    sim, self.config, headless=True,
                    width=self.config.capture_width,
                    height=self.config.capture_height,
                )

            img = self._renderer.headless_frame()
            if img is not None:
                self.frames.append(img)
        except RuntimeError:
            # FBO/GPU failure — skip this frame, don't crash the run
            pass

    def save_gif(self, path: str | None = None, fps: int = 20) -> str | None:
        """Assemble captured frames into an animated GIF.

        Frames are downscaled with LANCZOS to keep file sizes reasonable.
        Returns the output path, or None if no frames were captured.
        """
        if not self.frames:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        output = Path(path or self.config.capture_output)
        output.parent.mkdir(parents=True, exist_ok=True)

        # LANCZOS downscale: half-resolution for manageable GIF size
        scaled = [
            f.resize((f.width // 2, f.height // 2), Image.LANCZOS)
            if hasattr(f, "resize") else f
            for f in self.frames
        ]

        if len(scaled) == 1:
            scaled[0].save(str(output))
        else:
            scaled[0].save(
                str(output),
                save_all=True,
                append_images=scaled[1:],
                duration=int(1000 / fps),
                loop=0,
            )
        return str(output)

    def save_metrics_csv(self, path: str | None = None) -> str | None:
        """Export per-frame metrics as CSV.

        Returns the output path, or None if no metrics were captured.
        """
        if not self.metrics_history:
            return None

        output = Path(path or self.config.capture_metrics_csv)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(str(output), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        return str(output)

    def save_metrics_json(self, path: str | None = None) -> str | None:
        """Export metrics as JSON with metadata.

        Returns the output path, or None if no metrics were captured.
        """
        if not self.metrics_history:
            return None

        output = Path(path or self.config.capture_metrics_json)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Convert ndarray values to lists for JSON serialisation
        safe_history: list[dict] = []
        for entry in self.metrics_history:
            safe: dict = {}
            for k, v in entry.items():
                if hasattr(v, "tolist"):
                    safe[k] = v.tolist()
                elif hasattr(v, "item"):
                    safe[k] = v.item()  # numpy scalars
                else:
                    safe[k] = v
            safe_history.append(safe)

        data = {
            "metadata": {
                "seed": self.config.seed,
                "mode": self.config.mode,
                "num_boids": self.config.num_boids,
                "frame_count": len(safe_history),
            },
            "metrics": safe_history,
        }

        with open(str(output), "w") as f:
            json.dump(data, f, indent=2, default=str)
        return str(output)
