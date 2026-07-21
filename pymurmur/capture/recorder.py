"""Frame capture and metrics export for headless simulation runs.

Level 2 — optional recording layer. Called via SimulationEngine callback.

P8.7: Cinematic capture sweep — pre-warm, camera sweep (azim/elev/dist),
GIF optimize+disposal=2, env var overrides for width/height/frames/output.
P8.9: MPL fallback — when GPU capture fails, falls back to
MPLRecorder (matplotlib 3D scatter) instead of silently dropping frames.
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

        # P8.9: MPL fallback when GPU capture fails
        self._mpl_fallback: object | None = None  # MPLRecorder (lazy init)
        self._mpl_fallback_enabled: bool = capture_config.capture_mpl_fallback
        self._mpl_fallback_activated: bool = False

        # D16: Env vars are applied to config in __main__.py before
        # Recorder is constructed, so CLI > env > YAML.  Read config
        # directly here — no env var fallback that would invalidate CLI.
        self._capture_width: int = capture_config.capture_width
        self._capture_height: int = capture_config.capture_height
        self._capture_frames: int = capture_config.capture_frames
        self._capture_output: str = capture_config.capture_output
        self._prewarm: int = capture_config.capture_prewarm
        self._sweep: bool = capture_config.capture_sweep
        self._sweep_scale: float = capture_config.capture_scale
        # D19: Frame cap — prevent unbounded memory growth on long runs
        self._frame_cap: int = capture_config.capture_frame_cap

    def on_frame(self, sim: SimulationEngine) -> None:
        """Called every frame by run_headless() callback.

        Metrics are always captured every frame (I6.5).
        P8.7: Pre-warm skips only FBO capture for the first
        *capture_prewarm* frames.
        """
        self._frame_count += 1

        # Always capture metrics (I6.5: use to_dict for JSON-safe serialization)
        if sim.metrics:
            self.metrics_history.append(sim.metrics.snapshot().to_dict())
            # D19: Ring-buffer truncation — keep only the last _frame_cap entries
            if len(self.metrics_history) > self._frame_cap:
                self.metrics_history = self.metrics_history[-self._frame_cap:]

        # P8.7: Pre-warm — skip FBO capture for first N frames
        if self._frame_count <= self._prewarm:
            return

        # Capture FBO frame every capture_every frames (offset by pre-warm)
        effective_frame = self._frame_count - self._prewarm
        if self.with_viz and effective_frame % self.every == 0:
            self._capture_frame(sim)

    def _capture_frame(self, sim: SimulationEngine) -> None:
        """Compose Visualizer for headless frame capture (I6.1).

        P8.7: Applies cinematic sweep camera position when capture_sweep
        is enabled, based on the captured frame index.
        """
        try:
            from ..viz.visualizer import Visualizer
        except ImportError:
            return  # viz not available, skip frame capture

        try:
            if self._renderer is None:
                # P8.7: Use env-overridden capture dimensions
                self._renderer = Visualizer(
                    sim, self.config, headless=True,
                    width=self._capture_width,
                    height=self._capture_height,
                )

            # P8.7: Cinematic camera sweep (safe for mock renderers)
            if self._sweep and self._capture_frames > 0 and hasattr(self._renderer, "camera"):
                captured = len(self.frames)  # frames already captured
                t = captured / max(self._capture_frames, 1)
                t = min(t, 1.0)
                self._renderer.camera.cinematic_sweep(
                    t, scale=self._sweep_scale
                )

            img = self._renderer.headless_frame()  # type: ignore[attr-defined]
            if img is not None:
                self.frames.append(img)
                # D19: Ring-buffer truncation — keep only the last _frame_cap frames
                if len(self.frames) > self._frame_cap:
                    self.frames = self.frames[-self._frame_cap:]
        except RuntimeError:
            # P8.9: Fall back to matplotlib when GPU capture fails
            if self._mpl_fallback_enabled:
                self._fallback_to_mpl(sim)

    def _fallback_to_mpl(self, sim: SimulationEngine) -> None:
        """P8.9: Use MPLRecorder for GPU-free frame capture.

        Lazily creates the MPLRecorder on first GPU failure, then
        delegates frame capture to it.  Frames are merged into the
        main Recorder's frame list.
        """
        from .mpl_recorder import MPLRecorder

        if self._mpl_fallback is None:
            self._mpl_fallback = MPLRecorder(sim, self.config)
            # Inherit pre-warm state: reset mpl frame count to match
            # so capture_every gating aligns
            self._mpl_fallback._every = self.every
            self._mpl_fallback._prewarm = 0
            self._mpl_fallback._frame_count = self._frame_count
            self._mpl_fallback_activated = True

        mpl = self._mpl_fallback
        mpl.on_frame(sim)  # type: ignore[attr-defined]
        # Merge MPL frames into our list
        if hasattr(mpl, "frames") and mpl.frames:
            # Only take new frames we haven't seen yet
            existing = len(self.frames)
            new_frames = mpl.frames[existing:]
            self.frames.extend(new_frames)

    def save_gif(self, path: str | None = None, fps: int | None = None) -> str | None:
        """Assemble captured frames into an animated GIF.

        P8.7: Uses optimize=True and disposal=2 for smaller file size.
        Frames are downscaled with LANCZOS to keep file sizes reasonable.
        Returns the output path, or None if no frames were captured.

        C3: fps defaults to config.capture_fps when not explicitly passed.
        """
        if fps is None:
            fps = self.config.capture_fps
        if not self.frames:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        output = Path(path or self._capture_output)
        output.parent.mkdir(parents=True, exist_ok=True)

        # LANCZOS downscale: half-resolution for manageable GIF size
        scaled = [
            f.resize((f.width // 2, f.height // 2), Image.LANCZOS)  # type: ignore[attr-defined]
            if hasattr(f, "resize") else f
            for f in self.frames
        ]

        if len(scaled) == 1:
            scaled[0].save(str(output))  # type: ignore[union-attr]
        else:
            scaled[0].save(  # type: ignore[union-attr]
                str(output),
                save_all=True,
                append_images=scaled[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True,   # P8.7
                disposal=2,      # P8.7: clear each frame before next
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
