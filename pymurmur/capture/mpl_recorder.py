"""GPU-free matplotlib fallback recorder (P8.9).

Level 2 — optional capture layer.  Renders flock frames with
matplotlib 3D scatter plots when OpenGL/ModernGL is unavailable.

Dual-view: two subplots (15°/15° and 45°/45°) for depth perception.
Replaces the silent ``except RuntimeError: pass`` in the GPU recorder
with a working fallback path.

Warns on first activation so the user knows quality is reduced.
"""

from __future__ import annotations

import colorsys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.config import SimConfig
    from ..simulation.engine import SimulationEngine


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (h in [0,1]) to RGB, each channel in [0,1]."""
    return colorsys.hsv_to_rgb(h, s, v)


class MPLRecorder:
    """Capture flock frames as matplotlib 3D scatter → PIL Images.

    Pure-CPU fallback when ModernGL/GPU is unavailable.
    Config-driven: respects *capture_prewarm*, *capture_every*,
    *capture_mpl_dpi*, and capture dimensions.

    Usage::

        rec = MPLRecorder(sim, config)
        sim.run_headless(callback=rec.on_frame)
        rec.save_gif("output/murmuration.gif")
    """

    _WARNED: bool = False
    """Class-level flag to warn only once per process."""

    def __init__(
        self,
        simulation: SimulationEngine,
        config: SimConfig,
    ) -> None:
        self.sim = simulation
        self.config = config
        self._every: int = config.capture_every
        self._prewarm: int = config.capture_prewarm
        self._width: int = config.capture_width
        self._height: int = config.capture_height
        self._dpi: int = config.capture_mpl_dpi
        self._frame_count: int = 0
        self.frames: list[object] = []   # PIL Images
        self.metrics_history: list[dict] = []

        # Issue warning once per process
        if not MPLRecorder._WARNED:
            MPLRecorder._WARNED = True
            warnings.warn(
                "Matplotlib fallback active — rendering quality will be reduced. "
                "Install ModernGL for full GPU rendering.",
                stacklevel=2,
            )

    def on_frame(self, sim: SimulationEngine) -> None:
        """Called every frame by run_headless() callback.

        Always captures metrics.  Frames are captured every
        *capture_every* frames after the pre-warm period.
        """
        self._frame_count += 1

        # Always capture metrics
        if sim.metrics:
            self.metrics_history.append(sim.metrics.snapshot().to_dict())

        # Pre-warm: skip frame capture
        if self._frame_count <= self._prewarm:
            return

        effective = self._frame_count - self._prewarm
        if effective % self._every == 0:
            self._capture_frame(sim)

    def _capture_frame(self, sim: SimulationEngine) -> None:
        """Render a single frame as matplotlib 3D scatter → PIL Image."""
        try:
            img = self._render_frame(sim)
            if img is not None:
                self.frames.append(img)
        except Exception:
            # Don't crash the run on rendering errors
            pass

    def _render_frame(self, sim: SimulationEngine) -> object | None:
        """Draw dual-view 3D scatter and return PIL Image.

        Uses matplotlib Figure directly (not pyplot) for headless/CI safety.
        """
        import io

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from PIL import Image

        flock = sim.flock
        n = flock.N_active
        if n == 0:
            return None

        # Active bird data
        active = flock.active
        pos = flock.positions[active][:n]
        seeds = flock.seeds[active][:n]
        pred = flock.is_predator[active][:n]

        # Compute the bounding sphere for axis limits
        center = np.mean(pos, axis=0) if n > 0 else np.zeros(3)
        # Use np.maximum element-wise, then take scalar max
        half_range_arr = np.maximum(
            np.max(pos, axis=0) - center,
            center - np.min(pos, axis=0),
        )
        half_range = max(float(np.max(half_range_arr)), 100.0)
        pad = max(half_range * 0.15, 50.0)

        # Figure: 2× width for dual-view, single height
        fig_w = self._width * 2 / self._dpi
        fig_h = self._height / self._dpi
        fig = Figure(figsize=(fig_w, fig_h), dpi=self._dpi)

        # Split prey/predator
        prey_mask = ~pred[:n]
        pred_mask = pred[:n]

        # ── Left subplot: elev=15°, azim=15° ──
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.view_init(elev=15, azim=15)  # type: ignore[attr-defined]
        self._draw_scatter(ax1, pos, prey_mask, pred_mask, seeds, center, pad)

        # ── Right subplot: elev=45°, azim=45° ──
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.view_init(elev=45, azim=45)  # type: ignore[attr-defined]
        self._draw_scatter(ax2, pos, prey_mask, pred_mask, seeds, center, pad)

        fig.suptitle(
            f"frame {self._frame_count}  —  N={n}",
            fontsize=8,
        )

        # Render to PIL Image (Figure will be garbage-collected)
        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        buf.seek(0)

        return Image.open(buf)

    def _draw_scatter(
        self,
        ax,
        pos: np.ndarray,
        prey_mask: np.ndarray,
        pred_mask: np.ndarray,
        seeds: np.ndarray,
        center: np.ndarray,
        pad: float,
    ) -> None:
        """Draw prey + predator scatter points on a 3D axis."""
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        # Prey: coloured by seed hue
        n_prey = int(np.sum(prey_mask))
        if n_prey > 0:
            prey_colors = [_hsv_to_rgb(float(s), 0.65, 0.85) for s in seeds[prey_mask]]
            ax.scatter(
                x[prey_mask], y[prey_mask], z[prey_mask],
                c=prey_colors, s=3.0, alpha=0.8,
                marker=".",
            )

        # Predators: red, slightly larger
        n_pred = int(np.sum(pred_mask))
        if n_pred > 0:
            ax.scatter(
                x[pred_mask], y[pred_mask], z[pred_mask],
                c="#cc0000", s=12.0, alpha=0.9,
                marker="x",
            )

        # Axis limits centred on the flock
        ax.set_xlim(center[0] - pad, center[0] + pad)
        ax.set_ylim(center[1] - pad, center[1] + pad)
        ax.set_zlim(center[2] - pad, center[2] + pad)

        # Minimal ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.3)

    def save_gif(self, path: str | None = None, fps: int = 20) -> str | None:
        """Assemble captured frames into an animated GIF.

        Returns the output path, or None if no frames were captured.
        """
        if not self.frames:
            return None

        try:
            import PIL  # noqa: F401  # guard against missing dependency
        except ImportError:
            return None

        output = Path(path or self.config.capture_output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if len(self.frames) == 1:
            self.frames[0].save(str(output))  # type: ignore[attr-defined]
        else:
            self.frames[0].save(  # type: ignore[attr-defined]
                str(output),
                save_all=True,
                append_images=self.frames[1:],
                duration=int(1000 / fps),
                loop=0,
                optimize=True,
                disposal=2,
            )
        return str(output)

    def save_metrics_csv(self, path: str | None = None) -> str | None:
        """Export per-frame metrics as CSV."""
        import csv

        if not self.metrics_history:
            return None
        output = Path(path or self.config.capture_metrics_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        return str(output)
