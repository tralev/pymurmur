"""Performance diagnostics (Phase 10.3).

EMA frame timing (alpha=0.08), bottleneck classification,
and adaptive quality control for the render loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PerfSnapshot:
    """Current performance state — updated each frame."""

    # EMA-smoothed timings (ms)
    physics_ema: float = 0.0
    render_ema: float = 0.0
    total_ema: float = 0.0

    # Derived
    fps: float = 0.0
    bottleneck: str = "unknown"  # cpu | gpu | balanced
    cpu_fraction: float = 0.0

    # Adaptive quality flags
    reduce_resolution: bool = False
    reduce_count: bool = False

    # Counters
    frame_count: int = 0
    uptime_seconds: float = 0.0


class PerfDiagnostics:
    """Tracks and classifies frame-time performance.

    Uses EMA (alpha=0.08) to smooth timing measurements.
    Classifies bottleneck based on physics vs render time ratio.
    Provides adaptive quality recommendations when FPS drops.

    Usage:
        perf = PerfDiagnostics()
        for frame in loop:
            with perf.measure_physics():
                engine.step()
            with perf.measure_render():
                renderer.draw()
            perf.tick()
            snap = perf.snapshot()
            # snap.fps, snap.bottleneck, snap.reduce_resolution, etc.
    """

    ALPHA: float = 0.08
    TARGET_FPS: float = 60.0
    ADAPTIVE_THRESHOLD: float = 0.75  # trigger adaptive quality below 75% of target

    def __init__(self) -> None:
        self._physics_ema: float = 0.0
        self._render_ema: float = 0.0
        self._total_ema: float = 0.0
        self._frame: int = 0
        self._start_time: float = time.perf_counter()
        self._physics_start: float = 0.0
        self._render_start: float = 0.0
        self._last_physics_ms: float = 0.0
        self._last_render_ms: float = 0.0

    # ── Context managers for timing ────────────────────────────

    def measure_physics(self) -> _TimingBlock:
        """Context manager: measures physics step duration."""
        return _TimingBlock(self, "physics")

    def measure_render(self) -> _TimingBlock:
        """Context manager: measures render step duration."""
        return _TimingBlock(self, "render")

    def record_physics(self, ms: float) -> None:
        """Record physics step duration in milliseconds."""
        self._last_physics_ms = ms

    def record_render(self, ms: float) -> None:
        """Record render step duration in milliseconds."""
        self._last_render_ms = ms

    # ── Tick ────────────────────────────────────────────────────

    def tick(self) -> PerfSnapshot:
        """Advance one frame: update EMA, classify bottleneck.

        Call once per frame after physics + render timings are recorded.
        """
        self._frame += 1
        phys = self._last_physics_ms
        rend = self._last_render_ms
        total = phys + rend

        # EMA update
        if self._frame == 1:
            self._physics_ema = phys
            self._render_ema = rend
            self._total_ema = total
        else:
            self._physics_ema = self.ALPHA * phys + (1 - self.ALPHA) * self._physics_ema
            self._render_ema = self.ALPHA * rend + (1 - self.ALPHA) * self._render_ema
            self._total_ema = self.ALPHA * total + (1 - self.ALPHA) * self._total_ema

        return self._build_snapshot()

    def snapshot(self) -> PerfSnapshot:
        """Return current performance state without advancing."""
        return self._build_snapshot()

    def _build_snapshot(self) -> PerfSnapshot:
        """Shared derived metric computation."""
        total = max(self._total_ema, 0.01)
        fps = 1000.0 / total
        cpu_frac = self._physics_ema / total

        if cpu_frac > 0.6:
            bottleneck = "cpu"
        elif cpu_frac < 0.4:
            bottleneck = "gpu"
        else:
            bottleneck = "balanced"

        fp, pc = self.TARGET_FPS, self.ADAPTIVE_THRESHOLD
        adaptive = fps < fp * pc

        return PerfSnapshot(
            physics_ema=round(self._physics_ema, 3),
            render_ema=round(self._render_ema, 3),
            total_ema=round(self._total_ema, 3),
            fps=round(fps, 1),
            bottleneck=bottleneck,
            cpu_fraction=round(cpu_frac, 3),
            reduce_resolution=adaptive and bottleneck == "gpu",
            reduce_count=adaptive and bottleneck == "cpu",
            frame_count=self._frame,
            uptime_seconds=round(time.perf_counter() - self._start_time, 1),
        )

    def reset(self) -> None:
        """Reset all EMA state (e.g., after engine reset)."""
        self._physics_ema = 0.0
        self._render_ema = 0.0
        self._total_ema = 0.0
        self._frame = 0
        self._start_time = time.perf_counter()


class _TimingBlock:
    """Context manager for measuring a code block."""

    def __init__(self, parent: PerfDiagnostics, phase: str) -> None:
        self._parent = parent
        self._phase = phase
        self._start: float = 0.0

    def __enter__(self) -> _TimingBlock:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000.0  # to ms
        if self._phase == "physics":
            self._parent.record_physics(elapsed)
        elif self._phase == "render":
            self._parent.record_render(elapsed)
