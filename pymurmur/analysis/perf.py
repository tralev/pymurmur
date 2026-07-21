"""Performance diagnostics (Phase 10.3).

EMA frame timing (alpha=0.08), bottleneck classification,
and adaptive quality control for the render loop.

P8.6: QualityGovernor — EMA-based frame budget tracking with
degradation ladder (trails→scale→count) and hysteresis guard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PerfSnapshot:
    """Current performance state — updated each frame."""

    # EMA-smoothed timings (ms)
    physics_ema: float = 0.0
    render_ema: float = 0.0
    total_ema: float = 0.0

    # Derived
    fps: float = 0.0
    bottleneck: str = "unknown"  # legacy: cpu | gpu | balanced
    cpu_fraction: float = 0.0

    # S4.10: 4-category risk classifier
    risk_class: str = "unknown"   # cpu | vertex | fragment | mixed
    n_active: int = 0              # active bird count at snapshot time

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

    # S4.10: Thresholds for vertex vs fragment classification.
    # When render is the bottleneck (cpu_frac <= 0.4), N_active above
    # this value suggests vertex-bound (many instances to transform),
    # below suggests fragment-bound (fill-rate from resolution).
    # 10K matches the mesh registry's instanced→impostor transition
    # (recommend_render_mode): below 10K the GPU can handle full
    # instanced geometry, so render time is likely fill-rate.
    VERTEX_N_THRESHOLD: int = 10_000

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
        self._n_active: int = 0  # S4.10: last known bird count

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

    # ── S4.10: Risk classifier input ───────────────────────────

    def set_active_count(self, n: int) -> None:
        """S4.10: Record active bird count for vertex/fragment classification."""
        self._n_active = max(n, 0)

    def _classify_risk(self, cpu_frac: float) -> str:
        """S4.10: 4-category risk classifier.

        Logic:
            cpu_frac > 0.6  → "cpu"      (physics dominates)
            cpu_frac ≤ 0.4, N ≥ 10K → "vertex"  (many instances to transform)
            cpu_frac ≤ 0.4, N < 10K → "fragment" (fill-rate from resolution/blend)
            otherwise       → "mixed"    (balanced, or borderline)
        """
        if cpu_frac > 0.6:
            return "cpu"
        if cpu_frac <= 0.4:
            return "vertex" if self._n_active >= self.VERTEX_N_THRESHOLD else "fragment"
        return "mixed"

    def _build_snapshot(self) -> PerfSnapshot:
        """Shared derived metric computation."""
        total = max(self._total_ema, 0.01)
        fps = 1000.0 / total
        cpu_frac = self._physics_ema / total

        # Legacy 3-way bottleneck (backward compatible).
        # NOTE: this field and risk_class are independent classification
        # layers.  bottleneck always returns cpu/gpu/balanced based on
        # the simple cpu_frac ratio; risk_class uses N_active to split
        # the GPU path into vertex/fragment.  The reduce_* flags below
        # follow risk_class (S4.10), not bottleneck.
        if cpu_frac > 0.6:
            bottleneck = "cpu"
        elif cpu_frac < 0.4:
            bottleneck = "gpu"
        else:
            bottleneck = "balanced"

        # S4.10: 4-category risk classifier
        risk_class = self._classify_risk(cpu_frac)

        fp, pc = self.TARGET_FPS, self.ADAPTIVE_THRESHOLD
        adaptive = fps < fp * pc

        # S4.10: reduce_count for cpu/vertex (instance-heavy),
        # reduce_resolution for fragment/mixed (fill-bound).
        count_classes = {"cpu", "vertex"}
        res_classes = {"fragment", "mixed"}

        return PerfSnapshot(
            physics_ema=round(self._physics_ema, 3),
            render_ema=round(self._render_ema, 3),
            total_ema=round(self._total_ema, 3),
            fps=round(fps, 1),
            bottleneck=bottleneck,
            cpu_fraction=round(cpu_frac, 3),
            risk_class=risk_class,
            n_active=self._n_active,
            reduce_resolution=adaptive and risk_class in res_classes,
            reduce_count=adaptive and risk_class in count_classes,
            frame_count=self._frame,
            uptime_seconds=round(time.perf_counter() - self._start_time, 1),
        )

    def reset(self) -> None:
        """Reset all EMA state (e.g., after engine reset)."""
        self._physics_ema = 0.0
        self._render_ema = 0.0
        self._total_ema = 0.0
        self._frame = 0
        self._n_active = 0
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



class QualityGovernor:
    """P8.6: Adaptive quality governor with hysteresis.

    Tracks EMA frame time against a configurable budget and triggers
    degradation actions when the system falls below 78% of the target
    frame rate for a sustained period (1.8 s).  Recovery is gated by
    a separate 3.6 s window to avoid oscillation.

    Ladder (one step per 1.8 s):
        0. no-op (healthy)
        1. trails off
        2. render scale −0.15 (floor 0.75)
        3. N −18% (floor 512)

    Parameters
    ----------
    target_fps: Target frame rate (default 60).

    Usage::

        gov = QualityGovernor(target_fps=60)
        for frame in loop:
            gov.feed(frame_ms)
            if gov.should_degrade():
                action = gov.degradation_level  # 1, 2, or 3
                # ... apply action ...
    """

    # Tuning constants (P8.6 spec)
    ALPHA: float = 0.08          # EMA smoothing factor
    SPIKE_CAP_MS: float = 250.0   # clamp outliers
    HEALTHY_MARGIN: float = 1.12  # avg ≤ margin·budget → healthy
    DEGRADE_RATIO: float = 0.78   # fps < ratio·target → start degrade timer
    RECOVERY_RATIO: float = 0.85  # avg ≤ ratio·budget → start recovery timer
    DEGRADE_WINDOW: float = 1.8   # seconds below threshold before degrade
    RECOVERY_WINDOW: float = 3.6  # seconds healthy before recovery step
    RENDER_SCALE_STEP: float = 0.15  # −0.15 per step
    RENDER_SCALE_FLOOR: float = 0.75
    COUNT_STEP_FRACTION: float = 0.18  # −18% per step
    COUNT_FLOOR: int = 512

    def __init__(self, target_fps: int = 60) -> None:
        self._target_fps: float = float(max(target_fps, 24))
        self._budget_ms: float = 1000.0 / self._target_fps

        self._ema: float = 0.0          # EMA of frame time (ms)
        self._initialised: bool = False

        # Degradation state
        self._degradation_level: int = 0  # 0=healthy, 1=trails off, 2=scale, 3=count
        self._degrade_timer: float = 0.0   # seconds below threshold
        self._recovery_timer: float = 0.0  # seconds above recovery threshold
        self._last_action_clock: float = 0.0  # seconds since last degrade step

    # ── Public API ──────────────────────────────────────────────

    def feed(self, frame_ms: float) -> None:
        """P8.6: Update EMA with latest frame time (spike-capped at 250 ms)."""
        capped = min(frame_ms, self.SPIKE_CAP_MS)
        if not self._initialised:
            self._ema = capped
            self._initialised = True
        else:
            self._ema = self.ALPHA * capped + (1.0 - self.ALPHA) * self._ema

        # Advance timers
        dt = frame_ms / 1000.0  # convert to seconds
        self._last_action_clock += dt

        fps = 1000.0 / max(self._ema, 0.01)

        if fps < self.DEGRADE_RATIO * self._target_fps:
            self._degrade_timer += dt
            self._recovery_timer = 0.0
        elif self._ema <= self.RECOVERY_RATIO * self._budget_ms:
            self._recovery_timer += dt
            self._degrade_timer = 0.0
        else:
            # In between — reset both timers to avoid drift
            self._degrade_timer = max(self._degrade_timer - dt, 0.0)
            self._recovery_timer = max(self._recovery_timer - dt, 0.0)

    def should_degrade(self) -> bool:
        """P8.6: True when a degradation action should fire this frame."""
        if self._degradation_level >= 3:
            return False  # already at bottom of ladder
        if self._degrade_timer < self.DEGRADE_WINDOW:
            return False
        if self._last_action_clock < self.DEGRADE_WINDOW:
            return False  # one step per 1.8 s
        # Fire degradation
        self._degradation_level += 1
        self._degrade_timer = 0.0
        self._last_action_clock = 0.0
        return True

    def should_recover(self) -> bool:
        """P8.6: True when a recovery action should fire this frame."""
        if self._degradation_level <= 0:
            return False  # already healthy
        if self._recovery_timer < self.RECOVERY_WINDOW:
            return False
        # Fire recovery — move up one step
        self._degradation_level -= 1
        self._recovery_timer = 0.0
        self._last_action_clock = 0.0
        return True

    # ── Accessors ───────────────────────────────────────────────

    @property
    def degradation_level(self) -> int:
        """P8.6: Current degradation step (0=healthy, 3=max degraded)."""
        return self._degradation_level

    @property
    def ema_ms(self) -> float:
        """P8.6: Current EMA of frame time in milliseconds."""
        return self._ema

    @property
    def budget_ms(self) -> float:
        """P8.6: Target budget per frame in milliseconds."""
        return self._budget_ms

    @property
    def is_healthy(self) -> bool:
        """P8.6: True if EMA ≤ 1.12·budget."""
        return self._ema <= self.HEALTHY_MARGIN * self._budget_ms

    def reset(self) -> None:
        """P8.6: Reset all state (e.g. after engine reset)."""
        self._ema = 0.0
        self._initialised = False
        self._degradation_level = 0
        self._degrade_timer = 0.0
        self._recovery_timer = 0.0
        self._last_action_clock = 0.0
