"""Ecology extension — day/night cycle, logistic dusk roost, seasonal amplitude.

P4.8: Logistic dusk model (sigmoid ramp instead of linear), coherence gate
(smoothstep critical-mass threshold), seasonal amplitude (cosine year-cycle
modulation), and temperature-boosted roost pull.

O(1) per frame. Deterministic predator presence via Knuth hash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import Extension

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ..flock import PhysicsFlock
    from ._base import StepContext


class Ecology(Extension):
    """Day/night cycle with logistic dusk roosting behaviour.

    P4.8 additions:
      - logistic_dusk_factor: sigmoid ramp replacing hard linear cutoff
      - seasonal_factor: cosine year-cycle modulation (peak Jan, trough Jul)
      - coherence_gate: smoothstep flock-size threshold for roost pull
      - temperature_boost: warmer evenings → stronger roost pull
    """

    # ── Constants ─────────────────────────────────────────────
    _DAY_PEAK: float = 15.0     # day of year with peak murmuration (mid-Jan)
    _DAY_TROUGH: float = 197.0  # day of year with minimum activity (mid-Jul)
    _DUSK_CENTER: float = 20.0  # minutes before dusk at sigmoid midpoint

    def __init__(self, config: SimConfig) -> None:
        self._day: float = 172.0   # summer solstice
        self._day_dt: float = 1.0 / 600.0  # ~1 day per 10 seconds at 60fps
        self._roost_pos = np.array(config.ecology_roost, dtype=np.float32)
        self._critical_mass = config.ecology_critical_mass
        self._dusk_width = config.ecology_dusk_width
        self._seasonal_amplitude = config.ecology_seasonal_amplitude
        self._temperature_boost = config.ecology_temperature_boost
        self.predator_active: bool = True  # updated in apply(); public for I5.4
        self.coherence_factor: float = 1.0  # P4.8: exposed for force-weight gating
        self._last_int_day: int = int(self._day)

    # ── Static helpers (testable independently) ──────────────

    @staticmethod
    def day_length(day: float) -> float:
        """Hours of daylight for a given day of year."""
        return 12.0 + 4.5 * np.cos(2 * np.pi * (day - 172) / 365)

    @staticmethod
    def temperature(day: float) -> float:
        """Temperature in °C for a given day of year."""
        return 9.0 - 8.0 * np.cos(2 * np.pi * (day - 20) / 365)

    @staticmethod
    def predator_present(day: int) -> bool:
        """Deterministic per-day predator presence via Knuth multiplicative hash.

        S2.B8: reconciled to spec formula — `(day·2654435761 mod 1000)/1000
        < 0.296`. Returns True on ~29.6% of days, deterministically.
        """
        day_int = int(day)
        h = (day_int * 2654435761) % 1000
        return (h / 1000.0) < 0.296

    @staticmethod
    def dusk_hour(day: float) -> float:
        """Sunset hour (decimal) for a given day of year."""
        day_len = Ecology.day_length(day)
        return 12.0 + day_len / 2.0

    @staticmethod
    def logistic_dusk_factor(
        hour: float, dusk: float, dusk_width: float = 6.0
    ) -> float:
        """P4.8: Sigmoid dusk ramp — smooth transition into roost.

        S2.B8 assessment (2026-07-19): the roadmap spec sketches
        `1/(1+e^-z)` with `z = (hour-sunset)/(width/4)` (hour-unit z,
        midpoint exactly at sunset). That formula is under-specified for
        `dusk_width`, which is a documented, tested, YAML-exposed minutes
        parameter (`ecology_dusk_width`) — the spec never states whether
        `width` there is hours or minutes, and both readings change the
        transition steepness by 4x-240x. This form is blessed instead:
        same qualitative shape (0 well before dusk, 1 once roosting),
        keyed off the same `dusk_hour = 12 + day_length/2`, midpoint at a
        configurable offset before sunset rather than exactly at sunset
        (birds start settling before the sun is down, not after).

        Args:
            hour: current hour of day (0–24)
            dusk: sunset hour (from dusk_hour())
            dusk_width: logistic transition width in minutes

        Returns:
            Factor in [0, 1]: 0 = well before dusk, 1 = fully in roost window
        """
        minutes_before_dusk = (dusk - hour) * 60.0  # positive = before sunset
        # Guard against invalid width (zero or negative)
        if dusk_width <= 0:
            return 1.0 if minutes_before_dusk > Ecology._DUSK_CENTER else 0.0
        # Centred sigmoid: midpoint at _DUSK_CENTER minutes before dusk
        z = (minutes_before_dusk - Ecology._DUSK_CENTER) / dusk_width
        # Clamp to avoid overflow in exp
        z_clamped = max(min(z, 50.0), -50.0)
        # Sigmoid: z > 0 (well before dusk) → 0, z < 0 (past dusk) → 1
        return float(1.0 / (1.0 + np.exp(z_clamped)))

    @staticmethod
    def seasonal_factor(day: float, amplitude: float = 0.5) -> float:
        """P4.8: Cosine seasonal modulation of roost strength.

        Peak at day ~15 (mid-January murmuration season), trough at day ~197
        (mid-July, no murmurations).

        With default amplitude=0.5:
          - day 15 (peak):  1.0
          - day 197 (trough): 0.25

        Args:
            day: day of year (fractional)
            amplitude: modulation depth (0 = no seasonal effect, 0.5 = default)

        Returns:
            Factor with peak = 1.0, trough = 1.0 - 1.5*amplitude.
            Clamped to [0.05, 2.0] for safety.
        """
        phase = 2.0 * np.pi * (day - Ecology._DAY_PEAK) / 365.0
        raw = 1.0 - amplitude * 0.75 * (1.0 - np.cos(phase))
        return float(max(min(raw, 2.0), 0.05))

    @staticmethod
    def coherence_gate(n_active: int, critical_mass: int) -> float:
        """P4.8/S2.B8: Smoothstep gate on flock size.

        Returns 0 at or below 0.4×critical_mass, 1 at or above
        1.2×critical_mass, smoothstep-interpolated between — reconciled
        to the S2.B8 spec window (was smoothstep over [0, 1]·critical_mass).

        Args:
            n_active: number of active birds
            critical_mass: threshold for full roost behaviour

        Returns:
            Factor in [0, 1]
        """
        if n_active <= 0 or critical_mass <= 0:
            return 0.0
        lo = 0.4 * critical_mass
        hi = 1.2 * critical_mass
        t = min(max((float(n_active) - lo) / (hi - lo), 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)  # smoothstep

    @staticmethod
    def gated_weight(base_weight: float, n_active: int, critical_mass: int) -> float:
        """P4.8: Apply coherence gate to a flocking weight.

        Useful for modulating separation/alignment/cohesion weights based on
        whether the flock is large enough to exhibit murmuration behaviour.

        Args:
            base_weight: unmodified flocking weight
            n_active: number of active birds
            critical_mass: threshold for full behaviour

        Returns:
            base_weight * coherence_gate(n_active, critical_mass)
        """
        return base_weight * Ecology.coherence_gate(n_active, critical_mass)

    # ── Per-frame update ─────────────────────────────────────

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        """Advance day and apply logistic-dusk roost pull.

        P4.8: Combines logistic dusk ramp × coherence gate × seasonal amplitude
        × temperature boost for a physically plausible roosting pull.
        """
        # D3: Use actual dt from engine (P8.10 fixed-timestep).
        # Derive rate from the original _day_dt constant (1/600 per frame at
        # default fps=60).  day_per_second = _day_dt * fps = 1/600 * 60 = 0.1.
        self._day += ctx.dt * self._day_dt * 60.0

        # Check for predator presence on day boundary
        int_day = int(self._day)
        if int_day != self._last_int_day:
            self._last_int_day = int_day
            self.predator_active = self.predator_present(int_day)

        dusk = self.dusk_hour(self._day)
        hour = (self._day % 1) * 24.0

        # P4.8: Time-window guard — only activate roost pull within
        # ~2 hours before to 30 minutes after dusk.
        minutes_before_dusk = (dusk - hour) * 60.0
        if minutes_before_dusk < -30 or minutes_before_dusk > 120:
            self.coherence_factor = 1.0
            return

        # P4.8: Logistic dusk factor — smooth sigmoid transition
        dusk_factor = self.logistic_dusk_factor(hour, dusk, self._dusk_width)

        if dusk_factor <= 1e-6:
            self.coherence_factor = 1.0
            return  # No roost pull — well before dusk window

        # P4.8: Coherence gate — smoothstep flock-size threshold
        n_active = int(flock.active.sum())
        coherence = self.coherence_gate(n_active, self._critical_mass)
        self.coherence_factor = coherence

        # P4.8: Seasonal amplitude — modulate by time of year
        seasonal = self.seasonal_factor(self._day, self._seasonal_amplitude)

        # P4.8: Temperature boost — warmer weather increases roost pull
        temp = self.temperature(self._day)
        temp_factor = 1.0 + self._temperature_boost * max(temp / 25.0, 0.0)

        # Combined roost strength
        ramp = dusk_factor * coherence * seasonal * temp_factor

        if ramp <= 0.0:
            return

        active = flock.active
        for i in np.where(active)[0]:
            to_roost = self._roost_pos - flock.positions[i]
            flock.accelerations[i] += to_roost * 0.01 * ramp
