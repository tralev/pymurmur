"""Time-series flock-dynamics curves — MSD, density autocorrelation
(tau_rho), convex-hull density, and the B9 accel/opacity
cross-correlation.

Extracted from metrics.py (file-size split).
"""
from __future__ import annotations

import numpy as np


def compute_tau_rho(density_history: list[np.ndarray]) -> float:
    """Density autocorrelation time τ_ρ via exponential decay fit.

    Computes Pearson r(τ) between density histograms at lag τ,
    fits r(τ) ≈ exp(−τ / τ_ρ) to extract the characteristic timescale.

    Args:
        density_history: list of flattened density histograms, one per snapshot.

    Returns:
        τ_ρ in frame units (≥ 1). Returns -1 if histograms are unchanging.
    """
    if len(density_history) < 4:
        return 0.0

    n = len(density_history)
    max_lag = min(n - 1, 6)

    lags: list[int] = []
    corrs: list[float] = []

    for lag in range(1, max_lag + 1):
        pairs = [(density_history[t], density_history[t + lag]) for t in range(n - lag)]
        if not pairs:
            continue
        # Compute Pearson r across all pairs for this lag
        r_vals: list[float] = []
        for h0, h1 in pairs:
            h0c = h0 - h0.mean()
            h1c = h1 - h1.mean()
            denom = np.sqrt((h0c ** 2).sum() * (h1c ** 2).sum())
            if denom < 1e-10:
                continue  # zero-variance pair: undefined correlation, skip
            else:
                r_vals.append(float((h0c * h1c).sum() / denom))
        lags.append(lag)
        corrs.append(float(np.mean(r_vals)))

    if not lags or all(c <= 0 for c in corrs):
        return 0.0

    # Fit exponential decay: log(r) = -τ / τ_ρ → τ_ρ = -τ / log(r)
    # Only use positive correlations for the fit
    valid = [(lag_val, _c) for lag_val, _c in zip(lags, corrs) if _c > 0.01]
    if len(valid) < 2:
        return 0.0

    # Weighted average of per-lag estimates
    # τ_ρ = -lag / log(r), with corr clamped to <1 to avoid log(1)=0
    tau_estimates = []
    for lag, corr in valid:
        corr_safe = min(corr, 0.999)  # r=1 → large τ, not inf
        tau_estimates.append(-lag / np.log(corr_safe + 1e-10))

    return float(np.median(tau_estimates))


def compute_msd(snapshots: list[np.ndarray]) -> float:
    """Mean squared displacement from position snapshots.

    MSD = mean displacement² over the longest available lag.
    """
    if len(snapshots) < 2:
        return 0.0

    # Compare first and last snapshot
    pos0 = snapshots[0]
    pos1 = snapshots[-1]
    if len(pos0) != len(pos1):
        return 0.0

    disp = pos1 - pos0
    return float(np.mean(np.sum(disp ** 2, axis=1)))


# ── P9.2: MSD(τ) curve ────────────────────────────────────────

def compute_msd_curve(
    snapshots: list[np.ndarray],
    domain_size: tuple[float, float, float] = (1000.0, 1000.0, 1000.0),
    max_lag: int = 64,
) -> tuple[list[float], list[int], float, int | None]:
    """Compute MSD(τ) curve with unwrapped positions and log-spaced lags.

    P9.2: Unwraps positions across toroidal boundaries via min_image,
    computes MSD over log-spaced lags {1, 2, 4, …, max_lag}, fits
    a log-log slope, and detects the ballistic→diffusive crossover.

    Args:
        snapshots: list of (N, 3) position snapshots at evenly-spaced frames.
        domain_size: (W, H, D) for min-image unwrapping.
        max_lag: maximum lag in snapshot units (must be < len(snapshots)).

    Returns:
        (msd_vals, lags, slope, crossover) where:
        - msd_vals: MSD value at each lag.
        - lags: the log-spaced lag list.
        - slope: log-log slope over the first 3 lags (ballistic regime).
        - crossover: first lag where the per-lag exponent drops below 1.5,
          or None if never crosses.
    """
    T = len(snapshots)
    if T < 3:
        return [0.0], [1], 0.0, None

    N = snapshots[0].shape[0]
    if N == 0:
        return [0.0], [1], 0.0, None

    W, H, D = domain_size
    box = np.array([W, H, D], dtype=np.float32)

    # Build unwrapped trajectory: p_unwrap[0] = p[0],
    #   p_unwrap[t] = p_unwrap[t−1] + min_image(p[t] − p[t−1])
    unwrapped = [snapshots[0].copy()]
    for t in range(1, T):
        delta = snapshots[t] - snapshots[t - 1]
        # min_image per-axis: Δx − W·round(Δx/W)
        delta_unwrapped = delta - box * np.round(delta / box)
        unwrapped.append(unwrapped[-1] + delta_unwrapped)
    traj = np.stack(unwrapped, axis=0)  # (T, N, 3)

    # Log-spaced lags: 1, 2, 4, 8, …, max_lag
    lags: list[int] = []
    lag = 1
    while lag <= max_lag and lag < T:
        lags.append(lag)
        lag *= 2
    if not lags:
        lags.append(1)

    msd_vals: list[float] = []
    for lag in lags:
        count = T - lag
        if count < 1:
            msd_vals.append(0.0)
            continue
        # MSD[lag] = (1/(T−lag))·Σ_t ‖p_unwrap(t+lag) − p_unwrap(t)‖²
        diffs = traj[lag:] - traj[:count]  # (count, N, 3)
        sq_disp = np.sum(diffs * diffs, axis=2)  # (count, N)
        msd_vals.append(float(np.mean(sq_disp)))

    # Log-log slope: linear fit to log(MSD) vs log(lag) over first 3 lags
    if len(lags) >= 2:
        n_fit = min(3, len(lags))
        log_lags = np.log(np.array(lags[:n_fit], dtype=np.float64))
        log_msd = np.log(np.maximum(np.array(msd_vals[:n_fit], dtype=np.float64), 1e-12))
        slope, _ = np.polyfit(log_lags, log_msd, 1)
    else:
        slope = 0.0

    # Crossover: first lag where per-lag exponent drops below 1.5
    # Per-lag exponent: d(log MSD) / d(log lag) between consecutive lags
    crossover: int | None = None
    for i in range(1, len(lags)):
        if lags[i - 1] == 0:
            continue
        local_slope = (np.log(max(msd_vals[i], 1e-12))
                       - np.log(max(msd_vals[i - 1], 1e-12))) / np.log(lags[i] / lags[i - 1])
        if local_slope < 1.5 and crossover is None:
            crossover = lags[i]

    return msd_vals, lags, float(slope), crossover


# ── P9.3: Hull-volume density + autocorrelation time ───────────

def compute_convex_hull_density(positions: np.ndarray) -> float:
    """P9.3: Compute flock density via convex hull volume.

    ρ = N / ConvexHull(positions).volume

    Returns 0.0 if the hull is degenerate (coplanar, colinear, or
    fewer than 4 non-coplanar points).

    Args:
        positions: (N, 3) float32 array of bird positions.

    Returns:
        ρ ≥ 0 — density in birds per unit volume.
    """
    N = len(positions)
    if N < 4:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(positions)
        vol = hull.volume
        if vol <= 0:
            return 0.0
        return N / vol
    except Exception:
        return 0.0


def compute_tau_rho_hull(
    density_ring: list[float],
    interval: int = 10,
    buffer_size: int = 500,
) -> float:
    """S3.5: Density autocorrelation time from hull-density ring buffer.

    τ = interval · (0.5 + Σ_{lag≥1} r(lag))
    Stops summation at the first lag where r(lag) ≤ 0, **or** at
    lag = 0.25·buffer_size, whichever comes first — the cap keeps τ
    finite on slowly-varying series that never cross zero (S3.5).

    Uses the ring buffer convention where index 0 is the oldest sample
    and index -1 is the newest (reverse of the original spec, but both
    work identically for autocorrelation).

    Args:
        density_ring: list of hull-density samples ρ(t).
        interval: frames between consecutive samples (default 10).
        buffer_size: capacity of the ring buffer the samples were drawn
            from (default 500, matching MetricsCollector's
            `_hull_density_maxlen`) — the 0.25·buffer_size stop cap is
            relative to this capacity, not the current sample count.

    Returns:
        τ_ρ in frame units. Returns 0 if insufficient data or
        no positive correlations.
    """
    n = len(density_ring)
    if n < 4:
        return 0.0

    series = np.array(density_ring, dtype=np.float64)
    mean = series.mean()
    var = np.var(series)
    if var < 1e-12:
        return 0.0  # constant series → τ = 0

    # Compute autocorrelation r(lag) for lags 1..max_lag
    max_lag = min(n - 1, max(1, int(0.25 * buffer_size)))
    tau_sum = 0.5  # from the formula: 0.5 + Σ r(lag)

    for lag in range(1, max_lag + 1):
        # r(lag) = ⟨(ρ_t - ρ̄)(ρ_{t+lag} - ρ̄)⟩ / var
        # Using the oldest-to-newest convention:
        # series[0] is oldest, series[-1] is newest
        # lag=1: compare adjacent pairs (0,1), (1,2), ..., (n-2,n-1)
        head = series[:n - lag]
        tail = series[lag:]
        r = float(np.mean((head - mean) * (tail - mean)) / var)

        if r <= 0:
            break
        tau_sum += r

    return float(tau_sum * interval)


def _theta_accel_correlation(
    accel_mag: np.ndarray,
    theta_ring: list[float],
    interval: int,
    buffer_size: int,
) -> tuple[list[float] | None, int | None]:
    """Shared cross-correlation core for both the horizontal and 3D
    theta-accel variants — everything after "compute an acceleration
    magnitude series" is identical between them."""
    theta_arr = np.array(theta_ring[1:], dtype=np.float64)  # (n-1,), aligned

    m = len(accel_mag)
    if m < 4:
        return None, None

    accel_mean, accel_std = accel_mag.mean(), accel_mag.std()
    theta_mean, theta_std = theta_arr.mean(), theta_arr.std()
    if accel_std < 1e-12 or theta_std < 1e-12:
        return None, None

    max_lag = min(m - 1, max(1, int(0.25 * buffer_size)))
    curve: list[float] = []
    for lag in range(0, max_lag + 1):
        a = accel_mag[: m - lag]
        t = theta_arr[lag:]
        c = float(np.mean((a - accel_mean) * (t - theta_mean)) / (accel_std * theta_std))
        curve.append(c)

    peak_lag = int(np.argmax(np.abs(curve))) * interval
    return curve, peak_lag


def compute_theta_horizontal_accel_correlation(
    com_vel_ring: list[np.ndarray],
    theta_ring: list[float],
    interval: int = 10,
    buffer_size: int = 500,
) -> tuple[list[float] | None, int | None]:
    """B9 (Pearce et al. 2014): cross-correlation between horizontal
    COM acceleration and internal opacity, C(δt) = corr(a_horiz_COM(t),
    Θ(t+δt)).

    Real starling murmurations show opacity changing significantly
    within seconds of rapid horizontal acceleration — suggesting
    opacity mediates long-range 3D information transfer, faster than
    nearest-neighbour propagation. This computes that correlation curve
    from ring-buffer samples of COM velocity and Θ, sampled at the same
    cadence (mirrors compute_tau_rho_hull's ring-buffer/lag-cap
    convention, but as a cross-correlation between two series instead
    of an autocorrelation of one).

    com_vel_ring holds full 3D COM velocity samples, but only the
    horizontal x,y components are used (z is "up" in this codebase) —
    this is a deliberate, paper-faithful choice: Pearce et al.'s own
    Fig. 3c specifically measures horizontal acceleration of real
    flocks (their footage only gave them horizontal motion to measure).
    See compute_theta_accel_correlation_3d for the full-3D-acceleration
    sibling, which uses the vertical component too — a genuinely
    different quantity, not a "more correct" version of this one; keep
    both, don't collapse into one.

    Acceleration is derived as the per-sample-step difference of
    horizontal velocity; this is proportional to true acceleration but
    not divided by interval·dt_phys, since Pearson correlation is
    invariant to a positive linear rescaling of one series — the
    lag-δt curve and its peak are identical either way, so the division
    is skipped.

    Args:
        com_vel_ring: list of (3,) COM velocity samples, oldest first.
        theta_ring: list of Θ samples, same length, same instants.
        interval: frames between consecutive samples (default 10).
        buffer_size: capacity of the ring buffer the samples were drawn
            from (default 500) — the 0.25·buffer_size max-lag cap is
            relative to this capacity, matching compute_tau_rho_hull.

    Returns:
        (curve, peak_lag) — C(δt) for δt = 0..max_lag, indexed by
        sample-step (curve[i] is the correlation at a lag of i sample
        steps), and peak_lag = the δt (converted to **frame** units via
        `interval`, matching compute_tau_rho_hull's frame-unit
        convention) at which |C(δt)| is largest.
        (None, None) if there are too few samples or either series is
        degenerate (zero variance — e.g. a perfectly steady flock).
    """
    n = len(theta_ring)
    if n < 6 or len(com_vel_ring) != n:
        return None, None

    vel_arr = np.array(com_vel_ring, dtype=np.float64)  # (n, 3)
    accel_xy = np.diff(vel_arr[:, :2], axis=0)  # (n-1, 2)
    accel_mag = np.linalg.norm(accel_xy, axis=1)  # (n-1,)
    return _theta_accel_correlation(accel_mag, theta_ring, interval, buffer_size)


def compute_theta_accel_correlation_3d(
    com_vel_ring: list[np.ndarray],
    theta_ring: list[float],
    interval: int = 10,
    buffer_size: int = 500,
) -> tuple[list[float] | None, int | None]:
    """Full-3D-acceleration sibling of compute_theta_horizontal_accel_correlation:
    cross-correlation between internal opacity and the full 3D COM
    acceleration magnitude ‖Δv‖ (not just the horizontal ‖Δv_xy‖).

    Pearce et al.'s own Fig. 3c is horizontal-only because their real
    flock footage couldn't measure vertical motion; this simulation has
    genuine 3D velocity data, so it's a legitimate question whether
    vertical maneuvering also correlates with opacity changes — a
    different quantity from the paper's own result, not a replacement
    for it. Same ring-buffer/lag-cap/degeneracy-guard mechanics as the
    horizontal version; see its docstring for those details.

    Args:
        com_vel_ring: list of (3,) COM velocity samples, oldest first.
        theta_ring: list of Θ samples, same length, same instants.
        interval: frames between consecutive samples (default 10).
        buffer_size: capacity of the ring buffer the samples were drawn
            from (default 500).

    Returns:
        (curve, peak_lag) — same convention as
        compute_theta_horizontal_accel_correlation. (None, None) if
        there are too few samples or either series is degenerate.
    """
    n = len(theta_ring)
    if n < 6 or len(com_vel_ring) != n:
        return None, None

    vel_arr = np.array(com_vel_ring, dtype=np.float64)  # (n, 3)
    accel_3d = np.diff(vel_arr, axis=0)  # (n-1, 3)
    accel_mag = np.linalg.norm(accel_3d, axis=1)  # (n-1,)
    return _theta_accel_correlation(accel_mag, theta_ring, interval, buffer_size)


