"""Reynolds 1987 spatial mode — separation + alignment + cohesion + noise.

Two-pass architecture: Python cKDTree query → numpy force pass.

P2.2: Wrapped in SpatialMode(ForceMode) with @register("spatial").
P4.1: Hybrid metric+topological filter — neighbours capped at influence_count
      within visual_range.
P4.2: Standardised force pipeline — accumulate → accel_scale → clamp → noise.
P4.3: Predator boids — prey near predators get escape force, zeroed
      alignment/cohesion, and speed/perception/accel boosts.
P4.5: Per-frame parameter jitter — seeded random variation on
      separation/cohesion/alignment weights each frame.
P4.10: Numba-accelerated kernels for hot-loop operations — hybrid filter,
       predator detection, and predator escape.
Modularity pass 9: sep/align/coh/flow/predator_escape/forward_thrust are
       registered as named ForceTerm entries in SPATIAL_TERMS and composed
       via composeForces() (physics/forces/_base.py), the same S2.A5
       contract field.py's terms use — accel_scale/clamp/noise stay
       outside the composed terms, matching field.py's own precedent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._base import (
    ForceTerm,
    alignment_force,
    cohesion_force,
    composeForces,
    curl_flow,
    separation_force,
)
from ..plugins.force_mode import ForceFn, ForceMode, register
from ..plugins.neighbor_selection import NEIGHBOR_SELECTOR_REGISTRY
from ..plugins.noise_strategy import NOISE_STRATEGY_REGISTRY
from .spatial_helpers import (
    _apply_hybrid_filter,  # noqa: F401  # re-export
    _maybe_perception_filter,
    _predator_escape,
    _query_neighbors,
)

# P4.10: Try to import numba kernels; fall back to pure numpy.
# Always import both numba and numpy versions so runtime dispatch
# based on config.perf.use_numba can switch without re-import.
try:
    from ._kernels import _HAS_NUMBA as _KERNELS_HAS_NUMBA  # noqa: F401
    from ._kernels import (
        _numpy_hybrid_filter,
        _numpy_predator_detect,
        _numpy_predator_escape,
    )
    if _KERNELS_HAS_NUMBA:
        from ._kernels import (
            _numba_hybrid_filter,
            _numba_predator_detect,
            _numba_predator_escape,
        )
    else:
        _numba_hybrid_filter = _numpy_hybrid_filter
        _numba_predator_detect = _numpy_predator_detect
        _numba_predator_escape = _numpy_predator_escape
except ImportError:
    # D7: _kernels.py failed to import (missing or errored).
    # Inline the numpy fallback implementations instead of re-importing
    # from the same broken module.
    _KERNELS_HAS_NUMBA = False

    def _numpy_hybrid_filter(neighbor_idx: np.ndarray, positions: np.ndarray,
                              active: np.ndarray, visual_range: float,
                              influence_count: int) -> None:
        """Inline numpy fallback — hybrid metric+topological filter."""
        active_idx = np.where(active)[0]
        vr_sq = visual_range * visual_range
        for global_i in active_idx:
            nbrs = neighbor_idx[global_i]
            valid = nbrs > 0
            if not valid.any():
                continue
            nbr_indices = nbrs[valid]
            diffs = positions[nbr_indices] - positions[global_i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            in_range = dists_sq <= vr_sq
            nbr_indices = nbr_indices[in_range]
            if len(nbr_indices) > influence_count:
                nbr_indices = nbr_indices[:influence_count]
            neighbor_idx[global_i, :] = 0
            neighbor_idx[global_i, :len(nbr_indices)] = nbr_indices

    def _numpy_predator_detect(threatened: np.ndarray, neighbor_idx: np.ndarray,
                                is_predator: np.ndarray,
                                active: np.ndarray) -> None:
        """Inline numpy fallback — predator threat detection."""
        active_idx = np.where(active)[0]
        for global_i in active_idx:
            if is_predator[global_i]:
                continue
            nbrs = neighbor_idx[global_i]
            valid_nbrs = nbrs[nbrs > 0]
            if len(valid_nbrs) > 0 and is_predator[valid_nbrs].any():
                threatened[global_i] = True

    def _numpy_predator_escape(escape: np.ndarray, positions: np.ndarray,
                                neighbor_idx: np.ndarray,
                                is_predator: np.ndarray,
                                threatened: np.ndarray, active: np.ndarray,
                                escape_factor: float,
                                accel_boost: float,
                                box: np.ndarray | None = None) -> None:
        """Inline numpy fallback — predator escape force.

        S2.B3: box (3,) enables minimum-image (toroidal) escape distances.
        """
        from ...core.types import min_image
        wrap = box is not None and bool((box > 0).any())
        active_idx = np.where(active)[0]
        for global_i in active_idx:
            if not threatened[global_i]:
                continue
            nbrs = neighbor_idx[global_i]
            valid_nbrs = nbrs[nbrs > 0]
            if len(valid_nbrs) == 0:
                continue
            predator_mask = is_predator[valid_nbrs]
            if not predator_mask.any():
                continue
            predator_idx = valid_nbrs[predator_mask]
            diffs = positions[predator_idx] - positions[global_i]
            if wrap:
                assert box is not None  # wrap is only True when box is not None
                diffs = min_image(diffs, box)
            dists_sq = np.sum(diffs * diffs, axis=1)
            nearest = np.argmin(dists_sq)
            to_predator = diffs[nearest]
            d = np.sqrt(dists_sq[nearest])
            if d < 1e-6:
                continue
            direction = -to_predator / d
            escape[global_i] = direction * (
                escape_factor * accel_boost / (d * d))

    # No numba — both aliases point to numpy fallbacks
    _numba_hybrid_filter = _numpy_hybrid_filter
    _numba_predator_detect = _numpy_predator_detect
    _numba_predator_escape = _numpy_predator_escape


def _dispatch_kernels(config):
    """Return (hybrid, predator_detect, predator_escape) kernels.

    Selects numba-accelerated versions when numba is installed
    and config.perf.use_numba is True. Otherwise returns pure-numpy
    fallbacks.

    Safe for non-SimConfig configs (e.g., test FakeConfig): defaults
    to numpy fallbacks when config lacks a 'perf' attribute.
    """
    perf = getattr(config, 'perf', None)
    use_numba = getattr(perf, 'use_numba', False) if perf is not None else False
    if use_numba and _KERNELS_HAS_NUMBA:
        return _numba_hybrid_filter, _numba_predator_detect, _numba_predator_escape
    return _numpy_hybrid_filter, _numpy_predator_detect, _numpy_predator_escape

if TYPE_CHECKING:
    from ...core.config import SimConfig
    from ...core.types import SpatialIndex


# ── Modularity pass 9: spatial-mode term composition (S2.A5 contract) ──
# Ports field.py's ForceTerm/composeForces pattern to spatial mode —
# same shared infrastructure (_base.py), same "named, typed, runtime-
# togglable force contribution" contract. Unlike field mode's terms
# (which compute raw physics per term), spatial's terms are thin
# weight-multiplier wrappers around sep/align/coh/flow/escape/thrust
# arrays already computed earlier in compute() — a faithful 1:1
# extraction of the pre-existing weighted-sum expression, not a
# reformulation (verified byte-identical against golden trajectories).

@dataclass
class SpatialTermContext:
    """Shared per-frame context passed to every spatial-mode ForceTerm."""

    config: SimConfig
    positions: np.ndarray       # (N, 3) full-width
    velocities: np.ndarray      # (N, 3) full-width
    active: np.ndarray          # (N,) full-width bool
    sep: np.ndarray             # (N, 3) — from separation_force(), threat-zeroed
    align: np.ndarray           # (N, 3) — from alignment_force(), threat-zeroed
    coh: np.ndarray             # (N, 3) — from cohesion_force(), threat-zeroed
    sep_jitter: float
    align_jitter: float
    coh_jitter: float
    flow_contrib: np.ndarray    # (N, 3) — already fully scaled (flow_weight*0.22 baked in)
    escape_force: np.ndarray | None  # (N, 3) or None (no threatened birds this frame)


def _term_separation(fx: SpatialTermContext) -> np.ndarray:
    return fx.sep * (fx.config.separation_weight * fx.sep_jitter)


def _term_alignment(fx: SpatialTermContext) -> np.ndarray:
    return fx.align * (fx.config.alignment_weight * fx.align_jitter)


def _term_cohesion(fx: SpatialTermContext) -> np.ndarray:
    return fx.coh * (fx.config.cohesion_weight * fx.coh_jitter)


def _term_flow(fx: SpatialTermContext) -> np.ndarray:
    return fx.flow_contrib


def _term_predator_escape(fx: SpatialTermContext) -> np.ndarray:
    if fx.escape_force is None:
        return np.zeros((len(fx.positions), 3), dtype=np.float32)
    return fx.escape_force


def _term_forward_thrust(fx: SpatialTermContext) -> np.ndarray:
    """P11.5: Evolvable forward force — thrust toward cruise speed.
    F_fwd = w_fwd * (v0 - |v|) * v_hat : sign flips around v* = v0.
    """
    out = np.zeros((len(fx.positions), 3), dtype=np.float32)
    w_fwd = fx.config.spatial.w_fwd
    if w_fwd > 0.0:
        speeds = np.linalg.norm(fx.velocities, axis=1)
        moving = fx.active & (speeds > 1e-6)
        if moving.any():
            v_hat = fx.velocities[moving] / speeds[moving, np.newaxis]
            out[moving] = v_hat * (
                w_fwd * (fx.config.v0 - speeds[moving])
            )[:, np.newaxis]
    return out


# Order matches the pre-extraction expression exactly: sep+align+coh+flow
# (one Python expression), then += escape_force, then the masked
# forward-thrust addition — composeForces' left-to-right accumulation
# reproduces the same floating-point summation order bit-for-bit.
SPATIAL_TERMS: list[ForceTerm] = [
    ForceTerm("separation", fn=_term_separation),
    ForceTerm("alignment", fn=_term_alignment),
    ForceTerm("cohesion", fn=_term_cohesion),
    ForceTerm("flow", fn=_term_flow),
    ForceTerm("predator_escape", fn=_term_predator_escape),
    ForceTerm("forward_thrust", fn=_term_forward_thrust),
]


@register("spatial")
class SpatialMode(ForceMode):
    """Reynolds 1987 boids — separation + alignment + cohesion + noise."""

    needs_index = True

    @staticmethod
    def compute(
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        active: np.ndarray,
        index: SpatialIndex | None,
        rng: np.random.Generator,
        last_theta: np.ndarray,
        config: SimConfig,
    ) -> None:
        """Compute Reynolds boids forces: separation + alignment + cohesion + noise."""
        n_active = active.sum()
        if n_active == 0:
            return

        # Pass 1: Query neighbours via spatial index
        if index is None or not index.ready:
            return

        # C3: predator_perception_boost — fetched before the neighbour
        # query so predator rows can use a boosted visual_range.
        is_predator = getattr(config, '_is_predator', None)

        neighbor_idx = NEIGHBOR_SELECTOR_REGISTRY["hybrid"].select(
            positions, velocities, active, index, config,
            filter_mode=config.spatial.neighbor_filter,
            is_predator=is_predator,
        )

        # ── Noise mode: additive | maxwellian | none ──
        noise_mode = config.spatial.noise_mode

        # ── P4.3: Predator detection ──
        # Dispatch numba vs numpy kernels based on config.perf.use_numba
        _hybrid_kernel, _pred_detect_kernel, _pred_escape_kernel = \
            _dispatch_kernels(config)
        threatened: np.ndarray | None = None
        if is_predator is not None and is_predator.any():
            threatened = np.zeros(len(positions), dtype=bool)
            _pred_detect_kernel(
                threatened, neighbor_idx, is_predator, active,
            )

        # ── P11.5: Per-interaction perception cones + max distances ──
        # Defaults (max_dist=0, cos angle=−1) leave the shared neighbour
        # set untouched, so the hot path is unchanged unless genes are set.
        # S2.B1: separation_distance is an additional absolute gate (0 =
        # off, falls back to max_dist_sep); alignment_radius_ratio scales
        # visual_range for a tighter alignment-only subset (1.0 = no
        # extra restriction, falls back to max_dist_align).
        sep_gate = (
            config.spatial.separation_distance
            if config.spatial.separation_distance > 0.0
            else config.spatial.max_dist_sep
        )
        align_gate = config.spatial.max_dist_align
        if config.spatial.alignment_radius_ratio < 1.0:
            ratio_gate = config.spatial.alignment_radius_ratio * config.visual_range
            align_gate = ratio_gate if align_gate <= 0.0 else min(align_gate, ratio_gate)
        sep_idx = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            sep_gate,
            config.spatial.angle_sep)
        align_idx = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            align_gate,
            config.spatial.angle_align)
        coh_idx = _maybe_perception_filter(
            positions, velocities, neighbor_idx, active,
            config.spatial.max_dist_coh,
            config.spatial.angle_coh)

        # Pass 2: Assemble primitives
        # S1.5: Kernel selector — "sum" | "mean" | "unit" for separation
        sep = separation_force(
            positions, velocities, sep_idx, active,
            kernel=config.spatial.separation_kernel,
            kernel_radius=config.spatial.separation_kernel_radius,
            kernel_zone_width=config.spatial.kernel_zone_width)
        if config.spatial.neighbor_filter == "global":
            # S2.B1: degenerate "global" case — alignment/cohesion steer
            # toward the whole-flock mean velocity / centre of mass, no
            # radius, no kNN (separation stays local via sep_idx above).
            align = np.zeros((len(positions), 3), dtype=np.float32)
            coh = np.zeros((len(positions), 3), dtype=np.float32)
            global_active_idx = np.where(active)[0]
            if len(global_active_idx) > 0:
                mean_vel = velocities[global_active_idx].mean(axis=0)
                mean_pos = positions[global_active_idx].mean(axis=0)
                # S1.5 forms: alignment = normalize(v̄ - v_i); cohesion = limit3(p̄ - p_i, 1)
                steer = mean_vel - velocities[global_active_idx]
                steer_norms = np.linalg.norm(steer, axis=1)
                valid = steer_norms > 1e-6
                align[global_active_idx[valid]] = (
                    steer[valid] / steer_norms[valid, np.newaxis]
                )
                to_center = mean_pos - positions[global_active_idx]
                lengths = np.linalg.norm(to_center, axis=1)
                coh_vecs = to_center.copy()
                long = lengths > 1.0
                coh_vecs[long] = to_center[long] / lengths[long, np.newaxis]
                coh[global_active_idx] = coh_vecs
        else:
            align = alignment_force(
                positions, velocities, align_idx, active,
                kernel=config.spatial.alignment_kernel,
                fov_min=config.spatial.angle_align,
                kernel_radius=config.spatial.separation_kernel_radius,
                kernel_zone_width=config.spatial.kernel_zone_width)
            coh = cohesion_force(
                positions, velocities, coh_idx, active,
                kernel=config.spatial.cohesion_kernel,
                kernel_radius=config.spatial.separation_kernel_radius,
                kernel_zone_width=config.spatial.kernel_zone_width)
        # S2.B2/Modularity pass 8: noise strategies own their own side
        # effects (e.g. "velocity"'s config._spatial_velocity_noise
        # one-shot stash for flock.integrate() to consume, "maxwellian"'s
        # direct velocity mutation) — see plugins/noise_strategy.py.
        # Unrecognised noise_mode values fall back to "additive", matching
        # this dispatch's pre-registry else-branch default exactly.
        noise_strategy = NOISE_STRATEGY_REGISTRY.get(
            noise_mode, NOISE_STRATEGY_REGISTRY["additive"]
        )
        noise_full = noise_strategy.apply(positions, velocities, active, n_active, config, rng)

        # ── P4.3: Predator effect — escape replaces separation, zero align/coh ──
        escape_force: np.ndarray | None = None
        if threatened is not None and threatened.any():
            t_idx = threatened & active
            # mypy narrow: is_predator is not None (guarded above)
            assert is_predator is not None
            # Compute predator escape force for threatened prey
            escape_force = _predator_escape(
                positions, neighbor_idx, is_predator, threatened, active,
                config,
            )
            # Zero alignment and cohesion for threatened birds
            align[t_idx] = 0.0
            coh[t_idx] = 0.0
            sep[t_idx] = 0.0

        # ── P4.5: Per-frame parameter jitter ──
        jitter_sep = config.spatial.jitter_separation
        jitter_coh = config.spatial.jitter_cohesion
        jitter_align = config.spatial.jitter_alignment
        # Seeded U(0, jitter) — deterministic per frame via seeded rng
        sep_jitter = 1.0 + rng.uniform(0.0, jitter_sep) if jitter_sep > 0 else 1.0
        coh_jitter = 1.0 + rng.uniform(0.0, jitter_coh) if jitter_coh > 0 else 1.0
        align_jitter = 1.0 + rng.uniform(0.0, jitter_align) if jitter_align > 0 else 1.0

        # ── P4.8: Coherence gate — reduce align/cohesion for small flocks ──
        # P4.8: Runtime-private field set by ecology extension.
        # NOT the static config.spatial.coherence_factor — that's the default;
        # the runtime override is written via config._coherence_factor by ecology.
        coherence = getattr(config, '_coherence_factor', 1.0)
        if coherence < 1.0:
            align_jitter *= coherence
            coh_jitter *= coherence

        # ── P4.2: Standardised force pipeline ──
        # 1. Accumulate weighted forces (with jitter + optional global flow)
        flow_w = config.spatial.flow_weight
        flow_contrib: np.ndarray = np.zeros_like(align)
        if flow_w > 0.0:
            # S2.B11: shared curl-flow L0 primitive (_base.py), also used
            # by FieldMode — gain is flow_weight*0.22, per spec.
            flow_active_idx = np.where(active)[0]
            flow_center = np.mean(positions[flow_active_idx], axis=0)
            flow_t = getattr(config, '_field_time', 0.0)
            flow_unit_scale = getattr(config, 'field_unit_scale', None)
            flow_U = float(flow_unit_scale) if flow_unit_scale is not None else (
                0.4 * min(config.width, config.height, config.depth)
            )
            flow_seeds = np.arange(len(flow_active_idx), dtype=np.float32)
            flow_contrib[flow_active_idx] = curl_flow(
                positions[flow_active_idx], flow_center, flow_seeds, flow_t, flow_U,
            ) * (flow_w * 0.22)
        # Modularity pass 9: sep/align/coh/flow/escape/forward-thrust,
        # composed via the shared S2.A5 ForceTerm/composeForces contract
        # (same summation order as the pre-extraction expression — see
        # SPATIAL_TERMS' comment).
        fx = SpatialTermContext(
            config=config,
            positions=positions,
            velocities=velocities,
            active=active,
            sep=sep,
            align=align,
            coh=coh,
            sep_jitter=sep_jitter,
            align_jitter=align_jitter,
            coh_jitter=coh_jitter,
            flow_contrib=flow_contrib,
            escape_force=escape_force,
        )
        accelerations += composeForces(fx, SPATIAL_TERMS, n=len(positions))

        # 2. Apply acceleration scale (global intensity multiplier)
        acceleration_scale = config.spatial.acceleration_scale
        if acceleration_scale != 1.0:
            accelerations *= acceleration_scale

        # 3. Clamp to max_force
        acc_mags = np.linalg.norm(accelerations, axis=1)
        too_strong = acc_mags > config.max_force
        if too_strong.any():
            accelerations[too_strong] = (
                accelerations[too_strong] /
                acc_mags[too_strong, np.newaxis] * config.max_force
            )

        # 4. Add noise AFTER clamping (noise should not be clamped)
        accelerations += noise_full


# Backward compatibility alias — tests import spatial_forces directly
spatial_forces: ForceFn = SpatialMode.compute  # type: ignore[assignment]
spatial_forces.needs_index = True
