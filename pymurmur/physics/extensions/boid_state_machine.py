"""BoidStateMachine extension — generic, threshold-driven per-boid state.

Every other extension in this session (Predator, SpeedNoise,
NeighborAdaptiveSpeed, DynamicVisionRange) has an implicit "state" of
its own, but only Predator has anything resembling a state machine —
and that's a single scalar (approach/egress) on the one predator agent,
not a per-boid mechanism. This extension gives ordinary boids a small,
fixed, config-driven set of states (normal/isolated/crowded/threatened)
evaluated in priority order — first match wins, mirroring the
allocate_priority_budget binary-priority precedent — each mapping to a
speed-cap multiplier, composed the same way as every other multiplier
extension this session.

Not a general FSM with arbitrary states/transitions: the three
non-normal states and their trigger conditions are fixed; only the
thresholds and multipliers are configurable. A genuinely open-ended
state machine (arbitrary states, arbitrary transition logic) would
need a different design (e.g. a rule-expression language) — out of
scope here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import Extension
from .extension_registry import register_extension

if TYPE_CHECKING:
    from ..flock import PhysicsFlock
    from ._base import StepContext

STATE_NORMAL = 0
STATE_ISOLATED = 1
STATE_CROWDED = 2
STATE_THREATENED = 3


@register_extension("boid_state_machine_enabled", "boid_state_speed_mult")
class BoidStateMachine(Extension):
    """Assigns each active boid one of 4 states based on local neighbor
    count and (if the Predator extension is active) threat proximity,
    then applies that state's configured speed-cap multiplier.

    Priority order (first match wins): threatened > isolated > crowded
    > normal. A boid can't be both "isolated" and "crowded" (mutually
    exclusive by construction — neighbor count is either below the
    isolated threshold, above the crowded threshold, or between them),
    but threatened is checked first regardless of neighbor count, since
    fleeing takes priority over density-driven speed changes.
    """

    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None:
        cfg = ctx.config.boid_state_machine
        n = len(flock.positions)
        state = np.full(n, STATE_NORMAL, dtype=np.int8)
        mult = np.ones(n, dtype=np.float32)

        active_idx = np.where(flock.active)[0]
        if len(active_idx) == 0:
            flock.boid_state = state
            flock.boid_state_speed_mult = mult
            return

        index = flock._index
        neighbor_count = np.zeros(len(active_idx), dtype=np.float32)
        if index is not None and getattr(index, "ready", False):
            radius_sq = cfg.boid_state_neighbor_radius ** 2
            k = cfg.boid_state_sample_k
            for row, i in enumerate(active_idx):
                nbrs = index.query_knn(flock.positions[i], k)
                if len(nbrs) == 0:
                    continue
                diffs = flock.positions[nbrs] - flock.positions[i]
                dists_sq = np.sum(diffs * diffs, axis=1)
                neighbor_count[row] = np.count_nonzero(dists_sq <= radius_sq)

        threat_prox = ctx.threat_prox
        if threat_prox is None:
            threatened = np.zeros(len(active_idx), dtype=bool)
        else:
            threatened = threat_prox[active_idx] > cfg.boid_state_threatened_proximity_threshold

        isolated = neighbor_count < cfg.boid_state_isolated_neighbor_threshold
        crowded = neighbor_count > cfg.boid_state_crowded_neighbor_threshold

        row_state = np.full(len(active_idx), STATE_NORMAL, dtype=np.int8)
        row_mult = np.ones(len(active_idx), dtype=np.float32)

        # Lowest priority first, so higher-priority assignments overwrite.
        row_state[crowded] = STATE_CROWDED
        row_mult[crowded] = cfg.boid_state_crowded_speed_mult
        row_state[isolated] = STATE_ISOLATED
        row_mult[isolated] = cfg.boid_state_isolated_speed_mult
        row_state[threatened] = STATE_THREATENED
        row_mult[threatened] = cfg.boid_state_threatened_speed_mult

        state[active_idx] = row_state
        mult[active_idx] = row_mult

        flock.boid_state = state
        flock.boid_state_speed_mult = mult
