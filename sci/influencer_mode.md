# Influencer Mode

This document defines the influencer mode: a persistent tick-driven
3D Lissajous target with move-then-steer at unit speed, rank-based or
distance-based influence weights, density-scaled Gaussian position
initialization, and a desktop pilotable-flock mode.

---

## 1. Lissajous Target (P7.1)

The target follows a 3D Lissajous curve with primary (orbit) and
secondary (flutter) terms per axis:

```
x(t) = sin(t / f_px) · A_px  +  cos(t / f_sx) · A_sx
y(t) = cos((t + φ_y) / f_py) · A_py  +  sin((47 − t) / f_sy) · A_sy
z(t) = cos((t + φ_z) / f_pz) · A_pz  +  sin((t + 13) / f_sz) · A_sz  +  v_offset
```

The raw coordinates are scaled by `s` and offset to the domain centre:

```
s = scale · min(W/460, H/460, D/254)
T(t) = C + (T_raw(t) − (0, 0, v_offset)) · s + (0, 0, v_offset·s)
```

A persistent tick counter `_influencer_tick` advances by `tick_rate`
each substep, ensuring deterministic, repeatable trajectories.

Default coefficients:

| Axis | Primary freq | Primary amp | Secondary freq | Secondary amp | Phase offset |
|------|:-----------:|:-----------:|:--------------:|:-------------:|:------------:|
| x | 97 | 200 | 217 | 30 | 0 |
| y | 29 | 200 | 13 | 30 | 53 |
| z | 41 | 100 | 7 | 27 | 61 |

Vertical offset: `v_offset = 40`.

---

## 2. Move-Then-Steer (P7.2)

Influencer mode owns position updates (`owns_positions = True`).
Each substep advances positions along the current velocity, then
steers toward the target:

```
for each substep:
    p ← p + v · dt_sub                  // move
    d̂_new = steer(d̂_old, target)        // steer
    v = d̂_new · v0                       // constant speed
```

This is a direction-based, not acceleration-based, update.  Birds move
at constant speed `v0` and change direction only.  The engine's
`integrate()` is called with `move=False` for this mode — the
per-substep move here is the only position advance.

---

## 3. Influence Weights (P7.3)

Two influence modes determine how strongly each bird steers toward
the target:

### 3.1 Rank-Based

Birds are ranked by distance to the target.  Closer birds steer more
weakly (they are already near the target); farther birds steer more
strongly:

```
ranks = argsort(dists).argsort() / (N − 1)     // 0 = closest, 1 = farthest
influence = (1 − ranks · 0.8) ^ rank_exponent
```

The `rank_exponent` (default 4.0) controls how sharply influence
falls off with rank.

### 3.2 Distance-Based

Influence is a function of absolute distance, clipped to a range:

```
raw = near_dist_sq · s² / (d² + ε)
influence = clamp(raw, influence_min, influence_max)      // default: [0.01, 3.5]
```

Near the target, influence approaches `influence_max` (strong steer);
far away, it approaches `influence_min` (weak steer).

### 3.3 Steering Blend

The new direction is a blend of the old direction and the direction
toward the target:

```
t̂ = normalize(target − p)
d̂_new = normalize( d̂_old · (1 − influence) + t̂ · influence )
```

Zero-velocity birds are assigned a random Fibonacci-sphere direction
before blending.

---

## 4. Density-Scaled Initialization (P7.4)

Initial positions follow a Gaussian distribution whose spread scales
with flock size:

```
σ = N^(1/3) · separation · scale
positions = N(0, σ)₃ + C + U(0, 10·scale)³
```

The `σ ∝ N^(1/3)` scaling ensures the initial cloud volume grows
linearly with N — larger flocks start more spread out.  The shared
uniform offset `U(0, 10·scale)³` shifts the entire cloud (not
per-bird jitter), breaking symmetry from the domain centre.

---

## 5. Distance Diagnostics (P7.5)

After the final substep, per-frame diagnostics are stored on the
config for metrics collection:

```
target_dist_min  = min ‖p_i − T‖
target_dist_max  = max ‖p_i − T‖
```

The final substep's target position is stashed for the marker
renderer (`_influencer_target_pos`).

---

## 6. Pilot Mode (P7.6)

When a `PilotTarget` is active, all birds steer toward a
user-controlled position instead of the Lissajous target:

```
F_heading = pilot_heading · 0.12                     // alignment force
F_core    = (pilot_pos − p) · 0.22                    // core follow (unbounded)
if d > shell_radius:                                  // shell pull
    F_shell = t̂ · (d − shell_radius) · 0.42
F_total   = F_heading + F_core + F_shell
v ← v + F_total · dt_sub                              // velocity integration
|v| = v0                                             // constant speed
```

The pilot position is updated via WASD keyboard input.  The shell
radius can be expanded/contracted with scatter/gather toggles
(Shift/Alt), clamped to `[0.42, 2.2]`.

---

## 7. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `influencer_scale` | 1.0 | Spatial scale factor |
| `influencer_substeps` | 2 | Move-then-steer substeps per frame |
| `influencer_tick_rate` | 1.0 | Tick increment per substep |
| `influencer_influence_mode` | "rank" | "rank" or "distance" |
| `influencer_rank_exponent` | 4.0 | Rank falloff exponent |
| `influencer_near_dist_sq` | 1.0 | Near-distance reference for distance mode |
| `influencer_influence_min` | 0.01 | Minimum influence (far away) |
| `influencer_influence_max` | 3.5 | Maximum influence (near target) |
| `influencer_use_rank_override` | false | Force rank-based influence |
| `influencer_init_separation` | 1.0 | Initial Gaussian σ factor |
| `influencer_target_freq_primary` | (97, 29, 41) | Primary Lissajous frequencies |
| `influencer_target_freq_secondary` | (217, 13, 7) | Secondary Lissajous frequencies |
| `influencer_target_amp_primary` | (200, 200, 100) | Primary amplitudes |
| `influencer_target_amp_secondary` | (30, 30, 27) | Secondary amplitudes |
| `influencer_target_phase_offsets` | (0, 53, 61) | Phase offsets per axis |
| `influencer_target_vert_offset` | 40.0 | Vertical offset |

---

## 8. Taxonomy

Influencer mode is one of pymurmur's 7 interchangeable
force-computation strategies — a per-strategy dispatch registry: an
ABC (or shared-signature callable) plus a decorator populating a
lookup table, selected at runtime instead of branching on a hardcoded
if/elif chain. Its 6 siblings: spatial (the literal Reynolds triad),
projection (occlusion-geometry-driven boundary-seeking), field
(target-seeking blob/anchor compositing), vicsek (constant-speed
angle-coupling alignment), angle (turn-rate-limited Rodrigues-rotation
steering), and marl (deferred control under an external per-bird
policy).

Influencer mode is architecturally unique among the seven in two
ways: it is the only mode that owns bird positions directly
(`owns_positions = True` — every other mode derives position purely
from integrating a computed velocity/acceleration), and it is the
only mode driven by a persistent tick counter producing a fully
deterministic, choreographed trajectory rather than a per-frame
reactive computation from neighbour state.

## 9. Beyond pymurmur: Unimplemented Extensions

- **Multiple simultaneous targets/leaders.** Influencer mode has
  exactly one global target (the Lissajous curve, or the single pilot
  position in pilot mode) that every bird's influence weight is
  computed against. Field mode's leader/chaser groups demonstrate
  pymurmur already has machinery for per-group targets elsewhere, but
  influencer mode itself has no multi-target variant — birds could
  instead be partitioned into groups each following an independently
  phased Lissajous curve, or ranked against the nearest of several
  simultaneous targets rather than a single global one.
- **Obstacle-aware hard heading override.** Some implementations give
  obstacle avoidance a hard priority override — flocking (or in this
  case, target-steering) is fully suspended while an obstacle is close,
  rather than blended in as one more force. Influencer mode's
  move-then-steer update has no obstacle-avoidance term of its own at
  all; a bird would fly straight through an SDF obstacle unless
  something external intervened.
- **Banking/orientation animation tied to steering.** A visual
  concern, not a physics one — tilting a bird's rendered orientation
  proportional to its turn rate during target pursuit. Out of scope
  for the simulation core regardless; noted for completeness since
  influencer mode's persistent, smooth curve trajectory is the pymurmur
  mode best suited to showcase it.
