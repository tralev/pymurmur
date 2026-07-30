# Angle Mode (Rodrigues Rotation)

This document defines the angle-based steering mode: each bird stores its
heading as its velocity direction, and steering is performed via Rodrigues
rotation about an axis rather than through acceleration-based physics.
The mode implements unified neighbour behaviour zones, adaptive speed,
edge avoidance, per-frame heading jitter, and a dead-zone turn threshold.

---

## 1. Velocity Update

Angle mode steers via direct velocity assignment rather than force
accumulation.  Each frame, the bird's heading (unit velocity direction)
is rotated toward a computed target direction, then the result is scaled
to the current adaptive speed:

```
v_new = rotate_about(hdg, axis, Δθ) × speed
```

where `hdg` is the current heading (velocity direction, or a random
unit vector if stationary), `axis` is the rotation axis (hdg × target),
and `Δθ` is the capped turn angle.  The velocity magnitude is set
directly — not integrated from acceleration.

---

## 2. Rodrigues Rotation

The core rotation is the Rodrigues formula, rotating a vector `v` around
a unit axis `k` by angle `θ`:

```
rotate_about(v, k, θ) = v·cos θ + (k × v)·sin θ + k·(v·k)·(1 − cos θ)
```

For the per-bird steering step:

```
cos φ = clamp(hdg · target, −1, 1)
φ     = arccos(cos φ)                         // angular error
axis  = normalize(hdg × target)               // rotation axis

if φ > turn_threshold:                         // dead zone
    Δθ = min(φ, turn_rate × dt)               // cap turn angle
    hdg = rotate_about(hdg, axis, Δθ)
```

The rotation axis is `hdg × target`.  When hdg and target are parallel
or anti-parallel (cross product near zero), a fallback perpendicular
axis is chosen — first `hdg × (1,0,0)`, then `(0,1,0)` if that also
degenerates.

The dead zone (`turn_threshold`, default 0.5°) prevents jitter when
the heading is already near the target.  The turn rate is capped at
`turn_rate × dt` (default turn_rate = 120°/s), with a `max_turn_rate`
(default 540°/s) used for edge avoidance.

---

## 3. Unified Neighbour Modes

Neighbours are found via k-NN spatial index query (`k = n_neighbors + 1`).
For each bird, the nearest neighbour distance determines which of three
behaviour modes applies:

### 3.1 Flee (nearest neighbour < sep_radius)

Steer directly away from the closest neighbour:

```
to_nbr = p_nearest − p_self            // toroidal-aware if border_mode = "toroidal"
target = normalize(−to_nbr)
```

Flee uses **full turn_rate** — not gated by the coherence factor
(safety-critical, like separation in spatial mode).

### 3.2 Align + Cohere (nearest neighbour < align_radius)

Steer toward the normalized sum of the centroid direction and the mean
neighbour heading:

```
centroid = mean(p_neighbours)            // toroidal-aware
c_hat    = normalize(centroid − p_self)  // cohesion direction
m_hat    = normalize(Σ v̂_neighbours)     // mean heading direction
target   = normalize(c_hat + m_hat)
```

Alignment/cohesion turn rate is **gated by the coherence factor** —
reduced for small flocks at dusk (the coherence gate, shared with
spatial and projection modes).

### 3.3 Cohere Only (nearest neighbour < range_radius)

Steer toward the neighbour centroid with no alignment term:

```
target = normalize(centroid − p_self)
```

### 3.4 No Neighbours

If no neighbour lies within `range_radius`, no target is computed and
the bird maintains its current heading (subject to jitter and edge
avoidance only).

The three radii are scaled by `boid_size` (body radius `b`):

```
sep_radius   = sep_radius_bodies  × b      // default: 1.0b
align_radius = align_radius_bodies × b      // default: 5.0b
range_radius = range_radius_bodies × b      // default: 12.0b
```

---

## 4. Adaptive Speed

Isolated birds fly faster; surrounded birds fly at base speed.  The
speed bonus depends on the deficit — how many fewer neighbours a bird
has than the target count `n_neighbors`:

```
deficit   = n_neighbors − n_nbrs              // positive = isolated
if deficit > 0:
    new_speed = base_speed + bonus(deficit)
else:
    new_speed = base_speed
```

Three deficit laws are available:

| Mode | Bonus formula | Behaviour |
|------|--------------|-----------|
| Linear (default) | `deficit × 5.0` | Proportional, uncapped |
| Quadratic | `min(deficit_cap, deficit²)` | Gentle for small deficits, saturating |
| Softened | `min(deficit_cap, deficit² / 2)` | Half the quadratic gain |

The `deficit_cap = n_neighbors²` — the cap grows quadratically with the
target neighbour count, so a higher target allows a higher top speed.

Example: a bird missing 4 neighbours (target 7, actual 3) with the
linear law gets `bonus = 4 × 5 = 20`.  With the default
`base_speed = 150.0`, the bird flies at `150 + 20 = 170` — about 1.13×
its base speed, a modest nudge back toward the flock rather than a
dramatic multiple (the bonus formula's absolute scale is small
relative to this mode's actual cruise speed).

---

## 5. Edge Handling

Angle mode supports two boundary strategies: **margin** (cube domain)
and **sphere** (spherical domain).

### 5.1 Cube Margin

For each of the 6 domain faces, if the bird is within `margin` units
of the face, an edge target is the inward-facing face normal.  The
closest face wins:

```
if px < margin:          face_normal = (+1, 0, 0)
if px > width − margin:  face_normal = (−1, 0, 0)
// ... same for y, z axes

if closest_face_dist < margin:
    edge_target  = face_normal
    turn_rate_now = turn_rate + (1 − dist/margin) × (max_turn_rate − turn_rate)
```

The turn rate ramps linearly from `turn_rate` at the margin boundary to
`max_turn_rate` at the wall — birds turn harder the closer they get.

### 5.2 Sphere

```
dist = ‖p_self‖
if dist > sphere_radius − margin:
    edge_target  = normalize(−p_self)    // toward centre
    turn_rate_now = turn_rate + (1 − (sphere_radius − dist)/margin) × (max_turn_rate − turn_rate)
```

If the bird is outside the sphere radius (penetrated), `(sphere_radius − dist)`
is negative, so the edge factor exceeds 1.0 — clamped implicitly by the
turn cap.

Both strategies use **full turn_rate** — not gated by the coherence
factor (safety-critical).

### 5.3 Toroidal

No edge handling — wrap handles position, and the spatial index handles
neighbour queries with min-image distance correction.

---

## 6. Heading Jitter

Before computing the target, a random perturbation is applied:

```
jitter_rad  = uniform(−jitter_deg°, +jitter_deg°)   // converted to radians
jitter_axis = normalize(uniform(−1, 1, 3))           // random unit vector
hdg         = rotate_about(hdg, jitter_axis, jitter_rad)
```

Default `jitter_deg = 4°`.  Set to 0 to disable.

---

## 7. Edge + Flee Target Composition

When both edge avoidance and a neighbour-based target are active, they
are combined:

```
if edge_target is not None:
    if target is not None:
        target = normalize(target + edge_target)    // blend both
    else:
        target = edge_target                        // edge only
```

This enables a bird fleeing a neighbour near the wall to turn away from
both simultaneously — the averaged target direction balances social and
boundary avoidance.

---

## 8. Coherence Gate

A runtime gate reduces alignment/cohesion steering responsiveness for
small flocks at dusk.  The coherence factor (`_coherence_factor`,
set by the ecology extension, default 1.0) scales the turn rate:

```
if not is_fleeing and not edge_only and coherence < 1.0:
    gated_turn = turn_rate_now × coherence
else:
    gated_turn = turn_rate_now    // flee + edge: full rate
```

This is consistent with spatial and projection modes, which gate
alignment/cohesion weights but leave separation at full strength.
Angle mode's flee and edge avoidance are safety-critical and use
full `turn_rate` regardless.

---

## 9. Incremental Spatial Grid

Angle mode uses an incremental spatial grid update: each bird tracks
its last cell (`_angle_last_cell`), and only birds that cross cell
boundaries are re-filed.  This avoids a full grid rebuild each frame
for flocking modes where birds move incrementally rather than jumping
large distances.

---

## 10. Taxonomy

Angle mode is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table,
selected at runtime instead of branching on a hardcoded if/elif chain.
The six sibling strategies in the same family: projection (steering
derived from occlusion geometry — the least-occluded direction of the
visual sphere), spatial (classic Reynolds separation/alignment/cohesion
force summation), field (a large set of named force terms composited
into one blob-following acceleration), vicsek (constant-speed heading
averaging with tangent-plane noise), influencer (birds steer toward a
tick-driven scripted target rather than each other), and marl (steering
deferred entirely to an external controller).

Within that family, angle mode is architecturally distinct: it is the
only strategy that steers via **direct rotation of a heading vector**
(Rodrigues rotation toward a target direction, capped by a turn rate)
rather than by accumulating a force or acceleration and integrating it.
Every other mode either sums forces (spatial, field) or blends
directions into a target velocity (projection, vicsek, influencer) that
still passes through the same acceleration/integration machinery; angle
mode bypasses that machinery and assigns velocity directly from a
rotated heading and an independently-computed scalar speed.

## 11. Beyond pymurmur

Techniques from the broader flocking-simulation literature not
currently implemented in angle mode:

- **Reynolds wander** — a jittered target point on a circle projected
  ahead of the bird (`target = normalize(jitter_on_circle) × wanderRadius`,
  then steered toward `aheadDistance × forward + target`). Angle mode's
  own "no neighbours" case (§3.4) currently just holds the last heading;
  a wander term would give isolated birds a continuously exploring drift
  instead of a static line, at the cost of one extra per-bird persistent
  jitter-angle state variable (wander is stateful — the jitter point
  needs to persist and re-jitter incrementally, not redraw from scratch
  every frame, or the motion looks twitchy rather than wandering).
- **Banking/roll on turns** — tilting a secondary "up" orientation
  vector toward the current turn's acceleration direction (an
  exponential lerp of `up` toward `worldUp + turnAccel × bankGain`).
  Angle mode already computes a per-frame rotation axis and angle (§2);
  a bank angle could be derived from the same `axis`/`Δθ` pair as a
  purely cosmetic secondary orientation, without touching the heading
  update itself — this would be a rendering-layer addition, not a
  physics change, since angle mode's own state is heading-only.
- **Priority-ordered turn-rate stack** — rather than angle mode's
  three mutually-exclusive distance-banded modes (flee / align+cohere /
  cohere-only, §3), some implementations resolve competing steering
  targets by evaluating them in strict priority order and stopping once
  the turn-rate budget for the frame is spent (obstacle > flee >
  separation > alignment > cohesion > wander), rather than picking
  exactly one band by nearest-neighbour distance. This would let, e.g.,
  a distant flee target still contribute a small correction even while
  a closer align target dominates, instead of flee fully suppressing
  alignment whenever any neighbour is inside `sep_radius`.

## 12. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `turn_rate` | 120°/s | Base turn rate |
| `max_turn_rate` | 200°/s | Maximum turn rate (edge avoidance) |
| `turn_threshold` | 0.5° | Dead zone — no turn below this |
| `jitter_deg` | 4° | Per-frame heading jitter |
| `base_speed` | 150.0 | Base cruise speed |
| `sep_radius_bodies` | 1.0 | Separation radius in body units |
| `align_radius_bodies` | 5.0 | Alignment radius in body units |
| `range_radius_bodies` | 12.0 | Cohesion-only radius in body units |
| `n_neighbors` | 7 | Target neighbour count |
| `angle_speed_mode` | "linear" | Deficit law: linear, quadratic, softened |
