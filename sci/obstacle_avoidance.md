# Obstacle Avoidance

How a flocking simulation steers boids around static geometry. Every
approach must solve two sub-problems — *detection* (is an obstacle
close enough to matter, and where) and *response* (what force or
heading change to apply) — and implementations differ substantially in
both.

---

## 1. Detection & Response Strategies Surveyed

| Strategy | Detection | Response | Steering Override |
|----------|-----------|----------|:---:|
| Orthogonal projection | Per-obstacle perpendicular miss-distance check against current heading | Steer away from the closest threatening obstacle's perpendicular offset | Partial (weighted force) |
| Hard heading override | Nearest obstacle surface distance | Compute an avoid-point beyond the influence zone and set heading directly toward it | **Full** (flocking ignored while triggered) |
| See-ahead ray | Three sample points (ahead, half-ahead, self) checked against each obstacle | Steer away from the closest intersected obstacle | Partial (summed with other steering forces, high weight) |
| 8-directional raycast | A forward raycast hit triggers a search over 8 perpendicular candidate directions | Blend the original heading and best candidate direction by proximity, then raycast-verify the blend before adopting it | Partial (direction blend) |
| Single raycast feeler | A single forward raycast | `normal × (feelerDepth / dist)` — an inverse-distance force, entering the *highest-priority* slot of a force-accumulation stack | Highest priority in stack |
| Golden-spiral sampling | A forward raycast hit triggers a search | Sample ~1500 pre-generated, near-uniformly distributed directions (via the golden angle) for the first unobstructed path | Partial (added to separation) |
| SDF gradient + predictive TTC | Sign change in a signed-distance field (crossing from outside to inside) plus a closing-velocity check | A static fly-away force ramping with proximity, plus a predictive time-to-collision urgency term | Partial (added to acceleration) |

## 2. Notable Mechanisms

**Hard heading override** is the only approach that *completely
replaces* the flocking heading when triggered — separation, alignment,
and any target-seeking force are all ignored until the boid clears the
obstacle's influence zone. Every other surveyed strategy blends
avoidance with ongoing flocking as an additional or prioritized force,
never a full override.

**Golden-spiral sampling** is unique in exhaustively searching a large
pre-generated direction set (~1500 candidates, uniformly distributed
via the golden angle, filtered to a wide forward field of view) rather
than computing an avoidance direction analytically. This avoids local
minima that a purely analytic approach can fall into, at the cost of
doing many raycasts per avoidance decision.

**SDF gradient + predictive TTC** is unique in using signed distance
functions rather than any raycast or simple distance check — obstacle
geometry is *volumetric* (defined by the sign of a scalar field) rather
than surface-based, which is the only approach in the survey that
naturally supports compound shapes built from constructive solid
geometry (union/subtraction of primitives) without needing separate
per-shape-type collision logic.

## 3. Patterns

- **Most strategies are force-based, not overrides** — only the hard
  heading-override approach completely replaces steering; every other
  strategy adds obstacle avoidance as a weighted or prioritized force
  term alongside normal flocking.
- **Raycast-based detection dominates** — roughly half the surveyed
  strategies use some form of raycast (single feeler, multi-point
  see-ahead, 8-directional, or golden-spiral sampling); the rest use
  geometric projection, simple distance checks, or field-gradient
  evaluation.
- **Priority matters as much as detection mechanism** — some
  strategies give obstacle avoidance the highest possible priority
  (full override, or first slot in a priority stack), while others
  blend it in as one force among several, meaning a strong enough
  flocking pull can in principle still carry a boid toward danger.
- **Safety margins are implementation-specific tuning constants** in
  every surveyed strategy — there's no universal "correct" margin, only
  values tuned per simulation's scale and speed.

---

## 4. pymurmur's Approach: SDF Gradient + Predictive Time-to-Collision

pymurmur implements exactly one obstacle-avoidance strategy — the
SDF-gradient/predictive-TTC approach from the survey above — registered
as a single, hot-swappable strategy in the same per-strategy dispatch
pattern used throughout the codebase (an ABC plus a name-keyed lookup,
so a future second strategy could be added without touching call
sites). Obstacle geometry is defined independently of the avoidance
strategy itself.

### 4.1 Signed Distance Function Primitives

Three analytical primitives, each mapping an `(N,3)` position array to
an `(N,)` array of signed distances (negative = inside the shape,
positive = outside):

```
sphere(p)   = ‖p − center‖ − radius
box(p)      = ‖max(q, 0)‖ + min(max(q), 0),  where q = |p − center| − half_extents
cylinder(p) = combines a radial distance in one plane with a vertical
              distance along the height axis, using the same
              "outside term + inside term" SDF construction as box
```

Compound geometry is built via constructive solid geometry (CSG)
boolean operators applied directly to the distance values:

```
union(a, b)    = min(a, b)        // either shape's interior counts as inside
subtract(a, b) = max(a, −b)       // a, with b's volume carved out
```

Because these operate purely on scalar SDF values, arbitrarily nested
unions and subtractions of spheres, boxes, and cylinders compose into
one combined SDF for the whole scene with no special-casing per shape
type.

### 4.2 Gradient and Collision Detection

The gradient of the combined scene SDF is computed numerically via
central finite differences (one forward/backward evaluation per axis),
then normalized to a unit "away from surface" direction:

```
∇SDF(p) ≈ (SDF(p + ε·axis) − SDF(p − ε·axis)) / (2ε)   for each axis
away = ∇SDF / ‖∇SDF‖
```

A genuine collision (a bird actually entering an obstacle) is detected
by a sign flip between consecutive frames — `SDF` was positive
(outside) and is now negative (inside):

```
collided = sign(SDF_old) > 0  AND  sign(SDF_new) < 0
```

### 4.3 Static Fly-Away

Birds within `fly_away_max_dist` of a surface (but not yet inside it)
are pushed along the outward gradient, ramping linearly from zero at
the threshold distance to full strength exactly at the surface:

```
if 0 ≤ SDF(p) < fly_away_max_dist:
    ramp = 1 − SDF(p) / fly_away_max_dist
    a += away · static_weight · ramp
```

### 4.4 Predictive Time-to-Collision

Independently of the static term, a bird whose velocity is closing on
a surface fast enough to hit it within `min_time_to_collide` receives
an additional urgent steering push, even if it's currently farther away
than `fly_away_max_dist`:

```
closing = −∇SDF · v̂        // positive when approaching the surface
if closing > 0 and SDF(p) ≥ 0:
    ttc = SDF(p) / closing
    if ttc < min_time_to_collide:
        a += away · predictive_weight
```

This is what distinguishes the approach from a purely distance-based
static push: a fast-moving bird heading straight at a wall from far
away gets an early urgent correction that a proximity-only check would
miss entirely.

### 4.5 Kinematic Correction

As a last-resort backstop (not a steering force — a direct position
fix), any bird whose position ends up inside an obstacle after
integration is pushed back to the surface via a single Newton-like
step:

```
p ← p − SDF(p) · ∇SDF / ‖∇SDF‖²
```

only applied when `SDF(p) < 0` (genuinely inside). If the gradient is
near-zero (the degenerate case of a bird sitting exactly at a sphere's
centre, where the gradient is undefined), the position is perturbed
slightly first so the gradient becomes well-defined before correcting.

---

## 5. Taxonomy

pymurmur's obstacle-avoidance system is the ObstacleAvoidanceStrategy
plugin family — architecturally the same per-strategy dispatch pattern
(an ABC plus a name-keyed lookup table, so a call site selects a
strategy by name at runtime instead of branching on a hardcoded chain)
used throughout the codebase for every other swappable computation:
domain-edge handling, per-mode neighbour selection, post-integrate
speed enforcement, spatial-index selection, kernel dispatch for
separation/alignment/cohesion, and noise injection are each their own
such registry. Obstacle avoidance currently has exactly one registered
entry (§4's SDF-gradient-plus-predictive-TTC strategy) rather than
several — but the registry-of-one is still meaningful as an
*extensibility contract*, not just a count: any future second obstacle
strategy plugs in beside it without touching the call site that
dispatches by name, exactly as adding a new force mode or a new speed
policy does elsewhere in the codebase. The single current entry is not
a special case; it's simply the only strategy written so far.

## 6. Beyond pymurmur: Unimplemented Extensions

Two mechanisms from §1's survey that pymurmur's SDF-only approach does
not implement:

- **Golden-spiral direction sampling** (§1's golden-spiral strategy) —
  rather than following an analytic gradient, pre-generate a large set
  of near-uniformly distributed candidate directions (via the golden
  angle, filtered to a forward field of view) and raycast-search them
  for the first unobstructed path. This avoids local minima an
  analytic gradient method can fall into (e.g. a gradient that points
  directly into a concave pocket of the combined SDF), at the cost of
  many raycasts per avoidance decision — the tradeoff is exhaustive
  search robustness against a real per-frame cost that scales with
  sample count. Adding it would mean a second registered strategy
  requiring a raycast primitive against the same SDF scene (a
  root-finding walk along each candidate ray, rather than the current
  approach's local gradient evaluation).
- **Multi-point see-ahead sampling** (§1's see-ahead-ray strategy) —
  checking several sample points along the projected forward path
  (not just the bird's current position) against each obstacle,
  giving a graduated detection distance rather than relying entirely
  on the SDF value's own smooth ramp. Since the current SDF approach
  already gets a continuous distance field "for free," this would
  mostly matter for very fast-moving birds whose per-frame displacement
  is large relative to `fly_away_max_dist` — a case the current static
  fly-away term alone might under-react to before the predictive TTC
  term (§4.4) engages.

## 7. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|--------------|
| `static_weight` | 0.0 | Static fly-away force magnitude (0 disables the term entirely) |
| `predictive_weight` | 0.0 | Predictive time-to-collision force magnitude (0 disables the term) |
| `fly_away_max_dist` | 0.0 | Distance from a surface at which the static term begins ramping up |
| `min_time_to_collide` | 0.0 | Time-to-collision threshold below which the predictive term engages |
