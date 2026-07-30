# Steering Paradigms

This document defines the six core steering approaches found across 22
surveyed flocking implementations, from the classic Reynolds formulation
through GPU-era exponential smoothing to unique single-implementation
approaches.  Each paradigm represents a fundamentally different way of
computing a bird's new heading from its neighbours.

---

## 1. Reynolds Steering (`desired − velocity`)

The most common paradigm — 8 of 22 surveyed implementations use it.
The bird computes a desired velocity from separation, alignment, and
cohesion forces, then steers toward it by subtracting its current
velocity:

```
v_desired = separation + alignment + cohesion
steer     = clamp(v_desired − v_current, max_force)
v_new     = v_current + steer
```

The `desired − velocity` term gives the bird inertia — it accelerates
toward the target rather than snapping to it instantly.  The `max_force`
cap limits the per-frame acceleration, producing smooth curved
trajectories.

**Used by:** pymurmur spatial mode, pymurmur projection mode, murmuration
(§21), plus §02, §03, §12, §13, §14, §18 from the survey.

---

## 2. Force Accumulation (Factor Addition)

Forces from each behavioural rule are accumulated as direct velocity
additions rather than through a steering intermediary:

```
v_new = v_current + F_separation + F_alignment + F_cohesion
```

No `desired − velocity` subtraction, no max-force clamping.
The resulting velocity is typically clamped to a maximum speed
afterwards.

**Used by:** §01 (classic original), §06, §07a, §07b from the survey.

---

## 3. Angle-Based (Scalar Rotation, No Vectors)

Heading is stored as a scalar angle (2D) or axis-angle pair (3D).
Neighbour interactions compute angular deltas rather than vector
forces:

```
φ_new = φ_current + Δφ_separation + Δφ_alignment + Δφ_cohesion
v_new = speed · (cos φ_new, sin φ_new)
```

In 3D, the rotation is performed via Rodrigues' formula rotating the
heading around a computed axis toward a target direction.  This approach
never stores velocity vectors — only a direction and a speed scalar.

**Used by:** pymurmur angle mode, §04, §16, §17 from the survey.

---

## 4. Heading Blend (Normalized Sum)

Forces from each rule are computed as unit vectors (directions), summed,
and normalized to produce the new heading:

```
v_desired = normalize( d̂_separation + d̂_alignment + d̂_cohesion )
v_new     = speed · v_desired
```

Each rule contributes a direction with equal weight regardless of
distance.  The normalization step enforces unit-length output, so all
forces are inherently balanced — no tuning of relative weights needed.

**Used by:** §05, §11 from the survey. (pymurmur field mode does *not*
belong here despite superficially similar language — it composes
named force terms into an **acceleration** with a max-force clamp, no
normalization step; see §6.6.)

---

## 5. Exponential Smoothing (Lerp Toward Target)

The heading is blended toward a target direction via linear
interpolation (lerp), giving the bird persistent momentum:

```
v_target  = normalize(global_average_of_all_birds)
v_new     = normalize( lerp(v_current, v_target, smooth_factor) )
```

The smoothing factor controls inertia — low values produce slow,
smooth turns; high values produce quick snaps.  Unlike Reynolds
steering, the smoothing is applied to the direction directly rather
than through acceleration, so speed and direction are decoupled.

**Used by:** §05, §10a, §10b, §20 from the survey (all GPU compute
implementations).

---

## 6. Unique Approaches

Six implementations use steering paradigms not covered by the five
categories above.  Each is one of a kind in the survey:

### 6.1 Cosine-Zone Weighting (§08)

Forces from neighbours are weighted by a cosine of the angular
separation from the bird's heading, creating a smooth directional
sensitivity profile:

```
weight_j = cos(θ_j − θ_i)    // maximum broadside, zero perpendicular
v_new = Σ weight_j · v_j / Σ weight_j
```

### 6.2 Velocity Inertia Blend (§09)

The bird's current velocity is blended with the normalized flocking
force, giving strong directional persistence:

```
v_new = normalize( inertia·v_current + (1−inertia)·F_flocking )
```

### 6.3 Priority-Ordered Rule Selection (§15)

Behaviours are evaluated in priority order.  The first rule that
triggers sets the output; subsequent rules are ignored:

```
if obstacle_detected:     steer = avoid_obstacle()
elif neighbour_too_close: steer = separate()
elif has_neighbours:      steer = align_and_cohere()
else:                     steer = wander()
```

Each rule can consume the full force budget, so lower-priority rules
only execute if no higher-priority rule triggered. **Also used by
pymurmur** (as an optional, opt-in mode): a priority-stack allocator
combines obstacle-avoidance, predator-threat, and flocking forces
under a shared per-bird budget via the same binary-cutoff cascade —
tier 1 (obstacles) claims budget first and is clamped to it; tier 2
(threat) gets only what's left and is zeroed entirely, not scaled
down, if tier 1 alone already saturates the budget; tier 3 (flocking)
cascades the same way against tiers 1+2. This is disabled by default
(the ordinary weighted-sum composition is pymurmur's default across
every mode) but available as a config toggle.

### 6.4 Pearce Boundary-Seeking Projection (§21, §22)

The bird's heading is driven by the resolved direction `δ̂` of
light–dark domain boundaries on its view sphere — a visual projection
model rather than a neighbour-based steering rule.

### 6.5 Vicsek Constant-Speed Alignment (§22)

All birds move at fixed speed.  Each timestep, each bird aligns its
heading with the average direction of neighbours within a fixed radius,
with additive angular noise.  The alignment is blended with a memory
term via a coupling parameter η.

### 6.6 Field Force Compositing (§22)

Forces from up to 11 named terms (shell, target pull, slot repulsion,
tangential, buoyancy, curl flow, fold noise, field noise, viscous drag,
drift alignment, floating boundary) are composed via a shared context
and summed into a single acceleration with a max-force clamp.

---

## 7. Taxonomy

This document is itself a taxonomy — a cross-mode survey — rather than
a description of one thing, so its place in pymurmur's own architecture
is different from every other document in this collection: it maps
onto pymurmur's force-computation registry as a whole rather than onto
one entry in it. That registry is a per-strategy dispatch pattern: an
ABC plus a decorator populating a lookup table that a call site
consults at runtime instead of branching on a hardcoded chain,
currently holding 7 entries. Every pymurmur-specific paradigm named
above (§1 Reynolds, §3 Angle-based, §6.3's priority-stack note, §6.4
Pearce projection, §6.5 Vicsek, §6.6 field compositing) corresponds to
exactly one such registered strategy — this document is a reader's map
of *which* paradigm each registered mode uses, not a plugin family of
its own.

## 8. Beyond pymurmur

Paradigms surveyed above that no pymurmur mode currently uses:

- **§5 Exponential smoothing of heading** (not to be confused with
  pymurmur's own exponential-lerp *speed* enforcement policy, which
  smooths magnitude, not direction) — lerping the heading unit vector
  itself toward a target direction each frame
  (`v_new = normalize(lerp(v_current, v_target, smooth_factor))`)
  rather than accumulating a force and integrating it. None of
  pymurmur's 7 modes decouple direction-smoothing from
  force/acceleration this way; angle mode comes closest (it also
  bypasses acceleration) but caps by a turn-rate angle limit rather
  than a fractional lerp toward the target.
- **§2 Force accumulation without a clamp step** — pymurmur's own
  Reynolds-family modes (spatial, projection) both apply Reynolds
  steering (`desired − velocity`, clamped by `max_force`) rather than
  raw, unclamped factor addition; a mode implementing the older,
  simpler "sum forces directly onto velocity, clamp only the final
  speed" style would behave more twitchily under many simultaneous
  strong forces, since no single force is ever pre-limited before
  combining.
- **§6.1 Cosine-zone weighting as the primary steering paradigm** — a
  bird's entire heading computed as a cosine-weighted average over all
  neighbours (`v_new = Σ cos(θⱼ)·vⱼ / Σ cos(θⱼ)`) is different from how
  pymurmur uses angular weighting today: spatial mode has a
  FOV-weighted *alignment kernel* as one configurable option among
  several within its Reynolds pipeline, not a mode where cosine
  weighting alone determines the entire velocity update.

## 9. Summary

| Paradigm | Count | Mechanism | Inertia |
|----------|:-----:|-----------|:-------:|
| Reynolds (`desired − v`) | 8 | Steer toward computed target | Acceleration-based |
| Force accumulation | 4 | Direct factor addition to velocity | None (clamp only) |
| Angle-based | 4 | Scalar/axis-angle rotation | Turn rate cap |
| Heading blend | 2 | Normalized sum of unit directions | None |
| Exponential smoothing | 4 | Lerp toward target heading | Smoothing factor |
| Unique approaches | 6 | Various (one per implementation) | Varies |
