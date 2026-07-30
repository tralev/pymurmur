# Force Kernels

This document defines all 18 force kernels — the per-neighbour
weighting functions used by the spatial mode's separation, alignment,
and cohesion force primitives.  Each kernel operates on precomputed
per-neighbour geometry arrays and returns a force contribution vector.

---

## Part A: Separation Kernels (11)

Separation steers each bird away from its close neighbours.  A kernel
function `K` maps the per-neighbour geometry (difference vectors,
distances, and optional extra parameters like heading or closing
speed) to a force contribution for each neighbour.  The total
separation force is the sum over all neighbours:

```
F_sep_i = Σ_{j ∈ neighbors} K(diffs_j, dists_j, ...)
```

All separation kernels receive: `diffs` (vectors from bird to each
neighbour, shape `(..., k, 3)`), `dists` (Euclidean distances, shape
`(..., k)`), and `close` (boolean mask of valid neighbours).  Each
returns a 3D force vector — the summed contribution of all neighbours.

The unit vector pointing from neighbour `j` to bird `i` is:

```
r̂_ji = −diffs_j / dists_j
```

---

### A1. Sum (default)

```
w(d) = 1
F_sep = Σ_j r̂_ji / d_j²
```

The classic Reynolds separation: force falls as 1/d², so near
neighbours repel much more strongly than distant ones.  Magnitude is
density-dependent — a bird with many neighbours receives a larger
total force than one with few.

- Parameters: none
- Key property: simplest, most widely used

---

### A2. Mean

```
F_sep = (1/k) · Σ_j r̂_ji / d_j²     where k = number of neighbours
```

Density-invariant version of `sum`.  Dividing by the neighbour count
means a bird with 20 neighbours receives roughly the same total force
as a bird with 3, assuming similar average distance.  Prevents
over-separation in dense regions while maintaining the 1/d² falloff
per neighbour.

- Parameters: none
- Key property: density-invariant

---

### A3. Unit

```
F_sep = Σ_j r̂_ji
```

Distance-independent: every neighbour contributes exactly one unit
vector, regardless of how far away it is.  A bird with many
neighbours receives a much larger force than one with few, but the
force per neighbour is constant — there is no 1/d or 1/d² decay.

- Parameters: none
- Key property: all neighbours contribute equally; no distance falloff

---

### A4. Exponential (exp)

```
w(d) = exp( −(d − r) / r )
F_sep = Σ_j w(d_j) · r̂_ji
```

Exponential decay with length scale `r` (the kernel radius).  At
`d = r`, the weight is `exp(0) = 1`.  At `d = 2r`, the weight is
`exp(−1) ≈ 0.37`.  At `d = 0`, the weight is `exp(1) ≈ 2.72`.

- Parameters: `radius` (length scale)
- Key property: smooth, continuous, never reaches exactly zero

---

### A5. Linear Ramp

```
w(d) = max(r − d, 0)
F_sep = Σ_j w(d_j) · r̂_ji
```

Weight falls linearly from `r` at `d = 0` to `0` at `d = r`.
Neighbours beyond `r` contribute nothing — a hard cutoff.  Inside the
radius, nearer neighbours push harder (linearly).

- Parameters: `radius` (cutoff distance)
- Key property: hard cutoff, linear decay

---

### A6. Asymptotic

```
w(d) = max(r/d − 1, 0)
F_sep = Σ_j w(d_j) · r̂_ji
```

Weight is proportional to `1/d` minus a constant offset.  At `d = r`,
weight = 0 (hard cutoff).  As `d → 0`, weight → ∞ (strong repulsion
for very close neighbours).  Neighbours beyond `r` contribute nothing.

- Parameters: `radius` (cutoff distance)
- Key property: hard cutoff, diverges at zero distance

---

### A7. Velocity-Weighted

```
closing_speed_j = (v_i − v_j) · (−r̂_ji)   // positive = approaching
w(d, closing_speed) = max(closing_speed, 0) / d²
F_sep = Σ_j w(d_j, s_j) · r̂_ji
```

Since `r̂_ji` points from neighbour `j` toward bird `i`, the vector
`−r̂_ji` points from `i` toward `j` — the direction along which
closing is measured.  Positive closing speed means the distance
between bird and neighbour is decreasing.

The **only** kernel that depends on runtime dynamics, not just static
geometry.  Receding neighbours contribute **no separation force**
regardless of distance.  Approaching neighbours push proportionally
to how fast they're closing.  The `1/d²` base is the same as the
`sum` kernel, further scaled by closing speed.

- Parameters: `closing_speed` (per-neighbour scalar, computed at runtime)
- Key property: the only runtime-dynamic kernel; approaching birds
  repel, receding ones are ignored

---

### A8. Cosine Zone

```
cos_θ = r̂_ij · v̂_i                  (cosine of bearing from bird's heading)
w(d, θ) = (1 + cos_θ) / (2 · d²)
F_sep = Σ_j w(d_j, θ_j) · r̂_ji
```

A continuous weighting based on the angle between the bird's heading
and the bearing to each neighbour.  Neighbours directly ahead
(cos_θ = 1) get full weight; neighbours to the side (cos_θ = 0) get
half weight; neighbours behind (cos_θ → −1) get near-zero weight.

This is a **soft** version of the hard FOV cone cutoff — instead of
excluding neighbours outside the cone entirely, it smoothly attenuates
them.

- Parameters: `heading` (per-bird 3D unit vector)
- Key property: heading-dependent; smooth angular falloff, not a hard
  cone

---

### A9. Linear

```
F_sep = Σ_j r̂_ji / d_j
```

1/d falloff instead of 1/d² — nearer neighbours still push harder,
but the decay is shallower.  At d = 2, weight = 0.5; at d = 10,
weight = 0.1.

- Parameters: none
- Key property: shallow distance falloff (1/d vs 1/d²)

---

### A10. Nearest Only

```
F_sep = r̂_nearest
```

Only the single closest neighbour contributes — at full unit strength,
regardless of how many other neighbours exist or how close they are.
If multiple neighbours are at the exact same minimum distance (a tie),
all of them contribute (intentionally simple tie-breaking).

- Parameters: none
- Key property: hardest cutoff — exactly one (or tied) neighbour matters

---

### A11. Bell Zone

```
t = clip( |d − r| / w, 0, 1 )
bell(t) = cos(π · t) / 2 + 0.5
F_sep = Σ_j bell(d_j) · r̂_ji
```

A cosine-bell weighting centred at distance `r` with half-width `w`.
The weight is:

- 1.0 at `d = r` (zone centre — maximum repulsion)
- 0.5 at `d = r ± w/2`
- 0.0 at `d ≤ r − w` and `d ≥ r + w` (hard cutoff on both sides)

This is the only kernel where **nearer is not always stronger**.
Neighbours closer than `r − w` contribute zero force — there is a
preferred separation zone.

- Parameters: `radius` (zone centre), `zone_width` (half-width)
- Key property: non-monotonic in distance — the only kernel where
  being too close reduces repulsion

---

### Separation Parameter Requirements

| Kernel | needs_radius | needs_zone_width | needs_closing_speed | needs_heading |
|--------|:---:|:---:|:---:|:---:|
| sum | | | | |
| mean | | | | |
| unit | | | | |
| exp | ✓ | | | |
| linear_ramp | ✓ | | | |
| asymptotic | ✓ | | | |
| velocity_weighted | | | ✓ | |
| cosine_zone | | | | ✓ |
| linear | | | | |
| nearest_only | | | | |
| bell_zone | ✓ | ✓ | | |

---

### Separation: Choosing a Kernel

| If you want... | Use |
|----------------|-----|
| The classic Reynolds formula | `sum` |
| Density-invariant behaviour | `mean` |
| Distance-independent repulsion | `unit` |
| Smooth exponential falloff | `exp` |
| A hard distance cutoff with linear decay | `linear_ramp` |
| 1/d falloff with a hard cutoff | `asymptotic` |
| Response to closing speed, not just distance | `velocity_weighted` |
| Heading-dependent soft attenuation | `cosine_zone` |
| Shallow 1/d decay without a cutoff | `linear` |
| Only the closest bird matters | `nearest_only` |
| A preferred separation zone, not just "farther is better" | `bell_zone` |

---

## Part B: Alignment Kernels (4)

Alignment steers each bird toward the mean velocity of its neighbours.
All alignment kernels receive precomputed per-neighbour arrays plus
`neighbor_vel` (neighbour velocities, shape `(..., k, 3)`).  Each
returns a 3D desired velocity vector.

---

### B1. Unweighted (default)

```
⟨v⟩ = (1 / k) · Σ_{j ∈ neighbors} v_j
```

Plain arithmetic mean of neighbour velocities — every neighbour
contributes equally regardless of distance or direction.

---

### B2. FOV-Weighted

```
cos_θ = r̂_ij · v̂_i                    (cosine of bearing)
weight(θ) = clip( (cos_θ − fov_min) / (1 − fov_min), 0, 1 )
⟨v⟩ = Σ_j weight(θ_j) · v_j / Σ_j weight(θ_j)
```

Neighbours are weighted by how close they are to the bird's forward
direction.  At `cos_θ = 1` (dead ahead), weight = 1.  At
`cos_θ = fov_min` (edge of the FOV cone), weight = 0.  Neighbours
behind the cone contribute nothing.

The `fov_min` parameter is typically the existing cosine-threshold
alignment cone.  When `fov_min = −1` (full sphere), all neighbours
get weight = 1, reducing to the unweighted kernel.

This is the biologically-motivated kernel: a bird's field of view
is not omnidirectional, and birds ahead matter more for alignment
than birds behind.

- Parameters: `heading` (bird's unit velocity), `fov_min` (cosine
  threshold, typically −1 to 1)

---

### B3. Spherical Mean (3D)

```
v̂_j = v_j / ‖v_j‖                                (neighbour unit heading)
R = Σ_j v̂_j                                       (resultant vector, 3D)
d̄ = R / ‖R‖                                       (mean resultant direction)
mean_speed = (1/k) · Σ_j ‖v_j‖
⟨v⟩ = mean_speed · d̄
```

The true 3D generalization of a circular mean — the *mean resultant
direction* from directional statistics on the sphere S² (a circular
mean is the S¹-restricted special case of this same construction,
`atan2(Σsinθ, Σcosθ)` being algebraically the angle of
`normalize(Σ unit_2d(v))`). Every axis is treated identically: there
is no XY/Z split, no special-cased "ground plane." Neighbours with
near-zero speed (`‖v_j‖ ≤ 1e-6`) are excluded from both the resultant
sum and the speed average, since their heading is undefined. The
result is scaled by the mean neighbour speed so it's a real velocity,
not a bare unit heading.

- Parameters: `neighbor_vel` (standard), no extra params
- Key property: wraparound-aware for angular data in all three axes
  simultaneously; the correct choice for flocks with genuine vertical
  structure, where a 2D-projected circular mean would discard climb/
  dive information

---

### B4. Bell Zone

```
t = clip( |d − r| / w, 0, 1 )
bell(t) = cos(π · t) / 2 + 0.5
⟨v⟩ = Σ_j bell(d_j) · v_j / Σ_j bell(d_j)
```

The same cosine-bell weighting used for separation and cohesion,
applied to alignment.  Neighbours at the configured zone centre `r`
dominate the alignment direction; neighbours nearer OR farther than
the zone both weight toward zero.

- Parameters: `radius` (zone centre), `zone_width` (half-width)
- Key property: non-monotonic in distance; preferred alignment zone

---

## Part C: Cohesion Kernels (3)

Cohesion steers each bird toward the centre of mass of its neighbours.
The force is the vector from the bird's position to the (possibly
weighted) neighbour centre, clamped to unit length.

---

### C1. Unweighted (default)

```
⟨p⟩ = (1 / k) · Σ_{j ∈ neighbors} p_j
F_coh = clamp(⟨p⟩ − p_i, 1)
```

Plain mean of neighbour positions — every neighbour pulls equally
regardless of distance.

---

### C2. Inverse Distance

```
weight(d) = 1 / d
⟨p⟩ = Σ_j weight(d_j) · p_j / Σ_j weight(d_j)
F_coh = clamp(⟨p⟩ − p_i, 1)
```

Neighbours are weighted by `1/d` — nearer neighbours pull more
strongly toward the weighted centre.  The weight is applied to the
**unit direction** from the centre, not to the raw position
difference, to avoid a cancellation trap: weighting raw diffs by
`1/d` would cancel exactly since `diffs` already has magnitude `d`.

- Parameters: none (standard `dists` and `close` arrays)
- Key property: nearer neighbours pull harder

---

### C3. Bell Zone

```
t = clip( |d − r| / w, 0, 1 )
bell(t) = cos(π · t) / 2 + 0.5
⟨p⟩ = Σ_j bell(d_j) · p_j / Σ_j bell(d_j)
F_coh = clamp(⟨p⟩ − p_i, 1)
```

The same cosine-bell weighting applied to cohesion.  Neighbours at
the zone centre dominate the weighted centre; neighbours nearer or
farther both weight toward zero.  This produces a preferred cohesion
distance.

- Parameters: `radius` (zone centre), `zone_width` (half-width)
- Key property: non-monotonic; preferred cohesion distance

---

## Part D: Summary

### Alignment Kernels

| Kernel | Formula | Extra Params | Key Property |
|--------|---------|:---:|---|
| unweighted | Arithmetic mean of v_j | — | All neighbours equal |
| fov_weighted | FOV-cosine-weighted mean | heading, fov_min | Biologically motivated; front-biased |
| spherical_mean | Mean resultant direction of 3D headings | — | Wraparound-aware in all 3 axes |
| bell_zone | Cosine-bell-weighted mean | radius, zone_width | Preferred alignment distance |

### Cohesion Kernels

| Kernel | Formula | Extra Params | Key Property |
|--------|---------|:---:|---|
| unweighted | Arithmetic mean of p_j | — | All neighbours equal |
| inverse_distance | 1/d-weighted centre of mass | — | Nearer neighbours pull harder |
| bell_zone | Cosine-bell-weighted centre of mass | radius, zone_width | Preferred cohesion distance |

### Why 11 Separation Kernels but Only 4 Alignment / 3 Cohesion?

Separation is the hardest problem in flocking: it must prevent
collisions in dense regions without pushing birds so far apart that
cohesion fails.  The 11-kernel taxonomy reflects the many different
ways to balance distance-dependent repulsion — hard cutoffs, soft
falloff, heading-dependence, closing-speed-dependence, and
non-monotonic zone preferences.

Alignment is simpler: average neighbour velocities.  The variation
comes from *which* neighbours to weight more (front-biased, distance-
zoned) or *how* to average (linear vs. circular).  Cohesion is
simpler still: average neighbour positions with optional distance
weighting.

The asymmetry (11 / 4 / 3) reflects the genuine asymmetry in the
underlying problem — not a gap in the implementation.

---

## Part E: Taxonomy

These three kernel registries are a "plugin inside a plugin." The
mode-computation layer itself is a family of 7 interchangeable
force-computation strategies — a per-strategy dispatch registry: an
ABC (or shared-signature callable) plus a decorator populating a
lookup table, selected at runtime instead of branching on a hardcoded
if/elif chain. The three kernel registries documented here are a
*sub-component* consumed by one of those 7 strategies (primarily the
Reynolds-style mode), not a strategy family in their own right —
they answer "how does this one force term weight its neighbours,"
not "which force term runs this frame." Each of the three registries
(separation/alignment/cohesion) follows the identical dispatch
pattern independently: its own ABC, its own decorator, its own
lookup table, so a new kernel can be added to any one of the three
without touching the other two or the mode-selection layer above them.

---

## Part F: Beyond pymurmur

Candidate kernel variants seen in the broader flocking-simulation
literature that are not implemented here — framed as candidates, not
verified-correct recommendations:

**Exponential-decay cohesion.** Where separation already has an
exponential-falloff option (`exp(−(d − sepDist)/sepDist)`, strongest
at zero distance and decaying smoothly rather than the hard `1/d`
weighting of `inverse_distance`), cohesion has no equivalent —
only unweighted and inverse-distance centre-of-mass options exist.
An exponential-decay cohesion kernel (`w(d) = exp(−d/cohesion_scale)`)
would let nearby flockmates dominate the cohesion pull smoothly
rather than via a hard `1/d` singularity as `d → 0`.

**Velocity-differential separation for alignment/cohesion.** The
separation registry's closing-speed-weighted kernel (approaching
neighbours push harder, receding ones are ignored) has no analogue
in alignment or cohesion — a "velocity-differential cohesion" kernel
could weight the cohesion pull by how much *faster* a neighbour is
moving away, so a flockmate actively drifting apart gets pulled back
more strongly than one merely far away but holding station. Would
need a `neighbor_vel`-aware cohesion signature, which the current
cohesion-kernel contract (positions only) doesn't carry.

**Distance-override "nearest wins" for alignment/cohesion.** The
separation registry has a `nearest_only` kernel (react only to the
single closest neighbour, ignore the rest); no equivalent exists for
alignment or cohesion. A "nearest-only cohesion" would steer purely
toward the single closest neighbour's position rather than a
group centroid — a qualitatively different (more reactive, less
smooth) social-pull behaviour worth exploring for small-flock or
predator-evasion scenarios where the nearest flockmate matters more
than the group average.
