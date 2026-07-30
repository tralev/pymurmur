# Projection Model (Pearce et al. 2014)

This document defines the projection-based flocking model from
Pearce et al. (2014, PNAS): the velocity update equation, the
boundary-projection direction δ̂, spherical-cap occlusion geometry
and culling, topological neighbour selection, the resulting opacity
observables (Θ, Θ′), steric repulsion, and the coherence gate.

---

## 1. Velocity Update

The velocity of bird `i` at each timestep is a weighted blend of three
components — a boundary-projection term, an alignment term, and a
random walk term (Eq. 3 of the paper):

```
v_i,desired = φp · δ̂_i + φa · ⟨v̂⟩_σ + φn · η̂
```

where:

| Symbol | Meaning | Default |
|--------|---------|---------|
| φp | Projection weight (gradient toward domain boundary) | 0.03 |
| φa | Alignment weight (toward visible neighbours' mean heading) | 0.80 |
| φn | Noise weight (random walk on the unit sphere S²) | 1 − φp − φa |
| δ̂_i | Boundary-length-weighted projection direction (§2) | — |
| ⟨v̂⟩_σ | Mean unit velocity of the σ visible neighbours (§5) | — |
| η̂ | Uniform random unit vector on S² | — |

The unit partition φp + φa + φn = 1 is enforced (S1.4).  When φn is
clamped to max(0, 1 − φp − φa), a non-negative noise term is
guaranteed.

The bird's acceleration is computed via Reynolds steering from its
current velocity:

```
steering = v_i,desired − v_i
steering = clamp(steering, max_force)    // steering magnitude ≤ max_force
a_i += steering
```

The bird's speed is clamped to `v0` after integration (constant-speed
model, matching the paper).  A heading-blend inertia term
`projection_heading_inertia` adds an additional additive pull toward
the bird's own current heading, independent of the φp + φa + φn
partition.

---

## 2. Boundary Projection δ̂

δ̂ is a unit vector pointing toward the **least-occluded** region of
the domain boundary — the direction where the fraction of visible
neighbours is smallest, weighted by boundary arc length. This is a
boundary-length-weighted construction *inspired by* the paper's own
Eq. 1, not identical to it: the paper's Eq. 1 averages, unweighted,
over the directions of detected light–dark boundary edges on the
observer's retina (silhouette transition curves); this implementation
instead computes a sin(angular-radius)-weighted average over visible
neighbours' centre directions (below) — a documented approximation of
the same idea, not a literal transcription of the equation.

For each bird, the set of visible neighbours is determined via
spherical-cap occlusion (§3).  Each visible neighbour `j` at
direction d̂_j with angular radius α_j contributes an arc length
sin α_j to the boundary projection:

```
δ̂_i = Σ_{j ∈ visible} (sin α_j · d̂_j) / Σ_{j ∈ visible} sin α_j
```

where `d̂_j = (p_j − p_i) / ||p_j − p_i||` is the unit direction from
observer `i` to visible neighbour `j`.

|δ̂_i| ∈ [0, 1]:
- |δ̂_i| ≈ 1: bird on the flock boundary — all visible neighbours
  cluster in one direction, so their weighted directions sum to a
  strong outward vector.
- |δ̂_i| ≈ 0: bird surrounded — visible neighbours are evenly
  distributed, so the weighted directions cancel.

The δ̂ vector points away from the flock interior toward open sky,
giving each bird a gradient descent toward the edge.  This replaces
the explicit cohesion and separation terms in the original Reynolds
model — the φp term alone produces emergent cohesive behaviour
because birds on the periphery steer inward toward the centre (the
least-occluded region is always away from the flock).

---

## 3. Spherical-Cap Occlusion

The core geometric primitive: each bird subtends a solid angle on the
observer's visual sphere, modelled as a spherical cap whose angular
radius depends on the bird's body size, its distance from the
observer, and its orientation relative to the line of sight.

### 3.1 Body Radius

Each bird has a base body radius `b` (configurable; default 9.0 in
display units).  For an isotropic bird the effective radius is simply
`b_eff = b`.  For an anisotropic bird the effective radius depends on
the viewing angle ψ — the angle between the neighbour's velocity
direction (its long body axis) and the line of sight from the
observer:

```
b_eff = sqrt( (b · a · sin ψ)² + (b · cos ψ)² )
```

where `a` is the body axis ratio (anisotropy parameter, default 2.0) —
a prolate spheroid's long-axis elongation. When `a = 1` this reduces
to `b_eff = b` regardless of viewing angle. When `a > 1`, a bird seen
from the side (ψ ≈ π/2, sin ψ ≈ 1) appears wider — its full elongated
body is visible, `b_eff → b · a`; a bird seen head-on (ψ ≈ 0, cos ψ ≈ 1)
appears unchanged — only its cross-section is visible, `b_eff → b`.

The cosine of ψ is the absolute dot product of the unit sight
direction r̂ and the unit velocity direction v̂:

```
cos ψ = |r̂ · v̂|
sin ψ = sqrt(max(0, 1 − cos² ψ))
```

### 3.2 Angular Radius

For a bird at distance `d` with effective radius `b_eff`, the exact
angular radius α of the spherical cap it subtends is (P1.4 — Pearce
et al. 2014, exact asin formulation):

```
α = asin(min(b_eff / d, 1))
```

When `b_eff / d ≥ 1` (bird very close), α = π/2, meaning the cap
covers a full hemisphere.  The angular radius is always in [0, π/2].

### 3.3 Solid Angle

The solid angle Ω subtended by a spherical cap of angular radius α
is:

```
Ω = 2π · (1 − cos α)
```

This is the **exact** solid angle for a spherical cap — without the
small-angle approximation `Ω ≈ π · α²` used in earlier work.  The
fraction of the full visual sphere (4π steradians) occluded by one
bird is `Ω / (4π) = (1 − cos α) / 2`.

### 3.4 Blind Angle

Birds cannot see behind themselves.  A rear blind cone is defined by
`blind_deg` (configurable; default 60°).  The half-angle in radians
is `blind_deg / 2`, and the cosine threshold is:

```
cos(blind_half) = cos(radians(blind_deg / 2))
```

A neighbour whose direction d̂ relative to the observer satisfies
`−d̂ · v̂_observer ≥ cos(blind_half)` — i.e. it lies inside the rear
cone — is excluded from occlusion processing entirely.  It contributes
nothing to δ̂ or Θ.

When `blind_deg = 0`, the blind cone is disabled and all neighbours
are considered.

### 3.5 Occlusion Culling (Closest-First)

Visible neighbours are determined by closest-first occlusion culling
(P1.1 — true 3D spherical-cap occlusion, Pearce et al. 2014):

1. Sort all candidates by distance `d`, closest first.
2. For each candidate `j` in order:
   - Compute its direction `d̂_j = (p_j − p_obs) / d_j` and angular
     radius `α_j` (via §3.2).
   - Skip if inside the blind cone (§3.4).
   - For each already-visible neighbour `k`:
     - If `d̂_j · d̂_k ≥ cos α_k`, neighbour `j` is **occluded** —
       its entire cap lies within the cap of the closer neighbour `k`.
     - This is a conservative test: the `≥` operator is exact for
       the special case where one cap is fully inside another,
       which is the only case the Pearce model's δ̂ formulation
       handles.
3. If not occluded by any already-visible neighbour, `j` is visible.

The occlusion test is O(n_vis × M) per observer in the sequential
path, and parallelised across observers in the batched path (when
the number of active birds exceeds a threshold, default 100).

---

## 4. Topological Neighbour Selection

Neighbours are selected **topologically**: for each bird, the σ
nearest neighbours within `visual_range` are collected via a spatial
index (k-NN query).  σ is the topological neighbour count (default 4;
Ballerini et al. 2008 find 6–7 in the field).

The spatial index used is a `cKDTree` for projection mode (toroidal
distance when the domain wraps).  Neighbours beyond σ are ignored
entirely — this is the defining topological property: a bird 50 metres
from its 7th neighbour behaves identically to a bird 5 metres from its
7th neighbour, regardless of absolute density.

Birds with fewer than σ neighbours (e.g. on the flock edge) use
whatever neighbours are available.  Birds with zero neighbours
contribute no alignment or occlusion — they drift on noise alone.

---

## 5. Alignment from Visible Neighbours

After occlusion culling, the visible subset of the σ topological
neighbours contributes to alignment:

```
⟨v̂⟩_σ = (1 / n_visible) · Σ_{j ∈ visible} v̂_j
```

Only neighbours that survive both (a) the topological σ cutoff and
(b) the occlusion culling contribute.  Invisible neighbours (occluded
by closer birds) are excluded entirely.

If `n_visible = 0` (bird sees no neighbours — rare at the flock edge),
alignment contributes zero force.

---

## 6. Opacity Observables

The spherical-cap occlusion geometry produces three opacity metrics.

### 6.1 Internal Opacity Θ

Internal opacity Θ measures the fraction of the visual sphere blocked
by visible flock-mates — the probability that a random ray from the
observer's eye hits a bird rather than sky.

Θ is computed as the **probabilistic union** of solid angles from all
visible neighbours (P1.2):

```
Θ = 1 − ∏_{j ∈ visible} (1 − Ω_j / (4π))
```

where `Ω_j = 2π · (1 − cos α_j)` from §3.3.  The product form assumes
independent probabilities per neighbour — a good approximation because
the spatial correlation between distant birds' occluding caps is weak.

Θ ∈ [0, 1]:
- Θ = 0: no bird blocks the sky (observer on the flock edge, looking
  outward).
- Θ = 1: the entire visual sphere is blocked (observer deep inside a
  dense, optically thick region).

Pearce et al. (2014) note that Θ, unlike Θ′ below, is *not* tightly
bounded by flock density — it "could be nearly 1 even for very small
densities" from the inside, since a bird surrounded on all sides sees
mostly flock-mates regardless of how sparse the flock is overall.

Θ is computed **per observer, per frame** and averaged across the
flock for reporting.  It is only computed in projection mode — in
all other modes, Θ is NaN.

### 6.2 External Opacity Θ′

External opacity Θ′ (also called 3D voxel opacity) measures the
probability that a ray from **outside** the flock hits a bird —
i.e. how opaque the flock appears to an external observer (a predator
or a camera).

The flock's bounding box is subdivided into a 3D voxel grid.  For
each voxel, the number of birds inside it determines the local
opacity via a probabilistic model:

```
Θ_voxel = 1 − exp(−ρ_voxel · σ_bird)
```

where `ρ_voxel` is the number density in the voxel (birds per unit
volume) and `σ_bird = π · b²` is the cross-sectional area of a bird
(assuming isotropic projection).  The aggregate Θ′ is the
volume-weighted mean across occupied voxels.

This is a **3D extension** of Pearce et al.'s silhouette-based
external opacity — the original model only considered 2D projections
onto the retina. Pearce et al. (2014) report Θ′ ≈ 0.25–0.60 as the
emergent marginal-opacity range in field-observed starling flocks
(their Fig. 3b–e) — this is the empirical range this 3D extension is
calibrated against, not Θ (§6.1), which the paper treats as a
distinct, much-less-bounded quantity.

### 6.3 2D Silhouette Θ′

A faster, approximate 2D version of external opacity: the flock is
projected onto a disk (the silhouette plane) and rasterised into
pixels.  Each pixel's opacity is the fraction of birds whose
projected disks cover that pixel:

```
Θ′_2D = (1 / N_pixels) · Σ_pixels pixel_opacity(p)
```

with

```
pixel_opacity(p) = 1 − ∏_{birds b covering p} (1 − Ω_b / (4π))
```

This is the **disk rasterisation** approach (P9.4) — a 2D
approximation of the full 3D voxel Θ′.  It is cheaper to compute and
matches the Pearce paper's original 2D silhouette formulation.

---

## 7. Steric Repulsion (SI Extensions)

When `refinements` are enabled, a steric repulsion force is added
after the main velocity blend (P1.10 — the Pearce-SI extensions).
For each bird, the repulsion from all neighbours within a distance
threshold `steric_radius` (default 10.0) is:

```
F_steric_i = Σ_{j: d_ij < steric_radius} φ_s · r̂_ji / (d_ij² + ε)
```

where `φ_s` is the steric strength (default 0.6) and `r̂_ji` is the
unit vector from `j` to `i`.  The force is clamped to `max_force`.

The `steric_visible_only` flag (default `false`) restricts steric
repulsion to occlusion-visible neighbours only — neighbours behind
closer birds do not repel.  When `steric_visible_only` is `true`, the
Pearce-SI model is fully honoured: a bird only "feels" birds it can
actually see.

---

## 8. Coherence Gate

For small flocks, the projection and alignment weights are attenuated
by a coherence factor `γ ∈ [0, 1]`, modulated by the flock's size
relative to a critical mass threshold:

```
γ = smoothstep(N_active, 0.4 · N_crit, 1.2 · N_crit)
```

where `N_crit` is the critical mass (default 500, from Goodenough et
al. 2017 — murmurations typically exceed ~500 birds).  The smoothstep
function (Hermite interpolation) produces a smooth transition:

```
t = clamp((x − lo) / (hi − lo), 0, 1)
smoothstep(x, lo, hi) = t² · (3 − 2t)
```

When `γ < 1`:
```
φp_effective = φp · γ
φa_effective = φa · γ
```

Small flocks (<200 birds) get reduced directional pull, allowing them
to explore more freely without the strong boundary-gradient drive that
large flocks need to maintain cohesion.

---

## 9. Summary of Parameters

| Parameter | Symbol | Default | Range |
|-----------|--------|---------|-------|
| Projection weight | φp | 0.03 | [0, 1] |
| Alignment weight | φa | 0.80 | [0, 1] |
| Topological neighbour count | σ | 4 | 1–64 |
| Max visible neighbours | — | 64 | 1–64 |
| Body radius | b | 9.0 | >0 |
| Blind angle | — | 60° | 0–180° |
| Anisotropy | a | 2.0 | ≥1.0 |
| Steric strength | φ_s | 0.6 | ≥0 |
| Steric radius | — | 10.0 | >0 |
| Steric visible-only | — | false | bool |
| Coherence critical mass | N_crit | 500 | >0 |
| Heading inertia | — | 0.0 | [0, 1] |

### Opacity Observables

| Symbol | Name | Formula | Range |
|--------|------|---------|-------|
| Θ | Internal opacity | `1 − ∏(1 − Ω_j/4π)` per observer, flock mean | [0, 1] |
| Θ′ | External (3D voxel) opacity | `mean(1 − exp(−ρ_voxel · σ))` | [0, 1] |
| Θ′₂D | 2D silhouette opacity | Disk rasterisation, fraction of covered pixels | [0, 1] |
| α_j | Angular radius | `asin(min(b_eff / d, 1))` | [0, π/2] |
| Ω_j | Solid angle | `2π · (1 − cos α_j)` | [0, 2π] |
| b_eff | Effective body radius | `sqrt((b·a·sin ψ)² + (b·cos ψ)²)` | [b, b·a] |

---

## 10. Taxonomy

Projection is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table,
selected at runtime instead of branching on a hardcoded if/elif chain.
Every registered strategy in this family receives the same per-frame
inputs (flock state, config, timestep) and produces the same kind of
output (a velocity or acceleration update for every active bird), so
they're freely swappable at startup with no other code changes.

Its six siblings in this family: spatial (classic Reynolds
separation/alignment/cohesion, distance- and topology-filterable),
field (a blob/anchor force compositing many named terms into one
acceleration), vicsek (constant-speed angle-coupling alignment with
tangent-plane noise), influencer (a tick-driven Lissajous target with
rank-based influence weighting), angle (turn-rate-limited heading
rotation with adaptive speed), and marl (defers to an externally
supplied per-bird control signal rather than computing its own force
law).

Projection is the only member of this family whose steering signal is
derived from **occlusion geometry** — what fraction of the visual
sphere is blocked, and in which direction it's least blocked — rather
than a direct average of neighbours' positions or velocities. Every
other strategy in the family computes its steering from raw
neighbour geometry (distances, relative velocities, headings);
projection's δ̂ term is a geometric byproduct of visibility, not a
weighted average of anything.

## 11. Beyond pymurmur: Unimplemented Extensions

A few occlusion- and projection-related techniques from the broader
flocking-simulation literature are not implemented here:

- **Anisotropic occlusion with a full ellipsoid silhouette, not just a
  circular cap.** This document's spherical-cap occlusion already
  varies the cap's *radius* with viewing angle (§3.1's anisotropic
  `b_eff`), but the cap itself is still treated as circular. A bird's
  true silhouette from an oblique angle is an ellipse, not a
  foreshortened circle — modelling the cap as an ellipse (with both a
  major and minor angular radius) would be a more faithful projection
  of a prolate-spheroid body, at the cost of a more expensive
  overlap test between elliptical caps in the occlusion-culling pass.
- **Raycast-based occlusion instead of a closed-form angular-radius
  test.** Some implementations determine visibility by literally
  casting a small bundle of sample rays per neighbour and counting
  hits, rather than computing an exact spherical-cap overlap
  analytically. This trades exactness for generality — it would let
  occlusion account for non-spherical/non-ellipsoidal obstacles (e.g.
  another bird's actual wing-extended silhouette) at the cost of
  sampling noise and per-ray computational overhead that this
  document's closed-form `asin`/solid-angle approach avoids entirely.
- **A soft (probabilistic) blind cone instead of the current hard
  cutoff.** §3.4's rear blind cone is a hard boolean — a neighbour is
  either fully counted or fully excluded based on a fixed angular
  threshold. A soft falloff (visibility weight decreasing smoothly
  from 1 at dead-ahead to 0 at directly-behind, rather than a step
  function at `blind_deg`) would better model a bird's actual
  peripheral vision gradient, though it would also mean every
  neighbour contributes at least a little to δ̂ and Θ regardless of
  direction, changing the model's qualitative behaviour at the
  boundary.
- **Multi-frame occlusion memory.** Currently δ̂ and Θ are recomputed
  from scratch every frame with no temporal smoothing — a neighbour
  that flickers between visible and occluded frame-to-frame (e.g. two
  birds momentarily aligning along a sightline) causes δ̂ to jump
  discontinuously. A short exponential memory over recent visibility
  states per neighbour-pair would smooth this out, trading
  responsiveness for stability.
