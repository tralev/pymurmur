# Spatial Mode (Reynolds 1987)

This document defines the Reynolds-boids spatial mode: the
separation/alignment/cohesion triad, neighbour selection strategies,
noise models, speed enforcement policies, perception cones and dual
radii, species dynamics (predator–prey), and the coherence gate.

---

## 1. Velocity Update

The classic Reynolds (1987) boids model computes six independent
force terms — separation, alignment, cohesion (§3), an optional
global flow field (§3.4), an optional predator-escape override
(§8.2), and an optional forward-thrust term (§3.5) — and sums them:

```
a_i = w_sep · F_sep_i + w_align · F_align_i + w_coh · F_coh_i
    + F_flow_i + F_escape_i + F_fwd_i
```

Flow, escape, and forward-thrust are each zero unless explicitly
enabled (`flow_weight > 0`, a threat is active, `w_fwd > 0`
respectively). The net acceleration is then scaled by
`acceleration_scale` (default 0.3) and clamped:

```
a_i *= acceleration_scale
steering_mag = ||a_i||
if steering_mag > max_force: a_i *= max_force / steering_mag
```

Noise (§4) is added **after** this scale-and-clamp step, not before
it — it is deliberately never subject to `max_force`:

```
a_i += F_noise_i
```

After integration, the bird's speed is enforced by the configured
speed model.

---

## 2. Neighbour Selection (Pass 1)

Neighbours for each bird are collected via a spatial index (cKDTree or
SpatialHashGrid) using a k-NN query up to `influence_count` (default
7) candidates within `visual_range` (default 70.0).

Five filter modes exist — exactly one is active per run:

### 2.1 Hybrid (default)

Candidates are collected via k-NN (`influence_count` neighbours).
Then a metric filter is applied: only neighbours within
`visual_range` survive.  If more than `influence_count` survive, only
the closest `influence_count` are kept (topological cap).

```
candidates = kNN(p_i, k=influence_count)      // spatial index query
close = { j : ||p_j − p_i|| < visual_range }   // metric filter
neighbors = close[:influence_count]            // topological cap
```

This combines the best of both worlds: metric filtering ensures
distant birds are ignored in sparse flocks, and the topological cap
prevents over-saturation in dense flocks.

### 2.2 Metric

Only neighbours within `visual_range` are collected.  No topological
cap — all birds within range contribute.

```
neighbors = { j : ||p_j − p_i|| < visual_range }
```

Computational cost grows with density.  Use for small flocks where
every neighbour matters, or for deliberate density-dependent
behaviour studies.

### 2.3 Topological

The closest `influence_count` birds are collected regardless of
distance.  No metric filter — a bird 1000 units away is a neighbour
if it's the 7th closest.

```
neighbors = kNN(p_i, k=influence_count)
```

This is the pure Ballerini et al. (2008) topological interaction
model: a bird interacts with a fixed number of neighbours independent
of physical distance.  The σ = 6–7 finding from the field is
reproduced by setting `influence_count = 7`.

### 2.4 Global

Every active bird considers every other active bird.  No spatial
index is used for alignment or cohesion — these steer toward the
**whole-flock** mean velocity and centre of mass respectively.
Separation remains local (uses its own neighbour set built from the
spatial index).

```
⟨v⟩ = (1/N_active) · Σ_j v_j
⟨p⟩ = (1/N_active) · Σ_j p_j
F_align_i = (⟨v⟩ − v_i) / ||⟨v⟩ − v_i||         // unit steering toward mean velocity
F_coh_i = clamp(⟨p⟩ − p_i, 1)                    // toward centre, capped at unit distance
```

Computational cost is O(N) per bird (O(N²) total) — only tractable
for small flocks or benchmark comparisons.

### 2.5 None

The k-NN candidates are returned unfiltered — no `visual_range`
distance filter and no further topological capping beyond the k-NN
query's own `influence_count` limit.  Every candidate within the
k-NN result set contributes, even distant ones.

```
neighbors = kNN(p_i, k=influence_count)          // returned as-is, no additional filtering
```

This is a degenerate mode useful for debugging and for comparing the
effect of the hybrid filter against its absence.

---

## 3. Force Primitive Formulas

### 3.1 Separation Force

Separation steers each bird away from its close neighbours.  A kernel
function `K` maps the per-neighbour geometry (difference vectors,
distances, and optional extra parameters like heading or closing
speed) to a force contribution for each neighbour.  The total
separation force is the sum over all neighbours:

```
F_sep_i = Σ_{j ∈ neighbors} K(diffs_j, dists_j, ...)
```

Different kernels define different force profiles.  The default is
inverse-square repulsion with unit weight:

```
K(diffs_j, dists_j) = r̂_ji / d_ij²
```

where `r̂_ji = (p_i − p_j) / d_ij` is the unit vector from neighbour
`j` toward bird `i`.  Other kernels may use constant unit-direction
repulsion, exponential falloff, linear ramps, heading-dependent
weighting, closing-speed modulation, or non-monotonic bell-zone
profiles centred at a preferred separation distance — each defining
its own complete function without an additional external 1/d² factor.

### 3.2 Alignment Force

Alignment steers each bird toward the mean velocity of its neighbours:

```
F_align_i = (1 / |neighbors|) · Σ_{j ∈ neighbors} v_j − v_i
```

A configurable kernel can replace the unweighted mean with a weighted
average.  Four options exist: unweighted (all neighbours equal),
field-of-view-weighted (neighbours ahead of the bird contribute more),
spherical-mean (the true 3D mean resultant direction over full unit
headings — the S²-generalization of a circular mean, with no XY/Z
split), and bell-zone (neighbours at a preferred distance dominate).
The kernel receives the per-neighbour geometry and neighbour
velocities and returns a desired velocity vector.

### 3.3 Cohesion Force

Cohesion steers each bird toward the centre of mass of its neighbours:

```
F_coh_i = clamp( (1 / |neighbors|) · Σ_{j ∈ neighbors} p_j − p_i, 1 )
```

The `clamp(_, 1)` caps the force magnitude at 1, so cohesion
contributes a unit steering vector pointing toward the local centre,
regardless of distance.  A configurable kernel can replace the
unweighted mean with a weighted centre of mass.  Three options exist:
unweighted (all neighbours equal), inverse-distance (nearer neighbours
pull harder), and bell-zone (neighbours at a preferred distance
dominate).

### 3.4 Flow (Optional Global Field)

When `flow_weight > 0`, every active bird also receives a shared
curl-noise flow contribution, sampled around the flock's own centre
of mass:

```
F_flow_i = curl_flow(p_i, flock_centre, seed_i, t, U) · (flow_weight · 0.22)
```

`curl_flow` is the same shared L0 primitive field mode uses for its
own flow term — a divergence-free, time-varying pseudo-noise field
producing large-scale swirling drift rather than per-bird independent
noise. `U` is the same unit-scale convention used elsewhere
(`field_unit_scale`, or `0.4 · min(W, H, D)` if unset). Disabled by
default (`flow_weight = 0`).

### 3.5 Forward Thrust (Optional)

When `w_fwd > 0` (`spatial.w_fwd`, default 0), a bird also receives a
cruise-speed-seeking force whose sign flips around the target speed:

```
F_fwd_i = w_fwd · (v0 − ||v_i||) · v̂_i
```

Birds slower than `v0` are pushed forward (in their own current
heading); birds faster than `v0` are pushed backward, decelerating
them. Stationary birds (`||v_i|| ≈ 0`) receive no forward-thrust
contribution, since there is no `v̂_i` to push along.

---

## 4. Noise Models

Noise is added after the three force primitives.  Five strategies
exist:

### 4.1 Additive (default)

Standard Reynolds noise: a random vector of configurable magnitude
`noise_scale` (default 0.0 — off):

```
F_noise_i = noise_scale · η_i      where  η_i ~ Uniform(S²)
```

η is drawn per bird from a uniform distribution on the unit sphere
(via 3D Gaussian normalisation).

### 4.2 Maxwellian

A velocity perturbation rather than an acceleration perturbation.
Despite the name (inherited from the codebase's own convention),
this uses the **same** uniform-direction, fixed-magnitude generator
as additive noise (§4.1) — not an independent per-axis Gaussian:

```
η_i = noise_force(1, 1.0, rng)      // unit magnitude, uniform direction on S²
v_i += noise_scale · 0.1 · η_i
```

The two noise models differ only in (a) where the result is applied
(acceleration for additive, velocity for Maxwellian) and (b) the
effective scaling factor — not in the underlying distribution. This
is added to velocity *after* the acceleration step and *before* the
speed clamp, matching spec pipeline order (noise → speed enforcement,
not noise → acceleration).

### 4.3 None

No noise is applied.  Forces are entirely deterministic from the
neighbour configuration.

### 4.4 Seed-Sinusoidal

Deterministic per-bird sinusoidal noise driven by 3D value noise
sampled at each bird's identity (its seed index) and frame time:

```
F_noise_i = seed_noise3(seed_i, frame_t) · (noise_scale / 0.18)
```

`seed_noise3` is a coherent 3D noise function (sinusoidal basis)
producing output in [−0.18, +0.18] per axis.  The scaling factor
maps `noise_scale` onto the same effective range as additive noise.
This is deterministic given (seed, t) — independent of the RNG call
order elsewhere in the pipeline.

### 4.5 Velocity

A velocity-domain perturbation (not an acceleration): a cubic-shaped
random vector is added directly to velocity after integration:

```
u ~ Uniform³([0, 1])
noise_i = (u³ − 0.5) · noise_scale
v_i += noise_i
```

The cubic shape concentrates noise near zero while retaining an
occasional large perturbation.  This is stashed on the config for
the integrator to consume after the acceleration step, before the
speed clamp.

---

## 5. Speed Enforcement

After the acceleration step (and any velocity-domain noise), the
bird's speed is enforced by one of six policies. The four below
(band/clamp, fixed, ceiling, none) are the ones spatial mode
typically runs with; two further policies exist in the shared
speed-enforcement registry — noise-modulated (the speed cap is
continuously modulated by a deterministic position-sampled noise
field) and velocity-adaptive (speed exponentially lerps toward a
randomized-bonus target rather than being hard-clamped) — and are
equally available to spatial mode via `speed_mode`, though rarely
used with it. The policy is applied to all active birds with per-bird
speed caps `caps[i]` (configurable; default `v0`):

### 5.1 Band / Clamp (default)

Speed is clamped to the band [min_speed, caps]:

```
if ||v_i|| > caps[i]:   v_i *= caps[i] / ||v_i||
if ||v_i|| < min_speed: v_i *= min_speed / ||v_i||
```

where `min_speed = v0 · speed_min_factor` (default 0.3).  Both
`band` and `clamp` are aliases for the same strategy.

### 5.2 Fixed

Speed is exactly renormalised to `caps[i]`:

```
v_i = caps[i] · v_i / ||v_i||
```

Zero-velocity birds get a deterministic fallback direction (1, 0, 0)
to avoid NaN.  Used by modes requiring constant speed.

### 5.3 Ceiling

Speeds above `caps[i]` are scaled down; speeds below are left
unchanged:

```
if ||v_i|| > caps[i]:   v_i *= caps[i] / ||v_i||
```

No lower bound — birds can drift to arbitrarily slow speeds.

### 5.4 None

No speed enforcement.  Velocities pass through unchanged.  Used by
modes where speed control is handled externally.

---

## 6. Perception Cones and Dual Radii

### 6.1 Per-Interaction Perception Cones

Each force primitive (separation, alignment, cohesion) can be gated
by a perception cone and a maximum interaction distance:

- `max_dist_sep`, `max_dist_align`, `max_dist_coh`: maximum distance
  (0 = disabled) for the corresponding force.
- `angle_sep`, `angle_align`, `angle_coh`: cosine of the half-angle of
  the perception cone (−1 = full sphere, no filtering).

A neighbour `j` survives the perception filter for force `F` iff:

```
||p_j − p_i|| ≤ max_dist_F     (if max_dist_F > 0)
AND  cos⁻¹(r̂_ij · v̂_i) ≤ acos(angle_F)    (if angle_F > −1)
```

Neighbours outside the cone or beyond the distance limit are excluded
from that force primitive but remain available for the others.

### 6.2 Dual Radii

Separation and alignment can use different effective radii:

- `separation_distance` (default 0 = off): an absolute metric gate
  for separation neighbours — only birds within this distance
  contribute to separation.
- `alignment_radius_ratio` (default 1.0): scales `visual_range` for a
  tighter alignment-only subset.  At 0.75, alignment considers
  neighbours within 75% of the full visual range.

When both a distance gate and the alignment radius ratio are active,
the **tighter** of the two is used.

---

## 7. Per-Frame Parameter Jitter

To add organic variation, the separation, alignment, and cohesion
weights can be jittered randomly each frame:

```
w_effective = w · (1 + δ)    where  δ ~ Uniform[−jitter, +jitter]
```

Each of `jitter_separation`, `jitter_cohesion`, `jitter_alignment`
(default 0.0 = off) controls the jitter amplitude for that force.
A jitter of 0.5 means the weight varies ±50% each frame.

The jitter is seeded — the same seed reproduces the same sequence
of jitter values frame-by-frame.

---

## 8. Species Dynamics (Predator–Prey)

When predator boids are present (`n_predators > 0`), two distinct
species interact:

### 8.1 Predator Detection

Each non-predator bird checks its neighbour set for predators:

```
threatened[i] = any( is_predator[neighbors[i]] )
```

When numba is available, this is accelerated via a compiled kernel.
Otherwise a pure-numpy loop is used.

### 8.2 Prey Escape

Threatened prey birds experience a predator escape force that
**replaces** separation and zeroes out alignment and cohesion:

```
F_escape_i = escape_factor · accel_boost · r̂_away / d²
```

where `r̂_away` points away from the nearest predator and
`escape_factor` (default 10⁷) is large enough to dominate all other
forces.  Threated prey also get:

- `predator_speed_boost` (default 1.8×) — multiplicative speed cap
  increase.
- `predator_perception_boost` (default 1.5×) — expanded visual range
  for detecting threats.
- `predator_accel_boost` (default 1.4×) — multiplicative acceleration
  increase.

### 8.3 Predator Behaviour

Predator boids use the same force primitives (separation, alignment,
cohesion) as prey, but with:

- A boosted visual range (`predator_perception_boost` × normal range).
- Autonomous, cursor-following, or orbit threat modes.
- Configurable threat radius, strength, momentum, and split gain.

---

## 9. Coherence Gate

For small flocks, the alignment and cohesion weights are attenuated
by a coherence factor `γ ∈ [0, 1]`:

```
γ = smoothstep(N_active, 0.4 · N_crit, 1.2 · N_crit)
```

where `N_crit` is the critical mass (default 500).  The smoothstep
(Hermite interpolation) produces a smooth transition:

```
t = clamp((x − lo) / (hi − lo), 0, 1)
smoothstep(x, lo, hi) = t² · (3 − 2t)
```

When `γ < 1`:

```
w_align_effective = w_align · γ
w_coh_effective = w_coh · γ
```

Separation is **not** gated — it operates independently of flock
size to prevent collisions regardless of critical mass.

---

## 10. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| separation_weight | 4.5 | Weight of separation force |
| alignment_weight | 0.65 | Weight of alignment force |
| cohesion_weight | 0.75 | Weight of cohesion force |
| noise_scale | 0.0 | Magnitude of noise force |
| noise_mode | "additive" | Noise strategy (additive, maxwellian, none, seed_sinusoidal, velocity) |
| acceleration_scale | 0.3 | Global force scale factor |
| influence_count | 7 | Topological neighbour cap |
| speed_mode | "clamp" | Speed enforcement (band, fixed, ceiling, none, noise_modulated, velocity_adaptive) |
| flow_weight | 0.0 | Global curl-flow field contribution weight (§3.4) |
| spatial.w_fwd | 0.0 | Forward-thrust force weight toward v0 (§3.5) |
| neighbor_filter | "hybrid" | Neighbour selection strategy |
| separation_kernel | "sum" | Kernel for separation force |
| alignment_kernel | "unweighted" | Kernel for alignment force |
| cohesion_kernel | "unweighted" | Kernel for cohesion force |
| max_dist_sep / align / coh | 0.0 | Perception cone max distances |
| angle_sep / align / coh | −1.0 | Perception cone cos(½-angle) |
| separation_distance | 0.0 | Absolute separation gate |
| alignment_radius_ratio | 1.0 | Alignment subset radius |
| jitter_separation / cohesion / alignment | 0.0 | Per-frame weight jitter |
| predator_escape_factor | 10⁷ | Escape force magnitude |
| predator_speed_boost | 1.8 | Threatened speed multiplier |
| predator_perception_boost | 1.5 | Threatened visual range multiplier |
| predator_accel_boost | 1.4 | Threatened acceleration multiplier |
| coherence_factor | 1.0 | Small-flock attenuation |

---

## 11. Taxonomy

Spatial mode is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table,
selected at runtime instead of branching on a hardcoded if/elif chain.
Its 6 siblings: projection (occlusion-geometry-driven boundary-seeking,
no explicit separation/cohesion term), field (target-seeking blob/anchor
compositing with zero neighbour queries), vicsek (constant-speed
angle-coupling alignment with tangent-plane noise), influencer
(tick-driven Lissajous pursuit — the only mode that owns bird positions
directly rather than deriving them from forces), angle (turn-rate-limited
Rodrigues-rotation steering, no force accumulation at all), and marl
(deferred control under an external per-bird policy).

Spatial mode is the most literal Reynolds (1987) implementation of the
seven, and by far the most configurable: an independent, swappable
kernel choice per force primitive (11 separation profiles, 4 alignment
profiles, 3 cohesion profiles), five neighbour-selection filters, and
five noise strategies — a degree of runtime tunability none of the
other six modes expose to this extent.

## 12. Beyond pymurmur: Unimplemented Extensions

- **GPU compute-shader force evaluation.** Some flocking implementations
  run the entire separation/alignment/cohesion force loop as a GPU
  compute shader operating on all boids in parallel, with either a plain
  O(n²) all-pairs kernel or a parallel prefix-sum reduction for global
  aggregation. Spatial mode's force pass stays CPU-resident (numpy, with
  optional numba JIT) — there is no compute-shader path. Adding one
  would mean porting the kernel functions to a shader language and
  managing GPU-resident position/velocity buffers across frames.
- **Roosting / fixed-site attraction.** A gentle, decoupled
  vertical+horizontal pull toward a fixed "roost" point, separate from
  both the coherence gate (which only attenuates existing weights) and
  the Wander extension (a bounded random-travel attractor with no fixed
  destination). Would need a new opt-in force term or extension with its
  own target point and independent vertical/horizontal gains.
- **Parallel prefix-sum global reduction.** Spatial mode's "Global"
  neighbour-selection filter (§2.4) already computes a whole-flock mean
  velocity and centre of mass every frame, but does so as an O(N)
  numpy reduction. A GPU parallel-reduction implementation would make
  this O(log N) per step, though the resulting behaviour (every bird
  steering toward the same global mean) would be unchanged — only the
  computation's cost profile would differ.
