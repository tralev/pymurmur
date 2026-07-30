# Noise Models and Speed Policies

This document defines all noise models (5 strategies for injecting
randomness into the simulation) and all speed enforcement policies
(6 policies for post-integration speed control, plus the
deficit-based speed laws used by angle mode and the
neighbour-adaptive speed extension).

---

## Part A: Noise Models

Noise is injected after the three force primitives (separation,
alignment, cohesion) to prevent the flock from converging to a static
equilibrium.  Four of the five noise models (additive, maxwellian,
none, and seed-sinusoidal) share a single generation function
`noise_force(n, scale, rng)`.  The velocity model uses its own
cubic-shaped distribution (see §A5).  The shared function is:

```
g_i ~ N(0, scale) in each of 3 axes    // isotropic 3D Gaussian
η_i = (g_i / ||g_i||) · scale          // normalise to unit, then scale
```

This produces vectors of uniform magnitude `scale` with uniform
direction on the unit sphere S².  The distribution is isotropic — no
directional bias.  For `scale = 0`, an all-zero array is returned.

Five strategies exist.  Exactly one is active per run.

### A1. Additive (Default)

A random acceleration vector added to the net steering force:

```
a_noise_i = noise_force(1, noise_scale, rng)
```

The `noise_scale` parameter (default 0.0 — off) controls the
magnitude.  Each bird gets an independent draw per frame from the
seeded RNG.  The uniform-on-sphere direction combined with fixed
magnitude ensures no directional bias and a consistent perturbation
size across birds.

### A2. Maxwellian

A velocity perturbation applied **after** the acceleration step and
**before** the speed clamp.  Generates unit-scale noise vectors, then
scales them down:

```
η_i = noise_force(1, 1.0, rng)            // unit magnitude, uniform direction
v_i += noise_scale · 0.1 · η_i
```

The `0.1` factor scales the unit-magnitude perturbation to a
physically reasonable size for a velocity update.  Despite the name
"Maxwellian" (inherited from the codebase convention), the underlying
distribution is uniform-direction, fixed-magnitude — the same
`noise_force` function as additive noise.  The two models differ only
in (a) where the result is applied (acceleration vs velocity) and
(b) the effective scaling factor.

### A3. None

No noise is applied.  Forces are entirely deterministic from the
neighbour configuration — the flock evolves solely under the
separation, alignment, and cohesion forces with no stochastic
component.  `noise_force` is never called.

Useful for:
- Reproducibility studies (identical trajectories from the same seed).
- Comparing noise effects by toggling between "none" and "additive"
  at the same seed.

### A4. Seed-Sinusoidal

Deterministic per-bird sinusoidal noise driven by 3D value noise
sampled at each bird's identity (its frame-invariant seed) and the
current frame time:

```
a_noise_i = seed_noise3(seed_i, frame_t) · (noise_scale / 0.18)
```

where `seed_noise3` is a coherent 3D noise function based on
sinusoidal basis functions, producing output in [−0.18, +0.18]
per axis.  The `noise_scale / 0.18` factor maps the user-specified
`noise_scale` onto the same effective magnitude range as additive
noise.

Key property: for a given `(seed_i, frame_t)`, the noise is always
the **same** — independent of the RNG call order elsewhere in the
pipeline.  Two runs with the same seed at the same frame produce
byte-identical noise vectors regardless of how many other RNG calls
intervened.  This makes seed-sinusoidal noise useful for debugging
and for comparing force-mode variants: the noise contribution is
locked to the RNG state, not the RNG call order.

Caveat: `seed_i` is not a persistent per-bird identity — it is
generated as `arange(len(active_idx))` each frame, i.e. a bird's
**rank among currently-active birds**, not a fixed ID. This is stable
frame-to-frame only while the active-bird set itself doesn't change;
if birds are added, removed, or reordered between frames, a given
physical bird's `seed_i` (and hence its noise sequence) can shift.

### A5. Velocity

A velocity-domain perturbation using a cubic-shaped random vector
added directly to velocity after integration, before the speed clamp:

```
u ~ Uniform³([0, 1])                     // 3 independent uniform draws
noise_i = (u³ − 0.5) · noise_scale
v_i += noise_i
```

The cubic shape `u³` (rather than linear `u`) concentrates the
distribution near zero while retaining an occasional large
perturbation.  The shift by `−0.5` centres the distribution:
- Most noise values are near −0.5 (slightly negative).
- Occasional values near +0.5 (positive).
- Very rare values reach ±0.5 (the tails are suppressed by the
  cubic).

This is stashed on the config for the integrator to consume after the
acceleration step.  Like Maxwellian noise, it is a velocity-domain
perturbation — not an acceleration — but unlike Maxwellian it uses a
custom cubic-shaped distribution rather than the shared `noise_force`.

---

## Part B: Speed Enforcement Policies

After the acceleration step and any velocity-domain noise, each bird's
speed is enforced by one of six policies (seven registry entries,
since `clamp` is an alias of `band`).  The policy receives:

- `velocities`: (N, 3) array, mutated in place.
- `caps`: (N,) per-bird maximum speed (default `v0`).
- `min_speed`: (N,) per-bird minimum speed (default `v0 · speed_min_factor`).
- `speeds`: (N, 1) precomputed `||velocities||` per row.

### B1. Band / Clamp (Default)

Speed is clamped to the band [min_speed, caps]:

```
too_fast = (speeds > caps) & active
too_slow = (speeds < min_speed) & active

if any too_fast:
    velocities[too_fast] *= caps[too_fast] / speeds[too_fast]
if any too_slow:
    velocities[too_slow] *= min_speed[too_slow] / (speeds[too_slow] + 10⁻¹⁰)
```

Both `band` and `clamp` are registered aliases for the same strategy.
The `speed_min_factor` (default 0.3) sets the floor relative to `v0`:

```
min_speed = v0 · speed_min_factor
```

This is the standard variable-speed model — birds can fly at any
speed between the floor and their individual cap.

### B2. Fixed

Speed is exactly renormalised to the cap:

```
safe_speeds = speeds + 10⁻¹⁰                 // avoid division by zero
dirs = velocities / safe_speeds

zero_mask = (speeds < 10⁻⁶) & active
if any zero_mask:
    dirs[zero_mask] = (1, 0, 0)              // deterministic fallback

velocities[active] = dirs[active] * caps[active]
```

Zero-velocity birds get a deterministic direction (1, 0, 0) to avoid
NaN.  This enforces constant-speed motion — birds always fly at
exactly their cap speed.

### B3. Ceiling

Only the upper bound is enforced — birds can drift to arbitrarily
slow speeds:

```
too_fast = (speeds > caps) & active
if any too_fast:
    velocities[too_fast] *= caps[too_fast] / speeds[too_fast]
```

No lower bound.  Birds that receive little force naturally slow down,
producing more realistic "loitering" behaviour.

### B4. None

No speed enforcement — velocities pass through unchanged:

```
// no operation
```

Used by modes where speed control is handled externally (e.g. MARL
mode, where the RL policy owns velocity control end-to-end on a
different unit scale).

### B5. Noise-Modulated

The speed *cap* (not the enforced speed itself) is continuously
modulated by a deterministic 3D value-noise field sampled at each
bird's position, producing organic slow/fast zones without any
behavioural logic:

```
q = position · frequency                          // frequency = 0.01 cycles/unit
noise = sin(q.x·1.3 + cos(q.y·1.7 + 0.5)) · cos(q.z·1.1 + sin(q.x·0.9 + 1.7))
noise_01 = (clip(noise, −1, 1) + 1) / 2
mult = 0.5 + 1.5 · noise_01³
caps_i = base_caps_i · mult
```

Not literal simplex noise — the same deterministic sinusoidal field
construction the physics core's `curl_flow` primitive uses for its own
pseudo-noise. Only the cap is modulated; `min_speed` is untouched, so
slow zones still respect the mode's normal floor. If no per-bird
position is available to the strategy, it degrades to Band/Clamp (B1)
rather than silently no-opping.

### B6. Velocity-Adaptive

Speed smoothly approaches a randomized-bonus target via exponential
lerp at a fixed rate, rather than a hard per-frame clamp:

```
bonus ~ Uniform(0.85, 1.15)                   // re-rolled every frame, per bird
goal = caps · bonus
rate = clamp(3.0 · dt, 0, 1)                  // fixed lerp rate, 3.0 s⁻¹
v_new = lerp(v, direction(v) · goal, rate)
```

A simplified form of a taxonomy strategy that (in its full form) keys
the target speed on an external per-bird behavioural state
(normal/emergency acceleration); this policy has no channel for that
state, so it implements only the randomized-bonus + smooth-approach
mechanism at one fixed rate.

---

## Part C: Deficit-Based Speed Laws

The angle mode and the neighbour-adaptive speed extension use a
fundamentally different approach from the band/clamp/fixed/ceiling
policies: rather than enforcing speed after the fact, they modulate
the **speed cap** based on the **neighbour count deficit** — how
many fewer neighbours a bird has than the target count.  Isolated
birds get a higher speed cap; birds in dense clusters get a lower
one.

The core function is `adaptive_speed_bonus(deficit, mode, deficit_cap, linear_scale)`:

```
deficit = target_neighbor_count − actual_neighbor_count
positive = max(deficit, 0)
```

Three functional forms for the bonus exist:

### C1. Linear (Default)

```
bonus = positive · linear_scale          // uncapped
```

The bonus grows linearly with the deficit.  A bird missing 4
neighbours (target 7, actual 3) with `linear_scale = 5.0` gets
`bonus = 4 · 5 = 20`.  The speed multiplier is then
`1.0 + bonus / v0` — for `v0 = 4.0`, this gives a multiplier of
`1 + 20/4 = 6.0`, so the bird flies at 6× its base speed.

The linear mode is the only one that is uncapped — `linear_scale`
(default 5.0) directly controls sensitivity.

### C2. Quadratic

```
bonus = min(deficit_cap, positive²)      // capped at deficit_cap
```

The bonus grows quadratically but is capped at `deficit_cap`
(default `k_target²`, where `k_target` is the target neighbour
count — typically 4, so the cap is 16).  A bird with deficit 4
gets `bonus = min(16, 16) = 16`, capped.

The quadratic form produces a sharper transition: birds close to
the target get very little bonus (1² = 1), while very isolated
birds quickly reach the cap.

### C3. Softened

```
bonus = min(deficit_cap, positive² / 2.0)   // half of quadratic
```

Same as quadratic but halved.  A bird with deficit 4 gets
`bonus = min(16, 16/2) = 8`.  The softened form produces gentler
speed modulation than quadratic, with a more gradual transition
from the target neighbour count.

---

## Part D: Neighbour-Adaptive Speed Extension

The neighbour-adaptive speed extension (`NeighborAdaptiveSpeed`)
generalises the deficit-based speed law across **all 7 force modes**.
When enabled, it counts each bird's neighbours within a configurable
radius (default: `neighbor_adaptive_speed_radius`, 70.0 units) and
computes the deficit:

```
k = count of neighbours within radius
deficit = k_target − k
bonus = adaptive_speed_bonus(deficit, mode, k_target², linear_scale)
speed_multiplier = 1.0 + bonus / v0
```

The resulting `speed_multiplier` is stored on the flock and applied
by the integrator to scale the per-bird `max_speed` caps.  The three
mode variants (linear, quadratic, softened) use the same
`adaptive_speed_bonus` function from Part C, configured independently
via `neighbor_adaptive_speed_mode` and
`neighbor_adaptive_speed_linear_scale`.

The extension does not affect modes that do not maintain a spatial
index (field, influencer, MARL) — in those modes it no-ops with a
multiplier of 1.0 for every bird.

When both the neighbour-adaptive speed extension **and** angle mode's
deficit-based speed law are active simultaneously (possible by
configuring an angle-mode preset with the extension enabled), both
modulations compose multiplicatively — the extension scales the caps,
and angle mode further scales the cruise speed within those caps.

---

## Part F: Taxonomy

This document covers two architecturally distinct plugin families
that happen to sit next to each other in the per-frame pipeline:

**NoiseStrategy** (5 entries: additive, maxwellian, none,
seed_sinusoidal, velocity) — a per-strategy dispatch registry
dedicated to spatial mode's noise injection, structurally identical
to every other plugin family here: an ABC (or shared-signature
callable) plus a decorator populating a lookup table, chosen at
runtime instead of a hardcoded if/elif chain.

**SpeedModel** (6 distinct strategies, 7 registry entries counting
the `clamp`/`band` alias) — a separate registry dispatched from the
post-integration speed-enforcement step, shared across all 7 force
modes rather than being spatial-mode-specific.

They're documented together because they're pipeline-adjacent, not
because they're the same family: noise is injected during the
acceleration/velocity-composition stage, speed enforcement runs
immediately after integration, and a bird's frame experiences both in
sequence. Neither registry knows the other exists — swapping the
noise strategy has no effect on which speed model runs, and vice
versa.

## Part G: Beyond pymurmur

Candidate noise/speed techniques from the broader flocking-simulation
literature not implemented here — framed as candidates, not
verified-correct recommendations:

**True gradient (Perlin/Simplex) noise fields.** pymurmur's
`seed_sinusoidal` noise and `noise_modulated` speed cap both use a
custom deterministic sinusoidal field (a sum/product of a few sine
and cosine terms at fixed frequencies) rather than a proper coherent
gradient-noise function. A real Simplex or Perlin field would give
smoother, less obviously periodic spatial variation — visible at
close inspection, the current sinusoidal field has an underlying
regular structure a true gradient-noise octave stack would not. Would
require implementing (or depending on) an actual gradient-noise
primitive, a real dependency this codebase currently avoids.

**Persistent per-bird noise seeds.** §A4's caveat (above) already
flags that `seed_sinusoidal`'s per-bird seed is a rank-among-active-
birds index, not a stable identity. A version keyed to a genuinely
persistent per-bird ID (assigned once at spawn, never reindexed) would
make each bird's noise sequence track that specific bird across its
lifetime rather than shifting when the active set changes — useful
for visually distinguishing individual birds' long-run trajectories.

**Two-rate speed response (normal vs. emergency).** Several
implementations in the survey pair a slow, smooth speed-adjustment
rate for routine flocking with a much faster rate triggered by an
external event (a predator sighting, an imminent obstacle). None of
the 6 pymurmur speed models currently branch on an external urgency
signal — `velocity_adaptive` (§B6) always lerps at the same fixed
rate regardless of context. Would need a per-bird urgency channel fed
in from outside the speed-enforcement layer.

---

## Part E: Summary

### Noise Models

| Name | Domain | Effective Magnitude | Distribution | Key Property |
|------|--------|--------------------|--------------|--------------|
| additive | Acceleration | `noise_scale` | Uniform S², fixed magnitude | Default; isotropic random steering |
| maxwellian | Velocity | `noise_scale · 0.1` | Uniform S², fixed magnitude | Velocity-domain (same generator as additive) |
| none | — | 0 | — | Deterministic; no randomness |
| seed_sinusoidal | Acceleration | `noise_scale` | Sinusoidal, [−0.18,+0.18]/axis scaled | Deterministic given (seed, t); RNG-independent |
| velocity | Velocity | `noise_scale` | Cubic-shaped: (U³−0.5) per axis | Custom distribution; concentrates near zero |

### Speed Enforcement Policies

| Name | Formula | Key Property |
|------|---------|--------------|
| band / clamp | Clamp ‖v‖ to [min_speed, caps] | Variable-speed with floor and ceiling (default) |
| fixed | v = caps · v̂ | Constant-speed; exact renormalisation |
| ceiling | Clamp ‖v‖ ≤ caps | No lower bound; birds can drift slowly |
| none | — | No enforcement; external control |
| noise_modulated | `caps · (0.5 + 1.5·noise_01³)` | Cap varies by deterministic position noise |
| velocity_adaptive | `lerp(v, dir(v)·caps·bonus, 3.0·dt)` | Smooth exponential approach, randomized bonus |

### Deficit-Based Speed Law Variants

| Name | Bonus Formula | Cap | Key Property |
|------|-------------|-----|--------------|
| linear | `deficit · linear_scale` | None | Proportional to neighbour deficit |
| quadratic | `min(k_target², deficit²)` | k_target² | Sharp penalty for isolation |
| softened | `min(k_target², deficit² / 2)` | k_target² | Half of quadratic; gentler transition |

The final speed multiplier stored on the flock is:

```
mult = 1.0 + bonus / v0
```
