# Vicsek Mode

This document defines the Vicsek (1995) constant-speed alignment mode:
each bird moves at fixed speed and aligns its heading with the average
direction of neighbours within a fixed radius, with additive angular
noise and an optional memory term.  Predator-prey species dynamics
overlay the base model with fear-weighted fleeing and nearest-prey
hunting.

---

## 1. Base Model

Every bird moves at constant speed `v0`.  Each timestep, the new heading
is the average of neighbours' headings within radius `r`, perturbed by
tangent-plane noise, blended with a coupling parameter `η`:

```
u_avg    = Σ_{j∈N(i)} u_j / |N(i)|                  // unnormalized mean
u_target = normalize(u_avg)                        // unit direction
u_noisy  = normalize( u_old + √(2·D·dt) · n_⊥ )   // memory + tangent-plane noise
u_new    = normalize( η·u_target + (1−η)·u_noisy )  // blend
v_new    = v0 · u_new
```

where `D` is the diffusion coefficient and `dt` is the time step.

---

## 2. Neighbour Selection

Neighbours are all birds within a fixed Euclidean radius `r`
(`vicsek_radius_influence`).  The selection uses a ball-tree radius
query via `NEIGHBOR_SELECTOR_REGISTRY["ball_tree_radius"]`, producing
a sparse adjacency matrix and per-bird neighbour counts.

Single-bird flocks skip the neighbour query and apply only the noise
term to the old direction.

---

## 3. Memory Term with Tangent-Plane Noise

The memory term projects Gaussian noise onto the tangent plane of
the unit sphere at the bird's current heading, then blends with the
neighbour average:

```
g      ~ N(0, I₃)                               // isotropic Gaussian noise
g_∥    = (g · u_old) · u_old                     // parallel component
n_⊥    = g − g_∥                                 // tangent-plane component
u_noisy = normalize( u_old + √(2·D·dt) · n_⊥ )  // noise-perturbed (on S²)
u_new   = normalize( η·u_target + (1−η)·u_noisy )
```

The tangent-plane projection `n_⊥ = g − (g·u)·u` ensures the noise
moves the heading along the sphere surface rather than pulling it off
the unit sphere.  The noise magnitude is `√(2·D·dt)` — consistent with
the continuous-time Langevin equation for angular diffusion on S².

---

## 4. Coupling Parameter η

The coupling `η ∈ [0, 1]` controls the trade-off between alignment
and memory:

```
η = 1.0 → pure alignment (classic Vicsek — no memory, full neighbour averaging)
η = 0.0 → pure diffusion (random walk on S² — no alignment, pure noise)
η ≈ 0.5 → balanced memory + alignment
```

Birds without neighbours use `η = 0` implicitly — the `u_target` term
vanishes and only the noise term remains.

---

## 5. Heading-Blend Inertia

An optional post-processing step blends the bird's prior heading into
the fully-finalized new direction, independent of the memory term above:

```
u_final = normalize( inertia·u_old + (1−inertia)·u_new )     if inertia > 0
```

Default `inertia = 0.0` (disabled).  When enabled, the bird retains a
fraction of its previous heading regardless of the Vicsek blend, giving
it velocity persistence — the heading changes more smoothly, analogous
to the §09/§11 heading-blend implementations in the comparison survey.

---

## 6. Speed Model

All birds move at constant speed.  Prey use `v0` (`vicsek_velocity`,
default 1.0).  Predators use `v_pred` (`vicsek_velocity_predator`,
default 2.0) — faster than prey.

The speed_mode is `"fixed"`: velocities are set directly to
`direction × speed` each frame, not integrated from acceleration.

---

## 7. Time Step

Vicsek mode uses its own internal time step `dt` (`vicsek_time_step`,
default 0.1), independent of the simulation frame rate.  This
decouples the noise diffusion magnitude from the visual framerate —
the same `dt` produces the same angular diffusion regardless of FPS.

---

## 8. Predator-Prey Species Dynamics

When predator species are configured (via `_is_predator` boolean array),
three additional behaviours overlay the base Vicsek model.

### 8.1 Fear-Weighted Alignment Blending

Prey birds near predators blend their alignment with a flee direction.
Fear is driven by the **mean** distance to all predators within
range, not just the nearest one:

```
R_pred = vicsek_radius_predators
for each prey:
    near_dists = { ‖p_prey − p_pred_k‖ : predator k within R_pred }   // min-image
    if near_dists is empty: skip                       // no fear response
    fear = clamp((R_pred − mean(near_dists)) / R_pred, 0, 1)
    flee_dir = normalize( mean(p_prey − p_pred_k over near predators) )
    u_combined = normalize( (1−fear)·η·u_align + w_afraid·fear·flee_dir + (1−η)·u_noisy )
```

where `w_afraid = vicsek_weight_afraid` (default 3.0) weights the
flee term independently of `fear` — this is a genuine **three-term**
blend, not a simple interpolation: the noise term's `(1−η)` weight is
untouched by fear. Solo prey (no topological neighbours, so no
`u_align`) use a simpler 70%-flee/30%-existing-heading mix instead of
the full three-term blend.

### 8.2 Predator Hunting

Predators pursue the nearest prey within a detection radius scaled
off `R_pred`, with the resulting heading perturbed by hunting noise
rather than pointed exactly at the target:

```
detect_r = vicsek_detect_ratio · vicsek_radius_predators   // default 1.5× R_pred
for each predator:
    nearest_prey = argmin ‖p_pred − p_prey‖ over prey        // min-image
    if nearest_prey within detect_r:
        target = normalize(p_nearest_prey − p_pred)
        u_hunt = normalize(target + vicsek_predator_noise_ratio · η̂)   // η̂ ~ N(0,I₃), normalized
    else:
        u_hunt = random unit vector                     // random walk
```

### 8.3 All-Predator Flock

If no prey are present (all birds are predators), alignment and hunting
are skipped.  All birds perform a pure random walk at `v_pred`:

```
u = normalize( N(0, I₃) )
v = v_pred · u
```

---

## 9. Position Collisions

Collision resolution handles bird-bird overlap with species-aware
behaviour:

**Same-type collisions** (prey-prey or predator-predator): symmetric
push — both birds are displaced equally away from each other:

```
p_i ← p_i + 0.5 × overlap × r̂_{i←j}
p_j ← p_j − 0.5 × overlap × r̂_{i←j}
```

**Prey-predator collisions**: asymmetric — prey are pushed harder than
predators, reflecting the asymmetry of the interaction:

```
p_prey  ← p_prey  + (1 − α) × overlap × r̂_{prey←pred}
p_pred  ← p_pred  − α × overlap × r̂_{prey←pred}         // α < 0.5, so prey moves more
```

**Toroidal seams**: collisions across the wrap boundary use min-image
distance correction so a bird near `x ≈ 0` and one near `x ≈ width`
are treated as neighbours across the seam.

---

## 10. Taxonomy

Vicsek mode is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table,
selected at runtime instead of branching on a hardcoded if/elif chain.
The six sibling strategies: projection (occlusion-geometry-driven
steering), spatial (Reynolds force summation), field (named-term blob
compositing), influencer (scripted tick-driven targets), angle (direct
heading rotation, no force accumulation), and marl (external control).

Within that family, Vicsek mode is the most literal transcription of a
single named published model among the 7 — it implements the 1995
Vicsek self-propelled-particle model nearly directly (constant speed,
neighbour-heading averaging, additive noise), where the other modes
either combine several distinct published ideas (projection blends
Pearce's occlusion geometry with Reynolds-style noise) or implement an
original, non-literature-specific mechanism (field's blob/anchor
compositing, influencer's scripted targeting). It is also the only mode
whose species-overlay logic (§8, fear-weighted fleeing and predator
hunting) lives as inline mode-specific branching rather than being
dispatched through a separate registry — unlike the Threat behavioural
extension, which is a distinct opt-in plugin family available to every
mode, Vicsek's predator-prey handling only exists within Vicsek mode
itself.

## 11. Beyond pymurmur

Techniques from the broader flocking/self-propelled-particle literature
not currently implemented in Vicsek mode:

- **Topological (k-NN) neighbour selection for Vicsek specifically** —
  pymurmur's Vicsek mode uses a fixed metric radius (`r`,
  `vicsek_radius_influence`) for its neighbour set; some Vicsek-family
  implementations instead always average exactly the `k` nearest
  neighbours regardless of distance (a bird at the flock edge still
  averages over `k` birds, even if they're far away). This is the same
  topological-vs-metric distinction pymurmur's own projection mode
  already makes (topological σ), but Vicsek mode itself has no
  topological option — adding one would mean swapping the ball-tree
  radius query for a k-NN query and dropping the radius cutoff.
- **Roosting force** — a gentle, decoupled vertical/horizontal
  center-attraction pull toward a fixed "roost" point, distinct from
  ordinary cohesion: the vertical component pulls toward a target
  altitude with a much weaker gain than the horizontal pull toward a
  target ground position, so a flock settles into a stable altitude
  band rather than just clustering in 3D. Vicsek mode's own heading
  update (§1) has no positional target term at all — everything is
  heading-averaging plus noise, so adding roosting would mean
  introducing a small positional-pull contribution to `u_target`
  alongside the neighbour-heading average, a genuinely new term rather
  than a variant of an existing one.
- **Literal original-Vicsek angular noise** — the 1995 model's noise
  term is a uniform random perturbation to the heading *angle* directly
  (`θ_new = ⟨θ⟩ + η·Uniform(−π, π)`, with `η` a scalar noise amplitude
  in `[0, 1]`), not the continuous tangent-plane Gaussian diffusion this
  implementation uses (§3, a Langevin-style `√(2·D·dt)` term). The two
  are related in spirit (both are heading-randomizing) but produce
  different noise statistics — the original model's noise doesn't scale
  with a time step at all, so its phase-transition behaviour (ordered ↔
  disordered flocking as a function of noise amplitude and density) is
  not directly reproducible with pymurmur's diffusion-coefficient
  parametrization without a conversion.

## 12. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vicsek_couplage` (η) | 0.8 | Alignment-memory blend weight |
| `vicsek_diffusion` (D) | 0.8 | Angular diffusion coefficient |
| `vicsek_time_step` (dt) | 0.1 | Internal time step |
| `vicsek_radius_influence` (r) | 5.0 | Neighbour interaction radius |
| `vicsek_velocity` (v0) | 1.0 | Prey cruise speed |
| `vicsek_velocity_predator` | 2.0 | Predator cruise speed |
| `vicsek_heading_inertia` | 0.0 | Heading-blend inertia (0 = disabled) |
| `vicsek_radius_predators` | 80.0 | Fear/hunting detection radius `R_pred` (§8) |
| `vicsek_weight_afraid` | 3.0 | Flee-term weight in the fear-blend (§8.1) |
| `vicsek_detect_ratio` | 1.5 | Multiplier on `R_pred` for predator hunting radius (§8.2) |
| `vicsek_predator_noise_ratio` | 0.1 | Directional noise in predator hunting (§8.2) |
