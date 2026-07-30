# Field Mode (Force Compositing)

This document defines the field/blob anchor mode: an O(N), fully
vectorised force mode with no neighbour queries.  Up to 11 named force
terms are composed via a shared per-frame context and summed into a
single acceleration.  Birds are assigned to blob anchors via cyclic
phase weights, with a leader/chaser subgroup using golden-angle
stratified shells.

---

## 1. Force Compositing Contract

Field mode does not use the separation/alignment/cohesion triad.
Instead, up to 11 named force terms are evaluated independently and
summed:

```
a = Σ_{term in enabled_terms} term.compute(ctx)
```

Each term receives a shared `FieldTermContext` carrying active-sliced
positions, velocities, per-bird seeds, the current time, the flock
centroid, unit scale, targets, and config-derived gain parameters.
Terms can be disabled at runtime via the `disabled_terms` config list.

The 11 terms, in composition order:

| # | Term | Description |
|---|------|-------------|
| 1 | `shell` | Oscillating equilibrium-radius shell force |
| 2 | `target_pull` | Direct pull toward the assigned target |
| 3 | `slot_repulsion` | Quadratic kernel repulsion at fixed array offsets |
| 4 | `tangential` | Perpendicular orbital steering around target |
| 5 | `buoyancy` | Vertical lift force (z-up) |
| 6 | `curl_flow` | Rotational flow about the flock centroid |
| 7 | `fold_noise` | Ripple-modulated perturbation |
| 8 | `noise` | Deterministic per-bird jitter (`seed_noise3`) |
| 9 | `viscous_drag` | Speed-dependent damping |
| 10 | `drift_alignment` | Alignment to wander heading or static drift direction |
| 11 | `floating_boundary` | 1.45·R_blob dynamic soft boundary |

After composition, the total acceleration is clamped to `max_force`.

---

## 2. Target Assignment

### 2.1 Blob Anchors (P3.2)

Five Lissajous blob anchors `B₀`–`B₄` are computed at the current time
`t`, each offset from centre `C` by `U`. These are **not** one
parametrized family varying by a single frequency/phase — each of the
5 anchors hardcodes its own distinct sin/cos assignment, frequency,
phase, and amplitude per axis:

```
anchor_k(t) = C + U · (x_k(t), y_k(t), z_k(t))
```

| k | x_k(t) | y_k(t) | z_k(t) |
|---|--------|--------|--------|
| 0 | sin(0.19t)·0.74 | sin(0.31t+0.8)·0.48 | cos(0.23t)·0.62 |
| 1 | cos(0.17t+1.6)·0.68 | sin(0.37t+2.1)·0.54 | sin(0.29t+0.4)·0.72 |
| 2 | sin(0.27t+2.7)·0.58 | cos(0.21t+1.2)·0.42 | cos(0.33t+2.5)·0.68 |
| 3 | cos(0.24t+3.4)·0.70 | sin(0.33t+0.6)·0.50 | sin(0.18t+1.4)·0.58 |
| 4 | sin(0.14t+4.4)·0.48 | sin(0.47t+2.3)·0.62 | cos(0.26t+4.0)·0.70 |

Each anchor is hand-tuned so its 5-body constellation traces
non-repeating, mutually-offset Lissajous paths around the centre —
deliberately irregular rather than a symmetric rotation, so the blob
doesn't read as a rigid rotating shape.

Each bird is assigned a phase weight per anchor via a cyclic hash of
its seed, producing a blended target:

```
T_legacy_i = Σ_k phase_weight(k, seed_i) · anchor_k(t)
```

### 2.2 Leader/Chaser Groups (P3.3)

Birds are divided into `num_groups` (default 7) seed groups.  The top
`leader_fraction` (default ~16%) of each group are leaders; the
remainder are chasers.  Leaders use the blob-anchor targets directly.
Chasers are placed on golden-angle stratified shells around their
group's leader target and blend toward it via `chase_strength`:

```
for each chaser k in group g:
    shell_target = leader_target(g) + fibonacci_sphere_offset(k)
    target_k = lerp(T_legacy_k, shell_target, chase_strength)
```

---

## 3. Shell Force (P3.4)

Each bird is pulled toward or pushed away from its target to maintain
an oscillating equilibrium radius, via **one continuous formula**
driven entirely by `coh_eff` — there is no separate `sep_eff`-based
branch for the inside-the-shell case:

```
R_blob = (shell_radius_base + sin(seed·41 + t·0.29)·0.08 + sin(phase·2π + t·0.17)·0.05) · U
d = ‖p − target‖ ;  d̂ = (p − target) / d
F_shell = −d̂ · (d − R_blob) · coh_eff · 1.35 · (1 − chase_strength) · shell_influence
```

When `d > R_blob`, `(d − R_blob) > 0` and `−d̂` points inward, so
`F_shell` pulls the bird toward its target. When `d < R_blob`, the
sign of `(d − R_blob)` flips and `d̂` now dominates, so the same
formula pushes the bird back out — a single spring-like force with no
branch, not two differently-weighted regimes.

`sep_eff` appears only in a separate, smaller-radius **inner cavity**
push-out — the only place separation modulates this term:

```
inner = R_blob · (inner_radius_factor + (1 − chase_strength)·0.18 + sep_eff·0.012)
if d < inner:
    F_shell += d̂ · (inner − d) · sep_eff · 1.4
```

This keeps a small hollow core near the target even when the outer
shell force alone would let birds pile up at `d ≈ 0`.

---

## 4. Slot Repulsion (P3.5)

Birds repel neighbours at fixed array offsets — a deterministic pairing
that requires no spatial queries:

```
for offset in {±1, ±7, ±31}:
    j = (i + offset) mod N_active
    d = ‖p_i − p_j‖
    if d < r_slot:
        F += t̂_{i←j} · ((r_slot − d) / r_slot)² · separation
```

The quadratic kernel `((r_slot − d)/r_slot)²` produces strong repulsion
at close range that falls off smoothly.  The mod-wrap around the active
bird index ring ensures every bird has exactly 6 slot neighbours
regardless of spatial position.

---

## 5. Tangential Orbital (P3.6)

Birds orbit perpendicular to the line toward their target:

```
t̂ = normalize(target − p)
perp = t̂ × z_axis                              // orbit plane normal (z-up)
F = align · perp · sin(seed·41 + t·0.29) · tangent_pull
```

The sinusoidal modulation per bird creates a mix of clockwise and
counter-clockwise orbits, preventing the flock from rotating as a
solid body.

---

## 6. Buoyancy (P3.6)

A vertical (z-up) lift force:

```
F_z = flow · sin(seed·17 + t·0.13) · 0.5
```

Creates gentle vertical stratification — some birds float up, others
sink, modulated by the flow parameter.

---

## 7. Curl Flow (P3.6)

Rotational flow about the flock centroid `C`:

```
r = p − C
F = flow · flow_pull · cross(z_axis, r) · sin(seed·31 + t·0.09)
```

Produces a swirling motion around the vertical axis through the flock
centre.  The sinusoidal per-bird weight prevents solid-body rotation.

---

## 8. Fold Noise (P3.6, C3)

Ripple-modulated position perturbation:

```
F = ripple_env · flow · sin(seed·53 + t·0.23 + dot(p, C)·0.03) · random_dir(seed)
```

The `ripple_env` multiplier comes from the ripple extension — when a
ripple wave passes through, fold noise amplifies, creating visible wave
deformations that propagate through the flock.

---

## 9. Field Noise (C3)

Deterministic per-bird jitter via `seed_noise3`:

```
F = field_noise · seed_noise3(seed_i, t)
```

Unlike the random noise in spatial mode, this is fully deterministic
— same seed + same time always produces the same perturbation.

---

## 10. Viscous Drag (P3.6)

Speed-dependent damping:

```
F = −v · (0.002 + chase·0.002 + flow·0.001)
```

Slows birds proportionally to their speed.  Stronger when chase or flow
is active — prevents runaway acceleration under strong target pull.

---

## 11. Drift Alignment (P3.6)

Birds align with a wander heading or static drift direction:

```
if wander_heading exists:     d̂_wind = wander_heading
else if drift_direction ≠ 0:  d̂_wind = normalize(drift_direction)
else:                         skip

F = align · d̂_wind · (d̂_wind · v̂) · drift_pull
```

Alignment is weighted by how parallel the bird already is to the drift
direction — birds flying with the drift get less force than those
flying against it.

---

## 12. Floating Boundary (P3.12)

A soft spherical boundary at 1.45× the blob radius:

```
d = ‖p − C‖
if d > 1.45·R_blob:
    F = −t̂ · (d − 1.45·R_blob) · mu
```

Weaker than the shell force but always active — prevents birds from
drifting too far from the flock centre.

---

## 13. Blackening (P3.8)

When a predator threat is present, the cohesion and separation
effectiveness are modulated per bird based on threat proximity:

```
prox  = clamp(1 − dist/threat_radius, 0, 1)
black = 1 + blackening_gain · prox · 0.85
sep_eff = separation · (2 − black)    // weaker near threat
coh_eff = cohesion · black            // stronger near threat
```

Near a threat, birds cluster together (stronger cohesion) and relax
personal space (weaker separation) — the "safer together" response.

---

## 14. Summary of Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `field_cohesion` | 1.0 | Cohesion gain for shell force |
| `field_separation` | 1.0 | Separation gain for shell force + slot repulsion |
| `field_alignment` | 1.0 | Alignment gain for tangential, drift |
| `field_flow` | 1.0 | Flow gain for buoyancy, curl, fold noise, drag |
| `field_target_pull` | 0.22 | Direct pull toward target |
| `field_chase_strength` | 0.0 | Leader/chaser pursuit (0 = disabled) |
| `field_noise` | 0.035 | Deterministic jitter magnitude |
| `field_shell_radius_base` | 1.0 | Base shell equilibrium radius |
| `field_inner_radius_factor` | 0.15 | Inner cavity radius fraction |
| `field_shell_influence` | 1.0 | Shell force influence multiplier |
| `field_tangent_pull` | 0.08 | Tangential orbital strength |
| `field_drift_pull` | 0.10 | Drift alignment strength |
| `field_drift_direction` | (0, 0, 0) | Static drift heading (§11); ignored when a Wander extension heading exists |
| `field_flow_pull` | 1.0 | Curl/fold flow multiplier |
| `field_unit_scale` | auto | Spatial scale factor (or auto: 0.4 × min(W,H,D)) |
| `field_num_groups` | 7 | Leader/chaser seed groups |
| `field_leader_fraction` | 0.16 | Fraction of leaders per group |
| `disabled_terms` | [] | Term names to skip at runtime |

---

## 15. Taxonomy

Field mode is one of pymurmur's 7 interchangeable force-computation
strategies — a per-strategy dispatch registry: an ABC (or
shared-signature callable) plus a decorator populating a lookup table,
selected at runtime instead of branching on a hardcoded if/elif chain.
Its 6 siblings: spatial (the literal Reynolds 1987 triad), projection
(occlusion-geometry-driven boundary-seeking), vicsek (constant-speed
angle-coupling alignment), influencer (tick-driven Lissajous pursuit,
the only mode that owns bird positions directly), angle
(turn-rate-limited Rodrigues-rotation steering), and marl (deferred
control under an external per-bird policy).

Field mode is architecturally unlike all six siblings in one respect:
it is the only mode with zero neighbour queries — every one of its 11
force terms is a pure function of a bird's own state, its assigned
target, and shared context (time, flock centroid, unit scale), never
another bird's position or velocity directly. Where the other modes
compute *interactions*, field mode computes *choreography*: birds fly
toward moving target points rather than toward each other, and any
apparent flocking coherence is an emergent side-effect of birds sharing
nearby targets, not a direct force between them.

## 16. Beyond pymurmur: Unimplemented Extensions

- **Roosting / fixed-site landing.** A dedicated attractor toward a
  fixed "roost" location, decoupled into independent vertical and
  horizontal pull components, distinct from all 11 existing terms
  (none of which target a static site — `target_pull` follows the
  moving blob-anchor target, and `floating_boundary` is a containment
  force, not an attractor). Would need a new named term with its own
  fixed target point and gain.
- **Within-mode priority-ordered term evaluation.** All 11 terms
  currently compose additively, unconditionally, every frame. An
  alternative would evaluate them in priority order and let a
  high-priority term (e.g. `floating_boundary` breach) suppress
  lower-priority ones entirely rather than simply adding on top — the
  same "first-triggered wins" pattern used elsewhere in pymurmur for
  cross-system force budgeting, but not applied *within* field mode's
  own 11-term composition specifically.
- **GPU compute-shader term evaluation.** Field mode's O(N) vectorised
  numpy composition has no GPU-resident equivalent; a compute-shader
  port would let all 11 terms evaluate per-frame without a CPU
  round-trip, though at N ≤ a few thousand birds the current numpy
  path is already fast enough that this is a scaling concern, not a
  correctness one.
