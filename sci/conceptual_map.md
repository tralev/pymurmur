# Conceptual Map

This document maps the relationships among 22 surveyed flocking
implementations — how they relate by steering paradigm, which
techniques they share, and how ideas evolved from the classic
original through to the feature-complete engines.

---

## 1. Steering Paradigm Tree

Every implementation branches from one of six core steering approaches.
pymurmur (§22) appears under multiple branches because its seven force
modes span several paradigms: spatial and projection use Reynolds
steering, angle uses axis-angle rotation, field uses heading-blend
compositing, and vicsek is a unique constant-speed approach.

```
Boids Flocking (22 entries)
│
├─ Reynolds Steering (8) — desired − velocity
│  ├─ §02  JS Canvas          ▸ obstacles + noise
│  ├─ §03  Panda3D            ▸ seek/arrive demo (no flock)
│  ├─ §12  Unity GPU          ▸ reflective bounds + wall force
│  ├─ §13  Blazor             ▸ exponential separation
│  ├─ §14  Unity ECS          ▸ FOV cone + min/max speed
│  ├─ §18  Rust + Bevy        ▸ golden-spiral raycast + cage
│  ├─ §21  Python + PyOpenGL  ▸ Pearce δ̂̂ + H₂ + density scaling
│  └─ §22  Python (pymurmur)  ▸ 7 modes + SDF + Vicsek + field forces
│
├─ Force Accumulation (5) — direct factor addition
│  ├─ §01  Vanilla JS         ▸ classic original
│  ├─ §06  Processing         ▸ fixed speed + spherical confinement
│  ├─ §07a Processing         ▸ interactive slider blending
│  ├─ §07b Processing         ▸ k-NN topology + roosting + wing flap
│  └─ §22  Python (pymurmur)  ▸ field mode: 11 named terms composed into
│                                an acceleration + max-force clamp — no
│                                normalization step, so this sits with
│                                accumulation, not heading blend below
│
├─ Angle-Based (4) — scalar rotation, no vectors
│  ├─ §04  C + Raylib         ▸ priority zones
│  ├─ §16  Rust + Nannou      ▸ edge state machine + circular mean
│  ├─ §17  Python + Pygame    ▸ spatial grid + pixel trails + 7-nearest
│  └─ §22  Python (pymurmur)  ▸ Rodrigues rotation + unified zones
│
├─ Heading Blend (2) — normalized sum of unit directions
│  ├─ §05  Unity GPU          ▸ 3D Simplex noise + zone containment
│  └─ §11  Unity 3D           ▸ FOV + predators + skeletal trails
│
├─ Exponential Smoothing (3) — lerp toward target heading
│  ├─ §10a Unity GPU          ▸ O(N²) all-pairs global average
│  ├─ §10b Unity GPU          ▸ parallel prefix-sum reduction
│  └─ §20  Unreal Engine 5    ▸ hashed grid + bitonic sort + value noise
│
├─ Pearce Projection (2) — visual occlusion model
│  ├─ §21  Python + PyOpenGL  ▸ spherical-cap occlusion + H₂ + density
│  └─ §22  Python (pymurmur)  ▸ batched parallel occlusion
│
└─ Unique Approaches (3) — one of a kind
   ├─ §08  WebGL GPU          ▸ cosine-zone weighting + predator
   ├─ §09  Scala              ▸ velocity inertia + see-ahead ray
   └─ §15  Unity DOTS         ▸ priority-ordered stack + wander + banking
```

---

## 2. Technique Sharing Map

Cross-cutting features span steering paradigms — implementations
grouped by what they share, not how they steer.

| Technique group | Implementations |
|-----------------|----------------|
| **GPU Compute** | §05, §08, §10a, §10b, §12, §20 |
| **Spatial Partitioning** | §07b, §15, §17, §20, §21, §22 |
| **FOV Cone** | §11, §14, §22 |
| **Predators** | §08, §11, §21, §22 |
| **Procedural Noise** | §02, §05, §06, §11, §13, §15, §20, §21, §22 |
| **Obstacle Avoidance** | §02, §05, §09, §11, §15, §18, §22 |
| **Fixed Speed** | §06, §09, §22 (angle/vicsek/influencer) |

---

## 3. Platform and Dimension Split

```
CPU (18 entries)
├── 2D (7):  §01 JS, §02 JS, §04 C, §09 Scala, §13 C#, §16 Rust, §17 Python
└── 3D (11): §03 Python, §06 Java, §07a Java, §07b Java, §11 C#,
             §14 C#, §15 C#, §18 Rust, §19 GDScript, §21 Python, §22 Python

GPU Compute (6 entries) — all 3D
    §05 HLSL, §08 GLSL, §10a HLSL, §10b HLSL, §12 HLSL, §20 USF
```

All 6 GPU implementations are 3D.  No GPU implementation targets 2D.
The 7 2D implementations are all CPU-based.

---

## 4. Steering Lineage

How ideas evolved from the classic original through successive
implementations:

```
§01 · Classic Original (2010s)
│  └─ Factor-based force accumulation, steer-back boundaries
│
├─► §02 · Reynolds Formalization
│     └─ Proper steer = desired − velocity, 1/r² separation, obstacles
│
├─► §04 · Angle-based Branch
│     ├─► §16 · Edge state machine, circular mean
│     └─► §17 · Spatial grid, pixel trails, 7-nearest cap
│
├─► §05 · GPU Era Begins
│     ├─► §08 · Cosine-zone weighting, ping-pong textures, predator
│     ├─► §10a/§10b · Exponential smoothing, parallel reduction
│     ├─► §12 · Reynolds on GPU, reflective boundaries
│     └─► §20 · Hashed grid + bitonic sort, value noise
│
├─► §06/§07 · Processing Research
│     ├─► §07b · Topological k-NN (starling research)
│     └─► §06 · Physical measurements, spherical confinement
│
├─► §11 · Feature-Rich Peak
│     └─ FOV, predators, state machine, skeletal trails, raycast avoidance
│
├─► §13 · Web Assembly
│     └─ Exponential separation, 1/d weighted cohesion
│
├─► §14/§15 · ECS Architecture
│     ├─► §14 · FOV cone, min+max speed, multi-framework
│     └─► §15 · Priority stack, wander, banking, damping, Burst
│
├─► §18 · Physics Integration
│     └─ Golden-spiral raycast, Rapier physics, Lissajous target
│
├─► §21 · Scientific Murmuration
│     └─ Pearce 2014 projection, H₂ robustness, density scaling, τρ
│
└─► §22 · Feature-Complete Engine
      └─ 7 flocking modes, SDF+CSG obstacles, batched Pearce, Vicsek,
         field forces, influencer, angle mode, predator-prey species

§19 · Godot Skeleton (no flocking math)
```

---

## 5. Summary

| Relationship type | Description |
|-------------------|-------------|
| Steering paradigm tree | 6 branches grouping 22 implementations by core mechanism |
| Technique sharing | 7 cross-cutting technique groups spanning paradigms |
| Platform split | 18 CPU (7 2D + 11 3D) + 6 GPU (all 3D) |
| Steering lineage | Evolutionary path from classic → GPU → ECS → scientific → feature-complete |

---

## 6. Comparison Table

A one-row-per-implementation overview across five structural axes:
dimensionality, steering mechanism, separation kernel, boundary
handling, and time complexity.

| § | Dim | Steering | Separation | Boundaries | Complexity | Notable |
|:-:|:---:|----------|------------|------------|:----------:|---------|
| 01 | 2D | Force-accumulation | Linear factor | Steer-back | O(n²) | Classic original Reynolds |
| 02 | 2D | Reynolds (`desired−vel`) | 1/r² | Wrap | O(n²) | Obstacle avoidance via projection |
| 03 | 3D | Reynolds Seek/Arrive | — | — | N/A | Steering demo only (2 boids, no flock) |
| 04 | 2D | Rotation-based | Priority (closest) | Wrap | O(n²) | Three behavioral zones, scalar angles |
| 05 | 3D | Heading-blend | Equal-weight normalized | Zone-clamp | O(n²) GPU | 3D Simplex noise speed field |
| 06 | 3D | Force-accumulation | 1/r² | Spherical | O(n²) | Fixed speed, 7-neighbor cap |
| 07a | 3D | Force-accumulation | 1/r | Wrap (all 6 walls) | O(n²) | Interactive slider blending |
| 07b | 3D | Force-accumulation | 1/r | Wrap + wall-avoid | k-NN (6) | Topological neighbors + roosting + wing flap |
| 08 | 3D | Cosine-zone blend | Asymptotic 1/r² | Center attraction | O(n²) GPU | Predator, ping-pong textures, wing flap |
| 09 | 2D | Normalized + inertia | Equal-weight | Wrap | O(n²) | See-ahead ray obstacle avoidance |
| 10a | 3D | Exponential smoothing | Global avg (self-incl.) | — | O(n²) GPU | All-pairs global average |
| 10b | 3D | Exponential smoothing | Global avg (corrected) | — | O(log n) GPU | Parallel prefix-sum reduction |
| 11 | 3D | Direction-blend + momentum | FOV-weighted InverseLerp | 8-raycast avoidance | O(n²) | State machine, predators, skeletal trails |
| 12 | 3D | Reynolds | Velocity-dependent | Reflect / 1/d wall | O(n²) | CPU + GPU variants, C# limit bug noted |
| 13 | 2D | Reynolds | Exponential | Linear wall ramp | O(n²) | 1/d weighted cohesion, 30% jitter chance |
| 14 | 3D | Reynolds (alignment only) | Equal-weight | 1/d wall force | O(n²) | FOV cone, min+max speed clamp |
| 15 | 3D | Priority-ordered stack | 1/r | Spherical spring | Grid (opt.) | Wander, banking, velocity damping |
| 16 | 2D | Angle-based (±1° delta) | ±1° turn delta | Edge state machine | O(n²) | Circular mean, no velocity vectors |
| 17 | 2D | Angle-based | Distance-override | Edge-steer / Wrap | Grid | 7-nearest, circular mean, pixel trails |
| 18 | 3D | Reynolds | 1/r | Cage (constant push) | O(n²) | Golden-spiral raycast, Lissajous target |
| 19 | 3D | — | — | — | N/A | No flocking math — renderer skeleton only |
| 20 | 3D | Exponential smoothing | Linear ramp | Home force | Hashed grid + sort | Bitonic sort, value noise speed |
| 21 | 3D | Reynolds + Pearce δ̂ | 1/d² steric (Pearce SI) | Toroidal / Margin / Open | Grid (hash) | 3D spherical-cap occlusion, H₂ robustness, density scaling |
| 22 | 3D | Multi (7 modes) | 11 selectable kernels (sum/mean/unit/exp/…) | Toroidal / Open / Margin / Sphere / Sphere_soft (5 modes) | Grid / KDTree (N-adaptive) | SDF+CSG obstacles, 11-term field forces, batched Pearce, Vicsek predator-prey |

Row 22 (pymurmur) verified against the live registries: 7 force modes
(`MODE_REGISTRY`), 5 boundary modes, 11 separation kernels — the source
survey's original "sum/mean/unit" undersold the actual kernel count, so
this row has been corrected to reflect that.

---

## 7. Feature Matrix

A checklist of distinguishing techniques across all 22 entries.

| § | Reynolds | GPU | Spatial | FOV | Predators | Noise | Obstacles | Fixed-speed | Angle | 3D |
|:-:|:--------:|:---:|:-------:|:---:|:---------:|:-----:|:---------:|:-----------:|:-----:|:---:|
| 01 | | | | | | | | | | |
| 02 | ✓ | | | | | ✓ | ✓ | | | |
| 03 | ✓ | | | | | | | | | ✓ |
| 04 | | | | | | | | | ✓ | |
| 05 | | ✓ | | | | ✓ | ✓ | | | ✓ |
| 06 | | | | | | ✓ | | ✓ | | ✓ |
| 07a | | | | | | | | | | ✓ |
| 07b | | | ✓ | | | | | | | ✓ |
| 08 | | ✓ | | | ✓ | | | | | ✓ |
| 09 | | | | | | | ✓ | ✓ | | |
| 10a | | ✓ | | | | | | | | ✓ |
| 10b | | ✓ | | | | | | | | ✓ |
| 11 | | | | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| 12 | ✓ | ✓ | | | | | | | | ✓ |
| 13 | ✓ | | | | | ✓ | | | | |
| 14 | ✓ | | | ✓ | | | | | | ✓ |
| 15 | | | ✓ | | | ✓ | ✓ | | | ✓ |
| 16 | | | | | | | | | ✓ | |
| 17 | | | ✓ | | | | | | ✓ | |
| 18 | ✓ | | | | | | ✓ | | | ✓ |
| 19 | | | | | | | | | | ✓ |
| 20 | | ✓ | ✓ | | | ✓ | | | | ✓ |
| 21 | ✓ | | ✓ | | ✓ | ✓ | | | | ✓ |
| 22 | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Row 22 corrected from the source survey: the original marked
Fixed-speed and Angle blank for pymurmur, which is stale — angle,
vicsek, and influencer modes are all fixed-speed (§2's Technique
Sharing Map above already lists this), and angle mode is a full
Rodrigues-rotation implementation, not a gap. GPU is correctly blank —
pymurmur is CPU/numba, not a compute-shader implementation.

---

## 8. Cross-Reference Index

Where each mathematical technique appears across all 22 entries.

### Steering

| Technique | Sections |
|-----------|----------|
| Reynolds (`desired − velocity`) | 02, 03, 12, 13, 14, 18, 21, 22 |
| Force accumulation (factor addition) | 01, 06, 07a, 07b, 22 |
| Pearce boundary-seeking δ̂ (projection) | 21, 22 |
| Field force compositing (shell/target/orbital/etc.) | 22 |
| Vicsek constant-speed alignment | 22 |
| Influencer move-then-steer | 22 |
| Heading blend (normalized sum) | 05, 11 |
| Exponential heading smoothing | 05, 10a, 10b, 20 |
| Angle-based (scalar rotation, no vectors) | 04, 16, 17, 22 |
| Velocity inertia (current dir in blend) | 09, 11 |
| Priority-ordered rule selection | 04, 15 |
| Cosine-zone weighting | 08 |

pymurmur's field mode is listed only under Force Accumulation and
Field Force Compositing, not Heading Blend — see §1's note on why it
doesn't fit the normalized-sum category despite superficially similar
"combine several force terms" language.

### Separation

| Technique | Sections |
|-----------|----------|
| Linear factor (no distance falloff) | 01 |
| 1/r falloff (`r̂ / d`) | 07a, 07b, 15, 18 |
| 1/r² falloff (`r⃗ / d²`) | 02, 06, 21, 22 |
| Exponential decay (`exp(−(d−r)/r)`) | 13, 22 |
| Linear ramp (`r − d`) | 20, 22 |
| Equal-weight unit vectors | 05, 09, 14, 22 |
| Asymptotic (`r/d − 1`) | 08, 22 |
| Distance-override (nearest neighbor) | 17, 22 |
| Priority-zone (closest neighbor) | 04 |
| Fixed-angle delta (±1°) | 16 |
| FOV-weighted InverseLerp blend | 11 |
| Global average (center of mass) | 10a, 10b |
| Velocity-weighted (closing-speed modulated) | 22 |
| Cosine-zone weighted | 08, 22 |
| Bell-zone (preferred-distance, non-monotonic) | 22 |

pymurmur (§22) appears in far more rows here than the source survey's
single "sum/mean/unit" characterization suggested — its 11-kernel
separation registry covers most of the taxonomy in one configurable
implementation rather than one implementation, one kernel.

### Boundaries

| Technique | Sections |
|-----------|----------|
| Wrap-around (toroidal) | 02, 04, 07a, 07b, 09, 16, 17, 21, 22 |
| Steer-back / soft force nudge | 01, 08, 11, 13, 14, 15, 18, 20, 22 |
| Hard clamp / position override | 05, 09, 12, 21, 22 |
| Spherical confinement | 06, 15, 22 |
| No boundary | 03, 10a, 10b, 19 |
| SDF-based obstacle geometry (not domain boundary) | 22 |

### Neighbour Selection

| Technique | Sections |
|-----------|----------|
| Metric radius (single or multi-zone) | 01, 02, 04, 05, 06, 07a, 08, 09, 12, 13, 15, 16, 18, 20 |
| k-NN topological | 07b, 22 |
| FOV cone | 11, 14, 22 |
| Capped radius (N-nearest within range) | 06, 17, 22 |
| Global (all boids) | 10a, 10b |
| Occlusion-based visibility filtering | 21, 22 |

pymurmur (§22) is the only entry combining metric, k-NN/topological,
FOV, capped, and occlusion-based filtering — each available as a
per-mode selectable strategy rather than a fixed single approach.

---

## 9. Taxonomy

This document doesn't describe a single plugin family or a single
metric — it's a cross-cutting index into the different *kinds* of
content pymurmur's own scientific documentation is organized into.
That documentation set spans three broad categories: per-mode physics
documents (one per interchangeable force-computation strategy, each
covering that mode's specific steering formula — the largest category,
matching the 7-entry force-computation registry), per-plugin-family
documents (covering the smaller dispatch registries alongside it —
boundary handling, neighbour selection, obstacle avoidance, speed
enforcement, spatial-index acceleration, kernel dispatch, and noise
injection — each with a handful of interchangeable strategies of its
own), and per-metric documents (covering the scientific-observable
layer — spatial structure, order/motion, consensus robustness, opacity,
and derived cross-correlations — which are pure read-only functions of
flock state, architecturally separate from anything dispatched by a
registry). This document's own role is narrower than all of those: it
shows how the *steering* side of that space (the per-mode physics
documents specifically) relates to a much broader survey of
implementations outside this project, not to describe pymurmur's own
plugin architecture directly — that's what the per-mode and
per-plugin-family documents are each individually for.

## 10. Beyond pymurmur: Unimplemented Approaches

Whole steering/analysis paradigms from the wider collective-motion
field, not represented by any of pymurmur's 7 force modes nor by any
entry in the 22-implementation survey above:

**Cucker–Smale flocking.** A well-studied alternative to the Vicsek
model in the mathematics/control-theory literature, where each bird's
alignment weight toward a neighbour decays with distance via an
explicit kernel rather than a hard cutoff radius:
`w_ij = 1 / (1 + ‖p_i − p_j‖²)^β`, with the full velocity update
`dv_i/dt = (K/N) · Σ_j w_ij · (v_j − v_i)`. Unlike Vicsek's
constant-speed, radius-cutoff alignment (pymurmur's vicsek mode), it
has provable convergence-to-consensus conditions on the decay
exponent `β`, and no hard neighbour cutoff to tune. Would fit as an
eighth force mode, structurally closest to vicsek mode but with a
kernel-weighted rather than radius-gated neighbourhood and no
fixed-speed constraint.

**Model predictive / receding-horizon steering.** Every pymurmur mode
is purely reactive — it computes one frame's force or heading update
from the current instant's neighbour state. An MPC-style approach
instead optimises a short trajectory (a few frames ahead) against a
cost function (collision risk, alignment error, energy) and only
executes the first step of the optimised plan, re-solving next frame.
This is a fundamentally different computational shape — an
optimisation problem solved every frame rather than a closed-form
force evaluation — and none of the 22 survey entries nor pymurmur's
own 7 modes attempt it, likely because of the added per-frame
computational cost at flock scale.

**Imitation-learned steering.** pymurmur's MARL mode defers control to
an externally-trained *reinforcement*-learning policy (reward-driven,
trial-and-error training against a defined objective). A distinct
alternative from the collective-motion literature is *imitation*
learning: training a policy via supervised learning directly on
recorded real-bird trajectories (behavioural cloning), rather than
reward shaping — the resulting policy's stated goal is to reproduce
observed flocking statistics rather than to optimise an engineered
objective. This would need a distinct training pipeline (supervised
regression against trajectory data, not an RL environment loop) rather
than a variant of MARL mode's existing gym-environment machinery.
