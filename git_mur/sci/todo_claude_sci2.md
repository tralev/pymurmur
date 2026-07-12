# TODO — Ideas & Math from `sci/new2_sci.md` (crs48/murmuration v2) Not Implemented in the Codebase

Comparison of `sci/new2_sci.md` against the `pymurmur/` codebase. This document is
the v2 expansion of `sci/new1_sci.md` (same source repository), so a large block of
findings is shared — those are listed compactly in Part 1 with pointers to
`todo_claude_sci1.md`, which carries the file/line evidence. Part 2 gives full
treatment to the material that only appears in v2.

Already implemented from this document: the 3D spatial hash (§2.1) with 27-cell
top-k queries (code uses `argpartition` instead of worst-first insertion —
functionally equivalent, [flock.py:178-209](pymurmur/physics/flock.py#L178-L209)),
SoA float32 layout, the three ripple-train offsets, four theme palettes (unwired),
and the classic Reynolds grid force terms in rough form (with the §2.3 deviations
noted below).

---

## Part 1 — Gaps shared with `sci/new1_sci.md` (details in `todo_claude_sci1.md`)

All of the following v2 sections are unimplemented or simplified exactly as
catalogued in the sci1 audit:

- [ ] **§1.1–1.2 Multi-tier backends + GPU texture planning** — single CPU path,
  no count-based tier selection, no fallback cascade. *(sci1 §1)*
- [ ] **§3.1–3.3 Blob anchors, cyclic phase weights, leader/chaser** — field mode
  uses one CoM target; seeds unused; `field_chase_strength` dead. *(sci1 §2.1–2.2)*
- [ ] **§3.4 Shell force + inner-radius expansion** — no target radius, no cavity;
  v2 pins the inner-radius coefficient to `(0.28 + (1−chaseStrength)·0.18 +
  separation·0.012)`. *(sci1 §2.3, §2.7)*
- [ ] **§3.6 Ripple envelope math** — extension lacks the smoothstep rise/fall
  envelope, radius/width laws, moving origins, twist, and flow coupling.
  *(sci1 §2.5)*
- [ ] **§3.7 Buoyancy**, **§3.8 tangential orbital force**, **§3.9 curl flow field
  (exact form)**, **§3.10 viscous drag**, **§3.11 the full 13-term composition** —
  field mode implements 4 of 13 terms. *(sci1 §2.6, §3.4–3.7)*
- [ ] **§4.1 Inertial smooth turning**, **§4.2 bounded panic boost** (code compounds
  ×1.5 per frame instead), **§4.3 blackening** — all absent. *(sci1 §3.1–3.3)*
- [ ] **§5 Predator**: smoothed `attackDirection`/`turnAxis` state, real egress with
  `capture/pass/clear` distances, axis-angle steering, arc offset, threat wake,
  tangent split, wave amplification, threat modes (`cursor`/`orbit`) — the code
  predator teleport-resets and applies only a radial push. *(sci1 §4)*
- [ ] **§6 Bounded-unit-travel wander** with radial-pulse containment and
  forward-difference heading; code's Wander is a different two-frequency formula
  orbiting the domain *corner*. *(sci1 §8)*
- [ ] **§9.1–9.3 GPU trail rendering** (velocity-stretched impostors, accumulation
  blending, mode ladder) — `config.trails` dead. *(sci1 §5)*
- [ ] **§10 Sphere impostors + 4 depth cues + render-mode tiers** —
  `config.point_sprites` dead. *(sci1 §6)*
- [ ] **§11 Theme wiring** — palettes exist, `config.theme` never passed. *(sci1 §9)*
- [ ] **§14 Functional `ForceTerm`/`composeForces` composition** with per-term
  runtime toggling. *(sci1 §15)*
- [ ] **§15 Smoothed swarm-centre estimation** — every consumer recomputes a raw
  per-frame CoM. *(sci1 §14)*
- [ ] **§17 Perf: frame budget, healthy gate, cpu/vertex/fragment classification,
  78%/1.8 s degradation cascade** — EMA exists; everything downstream is missing and
  the advisory flags are consumed by nothing. *(sci1 §11)*
- [ ] **§18 Data-oriented rails: dt clamp [0, 1/20], NaN/isFinite guard,
  zero-allocation step()**. *(sci1 §7.5)*
- [ ] **§21 Test categories: simulation invariants (no NaN, bounded positions),
  soak test, visual smoke tests**. *(sci1 §12)*
- [ ] **§7–8 VR locomotion & environment** — out of scope per
  `functional_decomposition.md`; portable pieces are the pilot-aware forces and the
  §18 rails above. *(sci1 §7)*

## Part 2 — Material unique to v2, not implemented

### §1.3 Capability probing

- [ ] No startup capability detection exists at all. The Python analogue is real:
  probe for moderngl context version / standalone-context availability, numba
  importability, and scipy presence, and expose a `capability` report that backend
  selection (§1.1) and `--list-configs`-style diagnostics can read. Today a missing
  GPU crashes the visual path with a raw exception rather than degrading to
  headless.

### §2.3 Grid-mode force-term deviations (exact constants)

Where the code has counterparts, two v2-specified details are missing:

- [ ] **Separation normalised by neighbour count** — `· separation / max(1, found)`.
  [_base.py:separation_force](pymurmur/physics/forces/_base.py#L12-L39) sums raw
  1/d² contributions, so force magnitude scales with neighbour count instead of
  averaging — dense regions get disproportionately large kicks.
- [ ] **Cohesion capped at unit length** — `clamp(|c_avg − p_i|, 1)`. The code's
  `to_center / min(length, 1.0)` is *unbounded* for far centroids (also flagged as a
  correctness bug in `todo_claude2.md` §2).

### §3.5 Slot repulsion — v2 additions

Beyond the sci1 finding (bounded `proximity²` kernel, ± offsets, gain
`0.14 + chaseStrength·0.05`), v2 adds two specifics the code also lacks:

- [ ] **Cutoff radius law** `r_slot = 0.07 + separation·0.02` — the interaction
  radius scales with the separation setting; code has no cutoff at all.
- [ ] **Modulo wraparound** `other = positions[(i+offset) mod N]` — code's
  forward-slice pairing ([field.py:59-66](pymurmur/physics/forces/field.py#L59-L66))
  leaves the last `offset` birds without partners instead of wrapping.

### §4.4 Speed clamp — zero-speed convention

- [ ] Doc: zero-speed birds get `(minSpeed, 0, 0)` deterministically; code re-seeds
  a random direction from an **unseeded** RNG
  ([boid.py:53-57](pymurmur/physics/boid.py#L53-L57)). Either convention works, but
  the code's choice silently breaks same-seed reproducibility — adopt the
  deterministic fallback or thread the flock RNG through (`todo_claude2.md` §3).

### §5.3 Rodrigues rotation + turn-rate model

v2 supplies the concrete math the predator steering needs:

- [ ] **`rotate_around_axis`** (Rodrigues formula:
  `v·cosθ + (axis×v)·sinθ + axis·(axis·v)·(1−cosθ)`) — no rotation utility exists
  anywhere in the codebase (would also serve the §10 PyNBoids-style steering noted
  in `todo_claude3.md`).
- [ ] **Turn-rate parameterisation**: approach `0.54 + threatAcceleration·0.025`
  rad/s, orbit `0.42` rad/s, both scaled by `(1 − threatMomentum·0.24)` — no
  turn-rate concept exists; the code predator sets velocity directly.
- [ ] **Sign-aligned EMA of the turn axis** (`aligned = dot < 0 ? −axis : axis`
  before blending) — the anti-flip detail that keeps banked turns stable.

### §5.6 Threat force, spatial-hash-mode variant

- [ ] The v2 formulation applies **all four components with `broad = √proximity`**
  and a stronger push gain `(2.5 + vacuoleStrength·1.7)`:
  `F = push + wake + split + wave` as one bundle. The code has only the radial push
  with linear proximity and gain 0.5
  ([predator.py:54-58](pymurmur/physics/extensions/predator.py#L54-L58));
  `vacuoleStrength`, `splitGain`, `waveGain` have no config counterparts at all
  (`predator_split_gain` exists but is dead).

### §9.4 CPU trail lines (LineSegments)

The tier-independent trail fallback — new in v2 — is absent along with the GPU
trails:

- [ ] 5 segments per bird traced backward along velocity
  (`trailScale = 0.1·trailLength`), camera-plane perpendicular
  `(−v_y, v_x, 0)/√(v_x²+v_y²)`, sinusoidal ribbon wave
  `sin(progress·2π·2.6 + seed)·waveScale·progress²` (amplitude vanishing at the
  tip), drawn depth-test-off with uniform alpha. In moderngl terms: a `LINES` VAO
  with 2·5 vertices per bird — the natural first trail implementation for pymurmur
  since it needs no shader work, only a `previousPositions`-style buffer (see §18
  below).

### §12 Particle initialization — blob-based start

More specific than sci1's sphere init; all missing (only uniform-box init exists):

- [ ] **§12.1 Five hardcoded blob centres** (`(−0.48, 0.18, 0.12)`, …) assigned by
  `index mod 5` — visual interest from frame 0, and the natural seed layout for the
  §3 blob dynamics.
- [ ] **§12.2 ∛-uniform radial shells**: `radius = cbrt(rand)·(0.22 + rand·0.28)`
  with jitter 0.045 — uniform *volume* density per blob.
- [ ] **§12.3 Drift-biased tangential velocities**
  (`drift = (0.34±0.08, ±0.16, 0.08±0.08) + jitter(0.05)`) — coherent initial flow
  instead of the current isotropic random directions at fixed `0.8·v₀`.

### §13 Mulberry32 PRNG

- [ ] Not implemented, and in Python the right call is **not** to port it —
  `np.random.Generator(PCG64(seed))` is the ecosystem equivalent. The actionable gap
  it points at is the one already filed in `todo_claude2.md` §3: the codebase's
  stochastic terms bypass the seeded generator entirely. Implement seed-threading;
  skip the Mulberry32 port. Listed here so the decision is recorded.

### §16.1 Full preset parameter vectors

- [ ] v2 extends the preset table with **Inertia / Noise / Flow / Trail / Threat
  columns** (11 presets). None are shipped as `conf/*.yaml`, and several columns
  have no config field to map onto yet (`inertia` — §4.1; `chase` — §3.3;
  `trail` mode — §9; threat mode — §5.7). Shipping these presets is blocked on
  those parameters existing; track the dependency here.

### §19 Vec3 math library

- [ ] Not needed as a library (numpy fills the role), but two specific functions
  encode contracts the codebase currently gets wrong or lacks:
  - `limitLength3(a, M)` — the correct "cap at length M" (`scale by M/|a|` only when
    `|a| > M`), which is exactly what `cohesion_force` fails to do;
  - `isFinite3` — the NaN gate from §18 with a concrete signature. Add both as
    small helpers in `core/types.py` and use them at the force/integrate seams.

### §20 WebGPU compute pipeline

- [ ] No compute-shader tier exists. The documented structure (ping-pong storage
  buffers for positions/velocities, separate velocity and position kernels,
  workgroup 128, billboard rendering, device-loss fallback) maps to moderngl
  compute shaders (GL 4.3) and is the concrete blueprint for the "GPU tier" gap in
  §1. Independent of it, the **billboard vertex shader (6-vertex instanced quads)**
  is the same machinery §10's impostor mode needs — one implementation serves both.

### §21.1 Two v2-specific unit-test categories

- [ ] **Settings clamping / reducer invariants** — nothing validates `SimConfig`
  field ranges anywhere (a YAML with `sigma: -3` or `v0: 0` loads silently);
  add a `validate()` with clamps and a round-trip test.
- [ ] **Preset serialization round-trip** — `SimConfig.to_file→from_file` has no
  equality test; the nested-YAML flattening in
  [config.py:from_file](pymurmur/core/config.py#L136-L164) makes this an easy
  regression to introduce.
