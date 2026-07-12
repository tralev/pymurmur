# TODO — Ideas & Math from `sci/new1_sci.md` (crs48/murmuration) Not Implemented in the Codebase

Comparison of `sci/new1_sci.md` — the field/blob "lava lamp" simulation, autonomous
predator, trail/impostor rendering, and adaptive-quality reference — against the
`pymurmur/` codebase. Items are grouped by the source document's own sections.
*(simplified)* = a placeholder exists but does not match the documented math;
unmarked = missing entirely.

Overlaps with `todo_claude3.md` (which audited the consolidated reference) are
retained here with the **exact formulas and constants** this source document adds.

Already implemented from this document: 3D spatial hash grid with 27-cell queries,
SoA `Float32Array`-style layout, three ripple trains at offsets {0, 9.33, 18.67}
(envelope math differs — see §2.5), four monochrome theme palettes in the renderer
(unwired to config), per-mode performance benchmarks in `test/test_performance.py`.

---

## §1 Multi-Tier Simulation Architecture — missing entirely

- [ ] **Tiered backend selection by count** (CPU grid ≤1,200 → CPU field ≤30K →
  GPU tiers). The codebase has force *modes* the user picks manually; nothing selects
  a cheaper backend automatically when `num_boids` exceeds a tier ceiling, and there
  is no fallback cascade ending in "reduced count".
- [ ] **GPU compute tier**: no GPGPU simulation path at all (moderngl can express the
  ping-pong pattern via transform feedback / compute shaders). The texture-planning
  arithmetic (`S = ⌈√N⌉`, `C = S²`, texel `(i mod S, ⌊i/S⌋)`, `w` as active flag) is
  unimplemented and untested.
- [ ] **Adaptive degradation order** (trails → pixel ratio → particle count) — see
  §11 below; nothing consumes the perf flags.

## §2 Field-Based Blob Simulation — the core of this document, mostly absent

[field.py](pymurmur/physics/forces/field.py) implements a single-CoM sketch. Missing:

- [ ] **§2.1 Five blob anchors** `B₀…B₄` orbiting the flock centre with the
  documented per-axis Lissajous coefficients (e.g.
  `B₀ = C + (sin(t·0.19)·0.74, sin(t·0.31+0.8)·0.48, cos(t·0.23)·0.62)`).
- [ ] **§2.1 Cyclic phase weights**:
  `φᵢ = fract(seedᵢ·3.71 + t·0.022 + sin(seedᵢ·19 + t·0.11)·0.09)`,
  `w_k = max(0, 1 − |φ−c_k|_wrap·7.5)²` at centres {0, .2, .4, .6, .8}, and
  `T_legacy = Σ B_k w_k / Σ w_k`. Note `flock.seeds` exists for exactly this and is
  unused by field mode — every bird currently shares one target, so the documented
  "prevents collapse to a single point" property is lost.
- [ ] **§2.2 Leader/chaser dynamics**: 7 seed-derived groups, the lagged leader
  anchor formula, seed-dependent leader lag
  `hash(seed+9.17)·(1.1 + chaseStrength·2.4)` with ~16% leaders (`role ≥ 0.84`),
  **golden-angle stratified shell offsets** (golden angle 2.39996323,
  `shell = fract((slot+1)·0.754877)^(1/3)`, laminar breathing ×`(1+sin(...)·0.035)`),
  and the final blend `T = mix(T_legacy, chase_target, chaseStrength)`.
  `config.field_chase_strength` exists and is read by nothing.
- [ ] **§2.3 Shell force**: target *radius* rather than target point —
  `R_blob = (0.24 + (0.5+0.5·sin(seed·41+t·0.29))·0.16 + sin(φ·2π+t·0.17)·0.05)·R_pilot`,
  `F = −Δ̂·(d−R_blob)·cohesion·1.35·(1−chaseStrength)`. Code applies an
  unconditional constant-magnitude pull toward the CoM
  ([field.py:31-37](pymurmur/physics/forces/field.py#L31-L37)) — no shell, no
  hollow/layered structure.
- [ ] **§2.4 Slot repulsion kernel** *(simplified)*: doc uses **±**{1,7,31} offsets
  with the bounded kernel `proximity² = ((r_slot−d)/r_slot)²` active only within
  `r_slot`, gain `separation·(0.14 + chaseStrength·0.05)`. Code uses forward-only
  {1,7,31} with an **unbounded 1/d²** kernel and no cutoff radius
  ([field.py:53-66](pymurmur/physics/forces/field.py#L53-L66)) — distant slot pairs
  still attract force, near pairs blow up.
- [ ] **§2.5 Ripple envelope math** *(simplified in the Ripple extension)*:
  [ripple.py](pymurmur/physics/extensions/ripple.py) has the three offsets and a
  Gaussian `exp(−δ²)`, but is missing the rise/fall envelope
  `smoothstep(0.6,1.7,t)·(1−smoothstep(6.2,8.8,t))` (code pulses never die out —
  radius grows forever at `t·200`), the documented radius/width laws
  (`0.16 + t·0.16`, `0.11 + t·0.012`), **moving ripple origins** (Lissajous around C;
  code uses the CoM itself), the **twist component** (`+ ripple_twist·0.28`), and the
  flow coupling `·flow·(0.13 + waveGain·0.04)`. It is also O(N) per train in a
  Python loop rather than vectorised.
- [ ] **§2.6 Buoyancy**: `F_y += (sin(d·8 − t·1.1 + seed·17)·0.09 + (T_y −
  p_y)·0.24)·(0.75 + flow·0.25)` — no vertical undulation term exists; field-mode
  flocks can go perfectly planar.
- [ ] **§2.7 Inner-radius core cavity**:
  `inner = R_blob·(0.28 + shellInfluence·0.18 + separation·0.012)`; push out with
  `(inner−d)·separation·1.4` when inside. Nothing prevents collapse below a minimum
  radius (and without §2.3 there is no radius at all).
- [ ] **§2.8 Fold noise**: the second, higher-frequency undulation layer (spatial
  2.4–3.7, temporal 0.43–0.73, coupled to ripple activity via `flowPulse`). Code has
  only one low-frequency sine field.

## §3 Reynolds Steering Refinements

- [ ] **§3.1 Inertial smooth turning**:
  `v_final = lerp(v_raw, clamp_speed(v_raw), inertia)` with inertia 0.6–0.9.
  [boid.py:integrate](pymurmur/physics/boid.py#L39-L51) hard-clamps speed every step
  — there is no inertia parameter anywhere.
- [ ] **§3.2 Bounded panic speed boost**:
  `local_maxSpeed = maxSpeed·(1 + min(1.35, panic·(0.72 + waveGain·0.18 +
  vacuoleStrength·0.12)))`. The code's counterpart
  ([predator.py:68](pymurmur/physics/extensions/predator.py#L68)) multiplies
  velocity by 1.5 **every frame** a bird is in range — unbounded compounding rather
  than a raised ceiling.
- [ ] **§3.3 Blackening**: `sep_eff = separation·(2 − blackening)`,
  `coh_eff = cohesion·blackening` with
  `blackening = 1 + gain·threatProximity·0.85`. Code adds a fixed cohesion pull for
  panicked birds instead of modulating the existing force weights — no
  density-compression darkening effect.
- [ ] **§3.4 Tangential orbital force**:
  `F = normalize(axis × localDir)·alignment·0.035·(1−chaseStrength)` with the
  drifting seed-dependent axis — the term that produces laminar swirl around the
  shell. Absent.
- [ ] **§3.5 Curl flow field** *(simplified)*: doc composes sin+cos pairs per axis
  (`sin(p_y·2.8 + t·0.24 + seed) + cos(p_z·2.1 − t·0.17)`, …), then **normalizes**
  and scales by 0.08 (field) / 0.22 (grid). Code's flow
  ([field.py:44-49](pymurmur/physics/forces/field.py#L44-L49)) is three single sines
  at much lower spatial frequency (0.01/0.007), unnormalised, no seed term, no time
  term — a different (and weaker-structured) field.
- [ ] **§3.6 Viscous drag**: `F = −v·chaseStrength·(0.08 + flow·0.02)` — no damping
  term exists in field mode; only the global speed clamp bounds it.
- [ ] **§3.7 The full 12-term field composition** (shell, target pull, drift
  alignment, drag, orbital, slot spacing, flow, ripple, buoyancy, noise, threat,
  boundary) — code has 4 of 12 (target-pull-as-shell, drift alignment, flow-ish
  noise, slot spacing).

## §4 Autonomous Predator Model

[predator.py](pymurmur/physics/extensions/predator.py) has a two-phase skeleton;
essentially all the documented flight math is missing:

- [ ] **§4.1 Smoothed state**: no `attackDirection` / `turnAxis` — steering is
  instantaneous velocity assignment.
- [ ] **§4.2 Real egress phase**: doc flies *through and out* until
  `clear_distance = pass_distance·(0.72 + momentum·0.16)` with heading-away check
  (`dot < −0.12`); turn rate `0.54 + accel·0.025` rad/s on approach, lower on
  egress; `capture_radius = max(0.18, threatRadius·0.72)`. Code's "pass_through"
  branch **teleports** the predator to a random offset and immediately returns to
  approach ([predator.py:42-45](pymurmur/physics/extensions/predator.py#L42-L45)).
- [ ] **§4.3 `rotate_toward` axis-angle steering** with per-step angle cap and
  **exponentially smoothed turn axis** (prevents jitter, produces banked turns).
- [ ] **§4.4 Pass-through targeting**:
  `pass_distance = 0.92 + threatRadius·2.6 + momentum·1.32` and the sinusoidal
  `arc_offset` (lift/drift along the smoothed turn axis) for curved egress paths.
  `config.predator_momentum` exists and is dead.
- [ ] **§4.5 Threat wake**:
  `F = (r̂ − threatDir·0.35)·min(1.8,|v_threat|)·strength·√prox·0.42` — prey are
  pushed along the predator's motion, not just radially. Code force is purely radial.
- [ ] **§4.6 Tangent split** (both formulations): e.g.
  `F = (−away_z/d·1.45, away_y/d·0.28, away_x/d·1.45)·splitGain·√prox` — the
  XZ-biased lateral tear that opens the "hole" as the predator passes.
  `config.predator_split_gain` exists and is dead.
- [ ] **§4.7 Velocity wave amplification**: `F = v̂·waveGain·√prox·0.22` —
  directional acceleration along current heading, distinct from the §3.2 ceiling
  raise. Absent (no `waveGain` parameter exists).
- [ ] **§4.8 Threat modes**: only on/off exists. Missing `cursor` (threat at
  projected pointer — natural fit for the existing mouse handling), `orbit`
  (low-turn-rate 0.42 circling), and the mode enum itself.

## §5 Trail Rendering — missing entirely

`config.trails` (`off|velocity|accumulation`) is a dead field and
[renderer.py](pymurmur/viz/renderer.py) contains no trail path:

- [ ] **§5.1 Velocity-stretched impostors**: screen-space motion vector
  `project(p) − project(p − v·trailLength·0.12)`, stretch/point-size law
  (`1 + stretch·2.8`), shrinking head `headRadius = max(0.28, 1/(1+stretch·2.8))`,
  trapezoidal tail `0.22 + stretch·1.35` with sinusoidal waviness
  `sin(progress·(5.4+speed01·3.4)+seed)·waviness·stretch·0.18·envelope`.
- [ ] **§5.2 Screen-space accumulation**: fade-quad-over-previous-frame with
  `fadeOpacity = clamp(0.018, 0.32, 0.24 − persistence·0.19 − visibility·0.09)`,
  depth-only clear, then draw particles.
- [ ] **§5.3 The three-mode cost ladder** wired to config and to the §11 degradation
  cascade.

## §6 3D Particle Rendering

Only instanced tetrahedra exist. Missing:

- [ ] **§6.1 Sphere impostor fragment shader**: `p = uv·2−1`, `discard` outside unit
  disc, `z = √(1−r²)`, `edge = smoothstep(1.0, 0.72, r²)`, `shade = 0.55 + 0.45·z`.
  `config.point_sprites` is a dead field.
- [ ] **§6.2 Depth cues** (all four): size attenuation `∝ 1/depth^k`, depth fade
  `alpha·mix(1, 1−depth01, depthFade)`, velocity opacity `mix(0.65, 1, speed01)`,
  rim falloff `mix(1, 0.76, smoothstep(0.72, 1, r²))`. With the monochrome themes
  these are the only depth signals available; currently Blinn-Phong is the sole cue.
- [ ] **§6.3 Render-mode tiers** (`points` / `impostor-quads` / `instanced`) with
  count-based recommendation.

## §7 VR / Immersive Simulation — out of scope, except its safety rails

VR proper (SwarmPilotIntent/Rig, reference grid, medium modes, haptics) is excluded
by `functional_decomposition.md`. Three ideas port directly to desktop and are
missing:

- [ ] **§7.2 Pilot-aware forces** as a desktop "pilotable flock" mode:
  `F = heading·align·0.12 + (pilot−p)·coh·0.22 + (pilot−p)/d·(d−shellRadius)·0.42`
  — a keyboard-steered attractor with a shell (the existing influencer mode is the
  natural host).
- [ ] **§7.5 Data-oriented safety rails**: **`dt` clamp to [0, 1/20]** (the
  visualizer feeds raw `clock.tick()` dt into `sim.step()` — a window drag produces
  one huge Euler step), **NaN/isFinite guard** before writing positions, and
  **zero-allocation `step()`** (force primitives allocate `(N,3)` arrays every
  frame).

## §8 Flock Wander *(simplified)*

[wander.py](pymurmur/physics/extensions/wander.py) exists but implements a different,
weaker attractor:

- [ ] **`bounded_unit_travel`** multiscale composition (three frequency bands per
  axis, nested `sin(t·a + sin(t·b)·c)` phase modulation) with the **radial pulse
  normalisation** `scale = radial_pulse/max(1,|raw|)` that *guarantees* `|path| ≤ 1`.
  Code uses two-frequency products with no boundedness construction.
- [ ] **Speed/radius parameters** (`attractorSpeed·wanderSpeed`,
  `attractorRadius·wanderRadius`) — code hardcodes amplitude (100,100,50) and rate
  (0.001/frame).
- [ ] **Heading inference** `heading(t) = normalize(center(t+0.75) − center(t))` —
  needed by the drift-alignment force so birds align with where the flock is *going*.
- [ ] Fix while at it: the wander target orbits the **origin** `(±100, ±100, ±50)`,
  but the domain is `[0,1000]×[0,700]×[0,400]` — the attractor sits at a domain
  corner instead of wandering around the domain centre.

## §9 Monochrome Theme System *(one wire missing)*

Palettes exist in `renderer.py` (`THEMES`), but:

- [ ] `config.theme` is never passed by the Visualizer
  ([visualizer.py:39-44](pymurmur/viz/visualizer.py#L39-L44)) — one-line fix.
- [ ] The documented `color = mix(paper, ink, shade)`, `shade = 1 − rim·0.22`
  impostor shading model (lighting-free) — depends on §6.1 impostors; current
  shaders use Blinn-Phong instead.

## §10 Presets

- [ ] **None of the seven desktop presets are shipped** (Quiet Roost 3K / Ink Cloud
  18K / Predator Ripple 12K / Vacuole 10K / Silk Sheet 14K / Storm Turn 16K / Lava
  Lamp 16K) with their documented parameter vectors — natural to add as
  `conf/*.yaml` once field mode gains the parameters they set (`chase`, threat
  gains). The calm↔agitated / diffuse↔dense preset map is a good `--list-configs`
  organising principle.
- [ ] VR comfort presets — out of scope with VR itself.

## §11 Performance Diagnostics & Adaptive Quality *(simplified)*

[perf.py](pymurmur/analysis/perf.py) has EMA timing and a binary cpu/gpu ratio.
Missing:

- [ ] **Frame budget** `1000/max(24, targetFps)` and the "healthy if
  `avg ≤ budget·1.12`" gate.
- [ ] **Risk-based classification** (`likely-cpu` / `likely-vertex` /
  `likely-fragment` / `mixed`) using count, pixel-ratio, trail-mode, and scale
  thresholds — the current physics/render ratio cannot distinguish vertex- from
  fragment-bound.
- [ ] **The degradation cascade with hysteresis**: FPS < 78% of target sustained
  1.8 s → 1) trails off, 2) pixel ratio −0.15 (floor 0.75), 3) count −18%
  (floor 512). The code's `reduce_resolution`/`reduce_count` flags fire instantly
  (no sustain window) and **nothing consumes them** — `Visualizer.run()` never
  instantiates `PerfDiagnostics`, so the loop is unwired end to end.

## §12 Test & Validation Framework

Benchmarks and subsystem tests exist. Missing categories from the doc:

- [ ] **Simulation invariant tests**: no-NaN positions and bounded velocities after
  N steps, for every mode × boundary combination (would also catch the §7.5 gaps).
- [ ] **Soak test**: ≥10-minute run asserting stable memory (no unbounded growth in
  `metrics_history` / `frames` — the Recorder currently accumulates without bound).
- [ ] **Visual smoke tests**: headless-FBO screenshot comparison per mode (the
  infrastructure — `Renderer3D(headless=True)` + PIL — already exists).
- [ ] Bottleneck-classification unit tests (once §11 exists).

## §13 Particle Initialization

Only uniform-box init exists ([boid.py:random_positions](pymurmur/physics/boid.py#L187-L203)):

- [ ] **Volume-uniform sphere init**: `r = u^(1/3)·0.88`, `φ = acos(2w−1)` — the
  ∛ law for uniform density in 3D.
- [ ] **Tangential initial velocities**:
  `v = normalize(p × random_unit) · rand(minSpeed, maxSpeed)` — immediate orbital
  motion instead of the current random directions at fixed `0.8·v₀`.

## §14 Swarm Center Estimation

- [ ] **Exponentially smoothed centroid**:
  `center = mix(prev, centroid, 0.4–0.7)`. The predator, wander, ripple, and field
  modes each recompute the raw per-frame CoM independently — jittery for the
  predator's target and wasteful (4× redundant reductions per step). A shared,
  smoothed `flock.center` computed once per step covers all consumers.

## §15 Functional Force Composition

- [ ] **The `ForceTerm`/`composeForces` pattern**: pure terms
  `(context) → Vec3` composed by reduction, individually unit-testable and
  **runtime-togglable** (e.g. disabling noise for benchmarks). The codebase has
  mode-level dispatch but each mode is a monolith; there is no per-term toggle.
- [ ] **Exact grid-mode constants not matched** where counterparts exist:
  separation normalised by neighbour count (`/max(1,|N|)` — code doesn't normalise),
  cohesion `limit_length(p̄−p_i, 1)` (code's version is unbounded for far centroids —
  see `todo_claude2.md` §2), flow gain `0.22`, **seed-sinusoidal noise**
  `(sin(seed+t·1.17), sin(seed·1.31+t·1.41), cos(seed·0.73−t·1.23))·noise·0.18`
  (code uses Gaussian RNG noise — the doc's form is deterministic per seed, which
  would also fix reproducibility for this term).
- [ ] **Soft spherical boundary at `1.45·R_pilot`** with linear overshoot gain 1.6 —
  the code's `sphere` boundary mode hard-projects onto the radius instead.
- [ ] **Threat term with `vacuoleStrength`**:
  `push = r̂·strength·(1.1+vacuoleStrength)·prox` — the vacuole (hollow void)
  parameter has no counterpart anywhere.
