# TODO — Science & Math from `resulting_sci.md` Not Implemented in the Codebase

Comparison of the actual codebase (`pymurmur/`) against `resulting_sci.md` (the
consolidated 27-section scientific reference). This file lists **ideas, mechanisms,
and equations that are documented but absent or materially simplified** in the code.
Items marked *(simplified)* have a placeholder implementation that does not match the
documented math; unmarked items are missing entirely.

Already faithfully implemented (not repeated below): SoA arrays, hash-grid/cKDTree
dual index, four boundary modes, H₂ via Laplacian spectrum + J(m) minimisation, shape
PCA, trimmed gyration radius, voxel Θ′, Vicsek couplage blend + phase-diagram sweep
(`analysis/phase_diagram.py`), density-scaling sweep (`analysis/density_scaling.py`),
Knuth per-day predator hash, day-length model, tetrahedron instanced rendering with
gimbal-guard LookAt + Blinn-Phong + speed tint, orbit camera, EMA perf timing,
headless FBO→GIF capture.

---

## A. Core science gaps (highest impact on scientific validity)

### 1. The occlusion model doesn't occlude (§4) *(simplified)*

[occlusion.py](pymurmur/physics/occlusion.py) computes caps but skips the model's
defining step — every non-blind neighbour is marked visible:

- [ ] **Visibility test missing.** Doc: a neighbour j is visible **unless**
  `d̂_j · d̂_k ≥ cos α_k` for some nearer visible neighbour k (closest-first sweep).
  The code's loop appends every neighbour to `visible` unconditionally
  ([occlusion.py:91](pymurmur/physics/occlusion.py#L91)). Without occlusion, δ̂ and Θ
  are computed over the full σ-neighbour set and the Pearce density-regulation
  mechanism (steer toward light–dark boundaries) loses its physical basis.
- [ ] **Θ is not the documented probabilistic union.** Doc:
  `α = arcsin(min(b/d, 1))`, `Ω = 2π(1 − cos α)`, `Θ = 1 − Π_visible (1 − Ω_j/4π)`.
  Code: `theta += b_eff/d` (linear sum of small-angle cap radii, clamped at 1). The
  marginal-opacity observable (Θ ≈ 0.30 target) is therefore on a different scale
  than the paper's.
- [ ] **δ̂ weighting deviates.** Doc: `δ̂ = Σ sin α_j · d̂_j / Σ sin α_j`
  (boundary-length-weighted mean, `|δ̂| ∈ [0,1]` by construction). Code: weights are
  unnormalised `b/d` using the **isotropic** boid_size (ignoring the anisotropic
  `b_eff` it just computed), with a post-hoc clamp only when `|δ| > 1`
  ([occlusion.py:99-103](pymurmur/physics/occlusion.py#L99-L103)).
- [ ] Use exact `α = arcsin(min(b/d,1))` rather than the small-angle `b/d` throughout.

### 2. Pearce velocity update is missing its noise term (§1)

- [ ] Doc Eq.3: `v ∝ φp·δ̂ + φa·⟨v̂⟩_σ + φn·η̂` with constraint `φp + φa + φn = 1`.
  [projection.py:67-70](pymurmur/physics/forces/projection.py#L67-L70) blends only
  the first two terms — there is no `φn·η̂` random-unit-vector term and no
  representation of the weight constraint (no `phi_n`; `noise_scale` is only used by
  spatial mode). The stochastic term is what prevents artificial crystallisation in
  the paper's model.

### 3. Ecology envelope mostly decorative (§3) *(simplified)*

[ecology.py](pymurmur/physics/extensions/ecology.py):

- [ ] **`temperature()` is computed and never used** — the documented "temperature
  boosts roost pull by up to 20%" coupling is absent.
- [ ] **Roost pull is a linear ramp** (`to_roost * 0.01 * ramp`), not the documented
  logistic dusk factor.
- [ ] **`F_flee = φ_flee · (1 − d/R_d) · unit(r_i − r_p)`** — the Goodenough flee
  force (linear proximity taper) is not used by the predator; nor is the documented
  predator speed `~2·v₀`.
- [ ] Roost position and critical mass are hardcoded (500,350,40 / 500) instead of
  reading `config.ecology_roost` / `config.ecology_critical_mass`.

### 4. Metrics lack the physical-unit layer and several observables (§20)

[metrics.py](pymurmur/analysis/metrics.py):

- [ ] **Physical unit scaling** absent: `|v|_real = (8.94/FLIGHT_SPEED)·|v|_sim`,
  `|a|_real = (a_peak/MAX_FORCE)·|a|_sim` with `a_peak = 40 m/s²`, and mass
  `m = 0.075 kg` in force/power/angular momentum. All reported values are unitless.
- [ ] **Time-integrated energy `E_avg = Σ P_avg(t)·Δt`** not computed.
- [ ] **Nematic order parameters** missing: 2D `S = ⟨cos 2(θᵢ−θⱼ)⟩` and the 3D
  Q-tensor form `S = λ_max(Q)`, `Q = (1/N)Σ((3/2)û^αû^β − (1/2)δ_αβ)`. Only polar α
  exists — a two-lane/anti-parallel state is indistinguishable from disorder.
- [ ] **MSD is a single first-vs-last scalar** *(simplified)*. Doc:
  `MSD[n] = (1/(N_t−n))·Σ‖p(i+n)−p(i)‖²` per lag, with **minimum-image convention**.
  Code compares only the first and last snapshots and applies no wrap correction —
  under the default toroidal boundary, any wrap makes the value wrong.
- [ ] **τρ estimator differs** *(simplified)*: doc specifies
  `τρ = ∫ C_ρρ(Δt)/C_ρρ(0) dΔt` (trapezoidal integration of the autocorrelation);
  code fits `r(τ) ≈ exp(−τ/τρ)` and takes a median of per-lag estimates. Document or
  reconcile.
- [ ] **Number density `N / ((4/3)π·R_g³)`** (robust estimator) not computed.
- [ ] **Shape→m\* interpolation** missing: doc predicts
  `m* = interpolate(6.05, 9.78)` from thickness ratio; code computes thickness and
  finds m\* by J-minimisation but never implements/compares the shape-based
  prediction — which is the Young paper's actual claim (m\* depends on shape).
- [ ] Θ (internal opacity) is only populated by projection mode — in the other four
  modes `last_theta` is stale zeros; report it as N/A rather than 0.

---

## B. Force-mode math gaps

### 5. Reynolds spatial mode (§6)

- [ ] **Dual radii**: alignment neighbourhood at 0.75× of cohesion radius —
  `config.alignment_radius_ratio` exists but is read by nothing; all three primitives
  share one kNN set.
- [ ] **Separation distance threshold** (default 20): doc applies separation only
  within a short radius; code applies 1/d² separation to all k neighbours.
- [ ] **Perception radius**: `visual_range` never filters spatial-mode neighbours
  (pure topological kNN); doc's model is metric-radius based.
- [ ] **Force accumulation order** (rystrauss): forces → predator boost (1.4×) →
  `acceleration_scale` (0.3) → max_force clamp → velocity → **noise as direct
  velocity perturbation** → speed clamp → position. Code applies noise as a force and
  `config.acceleration_scale` is dead.
- [ ] Spherical wall theoretical form `Δv_wall = −μ·r̂/(R−|r|)` (asymptotic) — code
  uses hard projection + velocity subtraction *(simplified — acceptable, but the
  asymptotic form is the documented one)*.

### 6. Vicsek mode (§7)

- [ ] **Predator agent type** absent within vicsek mode: nearest-prey hunting with
  `R_detect = 1.5·R_predator`, 0.2× noise, and no-couplage direct pursuit.
- [ ] **Fear-weighted alignment**:
  `fear = clamp((R_pred − d̄_pred)/R_pred, 0, 1)`;
  `û = normalize((1−fear)·û_align + fear·û_flee)` with 3× neighbour-weight
  amplification when afraid.
- [ ] **Asymmetric collision resolution**: prey–prey half-correction each,
  prey–pred 100% correction to prey. `config.vicsek_radius_avoid` exists and is dead
  — no collision/avoidance step at all in vicsek mode.

### 7. Cosmic influencer mode (§8) *(heavily simplified)*

[influencer.py](pymurmur/physics/forces/influencer.py) diverges from every documented
mechanism:

- [ ] **Target trajectory is random, not Lissajous-smooth.** Doc: aperiodic C∞ path
  from 6 mutually-prime frequencies evaluated at *simulation time*
  (`sin(t/97)·200 + cos(t/217)·30`, …). Code draws `t = np.random.uniform(0, 2π)`
  **fresh each substep** — the target teleports randomly instead of tracing a path,
  which destroys the model's core "differential following of a smooth wanderer" idea.
- [ ] **Distance-based influence** `inf_A = clamp(100/d², 0.2, 0.8)` missing.
- [ ] **Rank-based influence**: doc ranks by distance **to the target** and maps to
  `seq(1 → 0.2)^1.8` (floor 0.055); code ranks by distance to the **CoM** with range
  `[0,1]^1.8` and no floor.
- [ ] **Move-then-steer sequence** (`p += d_old` before updating direction) — the
  one-step control lag that produces natural inertia — is missing; code uses standard
  force accumulation.
- [ ] **Density-scaled initialization** `multiplier = N^(1/3)·separation`, Gaussian
  cloud — missing (uniform-volume init is used for all modes).

### 8. Field/blob mode (§11, §12) *(heavily simplified)*

[field.py](pymurmur/physics/forces/field.py) is a minimal CoM-based sketch of crs48;
nearly all the documented machinery is absent:

- [ ] **5 Lissajous blob anchors** B₀–B₄ with independent trajectories — code uses
  the single flock CoM as the only attractor.
- [ ] **Cyclic phase weights**
  `φᵢ = fract(seedᵢ·3.71 + t·0.022 + sin(seedᵢ·19 + t·0.11)·0.09)`,
  `w_k = max(0, 1 − wrap-dist·7.5)²`, `T = Σ B_k w_k / Σ w_k` — missing entirely.
  Note the `flock.seeds` array exists precisely for this and is **unused by field
  mode**.
- [ ] **Leader/chaser groups** (7 groups, leader lag `hash(seed+9.17)·(1.1+cs·2.4)`,
  ~16% leaders, golden-angle shell offsets, `T = lerp(T_legacy, chase, cs)`) —
  missing; `config.field_chase_strength` is dead.
- [ ] **Shell force with breathing radius**
  `R_blob = (0.24 + sin_var·0.16 + sin_wobble·0.05)·R_pilot`,
  `F = −(Δ/d)·(d−R_blob)·cohesion·1.35·(1−cs)` plus **inner-radius outward
  expansion** — code applies an unconditional constant-magnitude CoM pull (no target
  radius, no hollow core).
- [ ] **Slot repulsion formula** *(simplified)*: doc uses offsets `{±1, ±7, ±31}`
  with bounded kernel `((r_slot−d)/r_slot)²` inside `r_slot` and gain 0.14; code uses
  forward-only `{1,7,31}` with an unbounded 1/d² kernel.
- [ ] **Fold noise** (2.4–3.7 spatial / 0.43–0.73 temporal undulation), **tangential
  orbital force** (`axis × localDir·alignment·0.035`), **buoyancy**
  (`sin(d·8−t·1.1)·0.09 + (T_y−p_y)·0.24`), **viscous drag**
  (`−v·cs·(0.08+flow·0.02)`), **curl-noise flow field** (normalized sin−cos
  composition; code's flow is three plain sines, not curl-like), and the **soft
  sphere boundary at 1.45·R_pilot** — all missing.
- [ ] **Functional force composition** (§12): the
  `composeForces(terms) = ctx → Σ term(ctx)` pattern — pure force terms, individually
  unit-testable and runtime-togglable — is not used; each mode is a monolith. This
  would also give the doc's per-term constants a natural home.

### 9. Angle-based steering (§10) — entirely absent

None of PyNBoids' mechanisms exist in any mode:

- [ ] Scalar-heading steering with wrapped `turnDir` and fixed turn rate (3D:
  axis-angle rotation toward target with max angle per frame).
- [ ] **Adaptive speed** `speed = base + (7 − neighborCount)²` (isolated birds rush
  to rejoin — cheap and visually significant).
- [ ] **Edge-distance turn-rate scaling**
  `turnRate += (1 − dist/margin)·(maxTurnRate − turnRate)`.
- [ ] Per-frame ±4° heading jitter (organic micro-instability).
- [ ] 3D cardinal-angle edge avoidance for the margin boundary mode.

---

## C. Predator, EvoFlock, and infrastructure gaps

### 10. Predator models (§15) *(heavily simplified)*

[predator.py](pymurmur/physics/extensions/predator.py) has approach/reset and a
radial push. Missing from the documented models:

- [ ] **crs48 flight dynamics**: `pass_distance = 0.92 + threatRadius·2.6 +
  momentum·1.32`, sinusoidal **arc offset** on egress, smoothed axis-angle
  `rotate_toward` steering (code teleport-resets to a random offset instead of an
  egress phase).
- [ ] **Threat wake force** `(r̂ − threatDir·0.35)·min(1.8,|v|)·strength·√prox·0.42`.
- [ ] **Tangent split** `(−away_z, away_y·0.28, away_x)/d·splitGain·√prox·1.45` —
  the XZ-biased term that carves the signature hole through the flock
  (`config.predator_split_gain` exists for this and is dead).
- [ ] **Wave amplification** `v̂·waveGain·√prox·0.22`.
- [ ] **Blackening modulation**: `sep_eff = sep·(2−blackening)`,
  `coh_eff = coh·blackening` — code approximates with a fixed cohesion pull; the
  documented version modulates the *existing* force weights.
- [ ] **Bounded panic speed**: doc `local_maxSpeed = maxSpeed·(1+min(1.35, …))`;
  code does `velocities *= 1.5` **every frame** a bird is within radius — unbounded
  compounding until the integrate clamp catches it; replace with the bounded form.
- [ ] **rystrauss behaviours**: escape steering, prey boosts (speed 1.8×, perception
  1.5×, accel 1.4×), and **hard-zero alignment/cohesion on predator detection**.
- [ ] **Runtime predator injection** (right-click spawn) and multiple predators.

### 11. EvoFlock GA (§5) *(simplified)*

[evoflock.py](pymurmur/analysis/evoflock.py) has islands/tournament/hypervolume, but:

- [ ] **No crossover.** The doc's core operator — 3-way tournament: evaluate, sort,
  delete worst, **uniform-crossover the best two**, insert offspring — is replaced by
  mutation-only reproduction of a single tournament winner.
- [ ] **4 parallel sims per evaluation, worst fitness returned** (anti-luck
  evaluation) — `eval_parallel` field exists, one sim is run.
- [ ] **SDF obstacle infrastructure absent**: zero-crossing collision detection and
  `p_corrected = p − SDF(p)·∇SDF/‖∇SDF‖`. Consequently the obstacle-avoidance
  objective `(f_cf)^500` is hardwired to 1.0 — a dead objective in the hypervolume.
- [ ] **Curvature objective is a proxy** (dispersion/α ratio), not the documented
  mean trajectory curvature `clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)`.
- [ ] **Speed objective unscaled**: targets [3,5] sim units with no mapping to the
  documented [19,21] m/s (blocked on the §20 unit layer).
- [ ] **15-parameter genome truncated to 10**, and two of those
  (`predictive_avoid_weight`, `static_avoid_weight`) are `setattr`'d onto SimConfig
  where **nothing reads them** — dead genes that add noise to the search. Missing:
  `forward_weight`, `max_dist_separate/align/cohere`, `angle_separate/align/cohere`
  (per-behaviour perception cones/radii), `fly_away_max_dist`,
  `min_time_to_collide`; and the fixed 7-topological-neighbour constraint.
- [ ] **Result persistence**: docstring promises `output/evolved.yaml`; `run()`
  never writes it.
- [ ] Reproduce the paper's headline experiment: **alignment emerges from separation
  alone** (evolve with alignment objective removed, verify α rises anyway).

### 12. Spatial partitioning (§13, §26)

- [ ] **Toroidal-corrected neighbour distance**: `config.use_toroidal_distance`
  (default True!) is read by nothing — both indexes and all force modes use raw
  Euclidean distances, so under the default toroidal boundary birds near opposite
  faces never see each other and flocks tear at the seam. Doc gives the corrected
  metric `dx = min(|x1−x2|, W−|x1−x2|)` and even warns about the sign bug
  (`fabs`) in the reference implementation. cKDTree supports this directly via
  `boxsize=`.
- [ ] Parallel neighbour search (doc's OpenMP two-phase pattern → Python analogue:
  `cKDTree.query(..., workers=-1)` and/or numba `prange` in the force pass).
- [ ] Incremental grid (boids migrate between cells on crossing) — optional; note it
  as the documented alternative to full O(N) rebuilds.

### 13. Multi-tier simulation & GPU compute (§14)

- [ ] The entire tiered-backend concept is absent: one CPU path serves all N. The
  documented ladder (grid → field O(n) → GPGPU ping-pong textures → compute-shader
  storage buffers, with automatic fallback and count reduction) has a natural
  moderngl analogue (transform feedback / compute shaders) that would make the 300K
  targets realistic. Square-texture particle packing (`S = ⌈√N⌉`) documented if a
  GPGPU tier is attempted.

---

## D. Rendering, UX, and operational gaps

### 14. Rendering (§16)

- [ ] **Sphere-impostor point sprites** (uv discard, `z = √(1−r²)`, Lambertian shade,
  edge smoothstep) — `config.point_sprites` is dead; only tetrahedra exist.
- [ ] **Depth cues**: size attenuation (`∝ 1/depth^k`), depth fade, velocity-based
  opacity, rim falloff — none implemented; these are the documented tricks that make
  16K+ flocks read as volumetric.

### 15. Trail rendering (§17) — entirely absent

`config.trails` (`off|velocity|accumulation`) is dead and
[renderer.py:5](pymurmur/viz/renderer.py#L5) falsely claims trail support. Missing,
in ascending cost per the doc: velocity-stretched impostors (stretch/headRadius/
tailLength/wave formulas), screen-space accumulation (per-frame fade-quad blend,
`fadeOpacity = clamp(0.018, 0.32, 0.24 − persistence·0.19 − visibility·0.09)`),
CPU trail lines, pixel-based fade.

### 16. Camera (§18)

- [ ] **Cinematic sweep** mode: half-orbit + gentle elevation bob + breathing zoom.
  (Orbit, auto-rotate, clamps are implemented.)

### 17. Performance diagnostics (§21) *(simplified)*

[perf.py](pymurmur/analysis/perf.py) has EMA + a cpu/gpu ratio. Missing:

- [ ] `frameMs = min(250, …)` spike cap (one GC pause currently poisons the EMA).
- [ ] `budget_ms = 1000/max(24, targetFps)` frame-budget concept.
- [ ] Four-way bottleneck heuristic (cpu / vertex / fragment / mixed) — code cannot
  distinguish vertex-bound from fragment-bound.
- [ ] **Staged adaptive degradation with hysteresis** (FPS < 78% of target sustained
  ≥ 1.8 s → 1. trails off, 2. pixel ratio −0.15 floor 0.75, 3. count −18% floor 512).
  Code sets advisory `reduce_*` booleans that **nothing consumes** — the visualizer
  never instantiates `PerfDiagnostics`, so the whole adaptive-quality loop is
  unwired.

### 18. Data-oriented safety rails (§22)

- [ ] **dt clamp to [0, 1/20]** — the visualizer feeds raw `clock.tick()` dt to
  `sim.step()`; a window drag or breakpoint produces one giant integration step.
- [ ] **NaN guard** (`isFinite` before writing positions) — one NaN currently
  propagates through the whole SoA within a few frames.
- [ ] Zero-allocation `step()`: per-frame allocations are pervasive (force arrays,
  cKDTree rebuild, per-bird lists). At minimum pre-allocate the force accumulation
  buffers.

### 19. Interaction & presets (§23, §24)

- [ ] **Preset hotkeys**: the documented a–h,w Pearce preset table (φp/φa/σ/mode per
  key) — `analysis/presets.py` exists but differs from the documented table and is
  wired to no key handler.
- [ ] **Runtime injection**: click-to-add boid, right-click predator, C = clear.
- [ ] Continuous parameter sliders (or any in-window UI); current stats live only in
  the title bar vs. the documented periodic stats display (speed m/s, accel, power,
  L, dispersion — blocked on §20 units).
- [ ] crs48 desktop preset table (Quiet Roost / Lava Lamp / Ink Cloud / Predator
  Ripple / Vacuole / Silk Sheet / Storm Turn with counts and weights) — none shipped
  as YAML configs.

### 20. Initialization & PRNG (§25)

- [ ] Velocity-init variants: uniform direction × random speed in [1, V₀]
  (code uses fixed 0.8·v₀), and tangent-to-sphere init (immediate orbital motion).
- [ ] Position-init variants: blob-based (5 centres, ∛-uniform shell) and
  density-scaled Gaussian (`N^(1/3)·sep`) — only uniform-volume exists.
- [ ] (Seedable-PRNG discipline itself is covered in `todo_claude2.md` §3 — the
  unseeded `np.random` calls break the determinism these docs assume.)

### 21. Build & tooling (§27)

- [ ] **Benchmark API**: `sim.benchmark(flock_size, num_steps) → per-step durations`
  (nanosecond-resolution list, not just EMA) — useful for the §4.3 scaling claims and
  CI perf regression.
- [ ] Env-var overrides for capture resolution/count/output (CLI flags exist;
  env-vars are the documented headless-farm interface).

---

## Explicitly out of scope (documented in `resulting_sci.md`, excluded by `functional_decomposition.md`)

Listed for completeness; implement only if scope changes:

- **MARL/PPO flocking (§9)** — centralized 6N-obs/3N-action PPO, two-layer control
  (RL 0.1× + rules 0.01×). The decomposition doc declares ML training external.
- **VR simulation (§19)** — SwarmPilotIntent/Rig state machine, haptics, Quest 2
  caps, medium modes. One portable idea: the **pilot-aware forces** (shell pull
  `(pilot−p)/d·(d−R)·0.42` + heading alignment) would make a good desktop
  "pilotable influencer" hybrid mode without any VR dependency.
- **C++/pybind11/OpenMP builds (§27)** — Python-native project; numba (see
  `todo_claude1.md`) is the sanctioned substitute.
