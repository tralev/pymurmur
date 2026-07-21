# roadmap0.md — Introduction, system contract, and index

## 0. Introduction — what this document set is

This six-file set is the complete, self-contained implementation
roadmap for **pymurmur**, a strictly-3D murmuration simulator (150 →
300 000 birds) with interchangeable physics models, scientific
observables, real-time 3D rendering, capture tooling, evolutionary
parameter tuning, and a MARL bridge. It unifies the design plan, the
test plan, and the science portfolio into one traceable structure, and
it is **adjusted to the current `pymurmur/` codebase**: a substantial
part of the plan is already implemented, so every work item in
[roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md) carries a
**Status** line, and [roadmap5.md](roadmap5.md) holds the codebase
audit (confirmed defects, unclear implementations, dead fields) and the
prioritized improvement steps.

**This file (roadmap0) contains no implementation phases.** It states
*what the system is*: goals, the subsystem decomposition, the protocol
contracts, the Level-0 atom inventory, conventions, and an index of all
work items. Its counterpart bookend, [roadmap5.md](roadmap5.md), also
contains no phases: it holds the audit, the improvement register, and
the appendixes (deliberately-excluded scope with its recorded math, and
the documentation-change checklist).

**How to read.**
- **roadmap0 (this file)** — introduction; system contract stated
  top-down (Macro→Micro, §2/§4) and bottom-up (Micro→Macro, §3);
  conventions (§5); index of every work item (§6).
- **[roadmap1.md](roadmap1.md)** — architecture foundation phases
  **D0–D9** (each with code sketches, file targets, acceptance) and
  test infrastructure **T0–T6** (harness, fixtures, contract suites).
- **[roadmap2.md](roadmap2.md)** — science portfolio part 1:
  correctness cluster **S1** and the five mode workstreams **S2**
  (tracks A–E). Each item gives *math* (3D form), *impl* (module /
  class / config path / linkage), and *tests* (concrete assertions +
  test file).
- **[roadmap3.md](roadmap3.md)** — science portfolio part 2: metrics &
  analysis **S3**, rendering & capture **S4**.
- **[roadmap4.md](roadmap4.md)** — science portfolio part 3: UX &
  tooling **S5**, EvoFlock **S6**, MARL bridge **S7**; the unified
  sequencing diagram, effort table, and definition of done.
- **[roadmap5.md](roadmap5.md)** — codebase audit and improvement
  register; Appendix A (deliberately excluded scope, with recorded
  mechanics/math kept for a future scope change); Appendix B
  (documentation-change checklist); conclusion.

Every feature PR carries its inline test block — the *accept* criterion
**is** the test. Item IDs (D·, T·, S·) are stable: tests and the other
files reference them. These six files reference **only each other**;
their provenance corpus (per-source audits and the predecessor gap
analyses) was verified item-by-item, fully absorbed, and retired — no
external document is needed to act on this plan.

**Decisions taken (user-confirmed, binding):**
1. **Nested config** — `SimConfig` composed of per-subsystem dataclasses.
2. **ForceMode classes** — modes are small stateful classes behind a
   protocol.
3. **Scope** — core physics/metrics/rendering/UX + EvoFlock + MARL
   bridge (excluded tiers: [roadmap5.md](roadmap5.md) Appendix A).
4. **Presets honor documented intent** — YAML values load as written;
   goldens re-pinned; behavior change release-noted.
5. **Never regress toward a reference.** Where the codebase is already
   ahead of the behaviours this plan was distilled from (modulo-wrapped
   grid cells, sphere boundary modes, the richer predator FSM, extra
   metrics, the plugin architecture), that lead is preserved — items in
   [roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md) upgrade, never
   downgrade, existing capability.

**Status legend used throughout roadmap1–roadmap4:** **DONE** (matches
contract) · **PARTIAL** (exists, incomplete) · **DIVERGES** (exists,
different semantics — decide & pin) · **MISSING** · **VERIFY**
(implemented but correctness unclear).

---

## 1. Non-negotiable goals

1. **Strictly 3D — simulation.** Every state vector is `(…, 3)`; no 2D
   fallbacks anywhere in `pymurmur/physics/`; z-up world; `depth > 0`
   enforced by config validation; invariance tests use random SO(3)
   rotations, never z-only. Guarded by test T1.3
   ([roadmap1.md](roadmap1.md)).
2. **Strictly 3D — visualization.** All rendering is 3D: instanced
   meshes / impostors in a perspective or orthographic 3D camera,
   depth-tested, with 3D trails, dual 3D viewports, and 3D capture. The
   GPU-free capture fallback renders 3D scatter projections (two 3D
   views), never a 2D simulation.
3. **Highly modular.** New capability = new module/class behind an
   existing protocol (ForceMode / Extension / SpatialIndex /
   render-mesh registry entry / metric / reward term / SDF primitive /
   preset) — never `if mode == ...` branches in shared code. The
   extension seams are the protocols in §4 below.
4. **Clean Macro→Micro.** The six-subsystem decomposition (§2) stays;
   every config field, protocol and dependency rule is traceable
   doc ↔ code (import matrix test T1.1, config-drift test T1.2,
   doc-drift test T1.4 fail CI on divergence —
   [roadmap1.md](roadmap1.md)).
5. **Clean Micro→Macro.** Level-0 primitives stay pure functions on
   arrays; assemblies compose them; composition is a **DAG** (no
   `flock ↔ forces` cycle); **no component ships without its
   composer** — a dead atom (defined but never composed) is an error,
   caught by T1.2-style drift guards.

---

## 2. Macro→Micro — subsystem decomposition and dependency matrix

Six subsystems, top-down:

```
core/        — config contract + Level-0 types/atoms (numpy/stdlib only)
physics/     — boid integration, occlusion, steric, obstacles, flock state,
               spatial indexes, force modes, behavioural extensions
simulation/  — SimulationEngine: the step orchestrator + command queue
analysis/    — observables (metrics, rewards, presets) and drivers
               (perf, evoflock, phase_diagram, density_scaling, gym_env)
viz/         — camera, shaders, renderer, trails, HUD, input, visualizer
capture/     — recorder(s): GIF/CSV/JSON export, GPU-free fallback
```

**Dependency matrix** (enforced by the architecture test T1.1,
[roadmap1.md](roadmap1.md)):

```
core/                    → numpy/stdlib only
physics/boid             → core                      (never flock/forces)
physics/occlusion|steric → core                      (pure numpy)
physics/obstacles        → core
physics/forces/*         → physics primitives, core  (read flock arrays;
                                                      no cKDTree construction)
physics/flock            → core                      (NEVER forces — cycle dead)
physics/extensions       → physics/flock(read), core
simulation/engine        → physics/*, analysis/{metrics,rewards}, core
analysis/{metrics,rewards,presets} → physics/flock(read), core     (tier F1)
analysis/{perf,evoflock,phase_diagram,density_scaling,gym_env}
                         → simulation, core                        (tier F2)
viz/                     → core, physics/flock(read), analysis/presets
                           (holds an engine *reference* from __main__;
                            imports no simulation modules)
capture/                 → simulation, viz, core
__main__ / scripts       → everything
```

Named regression edges (also enforced):
`physics.flock !→ physics.forces`; `viz.input_control !→ simulation`;
no `cKDTree(` constructed inside `forces/`; no module-level
`np.random.*` anywhere; no `print(` in package sources (logging only).

---

## 3. Micro→Macro — Level-0 atoms, conventions, composition rules

### 3.1 Level-0 atoms (all in `pymurmur/core/types.py`, unit-tested before first use)

- `min_image(Δ, box) = Δ − box · round(Δ/box)` — per-axis toroidal
  minimum-image displacement. The correct distance form is
  `d = min(|Δ|, L−|Δ|)` per axis (a signed-dx variant is a known bug
  class to test against).
- `rotate_about(v, k̂, θ) = v cosθ + (k̂×v) sinθ + k̂(k̂·v)(1−cosθ)`
  (Rodrigues rotation).
- `hash01(x) = fract(sin(x·12.9898)·43758.5453)` — deterministic hash to
  [0, 1).
- `smoothstep(a, b, x) = t²(3−2t), t = clamp((x−a)/(b−a), 0, 1)`.
- `fibonacci_sphere(n)` — golden-angle near-uniform unit vectors:
  `ga = π(3−√5) ≈ 2.399963`; `y_i = 1 − 2(i+0.5)/n`, `r_i = √(1−y_i²)`,
  `θ_i = ga·i`; row `(cos θ_i·r_i, y_i, sin θ_i·r_i)` → `(n,3) float32`,
  every row unit-length (z-up: swap axes). Shared atom — stratified
  shells and isotropic test fixtures *compose* it rather than re-derive
  the golden angle inline.
- `normalize3` — 0-safe vector normalization.
- `limit3(v, m)` — scale by `m/|v|` **only when** `|v| > m` (the correct
  "cap at length m"; sub-cap vectors pass through unscaled).
- `isfinite3(a)` — row-wise finiteness gate `(N,3) → (N,) bool` (the
  NaN-guard atom).
- `seed_noise3(seeds, t)` — deterministic per-bird sinusoids, bounded
  ±0.18/axis (consumed by the grid-tier noise mode,
  [roadmap2.md](roadmap2.md) S2.B11 — an atom must have a composer).

### 3.2 Array and randomness conventions

- All neighbour indices are **global capacity-space** rows; every
  primitive documents its index space.
- `active` may have holes at any time — every mode/metric must be
  correct under a holey mask (fixture + matrix tests in
  [roadmap1.md](roadmap1.md) T0.1/T4.1).
- `flock.rng` (a single seeded `np.random.Generator`, seeded from
  `config.flock.seed`) is the **only** randomness source anywhere.
  `seed=0` must be honored as a distinct seed (not conflated with
  unset). Recorded decision: the PRNG is numpy's default PCG64 via
  `np.random.default_rng` — no ports of source-specific generators
  (e.g. Mulberry32); determinism guarantees flow from the single
  seeded generator, not from a particular algorithm.
- Hot-path allocation hygiene: force primitives and `integrate()`
  accept/reuse preallocated scratch buffers (`out=`-style, e.g.
  `flock.scratch3: (capacity,3) float32`) — transient numpy temporaries
  are acceptable, unbounded per-frame growth is not.

### 3.3 Composition rules

- Level-0 atoms → force primitives (`physics/forces/_base.py`) → mode
  assemblies (`physics/forces/<mode>.py`) → engine
  (`simulation/engine.py`). Each level only composes the level below.
- One shared primitive, many composers: e.g. the curl-flow field is one
  Level-0 function with two composers (field mode gain 0.08, spatial
  mode gain 0.22 — [roadmap2.md](roadmap2.md) S2.A5/S2.B11); duplicate
  math is a defect.
- Force terms in composite modes are pure, named functions
  `(flock, cfg, cache) → (N,3)` registered in an ordered table and
  composed by reduction — per-term isolation for tests/benchmarks
  ([roadmap2.md](roadmap2.md) S2.A5 composition contract).

---

## 4. Protocol contracts

### 4.1 Configuration — `pymurmur/core/config.py`

`SimConfig` composes per-subsystem dataclasses. Target sections
(the current codebase implements most of these with `field_`-style
prefixed leaf names and a flat-access compatibility layer; the
retirement path and gaps are in [roadmap1.md](roadmap1.md) D1 and
[roadmap5.md](roadmap5.md)):

```python
@dataclass
class DomainConfig:
    width: float = 1000.0; height: float = 700.0; depth: float = 400.0

@dataclass
class BoundaryConfig:
    mode: str = "toroidal"            # toroidal | open | margin | sphere | sphere_soft
    sphere_radius: float = 300.0
    avoidance_factor: float = 0.05
    margin: float = 42.0
    use_toroidal_distance: bool = True

@dataclass
class FlockConfig:
    num_boids: int = 150; boid_size: float = 9.0
    v0: float = 4.0; max_force: float = 0.15
    visual_range: float = 70.0
    seed: int | None = None
    velocity_init: str = "fixed"       # fixed | cube | speed_uniform | tangential | drift
    position_init: str = "box"         # box | sphere | sphere_shell | gaussian | grid | blob
    speed_min_factor: float = 0.3      # promoted from the 0.3 hardcode in integrate()
    n_predators: int = 0               # species column

# one dataclass per mode / feature group:
@dataclass ProjectionConfig: phi_p, phi_a, sigma, refinements, steric, blind_deg,
                             anisotropy, max_visibility, max_occlusion_neighbors=64
@dataclass SpatialConfig:    separation_weight, alignment_weight, cohesion_weight,
                             noise_scale, acceleration_scale, separation_distance,
                             neighbor_filter, influence_count, alignment_radius_ratio,
                             separation_kernel, noise_mode, speed_mode,
                             flow_weight,     # grid-tier curl flow
                             parameter_jitter, jitter_separation/cohesion/alignment,
                             predator_* (boosts, escape_factor)
@dataclass FieldConfig:      unit_scale, chase_strength, shell_influence, target_pull,
                             drift_pull, tangent_pull, flow_pull, wave_gain, inertia,
                             separation, alignment, cohesion, flow,
                             disabled_terms   # per-term toggles
@dataclass VicsekConfig:     couplage, diffusion, time_step, velocity, radius_influence,
                             radius_avoid, radius_predators, weight_afraid,
                             predator_noise_ratio, detect_ratio, velocity_predator
@dataclass InfluencerConfig: rank_exponent, substeps, scale, influence_mode,
                             near_dist_sq, init, separation,
                             traj_primary_amp, traj_secondary_amp, traj_periods,
                             traj_phase, traj_z_bias,  # optional path-shaping
                             pilot_enabled, shell_radius, pilot_speed
@dataclass AngleConfig:      turn_rate, max_turn_rate, turn_threshold, jitter_deg,
                             margin, speed_mode, base_speed, neighbors,
                             sep/align/range_radius_bodies
@dataclass MarlConfig:       action_scale, velocity_cap, rule_weight,
                             separation_radius, episode_steps, reward_* weights
@dataclass ThreatConfig:     mode, radius, strength, momentum, acceleration,
                             split_gain, vacuole_strength, blackening_gain
@dataclass EcologyConfig:    roost, critical_mass, roosting_enabled,
                             seasonal_size, peak_size, predator_presence,
                             wander_*, ripple_*
@dataclass MetricsConfig:    detail_level, interval, bird_mass_kg, cruise_speed_ms,
                             acc_peak_ms2, readout_smooth, altitude_target,
                             history_cap    # ring-buffer cap (soak guard)
@dataclass VizConfig:        fps, window_width/height, theme, trails, trail_length,
                             point_sprites, per_bird_color, dual_view, background,
                             background_top/bottom,      # gradient sky colours
                             bird_mesh, flap_period,     # mesh registry + flap
                             show_grid, auto_rotate, hud
@dataclass CaptureConfig:    width, height, frames, every, fps, output, metrics_csv,
                             metrics_json, with_viz, sweep, prewarm
@dataclass PerfConfig:       use_numba, fastmath, num_threads, spatial_index,
                             instance_buffer_chunk, adaptive_quality, target_fps

@dataclass
class SimConfig:
    domain: DomainConfig; boundary: BoundaryConfig; flock: FlockConfig
    mode: str = "projection"
    projection: ProjectionConfig; spatial: SpatialConfig; field: FieldConfig
    vicsek: VicsekConfig; influencer: InfluencerConfig; angle: AngleConfig
    marl: MarlConfig; threat: ThreatConfig; ecology: EcologyConfig
    metrics: MetricsConfig; viz: VizConfig; capture: CaptureConfig; perf: PerfConfig
    # (all via field(default_factory=...))
```

**Loader contract** (`from_file`): YAML **section name = SimConfig field
name**; build each sub-config with
`cls(**{k: v for k, v in section.items() if k in sub_fields})`;
`warnings.warn` for unknown keys; scalars (`mode`) handled explicitly.
**No flattening — a `capture.width`-overwrites-`domain.width` collision
must be structurally impossible.** `to_file` emits the same nested
shape. `validate()` clamps ranges (σ ≥ 1, v0 > 0, weights ≥ 0,
**depth > 0** — strictly-3D) and runs after load.

**Live-mutation contract:** input handlers mutate sub-config fields in
place (`cfg.projection.phi_p += 0.01`); live-vs-static field tables
live in each sub-config's docstring. Every config leaf field must be
read by ≥ 1 non-config module (drift guard T1.2).

### 4.2 ForceMode protocol — `pymurmur/physics/forces/_mode.py`

```python
class ForceMode(ABC):
    name: ClassVar[str]
    needs_index: ClassVar[bool] = False        # spatial index rebuild wanted?
    speed_mode: ClassVar[str] = "band"         # band | fixed | ceiling | none
    owns_positions: ClassVar[bool] = False     # True → integrate(move=False)

    def reset(self, flock: PhysicsFlock, config: SimConfig) -> None: ...
    @abstractmethod
    def step(self, flock: PhysicsFlock, config: SimConfig, dt: float) -> None: ...

MODE_REGISTRY: dict[str, type[ForceMode]] = {}
def register(cls): MODE_REGISTRY[cls.name] = cls; return cls
```

Modes are **stateful instances**: per-mode state (influencer
`self.tick`, field `self.t` + group caches) lives on the instance,
created from the registry, re-instantiated on `config.mode` change or
`reset()`. All imports at module top — never inside per-bird loops. New
mode = one file + `@register`; the `M` key cycles
`sorted(MODE_REGISTRY)`. The `needs_index` flag replaces any hardcoded
mode-set. The declared `speed_mode` and `owns_positions` **must be
consumed by the engine** — an unconsumed flag is untruthful and fails
the mode-contract test.

### 4.3 Engine step — `pymurmur/simulation/engine.py`

`SimulationEngine` owns `self.mode`; `step()` is literally the Level-2
diagram:

```python
def step(self, dt, control=None):
    self._drain_commands()                                         # command queue
    ctx = StepContext(frame=self.frame, dt=dt, rng=self.flock.rng,
                      center=self.flock.center, config=self.config)
    self.extensions.pre_step(self.flock, ctx)
    if control is not None: self._apply_control(control)           # MARL/pilot hook
    if self.mode.needs_index: self.flock.rebuild_index(self.config)
    self.mode.step(self.flock, self.config, dt)
    self.flock.stash_accelerations()                               # metrics see pre-reset a
    integrate(self.flock, self.config, dt,
              speed_mode=self.mode.speed_mode,
              move=not self.mode.owns_positions, ...)
    self.flock.update_center()
    self.metrics.collect(self.flock, self.frame, ctx)
    self.frame += 1
```

`PhysicsFlock` never imports forces — pure state + index. DAG:
`engine → {modes, flock, extensions, metrics}`,
`modes → {flock(read), primitives, core}`.

### 4.4 Flock state contract — `pymurmur/physics/flock.py`

```python
self.rng: np.random.Generator            # seeded from config.flock.seed — the ONLY
                                          # randomness source anywhere
self.is_predator: (N,) bool              # species column; carried by
                                          # _extend/add_boids/remove_boids
self.center: (3,) float32                # smoothed centroid:
                                          # center += 0.5*(centroid - center) per step
self.prev_positions: (N,3) float32       # last wrapped positions (MSD unwrap,
                                          # ring trails, render interpolation)
self.last_accelerations: (N,3) float32   # stashed pre-integrate (physical metrics)
self.max_speed: (N,) float32 | None      # per-bird ceiling (panic; None = scalar v0)
```

`FlockArrays` is **composed** (`flock.arrays: FlockArrays`, attribute
properties forward) — the Level-0 state contract is real, not
decorative.

### 4.5 Integration contract — `pymurmur/physics/boid.py`

`integrate()` is the single motion authority:

```python
integrate(flock, config, dt, *,
          speed_mode="band",          # band | fixed | ceiling | none
          move=True,                  # False: boundary enforcement only
          inertia=0.0,                # v = lerp(v_raw, v_clamped, 1-inertia)
          noise_velocity=None)        # (N,3) additive post-integration noise or None
```

Band floor reads `cfg.flock.speed_min_factor`. Safety rails at callers:
the visualizer clamps `dt ∈ [0, 1/20]` behind a fixed-timestep
accumulator; the engine applies an `isfinite3` position guard
(offenders reset to `flock.center`). Zero-allocation rail per §3.2.
Boundary handlers: toroidal wrap; open; margin push; `sphere*`
**centred on the domain centre** (an origin-centred sphere makes every
bird permanently "outside" — a live bug, see
[roadmap5.md](roadmap5.md)).

### 4.6 SpatialIndex protocol — `pymurmur/core/types.py`

```python
class SpatialIndex(Protocol):
    def rebuild(self, positions, active, box: tuple | None) -> None: ...
    def query_knn(self, pos, k) -> np.ndarray: ...        # GLOBAL indices, closest-first
    def query_radius(self, pos, r) -> np.ndarray: ...     # GLOBAL indices
    def query_knn_batch(self, positions, k, workers=-1) -> np.ndarray: ...
```

- `KDTreeIndex`: maps compacted→global via `np.where(active)[0]`;
  passes `boxsize=box` when toroidal (activates
  `boundary.use_toroidal_distance`).
- `SpatialHashGrid`: modulo-wrapped cell keys; query cell range derived
  from the radius argument; k-NN selection on **squared** distances;
  optional incremental maintenance behind the same protocol.
- Modes consume `flock.index` (batch calls,
  `workers=config.perf.num_threads`); private `cKDTree` builds in
  `forces/` are forbidden.
- Pair vectors that cross the seam use `min_image`.

### 4.7 Extension protocol — `pymurmur/physics/extensions/_base.py`

```python
@dataclass
class StepContext:
    frame: int; dt: float
    rng: np.random.Generator
    center: np.ndarray
    config: SimConfig
    threat_prox: np.ndarray | None = None   # written by Threat, read by modes (blackening)

class Extension(ABC):
    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None: ...
```

`ExtensionManager.pre_step(flock, ctx)` re-reads `ctx.config` each
frame (live toggles work). Ecology exposes a **public read surface** —
`predator_active`, `hour`, `day`, `roost_position` properties — so HUD,
metrics and tests read simulated-calendar state without touching
privates. The Threat FSM replaces the simple predator under this
protocol.

### 4.8 Renderer contract — `pymurmur/viz/`

One schema, one VAO builder, layered passes:

```python
@dataclass
class InstanceSchema:                    # single source of truth
    floats: int = 8                      # pos.xyz, vel.xyz, flag, hue
    layout: str = "3f 3f 1f 1f/i"
    attrs = ("in_InstancePos", "in_InstanceVel", "in_Flag", "in_Hue")

class Renderer3D:
    def _build_vao(self): ...            # at init AND after every buffer growth
    def begin_frame(self, camera, viewport=None, fade=False): ...
    def draw_birds(self, flock): ...     # packs pos/vel/is_predator/seeds
    def draw_layer(self, name, ...): ... # threat/influencer markers, ring trails, HUD quads
```

Contract items: headless FBO **with depth attachment** (otherwise
captures render in draw order, not depth order); matrix uploads via
`_mat4_bytes = np.array(m.to_list(), np.float32).tobytes()` (PyGLM
memory-layout hazard); `theme` and mesh
(`tetra | winged | impostor | ellipsoid | cone | arrow | points`) from
`config.viz`; dual-view = two `(camera, viewport)` passes; HUD =
orthographic pixel-space pass at frame end. `Visualizer.frame()` is
**render-only** (the loop owner steps the simulation), so `Recorder`
reuses the Visualizer instead of hand-building a renderer.

### 4.9 Engine seams — control, commands, quality

1. **Control hook**: `step(dt, control: ndarray | None)` — scaled,
   clipped per-bird Δv (MARL, pilot, choreography).
2. **Command queue**: `engine.enqueue(cmd)` with
   `AddBoids(n) / RemoveBoids(n) / SpawnAt(pos, predator) / Reset /
   Clear`, drained at the top of `step()`. `input_control` translates
   keys/mouse into commands — simulation-lifecycle logic leaves the
   render loop; the headless path gains the same capabilities.
3. **Quality governor**: budget, spike cap, risk classifier, hysteresis
   ladder live in `analysis/perf.py` (pure logic); the Visualizer feeds
   timings and *applies* actions (dependency direction preserved).

### 4.10 Analysis tiers and export schema

- *Observables* (`metrics.py`, `rewards.py`): pure, importable by the
  engine tier. *Drivers* (`evoflock.py`, `phase_diagram.py`,
  `density_scaling.py`, `gym_env.py`, `perf.py`): sit above the engine,
  may import `simulation`.
- `FlockMetrics.to_dict()` (ndarray→list, numpy scalars→python) is the
  declared export interface; Recorder and the gym env consume it —
  never `snapshot().__dict__`.
- **Mode-gated observables export honestly**: fields only a specific
  mode populates (Θ/Θ′ from projection, `target_dist_*` from
  influencer) are `None`/absent in other modes — never stale zeros
  masquerading as measurements.
- Display smoothing (EMA readout) is **display-only** — export and
  analysis paths always carry raw per-frame values.

### 4.11 Public facade

`pymurmur/__init__.py` exports `SimConfig`, `SimulationEngine`,
`Simulation(**params)`, `Recorder`;
`SimulationEngine.benchmark(flock_size, num_steps)` returns per-step
timings.

---

## 5. File-location conventions

Code under `pymurmur/<subsystem>/…`; tests mirror the package: code in
`pymurmur/physics/forces/field.py` → tests in
`test/physics/forces/test_field.py`; golden data in `test/data/`;
dependency-gated examples in `scripts/`; presets in `conf/`. Markers in
`pytest.ini`: `@pytest.mark.gl` (auto-skip without a GL context),
`@pytest.mark.slow`, `@pytest.mark.golden`.

---

## 6. Index of work items

| IDs | Content | File |
|-----|---------|------|
| D0 | Safety net: goldens, architecture test, entry-point freeze, `fibonacci_sphere` | [roadmap1.md](roadmap1.md) |
| D1 | Nested configuration layer + migration + T2 config suite | [roadmap1.md](roadmap1.md) |
| D2 | ForceMode protocol & registry + T3.2 mode-contract suite | [roadmap1.md](roadmap1.md) |
| D3 | Flock state contract + T4.1–T4.3 matrices | [roadmap1.md](roadmap1.md) |
| D4 | Integration contract + T3.3 | [roadmap1.md](roadmap1.md) |
| D5 | SpatialIndex protocol + T3.1 conformance | [roadmap1.md](roadmap1.md) |
| D6 | Extension protocol widening + T3.4 | [roadmap1.md](roadmap1.md) |
| D7 | Renderer contract + T5 viz/capture suites | [roadmap1.md](roadmap1.md) |
| D8 | Engine seams (control, commands, quality) + T3.5/T6.1/T6.2 | [roadmap1.md](roadmap1.md) |
| D9 | Analysis split, facade, exports, cleanup, doc sync | [roadmap1.md](roadmap1.md) |
| T0 | Harness & fixtures (T0.1–T0.3) | [roadmap1.md](roadmap1.md) |
| T1 | Architecture & drift guards (T1.1–T1.5) | [roadmap1.md](roadmap1.md) |
| T4.4 | Metamorphic metric invariances | [roadmap1.md](roadmap1.md) |
| T6.3 | Soak / bounded-memory test | [roadmap1.md](roadmap1.md) |
| S1.1–S1.8 | Scientific correctness cluster (occlusion, Θ, δ̂, φn, kernels, steric, Vicsek, metric fixes) | [roadmap2.md](roadmap2.md) |
| S2.A1–A9 | Track A — field/blob mode + wander, ripples, threat FSM, blob presets | [roadmap2.md](roadmap2.md) |
| S2.B1–B11 | Track B — Reynolds variants, ecology, init variants, numba, grid flow | [roadmap2.md](roadmap2.md) |
| S2.C1–C8 | Track C — angle mode | [roadmap2.md](roadmap2.md) |
| S2.D1–D4 | Track D — Vicsek predator–prey | [roadmap2.md](roadmap2.md) |
| S2.E1–E6 | Track E — influencer (+ emergent-stretching signature test) | [roadmap2.md](roadmap2.md) |
| S3.1–S3.11 | Metrics & analysis suite (incl. S3.6a marginal opacity) | [roadmap3.md](roadmap3.md) |
| S4.1–S4.11 | Rendering & capture (incl. S4.4a mesh registry) | [roadmap3.md](roadmap3.md) |
| S5.1–S5.6 | UX & tooling (presets, title, HUD, interaction, CLI, logging) | [roadmap4.md](roadmap4.md) |
| S6.1–S6.6 | EvoFlock (SSGA, objectives, SDF obstacles, genes, protocol) | [roadmap4.md](roadmap4.md) |
| S7.1–S7.3 | MARL bridge (marl mode, gym env, train/rollout scripts) | [roadmap4.md](roadmap4.md) |
| — | Unified sequencing, effort table, definition of done | [roadmap4.md](roadmap4.md) |
| — | Codebase audit: defects, unclear items, dead fields | [roadmap5.md](roadmap5.md) |
| — | Improvement steps (prioritized) | [roadmap5.md](roadmap5.md) |
| App. A | Deliberately excluded scope (recorded mechanics/math) | [roadmap5.md](roadmap5.md) |
| App. B | Documentation-change checklist | [roadmap5.md](roadmap5.md) |
| App. C | Risk register | [roadmap5.md](roadmap5.md) |

### 6.1 Legacy P-phase numbering → item mapping

The codebase carries ~550 comments and test names using a retired
**P0–P14 phase numbering** (e.g. `# P8.5`, `test_p8_10_accumulator.py`,
preset headers "Roadmap ref: P3.10"). Those comments remain valid
pointers into this document set via this table — do not renumber the
code:

| P-phase | Content | Lives in |
|---|---|---|
| P0 | Foundations: goldens, arch-test skeleton, invariant fuzz, flock columns (rng/center/species/prev/stash/max_speed), integration variants + rails, capability probe, math atoms, H₂ inf fix, SDF primitives, position-init variants | D0/D3/D4 + T0 ([roadmap1.md](roadmap1.md)); S1.8, S2.B9, S6.4 ([roadmap2.md](roadmap2.md)/[roadmap4.md](roadmap4.md)) |
| P1 | Scientific correctness (occlusion, Θ union, δ̂, steric, kernels, Vicsek memory, thickness, Θ-N/A) | S1.1–S1.8 ([roadmap2.md](roadmap2.md)) |
| P2 | Contracts: nested config, ForceMode registry, SpatialIndex, StepContext/extensions, InstanceSchema/VAO, `_mat4_bytes`, holey-mask suite, ForceTerm/composeForces | D1/D2/D5/D6/D7 ([roadmap1.md](roadmap1.md)); S2.A5 composition contract ([roadmap2.md](roadmap2.md)) |
| P3 | Field/blob mode + threat FSM + wander/ripple + field presets | S2.A1–A9 ([roadmap2.md](roadmap2.md)) |
| P4 | Reynolds variants, predator species, physical metrics, jitter, ecology, numba kernels | S2.B1–B11 ([roadmap2.md](roadmap2.md)) |
| P5 | Angle mode | S2.C1–C8 ([roadmap2.md](roadmap2.md)) |
| P6 | Vicsek predator–prey | S2.D1–D4 ([roadmap2.md](roadmap2.md)) |
| P7 | Influencer (incl. P7.6 pilot mode) | S2.E1–E6 ([roadmap2.md](roadmap2.md)) |
| P8 | Rendering & capture (impostors, depth cues, trails, winged mesh, themes, governor, sweep, dual-view, mpl fallback, accumulator, density) | S4.1–S4.11 + D7 ([roadmap3.md](roadmap3.md)/[roadmap1.md](roadmap1.md)) |
| P9 | Metrics & analysis (nematic, MSD, hull-τρ, silhouette, m*, η(m), gyration, motion metrics, rewards, export) | S3.1–S3.11 ([roadmap3.md](roadmap3.md)) |
| P10 | UX & tooling (presets a–h,w, title, HUD, spawning, CLI/facade, φ-constraint) | S5.1–S5.5 + D8.2, S1.4 ([roadmap4.md](roadmap4.md)/[roadmap1.md](roadmap1.md)/[roadmap2.md](roadmap2.md)) |
| P11 | EvoFlock | S6.1–S6.6 ([roadmap4.md](roadmap4.md)) |
| P12 | MARL bridge | S7.1–S7.3 ([roadmap4.md](roadmap4.md)) |
| P13 | Scaling & performance (budgets, checkpoints, memory audit, soak, determinism matrix) | D8/T6 + T4.3 ([roadmap1.md](roadmap1.md)) |
| P14 | Guard rails (DAG matrix, config drift, 3D guard, doc drift, collection count) | T1.1–T1.5 ([roadmap1.md](roadmap1.md)) |

**Caution:** phase-completion claims attached to the retired P-scheme
("P1 ✅", "P8.6 wired ✅") are not authoritative — several are
contradicted by the code (see [roadmap5.md](roadmap5.md) §2 preamble).
The per-item Status lines in
[roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md) are the audit of
record.

---

## 7. Glossary (auxiliary reference)

Symbols and project terms used across the set, in their 3D-simulation
meaning.

**Observables.** `α` — polar order `‖Σ û_i‖/N ∈ [0,1]`; cannot
distinguish anti-parallel from isotropic. `S` — nematic order, largest
eigenvalue of the traceless Q-tensor
`Q_αβ = (1/N)Σ((3/2)û^α û^β − ½δ_αβ)`; invariant under `û→−û` and
SO(3). `Θ` — internal opacity: probabilistic union of neighbour solid
angles `Ω = 2π(1−cos α)`. `δ̂` — boundary-length-weighted mean of
visible-neighbour directions; `|δ̂|→1` at the flock edge, `→0` inside.
`Θ′` — external opacity (3D voxel occupancy; the 2D silhouette variant
is the codebase's **only** 2D projection, diagnostic-only, never fed
back into simulation). `H₂` — consensus robustness from the k-NN graph
Laplacian spectrum (`inf` when disconnected). `η(m)` — marginal
robustness per added neighbour. `R_g` — robust gyration radius (median
centroid, top-15 % trim). `m*` — shape-driven suggested neighbour
count. `MSD(τ)` — mean squared displacement over unwrapped positions;
slope 2 ballistic, 1 diffusive. `hull-τρ` — convex-hull density
autocorrelation time. `CoM` — instantaneous centroid; distinct from
`flock.center` (EMA-smoothed).

**Physics/config.** `v0` — cruise speed; band mode clamps to
`[speed_min_factor·v0, v0]`. `σ` — projection-mode topological
neighbour count. `φp/φa/φn` — projection weights, `φp+φa ≤ 1`,
`φn = 1−φp−φa`. `dt` — physics timestep, clamped `[0, 1/20]` behind
the fixed-timestep accumulator. SoA — structure-of-arrays layout.
SDF/CSG — signed distance functions and their min/max composition.
min-image — per-axis toroidal shortest displacement. Rodrigues —
axis-angle rotation atom. Lissajous — incommensurate-frequency
parametric curves (blob anchors, wander, influencer target).

**Project concepts.** *Holey mask* — `flock.active` may have arbitrary
`False` holes; every assembly must be correct under it. *Golden
trajectory* — pinned seeded run (`test/data/golden_<mode>.npz`,
`atol=1e-3`); deliberate physics changes re-pin in the same commit.
*Composer* — the assembly consuming an atom; atoms without composers
are deleted. *ForceMode / ForceTerm / StepContext / Extension /
SpatialIndex / InstanceSchema / QualityGovernor / MurmurationEnv* — the
protocol contracts of §4. *Impostor* — camera-facing quad with a
sphere-shaded disc fragment. *Alpha-accumulation* — low-α, no
depth-write density rendering. *Dual-view* — two (camera, viewport)
passes per frame.
