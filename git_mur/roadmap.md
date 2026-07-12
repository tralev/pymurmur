# roadmap.md — Unified implementation roadmap (design · tests · science)

**Supersedes and merges** `design_roadmap.md`, `test_roadmap.md`,
`sci_roadmap.md`, `test_sci_roadmap.md` (retired). Target architecture:
[`arch.md`](arch.md). Provenance for every item: the audit corpus in
**`sci/`**. Inline citation shorthand used throughout resolves there:
*todo_claude* → `sci/todo_claude.md`, *todo_claude1/2* →
`sci/todo_claude1.md` / `sci/todo_claude2.md`, *git0…git7* →
`sci/todo_claude_git0.md` … `sci/todo_claude_git7.md` (per-source expanded
specs, cited as *[→gitN Rx]*). The consolidated 27-section science reference
(`resulting_sci.md`) was fully absorbed — its coverage was verified
section-by-section against this roadmap and the codebase, so it has been
retired; its per-source detail survives in the `todo_claude_sci1…11.md`
audit files.

**Decisions taken (user-confirmed):**
1. **Nested config** — `SimConfig` composed of per-subsystem dataclasses.
2. **ForceMode classes** — modes are small stateful classes behind a protocol.
3. **Scope** — core physics/metrics/rendering/UX + EvoFlock + MARL bridge
   (excluded tiers in Appendix B).
4. **Presets honor documented intent** — YAML values load as written;
   goldens re-pinned; behavior change release-noted.

**Non-negotiable goals, enforced by tests in this file:**
- *Strictly 3D:* every vector is `(…,3)`; no 2D fallbacks; z-up world;
  guard tests T1.3.
- *Highly modular:* new capability = new module/class behind an existing
  protocol (ForceMode / Extension / SpatialIndex / render layer / metric /
  reward term / SDF / preset) — never `if mode == ...` branches in shared
  code. Extension-points table: `arch.md` §12.
- *Clean Macro→Micro:* the six-subsystem decomposition stays; every config
  field, protocol and dependency rule traceable doc ↔ code (import matrix
  T1.1, config-drift T1.2, doc-drift T1.4 fail CI on divergence).
- *Clean Micro→Macro:* Level-0 primitives stay pure functions on arrays;
  assemblies compose them; composition is a **DAG** (the flock↔forces cycle
  dies in D2); no component ships without its composer (no dead atoms).

**How to read.** Part I = architecture phases **D0–D9** (each with code
sketches, file targets, acceptance). Part II = test infrastructure
**T0–T6** (harness, fixtures, contract suites). Part III = science
portfolio **S1–S7**; each item gives *math* (3D form), *impl* (module /
class / config path / linkage), and *tests* (concrete assertions + test
file). Part IV = unified sequencing. Appendices: traceability, exclusions,
doc-update checklist.

**File-location conventions.** Code under `pymurmur/<subsystem>/…`; tests
mirror the package: code in `pymurmur/physics/forces/field.py` → tests in
`test/physics/forces/test_field.py`; golden data in `test/data/`;
dependency-gated examples in `scripts/`; presets in `conf/`. Markers in
`pytest.ini`: `@pytest.mark.gl` (auto-skip without a GL context),
`@pytest.mark.slow`, `@pytest.mark.golden`.

**Global helpers assumed once landed (D-phases):** `flock.rng` (single
seeded `np.random.Generator`), `flock.center`, `flock.is_predator`,
`flock.prev_positions`, `flock.last_accelerations`, `flock.max_speed`,
`min_image(Δ, box) = Δ − box·round(Δ/box)`,
`rotate_about(v, k̂, θ) = v cosθ + (k̂×v) sinθ + k̂(k̂·v)(1−cosθ)` (Rodrigues),
`hash01(x) = fract(sin(x·12.9898)·43758.5453)`,
`smoothstep(a,b,x) = t²(3−2t), t = clamp((x−a)/(b−a), 0, 1)`,
`fibonacci_sphere(n)` (golden-angle near-uniform unit vectors, Level-0 atom —
see D0.4), `normalize3` (0-safe), `limit3(v, m)`. All in
`pymurmur/core/types.py`, unit-tested before first use.

---

# Part I — Architecture foundation (D0–D9)

## D0 — Safety net before any refactor  *(½ day)*

1. **Golden trajectories** per existing mode: seeded 15-bird × 30-frame
   runs → `test/data/golden_<mode>.npz` (positions+velocities float32),
   `assert_allclose(atol=1e-3)` — tolerance, not bit-exactness (libm ulp
   differences amplify across platforms). Regeneration script
   `test/regenerate_golden.py`; policy: a deliberate physics change re-pins
   **in the same commit**, CI fails otherwise. (Full harness: T0.2.)
2. **Architecture test** (`test/test_architecture.py`): the D9 dependency
   matrix as data; AST-walk every module (function-level imports included);
   fail on any edge outside it. (Full spec: T1.1.)
3. **Entry-point freeze:** smoke tests — `--no-viz` runs 50 frames for each
   preset without crashing.
4. **`fibonacci_sphere(n)` Level-0 atom** *(predecessor audit)* — seed the
   `core/types.py` helper set early (S2.A3's stratified shells and isotropic
   test fixtures *compose* it rather than re-derive the golden angle inline —
   Micro→Macro: shared atom, one definition):
   `ga = π(3−√5) ≈ 2.399963`; `y_i = 1 − 2(i+0.5)/n`, `r_i = √(1−y_i²)`,
   `θ_i = ga·i`; row `(cos θ_i·r_i, y_i, sin θ_i·r_i)` → `(n,3) float32`,
   every row unit-length (z-up: swap axes). *tests* (`test/core/test_types.py`):
   all rows norm 1 ± 1e-6; n=0 → shape (0,3); n=1 → one unit vector (no NaN);
   n=256 → each of the 8 octants holds ≥ 1 point (coarse uniformity).

**Accept:** suite green; goldens committed; import matrix enforced;
`fibonacci_sphere` atom tested.

## D1 — Nested configuration layer  *(2 days; the enabling refactor)*

**Problem (todo_claude1 §1/§6, git0 F0):** one flat dataclass approaching
150 fields; the YAML loader flattens section keys unprefixed — whole
sections are silently dropped and `capture.width` **overwrites the domain
width** in every shipped preset.

**Target shape** (`pymurmur/core/config.py`):

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
    velocity_init: str = "fixed"       # fixed | cube | speed_uniform | tangential
    speed_min_factor: float = 0.3      # promoted from the 0.3 hardcode in integrate()
    n_predators: int = 0               # species column (D3)

# one dataclass per mode / feature group:
@dataclass ProjectionConfig: phi_p, phi_a, sigma, refinements, steric, blind_deg,
                             anisotropy, max_visibility, max_occlusion_neighbors=64
@dataclass SpatialConfig:    separation_weight, alignment_weight, cohesion_weight,
                             noise_scale, acceleration_scale, separation_distance,
                             neighbor_filter, influence_count, alignment_radius_ratio,
                             separation_kernel, noise_mode, speed_mode,
                             parameter_jitter, jitter_separation/cohesion/alignment,
                             predator_* (boosts, escape_factor)
@dataclass FieldConfig:      unit_scale, chase_strength, shell_influence, target_pull,
                             drift_pull, tangent_pull, flow_pull, wave_gain, inertia,
                             separation, alignment, cohesion, flow
@dataclass VicsekConfig:     couplage, diffusion, time_step, velocity, radius_influence,
                             radius_avoid, radius_predators, weight_afraid,
                             predator_noise_ratio, detect_ratio, velocity_predator
@dataclass InfluencerConfig: rank_exponent, substeps, scale, influence_mode,
                             near_dist_sq, init, separation,
                             traj_primary_amp, traj_secondary_amp, traj_periods,
                             traj_phase, traj_z_bias   # optional path-shaping (S2.E1)
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
                             acc_peak_ms2, readout_smooth, altitude_target
@dataclass VizConfig:        fps, window_width/height, theme, trails, trail_length,
                             point_sprites, per_bird_color, dual_view, background,
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

**Loader** (`from_file`): YAML **section name = SimConfig field name**;
build each sub-config with
`cls(**{k: v for k, v in section.items() if k in sub_fields})`,
`warnings.warn` for unknown keys; scalars (`mode`) handled explicitly.
**No flattening — the collision bug is structurally impossible.** `to_file`
emits the same nested shape. `validate()` clamps ranges
(σ ≥ 1, v0 > 0, weights ≥ 0, depth > 0 — strictly-3D) and runs after load.

**Migration mechanics (reviewable diff):**
1. Introduce nested classes; keep temporary **property shims** on
   `SimConfig` for every legacy flat name (`phi_p → self.projection.phi_p`,
   `capture_width → self.capture.width` …) so call sites keep working.
2. Migrate call sites package-by-package to dotted access
   (physics → viz → capture → analysis), deleting each shim as its last
   user converts.
3. Rewrite `conf/*.yaml` to the nested schema.
4. Delete the shim block; test asserts no legacy names remain.

**Live-mutation contract:** input handlers mutate sub-config fields in
place (`cfg.projection.phi_p += 0.01`); live-vs-static field tables live in
each sub-config's docstring.

**Tests (T2, `test/core/test_config.py`):**
1. Round-trip `to_file → from_file` equality on every field, nested.
2. Preset survival (parametrized over `conf/*.yaml`): `domain.*` match the
   file's `domain:` section (the historical corruption case) + one sentinel
   per section (e.g. `vicsek.couplage`) equals the YAML value.
3. Unknown keys warn (`pytest.warns`); known keys don't.
4. Validation: σ = −3, v0 = 0, depth = 0 raise/clamp per contract.
5. Live-mutation smoke: mutate `cfg.projection.phi_p` mid-run → next step
   reads it.
6. Shim retirement: `not hasattr(SimConfig(), "phi_p")` after step 4.

**Accept:** all presets load with correct domains; round-trip green; zero
legacy flat-field access; `arch.md` §2.1 updated to the nested contract.

## D2 — ForceMode protocol and registry  *(2 days; breaks the E-cycle)*

**Problem (todo_claude1 §6, todo_claude2 §5, git0 F2/F3):** modes are
stateless `(flock, config)` functions with nowhere to keep time/caches;
`PhysicsFlock.step` lazily imports `forces` (a Level-1 cycle); vicsek and
influencer need to own speed/positions and today fight `integrate()`.

**Target** (`pymurmur/physics/forces/_mode.py`):

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

- The five existing modes become `@register class SpatialMode(ForceMode)` …
  — mechanical wraps; per-mode state becomes instance attributes
  (influencer `self.tick`, field `self.t` + group cache). Lazy imports move
  to module top (never inside per-bird loops — todo_claude2 §2 hygiene).
- **Orchestration moves up:** `SimulationEngine` owns `self.mode`
  (instantiated from the registry; re-instantiated on `config.mode` change
  or `reset()`); `step()` becomes literally the Level-2 diagram:

```python
def step(self, dt, control=None):
    self._drain_commands()                                         # D8
    ctx = StepContext(frame=self.frame, dt=dt, rng=self.flock.rng,
                      center=self.flock.center, config=self.config)
    self.extensions.pre_step(self.flock, ctx)
    if control is not None: self._apply_control(control)           # D8
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

- `PhysicsFlock` **no longer imports forces** — pure state + index. DAG:
  `engine → {modes, flock, extensions, metrics}`,
  `modes → {flock(read), primitives, core}`.
- New mode = one file + `@register` (`M` cycles `sorted(MODE_REGISTRY)`);
  `needs_index` replaces the hardcoded `_INDEX_MODES` set.

**Tests (T3.2, `test/physics/forces/test_mode_contract.py`,
parametrized over `MODE_REGISTRY.values()`):** registered & instantiable;
step respects the active mask (inactive rows untouched); reset + same seed
→ same trajectory; `speed_mode` honored (T0.3 speed contract);
AST: no module-level `np.random.*` in mode sources; `owns_positions` flag
truthful (move=False modes actually moved birds).

**Accept:** import cycle gone (architecture test); all five modes green
against goldens (pure refactor); `M` cycles registry order; `arch.md` §2.2
updated to the class protocol.

## D3 — Flock state contract  *(1 day)*

`PhysicsFlock` gains the columns the portfolio needs, as **one documented
contract** (todo_claude2 §1's lesson: write conventions down):

```python
self.rng: np.random.Generator            # seeded from config.flock.seed — the ONLY
                                          # randomness source anywhere (git0 F1)
self.is_predator: (N,) bool              # species column (git0 F6); carried by
                                          # _extend/add_boids/remove_boids
self.center: (3,) float32                # smoothed centroid (git0 F8):
                                          # center += 0.5*(centroid - center) per step
self.prev_positions: (N,3) float32       # last wrapped positions (MSD unwrap,
                                          # ring trails, render interpolation)
self.last_accelerations: (N,3) float32   # stashed pre-integrate (physical metrics)
self.max_speed: (N,) float32 | None      # per-bird ceiling (panic; None = scalar v0)
```

**Array conventions (docstring in `core/types.py`):** all neighbour indices
are **global capacity-space** rows; every primitive states its index space;
`active` may have holes at any time — every mode/metric must be correct
under a holey mask. `FlockArrays` is **composed**
(`flock.arrays: FlockArrays`, attribute properties forward) — the Level-0
contract becomes real, not decorative.

**Tests (T4):**
- *T4.1 holey-mask matrix:* every mode × `holey_flock` fixture × 20 steps —
  no exception, invariants hold, inactive rows untouched, force clamps
  applied to active rows only.
- *T4.2 lifecycle:* `add_boids` past capacity (`_extend`), `remove_boids`,
  `SpawnAt`, `Clear` — species column and `seeds` carried; index rebuilt;
  metrics survive `N_active == 0`.
- *T4.3 determinism matrix:* same seed → bit-identical positions after 100
  steps, for (mode × `num_threads ∈ {1,-1}` × jitter on/off × numba on/off
  once S2.B10 lands); two in-process runs + one subprocess run.

**Accept:** matrices green; seeds-based features read documented semantics.

## D4 — Integration contract  *(½ day)*

`integrate()` (`physics/boid.py`) becomes the single motion authority:

```python
integrate(flock, config, dt, *,
          speed_mode="band",          # band | fixed | ceiling | none
          move=True,                  # False: boundary enforcement only
          inertia=0.0,                # v = lerp(v_raw, v_clamped, 1-inertia)
          noise_velocity=None)        # (N,3) additive post-integration noise or None
```

Band floor reads `cfg.flock.speed_min_factor`. Safety rails at callers:
visualizer clamps `dt ∈ [0, 1/20]` behind a fixed-timestep accumulator;
engine applies an `np.isfinite` position guard (offenders reset to
`flock.center`). Boundary handlers keep their shapes but `sphere*` centres
on the **domain centre** (bug fix — currently origin-centred, so every
bird is permanently "outside").

**Tests (T3.3, `test/physics/test_boid.py`):** all
(speed_mode × move × inertia ∈ {0, 0.8}) combinations vs hand-computed
expectations; per-bird `max_speed` respected; `speed_min_factor` honored;
toroidal wrap exactness; margin containment; sphere centred on C
(regression); NaN guard heals.

**Accept:** matrix green; centre-initialised flock stays centred in sphere
mode; NaN injection self-heals. *(Physics-visible: re-pin sphere golden.)*

## D5 — SpatialIndex protocol  *(1 day)*

**Problem (todo_claude2 §1):** the two indexes disagree on index space
(hash grid returns global rows, KDTree returns compacted-active rows) and
neither is wrap-aware; force modes bypass them and build private trees.

**Target** (`core/types.py`):

```python
class SpatialIndex(Protocol):
    def rebuild(self, positions, active, box: tuple | None) -> None: ...
    def query_knn(self, pos, k) -> np.ndarray: ...        # GLOBAL indices, closest-first
    def query_radius(self, pos, r) -> np.ndarray: ...     # GLOBAL indices
    def query_knn_batch(self, positions, k, workers=-1) -> np.ndarray: ...
```

- `KDTreeIndex`: maps compacted→global via `np.where(active)[0]` (bug fix);
  passes `boxsize=box` when toroidal (activates the dead
  `boundary.use_toroidal_distance`).
- `SpatialHashGrid`: modulo-wrapped cell keys (predecessor parity,
  todo_claude E11); query cell range derived from the radius (currently
  ignored); k-NN selection on **squared** distances; optional incremental
  maintenance (S2.C6) behind the same protocol.
- Modes consume `flock.index` (batch calls,
  `workers=config.perf.num_threads`); private `cKDTree` builds in `forces/`
  deleted (removes the double-build).
- Pair vectors that cross the seam use `min_image` (corrected form —
  `d = min(|Δ|, L−|Δ|)` per axis; the rystrauss source's signed-dx bug is
  the cautionary tale).

**Tests (T3.1, `test/physics/test_index_contract.py`, parametrized over
both implementations, on the holey fixture):** returns global indices;
knn closest-first, no self, no dupes; radius complete vs brute force;
radius argument honored (grid); toroidal cross-seam (x=1 vs x=W−1 mutual at
r=3); implementations interchangeable (identical neighbour sets); batch ==
per-row.

**Accept:** conformance suite green on both; zero `cKDTree(` constructions
inside `forces/` (grep test). *(Physics-visible: re-pin affected goldens.)*

## D6 — Extension protocol widening  *(½ day)*

**Problem (todo_claude1 §6):** `Extension.apply(flock)` has no config, dt,
frame, or rng — why predator/ecology params are dead and dt is hardcoded;
manager builds its list once, so live toggles (T/K) do nothing.

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

`ExtensionManager.pre_step(flock, ctx)` re-reads `ctx.config` each frame
(live toggles work); Ecology's `_predator_active` becomes a public
property; the Threat FSM (S2.A8) replaces `Predator` under this protocol.

**Tests (T3.4):** extensions receive real dt/frame/rng; flipping
`threat.mode` / `ecology.roosting_enabled` mid-run takes effect next frame;
`threat_prox` published shape `(N,)` in [0,1]; `predator_active` public.

## D7 — Renderer contract  *(1 day)*

One schema, one VAO builder, layered passes (git0 F5/W10; todo_claude
E1–E3):

```python
@dataclass
class InstanceSchema:                    # single source of truth
    floats: int = 8                      # pos.xyz, vel.xyz, flag, hue
    layout: str = "3f 3f 1f 1f/i"
    attrs = ("in_InstancePos", "in_InstanceVel", "in_Flag", "in_Hue")

class Renderer3D:
    def _build_vao(self): ...            # at init AND after every buffer growth (E2)
    def begin_frame(self, camera, viewport=None, fade=False): ...
    def draw_birds(self, flock): ...     # packs pos/vel/is_predator/seeds
    def draw_layer(self, name, ...): ... # threat marker, ring trails, HUD quads
```

Contract items: headless FBO **with depth attachment** (E1 — captures
currently render in draw order); matrix uploads via `_mat4_bytes` =
`np.array(m.to_list(), np.float32).tobytes()` (E3, macOS PyGLM layout
hazard); `theme` and mesh (`tetra|winged|impostor|ellipsoid|cone|arrow`,
S4.4a) from `config.viz`;
dual-view = two `(camera, viewport)` passes; HUD = orthographic pixel-space
pass at frame end. `Visualizer.frame()` becomes **render-only** (it
currently steps the simulation — the loop owner steps), letting `Recorder`
reuse it instead of hand-building a renderer (todo_claude1 §5).

**Tests (T5, `test/viz/`, `@gl` except 5.4):**
1. FBO has a depth attachment; VAO rebuilt after growth (grow past chunk →
   draw → nearest bird's colour wins at overlap); schema packs flag/hue;
   `_mat4_bytes` equals `to_list()` reference.
2. Render purity: `headless_frame()` twice → `sim.frame` unchanged;
   Recorder path reuses Visualizer.
3. Mode smoke: each mesh × each trail mode × dual-view renders one frame
   without GL errors; loose screenshot-hash regression.
4. *(no gl)* GPU-free capture: with moderngl monkeypatch-failing,
   `--no-viz --capture` produces ≥ 1 GIF frame via the matplotlib fallback
   with an asserted warning (replaces today's silent `except: pass`).
5. Capture pipeline: pre-warm frames not captured; sweep moves the camera
   between captured frames; env-override precedence YAML < env < CLI; GIF
   saved `optimize=True, disposal=2`; `capture.width/height` honored.

## D8 — Engine seams: control, commands, quality  *(1 day)*

1. **Control hook** (git0 F7): `step(dt, control: ndarray | None)` —
   scaled, clipped per-bird Δv (MARL, pilot, choreography).
2. **Command queue** (todo_claude1 §2): `engine.enqueue(cmd)` with
   `AddBoids(n) / RemoveBoids(n) / SpawnAt(pos, predator) / Reset / Clear`,
   drained at the top of `step()`. `input_control` translates keys/mouse
   into commands — simulation-lifecycle logic leaves the render loop; the
   headless path gains the same capabilities.
3. **Quality governor** (git6 R11): `PerfDiagnostics` grows budget, spike
   cap, risk classifier, hysteresis ladder (S4.10); Visualizer feeds
   timings and applies actions. Governor logic in `analysis/perf.py`;
   *application* is viz-side (dependency direction preserved).

**Tests (T3.5, `test/simulation/test_engine.py`; T6):** commands drained at
step start, headless; `step(control=0)` bit-identical to `step()`; control
clipping bounds; queue survives interleaving with reset; metrics survive
`N_active == 0`; governor pure-logic ladder test (T6.2);
`benchmark(num_steps=50)` returns 50 positive floats (T6.1) + `@slow`
per-mode step-time bounds at N = 2 000 (×3 headroom) and the arch.md
scaling checkpoints once S2.B10 lands.

## D9 — Analysis split, facade, exports, cleanup, doc sync  *(1 day)*

1. **Analysis tiers (todo_claude1 §4):** *observables*
   (`metrics.py`, `rewards.py`: pure, imported by B) vs *drivers*
   (`evoflock.py`, `phase_diagram.py`, `density_scaling.py`, `gym_env.py`,
   `perf.py`: sit above B, may import `simulation`).
2. **Metrics export schema (todo_claude1 §5):** `FlockMetrics.to_dict()`
   (ndarray→list, numpy scalars→python) as the declared interface;
   Recorder and gym env consume it — no more `snapshot().__dict__`.
   *Test (T3.6):* `to_dict()` JSON round-trips; key set pinned; Recorder
   CSV headers equal the schema.
3. **Public facade:** `pymurmur/__init__.py` exports `SimConfig`,
   `SimulationEngine`, `Simulation(**params)`, `Recorder`;
   `SimulationEngine.benchmark(flock_size, num_steps)`.
4. **Dead-code retirement (todo_claude2 §4):** delete `BoidView`; the
   `ForceKernel` type is fulfilled by S2.B10's numba kernels or deleted;
   legacy config shims removed; every remaining atom has a composer.
5. **Dependency matrix** (enforced by the architecture test):

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

6. **Docs to change (see Appendix C):** `arch.md` module map + rules +
   force-mode table (generated from `MODE_REGISTRY`) + §2 two-view
   summaries kept in sync (the `functional_*.md` companions are retired —
   `arch.md` is the single reference); add the doc-drift test T1.4;
   retire/slim `test.md` (superseded by this file).

**Accept:** architecture test enforces the matrix;
`pymurmur.Simulation(num_boids=50).benchmark(num_steps=10)` works;
doc-drift test green; grep finds no `BoidView`.

---

# Part II — Test infrastructure (T0–T6)

*(Feature tests live inline with their S-items in Part III; this part is
the harness they stand on.)*

## T0 — Harness & fixtures  *(with D0; 1 day)*

**T0.1 Shared fixtures (`test/conftest.py`):**

```python
@pytest.fixture
def cfg():                     # small deterministic config
    c = SimConfig(); c.flock.num_boids = 40; c.flock.seed = 7; return c

@pytest.fixture
def engine(cfg): return SimulationEngine(cfg)

@pytest.fixture
def holey_flock(cfg):          # THE composition fixture: 100 birds, 20 inactive mid-array
    cfg.flock.num_boids = 100
    f = PhysicsFlock(cfg)
    f.active[np.arange(15, 95, 4)] = False
    return f

@pytest.fixture
def rng(): return np.random.default_rng(0)

def make_engine(mode, n=30, seed=7, **overrides) -> SimulationEngine: ...
def run_steps(engine, k) -> tuple[np.ndarray, np.ndarray]:  # positions, velocities
```

**T0.2 Golden harness** — as D0.1; parametrized
`test_golden.py::test_matches_golden[mode]` over `MODE_REGISTRY`.

**T0.3 Invariant fuzz** (`test/physics/test_invariants.py`) — parametrized
(mode × boundary), 200 random seeded states each: no NaN after 50 steps;
toroidal positions in `[0, L)` every axis; speed contract per mode
(band within `[min_factor·v0−ε, v0+ε]`; fixed `== v0 ± 1e-5`; ceiling
`≤ v0`); inactive rows bit-identical across steps.

## T1 — Architecture & drift guards  *(with D0/D9; 1 day)*

- **T1.1 Import-rule matrix** (`test/test_architecture.py`): D9 matrix as
  data; AST walk (function-level imports too); named regression edges
  (`physics.flock !→ physics.forces`; `viz.input_control !→ simulation`;
  no `cKDTree(` in `forces/`; no module-level `np.random.*`).
- **T1.2 Config-usage drift:** every `SimConfig` leaf field (recursed) must
  be read by ≥ 1 non-config module (AST attribute-access scan); fail with
  the orphan list. Retro-covers the ~20 dead fields the audits found.
- **T1.3 Strictly-3D guard:** AST scan for `(…, 2)`-shaped spatial arrays
  in `physics/` → fail; `validate()` enforces `depth > 0`; invariance tests
  use random SO(3), not z-only.
- **T1.4 Doc-drift** (`test/test_docs.py`): every module path named in
  `arch.md` exists; every intra-repo markdown link in `arch.md` and
  `roadmap.md` resolves (GitHub slug rules); the `arch.md` force-mode table
  lists exactly `sorted(MODE_REGISTRY)`.
- **T1.5 Collection-count guard:** collected-test count pinned per
  subpackage (update deliberately).

## T2 — Config suite — folded into D1 (above).
## T3 — Contract suites — folded into D2–D6, D8, D9.2 (above).
## T4 — Composition/determinism — folded into D3 (above); plus:

**T4.4 Metamorphic invariances**
(`test/analysis/test_metrics_invariance.py`) — random flocks, 30 trials:
order parameter & nematic S invariant under random SO(3) rotation;
dispersion/gyration translation-invariant; `[0,1]`-metrics within bounds;
permutation invariance (shuffling bird order changes nothing).

## T5 — Viz/capture suites — folded into D7 (above).
## T6 — Perf/quality guards — folded into D8 (above).

---

# Part III — Science portfolio (S1–S7)

Each item: *math* → *impl* (file, class, config path, linkage) → *tests*
(assertions; test file per the mirror convention). Design-phase gates:
S1 needs D0–D2; S2 needs D1–D6; S3 needs D3; S4 needs D7; S5 needs D8;
S6 needs S2-B + S6.4; S7 needs D8.

## S1 — Scientific correctness cluster  *(≈2 days; re-pin goldens)*

**S1.1 Occlusion culling** *(todo_claude §1/§4)* — *math:* closest-first
sweep; j visible iff no accepted nearer k has `d̂_j·d̂_k ≥ cos α_k`; exact
`α = asin(min(b_eff/d, 1))`; candidates pre-filtered by
`cfg.projection.max_visibility`, capped at nearest
`cfg.projection.max_occlusion_neighbors` (64). *impl:*
`physics/occlusion.py::spherical_cap_occlusion` (signature stable;
array-native hot path — no per-neighbour `SimpleNamespace`; steric import
at module top). *tests* (`test/physics/test_occlusion.py`): collinear
(30,0,0),(60,0,0),(90,0,0) → `visible == [0]` (fix the legacy test
asserting `[2,1,0]` — it enshrines the bug); separated axes → all 3
visible; property (100 random configs): closest-first, duplicate-free, no
bird inside a nearer accepted cap, none in the blind cone, count ≤ cap.
**S1.1a Anisotropy identity** *(todo_claude T4)*: `anisotropy=1.0` vs
default → identical `(δ̂, visible, Θ)`.

**S1.2 Θ probabilistic union** *(todo_claude §2)* — *math:*
`Ω_j = 2π(1−cos α_j)`, `Θ = 1 − Π_visible(1 − Ω_j/4π)` (running product).
*tests:* `Θ(two separated caps) = 1−(1−Ω/4π)²` to 1e-6; metamorphic
sub-additivity `Θ₁ < Θ₁₂ < Θ₁+Θ₂`, monotone in neighbour count;
Θ ∈ [0,1] always.

**S1.3 δ̂ boundary-length weighting** *(todo_claude §3)* — *math:*
`δ̂ = Σ sin α_j d̂_j / Σ sin α_j`, no magnitude clamp; |δ̂| *is* the density
signal. *tests:* octahedral surround → |δ̂| < 1e-2; single neighbour →
|δ̂| = 1 ± 1e-6; property |δ̂| ≤ 1; rotation-equivariance
δ̂(Rp) = R δ̂(p) for random SO(3) R.

**S1.4 Pearce noise term φn + weight constraint** *(git0 W1.4;
todo_claude E5)* — *math:* `v ∝ φp δ̂ + φa ⟨v̂⟩_σ + φn η̂`,
`φn = 1 − φp − φa`, η̂ uniform on S² from `flock.rng`; input handler
renormalises the pair. *impl:* `ProjectionMode.step`;
`viz/input_control.py`. *tests:* hammer φp↑ 100× → `φp+φa ≤ 1` always
(and symmetric); behavioural: φn = 0.2 keeps residual heading variance >
φn = 0 (same seed).

**S1.5 Force-kernel corrections** *(git0 W3.1)* — *math:* separation
`Σ r̂/d²` (unit vector — current is 1/d); cohesion `normalize(p̄−p_i)`
(currently unbounded); `noise_force` output ×scale (δ currently
discarded). *impl:* `physics/forces/_base.py`. *tests*
(`test/physics/forces/test_kernels.py`): one neighbour at d = 2 →
separation magnitude 1/4; cohesion magnitude ≤ 1 always; noise mean
magnitude ≈ scale (±10 %, 10⁴ draws), zero at scale 0.

**S1.6 Steric clamp** *(todo_claude §14)* — *math:* `‖F‖ ≤ max_force`
after the 1/d² sum. *impl:* `physics/steric.py` (`max_force` param; caller
passes `cfg.flock.max_force`). *test:* pair at d = 0.01, strength 0.6,
cap 0.15 → ‖F‖ == 0.15 exactly.

**S1.7 Vicsek update corrected** *(git2 R1)* — *math:*
`û_noisy = normalize(û_old + √(2DΔt)·n_⊥)` — the missing **memory term**;
tangent-plane noise `n_⊥ = g − (g·û)û, g ~ N(0, I₃)` (in 3D, isotropic
noise biases angular diffusion — project it out);
`û_new = normalize(η û_target + (1−η) û_noisy)`; constant speed via
`speed_mode="fixed"`; `cfg.vicsek.time_step` live. *impl:*
`physics/forces/vicsek.py::VicsekMode`. *tests*
(`test/physics/forces/test_vicsek_core.py`): lone-bird heading
autocorrelation `⟨û_t·û_{t+1}⟩` > 0.99 at D = 0.01 and < 0.5 at D = 4
(memory live); D ∈ {0.1, 2.0} at η = 0.7 → settled α differs > 0.3
(D live); |v| == `cfg.vicsek.velocity` ± 1e-5 every frame; property
`|n_⊥·û| < 1e-6`.

**S1.8 Metrics formula fixes** *(todo_claude §10-prereq/E13;
git0 W9.1/9.3)* — thickness `= √(λ₃/λ₁) ∈ (0,1]` (currently `√(λ₂/λ₃)`);
`compute_h2` disconnected → `inf` (currently 0.0 — conflates
"disconnected" with "perfectly robust"); `find_optimal_m` skips non-finite
J; symmetrization `max(A, Aᵀ)`. *impl:* `analysis/metrics.py`. *tests*
(`test/analysis/test_metric_fixes.py`): thin-line flock thickness < 0.2
and ∈ (0,1], round cloud ≈ 1; two far pairs at m=1 → `math.isinf`; hand
3-node directed graph → max-form Laplacian.

## S2 — Mode workstreams  *(parallel tracks; ≈11½ days)*

### Track A — Field/blob + threat  *(git6; ≈4 days)*
Files: `physics/forces/field.py::FieldMode`,
`physics/extensions/{wander,ripple,predator}.py`; tests
`test/physics/forces/test_field.py`,
`test/physics/extensions/test_threat.py`. Unit scale
`U = cfg.field.unit_scale or 0.4·min(W,H,D)`; `C` = domain centre;
seed-derived quantities cached in `reset()`.

**S2.A1 Wander path** — *math* (verified `boundedUnitTravel`):
```
raw_x = sin(t·0.47 + sin(t·0.19)·1.15)·0.82 + sin(t·1.07+1.4)·0.38 + cos(t·0.23+2.1)·0.22
raw_y = cos(t·0.43+0.6 + sin(t·0.13)·0.9)·0.78 + sin(t·0.91+2.8)·0.42 + cos(t·0.29+0.4)·0.24
raw_z = sin(t·0.39+1.1 + cos(t·0.17)·1.05)·0.80 + cos(t·0.97+0.2)·0.40 + sin(t·0.21+2.6)·0.22
pulse = 0.72 + 0.28·(0.5 + 0.5·sin(t·0.41 + cos(t·0.17)))
path(t) = raw · pulse / max(1, ‖raw‖)              ⇒ ‖path‖ ≤ 1 guaranteed
wander_center(t) = C + path(t·speed)·radius·U
heading(t) = normalize(path(t+0.75) − path(t))
```
*impl:* rewrite `extensions/wander.py` (fixes the domain-corner bug).
*tests:* `‖path‖ ≤ 1` for 10⁶ fuzzed t; heading unit & continuous
(‖h(t+ε)−h(t)‖ < 0.05); attractor in-domain over 10⁴ frames.

**S2.A2 Blob anchors + phase weights** — *math:* five Lissajous anchors
about `flock.center` (coefficients: git6 R3);
`φᵢ = fract(seedᵢ·3.71 + t·0.022 + sin(seedᵢ·19 + t·0.11)·0.09)`;
`w_k = max(0, 1 − wrap(φ, c_k)·7.5)²`, c_k ∈ {0,.2,.4,.6,.8};
`T_legacy = Σ B_k w_k / Σ w_k` (Σw > 0 provably). *tests:*
`Σ_k w_k(φ) > 0` fuzzed; anchors at fixed t match hand values;
2 000 birds → k-means finds ≥ 4 clusters at t = 30 s; per-bird target
variance > 0.

**S2.A3 Leader/chaser** — *math:* 7 seed groups; lagged anchor (git6 R4);
`lag = hash01(seed+9.17)(1.1+2.4·chase)`; leaders
`hash01(seed+5.91) ≥ 0.84` (~16 %); golden-angle shells
(`ga = 2.39996323`, `y = 1−2·fract((slot+0.5)·0.618034 + gs·0.13)`,
`shell = fract((slot+1)·0.754877)^{1/3}`,
breath `1+sin(t·0.13+gs·12)·0.035`);
`T = lerp(T_legacy, chase_target, chase)` — activates
`cfg.field.chase_strength`. *tests:* leader fraction 0.16 ± 0.02 over 10⁴
seeds; group membership stable; chase = 0 ≡ S2.A2 targets (allclose);
chase = 0.8 → 7-cluster structure, leaders' anchor-distance < followers'.

**S2.A4 Shell + cavity** — *math:*
`R_blob = (0.24 + (0.5+0.5 sin(seed·41+t·0.29))·0.16 + sin(φ·2π+t·0.17)·0.05)U`;
`F = −d̂(d−R_blob)·coh·1.35(1−chase)`; inner floor
`R_blob(0.28+(1−chase)·0.18+sep·0.012)`, push-out ×`sep·1.4`. *tests:*
settled blob — central voxel density < 0.3× shell band; R_blob FFT shows
both documented oscillation frequencies.

**S2.A5 Remaining terms (full 13-term composition)** — *math:* slot
repulsion offsets **±{1,7,31}** mod-wrapped, kernel
`((r_slot−d)/r_slot)²` inside `r_slot = (0.07+sep·0.02)U`, gain
`sep(0.14+chase·0.05)`; tangential
`normalize(axis×(p−T))·align·0.035(1−chase)` (drifting seed axis);
buoyancy (z-up)
`F_z += (sin(8d/U−1.1t+17·seed)·0.09 + 0.24(T_z−p_z)/U)(0.75+0.25·flow)`;
curl flow (normalized sin+cos pairs ×0.08); fold band (spatial 2.4–3.7,
temporal 0.43–0.73, × ripple envelope sum); drag
`−v·chase(0.08+0.02·flow)`; drift alignment to `heading(t)·v0`; target
pull `(T−p)/U·coh·target_pull`. *tests:* each term unit-pinned on hand
inputs (slot kernel zero outside r_slot & continuous at it; buoyancy
z-only; drag anti-parallel; flow/fold normalized pre-gain); 10⁴-frame
NaN/speed fuzz all-terms-on; tangential on → nonzero sign-stable angular
momentum about blob axes.

**S2.A6 Ripples** — *math:* trains {0, 9.33, 18.67};
`env = ss(0.6,1.7,τ)(1−ss(6.2,8.8,τ))`; `radius = (0.16+0.16τ)U`;
`width = (0.11+0.012τ)U`; moving Lissajous origins about C; twist
`+(heading×F_r)·0.28`; gain `flow(0.13+0.04·waveGain)`. *impl:* vectorised
in `FieldMode`; `extensions/ripple.py` a thin wrapper for other modes.
*tests:* env zero outside [0.6, 8.8], peak in [1.7, 6.2]; origins move;
paused-flock radial histogram shows 3 rings; < 5 ms at N = 100 k.

**S2.A7 Inertia / bounded panic / blackening** — *math:* inertia lerp
(D4); panic ceiling
`v0(1+min(1.35, panic(0.72+0.18·wave+0.12·vacuole)))` via per-bird
`max_speed` (kills the compounding ×1.5 bug); blackening
`sep_eff = sep(2−black)`, `coh_eff = coh·black`,
`black = 1+gain·prox·0.85` (prox from S2.A8 via `ctx.threat_prox`).
*tests:* max speed ≤ 2.35·v0 across 10⁴ panic frames; wake-region pair
distance drops during a pass; inertia 0.8 → per-frame |Δ‖v‖| < 0.2·v0.

**S2.A8 Threat FSM + force bundle** — *math (source-verified):*
`capture = max(0.18, 0.72R)U`; `pass = (0.92+2.6R+1.32·mom)U`;
`clear = pass(0.72+0.16·mom)` + heading gate `dot < −0.12`; turn rate
`(0.54+0.025·accel)(1−0.24·mom)` (orbit 0.42·…); sign-aligned EMA turn
axis; Rodrigues `rotate_toward` capped at `rate·dt`; egress arc
`broad = R·(0.36 chase | 0.24 orbit)·U`, lift `sin(0.18t+0.7)·broad`,
drift `cos(0.13t+1.4)·broad·0.72`; force bundle with `broad = √prox`:
push `â·strength(2.5+1.7·vacuole)·broad`; wake
`(â−dir·0.35)·min(1.8, ‖v_t‖/v0)·strength·broad·0.42`; split
`(−â_y, â_x, 0.28/1.45·â_z)·splitGain·broad·1.45` (horizontal tear,
z-up); wave `v̂·waveGain·broad·0.22`. Modes off/cursor/orbit/autonomous.
*impl:* `extensions/predator.py → Threat(Extension)`; publishes
`ctx.threat_prox`; **rendered** via the flag channel (currently the
predator is invisible). *tests:* `rotate_toward` Rodrigues-exact and
capped; phase transitions at exact distances with the dot gate; trace
continuous (max step < 3·speed·dt), crosses and exits ≥ clear; evacuated
region horizontally biased (xy-extent > z-extent); `threat_prox ∈ [0,1]`
shape (N,); *(gl)* red marker visible in all themes.

**S2.A9 Blob init + presets** — *math:* 5 fixed centres
((−0.48,0.18,0.12) …), ∛-uniform shells `r = cbrt(u)(0.22+u'·0.28)U`,
jitter 0.045U, drift-biased tangential velocities; presets
quiet_roost / lava_lamp / ink_cloud / predator_ripple / vacuole /
silk_sheet / storm_turn as `conf/field_*.yaml` (full vectors: git6 R12).
*tests:* frame-0 lobes; init densities equal across N (±10 %); presets
load with documented values.

### Track B — Reynolds variants (Starlings + rystrauss)  *(git5, git3; ≈3½ days)*
Files: `physics/forces/spatial.py`, `physics/forces/_base.py`,
`physics/forces/_kernels.py`, `physics/extensions/ecology.py`,
`physics/boid.py`; tests
`test/physics/forces/test_spatial_variants.py`,
`test/physics/extensions/test_ecology.py`.

**S2.B1 Hybrid filter + dual radii** — neighbour iff `d < visual_range`
AND among first `cfg.spatial.influence_count` (7); alignment subset
`d < alignment_radius_ratio·R` (0.75 — field goes live);
`separation_distance` (20) as a metric gate. Extend `cfg.spatial.
neighbor_filter` to `knn | hybrid | global` *(sci7 §3.2–3.3)*: the `global`
degenerate case steers alignment/cohesion toward the **whole-flock** mean
velocity / CoM (no radius, no kNN) — the same behaviour the `marl` mode's
embedded rules use (S7.1), exposed here as a general spatial-mode option for
studying global vs local coupling. *tests:* hand cluster — neighbour set
respects radius AND cap; alignment set ⊆ cohesion set; `global` → every
bird's cohesion target equals the flock CoM.

**S2.B2 Update-order fidelity** — order: predator boost(×1.4) →
`acceleration_scale`(0.3) → limit(max_force) → `v += a` → velocity noise
`(U³−0.5)·noise_scale` (when `noise_mode="velocity"`) → ceiling limit →
move; `speed_mode ∈ {band, ceiling, fixed}`. *tests:* monkeypatched-stage
order recording; "ceiling" allows |v| < 0.3v0; "fixed" → |v| ≡ v0.

**S2.B3 Predator boids (species)** — boosts 1.8× speed / 1.5× perception /
1.4× acceleration; escape
`normalize(p_prey−p_pred)·cfg.spatial.predator_escape_factor (10⁷)`
**replacing** separation; **hard-zero** align+cohesion when any predator
is perceived; predators flock among themselves. *tests:* hand
neighbourhood → align/coh contributions exactly zero; escape wins the sum
pre-limit; flash-expansion (mean NN distance doubles in 30 frames); two
predators' pair distance stabilises.

**S2.B4 Physical metrics** — `k_v = cruise_ms/v0` (8.94 m/s default),
`k_a = acc_peak/max_force` (40 m/s²), m = 0.075 kg; `F = m·k_a⟨|a|⟩` (N);
`P = m⟨|k_a a · k_v v|⟩` (W); `E = Σ P·Δt` (J);
`L = m(r−CoM)×(k_v v)`; reads `flock.last_accelerations`. *impl:*
`analysis/metrics.py` + `cfg.metrics.*`. *tests:* hand-set v →
`speed_real == k_v·|v|` exactly; E ≈ P̄·elapsed ± 1 %; stash test —
metrics see nonzero pre-reset accelerations.

**S2.B5 Parameter jitter** — effective weights per frame from `flock.rng`:
`sep + U(0, 0.5)`, `coh + U(0, 0.1)`, `align + U(0, 0.005)`; config never
mutated. *tests:* spacing-series std(on) > std(off), same seed; config
unchanged after run; determinism holds.

**S2.B6 Parallel two-phase** — batched
`index.query_knn_batch(pos, k, workers=cfg.perf.num_threads)` + fully
vectorised gather/reduce force pass (`positions[neighbor_idx]` shape
(n,k,3), masked padding, axis-1 reductions — no per-bird Python loops).
*tests:* ≥3× at N = 20 k vs recorded loop baseline (`@slow`); identical
results across worker counts (T4.3).

**S2.B7 Sphere centring + asymptotic wall** — centre on C (D4); add
`boundary.mode = "sphere_soft"`: `Δv = −μ r̂ / max(R−r, 0.05R)` applied
inside the shell margin. *tests:* centre-initialised flock ‖CoM−C‖ < 0.1R
over 5 000 frames, both modes; soft mode never crosses R.

**S2.B8 Ecology completion** *(todo_claude §5–8 in full)* — *math:*
logistic dusk `1/(1+e^{−z})`, `z = (hour−sunset(day))/(width/4)` clamped
|z| > 60, `sunset = 12 + day_length/2`; `is_roosting_time(hour, day,
0.5)`; cold boost `roost_strength = base·dusk·max(0, 1+0.2(T_mean −
T(day))/T_amp)`; `roost_force = unit(roost−p)·roost_strength`;
**coherence gate** `coherence(N) = smoothstep over [0.4, 1.2]·N_crit`,
`gated_weight(w, N) = w·coherence(N)` **applied to the flocking weights**
(φa/φp or spatial weights); **seasonal model**
`seasonal_size_factor(day)` = cosine, 1.0 at PEAK_DAY = 15, MIN_FACTOR =
0.25 at +182; `flock_size_for_day(day, peak_size, min_size=0) → int`
driving N via the command queue when `cfg.ecology.seasonal_size`;
`is_murmuration_season(day)` (Oct–Mar) gating roost/predator behaviour;
**stochastic predator presence** `predator_present(day, rng=None)` —
Goodenough (2017) empirical rate `PREDATOR_RATE = 0.296`; deterministic
per-day Knuth-hash branch (`((day·2654435761) mod 1000)/1000 < RATE`,
reproducible) OR a true draw `rng.random() < RATE` when an rng is supplied
(from `ctx.rng`), selected by `cfg.ecology.predator_presence:
deterministic|stochastic` (the current code is deterministic-only and uses
a `77/256` shortcut that drifts from the cited rate — use `RATE`);
`cfg.ecology.roost / critical_mass` live. *impl:* free functions +
`Ecology(Extension)` in `physics/extensions/ecology.py`. *tests:*
`seasonal_size_factor(PEAK_DAY) ≈ 1.0`, `(+182) ≈ MIN_FACTOR`;
`flock_size_for_day` ints in range, curve-shaped; season Jan-in/Jul-out;
`dusk_factor(0)=0, (40)=1`, 0.5 at sunset; colder → stronger
(day 20 > day 200, same hour); `gated_weight(0.8, 10) ≈ 0`,
`(0.8, 600) > 0.7`; behavioural — gate on: α(N=50) < α(N=800) identical
params; seasonal N tracks the curve over a simulated year;
`predator_present` deterministic same-day-same-result and yearly frequency
0.296 ± 0.03, seeded-rng frequency 0.296 ± 0.01 over 10⁴ draws.

**S2.B9 Velocity-init variants** *(todo_claude E12)* — `cube`:
`(U³−0.5)·2v0` (E|v| ≈ 0.816·v0); `speed_uniform`: uniform direction ×
`U(min(1, 0.3v0), v0)`; `tangential`: `normalize(p−C)×random_unit ·
U(1, v0)`; selector `cfg.flock.velocity_init`. *tests:* cube mean ≈
0.816·v0 (±5 %, 10⁴ birds); speed_uniform in-band, non-constant;
tangential ⊥ radial (dot < 1e-5).

**S2.B10 Numba force kernels + precision policy** *(arch.md two-pass;
dead `use_numba`)* — two-pass: batched index query (Python/scipy) →
`@njit(parallel=True)` kernel over
`(positions, velocities, accelerations, active, neighbor_idx, weights…)`,
fulfilling the `ForceKernel` contract (or D9.4 deletes it).
`cfg.perf.use_numba` gates; `cfg.perf.fastmath` allowed **only** when
`metrics.detail_level == 0` (visual runs) — IEEE kernels whenever
observables are exported; lazy import; numpy path stays the reference.
*impl:* `physics/forces/_kernels.py`, consumed by SpatialMode/VicsekMode.
*tests:* numba ≡ numpy within `atol=1e-5` (fastmath off), same seeds,
N = 2 000; exporting metrics with fastmath on raises/warns; `@slow`
N = 50 k step within arch.md budget ×2.

### Track C — Angle mode (PyNBoids paradigm)  *(git4; ≈2 days)*
Files: `physics/forces/angle.py::AngleMode` (new; `speed_mode="fixed"`,
per-bird speeds); tests `test/physics/forces/test_angle.py`.

**S2.C1 Steering core** — dead zone (no turn if error <
`turn_threshold`°, anti-oscillation); 3D axis-angle:
`φ = acos(clamp(ĥ·t̂, −1, 1))`, axis `normalize(ĥ×t̂)` (any ⊥ axis when
parallel/anti-parallel), rotate `rotate_about(ĥ, k̂, min(φ, rate·dt))` —
never overshoot. *tests:* 180°-behind target turns through π/rate seconds
± 1 frame; per-frame heading change ≤ rate·dt + jitter; dead-zone hold
exact.

**S2.C2 Unified neighbour modes** — 7 closest within `b·12`
(b = boid_size): nearest < `b·1` → steer away from nearest (exclusive
flee state); nearest < `b·5` → toward `normalize(ĉ + m̂)` (centroid +
mean heading; 3D replaces `atan2(Σsin, Σcos)`); else → ĉ only. *tests:*
forced-close pair separates; mid-range cluster contracts AND α rises; far
cluster contracts only.

**S2.C3 Adaptive speed** — `s = base + (7−m)·5` (linear) | `+(7−m)²`
(quadratic, cap 49) | `+min(49, (7−m)²/2)` (softened), per
`cfg.angle.speed_mode`. *tests:* m = 0 → base+35 (linear); m ≥ 7 → base;
median 7th-NN distance converges (self-regulating density).

**S2.C4 Edge handling** — inside `margin`: steering target overridden by
the nearest face's inward normal (±x/±y/±z, sequence priority); spherical
variant `t̂ = normalize(C−p)`; turn rate
`rate += (1 − edgeDist/M)(maxRate − rate)`. *tests:* margin boundary,
10⁴ frames, zero escapes at max speeds; birds arc (mean tangential speed
at the wall > 0 — no sticking).

**S2.C5 Heading jitter** — ±`jitter_deg`° rotation about a random axis
**before** steering (steering compensates), from `flock.rng`. *tests:*
steering-off distribution bounded ±4°, ~symmetric; net track endpoint
within 2 % of jitter-off run.

**S2.C6 Incremental grid** — per-bird `last_cell`; re-file only on cell
crossing (vs full rebuild); behind the SpatialIndex protocol. *tests:*
neighbour sets == full-rebuild sets over 500 random-walk frames; touches
< 10 % of birds/frame at N = 5 k.

**S2.C7 Body-unit radii** — `sep/align/range_radius_bodies` scale with
`boid_size` (scale invariance; also offered to spatial mode as
`radii_in_bodies`). *tests:* doubling boid_size doubles all three
thresholds; 2×-scale behavioural smoke.

### Track D — Vicsek predator–prey  *(git2 R2–R5; ≈2 days)*
Files: `physics/forces/vicsek.py`; tests
`test/physics/forces/test_vicsek_species.py`.

**S2.D1 Species dynamics** — *math:* fear
`= clamp((R_pred − d̄_pred)/R_pred, 0, 1)` (d̄ = mean distance to
predators within R_pred, min-image);
`û_combined = normalize((1−fear)·û_align + fear·û_flee)`,
`û_flee = normalize(Σ (p_prey − p_k)_mi / |P|)` (random unit if none);
neighbour weights ×`weight_afraid` (3.0) while afraid; predator update:
nearest prey within `detect_ratio·R_pred` (1.5×), then
`û = normalize(û_target + predator_noise_ratio·η̂)` — **no couplage**;
random-walk fallback; all-predators early-out (pure random walk, skip all
interaction). *tests:* stationary predator at flock centre → prey inside
R_pred have ⟨û·r̂⟩ > 0.8 within 5 steps; monotone pursuit (≥ 90 % of
steps close distance); n_prey = 0 → α ≈ 1/√N for all η, D; afraid birds
align to neighbours more strongly than calm (two-group setup).

**S2.D2 Asymmetric position collisions** — *math:* same-type pairs at
`d < R_avoid`: each moves `(R_avoid − d)/2` along min-image n̂;
prey–predator at `d < R_pred`: **prey takes the full** `(R_pred − d)`
correction, predator unmoved; applied after move, before wrap;
`np.add.at` accumulation. Activates `cfg.vicsek.radius_avoid`. *tests:*
hand pair corrections exact (both cases); seam-crossing pair corrected;
100 steps → no same-type pair < 0.5·R_avoid; predator trace unaffected by
contacts.

**S2.D3 Prey-only metrics in vicsek mode** — *test:* α of aligned prey +
one orthogonal predator == 1.0.

### Track E — Influencer (murmuratR)  *(git1; ≈1½ days)*
Files: `physics/forces/influencer.py::InfluencerMode`
(`owns_positions=True`, `speed_mode="fixed"`, instance `tick`); tests
`test/physics/forces/test_influencer.py`.

**S2.E1 Trajectory** — *math (verbatim `target_pos`):*
```
T_raw(t) = ( sin(t/97)·200 + cos(t/217)·30,
             cos((t+53)/29)·200 + sin((47−t)/13)·30,
             cos((t+61)/41)·100 + sin((t+13)/7)·27 + 40 )
s = cfg.influencer.scale · min(W/460, H/460, D/254)
T(t) = C + (T_raw(t) − (0,0,40))·s + (0, 0, 40s)
```
Persistent tick (one per substep) replaces the current random-t teleport.
*Optional path-shaping* *(sci6 §2.3)*: lift the hardcoded coefficients to
`cfg.influencer` fields (defaulting to the verbatim values above) so the
trajectory is tunable — `traj_primary_amp` (200,200,100 → range of motion),
`traj_secondary_amp` (30,30,27 → local flutter), `traj_periods`
(97,217,29,13,41,7 → looping-vs-wandering character; keep mutually prime for
aperiodicity), `traj_phase` (53,47,61,13 → starting position), `traj_z_bias`
(40 → mean altitude). Each optional; unset = the canonical murmuratR path.
*tests:* `_target_pos` at t ∈ {0, 970, 2170} equals hand values (s = 1,
460×460×254 domain, default coefficients); in-domain for scale ≤ 1; step
distance varies; overriding `traj_primary_amp` scales the path extent
proportionally.

**S2.E2 Move-then-steer at unit speed** — per substep:
`p += d̂·v0·dt` (OLD direction) → recompute `t̂, dist` (guard
`x += (x==0)`) → `d̂ ← normalize(d̂(1−inf) + t̂·inf)` → `tick += 1`.
*tests:* frozen target → convergence to hover/orbit; one-step lag — after
a target jump, headings change only on the following substep; |v| ≡ v0.

**S2.E3 Influence** — rank by **distance-to-target** (not CoM):
`inf_sorted[i] = (1 − (i/(N−1))·0.8)^rank_exponent` (1.8 → floor
0.2^1.8 ≈ 0.055); distance alternative
`clamp(near_dist_sq·s²/d², 0.2, 0.8)` behind
`cfg.influencer.influence_mode`. *tests:* exactly one bird at 1.0;
min ≈ 0.055 ± 1e-3; monotone non-increasing in target distance;
exponent 1.0 → linear.

**S2.E4 Density-scaled init** — Gaussian `σ = N^{1/3}·separation·s`
(sep 0.5) + shared random offset `C + U(0, 10s)³`; **zero initial
directions** (first blend heads every bird at the target, weighted).
*tests:* init density equal across N ∈ {100, 1 000, 8 000} (±10 %);
frame-0 headings ∝ influence toward target.

**S2.E5 Diagnostics** — per-frame `min/max ‖p − T‖` →
`FlockMetrics.target_dist_min/max` + window title. *tests:* CSV contains
both columns, min ≤ max, finite.

## S3 — Metrics & analysis suite  *(≈2 days; after D3)*
Files: `analysis/metrics.py`, `analysis/rewards.py`,
`analysis/phase_diagram.py`, `analysis/density_scaling.py`; tests
`test/analysis/`.

**S3.1 Nematic order** — `Q = (3/2)(ûᵀû)/N − ½I` (3×3, traceless),
`S = λ_max(Q)`; O(N); `order: polar|nematic` option in the phase-diagram
sweep. *tests:* two anti-parallel half-flocks → α < 0.05, S > 0.95;
isotropic 500 birds → both < 0.15; S invariant under per-bird û → −û and
SO(3).
Also add a `quick=True` **snapshot mode** to `phase_diagram.sweep` *(sci8 §6)*:
one single-step angle update per (η, D) grid cell on the current
configuration instead of a full settled run — ~200× cheaper, for interactive
parameter-space exploration (the settled-run path stays the default and the
scientifically correct one). *tests:* `quick=True` returns the same grid shape
as the settled sweep and runs in a fraction of the time.

**S3.2 MSD(τ) curve** — unwrapped accumulation
`p_unwrap += min_image(p_t − p_{t−1})`; per-lag
`MSD(τ) = ⟨‖p_unwrap(t+τ) − p_unwrap(t)‖²⟩` at log-spaced lags
{1,2,4,…,64}; crossover `τ_cross: d log MSD/d log τ ≈ 1.5`. *tests:*
D = 0 aligned flight → slope 2.0 ± 0.1 all lags; strong-noise walkers →
1.0 ± 0.2 for τ ≥ 4; seam crossing contributes MSD(1) = (v dt)² ± 1e-4.

**S3.3 Shape→m\*** *(todo_claude §9)* —
`m* = 9.78 + clamp((aspect−1)/2, 0, 1)·(6.05 − 9.78)`; `suggested_m`
field on `FlockMetrics`. *tests:* endpoints (aspect 1 → 9.78, ≥ 3 →
6.05); monotone; thin flock ≤ 7, round ≥ 8.

**S3.4 η(m)** *(todo_claude §10)* — `η(m) = (H₂(m₀) − H₂(m))/(m − m₀)`;
+∞ when m first connects the graph; 0.0 when both disconnected (needs
S1.8's `inf`). *tests:* connectivity transition → `math.isinf`; both
disconnected → 0.0; telescoping sum property.

**S3.5 Hull-volume τρ** *(todo_claude §11)* —
`ρ = N / ConvexHull(positions).volume` (0 if degenerate/< 4 points); ring
buffer (sample every 10 frames, 500 slots);
`τ = interval·(0.5 + Σ_{lag≥1} r(lag))`, stop at first `r ≤ 0`. *tests:*
cube hull = edge³ ± 1e-3; coplanar → 0; constant series → τ == 0 (not
NaN); period-P oscillation → τ ∈ [P/6, P].

**S3.6 Θ′ silhouette** *(todo_claude §12)* — project positions ⊥ an
observer axis, rasterize disks of radius `boid_size`, coverage = union
fraction (overlaps counted once); **additional field** beside the voxel
Θ′. *tests:* flat wall ⊥ axis → silhouette ≈ 1 while voxel Θ′ ≪ 1; two
co-projected birds == one.

**S3.6a Marginal-opacity validation** *(predecessor audit; Pearce 2014
headline result)* — the projection model's *raison d'être*: a flock steering
on δ̂ self-regulates its density to **marginal opacity**. Reference (documented,
not runtime-enforced): silhouette `Θ′ ~ N(µ=0.30, σ²=0.059)` fitted across 118
real flocks. *impl:* module constants `MARGINAL_OPACITY_MEAN = 0.30`,
`MARGINAL_OPACITY_STD = 0.243` in `analysis/metrics.py` (Pearce citation);
**no new physics** — consumes the S3.6 *silhouette* Θ′ (not the voxel metric).
*tests* (`test/analysis/test_marginal_opacity.py`, `@slow`):
**scientific regression** — a seeded projection-mode run (N≈150, ~300 settle
frames) has time-averaged silhouette Θ′ ∈ [0.05, 0.55] (a loose µ±3σ band; the
claim is "marginally opaque", not exactly 0.30 — domain/N/φ shift the operating
point); if a future physics change breaks self-regulation, δ̂ or the φ weights
are wrong. Plus a constants-documented guard.

**S3.7 Robust gyration + number density + ideal exponent**
*(todo_claude §13)* — **median** centroid; one-sided top-15 % trim
(`keep = 0.85`); `R_g = √mean(r²_kept)`;
`ρ = N_kept / ((4/3)πR_g³)`; density-scaling sweep reports
`ideal_density_exponent = −0.5` beside fitted β (keep toroidal-vs-open +
R² framing). *tests:* one 10 000-unit outlier moves R_g < 5 %; degenerate
flock density → 0; sweep carries `−0.5` and finite β.

**S3.8 Motion metrics** *(todo_claude §15; git7 R4)* —
`velocity_deviation = (1/N)Σ‖v̄ − v_i‖` (catches speed dispersion α
misses); `boundary_overshoot = Σ max(0, ‖p−C‖ − R_dom)`;
`altitude_deviation = (1/N)Σ|z_i − z_target|` *(sci7 §2.5;
`z_target` = `cfg.metrics.altitude_target`, default domain-centre z)* —
mean vertical spread from a target altitude, a strictly-3D observable that
pairs with roosting/ecology and feeds the reward module's altitude term
(S3.9); normalized angular momentum `‖⟨r×v⟩‖/(v0·R_g)`; L about CoM with mass
(S2.B4). *tests:* equal headings + mixed speeds → deviation > 0 while α == 1;
overshoot 0 inside, > 0 for planted outliers; altitude_deviation 0 for a flat
sheet at `z_target`, grows with vertical spread; L translation-invariant;
normalized L O(1) across ×10 domain scale (±10 %).

**S3.9 Rewards module** — `analysis/rewards.py`: weighted composite over
named metric terms; `reward_faithful_signs` flag (source's +alignment
quirk vs corrected both-negative); shared by MARL (S7) and EvoFlock
scalarization. *tests:* perfect flock → corrected reward 0 (max);
faithful flag flips the alignment sign; per-term weight linearity.

**S3.10 Export schema** — `FlockMetrics.to_dict()` (D9.2) adopted
end-to-end; new fields (suggested_m, nematic, msd_curve, target_dist_*,
*_real) included. *tests:* JSON round-trip; pinned key set; Recorder CSV
headers == schema.

**S3.11 EMA-smoothed readout** *(predecessor audit; FlockMetrics3D)* — the
HUD/title/console readout smooths the fast fields (α, Θ, Θ′, L, σ_r) with a
one-pole EMA so it is stable frame-to-frame: `ema ← ema + s·(raw − ema)`,
`cfg.metrics.readout_smooth: float = 0.04` (0 = raw). **Strict display/analysis
separation** (modularity): smoothing is *display-only* — `to_dict()`, CSV/JSON
export, and every science/validation path keep **raw** per-frame values; never
smooth what you later analyse. *impl:* a `MetricsReadout` view (or `smoothed`
property) on `MetricsCollector` updated in `collect()`; consumed only by the
S5.2 title and S5.6 log; Recorder stays on raw. *tests*
(`test/analysis/test_metrics_readout.py`): EMA converges to a constant raw
stream and approaches a step monotonically without overshoot; `readout_smooth=0`
→ passthrough; `to_dict()` values equal the raw snapshot even while the readout
lags (raw/display separation asserted).

## S4 — Rendering & capture  *(≈4 days; after D7)*
Files: `viz/{renderer,shaders,camera,hud,visualizer}.py`,
`capture/{recorder,mpl_recorder}.py`; tests `test/viz/`,
`test/capture/` (`@gl` unless noted).

**S4.1 Sphere impostors** — billboard quads; fragment: `p = uv·2−1`,
`r² = p·p`, discard > 1, `z = √(1−r²)`, `shade = 0.55+0.45z`,
`color = mix(paper, ink, shade(1−0.22·rim))`; `cfg.viz.point_sprites`.
*tests:* shader compiles; centre pixel brighter than rim; corners =
background.

**S4.2 Depth cues** — size ∝ 1/depth^k; alpha ×
`mix(1, 1−depth01, fade) · mix(0.65, 1, speed01) ·
mix(1, 0.76, ss(0.72, 1, r²))`. Plus **Fresnel rim lighting** *(sci9 §4;
distinct from the impostor-disc rim above)* — the 3D generalisation of the
source's 1-px outline: `rim = pow(1 − max(N·V, 0), k)` (k ≈ 2–3), added to the
mesh fragment shader as a view-angle silhouette highlight (a depth/shape cue
that reads on solid meshes where the disc-`r²` rim does not apply). *tests:*
near bird renders larger & more opaque than far (pixel-area/alpha probe);
edge-on mesh pixels (low N·V) brighter than face-on.

**S4.3 Trails ×3** (`cfg.viz.trails`) — *velocity:* impostor stretched
along `proj(p) − proj(p − v·len·0.12)`; head
`max(0.28, 1/(1+2.8s))`; tail `0.22+1.35s`; wave
`sin(prog(5.4+3.4·speed01)+seed)·wav·s·0.18`. *accumulation:* fade quad at
`clamp(0.24 − 0.19·persist − 0.09·vis, 0.018, 0.32)`
(persist = clamp(len/5), vis = clamp(opacity)); depth-only clear, then
draw. *ring:* K = `trail_length` past positions (from `prev_positions`
lineage) as shrinking/fading sprites. *tests:* velocity — lit extent
along motion > ⊥; accumulation — persists ≈ 1/fadeOpacity frames, clears
when paused; ring — K sprites monotone size/alpha; `off` pixel-identical
baseline.

**S4.4 Winged mesh + flap** — 6-triangle body+wings+tail (vertex table:
git5 R6; forward +Z, wingspan ±8 on X); per-vertex flap weight (1.0 at
wing tips); `u_Flap = ±0.5` toggled every `⌊frame/flap_period⌋`
(period 100); applied to mesh-y **before** the LookAt rotation
(local-up flap). *tests:* geometry counts; tip y toggles at exact frame
boundaries; bird flying +z flaps in world-xy.

**S4.4a Mesh-registry entries + theme material sets** *(sci9 §3–4;
high-modularity — the `cfg.viz.bird_mesh` selector + `shaders.py` mesh table
is the extension seam, arch.md §12)* —
- **Speed-stretched ellipsoid** *(Option C)*: an ellipsoid mesh scaled along
  the velocity axis by the speed ratio — one extra per-instance factor
  `(1, 1, clamp(|v|/v0, lo, hi))` applied in the existing LookAt vertex shader
  before rotation; a cheap motion cue that reads even with trails off.
- **Cone / arrow procedural meshes** *(Options A/B)*: registry entries beside
  `tetra | winged | impostor | ellipsoid` — proves the mesh table is truly
  pluggable (no shader branching; one entry each).
- **Theme-driven material sets**: promote the Blinn-Phong `ambient`/`diffuse`
  pair from the hardcoded `(0.15,0.17,0.22)/(0.65,0.68,0.78)` constants to a
  per-theme table (e.g. dark scheme `diffuse (0,0.8,0) / ambient (0,0.2,0)`),
  driven by `cfg.viz.theme` (which D7 already threads to the renderer) so mesh
  shading matches the scheme rather than one fixed palette.
*tests:* each registered mesh (`tetra|winged|impostor|ellipsoid|cone|arrow`)
renders one frame without GL error (`@gl` smoke); a bird at 2·v0 renders
longer along its heading than at 0.5·v0 (ellipsoid stretch); switching
`theme` changes the sampled mesh ambient/diffuse (pixel probe).

**S4.5 Gradient sky** — fullscreen quad, top (0.60, 1, 1) → bottom
(0.686, 0.933, 0.933), theme-overridable; drawn first, depth off.
*tests:* top/bottom row pixel colours; flat mode unchanged.

**S4.6 Colour channels** — per-bird hue from `seeds` (HSV h = seed·360,
S = V = 0.9) via the schema hue float; predator flag → red, ×1.3–1.5
scale; heading-hue debug theme (azimuth → hue). *tests:* hue stable
across frames; predator red in all themes; +x vs −x flight differs ≈ 180°
in hue.

**S4.7 Alpha-accumulation density mode** — α ≈ 0.2 sprites, blending on,
depth-write off (murmuratR aesthetic). *tests:* cluster centre darker
than single bird.
*Optional exotic variant* *(sci4 §8.3 Option C — low priority; the S4.3 ring
and accumulation trails are the recommended forms first)*: a **64³ volumetric
accumulation texture** — splat bird positions into a low-res 3D density grid,
apply a per-frame 3D blur+fade, and raymarch it as a volumetric overlay for a
true smoke-like density field. Behind `cfg.viz.trails: "volumetric"`; GL 4.3
compute or a slice-stack fallback. *tests:* occupied voxel fraction tracks
flock compactness; fade decays a static splat over ~N frames.

**S4.8 Views** — dual-viewport (elev/azim 15°/15° + 45°/45°, two
camera/viewport passes — exposes planar flocks); orthographic top/side
presets (keys 7/8/9); fixed capture framing option. *tests:* halves
differ and both contain birds; ortho — equal pixel size at different
depths.

**S4.9 Capture pipeline** *(todo_claude E7–E10)* — cinematic sweep
(`azim = 45°+180°t`, `elev = 25°+0.15 sin 2πt`,
`dist = (650+100 sin 1.5πt)·scale`); pre-warm
(`capture.prewarm = 60` un-captured settle frames);
`CAPTURE_W/H/FRAMES/OUT` env overrides (YAML < env < CLI); GIF
`optimize=True, disposal=2`; matplotlib GPU-free fallback
(`capture/mpl_recorder.py`, dual-view scatter → GIF) replacing the silent
frame loss. *tests:* folded into D7-T5.4/5.5 + first-captured-frame
dispersion < unwarmed frame-0.

**S4.10 Adaptive quality** — EMA `avg = 0.92·avg + 0.08·min(250, frame_ms)`;
budget `1000/max(24, target_fps)`; healthy if `avg ≤ 1.12·budget`; risks
cpu/vertex/fragment → classification; ladder (trails off → render scale
−0.15 floor 0.75 → N −18 % floor 512) when fps < 78 % of target for
≥ 1.8 s, one step per 1.8 s. *tests:* synthetic series → ladder order,
spacing, per-action effects, recovery stop (pure logic, no GL).

**S4.11 Fixed-timestep accumulator + interpolation** —
`acc += clamp(frame_dt, 0, 1/20); while acc ≥ dt_phys: step(dt_phys)`;
optional render lerp `prev_positions → positions` by `acc/dt_phys`.
*tests:* same seed at simulated 30 vs 60 fps → identical physics; rendered
position between the two states.

## S5 — UX & tooling  *(≈2 days; after D8)*
Files: `analysis/presets.py`, `viz/{input_control,hud,visualizer}.py`,
`__main__.py`, `__init__.py`; tests `test/viz/test_input.py`,
`test/test_cli.py` (no GL).

**S5.1 Preset keys a–h,w** *(todo_claude E4)* — the predecessor's exact
table (a: 0.04/0.80/6/proj · b: 0.18/0.70/7/proj · c: 0.06/0.45/3/proj ·
d: 0.25/0.55/8/spatial · e: 0.10/0.75/6/proj · f: 0.02/0.85/3/proj ·
w: 0.08/0.82/10/spatial · h: 0.35/0.58/9/spatial) with labels/
descriptions printed on apply; key range skips `g` (grid). *tests:*
synthetic KEYDOWN 'b' → config equals the row; 'g' still toggles grid;
description printed (capsys).

**S5.2 Full title readout** *(E6)* — mode, N, φp/φa/σ,
`α Θ Θ' L σr`, τρ, FPS (+ physical units), rebuilt every 20th frame.
*tests:* token presence; cadence.

**S5.3 Slider HUD** — 5 sliders: sep 1–5 (3.0) → `spatial.
separation_weight`; coh 0–2 (0.2); align 0–0.5 (0.02); avoid 0–1 (0.05)
→ `boundary.avoidance_factor`; noise 0–0.5 (0.05). Ortho-pass track +
knob quads; `value(mx) = low + (high−low)·clamp((mx−x0)/w, 0, 1)`;
hit-rect ±6 px; drag locks (suppresses orbit); TAB toggles panel.
*tests:* mapping endpoints/midpoint pinned; drag writes the bound nested
field; TAB restores orbit-drag.

**S5.4 Interaction** — mouse spawn via cursor-ray unprojection:
`ndc = (2mx/w−1, 1−2my/h)`; `ray_eye = P⁻¹·(ndc, −1, 1)` with
`(x, y, −1, 0)`; `r̂ = normalize((V⁻¹·ray_eye).xyz)`;
`depth = median((p_i − o)·f̂)`; `spawn = o + r̂·depth/(r̂·f̂)` →
`SpawnAt` command; right-click → predator; `C` clear; `Q` quit;
PageUp/Dn: `flock.v0 ± 0.1` floor 0.3 (live). *tests:* synthetic-camera
unprojection hand-computed; right-click sets is_predator; C survives
metrics; PageUp floor respected.

**S5.5 CLI + facade** — repeatable `--set key.subkey=value` typed against
the nested schema + `--print-config`; `--fullscreen`;
`--light-scheme` → theme; `pymurmur.Simulation(**params)`;
`benchmark(flock_size, num_steps) → list[float]` (perf_counter_ns).
*tests:* `--set spatial.separation_weight=6 --set flock.num_boids=500`
reflected in `--print-config`; unknown key exits with the field list;
facade benchmark returns 20 positive floats.

**S5.6 Run logging to `output/`** *(headless-first observability)* —
every run (visual or `--no-viz`) writes a structured log
`output/run-<UTC-timestamp>.log` via the stdlib `logging` module:
run header (resolved config echo = `--print-config` content, seed, mode,
N, package version), one metrics line every `metrics.interval` frames
(the `FlockMetrics.to_dict()` payload of the fast fields), lifecycle
events (commands drained, mode switches, governor actions, golden-relevant
resets), and a run footer (frames, wall time, mean step ms). `--log-level
{debug,info,warning}` CLI flag; `viz`-only console echo at warning+.
No `print()` calls anywhere in `pymurmur/` (architecture-test rule —
extensions/presets currently print; route through `logging`). *impl:*
`core/logging.py` (setup helper), wired in `__main__.py`; engine/extension
call sites swap `print` → `logger`. *tests* (`test/test_cli.py`): a 30-frame
headless run creates the file with header/footer and ≥ 1 metrics line;
`--log-level debug` increases line count; AST guard — no `print(` in
package sources.

## S6 — EvoFlock  *(≈3 days; after S2-B + S6.4)*
Files: `analysis/evoflock.py`, `physics/obstacles.py` (new); tests
`test/analysis/test_evoflock.py`.

**S6.1 SSGA fidelity** — per update: select 3 → **evaluate all 3**
(fitness cache keyed on genome) → sort → **delete the worst of the 3**
(negative selection) → **uniform crossover** of the best two (each gene
from a random parent) + per-gene Gaussian mutation → insert in the freed
slot. Founders evaluated. *tests:* worst-of-3 gone; child mixes genes
from both parents (disjoint-value parents); all three finite fitness;
cache prevents re-simulation (call counter).

**S6.2 Worst-of-4 evaluation** — 4 sims per candidate, fixed per-sim
seeds, min-reduction (`eval_parallel` live; deterministic order).
*tests:* monkeypatched objectives [0.9, 0.8, 0.95, 0.7] → fitness 0.7;
seeds recorded.

**S6.3 Objectives** — separation on **nearest-neighbour** distance per
boid-step, trapezoid over body diameters (0 below 2, ramp 2→2.5,
plateau ≤ 4, ramp 4→5, 0 above); speed on `speed_real` band
[19, 21] m/s (ramps [18, 22]); curvature `κ = |v×a|/|v|³` per boid-step,
`score = clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)`; hypervolume
`F = Π max(o_k, 0.01)`. *tests:* trapezoid pinned at d/body ∈ {1.9→0,
2.5→1, 4→1, 5→0}; helix trajectory κ matches analytic ± 2 %; speed uses
`speed_real`.

**S6.4 SDF obstacle layer** — `physics/obstacles.py`: sphere
`‖p−c‖−r`; box `max(|p|−b)` (componentwise); cylinder; union = min,
subtract = max(a, −b); collision when
`sign(SDF(p_old)) ≠ sign(SDF(p_new))`; kinematic correction
`p ← p − SDF(p)·∇SDF/‖∇SDF‖` (numeric gradient ok); per-step collision
counter feeds `(f_cf)^500`. *tests:* SDF signs & surface zeros;
composition; zero-crossing on a straight path; correction lands
|SDF| < 1e-4; behavioural — obstacle course: collisions > 0 with zero
avoidance, ≈ 0 with evolved weights (`@slow`).

**S6.5 Missing behaviours/genes** — forward force
`w_fwd·sign(v* − |v|)·û`; per-behaviour `max_dist_{sep,align,coh}` and
`angle_{sep,align,coh}` perception cones (cos α ∈ [−1, 1]);
`fly_away_max_dist`; predictive avoidance (`min_time_to_collide`
look-ahead); fixed k = 7 topological neighbours; integer gene for σ;
`flock.speed_min_factor` as a gene. *tests:* forward force sign flips
around v*; cones exclude behind-cone birds (hand geometry); k enforced;
σ integer after decode.

**S6.6 Protocol** — persist best genome + Pareto front + per-run seeds +
objective scores to `output/evolved.yaml`; ship confined
(enclosure + obstacles) and open (`boundary: open`) evaluation configs.
*tests:* run(n_runs=2) writes the file; **experiment (`@slow`):** evolve
with NO alignment objective on the confined config → best genome's
settled α > 0.5 (the paper's emergent-alignment headline).

## S7 — MARL bridge  *(≈2 days; after D8 + S3.9)*
Files: `physics/forces/marl.py::MarlMode`, `analysis/gym_env.py`,
`scripts/{train_marl,rollout_marl}.py`; tests
`test/analysis/test_marl.py` (`pytest.importorskip("gymnasium")` where
needed). Unit map: `U = min(W,H,D)/6`, `v_cap = marl.velocity_cap·U`.

**S7.1 "marl" mode (deferred global rules)** — engine order for this
mode: control applies first (D8: `v += a_ext·action_scale·v_cap`,
component clip ±v_cap), **move**, then rules prep the *next* step:
`v += rule_weight·(F_sep(d < separation_radius·U) + (v̄ − v) +
(CoM − p))` with rule_weight 0.01 (global neighbourhood — no radius on
align/cohere). *tests:* two-step hand trace shows positions at step k
depend on rules from k−1 only; 0.01 scaling; clip bounds.

**S7.2 Gymnasium wrapper** — lazy import; `MurmurationEnv(config)`:
`observation_space = Box(−1, 1, (6N,))` — `concat((p−C)/3U, v/v_cap)`;
`action_space = Box(−1, 1, (3N,))`; seeded `reset` →
`p ~ C + U(−1,1)³·U`, `v ~ U(−0.1, 0.1)³·U`; truncate at
`marl.episode_steps` (500); reward from S3.9. *tests:*
`gymnasium.utils.env_checker.check_env` passes; obs ∈ [−1,1] over 500
random steps; same seed + same actions → identical obs; truncation at
500.

**S7.3 Scripts (dependency-gated)** — `train_marl.py`: PPO("MlpPolicy"),
5 000 timesteps, save `output/marl_ppo`; `rollout_marl.py`: 500
deterministic-predict steps → dual-view GIF; docstring notes the
centralized-MLP quadratic scaling and points to IPPO for large N.
*tests:* `@slow`, skip without sbl3 — 200-timestep learn() smoke; rollout
GIF ≥ 1 frame; **experiment:** trained policy's mean dispersion < random
policy's by ≥ 20 %.

---

# Part IV — Unified sequencing

```
D0 ─► D1 ─► D2 ─► D3 ─► D4 ─┬─► D5 ─┐
                            ├─► D6 ─┤
                            ├─► D7 ─┼─► D8 ─► D9
                            └───────┘
S1 (after D2) ─► S2 tracks A–E (parallel, after their gates)
S3 (after D3) · S4 (after D7) · S5 (after D8)
S6 (after S2-B + S6.4) · S7 (after D8 + S3.9)
```

| Phase | Days | Notes |
|-------|------|-------|
| D0–D9 foundation | 10½ | D2 and D5-golden re-pins; D5–D8 parallel after D4 |
| T0–T6 infrastructure | ≈6½ | interleaved with D-phases (not additive serial) |
| S1 correctness | 2 | re-pin projection/spatial/vicsek goldens |
| S2 tracks A/B/C/D/E | 4 / 3½ / 2 / 2 / 1½ | parallel; golden per track |
| S3 metrics | 2 | |
| S4 rendering/capture | 4 | independent of S2 physics tracks |
| S5 UX/tooling | 2 | |
| S6 EvoFlock | 3 | |
| S7 MARL bridge | 2 | |

**Total ≈ 38 working days single-track** (foundation + science + inline
tests); with two parallel streams (physics vs rendering/UX/tooling)
≈ **5–6 calendar weeks**. Every feature PR carries its inline test block —
the *accept* criterion **is** the test. If time is short, the highest
value-per-day cut: **D0–D4 → S1 → S2-A → S4.1–4.3 + S4.9**.

**Definition of done:** T1.2 reports zero orphan config fields; T4.1/T4.3
matrices green across all registered modes; golden set covers every mode;
full non-GL suite runs headless in CI; the two `@slow` experiments (S6.6
emergent alignment, S7.3 trained-beats-random) pass nightly.

---

# Appendix A — Input-coverage traceability

**todo_claude.md** — Part 1: §1→S1.1 · §2→S1.2 · §3→S1.3 · §4→S1.1 ·
§5→S2.B8 · §6→S2.B8 · §7→S2.B8 · §8→S2.B8 · §9→S3.3 · §10→S1.8+S3.4 ·
§11→S3.5 · §12→S3.6 · §13→S3.7 · §14→S1.6 · §15→S3.8. Part 2: T1→S1.1
tests · T2→S1.2 · T3→S1.3 · T4→S1.1a · T5→S1.1 property · T6→S1.6 ·
T7→S2.B8 · T8→S3.3 · T9→S1.8+S3.4 · T10→S3.5 · T11→S3.7 · T12→T4.4 ·
T13→T0.3 · T14→T0.2/D0 · T15→T1.5 · T16→T1.4. Part 3: E1–E3→D7 ·
E4→S5.1 · E5→S1.4 · E6→S5.2 · E7–E10→S4.9 · E11→D5 · E12→S2.B9 ·
E13→S1.8. Part 4 → absorbed into D/S phasing.

**todo_claude1.md** — §1→D1+T1.2 · §2→D6/D8 · §3→D5 · §4→D9 ·
§5→D7/D9.2/S4.9 · §6→D6/D2/D3/D1/D9 · §7→D0.2/T1.1–1.2.
**todo_claude2.md** — §1→D3/D5/T3.1/T4.1 · §2→S1.1/S1.5/S2.B6/D2-hygiene ·
§3→D3(rng) · §4→D9.4/S5.1/D7/S4.3 · §5→D2/D9.1/D1/D7 ·
§6→D7/T4.1/D9.6/test-pyramid.

**git1 (murmuratR)** — R0→D1 · R1→D2/S2.E1 · R2→S2.E1 · R3→S2.E2 ·
R4→S2.E3 · R5→S2.E4 · R6→S2.E5 · R7→S4.7/S4.8 · R8→S2.E tests.
**murmuratR 2nd-pass** (`todo_claude_sci6.md`, now merged) — §1.3 persistent
tick + protocol state slot→S2.E1/D2 · move-then-steer→S2.E2 · §2 Lissajous
(divisor freqs, two-scale amplitudes, +40 bias, C-centred)→S2.E1 · **§2.3
tunable trajectory params→S2.E1** (new optional `traj_*` fields) · §3
direction-state/distance-influence/rank-by-target/0.055-floor→S2.E2/E3 · §5
density init + zero directions→S2.E4 · §6 substep semantics + distance
diagnostics→S2.E2/E5 · §8 alpha-accum→S4.7, ortho views + fixed framing→S4.8.
**git2 (collective-motion)** — R0→D1 · R1→S1.7 · R2→D3/S2.D1 · R3→S2.D1 ·
R4→S2.D1 · R5→S2.D2 · R6→D5 · R7→S3.1 · R8→S3.2 · R9→S4.6 · R10→tests.
**collective-motion 2nd-pass** (`todo_claude_sci8.md`, now merged) — §0 config
loader→D1 · §1 Vicsek core (memory term, √(2DΔt) amplitude, tangent-plane
noise, fixed-speed contract)→S1.7/D4 · §2 predator-prey (fear blend,
weight_afraid, hunting, prey-only coupling, noise ratio, config fields)→
S2.D1/D1-VicsekConfig/D3 · §3 asymmetric collisions (np.add.at, min-image
before wrap)→S2.D2 · §4 MSD(τ) crossover + min-image + nematic Q-tensor→
S3.2/S3.1 · §5 prey/predator distinction + heading-hue→S4.6 · §6 phase-diagram
quick-snapshot mode→**S3.1**.
**git3 (rystrauss/boids)** — R0→D1 · R1→S2.B2(+S1.5) · R2→D5 · R3→S2.B3 ·
R4→S2.B6 · R5→S5.4 · R6→S5.5 · R7→D9.3/S5.5 ·
R8→S4.6/D7/S2.B9/S4.11 · R9→tests.
**rystrauss/boids 1st-pass** (`todo_claude_sci5.md`, verified — **no additions
needed**) — the source git3 was built from; all items already mapped:
§1 toroidal distance→D5/F4 · §2 predator boids→S2.B3 · §4 parallel→S2.B6/
S2.B10 · §5 sep_distance/perception/accel_scale/ceiling→S2.B1/S2.B2/S1.5 ·
§6 benchmark→S5.5 · §7 mouse spawn + facade→S5.4/S5.5 · §8 `--set`→S5.5 ·
§9 velocity noise→S2.B2 · §10 cube init→S2.B9 · §11 predator render/
accumulator/themes→S4.6/S4.11/D7 · §12 PRNG→F1/D3.
**rystrauss/boids 2nd-pass** (`todo_claude_sci9.md`, now merged) — Part 1
shared with git3 above (velocity init→S2.B9, seeded PRNG→D3/F1, accumulator→
S4.11, predator distinction→S4.6, hard-zero+escape→S2.B3, accum order→S2.B2,
toroidal kd-tree→D5, facade/benchmark→S5.5, CLI/spawn→S5.4/S5.5/S2.B6). Part 2
unique: render interpolation→S4.11 · fastmath policy→S2.B10 · ghost-cell wrap→
redundant (D5 modulo keys meet the goal) · speed-stretched ellipsoid + cone/
arrow meshes + theme material sets→**S4.4a** · Fresnel rim→**S4.2**.
**git4 (PyNBoids)** — R0→D1/D2 · R1→S2.C1 · R2→S2.C2 · R3→S2.C3 ·
R4→S2.C4 · R5→S2.C5 · R6→S2.C6/D5 · R7→S4.3 · R8→S2.C7/S4.6 ·
R9→excluded (App. B) · R10→tests.
**PyNBoids 2nd-pass** (`todo_claude_sci4.md`, verified — one addition) —
§1 angle steering→S2.C1 · §2 gated neighbour modes→S2.C2 · §3 adaptive speed→
S2.C3 · §4 cardinal edge avoidance→S2.C4 · §5 heading jitter→S2.C5 ·
§6 incremental grid + squared-dist kNN→S2.C6/D5 · §8 pixel-fade→S4.3
accumulation, ring-buffer trails→S4.3, **64³ volumetric (exotic)→S4.7** ·
§9 multi-flock + §11 screensaver + §12 desktop-overlay→excluded (App. B) ·
§10 body-unit radii→S2.C7 · §13 per-bird hue→S4.6, mesh selector→S4.4a.
**git5 (Starlings)** — R0→D1 · R1→S2.B1/S1.5/D4-fixed · R2→S2.B7/D4 ·
R3→S2.B4 · R4→S2.B5 · R5→S5.2/S5.3/S5.4 · R6→S4.4 · R7→S4.5 ·
R8→excluded (App. B) · R9→tests.
**Starlings 2nd-pass** (`todo_claude_sci3.md`, verified — **no additions
needed**) — the source git5 was built from; all items mapped: §1 kernel fixes
(sep 1/d², cohesion normalize, noise ×scale)→S1.5, fixed-speed→S2.B/D4 ·
§2 asymptotic wall + domain-centred sphere→S2.B7/D4 · §3 energy/physical-unit/
mass metrics→S2.B4 · §4 hybrid dual-constraint filter + split radii +
accept-first→S2.B1 · §5 parameter jitter→S2.B5 · §6 speed keys/sliders/
metrics panel→S5.4/S5.3/S5.2 · §7 winged mesh + flap + gradient sky→S4.4/S4.5 ·
§8 Hildenbrandt–Hemelrijk flight physics→excluded (App. B).
**git6 (crs48/murmuration)** — R0→D1 · R1→D3 · R2→S2.A1 · R3→S2.A2 ·
R4→S2.A3 · R5→S2.A4 · R6→S2.A5 · R7→S2.A6 · R8→S2.A7/D4 · R9→S2.A8/D6 ·
R10→S4.1–4.3/D7 · R11→S4.10/D8/D4-rails · R12→S2.A9 ·
out-of-scope→App. B.
**git7 (BirdMurmuration)** — R0→D1 · R1→D8 · R2→S7.1 · R3→S7.2 ·
R4→S3.8/S3.9 · R5→S7.3 · R6→S4.8/S4.9 · R7→tests.
**BirdMurmuration 2nd-pass** (`todo_claude_sci7.md`, now merged) — A: gym env→
S7.2 · control hook→D8 · deferred two-layer rules→S7.1 · gated train/rollout
scripts→S7.3. B: velocity_deviation + boundary_overshoot→S3.8 · **altitude
deviation `Σ|z−z_target|`→S3.8** (was missing) · rewards module→S3.9 ·
hard-radius separation→S2.B1 · **general `neighbor_filter: global`→S2.B1**
(was only in the marl mode) · dual-view→S4.8 · matplotlib GPU-free fallback→
S4.9. C: PPO training + centralized-policy-at-scale → excluded (Appendix B).
**git0** — F0–F8→D1/D3/D2/D4/D5/D7/D3/D8/D3 · W1→S1.1–1.4+S2.B7/B8 ·
W2→S2.A · W3→S1.5+S2.B · W4→S2.C · W5→S1.7+S2.D+S3.1/3.2 · W6→S2.E ·
W7→S7+S3.8/3.9+S4.8/4.9 · W8→S6 · W9→S1.8+S3 · W10→S4 · W11→S5 ·
P0–P8 phasing superseded here.
**arch.md** — module map/rules→D9 · numba two-pass→S2.B10 · scaling
table→S2.B10+D8-T6.1 · force-mode table→registry-generated (D9.6).

**predecessor audit** (`/Users/tralev/Developer/murmuration` vs current code
+ this roadmap; former `murmur2.md`, now merged) — four items that were in
the predecessor yet absent from *both* the `pymurmur/` codebase and this
roadmap: EMA-smoothed HUD readout → **S3.11**; marginal-opacity validation
(Pearce 0.30 target) → **S3.6a**; `fibonacci_sphere` Level-0 atom →
**D0.4** (+ global helpers); stochastic `predator_present(day, rng)` at
Goodenough's 0.296 rate → **S2.B8**. Everything else in the predecessor
(H₂/η(m)/m*, hull-τρ, silhouette Θ′, ecology functions, density estimators,
occlusion/steric, boundary/determinism/golden tests, presets, camera) was
already covered by the S/T/D items above.

# Appendix B — Deliberately excluded (scope decision)

Screensaver + desktop-overlay modes (git4 R9); GPU-compute simulation
backends (WebGPU/GPGPU blueprints, git6); Hildenbrandt–Hemelrijk flight
physics (git5 R8); CMA-ES benchmark and GP model evolution (EvoFlock
research directions); multi-flock parallax scenes (git4 R9); VR/XR
(arch.md "does not do"). Each stays documented in its source spec under
`sci/` if scope changes.

# Appendix C — Documentation-change checklist

| When | File | Change |
|------|------|--------|
| D1 | `arch.md` §2.1 | nested-config contract; live-vs-static tables into sub-config docstrings |
| D1 | `conf/*.yaml` | rewrite to the nested schema (documented-intent values) |
| D2 | `arch.md` §2.2, §6 | ForceMode protocol; force-mode table generated from `MODE_REGISTRY` |
| D5/D6/D7 | `arch.md` §5/§7/§8 | SpatialIndex, StepContext/extensions, InstanceSchema/renderer contract |
| D9 | `arch.md` §4/§5 | module map + dependency matrix final; doc-drift test enabled |
| D9 | `test.md` | retire or slim (superseded by this file) |
| S2.A9/S6.6 | `conf/` | field_* presets; evo confined/open configs |
| each S-track | `test/data/` | golden re-pins in the same commits as physics changes |
| continuous | this file | tick items; keep IDs stable (tests reference them) |
