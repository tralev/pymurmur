# roadmap1.md — Foundation phases (D0–D9) + test infrastructure (T0–T6)

Implements the contracts in [roadmap0.md](roadmap0.md). Each item carries
code targets, acceptance tests, and a **Status** line against the current
`pymurmur/` codebase (details and bug list: [roadmap5.md](roadmap5.md)).
Science items build on these phases: [roadmap2.md](roadmap2.md),
[roadmap3.md](roadmap3.md), [roadmap4.md](roadmap4.md). Sequencing table:
[roadmap4.md](roadmap4.md) §Sequencing.

Status legend ([roadmap0.md](roadmap0.md) §0): **DONE** (matches
contract) · **PARTIAL** (exists, incomplete) · **DIVERGES** (exists,
different semantics — decide & pin) · **MISSING** · **VERIFY**
(implemented but correctness unclear).

---

## D0 — Safety net before any refactor  *(½ day)*

1. **Golden trajectories** per existing mode: seeded 15-bird × 30-frame
   runs → `test/data/golden_<mode>.npz` (positions+velocities float32),
   `assert_allclose(atol=1e-3)` — tolerance, not bit-exactness (libm ulp
   differences amplify across platforms). Regeneration script
   `test/regenerate_golden.py`; policy: a deliberate physics change
   re-pins **in the same commit**, CI fails otherwise. (Harness: T0.2.)
2. **Architecture test** (`test/test_architecture.py`): the
   [roadmap0.md](roadmap0.md) §2 dependency matrix as data; AST-walk every
   module (function-level imports included); fail on any edge outside it.
   (Full spec: T1.1.)
3. **Entry-point freeze:** smoke tests — `--no-viz` runs 50 frames for
   each preset without crashing.
4. **`fibonacci_sphere(n)` Level-0 atom** — math in
   [roadmap0.md](roadmap0.md) §3.1. *tests*
   (`test/core/test_types.py`): all rows norm 1 ± 1e-6; n=0 → shape
   (0,3); n=1 → one unit vector (no NaN); n=256 → each of the 8 octants
   holds ≥ 1 point (coarse uniformity).

**Accept:** suite green; goldens committed; import matrix enforced;
`fibonacci_sphere` atom tested.

**Status: DONE (verify golden coverage).** Goldens exist
(`test/data/golden_{field,influencer,spatial}[_sphere].npz`),
`test/regenerate_golden.py` and `test/test_architecture.py` exist,
`fibonacci_sphere` is implemented in `core/types.py`. **Verify** that a
golden exists for *every* registered mode (projection, vicsek, angle
included) and that `test_golden.py` is parametrized over
`MODE_REGISTRY`, not a hand list.

## D1 — Nested configuration layer  *(2 days; the enabling refactor)*

**Problem:** a flat config approaching 150 fields; a YAML loader that
flattens section keys unprefixed silently drops whole sections and lets
`capture.width` overwrite the domain width.

**Target shape:** the nested `SimConfig` of [roadmap0.md](roadmap0.md)
§4.1, with the loader/validate/live-mutation contracts stated there.

**Migration mechanics (reviewable diff):**
1. Introduce nested classes; keep temporary **property shims** on
   `SimConfig` for every legacy flat name (`phi_p → self.projection.phi_p`,
   `capture_width → self.capture.width` …) so call sites keep working.
2. Migrate call sites package-by-package to dotted access
   (physics → viz → capture → analysis), deleting each shim as its last
   user converts.
3. Rewrite `conf/*.yaml` to the nested schema.
4. Delete the shim block; a test asserts no legacy names remain.

**Tests (T2, `test/core/test_config.py`):**
1. Round-trip `to_file → from_file` equality on every field, nested.
2. Preset survival (parametrized over `conf/*.yaml`): `domain.*` match the
   file's `domain:` section (the historical corruption case) + one
   sentinel per section (e.g. `vicsek.couplage`) equals the YAML value.
3. Unknown keys warn (`pytest.warns`); known keys don't.
4. Validation: σ = −3, v0 = 0, depth = 0 raise/clamp per contract.
5. Live-mutation smoke: mutate `cfg.projection.phi_p` mid-run → next step
   reads it.
6. Shim retirement: `not hasattr(SimConfig(), "phi_p")` after step 4.

**Accept:** all presets load with correct domains; round-trip green; zero
legacy flat-field access.

**Status: PARTIAL / DIVERGES.** The codebase has composed sub-dataclasses
(`DomainConfig` … `CaptureConfig`) with sub-config accessors, YAML
round-trip, aggregated `validate()`, `__copy__`, `__eq__` — good. But:
- Flat access is the *primary* interface via a `_FIELD_MAP` +
  `__getattr__/__setattr__` delegation; call sites everywhere use
  `config.width`-style names. That is the "shims" stage of the migration,
  never retired (only `phi_p` is setter-only).
- The loader still **flattens** nested YAML with section-aware key
  normalisation — the collision class the contract forbids is prevented
  only by prefixed leaf names (`capture_width`), not structurally.
- Leaf names carry prefixes (`field_separation` inside `FieldConfig`)
  instead of the clean nested names.
- **Missing sections/fields:** `AngleConfig` (angle-mode params are
  currently read via `getattr(config, 'turn_rate', …)` defaults — dead
  knobs), `MarlConfig`, `ThreatConfig` extras (`mode`, `acceleration`,
  `vacuole_strength`, `blackening_gain` are getattr-only),
  `boundary.margin`, `flock.speed_min_factor`, `flock.n_predators`,
  `spatial.separation_distance/neighbor_filter/separation_kernel/
  noise_mode/speed_mode/flow_weight`, `field.unit_scale/flow_pull/
  disabled_terms`, `projection.max_visibility/max_occlusion_neighbors`,
  `metrics.readout_smooth/altitude_target/history_cap`,
  `viz.per_bird_color/background/background_top/bottom/bird_mesh/
  flap_period/hud`, `perf.use_numba/fastmath/num_threads/
  adaptive_quality`, `velocity_init: "drift"` alias.
- Decide explicitly (record in [roadmap0.md](roadmap0.md) if amended):
  either finish the migration to pure nested access, or declare the
  flat-map layer permanent and delete the retirement items. Until
  decided, T2.6 cannot land. Dead-field list: [roadmap5.md](roadmap5.md).
- **First concrete retirement — finish `phi_p`.** `phi_p` is already
  setter-only (`_SETTER_ONLY`, excluded from `__eq__`, validated via
  `cfg.projection.phi_p`); complete it: remove the `phi_p` entry from
  `_SETTER_ONLY` (writes go through `cfg.projection.phi_p` like reads),
  keep the explicit sub-config validation, and update the config tests
  (`__eq__` exclusion note, round-trip writer) in the same commit. This
  is the template PR for retiring every other flat name in migration
  step 4.

## D2 — ForceMode protocol and registry  *(2 days; breaks the E-cycle)*

**Problem:** modes as stateless functions have nowhere to keep
time/caches; lazy imports create a Level-1 cycle; vicsek and influencer
need to own speed/positions and otherwise fight `integrate()`.

**Target:** the protocol of [roadmap0.md](roadmap0.md) §4.2 and the
engine step of §4.3. The five existing modes become
`@register class SpatialMode(ForceMode)` … — mechanical wraps; per-mode
state becomes instance attributes (influencer `self.tick`, field
`self.t` + group cache). Lazy imports move to module top (never inside
per-bird loops). `PhysicsFlock` no longer imports forces. New mode = one
file + `@register` (`M` cycles `sorted(MODE_REGISTRY)`); `needs_index`
replaces the hardcoded index-mode set.

**Tests (T3.2, `test/physics/forces/test_mode_contract.py`,
parametrized over `MODE_REGISTRY.values()`):** registered & instantiable;
step respects the active mask (inactive rows untouched); reset + same
seed → same trajectory; `speed_mode` honored (T0.3 speed contract);
AST: no module-level `np.random.*` in mode sources; `owns_positions`
flag truthful (move=False modes actually moved birds).

**Accept:** import cycle gone (architecture test); all modes green
against goldens (pure refactor); `M` cycles registry order.

**Status: PARTIAL / DIVERGES.** `MODE_REGISTRY` + `@register` + six mode
classes exist (`projection, spatial, field, vicsek, influencer, angle`)
and `mode_needs_index()` reads the class flag — good. But:
- `compute()` is a **static 8-array-arg function**, not the stateful
  `step(flock, config, dt)` instance method; there is no `reset()`; mode
  time lives in config privates (`config._field_time`,
  `config._influencer_tick` written by the engine each step).
- The engine **ignores** `speed_mode` and `owns_positions`:
  `flock.integrate()` always runs `speed_mode="band"`, `move=True`.
  `InfluencerMode` declares `owns_positions=True` yet never moves
  positions — the flag is untruthful today (see
  [roadmap2.md](roadmap2.md) S2.E2 and [roadmap5.md](roadmap5.md)).
- Remaining work: convert compute→stateful step (or formally amend the
  contract), wire `speed_mode`/`owns_positions` through
  `engine._step_physics → flock.integrate`, move per-mode state onto
  instances, add the T3.2 contract suite (a
  `test_mode_contract.py` exists — **verify** it asserts the truthfulness
  items above).

## D3 — Flock state contract  *(1 day)*

The columns of [roadmap0.md](roadmap0.md) §4.4, plus the array
conventions of §3.2, as one documented contract. `FlockArrays` composed
(`flock.arrays`) with forwarding properties.

**Tests (T4):**
- *T4.1 holey-mask matrix:* every mode × `holey_flock` fixture × 20 steps
  — no exception, invariants hold, inactive rows untouched, force clamps
  applied to active rows only.
- *T4.2 lifecycle:* `add_boids` past capacity (`_extend`),
  `remove_boids`, `SpawnAt`, `Clear` — species column and `seeds`
  carried; index rebuilt; metrics survive `N_active == 0`.
- *T4.3 determinism matrix:* same seed → bit-identical positions after
  100 steps, for (mode × `num_threads ∈ {1,-1}` × jitter on/off × numba
  on/off once [roadmap2.md](roadmap2.md) S2.B10 lands); two in-process
  runs + one subprocess run.

**Accept:** matrices green; seed-based features read documented
semantics.

**Status: PARTIAL.** All contract columns exist (`rng`, `is_predator`,
`center` EMA, `prev_positions`, `last_accelerations`, `max_speed`) and
are carried through `_extend/add_boids/remove_boids/spawn_at`. Gaps:
- **Seed bug:** `np.random.default_rng(config.seed if config.seed else 0)`
  conflates `seed=0` with `seed=None` and makes unseeded runs
  deterministic-0 — fix to honor `None` (entropy) vs integer seeds.
- `FlockArrays` composition not done (arrays live directly on
  `PhysicsFlock` — acceptable if the contract is amended; decide).
- `spawn_at` hardcodes speed `4.0` instead of `config.v0` and ignores
  `velocity_init`.
- Determinism matrix: verify T4.3 exists at full breadth (worker counts,
  subprocess run).

## D4 — Integration contract  *(½ day)*

`integrate()` per [roadmap0.md](roadmap0.md) §4.5 — speed modes, `move`,
`inertia`, `noise_velocity`, band floor from
`cfg.flock.speed_min_factor`, dt clamp + `isfinite3` guard at callers,
zero-allocation scratch rail, and **sphere boundaries centred on the
domain centre**.

**Tests (T3.3, `test/physics/test_boid.py`):** all
(speed_mode × move × inertia ∈ {0, 0.8}) combinations vs hand-computed
expectations; per-bird `max_speed` respected; `speed_min_factor`
honored; toroidal wrap exactness; margin containment; sphere centred on
C (regression); NaN guard heals; allocation hygiene (`@slow`) —
tracemalloc delta between frame 100 and frame 600 of a headless
N = 2 000 run < 1 MB (no per-frame `(N,3)` churn accumulating).

**Accept:** matrix green; centre-initialised flock stays centred in
sphere mode; NaN injection self-heals. *(Physics-visible: re-pin sphere
golden.)*

**Status: PARTIAL, one live bug.** The kernel implements all four speed
modes, `move`, `inertia`, dt clamp `[0, 0.05]`, deterministic zero-speed
fallback `(minSpeed, 0, 0)`, and the NaN→center guard. Gaps:
- **BUG — sphere boundary is origin-centred**
  (`_sphere_soft` uses `‖p‖`, not `‖p − C‖`): with the default domain
  `[0,1000]×[0,700]×[0,400]` every bird is permanently "outside" and gets
  projected onto the origin-sphere. Fix + re-pin the `*_sphere` goldens.
- `speed_min_factor` is a function default (0.3), not a config field.
- `noise_velocity` parameter absent.
- No scratch buffers (`flock.scratch3`) — allocation-hygiene rail not
  started; add the tracemalloc test.

## D5 — SpatialIndex protocol  *(1 day)*

**Problem:** the two indexes historically disagreed on index space and
neither was wrap-aware; force modes bypass them and build private trees.

**Target:** the protocol of [roadmap0.md](roadmap0.md) §4.6.

**Tests (T3.1, `test/physics/test_index_contract.py`, parametrized over
both implementations, on the holey fixture):** returns global indices;
knn closest-first, no self, no dupes; radius complete vs brute force;
radius argument honored (grid); toroidal cross-seam (x=1 vs x=W−1 mutual
at r=3); implementations interchangeable (identical neighbour sets);
batch == per-row.

**Accept:** conformance suite green on both; zero `cKDTree(`
constructions inside `forces/` (grep test). *(Physics-visible: re-pin
affected goldens.)*

**Status: PARTIAL.** `KDTreeIndex` maps compacted→global; the hash grid
has modulo-wrapped cell keys and min-image knn distances. Gaps:
- `KDTreeIndex.rebuild` never passes `boxsize` —
  `boundary.use_toroidal_distance` is dead for the KDTree path.
- `SpatialHashGrid.query_radius(pos, radius)` **ignores the radius**
  (always the fixed 27-cell neighbourhood).
- `query_knn_batch` is not part of the protocol; modes reach for the raw
  `.tree` property instead.
- Private tree builds persist: `spatial.py::_query_neighbors`,
  `angle.py`, `vicsek.py` all construct `cKDTree` when
  `index.tree is None` (hash-grid case) — the grep test would fail.
- `rebuild(positions, active, box)` signature lacks the `box` argument.

## D6 — Extension protocol widening  *(½ day)*

**Target:** `StepContext` + `Extension.apply(flock, ctx)` per
[roadmap0.md](roadmap0.md) §4.7; manager re-reads config each frame
(live T/K toggles); Ecology public read surface (`predator_active`,
`hour`, `day`, `roost_position`); the Threat FSM
([roadmap2.md](roadmap2.md) S2.A8) replaces the simple predator.

**Tests (T3.4):** extensions receive real dt/frame/rng; flipping
`threat.mode` / `ecology.roosting_enabled` mid-run takes effect next
frame; `threat_prox` published shape `(N,)` in [0,1]; `predator_active`,
`hour`, `day`, `roost_position` public and consistent with the advance
rate.

**Status: MOSTLY DONE.** `StepContext` (frame/dt/rng/center/config/
threat_prox) and per-frame lazy lifecycle toggles are implemented; the
Predator publishes `ctx.threat_prox`. Gaps: Ecology exposes only
`predator_active` and `coherence_factor` — add `hour`, `day`,
`roost_position` properties; there is no `threat.mode` field yet
(ThreatConfig, [roadmap2.md](roadmap2.md) S2.A8). Also **BUG:** the
Wander extension reads `cfg.wander_speed` / `cfg.attractor_radius` via
getattr, but the config defines `wander_attractor_speed` /
`wander_attractor_radius` — both fields are dead and wander silently
runs on hardcoded defaults (speed 1.0 vs configured 0.10).

## D7 — Renderer contract  *(1 day)*

**Target:** [roadmap0.md](roadmap0.md) §4.8 — one InstanceSchema, VAO
rebuilt at init and after every buffer growth, headless FBO with depth
attachment, `_mat4_bytes` uploads, mesh registry + themes from
`config.viz`, dual-view as two passes, HUD as ortho pass,
`Visualizer.frame()` render-only, `draw_layer` for markers.

**Tests (T5, `test/viz/`, `@gl` except 5.4):**
1. FBO has a depth attachment; VAO rebuilt after growth (grow past chunk
   → draw → nearest bird's colour wins at overlap); schema packs
   flag/hue; `_mat4_bytes` equals `to_list()` reference.
2. Render purity: `headless_frame()` twice → `sim.frame` unchanged;
   Recorder path reuses Visualizer.
3. Mode smoke: each mesh × each trail mode × dual-view renders one frame
   without GL errors; loose screenshot-hash regression.
4. *(no gl)* GPU-free capture: with moderngl monkeypatch-failing,
   `--no-viz --capture` produces ≥ 1 GIF frame via the matplotlib
   fallback with an asserted warning (replaces a silent `except: pass`).
5. Capture pipeline: pre-warm frames not captured; sweep moves the
   camera between captured frames; env-override precedence
   YAML < env < CLI; GIF saved `optimize=True, disposal=2`;
   `capture.width/height` honored.
6. End-to-end headless capture: a short headless run through
   `Visualizer → headless_frame() → Recorder → save_gif()` produces a
   file that PIL re-opens as a valid multi-frame GIF (≥ 1 frame,
   expected dimensions) — exercises impostors, sweep/pre-warm, and the
   fallback seam together rather than only in isolation.

**Status: PARTIAL.** Implemented: `InstanceSchema` dataclass (6 floats
pos+vel, with hue/scale in a **separate** colour VBO — functionally
covers flag/hue but diverges from the 8-float single-schema contract;
decide and pin — `test/viz/test_renderer.py::
test_renderer_single_memcpy` is the pin: it currently asserts **2**
`vbo.write` calls per frame; merging to the 8-float schema restores the
single-memcpy contract and re-tightens that test to 1 write),
`_build_vao` re-run after growth, `_mat4_bytes`,
theme tables threaded, dual-view passes, render-only
`frame()/headless_frame()`, Recorder reusing Visualizer, mpl fallback.
Gaps:
- **BUG — headless FBO has no depth attachment**
  (`framebuffer(color_attachments=[texture])` only) — captures render in
  draw order.
- `draw_layer` missing — the threat and influencer targets are invisible
  ([roadmap2.md](roadmap2.md) S2.A8/S2.E5).
- Mesh registry limited to tetra/winged/impostor
  ([roadmap3.md](roadmap3.md) S4.4a adds ellipsoid/cone/arrow/points).
- Env-override precedence is env > CLI (see [roadmap5.md](roadmap5.md)).

## D8 — Engine seams: control, commands, quality  *(1 day)*

Per [roadmap0.md](roadmap0.md) §4.9.

**Tests (T3.5, `test/simulation/test_engine.py`; T6):** commands drained
at step start, headless; `step(control=0)` bit-identical to `step()`;
control clipping bounds; queue survives interleaving with reset; metrics
survive `N_active == 0`; governor pure-logic ladder test (T6.2);
`benchmark(num_steps=50)` returns 50 positive floats (T6.1) + `@slow`
per-mode step-time bounds at N = 2 000 (×3 headroom) and scaling
checkpoints once [roadmap2.md](roadmap2.md) S2.B10 lands.

**Status: PARTIAL.**
- Command queue: DONE as a concrete `CommandQueue` (add/remove/reset/
  spawn/clear) drained in `step()` and by the viz loop — equivalent to
  the contract (generic `enqueue(cmd)` optional).
- **Control hook MISSING:** `step(frame_dt)` has no `control` parameter.
  Prerequisite for the MARL bridge ([roadmap4.md](roadmap4.md) S7).
- Quality governor: `QualityGovernor` (EMA, spike cap 250 ms, 0.78/1.8 s
  degrade, 3.6 s recovery, 3-step ladder) and
  `Visualizer._apply_quality_actions()` exist, **but the render loop
  never calls `governor.feed()` nor `_apply_quality_actions()`** —
  adaptive quality is dead code. Wire it; risk classifier still to add
  ([roadmap3.md](roadmap3.md) S4.10).

## D9 — Analysis split, facade, exports, cleanup, doc sync  *(1 day)*

1. **Analysis tiers** per [roadmap0.md](roadmap0.md) §4.10. —
   **Status: DONE** (observables vs drivers split matches).
2. **Metrics export schema:** `FlockMetrics.to_dict()` as the declared
   interface; Recorder and gym env consume it. *Test (T3.6):*
   `to_dict()` JSON round-trips; key set pinned; Recorder CSV headers
   equal the schema. — **Status: DONE** (`to_dict` with NaN/inf→None;
   Recorder consumes it). Schema extensions:
   [roadmap3.md](roadmap3.md) S3.10.
3. **Public facade:** exports + `benchmark`. — **Status: DONE**
   (`Simulation` facade with `run/metrics_history/benchmark`).
4. **Dead-code retirement:** delete `BoidView`; the `ForceKernel` type is
   fulfilled by numba kernels ([roadmap2.md](roadmap2.md) S2.B10) or
   deleted; legacy config shims removed; every remaining atom has a
   composer. — **Status: PARTIAL** — dead config fields and unconsumed
   atoms remain; register in [roadmap5.md](roadmap5.md).
5. **Dependency matrix** — [roadmap0.md](roadmap0.md) §2, enforced by the
   architecture test. — **Status: VERIFY** that
   `test/test_architecture.py` encodes exactly that matrix including the
   named regression edges.
6. **Doc sync:** the force-mode table and module map in
   [roadmap0.md](roadmap0.md) are kept in sync with
   `sorted(MODE_REGISTRY)` and the package layout; doc-drift test T1.4
   checks links **among roadmap0–roadmap5 only**. The per-phase
   documentation-change checklist lives in
   [roadmap5.md](roadmap5.md) Appendix B; the item index in
   [roadmap0.md](roadmap0.md) §6.

**Accept:** architecture test enforces the matrix;
`pymurmur.Simulation(num_boids=50).benchmark(num_steps=10)` works;
doc-drift test green; grep finds no `BoidView`.

---

# Test infrastructure (T0–T6)

*(Feature tests live inline with their items in
[roadmap2.md](roadmap2.md)/[roadmap3.md](roadmap3.md)/
[roadmap4.md](roadmap4.md); this part is the harness they stand on.)*

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

**T0.3 Invariant fuzz** (`test/physics/test_invariants.py`) —
parametrized (mode × boundary), 200 random seeded states each: no NaN
after 50 steps; toroidal positions in `[0, L)` every axis; speed
contract per mode (band within `[min_factor·v0−ε, v0+ε]`; fixed
`== v0 ± 1e-5`; ceiling `≤ v0`); inactive rows bit-identical across
steps.

**Status: PARTIAL** — `conftest.py`, `helpers.py`, `test_golden.py`
exist; **verify** the holey fixture and the full T0.3 matrix breadth.

## T1 — Architecture & drift guards  *(with D0/D9; 1 day)*

- **T1.1 Import-rule matrix** (`test/test_architecture.py`): the
  [roadmap0.md](roadmap0.md) §2 matrix as data; AST walk (function-level
  imports too); named regression edges
  (`physics.flock !→ physics.forces`; `viz.input_control !→ simulation`;
  no `cKDTree(` in `forces/`; no module-level `np.random.*`).
- **T1.2 Config-usage drift:** every `SimConfig` leaf field (recursed)
  must be read by ≥ 1 non-config module (AST attribute-access scan); fail
  with the orphan list. Retro-covers the known dead fields
  ([roadmap5.md](roadmap5.md) §3).
- **T1.3 Strictly-3D guard:** AST scan for `(…, 2)`-shaped spatial arrays
  in `physics/` → fail; `validate()` enforces `depth > 0`; invariance
  tests use random SO(3), not z-only.
- **T1.4 Doc-drift** (`test/test_docs.py`): every module path named in
  roadmap0–roadmap5 exists; every intra-repo markdown link across
  roadmap0–roadmap5 resolves; the force-mode list in
  [roadmap0.md](roadmap0.md) matches `sorted(MODE_REGISTRY)`.
- **T1.5 Collection-count guard:** collected-test count pinned per
  subpackage (update deliberately).

**Status: PARTIAL** — `test_architecture.py`, `test_config_drift.py`,
`test_docs.py`, `test_collection_count.py` exist. Given the dead fields
found in [roadmap5.md](roadmap5.md), either T1.2 is not enforcing the
"every leaf read" rule or it has an allowlist — tighten it; it is the
single highest-leverage guard in this plan.

## T2 — Config suite — folded into D1 (above).
## T3 — Contract suites — folded into D2–D6, D8, D9.2 (above).
## T4 — Composition/determinism — folded into D3 (above); plus:

**T4.4 Metamorphic invariances**
(`test/analysis/test_metrics_invariance.py`) — random flocks, 30 trials:
order parameter & nematic S invariant under random SO(3) rotation;
dispersion/gyration translation-invariant; `[0,1]`-metrics within
bounds; permutation invariance (shuffling bird order changes nothing).

**Status: PARTIALLY DEDUPLICATED — finish the job.** The deleted
`test/analysis/test_metrics_invariance.py` overlapped
`test_metrics.py`, which still carries the nematic invariances
(`test_compute_nematic_invariant_under_sign_flip`,
`test_compute_nematic_SO3_invariant`) — so the deletion reads as an
intentional dedup, **but the rest of the T4.4 matrix has no home**: α
rotation-invariance, dispersion/gyration translation-invariance,
permutation invariance (bird-order shuffle changes nothing), and the
`[0,1]`-bounds sweep are currently untested. Land those (in
`test_metrics.py` or a restored invariance file) to complete T4.4.

## T5 — Viz/capture suites — folded into D7 (above).
## T6 — Perf/quality guards — folded into D8 (above); plus:

**T6.3 Soak / bounded-memory test**
(`test/test_soak.py`, `@slow`, nightly): one long headless run
(≥ 20 000 frames, Recorder + metrics attached, N ≈ 500) — tracemalloc
growth after a 500-frame warm-up < 5 % of the warm-up footprint; the two
known unbounded accumulators become capacity-capped ring buffers
(`MetricsCollector.history` capped by `cfg.metrics.history_cap`,
Recorder frame list capped by `cfg.capture.frames`) and the test asserts
their lengths saturate at the caps; positions finite and in-domain
throughout (T0.3 invariants re-checked at soak scale).

**Status: MISSING** — both accumulators are currently unbounded; no
`history_cap` field; no soak test.
