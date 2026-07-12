# TODO — Improving the Macro-to-Micro Design

Comparison of the actual codebase (`pymurmur/`) against `functional_decomposition.md`
(Level 0 goal → Level 1 subsystems → Level 2 meso design → Level 3 module interfaces → Level 4 nano details).

The macro levels (0–2) are healthy: the six-subsystem split is real in the code, the
engine is genuinely headless, and viz/capture are optional layers. The weaknesses are
all in **traceability** — places where a macro-level promise (a config field, a
dependency rule, a documented mechanism) never arrives at the micro level, or where
the micro level grew structure the macro levels don't know about. Items below are
ordered roughly by how much they undermine the macro→micro chain.

---

## 1. Contract integrity — macro promises with no micro implementation

`SimConfig` is described (Level 2) as "the shared parameter contract between every
subsystem", but several fields are **dead ends**: they exist in the contract and in
YAML presets, yet nothing downstream reads them.

- [ ] **`use_numba` is dead.** No numba code exists anywhere in the package
  (`grep numba` only hits comments). Either implement the documented numba JIT pass
  or remove the field and correct the doc's Level 4 performance claims.
- [ ] **`spatial_index: "auto" | hash_grid | kdtree` is ignored.**
  [flock.py:41-45](pymurmur/physics/flock.py#L41-L45) hardcodes the `N >= 5000`
  threshold and never consults `config.spatial_index`. Honor the field (explicit
  override + auto default).
- [ ] **All predator params are ignored.** `predator_threat_radius`, `predator_strength`,
  `predator_momentum`, `predator_split_gain` exist only in
  [config.py:94-97](pymurmur/core/config.py#L94-L97).
  [predator.py](pymurmur/physics/extensions/predator.py) hardcodes 200.0, 0.5, 8.0,
  and `dt = 0.016`, and its constructor takes no config at all.
- [ ] **All ecology params are ignored** (`ecology_roost`, `ecology_critical_mass`) —
  same pattern: `Ecology()` is constructed with no arguments.
- [ ] **`capture_width` / `capture_height` are ignored.**
  [recorder.py:62-66](pymurmur/capture/recorder.py#L62-L66) builds its headless
  renderer with `window_width`/`window_height` instead.
- [ ] **Add a drift test**: a test that asserts every `SimConfig` field name is
  referenced by at least one non-config module. This turns "the config is the
  contract" from prose into an executable invariant, and would have caught all five
  items above.

## 2. Live-mutability promises not delivered

Level 2 explicitly lists `predator_enabled`, `roosting_enabled` as **live-mutable**
and binds them to the T/K keys.

- [ ] **Toggling extensions at runtime does nothing.**
  [ExtensionManager.__init__](pymurmur/physics/extensions/__init__.py#L27-L40)
  instantiates only the extensions enabled *at construction time*;
  `pre_step()` never re-reads the config. Pressing T/K mutates flags nobody looks at
  until reset. Fix: check `config.*_enabled` each `pre_step()` (lazily create/drop
  extension instances), and pass `config` into `pre_step` — see item 6.
- [ ] **Live `num_boids` only works through the viz path.**
  [visualizer.py:80-92](pymurmur/viz/visualizer.py#L80-L92) applies
  `pending_add/remove/reset` inside the render loop, so the headless engine can never
  respond to these commands. Move this into a small command queue that
  `SimulationEngine.step()` (or a `pre_step` hook) drains — then the same control
  surface works headless, and Subsystem C stops containing simulation-lifecycle logic.

## 3. The spatial-index abstraction is designed but bypassed

Level 2/4 present the spatial index as *the* scaling mechanism ("rebuild spatial index
(hash grid or cKDTree)" → forces read it).

- [ ] **Force modes build their own trees, so the index rebuild is wasted work.**
  `flock.step()` rebuilds `self._index` for spatial/projection modes
  ([flock.py:52-53](pymurmur/physics/flock.py#L52-L53)), but
  [spatial.py:_query_neighbors](pymurmur/physics/forces/spatial.py#L62-L82) and
  [vicsek.py](pymurmur/physics/forces/vicsek.py#L48-L49) each construct a **second
  cKDTree from scratch every frame** and never call `flock.get_index()`. Two tree
  builds per frame at 300K birds directly contradicts the Level 4 performance budget.
  Make force modes consume the flock's index, or delete the flock-owned index.
- [ ] **The two index implementations disagree on index space.**
  `SpatialHashGrid.query_knn` returns **global** bird indices;
  `KDTreeIndex.rebuild` builds from `positions[active]` so its `query_knn` returns
  **compacted active-subset** indices ([flock.py:225-236](pymurmur/physics/flock.py#L225-L236)).
  Any caller works with one and silently breaks with the other once inactive slots
  exist. Define the contract (recommend: always global indices) in a shared
  `SpatialIndex` Protocol in `core/types.py`, and test both implementations against it.
- [ ] **The index never migrates as N changes.** It's chosen once in `__init__`;
  `add_boids()` can grow a hash-grid flock far past 5,000. Re-evaluate the choice when
  `N_active` crosses the threshold (or on `_extend`).
- [ ] **Vicsek needs neighbors but is excluded from `_INDEX_MODES`**
  ([flock.py:47](pymurmur/physics/flock.py#L47)) — consistent only because of the
  bypass above. Once forces use the shared index, add vicsek to the set. Also make
  `_INDEX_MODES` data-driven: let each force mode declare `needs_index = True/False`
  next to its registration in `_DISPATCH`, so adding a mode (Level 4 §4.6 extension
  point) is one edit, not two.
- [ ] **Batch the k-NN query.** `_query_neighbors` loops per bird in Python calling
  `tree.query(pos, ...)`. `cKDTree.query` accepts the whole `(N, 3)` array — one
  vectorised call replaces the N-iteration Python loop. Without this (and item 1's
  numba work) the doc's "300K @ 17 ms spatial" table is fiction; either fix the code
  or fix the table.

## 4. Micro structure the macro decomposition doesn't know about

- [ ] **The Level 3 module map no longer matches the tree.** The doc lists a flat
  layout (`pymurmur.py`, `simulation.py`, `forces.py`, …); the code is a nested
  package (`core/`, `physics/forces/*`, `physics/extensions/*`, `viz/`, `analysis/`,
  `capture/`, `simulation/`). The nesting is *better* than the doc — it makes Level 1
  subsystems visible as directories — so update the doc's §3.1/§3.2 to the real
  layout rather than flattening the code.
- [ ] **Subsystem F grew four unmapped modules**: `analysis/perf.py`,
  `analysis/phase_diagram.py`, `analysis/density_scaling.py`, `analysis/evoflock.py`.
  None appear in any decomposition level. Add them to the doc — and note that they
  change the architecture (next item).
- [ ] **F now depends on B, inverting a Level 1 arrow.** The doc says B *owns* F and
  `metrics → physics_flock, config, types` only. But `evoflock.py`,
  `phase_diagram.py`, and `density_scaling.py` all import `SimulationEngine`.
  That's defensible (they are *experiment drivers*, not observables), but it should be
  a deliberate macro decision: split Subsystem F into **F1 Metrics** (pure, owned by B)
  and **F2 Experiments/Analysis drivers** (sits above B, like Subsystem D), and update
  the dependency rules accordingly.
- [ ] **Config fields grew past the documented contract.** The doc's `SimConfig`
  listing is missing `field_*`, `vicsek_*`, `influencer_*`, `boundary_sphere_*`,
  `theme`, `trails`, `point_sprites`, `capture_fps`, etc. Either regenerate that
  section from the dataclass (a 10-line script) or replace the doc's full listing with
  the grouping rationale and a pointer to `core/config.py` as the source of truth.

## 5. Broken/loose seams between subsystems

- [ ] **`Visualizer.frame()` and `headless_frame()` step the simulation**
  ([visualizer.py:49-64](pymurmur/viz/visualizer.py#L49-L64)). Rendering a frame
  mutating physics state violates the Level 0 principle "Simulation is pure…
  Visualization is optional" at the micro level (capturing a still advances the
  world). Make render methods take flock state and *only draw*; stepping belongs to
  the loop owner (`run()` / `run_headless()`).
- [ ] **`Recorder` duplicates the Visualizer instead of using it.** Level 2 says
  `Recorder` holds `Visualizer(sim, headless=True)` and calls `vis.headless_frame()`;
  the real [recorder.py:55-75](pymurmur/capture/recorder.py#L55-L75) hand-builds its
  own `Renderer3D` + `OrbitCamera` inline. Once the previous item lands
  (render-only `headless_frame`), delete the duplication and restore D → C as designed.
- [ ] **`except Exception: pass`** around the whole capture block
  ([recorder.py:79-80](pymurmur/capture/recorder.py#L79-L80)) silently produces
  frameless scientific runs. Catch the specific import/context errors once at
  Recorder construction, log a warning, and set `with_viz = False` — never per-frame
  blanket swallowing.
- [ ] **`snapshot().__dict__` leaks representation** across the D–F boundary
  ([recorder.py:52](pymurmur/capture/recorder.py#L52)). Give `FlockMetrics` an
  explicit `to_dict()` (handling ndarray → list there), so the export schema is a
  declared interface instead of whatever the dataclass happens to contain.
- [ ] **`ExtensionManager` reaches into `Ecology._predator_active`**
  ([extensions/__init__.py:58](pymurmur/physics/extensions/__init__.py#L58)) —
  a private attribute as a cross-object protocol. Promote it to a public property or,
  better, have `pre_step` pass a shared per-frame context (see item 6).

## 6. Interface-level (Level 3) refinements

- [ ] **Widen the `Extension` protocol.** `apply(flock)` gives extensions no `config`
  (why the predator/ecology params are dead, item 1), no `dt` (why the predator
  hardcodes 0.016), and no `frame`. Change to
  `apply(flock, config, dt, frame)` or a `StepContext` dataclass; this single change
  unblocks items 1 and 2 and removes the `_predator_active` hack (ordering/gating can
  live in the context).
- [ ] **Break the `flock ↔ forces` cycle explicitly.** Today it's papered over with a
  lazy import inside `PhysicsFlock.step()` ([flock.py:56](pymurmur/physics/flock.py#L56)).
  Cleanest: move orchestration up — `SimulationEngine.step()` calls
  `flock.rebuild_index()`, `compute_all_forces(flock, config)`, `flock.integrate(...)`
  itself. Then `flock.py` is pure state+index (never imports forces), `forces` depends
  on flock one-way, and the engine visibly *is* the Level 2 step diagram.
- [ ] **Either use `FlockArrays` or delete it.** The Level 3 shared type is never
  instantiated — `PhysicsFlock` re-declares the same six arrays as loose attributes.
  Composing it (`self.arrays = FlockArrays(...)` or `PhysicsFlock(FlockArrays)`) would
  let forces/metrics/renderer accept the plain data container instead of the full
  flock, shrinking those interfaces. `ForceKernel` in `core/types.py` is likewise
  unused (no kernels exist — see item 1); delete or implement.
- [ ] **Split `SimConfig` into per-subsystem sections.** ~90 flat fields spanning six
  subsystems is a god-object; the YAML is already nested (`domain:`, `flock:`,
  `capture:`…) and `from_file` flattens it away. Mirror the Level 1 decomposition in
  the type: `SimConfig` composed of `DomainConfig`, `FlockConfig`, per-mode configs,
  `VizConfig`, `CaptureConfig`, `PerfConfig`. Each subsystem then receives only its
  slice, and "which subsystem owns this parameter" becomes a compiler-visible fact
  instead of a comment convention. (Keep flat attribute aliases during migration.)
- [ ] **Give the package a public facade.** `pymurmur/__init__.py` exports nothing.
  Re-export the outside-in surface: `SimConfig`, `SimulationEngine`, `Recorder`
  (viz stays import-on-demand). Library users shouldn't need to know internal module
  paths that Level 3 says may change.

## 7. Enforcement — make the decomposition executable

- [ ] **Extend `test/test_imports.py` to the full §3.2 dependency matrix.** It
  currently checks ~4 of the rules. Generalise it: encode the whole allowed-imports
  table from the doc as data, walk every module's AST (it already catches
  function-level lazy imports), and assert no edge outside the table. Every future
  drift item in this file then becomes a failing test instead of an archaeology
  session.
- [ ] **Add the config-usage drift test** (item 1) and a **doc-sync check** (item 4,
  e.g. regenerate the module map / config listing into the doc via a script in
  `scripts/`), so the macro document and the micro code cannot silently diverge again.
