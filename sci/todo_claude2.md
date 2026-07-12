# TODO — Improving the Micro-to-Macro Design

Comparison of the actual codebase (`pymurmur/`) against `functional_synthesis.md`
(Level 0 atomic components → Level 1 assemblies → Level 2 subsystems → Level 3 system,
built bottom-up, inside-out).

Where `todo_claude1.md` asked "do macro promises reach the code?", this file asks the
bottom-up question: **are the atomic components actually atomic, pure, and
independently testable — and do the assemblies compose them correctly?** The verdict:
the Level 0 catalogue exists and is mostly well-shaped (pure functions on arrays,
explicit scalars, no upward imports), but the *composition seams* between levels are
where the design breaks. Contracts between Level 0 and Level 1 are unwritten and
inconsistent, several components were built and never composed into anything, and the
determinism plumbing that Level 0 provides is dropped on the floor at Level 1.

---

## 1. Composition contracts are unwritten — and the levels disagree

The doc's core claim is "a component at Level N only depends on Level N−1" and that
assemblies just wire components together. But wiring only works if the parts agree on
conventions, and they currently don't:

- [ ] **The two spatial indexes are not interchangeable, despite §0.4 calling them
  "two interchangeable components".** `SpatialHashGrid.query_knn` returns **global**
  bird indices; `KDTreeIndex.rebuild` builds from `positions[active]`, so its
  `query_knn` returns **compacted active-subset** indices
  ([flock.py:225-236](pymurmur/physics/flock.py#L225-L236)). Any composition that
  works with one silently mis-addresses birds with the other once `active` has holes.
  Write the index-space contract down (recommend: global indices always), fix
  `KDTreeIndex` to map back through `np.where(active)[0]`, and add an
  interchangeability test that runs both implementations on the same holey-mask flock
  and asserts identical neighbor sets.
- [ ] **Level 0 primitives and Level 1 assemblies disagree on array shape and row
  space.** `separation_force`/`alignment_force`/`cohesion_force`
  ([_base.py](pymurmur/physics/forces/_base.py)) return `(N_capacity, 3)` arrays and
  iterate global indices, but [spatial.py:34-50](pymurmur/physics/forces/spatial.py#L34-L50)
  builds `neighbor_idx` with **compacted** rows `(n_active, k)` and then does
  `flock.accelerations[active] += sep * w` — adding an `(N_capacity, 3)` array to an
  `(n_active, 3)` slice. This only runs because `n_active == N_capacity` at startup;
  **press `-` (remove birds) in spatial mode and the composition crashes** (shape
  mismatch), and `neighbor_idx[i]` indexed by global `i` reads the wrong row before
  that. `noise_force(n_active, ...)` mixes a third shape into the same sum. Fix:
  declare one convention (recommend: everything in global/capacity space, rows for
  inactive birds unused) in `core/types.py`, and make every primitive and assembly
  state its index space in its docstring.
- [ ] **Add composition tests with a holey `active` mask.** Every bug above is
  invisible in the all-active happy path the current subsystem tests exercise. A
  single fixture — 100 birds, 20 deactivated mid-array — run through all five force
  modes and both indexes would catch this entire class permanently.
- [ ] **`projection._topological_neighbors` does `isinstance` checks against both
  index classes and then calls the same method on each branch**
  ([projection.py:94-99](pymurmur/physics/forces/projection.py#L94-L99)) — evidence
  that no `SpatialIndex` protocol exists. Define the Protocol in `core/types.py`
  (`rebuild`, `query_knn`, `ready`) so composition is against an interface, not
  concrete classes.

## 2. Level 0 purity and testability promises not met

§0 promises components that are "pure functions on arrays", vectorised, and
unit-testable in isolation.

- [ ] **The force primitives are per-bird Python loops** — each iterates
  `for i in np.where(active)[0]` ([_base.py:26](pymurmur/physics/forces/_base.py#L26)
  et al.), directly contradicting §0.3/"zero per-bird Python loops" and making the
  §4.3 performance table (300K @ 17 ms) unreachable. `neighbor_idx` is a rectangular
  `(N, k)` array — all three primitives vectorise with one gather
  (`positions[neighbor_idx]`, shape `(N, k, 3)`) and axis-1 reductions. Do that, or
  implement the numba `ForceKernel` path the doc designs (`use_numba` is currently a
  dead config flag; no numba code exists).
- [ ] **`cohesion_force` violates its own stated contract.** Docstring says
  `F_coh = limit_length(avg(p_j) - p_i, 1)`, but the code returns
  `to_center / min(length, 1.0)` ([_base.py:93](pymurmur/physics/forces/_base.py#L93))
  — **unbounded** for far centres (length ≥ 1 divides by 1.0). Level 0 components are
  supposed to be the trustworthy foundation; add unit tests that assert each
  primitive's documented formula and bound, then fix the formula.
- [ ] **The occlusion SoA adapter allocates Python objects in the hot path.**
  `spherical_cap_occlusion_soa` wraps each of σ neighbours in a `SimpleNamespace`,
  per bird, per frame — and `projection_forces` drives it from a per-bird Python loop
  ([projection.py:38](pymurmur/physics/forces/projection.py#L38)). The doc presents
  the adapter as a stopgap ("temporary objects for σ neighbors only"); at 300K birds
  that is 300K × σ allocations per frame. Promote occlusion to a true array kernel
  (inputs `(σ, 3)` arrays, no object wrapping) — the doc's own bottom-up method says
  rebuild the atom, and every level above inherits the win.
- [ ] **Move the lazy `from ..steric import steric_force` out of the per-bird loop**
  ([projection.py:82](pymurmur/physics/forces/projection.py#L82)) — module import
  machinery invoked N times per frame for no layering benefit (steric is Level 0 and
  freely importable at module top).

## 3. Determinism: the rng plumbing exists at Level 0 but is never composed upward

For a scientific tool, this is the most consequential micro-to-macro gap. The Level 0
helpers were built right — `noise_force`, `random_positions`, `random_unit_sphere`
all accept an `rng: np.random.Generator` parameter — and then **no assembly ever
passes one**:

- [ ] `spatial_forces` calls `noise_force(n, scale)` with no rng; `integrate`'s
  zero-speed re-seed calls `random_unit_sphere(nz)` unseeded
  ([boid.py:57](pymurmur/physics/boid.py#L57)).
- [ ] `vicsek_forces`, `influencer_forces`, and `Predator` bypass the plumbing
  entirely with module-level `np.random.*`
  ([vicsek.py:42](pymurmur/physics/forces/vicsek.py#L42),
  [influencer.py:34](pymurmur/physics/forces/influencer.py#L34),
  [predator.py:45](pymurmur/physics/extensions/predator.py#L45)).
- [ ] `add_boids` creates a fresh unseeded `default_rng()` each call
  ([flock.py:83](pymurmur/physics/flock.py#L83)).

Net effect: `config.seed` determines frame 0 and nothing after it — two runs with the
same seed diverge on the first noisy frame. Fix bottom-up, the way the doc prescribes:
`PhysicsFlock` owns a single `Generator` seeded from `config.seed`, threads it into
every stochastic primitive/extension (the parameters already exist!), and a test
asserts *same seed → bit-identical positions after 100 steps* for every mode. Note
`evoflock.py` already does this correctly (`self._rng` seeded from config) — copy
that pattern down into the physics layer.

## 4. Components built bottom-up and then never composed (dead atoms)

Bottom-up construction's failure mode is inventory that never gets assembled. The
codebase has accumulated exactly that:

- [ ] **`PRESETS` is composed into nothing.** §1.5 + §2.5 promise keyboard keys 1-9
  applying presets, and §4.1 declares `input_control → presets` as a dependency.
  [presets.py](pymurmur/analysis/presets.py) exists, matches the doc exactly — and no
  module imports it; [input_control.py](pymurmur/viz/input_control.py) has no preset
  bindings. Wire the 1-9 keys (apply the dict onto `SimConfig`), or delete the module.
- [ ] **`BoidView` is defined and never used**
  ([boid.py:146-182](pymurmur/physics/boid.py#L146-L182)) — the doc says occlusion
  and forces use it; they take raw arrays instead (which is fine — then delete it).
- [ ] **`FlockArrays` and `ForceKernel` in `core/types.py` are never instantiated /
  referenced** — the "fundamental data contract every component agrees on" is agreed
  on by zero components. Compose (`PhysicsFlock` holding a `FlockArrays`) or remove.
- [ ] **Renderer supports themes; the Visualizer never passes one.** `Renderer3D`
  takes `theme:` and the shaders have the uniforms, but
  [visualizer.py:39-44](pymurmur/viz/visualizer.py#L39-L44) doesn't forward
  `config.theme`, so the `theme` config field (and `ink|inverse|paper|graphite` docs)
  is dead at the composition seam — a one-line wire-up.
- [ ] **`trails` and `point_sprites` config fields have no implementation at all**;
  the doc's §2.4 `TrailRenderer` doesn't exist and
  [renderer.py:5](pymurmur/viz/renderer.py#L5) claims "velocity trail rendering" it
  doesn't contain. Implement or strip the fields + docstring.
- [ ] Adopt a working rule to prevent recurrence: **a Level 0/1 component merges only
  together with the assembly that composes it** (or with a test that is its only
  consumer, explicitly marked). Uncomposed inventory is drift waiting to happen.

## 5. The composition graph isn't the DAG the doc draws

- [ ] **`physics_flock` ↔ `forces` is a mutual dependency between two Level 1
  assemblies.** The doc's §4.1 table itself lists both `physics_flock → forces` and
  `forces → physics_flock`; the code papers over the cycle with a lazy import inside
  `PhysicsFlock.step()` ([flock.py:56](pymurmur/physics/flock.py#L56)). Bottom-up
  composition should be a strict DAG. Cleanest fix: `PhysicsFlock` stops calling
  `compute_all_forces`; the Level 2 `SimulationEngine` sequences
  `rebuild_index → forces → integrate` (it already owns the step), leaving flock as
  pure state+index and forces depending on flock one-way.
- [ ] **Four new assemblies exist above the doc's layer diagram**: `analysis/perf.py`,
  `phase_diagram.py`, `density_scaling.py`, `evoflock.py` all compose
  `SimulationEngine` — they are Level 3 assemblies (experiment drivers) that the
  synthesis doc doesn't know about. Add them to §3.3 and the layer stack (they slot in
  beside `pymurmur.py`, not inside Level 1 metrics), and give them the same dependency
  rules (may import Level ≤2, never viz).
- [ ] **Mode assemblies consume god objects.** Every force mode takes
  `(flock: PhysicsFlock, config: SimConfig)` — the entire world — while the Level 0
  primitives underneath take exactly the arrays they need. The doc's inside-out
  discipline argues for the modes taking narrow inputs too: `(arrays, index,
  mode_params, rng)`. This would also make each mode testable without constructing a
  `PhysicsFlock` (build arrays by hand), which is the doc's stated reason for Level 0
  purity.
- [ ] **`Recorder` re-implements instead of composing.** §2.6 shows Recorder
  assembling a headless `Visualizer`; the code hand-builds its own `Renderer3D` +
  `OrbitCamera` inline ([recorder.py:55-75](pymurmur/capture/recorder.py#L55-L75)).
  Requires first making `Visualizer.headless_frame()` render-only — today it *steps
  the simulation* as a side effect ([visualizer.py:49-56](pymurmur/viz/visualizer.py#L49-L56)),
  which is why Recorder couldn't reuse it. Fix the atom, then restore the composition.

## 6. Composition-seam defects a bottom-up test suite would have caught

- [ ] **Renderer buffer growth orphans the VAO.** `update_instances` replaces
  `self._instance_vbo` when the flock outgrows the chunk
  ([renderer.py:104-109](pymurmur/viz/renderer.py#L104-L109)), but `self._vao` was
  built once in `__init__` against the *old* buffer and is never rebuilt — after the
  first growth, draws read a stale (deallocated) buffer. Rebuild the VAO after
  reallocation, and add a headless test that adds >50K birds and renders.
- [ ] **`spatial_forces` clamps accelerations of inactive birds too** — the
  max-force clamp at [spatial.py:53-59](pymurmur/physics/forces/spatial.py#L53-L59)
  operates on the full array; harmless today, wrong once inactive slots carry stale
  data. Mask it. (Falls out of the item-1 shape-contract work.)
- [ ] **Where the code is better than the doc, update the doc** — bottom-up design
  treats the components as the source of truth, and here they are: `integrate()` takes
  explicit scalars instead of the doc's `config` object (keeping Level 0 free of the
  config dependency §4.1 claims it has), and `InputControl.handle_events()` drops the
  doc's `viz` parameter (cleaner layering). Sync §0.5, §2.5, §3.3 (flat module map →
  real nested packages), and fix the §0.8 signatures (`random_positions(n, w, h, d,
  rng)`, not `(N, config)`).
- [ ] **Grow the test pyramid from the bottom, as the doc's build order promises**
  ("build Level 0 → test → assemble Level 1 → test → …"). Concretely: property tests
  for each Level 0 primitive (documented formula, output shape, bound, rng
  determinism); contract tests both index implementations must pass; holey-mask
  composition tests per force mode (item 1); a same-seed reproducibility test per mode
  (item 3). The existing `test/test_subsystem_*.py` files test top-down slices — the
  bottom-up complement is what's missing, and it's the layer that would have caught
  nearly every item in this file.
