# Test Suite — pymurmur 3D Murmuration Simulation

> **STATUS: ACTIVE.** This document describes the test suite's organization and
> conventions.
> **~2,260 tests collected; ~2,220 run in the fast suite.**
> **Organization: Top-Down / Macro-to-Micro.** The `test/` tree is layered by
> altitude, mirroring `arch.md`'s functional decomposition: Level 0 (system
> goal) down to Level 4 (implementation details). Directory names carry the
> level prefix (`l0_` … `l4_`) so the tree lists in macro-to-micro order.
> **Design docs:** `arch.md` — single architecture reference.
> **Framework:** `pytest` with `numpy` test helpers. GPU tests use `@pytest.mark.gpu`.

## Test Cheatsheet

### Run Commands

```bash
# All fast tests (marker-based: auto-skips GPU where unavailable)
python3 -m pytest test/ -q -m "not slow"

# CI: fast suite — what runs on every commit
pytest test/ -v -m "not slow and not gl and not gpu" --timeout=60 --tb=short

# Guard rails (architecture, docs, golden, collection count, config drift…)
pytest -m guard -v

# Golden trajectory tests only
pytest -m golden -v

# One altitude level (macro → micro)
pytest test/l0_system/ -v          # system goal
pytest test/l1_subsystems/ -v      # A–F isolation
pytest test/l2_integration/ -v     # subsystem wiring
pytest test/l3_modules/ -v         # module interfaces
pytest test/l4_crosscutting/ -v    # guards + perf

# One module's tests
pytest test/l3_modules/physics/forces/test_field.py -v

# Match by test name pattern
pytest test/ -k "test_vicsek" -v

# Performance & scaling (slow, nightly/PR-merge only)
pytest -m slow test/l4_crosscutting/perf/ -v

# Coverage (do NOT use the pytest-cov plugin — it reloads numpy and
# breaks ~440 tests; use coverage-run instead):
python3 -m coverage run --branch --include="pymurmur/*" -m pytest -q -m "not slow"
python3 -m coverage report --show-missing
```

### Pytest Markers

| Marker | Purpose |
|--------|---------|
| `slow` | Performance/benchmark tests, run on PR merge or nightly |
| `gpu` / `gl` | Tests requiring a ModernGL GPU context |
| `golden` | Golden trajectory regression tests |
| `guard` | Guard-rail tests (architecture, docs, golden, collection count, drift) |
| `numba` | Tests requiring numba JIT |
| `pygame` | Tests requiring a pygame event loop |
| `e2e` | End-to-end tests that run the full simulation |
| `integration` | Tests wiring multiple modules together |
| `acceptance` | Phase-level acceptance tests |
| `phase1`/`phase3`/`phase4` | Phase acceptance-criteria groupings |

### Shared Fixtures (`test/conftest.py`)

| Fixture | Returns | Use for |
|---------|---------|---------|
| `default_config` | `SimConfig()` with projection defaults | Any test needing a config |
| `spatial_config` | `SimConfig(mode="spatial", num_boids=200)` | Spatial mode tests |
| `small_flock` | `PhysicsFlock(config, N=50)` | Fast unit tests |
| `two_bird_flock` | `PhysicsFlock(config, N=2)` | Neighbor/pair tests |
| `known_positions` | `(4,3) float32` array | Deterministic position tests |
| `known_velocities` | `(4,3) float32` array | Deterministic velocity tests |
| `neighbor_idx` | `(4,3) int32` pre-computed neighbor indices | Force primitive tests |
| `gpu_available` | `bool` (session-scoped) | Conditional GPU test skipping |
| `numba_available` | `bool` (session-scoped) | Conditional numba test skipping |

Shared helpers (`_step_flock`, `_call_force`) live in `test/helpers.py` and are
imported as `from test.helpers import …`.

### Conventions & Gotchas

- Python is 3.9 — test files using `X | Y` annotations need
  `from __future__ import annotations`.
- `capture_prewarm` defaults to 60: Recorder frame-capture tests must set
  `cfg.capture_prewarm = 0` or the first 60 frames are silently skipped.
- Neighbor-index arrays use 0 as the padding value (`nbrs > 0` filters), so
  bird index 0 never appears as a neighbor slot value.
- `SimConfig.from_file(path)` is the loader (not `from_yaml`); validation
  requires `capture.frames >= 1`.
- Every new module import edge must be registered in `ALLOWED_EDGES` in
  `test/l4_crosscutting/guards/test_architecture.py`.

---

## Directory Organization — Macro to Micro

All input files (YAML configs) MUST be placed within `conf/`. All output files
MUST be placed within `output/`. All test files MUST be placed within `test/`.

The tree descends the same way `arch.md` does: start at the system goal,
decompose into subsystems, wire them, then test each module interface, and
finally pin the implementation details. Reading the tree top-to-bottom **is**
reading the architecture top-down.

```
test/
├── __init__.py
├── conftest.py                  # shared fixtures (see cheatsheet)
├── helpers.py                   # _step_flock, _call_force
├── regenerate_golden.py         # golden trajectory generator (not collected)
├── data/                        # golden .npz baselines (CI-validated paths)
│
├── l0_system/                   # ── Level 0: SYSTEM GOAL (macro) ──
│   │                            # "load config, run modes, viz/capture
│   │                            #  optionally, run headlessly"
│   ├── test_cli.py              # __main__ dispatch: parse_args, load_config
│   ├── test_cli_e2e.py          # CLI flags end-to-end (--set, --print-config…)
│   ├── test_probe.py            # --probe capability probing
│   ├── test_config_resolution.py# name → conf/*.yaml → path resolution
│   ├── test_config_files.py     # every shipped conf/*.yaml is valid
│   ├── test_facade.py           # pymurmur.Simulation public API
│   ├── test_e2e.py              # headless end-to-end scenarios
│   ├── test_mode_switch_no_crash.py  # cycle all modes live (guard)
│   └── acceptance/              # phase gates: whole-system acceptance criteria
│       └── test_phase1.py, test_phase3.py, test_phase4.py
│
├── l1_subsystems/               # ── Level 1: FUNCTIONAL DECOMPOSITION ──
│   │                            # each subsystem isolated, deps mocked
│   ├── test_subsystem_a.py      # A — Entry & Configuration
│   ├── test_subsystem_b.py      # B — Simulation Engine
│   ├── test_subsystem_c.py      # C — Visualization & Input
│   ├── test_subsystem_d.py      # D — Capture & Export
│   ├── test_subsystem_e.py      # E — Physics & Forces
│   └── test_subsystem_f.py      # F — Metrics & Analysis
│
├── l2_integration/              # ── Level 2: SUBSYSTEM WIRING (meso) ──
│   ├── test_engine_pipeline.py  # 6-stage engine order under live mutations
│   ├── test_render_contract.py  # frame()/headless_frame() never step the sim
│   ├── test_capture_pipeline.py # step → on_frame → serialize round-trip
│   ├── test_config_contract.py  # facade, field map, nested↔flat contract
│   └── test_cross_subsystem.py  # index swap, threat pipeline, instance schema
│
├── l3_modules/                  # ── Level 3: MODULE INTERFACES (micro) ──
│   │                            # one mirror directory per pymurmur package;
│   │                            # test for pymurmur/<pkg>/<mod>.py lives at
│   │                            # l3_modules/<pkg>/test_<mod>*.py
│   ├── core/                    # test_types, test_config, test_config_validation
│   ├── physics/                 # boid, flock, occlusion, steric, obstacles,
│   │   │                        # composition, spatial-index contract,
│   │   │                        # edge cases, field×extensions integration
│   │   ├── forces/              # one file per force mode + kernels, terms,
│   │   │                        # primitives, mode contract
│   │   └── extensions/          # manager, predator, threat, wander, ripple, ecology
│   ├── simulation/              # engine step order, fixed-timestep accumulator
│   ├── viz/                     # renderer, shaders, camera, trails (4 modes),
│   │                            # wings, colour, density, dual view, hud, input
│   ├── capture/                 # recorder, mpl_recorder, cinematic sweep
│   └── analysis/                # metrics (schema/motion/expensive), h2, presets,
│                                # perf+quality governor, phase diagram, density
│                                # scaling, rewards, evoflock, evolved-yaml guard,
│                                # marl (P12 placeholder)
│
└── l4_crosscutting/             # ── Level 4: IMPLEMENTATION DETAILS (nano) ──
    ├── guards/                  # repo guard-rails (CI: guard-rails.yml, -m guard)
    │   ├── test_architecture.py     # ALLOWED_EDGES import-DAG enforcement
    │   ├── test_imports.py          # no-upward-import rules
    │   ├── test_docs.py             # arch.md ↔ roadmap link validation
    │   ├── test_golden.py           # golden trajectory regression (test/data)
    │   ├── test_config_drift.py     # every SimConfig field used in source
    │   └── test_collection_count.py # per-level floors — suite never shrinks
    └── perf/                    # slow benchmarks (nightly / PR merge)
        ├── test_performance.py  # per-mode step-time budgets, memory
        └── test_scaling.py      # O(N) / O(N log N) scaling fits
```

**Placement rules (decide top-down):**

1. Does it verify a **system-level goal** — CLI, facade, an end-to-end run, a
   phase acceptance gate? → `l0_system/` (gates in `l0_system/acceptance/`).
2. Does it isolate **one subsystem A–F** with the rest mocked? → `l1_subsystems/`.
3. Does it wire **multiple subsystems** and assert their contract? → `l2_integration/`.
4. Does it exercise **one module's interface** (possibly with direct
   collaborators)? → `l3_modules/<pkg>/`.
5. Does it pin an **implementation invariant** — imports, goldens, drift,
   collection count, step-time? → `l4_crosscutting/` (`guards/` or `perf/`).

File names are module-first (`test_<module>[_<aspect>].py`), not phase-first;
phase provenance lives in docstrings (`P8.10: …`). The `guard` marker — not
the directory — is what CI's guard-rails workflow selects, so module-shaped
guards (e.g. `l3_modules/analysis/test_evolved_yaml.py`,
`l3_modules/physics/forces/test_vicsek_core.py`) stay with their modules.

---

## Level ↔ arch.md Mapping

| Test level | arch.md Level | What's tested |
|------------|---------------|---------------|
| `l0_system/` | Level 0 — System Goal | CLI dispatch, facade, config resolution, E2E, phase gates |
| `l1_subsystems/` | Level 1 — Functional Decomposition (A–F) | Each subsystem in isolation with mocked dependencies |
| `l2_integration/` | Level 2 — Subsystem Design (Meso) | Engine/capture pipelines, render + config contracts |
| `l3_modules/` | Level 3 — Module Interfaces (Micro) | Individual modules and their composition |
| `l4_crosscutting/` | Level 4 — Implementation Details (Nano) | Import DAG, goldens, drift, collection floors, perf/scaling |

Subsystem key: A = Entry & Configuration, B = Simulation Engine,
C = Visualization & Input, D = Capture & Export, E = Physics & Forces,
F = Metrics & Analysis.

---

## Guard Rails (`l4_crosscutting/guards/` + `.github/workflows/guard-rails.yml`)

Guards run on every commit via `pytest -m guard`. Each guard fails loudly and
explains how to update itself when a change is intentional:

| Guard | Protects against |
|-------|-----------------|
| `test_architecture.py` | New import edges not registered in `ALLOWED_EDGES`; layering violations |
| `test_imports.py` | Upward imports (physics→viz, engine→pygame, …) |
| `test_golden.py` | Behavioural drift in the 5 force modes (bit-exact vs `test/data/golden_*.npz`; regenerate with `test/regenerate_golden.py`) |
| `test_docs.py` | Broken intra-repo links in `arch.md` / `roadmap_deepseek.md` |
| `test_config_drift.py` | Orphan `SimConfig` fields no source file reads |
| `test_collection_count.py` | Silent test loss — floors per level, per module mirror, and in total |

System-level guards live at their altitude (`l0_system/test_mode_switch_no_crash.py`);
module-level guards live with their module. The marker selects them all.

---

## Index A — Implemented Ideas → Test Files

Where to look when you want the tests for a concept. Paths are relative to
`test/`; `l3m` abbreviates `l3_modules`.

| Idea / invariant | Test files |
|---|---|
| **Determinism** (same seed → bit-identical; numba↔numpy parity) | `l4_crosscutting/guards/test_determinism.py`, `l3m/physics/test_flock.py` |
| **Golden trajectory regression** (5 modes × 2 boundaries, bit-exact) | `l4_crosscutting/guards/test_golden.py` (+ generator `regenerate_golden.py`) |
| **Import-DAG / layering enforcement** (`ALLOWED_EDGES`) | `l4_crosscutting/guards/test_architecture.py`, `test_imports.py`; waiver-removal in `l2_integration/test_render_contract.py` |
| **Suite never silently shrinks** (collection floors) | `l4_crosscutting/guards/test_collection_count.py` |
| **Docs stay linked & in sync** (arch ↔ roadmap) | `l4_crosscutting/guards/test_docs.py` |
| **No orphan config fields** | `l4_crosscutting/guards/test_config_drift.py` |
| **Config system** (nested dataclasses, YAML I/O, validation, flat↔nested map) | `l3m/core/test_config.py`, `test_config_validation.py`, `l2_integration/test_config_contract.py` |
| **Config loading & resolution** (search path, shipped files) | `l0_system/test_config_resolution.py`, `test_config_files.py` |
| **CLI dispatch & flags** (`--set`, `--probe`, `--list-configs`…) | `l0_system/test_cli.py`, `test_cli_e2e.py`, `test_probe.py` |
| **Public facade** (`pymurmur.Simulation`) | `l0_system/test_facade.py`, `l2_integration/test_config_contract.py` |
| **Headless end-to-end runs** | `l0_system/test_e2e.py`, `test_mode_switch_no_crash.py` |
| **Phase acceptance gates** | `l0_system/acceptance/test_phase{1,3,4}.py` |
| **Subsystem isolation (A–F)** | `l1_subsystems/test_subsystem_{a..f}.py` |
| **Engine step order & command queue** | `l3m/simulation/test_engine.py`, `l2_integration/test_engine_pipeline.py` |
| **Fixed-timestep accumulator + render lerp** | `l3m/simulation/test_accumulator.py` |
| **Force-mode contract** (registry, needs_index, active mask) | `l3m/physics/forces/test_mode_contract.py` |
| **Force primitives** (sep/align/cohere/noise; property-based) | `l3m/physics/forces/test_base.py`, `test_force_primitives_properties.py`, `test_force_terms.py` |
| **Projection mode** (Pearce occlusion-driven) | `l3m/physics/forces/test_projection.py`, `l3m/physics/test_occlusion.py` |
| **Spatial mode** (+ hybrid filter, variants, numba kernels) | `l3m/physics/forces/test_spatial_variants.py`, `test_forces_hybrid.py`, `test_kernels.py` |
| **Field/blob mode** (anchors, targets, cavity, buoyancy) | `l3m/physics/forces/test_field.py`, `test_field_units.py`, `l3m/physics/test_field_integration.py` |
| **Vicsek mode** (core, memory term, species predator-prey) | `l3m/physics/forces/test_vicsek.py`, `test_vicsek_core.py`, `test_vicsek_species.py` |
| **Influencer mode** | `l3m/physics/forces/test_influencer.py` |
| **Angle mode** | `l3m/physics/forces/test_angle.py` |
| **Force dispatch** | `l3m/physics/forces/test_forces.py` |
| **Flock state & spatial indices** (SoA, RNG, hash grid, KD-tree) | `l3m/physics/test_flock.py`, `test_spatial_index_contract.py` |
| **Integration kernel & boundaries** | `l3m/physics/test_boid.py` |
| **Steric repulsion** | `l3m/physics/test_steric.py` |
| **Obstacles / SDF primitives** | `l3m/physics/test_obstacles.py` |
| **Active-mask (holey array) survival** | `l3m/physics/test_holey_mask_composition.py`, `test_composition.py` |
| **Extensions** (manager, predator FSM, threat, wander, ripple, ecology) | `l3m/physics/extensions/*` |
| **Physics edge cases / branch gaps** | `l3m/physics/test_edge_cases.py` |
| **Metrics** (order params, gating, schema, motion, expensive, H₂) | `l3m/analysis/test_metrics*.py`, `test_h2.py`, `test_cross_element.py` |
| **Presets** | `l3m/analysis/test_presets.py` |
| **Perf diagnostics + adaptive quality governor** | `l3m/analysis/test_perf.py`, `test_quality.py`; ladder actions in `l3m/viz/test_visualizer_quality.py` |
| **Phase diagram / density scaling experiments** | `l3m/analysis/test_phase_diagram.py`, `test_density_scaling.py` |
| **Rewards (shared MARL/Evo scalarization)** | `l3m/analysis/test_rewards.py` |
| **EvoFlock SSGA evolution** (+ evolved artifact guard) | `l3m/analysis/test_evoflock.py`, `test_evolved_yaml.py` |
| **MARL bridge (P12 — pending)** | `l3m/analysis/test_marl.py` (placeholder stub) |
| **Renderer** (impostors, depth cues, buffers, HUD GL) | `l3m/viz/test_renderer.py` |
| **Shaders & meshes** (GLSL, tetra/wings, sky) | `l3m/viz/test_shaders.py`, `test_wings.py` |
| **Camera** (orbit, cinematic sweep) | `l3m/viz/test_camera.py`, `l3m/capture/test_cinematic.py` |
| **Trails (4 modes + growth/degenerate/FBO lifecycle)** | `l3m/viz/test_trails.py` |
| **Colour channels & themes** | `l3m/viz/test_colour.py` |
| **Density (alpha-accumulation) mode** | `l3m/viz/test_density.py` |
| **Dual view** | `l3m/viz/test_dual_view.py`, `l3m/viz/test_cross_element.py` |
| **HUD sliders (logic)** | `l3m/viz/test_hud.py` |
| **Input → config bridge** | `l3m/viz/test_input.py` |
| **Render purity contract** (render never steps sim) | `l2_integration/test_render_contract.py` |
| **Capture** (recorder, GIF/CSV/JSON, prewarm, mpl fallback) | `l3m/capture/test_recorder.py`, `test_mpl_recorder.py`, `l2_integration/test_capture_pipeline.py` |
| **Step-time / memory budgets; complexity claims** | `l4_crosscutting/perf/test_performance.py`, `test_scaling.py` |

## Index B — Test Files → Implemented Ideas

One line per file: what idea(s) it pins down. Grouped macro → micro.

### l0_system/ — system goal
| File | Idea |
|---|---|
| `test_cli.py` | `__main__` arg parsing, config loading, `--list-configs`, main() dispatch |
| `test_cli_e2e.py` | CLI flags end-to-end: `--set`, `--print-config`, `--fullscreen` (guard) |
| `test_probe.py` | `--probe` capability probing (GPU/numba/pygame detection) |
| `test_config_resolution.py` | `load_config()` search-path order: name → `conf/` → path |
| `test_config_files.py` | Every shipped `conf/*.yaml` parses and has required sections |
| `test_facade.py` | `pymurmur.Simulation` public API: construct, run, metrics, benchmark |
| `test_e2e.py` | Full headless simulation scenarios across modes |
| `test_mode_switch_no_crash.py` | Live-cycling all force modes never crashes (guard) |
| `acceptance/test_phase1.py` | P1 gates: occlusion, steric, boundary δ |
| `acceptance/test_phase3.py` | P3 gates: field mode, wander, ripple, predator, presets |
| `acceptance/test_phase4.py` | P4 gates: spatial golden, fuzz, presets, predator |

### l1_subsystems/ — A–F isolation
| File | Idea |
|---|---|
| `test_subsystem_a.py` | Entry & Configuration isolated: defaults, YAML flattening, roundtrip |
| `test_subsystem_b.py` | Engine isolated: step order, headless run, reset, live mutation |
| `test_subsystem_c.py` | Viz & Input isolated: no simulation imports, wiring |
| `test_subsystem_d.py` | Capture isolated: metrics-only mode, GIF/CSV/JSON validity |
| `test_subsystem_e.py` | Physics isolated: two-pass architecture, all modes' force validity |
| `test_subsystem_f.py` | Metrics isolated: O(N) fast metrics, gating levels, intervals |

### l2_integration/ — subsystem wiring
| File | Idea |
|---|---|
| `test_engine_pipeline.py` | 6-stage engine order holds across steps + mutations; only engine imports flock+forces |
| `test_render_contract.py` | `frame()`/`headless_frame()` never step the sim; flock↔forces cycle break |
| `test_capture_pipeline.py` | step → on_frame → serialize round-trip; buffer growth under mutations |
| `test_config_contract.py` | Facade re-exports, `_FIELD_MAP` completeness, nested↔flat integrity, no GL imports in config classes |
| `test_cross_subsystem.py` | Index swap mid-run, threat/evasion pipeline, InstanceSchema packing |

### l3_modules/core/
| File | Idea |
|---|---|
| `test_types.py` | Math helpers (Rodrigues, min_image, smoothstep…), FlockArrays, protocols |
| `test_config.py` | SimConfig nested dataclasses, YAML I/O, unknown-key warning |
| `test_config_validation.py` | `validate()` cross-field rules and range clamps |

### l3_modules/physics/ (+ forces/, extensions/)
| File | Idea |
|---|---|
| `test_boid.py` | `integrate()`: speed clamps, boundary modes, dt scaling; array init helpers |
| `test_flock.py` | PhysicsFlock lifecycle, single RNG, indices, add/remove, determinism |
| `test_occlusion.py` | Spherical-cap occlusion: δ̂/Θ ranges, blind angle, anisotropy, SoA parity |
| `test_steric.py` | 1/d² steric repulsion behaviour |
| `test_obstacles.py` | SDF primitives (5), collision helpers, ObstacleScene |
| `test_composition.py` | Holey-mask contract for composed force pipelines |
| `test_holey_mask_composition.py` | All modes survive inactive slots mid-array |
| `test_spatial_index_contract.py` | KDTreeIndex global indices, boxsize toroidal queries |
| `test_edge_cases.py` | Branch gaps: predator anti-parallel, wander fallbacks, field cavity/clamp |
| `test_field_integration.py` | Field × ripple/wander coupling, egress arc, cycle periodicity |
| `forces/test_base.py` | Force primitive kernels (sep 1/d², bounded cohesion, noise scale) |
| `forces/test_force_primitives_properties.py` | Property-based invariants per primitive |
| `forces/test_force_terms.py` | ForceTerm compose/enable/gain reducer |
| `forces/test_forces.py` | MODE_REGISTRY dispatch, invalid mode, mid-run switch |
| `forces/test_forces_hybrid.py` | Hybrid filter, predator escape, jitter, coherence gate, batch query |
| `forces/test_kernels.py` | Numba kernel ↔ numpy parity (numba-marked) |
| `forces/test_mode_contract.py` | ForceMode ABC: instantiable, active mask, no import cycle |
| `forces/test_projection.py` | Projection mode: δ computation, θ cache, blind/anisotropy effects |
| `forces/test_spatial_variants.py` | Spatial-mode weight isolation (sep/align/cohere/noise only) |
| `forces/test_field.py` / `test_field_units.py` | Field mode L1 behaviour / L0 functions (anchors, targets, phases) |
| `forces/test_vicsek.py` / `test_vicsek_core.py` / `test_vicsek_species.py` | Order transition / memory + tangent noise (guard) / predator-prey (guard) |
| `forces/test_influencer.py` | Influencer parity + Lissajous target tracking |
| `forces/test_angle.py` | Angle-mode steering |
| `extensions/test_extensions.py` | ExtensionManager + all extensions no-op/enable matrix |
| `extensions/test_predator.py` / `test_threat.py` | Threat FSM, panic/blackening / FSM baselines |
| `extensions/test_wander.py` / `test_ripple.py` / `test_ecology.py` | Bounded wander path / ripple envelopes / day-length & roosting |

### l3_modules/simulation/
| File | Idea |
|---|---|
| `test_engine.py` | SimulationEngine: init, step, frame count, reset, no-viz imports |
| `test_accumulator.py` | Fixed-dt accumulator, spike clamp, lerped render_positions |

### l3_modules/viz/
| File | Idea |
|---|---|
| `test_renderer.py` | Renderer3D: headless FBO, instances, impostors, depth cues, HUD GL |
| `test_shaders.py` | GLSL sources compile-ready; tetrahedron mesh geometry |
| `test_camera.py` | OrbitCamera math: rotate/zoom clamps, auto-rotate, matrices |
| `test_trails.py` | 4 trail modes, buffer growth, degenerate inputs, FBO lifecycle, renderer wiring |
| `test_wings.py` | Winged flapping mesh, gradient sky quad |
| `test_colour.py` | Per-bird hue packing, theme material tables |
| `test_density.py` | Alpha-accumulation density rendering |
| `test_dual_view.py` | Dual-view + orthographic presets |
| `test_hud.py` | SliderHUD value mapping, hit-test, config writes (CPU) |
| `test_input.py` | Keyboard/mouse → SimConfig bridge, full key map |
| `test_visualizer_quality.py` | P8.6 degradation ladder + recovery actions (CPU, mocked GL) |
| `test_cross_element.py` | Phase 8 render/capture cross-element wiring |

### l3_modules/capture/
| File | Idea |
|---|---|
| `test_recorder.py` | Recorder: capture_every, prewarm, GIF/CSV/JSON, no-PIL fallback |
| `test_mpl_recorder.py` | Matplotlib fallback recorder |
| `test_cinematic.py` | Cinematic sweep math, prewarm skip, env overrides |

### l3_modules/analysis/
| File | Idea |
|---|---|
| `test_metrics.py` | FlockMetrics + collector: order params, gating, snapshot |
| `test_metrics_schema.py` | `to_dict()` JSON round-trip schema |
| `test_metrics_motion.py` | Silhouette, η(m), robust gyration, motion metrics |
| `test_metrics_expensive.py` | Shape PCA, gyration, MSD, θ′, τ_ρ |
| `test_h2.py` | H₂ k-NN Laplacian robustness, cost-optimal m* |
| `test_cross_element.py` | P9 metric chains working together |
| `test_presets.py` | PRESETS validity + non-mutation |
| `test_perf.py` | EMA timing, bottleneck classification, QualityGovernor internals |
| `test_quality.py` | Governor hysteresis, ladder state machine (unit) |
| `test_phase_diagram.py` | Vicsek η×D sweep |
| `test_density_scaling.py` | N-sweep power-law fits (sweeps are `@slow`), save/load round-trip |
| `test_rewards.py` | Weighted composite reward, linearity |
| `test_evoflock.py` | SSGA: worst-of-4, crossover, objectives, SDF collisions |
| `test_evolved_yaml.py` | Evolved-config artifact validity (guard) |
| `test_marl.py` | **P12 placeholder** — module-skipped until MARL bridge lands |

### l4_crosscutting/
| File | Idea |
|---|---|
| `guards/test_architecture.py` | Import-DAG matrix (`ALLOWED_EDGES`) enforcement |
| `guards/test_imports.py` | AST-level upward-import bans |
| `guards/test_golden.py` | Bit-exact trajectories vs `test/data/golden_*.npz` |
| `guards/test_determinism.py` | Registry-wide same-seed identity, seed divergence, numba↔numpy parity |
| `guards/test_docs.py` | arch/roadmap intra-repo links resolve; retired schemes absent |
| `guards/test_config_drift.py` | Every SimConfig field referenced in source |
| `guards/test_collection_count.py` | Per-level + per-module collection floors |
| `perf/test_performance.py` | FPS/memory budgets per mode & scale (`@slow`) |
| `perf/test_scaling.py` | O(N), O(N log N), O(1) complexity claims (`@slow`) |

---

## Suite Audit — 2026-07-19

Implemented in this audit: registry-parametrized **determinism guard** (CI
already had the slot), **visualizer degradation-ladder tests** (the P8.6
ladder actions were entirely untested), **trail buffer-growth / degenerate /
FBO-lifecycle tests** and **Renderer3D trail-wiring tests** (incl. the
accumulation restore-blit path), and **HUD GL helper tests**.

Known remaining gaps (acceptable, tracked):

- `l3m/analysis/test_marl.py` is a stub until Phase 12 lands.
- `pymurmur/viz/visualizer.py`'s pygame `run()` loop interaction branches
  (HUD toggle, cursor spawns) are only partially covered — they need a real
  event loop; the extracted logic (`_apply_quality_actions`) is now fully
  tested.
- `_kernels.py` shows low line coverage in fast runs — a tracing artifact:
  numba-jitted bodies bypass the coverage tracer; parity is asserted by
  `forces/test_kernels.py` and `guards/test_determinism.py`.
- `analysis/density_scaling.py` sweep body is `@slow`-only by design.

---

## Execution Strategy

Execution is bottom-up (micro first — a broken module fails everything above
it), even though the organization is top-down:

```
Every commit   →  pytest test/ -m "not slow and not gl and not gpu"   (~55 s)
Every commit   →  pytest -m guard                                     (guard-rails.yml)
PR merge       →  + GPU suites (l3_modules/viz, gpu-marked)           (needs display/Xvfb)
Nightly        →  + pytest -m slow test/l4_crosscutting/perf/         (benchmarks, scaling)
```

Bottom-up debugging order when a wide breakage appears:
`l3_modules/core → l3_modules/physics → l3_modules/simulation →
l3_modules/{analysis,capture,viz} → l2_integration → l1_subsystems →
l0_system → l4_crosscutting`.

---

## History

- **2026-07-19 (3)** — suite audit: added determinism guard, visualizer
  quality-ladder tests, trail growth/degenerate/FBO tests, renderer trail
  wiring + HUD GL tests; added Index A/B (ideas ↔ files) to this document.
- **2026-07-19 (2)** — reorganized Macro-to-Micro: five altitude levels
  `l0_system` → `l4_crosscutting`; module mirrors nested under `l3_modules/`;
  phase acceptance gates under `l0_system/acceptance/`.
- **2026-07-19 (1)** — root-level files foldered; phase-named files renamed
  module-first; dead stub `test_composers.py` deleted; collection-count guard
  implemented.
- Earlier planning history (per-test tables for P0–P10) is preserved in git
  history of this file and in `roadmap_deepseek.md` Part II.

*Derived from `arch.md` (§2). July 2026.*
