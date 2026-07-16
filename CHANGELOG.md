# Design Roadmap v2 — Final Changelog

**Date:** July 14, 2026
**Tests:** 862 passing, 0 failures, 13 xfail, 8/8 architecture
**Scope:** +3,817 / −7,822 lines across 87 files, 7 iterations + 3 bonus items

---

## Iteration 1 — Foundations & Determinism

| File | Δ | Description |
|---|---|---|
| `pymurmur/physics/forces/_base.py` | +135/−42 | Vectorised gather+reduce force primitives (dual-path: dense + ragged fallback). Fixed separation docstring from 1/d² to 1/d. |
| `pymurmur/physics/forces/spatial.py` | +73/−28 | `_query_neighbors` uses shared flock index (KDTreeIndex) with fallback for SpatialHashGrid |
| `pymurmur/physics/forces/vicsek.py` | +142/−46 | Batched `query_ball_tree` + sparse matvec replaces per-bird `query_ball_point` |
| `pymurmur/physics/forces/projection.py` | +112/−70 | Batched occlusion via `spherical_cap_occlusion_batched`; steering/alignment in batch; steric per-bird |
| `pymurmur/physics/occlusion.py` | +345/−65 | I1.3 array kernel: pre-allocated numpy arrays, batched `spherical_cap_occlusion_batched`, vectorised effective radii |
| `pymurmur/physics/forces/field.py` | +23/−15 | Uniform array-based signature |
| `pymurmur/physics/forces/influencer.py` | +26/−3 | Uniform array-based signature |
| `pymurmur/physics/forces/__init__.py` | +28/−3 | `compute_all_forces` unpacker, `mode_needs_index` helper |
| `pymurmur/physics/steric.py` | +9/−4 | Steric import at module top (I1.4) |
| `test/physics/test_force_primitives_properties.py` | +450 (new) | I1.7: 38 property tests across 4 primitives (separation, alignment, cohesion, noise) |
| `test/physics/test_occlusion.py` | +242/−6 | I1.3: 10 batched occlusion unit tests |
| `test/physics/test_flock.py` | +132/−24 | I1.6: per-mode determinism (5 modes) + parametric sweep |

---

## Iteration 2 — Contract Enforcement & Dead Inventory

| File | Δ | Description |
|---|---|---|
| `pymurmur/core/config.py` | +403/−56 | Deleted `use_numba`, `trails`, `point_sprites`; wired predator/ecology params; `config.theme` → Visualizer |
| `pymurmur/viz/visualizer.py` | +25/−18 | Reads `config.theme`; accepts `width`/`height` overrides (I6.2) |
| `pymurmur/viz/input_control.py` | +25 (new) | Keys 1–7 apply PRESETS |
| `pymurmur/capture/recorder.py` | +29/−31 | Uses `capture_width`/`capture_height`; composes Visualizer (I6.1) |
| `pymurmur/core/types.py` | +45/−8 | Added `SpatialIndex` Protocol |
| `conf/*.yaml` | −12 | Removed dead fields (`use_numba`, `trails`, `point_sprites`) |
| `test/test_config_drift.py` | +128/−3 | AST-based config-usage drift detector |

---

## Iteration 3 — Spatial Index & Shape Contracts

| File | Δ | Description |
|---|---|---|
| `pymurmur/physics/flock.py` | +90/−31 | Shared index consumed by all modes; global indices in `KDTreeIndex.query_knn`; N≥5000 auto-switch; `_reevaluate_index` on add/remove |
| `pymurmur/core/types.py` | (in I2) | `SpatialIndex` Protocol with `ready`, `rebuild`, `query_knn`, `tree` |
| `test/physics/test_spatial_index_contract.py` | +111/−2 | Both index impls return identical global indices on holey masks |
| `test/physics/test_holey_mask_composition.py` | +95 (new) | All 7 force modes survive interspersed inactive birds |

---

## Iteration 4 — Simulation Purity & Control Surface

| File | Δ | Description |
|---|---|---|
| `pymurmur/simulation/engine.py` | +84/−5 | I4.2: `engine.step()` orchestrates rebuild → compute → integrate (breaks flock↔forces cycle). I4.3: `CommandQueue` with `enqueue_add/remove/reset` + `drain_commands()` |
| `pymurmur/viz/visualizer.py` | (in I2) | I4.1: `frame()`/`headless_frame()` are pure render — no `step()` side effect |
| `pymurmur/capture/recorder.py` | (in I2) | I6.1: composes `Visualizer(sim, headless=True)` |
| `test/test_architecture.py` | −3 | Removed `KNOWN_VIOLATIONS` waiver for `flock→forces` |
| `test/viz/test_renderer.py` | +38/−21 | I6.4: VAO rebuilt on instance buffer reallocation |

---

## Iteration 5 — Extension Protocol & Live Mutability

| File | Δ | Description |
|---|---|---|
| `pymurmur/physics/extensions/_base.py` | +27/−3 | I5.1: `StepContext` dataclass (frame, dt, rng, center, config, threat_prox). I5.2: `Extension.apply(flock, ctx)` |
| `pymurmur/physics/extensions/__init__.py` | +60/−24 | I5.3: `pre_step` checks `config.*_enabled` each frame; lazy-create/drop |
| `pymurmur/physics/extensions/predator.py` | +22/−8 | Uses `ctx.dt`, `ctx.rng`, `ctx.config`; sets `ctx.threat_prox`; removed dead `self._config` |
| `pymurmur/physics/extensions/ecology.py` | +17/−2 | I5.4: `eco.predator_active` public; renamed `self._dt` → `self._day_dt` |
| `pymurmur/physics/extensions/wander.py` | +3/−0 | Accepts `ctx` parameter |
| `pymurmur/physics/extensions/ripple.py` | +3/−0 | Accepts `ctx` parameter |
| `pymurmur/simulation/engine.py` | (in I4) | Builds `StepContext` and passes to `extensions.pre_step(flock, ctx)` |
| `test/physics/extensions/test_extensions.py` | +78/−56 | Updated for `apply(flock, ctx)` signature |

---

## Iteration 6 — Seams: Capture, Viz & Metrics

| File | Δ | Description |
|---|---|---|
| `pymurmur/capture/recorder.py` | (in I2) | I6.1: composes Visualizer. I6.3: targeted `ImportError`/`RuntimeError` instead of bare `except Exception: pass` |
| `pymurmur/viz/renderer.py` | +32/−6 | I6.4: VAO rebuilt on buffer reallocation; stored mesh VBO/IBO |
| `pymurmur/analysis/metrics.py` | +36/−4 | I6.5: `FlockMetrics.to_dict()` — ndarray→list, numpy NaN→null, scalar→Python scalar |
| `test/analysis/test_metrics_schema.py` | +80 (new) | I6.6: 7 JSON round-trip schema tests |

---

## Iteration 7 — Architecture Alignment

| File | Δ | Description |
|---|---|---|
| `pymurmur/core/config.py` | (in I2) | I7.1: `SimConfig` split into 17 composed sub-dataclasses (`DomainConfig`, `FlockConfig`, `BoundaryConfig`, etc.) with `__getattr__`/`__setattr__` delegation + `__copy__` |
| `pymurmur/__init__.py` | +7 (new) | I7.2: public facade exports `SimConfig`, `SimulationEngine`, `Recorder` |
| `test/test_config_drift.py` | (in I2) | Uses `_ALL_FIELD_NAMES` for sub-config field tracking |
| `test/analysis/test_presets.py` | +5/−3 | Uses `_ALL_FIELD_NAMES` instead of `__dataclass_fields__` |
| `test/test_subsystem_f.py` | +2/−2 | Sub-config field reference updates |

---

## Bonus Items

| Item | Files | Description |
|---|---|---|
| **I1.3 occlusion array kernel** | `pymurmur/physics/occlusion.py` (+345), `pymurmur/physics/forces/projection.py` (+112), `test/physics/test_occlusion.py` (+242) | Batched spherical-cap occlusion; zero Python allocations in hot path |
| **I1.7 property tests** | `test/physics/forces/test_force_primitives_properties.py` (+450 new) | 38 fuzzy/property tests across all 4 Level 0 primitives |
| **Full-mode determinism** | `test/physics/test_flock.py` (+132) | 5-mode parametric sweep + per-mode determinism tests |
| **Golden regeneration** | `test/data/golden_*.npz` (3 files regenerated) | projection, vicsek, influencer golden files regenerated |
| **Shared test helper** | `test/helpers.py` (+8) | `_call_force(fn, flock, cfg)` reduces 80-char boilerplate |

---

## Documentation & Roadmap

| File | Δ | Description |
|---|---|---|
| `roadmap_deepseek.md` | +206/−213 | Phase cross-references updated |
| `sci/todo_claude*.md` (14 files) | −5,354 (deleted) | Superceded todo files removed |

---

## Summary

| Category | Files | Net Δ |
|---|---|---|
| Core `pymurmur/` | 22 | +1,655/−376 |
| Test suite | ~40 | +1,350/−250 |
| Config presets | 8 | +12/−22 |
| Documentation | 2 | +939/−361 |
| Deleted sci/todo | 14 | −5,354 |
| **Total** | **87** | **+3,817/−7,822** |

**Key outcomes:**
- `flock↔forces` import cycle broken
- All 5 force modes deterministic + golden-verified
- 0 orphan config fields (AST drift detector)
- Public package facade (`from pymurmur import SimConfig, SimulationEngine, Recorder`)
- `SimConfig` split into 17 composed sub-dataclasses
- Extension lifecycle: per-frame enable check, T/K toggles without reset
- Pure render: `Visualizer.render()` never steps the simulation
- Batched occlusion: zero Python object allocations in hot path
- 38 property tests for Level 0 force primitives
