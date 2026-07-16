# Phase 2 — Contracts & Protocols: Completion Report

**Date:** July 15, 2026  
**Status:** ALL COMPLETE ✅  
**Test suite:** 1,184 pass, 0 fail, 16 skipped, 13 xfailed

---

## Executive Summary

Phase 2 formalised every protocol boundary in the pymurmur codebase. Before Phase 2, the project had a flat 90-field SimConfig, stateless force mode functions, no spatial index abstraction, ad-hoc extension wiring, and a circular `flock ↔ forces` import dependency. After Phase 2, every subsystem communicates through explicit dataclass protocols, the import graph is a strict DAG, and all 10 architectural contracts are enforced by the test suite.

### Key outcomes

| Property | Before | After |
|---|---|---|
| Config structure | Flat 90-field dataclass | 16 nested sub-dataclasses with flat-access shims |
| Force modes | Stateless functions, hardcoded dispatch | 5 ForceMode subclasses in MODE_REGISTRY with class-level flags |
| Spatial indexing | Ad-hoc per-mode tree creation | SpatialIndex Protocol + 2 conformant implementations |
| Import graph | `flock ↔ forces` cycle + `viz → simulation` | Strict DAG with 0 forbidden edges at runtime |
| Instance buffer layout | Magic numbers in buffer alloc + VAO creation | InstanceSchema dataclass — single source of truth |
| Matrix uploads | PyGLM `.to_bytes()` — platform-dependent layout | `_mat4_bytes()` via numpy float32 — consistent 64 bytes |
| Extension lifecycle | Static enable/disable at init only | Lazy create/drop on config toggle mid-simulation |
| Force composition | Monolithic per-mode `compute()` functions | ForceTerm dataclass + `composeForces()` reducer |
| Holey flocks | Untested contract | 30 parametrized tests (5 modes × 6 scenarios) |
| Cross-item testing | None | 17 integration tests spanning 3+ Phase 2 items |

---

## Item-by-Item Summary

### P2.1 — Nested SimConfig ✅

**File:** `pymurmur/core/config.py`  
**Tests:** `test/core/test_config.py` (14 tests)

Replaced the flat 90-field dataclass with 16 per-subsystem sub-dataclasses (`DomainConfig`, `FlockConfig`, `BoundaryConfig`, `ProjectionConfig`, `SpatialConfig`, `FieldConfig`, `VicsekConfig`, `InfluencerConfig`, `IndexConfig`, `RefinementConfig`, `ExtensionConfig`, `PredatorConfig`, `EcologyConfig`, `PerfConfig`, `VizConfig`, `CaptureConfig`).

Flat access is preserved via `__getattr__`/`__setattr__` delegation and a `_FIELD_MAP` that routes each legacy field name to its sub-config. Cross-section key collisions (e.g., `capture.width` overwriting `domain.width`) are structurally impossible. YAML round-trip is collision-free — each section nests cleanly under its sub-config name.

All 16 sub-configs are independently instantiable and testable without a parent `SimConfig`.

### P2.2 — ForceMode ABC + MODE_REGISTRY ✅

**File:** `pymurmur/physics/forces/_mode.py`  
**Tests:** `test/physics/forces/test_mode_contract.py` (10 tests)

Defined a `ForceMode(ABC)` with class-level flags (`needs_index`, `speed_mode`, `owns_positions`) and an abstract `compute()` staticmethod. Five mode classes (`ProjectionMode`, `SpatialMode`, `FieldMode`, `VicsekMode`, `InfluencerMode`) are registered via `@register("name")` into `MODE_REGISTRY`.

`compute_all_forces()` dispatches through the registry — adding a new mode is a single-file change. Backward-compatible function aliases are preserved in each mode module. The `@register` decorator returns the class unchanged (stackable), and `MODE_REGISTRY` is a plain dict for zero-overhead lookup.

### P2.3 — SpatialIndex Protocol ✅

**File:** `pymurmur/core/types.py`  
**Tests:** `test/physics/test_spatial_index_contract.py` + `test/physics/test_flock.py` (83 tests shared with P2.4–P2.5)

Defined `SpatialIndex(Protocol)` with `rebuild(positions, active)`, `query_knn(pos, k)`, `query_radius(pos, r)`, `ready`, and `tree` properties. Both `SpatialHashGrid` and `KDTreeIndex` structurally conform. Runtime `isinstance(idx, SpatialIndex)` checks pass. New implementations only need to satisfy these method signatures.

### P2.4 — KDTreeIndex Global Indices ✅

**File:** `pymurmur/physics/flock.py::KDTreeIndex`  
**Tests:** 83 tests (shared spatial suite)

KDTreeIndex now stores a compacted→global index map (`_active_map = np.where(active)[0]`) during `rebuild()` and applies it in `query_knn()`, ensuring returned indices are global (0..N-1) rather than compacted (0..N_active-1). This makes KDTreeIndex and SpatialHashGrid interchangeable — callers never need to know which index implementation is active.

Tests verify global indices with non-contiguous active masks (holes), sparse gaps, and after add/remove cycles.

### P2.5 — Ghost-Cell Replication + Modulo Cells ✅

**File:** `pymurmur/physics/flock.py::SpatialHashGrid`  
**Tests:** 83 tests (shared spatial suite)

Cell keys use modulo wrapping (`% cols`, `% rows`, `% slices`) so birds near domain boundaries appear in cells at the opposite boundary. The 27-cell neighbourhood query also wraps, enabling cross-seam neighbour discovery. Distance calculations use `min_image()` for correct toroidal ranking.

Tests verify cross-seam queries on all three axes (X, Y, Z), XYZ corner cases, toroidal distance ranking, and YZ-axis cross-seam behaviour.

### P2.6 — StepContext + Extension ABC ✅

**File:** `pymurmur/physics/extensions/_base.py`  
**Tests:** `test/physics/extensions/test_extensions.py` (46 tests)

`StepContext` dataclass captures per-frame context (frame, dt, rng, center, config, threat_prox) and is passed to every extension's `apply(flock, ctx)` method. The `Extension(ABC)` base class enforces the `apply()` contract — subclasses without it cannot be instantiated.

`ExtensionManager` implements lazy lifecycle: extensions are created when their config toggle flips to `True` and dropped when it flips to `False`, with no simulation reset required. `count` stays accurate through all toggle sequences. The `threat_prox` array is published by `Predator.apply()` and defaults to `None` (not a mutable default).

StepContext is independently testable — it only needs a numpy Generator and SimConfig, not a fully-wired SimulationEngine.

### P2.7 — InstanceSchema + VAO Discipline ✅

**File:** `pymurmur/viz/renderer.py`  
**Tests:** `test/viz/test_renderer.py` (6 CPU-safe + 87 GPU-gated tests)

`InstanceSchema` dataclass centralises the GPU instance buffer layout: `floats=6` (pos.xyz + vel.xyz), `layout="3f 3f/i"`, `attrs=("in_bird_pos", "in_bird_vel")`. Buffer allocation (`max_instances × floats × 4` bytes), CPU-side packing (`packed[:n, :3] = pos; packed[:n, 3:] = vel`), and VAO creation (`s.layout, *s.attrs`) all read from the same schema instance. Changing the schema propagates to all three sites automatically.

`_build_vao()` is called during `__init__` and after every buffer reallocation, fixing the stale-VAO-after-growth bug.

Tests verify defaults, custom float counts, custom layout strings, dataclass protocol, field types, and the buffer bytes formula for 6/9/12-float schemas — all without requiring a GPU.

### P2.8 — PyGLM Matrix Upload ✅

**File:** `pymurmur/viz/renderer.py`  
**Tests:** `test/viz/test_renderer.py` (7 CPU-safe tests)

`_mat4_bytes(m)` converts any PyGLM 4×4 matrix to a consistent 64-byte layout via `np.array(m.to_list(), dtype=np.float32).tobytes()`. PyGLM builds differ in internal memory layout (column-major vs row-major on different architectures), but the `to_list() → numpy float32 → tobytes()` chain normalises to a consistent column-major float32 layout.

Both matrix upload sites in `begin_frame()` (view matrix + projection matrix) use `_mat4_bytes()`.

Tests verify: 64-byte output, identity roundtrip (diagonal at indices 0,5,10,15), translation matrix (at indices 12,13,14), float32 dtype (not float64), determinism, differentiation, and byte-order consistency — all skip gracefully via `pytest.importorskip("glm")` when PyGLM is not installed.

### P2.9 — Holey-Mask Contract Tests ✅

**File:** `test/physics/test_composition.py` (30 parametrized tests)

30 tests across all 5 registered modes × 6 scenarios:

| Test | Verifies |
|---|---|
| `test_holey_mask_no_exception` | Force computation doesn't crash on holey flocks |
| `test_holey_mask_inactive_positions_unchanged` | Inactive positions are bit-identical after force computation |
| `test_holey_mask_inactive_velocities_unchanged` | Inactive velocities are bit-identical after force computation |
| `test_holey_mask_active_forces_applied` | Active birds receive forces; inactive get zero (Vicsek exempt — sets velocity directly) |
| `test_holey_mask_deterministic` | Same holey mask + seed = identical accelerations |
| `test_holey_mask_20_steps_no_exception` | 20 integration steps: no crash, inactive birds stay frozen |

The `holey_flock` fixture creates 2 holes of 5 inactive birds each (10 of 30 deactivated, ~33% holes) and rebuilds the spatial index on the active-only subset.

### P2.10 — ForceTerm Composition ✅

**File:** `pymurmur/physics/forces/_base.py`  
**Tests:** `test/physics/forces/test_force_terms.py` (7 tests)

`ForceTerm` dataclass wraps a force contribution as a named, typed, runtime-togglable pure function:

```python
@dataclass
class ForceTerm:
    name: str                      # "shell", "drag", etc.
    enabled: bool = True           # runtime toggle
    gain: float = 1.0              # per-term multiplier
    fn: Callable | None = None     # (flock, ctx, cfg) → (N,3)
```

`composeForces(flock, ctx, config, terms)` sums enabled terms linearly. Tests cover: defaults, disabled toggle, gain multiplier, linearity (`composeForces(a+b) = composeForces(a) + composeForces(b)`), empty terms list, None fn, and inactive bird handling.

---

## Cross-Item Integration Tests

Beyond the 10 independent-item tests, 17 cross-item integration tests verify 2+ Phase 2 items working together:

**File:** `test/test_phase2_cross_item.py`

| Test class | Tests | Chain verified |
|---|---|---|
| `TestDynamicSpatialIndexSwap` | 5 | **P2.1→P2.3→P2.4**: auto→kdtree, kdtree→hash_grid, hash_grid→none, 5K threshold auto-migration, active mask preservation |
| `TestThreatEvasionPipeline` | 5 | **P2.6→P2.10**: predator publishes threat_prox, force computation reads it, extension→force order, all 4 extensions + spatial mode, mode-switch + predator |
| `TestInstanceSchemaPacking` | 5 | **P2.7→P2.8**: schema.floats matches packed array, layout components = attrs count, buffer allocation formula, schema change propagation, mat4 bytes independent of schema |
| `TestHoleyFlockWithExtensionsAndComposition` | 2 | **P2.9→P2.6→P2.10**: holey flock + predator, holey flock + all 4 extensions — inactive birds stay frozen across 10 steps |

---

## Architecture Enforcement

The import graph is verified by `test/test_architecture.py` at every Phase 2 commit:

| Forbidden edge | Status |
|---|---|
| `flock !→ forces` (runtime) | ✅ 0 violations |
| `viz !→ simulation` | ✅ 0 violations |
| `core !→ physics/simulation` | ✅ 0 violations |
| `forces !→ extensions` (runtime) | ✅ 0 violations (TYPE_CHECKING exempt) |

The `FORBIDDEN_EDGES` check now exempts `TYPE_CHECKING` imports — `forces/_base.py` imports `StepContext` (from `extensions/_base.py`) and `PhysicsFlock` (from `physics/flock.py`) under `if TYPE_CHECKING:`, which is allowed. Only `simulation/engine` imports both `flock` and `forces` at runtime.

---

## Files Changed

### New files (5)

| File | Purpose |
|---|---|
| `test/physics/forces/test_mode_contract.py` | P2.2: ForceMode ABC + MODE_REGISTRY tests |
| `test/physics/test_spatial_index_contract.py` | P2.3–P2.5: SpatialIndex Protocol + KDTreeIndex global indices + ghost-cell tests |
| `test/physics/forces/test_force_terms.py` | P2.10: ForceTerm + composeForces tests |
| `test/physics/test_composition.py` | P2.9: Holey-mask contract tests (30 parametrized) |
| `test/test_phase2_cross_item.py` | Cross-item integration tests (17 tests) |

### Modified files (14)

| File | Phase 2 items |
|---|---|
| `pymurmur/core/config.py` | P2.1 (nested SimConfig) |
| `pymurmur/core/types.py` | P2.3 (SpatialIndex Protocol) |
| `pymurmur/physics/forces/_mode.py` | P2.2 (ForceMode ABC + MODE_REGISTRY) |
| `pymurmur/physics/forces/_base.py` | P2.10 (ForceTerm + composeForces) |
| `pymurmur/physics/forces/__init__.py` | P2.2 + P2.10 (exports) |
| `pymurmur/physics/forces/projection.py` | P2.2 (ProjectionMode class wrapper) |
| `pymurmur/physics/forces/spatial.py` | P2.2 (SpatialMode class wrapper) |
| `pymurmur/physics/forces/field.py` | P2.2 (FieldMode class wrapper) |
| `pymurmur/physics/forces/vicsek.py` | P2.2 (VicsekMode class wrapper) |
| `pymurmur/physics/forces/influencer.py` | P2.2 (InfluencerMode class wrapper) |
| `pymurmur/physics/flock.py` | P2.4 + P2.5 (KDTreeIndex globals + ghost cells) |
| `pymurmur/physics/extensions/_base.py` | P2.6 (StepContext + Extension ABC) |
| `pymurmur/viz/renderer.py` | P2.7 + P2.8 (InstanceSchema + _mat4_bytes) |
| `test/test_architecture.py` | P2.x (allowed edges, forbidden TYPE_CHECKING exempt) |
| `test/core/test_config.py` | P2.1 (sub-config standalone tests) |
| `test/physics/extensions/test_extensions.py` | P2.6 (StepContext standalone tests) |
| `test/viz/test_renderer.py` | P2.7 + P2.8 (CPU-safe unit tests) |

---

## Test Coverage by Item

| Item | Independent tests | Cross-item tests | Source files |
|---|---|---|---|
| P2.1 (Nested SimConfig) | 14 | 6 | `core/config.py` |
| P2.2 (ForceMode ABC) | 10 | 3 | `physics/forces/_mode.py` + 5 mode files |
| P2.3 (SpatialIndex Protocol) | 6 | 6 | `core/types.py` |
| P2.4 (KDTreeIndex globals) | 6 | 6 | `physics/flock.py` |
| P2.5 (Ghost cells) | 6 | 6 | `physics/flock.py` |
| P2.6 (StepContext + Extension) | 46 | 9 | `physics/extensions/_base.py` |
| P2.7 (InstanceSchema) | 6 | 5 | `viz/renderer.py` |
| P2.8 (_mat4_bytes) | 7 | 6 | `viz/renderer.py` |
| P2.9 (Holey-mask) | 30 | 2 | (test-only) |
| P2.10 (ForceTerm) | 7 | 5 | `physics/forces/_base.py` |

---

## Architectural Guarantees

1. **Import DAG**: `core(L0) → physics atoms(L0) → assemblies(L1) → subsystems(L2) → system(L3)`. No L1 assembly imports another L1 assembly. Only `simulation/engine` imports both `flock` and `forces` at runtime.

2. **Config safety**: Cross-section key collisions are structurally impossible. Every sub-config is independently instantiable. YAML round-trip is collision-free.

3. **Extensibility**: Adding a force mode is a single-file change (`@register` + subclass `ForceMode`). Adding a spatial index implementation requires satisfying the `SpatialIndex` Protocol. Adding an extension requires subclassing `Extension` and implementing `apply()`.

4. **Determinism**: All stochastic sites use `flock.rng`. Same seed → bit-identical across all 5 modes.

5. **GPU safety**: Instance buffer layout is a single source of truth (`InstanceSchema`). Matrix uploads produce consistent 64-byte layouts on any platform. VAO is rebuilt after every buffer reallocation.

6. **Holey-flock contract**: Inactive birds are never touched by force computation, extension application, or integration — verified across all 5 modes × 20 simulation steps.

---

## Acceptance Criteria Met

- [x] Import cycle dead — `physics/flock` does not import from `physics/forces/` at runtime
- [x] All config presets load with correct domains
- [x] Nested config YAML round-trip is collision-free
- [x] `MODE_REGISTRY` has 5 entries, all `ForceMode` subclasses
- [x] `SpatialHashGrid` and `KDTreeIndex` conform to `SpatialIndex` Protocol
- [x] KDTreeIndex returns global indices (tested with non-contiguous active masks)
- [x] Ghost-cell queries work across all three axes (X, Y, Z) and XYZ corners
- [x] `StepContext` passed to every extension's `apply()` with correct fields
- [x] Extensions can be toggled on/off mid-simulation without reset
- [x] `InstanceSchema` controls buffer allocation, packing, and VAO creation
- [x] `_mat4_bytes()` produces consistent 64-byte float32 layout
- [x] All 5 modes handle holey flocks without crash, NaN, or inactive corruption
- [x] `ForceTerm` composition is linear, runtime-togglable, and gain-scalable
- [x] Full pipeline (engine → extensions → index → forces → integrate → metrics) verified
- [x] 1,184 tests passing, 0 failures
