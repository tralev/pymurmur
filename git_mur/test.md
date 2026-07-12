# Test Plan — pymurmur 3D Murmuration Simulation

> **STATUS: LEGACY.** This plan documents the *current* flat test suite
> (`test/test_subsystem_*.py`, `test/test_imports.py`, …). The forward
> test plan is **`roadmap.md`** — Part II (infrastructure T0–T6) and the
> inline test blocks of Part III (S-items). Per roadmap Appendix C, this
> file is retired or slimmed at phase D9, once the tree migrates to
> mirrored subpackages (`test/core/`, `test/physics/`, …). Until then it
> remains the reference for the existing tests only. Do not add new test
> specs here.

> **Organization:** Top-Down / Macro-to-Micro, matching the original five refinement levels (pre-`roadmap.md`).
> **Design docs:** `arch.md` — single architecture reference; §2 contains both views (Top-Down / Functional Decomposition and Bottom-Up / Component Assembly).
> **Strategy:** Test at each refinement level — System Goal validation → Subsystem isolation → Module unit tests → Implementation detail validation. Test the smallest components first (even in a Top-Down doc, execution is bottom-up).
> **Framework:** `pytest` with `numpy` test helpers. GPU-dependent tests use `pytest.mark.skipif` when no GPU context is available.

## Test Cheatsheet

Quick reference for writing and running tests. Keep this section updated as the test suite evolves.

### Run Commands

```bash
# All fast tests (skip GPU, skip slow) — run this for quick feedback
pytest test/ -v -m "not slow" --ignore=test/viz/test_renderer.py --ignore=test/viz/test_input.py

# CI: fast suite — what runs on every commit
pytest test/core/ test/physics/ test/simulation/ test/analysis/ test/capture/ test/viz/test_camera.py test/test_imports.py test/test_architecture.py test/test_docs.py test/test_golden.py -v

# Guard rails (architecture, docs, golden, config drift, collection count) — run on every commit
pytest test/test_architecture.py test/test_docs.py test/test_golden.py -v

# Single file with verbose output
pytest test/physics/forces/test_base.py -v -s

# Match by test name pattern
pytest test/ -k "test_steric" -v

# Performance & scaling (slow, nightly/PR-merge only)
pytest -m slow test/test_performance.py test/test_scaling.py test/test_config_files.py -v

# Full suite including GPU + pygame (requires display)
pytest test/ -v
```

> **Phase-by-phase commands:** See "Test Execution Strategy" at the bottom of this document — those commands are kept in sync with `roadmap.md` phases and are the authoritative reference.

### Pytest Markers

| Marker | Purpose | Example |
|--------|---------|---------|
| `slow` | Performance/benchmark tests, run on PR merge or nightly | `@pytest.mark.slow` |
| `gpu` | Tests requiring ModernGL GPU context | `@pytest.mark.gpu` |
| `numba` | Tests requiring numba JIT | `@pytest.mark.numba` |
| `pygame` | Tests requiring pygame event loop | `@pytest.mark.pygame` |
| `golden` | Golden trajectory regression tests | `@pytest.mark.golden` |
| `e2e` | End-to-end tests that run the full simulation | `@pytest.mark.e2e` |
| `integration` | Tests wiring multiple modules together | `@pytest.mark.integration` |

### Conditional Skip Patterns

```python
import pytest

# Skip if GPU unavailable
@pytest.mark.skipif(not gpu_available, reason="GPU required")
def test_renderer_init():
    ...

# Skip if numba unavailable
@pytest.mark.skipif(not numba_available, reason="numba required")
def test_numba_jit():
    ...

# Skip if pygame unavailable
try:
    import pygame
except ImportError:
    pytest.skip("pygame not installed", allow_module_level=True)
```

### Shared Fixtures (`conftest.py`)

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

### Common Test Patterns

```python
# 1. Test vectorised output shape
assert result.shape == (N, 3)
assert result.dtype == np.float32

# 2. Test zero output for edge case
result = force_fn(positions, velocities, empty_neighbors)
assert np.allclose(result, 0.0)

# 3. Test inactive birds produce no force
result = force_fn(positions, velocities, neighbors)
assert np.allclose(result[~active], 0.0)

# 4. Test reproducibility with seeding
result_1 = generate_with_seed(42)
result_2 = generate_with_seed(42)
assert np.array_equal(result_1, result_2)

# 5. Test direction sanity with dot product
force = separation_force(pos_i, pos_j)
assert np.dot(force, pos_i - pos_j) > 0  # away from neighbor

# 6. Test monotonic behavior
force_near = compute(dist=1.0)
force_far = compute(dist=10.0)
assert np.linalg.norm(force_near) > np.linalg.norm(force_far)

# 7. Test speed clamping
velocities = np.array([[100, 0, 0]])
integrate(positions, velocities, accelerations, dt=1.0, v0=4.0)
assert np.linalg.norm(velocities[0]) <= 4.1

# 8. Test boundary wrapping
pos = np.array([[domain_width + 1.0, 0, 0]])
apply_toroidal(pos, domain_width, domain_height, domain_depth)
assert pos[0, 0] == 1.0

# 9. Test no-crash for zero birds
result = force_fn(empty_positions, empty_velocities, empty_neighbors)
assert result.shape == (0, 3)

# 10. Test O(N) scaling (rough)
import time
t_small = timeit(lambda: fn(small_flock), number=10)
t_large = timeit(lambda: fn(large_flock), number=10)
ratio = t_large / t_small
ratio_N = len(large_flock) / len(small_flock)
assert ratio < ratio_N * 1.5  # roughly linear
```

---

## Directory Organization

All input files (YAML configs) MUST be placed within the `conf/` directory. All output files (captures, metrics CSV/JSON, evolved configs) MUST be placed within the `output/` directory. All test files MUST be placed within the `test/` directory — mirroring the `pymurmur/` sub-package structure. All Docker, Docker Compose, and CI configuration files MUST be placed within the `ci/` directory.

```
test/
├── __init__.py
├── conftest.py
├── core/
│   ├── test_types.py
│   └── test_config.py
├── simulation/
│   └── test_engine.py
├── physics/
│   ├── test_boid.py
│   ├── test_flock.py
│   ├── test_occlusion.py
│   ├── test_steric.py
│   ├── forces/
│   │   ├── test_base.py
│   │   ├── test_forces.py
│   │   └── test_projection.py
│   └── extensions/
│       └── test_extensions.py
├── viz/
│   ├── test_renderer.py
│   ├── test_camera.py
│   ├── test_shaders.py
│   └── test_input.py
├── capture/
│   └── test_recorder.py
├── analysis/
│   ├── test_metrics.py
│   └── test_presets.py
├── test_subsystem_a.py
├── test_subsystem_b.py
├── test_subsystem_c.py
├── test_subsystem_d.py
├── test_subsystem_e.py
├── test_subsystem_f.py
├── test_performance.py
├── test_scaling.py
├── test_e2e.py
├── test_imports.py
├── test_config_files.py
└── test_config_resolution.py
```

---

## Mapping arch.md Levels to Test Sections

| arch.md Level (Top-Down) | Test Section | What's tested |
|--------------------------|-------------|---------------|
| Level 0 — System Goal | § System-Wide Tests | CLI dispatch, all modes functional, E2E scenarios |
| Level 1 — Functional Decomposition (A–F) | § Subsystem Isolation Tests | Each subsystem A–F tested independently with mocked dependencies |
| Level 2 — Subsystem Design (Meso) | § Integration Tests | Subsystem internals wired correctly (SimulationEngine, Visualizer, Recorder) |
| Level 3 — Module Interfaces (Micro) | § Unit Tests | Individual modules: types, forces, occlusion, spatial index, integration, camera, shaders |
| Level 4 — Implementation Details (Nano) | § Cross-Cutting Tests | Performance, scaling, config validation, imports, reuse |

---

## System-Wide Tests (arch.md Level 0 — System Goal)

Verify that the full system meets its stated goals: load config, run 5 modes, visualize optionally, capture optionally, run headlessly.

### SYS.1 Entry Point Dispatch (`test/__main__.py` or `test/test_subsystem_a.py`)

| Test | What it verifies |
|------|-----------------|
| `test_cli_no_args` | `python -m pymurmur` runs with built-in defaults (projection, N=150) |
| `test_cli_config_name` | `--config murmuration_spatial` loads `conf/murmuration_spatial.yaml` |
| `test_cli_config_path` | `--config /path/to/custom.yaml` loads from absolute path |
| `test_cli_config_not_found` | `--config nonexistent` raises `FileNotFoundError` |
| `test_cli_config_no_extension` | `--config murmuration_spatial` auto-appends `.yaml` |
| `test_cli_list_configs` | `--list-configs` prints all 7 config files with descriptions |
| `test_cli_list_configs_includes_new` | A new `.yaml` dropped in `conf/` appears in `--list-configs` output |
| `test_cli_no_viz` | `--no-viz` runs without importing pygame/ModernGL |
| `test_cli_capture` | `--capture` creates `output/` directory with .gif and .csv |
| `test_cli_help` | `--help` prints usage and exits |

### SYS.2 Config Resolution (`test/test_config_resolution.py`)

| Test | What it verifies |
|------|-----------------|
| `test_resolve_shipped_first` | `load_config("murmuration_spatial")` finds `conf/murmuration_spatial.yaml` before any path |
| `test_resolve_path_fallback` | `load_config("/tmp/custom.yaml")` resolves to absolute path when `conf/` has no match |
| `test_resolve_none_returns_default` | `load_config(None)` returns `SimConfig()` with all defaults |
| `test_resolve_name_with_dot_yaml` | `load_config("murmuration.yaml")` still resolves correctly (don't double-append) |
| `test_resolve_missing_raises` | `load_config("nonexistent")` raises `FileNotFoundError` with helpful message listing `conf/` contents |

### SYS.3 All Modes Functional

| Test | What it verifies |
|------|-----------------|
| `test_all_five_modes_start` | Each of the 5 modes runs 10 steps without crash |
| `test_mode_switch_no_crash` | Switching mode mid-simulation (M key or config.mode change) doesn't crash |
| `test_project_mode_uses_occlusion` | Projection mode calls `spherical_cap_occlusion` (spy/mock) |
| `test_spatial_mode_uses_cKDTree` | Spatial mode with N≥5000 builds a cKDTree |
| `test_field_mode_no_neighbor_queries` | Field mode never queries spatial index |
| `test_influencer_mode_no_neighbor_queries` | Influencer mode never queries spatial index |
| `test_vicsek_mode_constant_speed` | Vicsek mode maintains constant speed for all birds |

---

## Subsystem Isolation Tests (arch.md Level 1 — Functional Decomposition A–F)

Each of the 6 subsystems from arch.md's functional decomposition is tested in isolation with mocked dependencies.

### A — Entry & Configuration (`test_subsystem_a.py`)

| Test | What it verifies |
|------|-----------------|
| `test_simconfig_all_defaults` | `SimConfig()` has all 40+ documented default values |
| `test_simconfig_from_file_flattens_nested` | `domain.width` in YAML → `config.width` |
| `test_simconfig_from_file_all_five_modes` | YAML with all 5 mode sections parses correctly; only active mode's fields are applied |
| `test_simconfig_from_file_unknown_keys_ignored` | Extra YAML keys don't raise errors (forward-compatible) |
| `test_simconfig_to_file_roundtrip` | `config.to_file(path)` → `SimConfig.from_file(path)` produces identical config |
| `test_simconfig_live_mutable_vs_static` | `phi_p` and `sigma` are tagged mutable; `width` and `boid_size` are tagged static |
| `test_simconfig_performance_defaults` | `use_numba=True`, `spatial_index="auto"`, `metrics_detail_level=1`, `metrics_interval=60`, `instance_buffer_chunk=50000` |
| `test_load_all_seven_shipped_configs` | All 7 `conf/*.yaml` parse without error |
| `test_config_search_path_order` | `load_config(name)` tries `conf/{name}.yaml` first, then `{name}` as path, then raises |

### B — Simulation Engine (`test_subsystem_b.py`)

| Test | What it verifies |
|------|-----------------|
| `test_engine_imports_no_viz` | `simulation/engine.py` does not import `pygame`, `moderngl`, `PyGLM`, `Pillow` |
| `test_engine_step_order` | Extensions.pre_step → flock.step → metrics.collect (verified via mock/spy) |
| `test_engine_headless_no_callback` | `run_headless(steps=100)` completes without error |
| `test_engine_headless_with_callback` | Callback called exactly once per step |
| `test_engine_reset_restores_initial_state` | After `reset()`, frame=0, flock is fresh, metrics are empty |
| `test_engine_config_live_mutation` | Mutating `config.phi_p` between steps affects next step's forces |

### C — Visualization & Input (`test_subsystem_c.py`)

| Test | What it verifies |
|------|-----------------|
| `test_renderer_no_simulation_imports` | `viz/renderer.py` does not import `simulation` |
| `test_input_control_no_simulation_imports` | `viz/input_control.py` does not import `simulation` |
| `test_camera_no_moderngl_import` | `viz/camera.py` does not import `moderngl` |
| `test_shaders_no_moderngl_import` | `viz/shaders.py` does not import `moderngl` |
| `test_input_to_config_bridge` | `InputControl` only mutates `SimConfig` fields — no direct simulation access |
| `test_visualizer_starts_without_crash` | `Visualizer(sim, config)` initializes renderer + camera + trails |
| `test_full_keyboard_map_coverage` | All 21 documented key bindings are handled without unhandled exceptions |

### D — Capture & Export (`test_subsystem_d.py`)

| Test | What it verifies |
|------|-----------------|
| `test_recorder_no_viz_mode` | `Recorder(sim, capture_config, with_viz=False)` captures only metrics, no frames |
| `test_recorder_with_viz_mode` | `with_viz=True` captures both frames and metrics |
| `test_recorder_gif_output_valid` | Output .gif has correct magic bytes (GIF89a/GIF87a), non-zero size |
| `test_recorder_csv_columns_match_metrics` | CSV column count equals `FlockMetrics` field count |
| `test_recorder_json_valid_metadata` | JSON output includes `seed`, `mode`, `num_boids`, `frame_count` metadata |
| `test_recorder_empty_run_no_crash` | Zero frames → `save_gif()` handles gracefully |

### E — Physics & Forces (`test_subsystem_e.py`)

| Test | What it verifies |
|------|-----------------|
| `test_flock_soa_memory_budget` | `FlockArrays` at N=300K uses < 15 MB (validate size estimate) |
| `test_spatial_index_auto_select` | N < 5000 → `SpatialHashGrid`, N ≥ 5000 → `KDTreeIndex` |
| `test_two_pass_architecture` | Spatial force: cKDTree query pass returns `neighbor_idx`, force pass reads it |
| `test_numba_jit_fallback` | When numba unavailable, numpy path produces bit-identical results |
| `test_all_five_modes_return_valid_forces` | Every mode function returns forces within `[-max_force, max_force]` |
| `test_mode_dispatch_unknown_raises` | `compute_all_forces(flock, config)` with invalid mode raises `KeyError` |
| `test_extensions_pre_step_before_forces` | Extensions forces are applied before main force computation |
| `test_occlusion_soa_adapter_bit_identical` | `spherical_cap_occlusion_soa` matches original for identical inputs |

### F — Metrics & Analysis (`test_subsystem_f.py`)

| Test | What it verifies |
|------|-----------------|
| `test_metrics_fast_o_n` | All 9 fast metrics compute in O(N) time (validate timing scales linearly) |
| `test_metrics_gating_level_0` | `detail_level=0` → zero metrics computed, returns empty `FlockMetrics` |
| `test_metrics_gating_level_1` | `detail_level=1` → only fast metrics populated, expensive fields are `None` or `NaN` |
| `test_metrics_gating_level_2` | `detail_level=2` → all 15 metrics populated |
| `test_metrics_interval_respected` | Expensive metrics computed every `interval` frames, not every frame |
| `test_metrics_collector_snapshot` | `snapshot()` returns a `FlockMetrics` with correct types for all fields |
| `test_presets_all_valid_fields` | Every preset dict contains only valid `SimConfig` field names |
| `test_presets_do_not_mutate_originals` | Applying a preset doesn't modify the `PRESETS` dictionary |

---

## Unit Tests — Individual Modules (arch.md Level 3 — Module Interfaces)

Every Level 0 component is independently testable with no project imports beyond `types.py`. All tests are pure unit tests.

### 0.1 Data Types (`test/core/test_types.py`)

| Test | What it verifies |
|------|-----------------|
| `test_flock_arrays_creation` | `FlockArrays` with N=150 has correct shapes: positions (150,3), velocities (150,3), accelerations (150,3), seeds (150,), last_theta (150,), active (150,) |
| `test_flock_arrays_default_active` | All `active` entries are `True` on creation |
| `test_flock_arrays_n_active` | `N_active` property equals `active.sum()` |
| `test_flock_arrays_dtype` | All arrays use `float32` dtype |
| `test_flock_arrays_zero_birds` | N=0 produces empty arrays without error |
| `test_force_func_protocol` | A function matching `ForceFunc` signature passes `isinstance` check |
| `test_force_kernel_signature` | A function matching `ForceKernel` accepts 11 args (5 arrays + 6 scalars) |

### 0.2 BoidView (`test/physics/test_boid.py`)

| Test | What it verifies |
|------|-----------------|
| `test_boid_view_pos` | `BoidView(idx, flock).pos` returns correct position from `flock.positions[idx]` |
| `test_boid_view_vel` | `BoidView(idx, flock).vel` returns correct velocity from `flock.velocities[idx]` |
| `test_boid_view_read_only` | Attempting to assign `view.pos = ...` raises `AttributeError` |
| `test_boid_view_uses_slots` | `BoidView` has `__slots__`, no `__dict__` |
| `test_boid_view_out_of_bounds` | `BoidView(-1, flock)` or `BoidView(N, flock)` raises `IndexError` |
| `test_boid_view_after_deactivate` | View still returns correct data after `active[idx] = False` |

### 0.3 Force Primitives (`test/physics/forces/`)

**Setup:** Create a small flock (N=10) with known positions and velocities, a pre-computed `neighbor_idx` array.

| Test | What it verifies |
|------|-----------------|
| `test_separation_force_zero_distance` | Two birds at identical positions produce non-zero repulsion |
| `test_separation_force_direction` | Force points away from neighbor: `dot(force, pos_i - pos_j) > 0` |
| `test_separation_force_falls_with_distance` | Force magnitude decreases as neighbor distance increases |
| `test_separation_force_no_neighbors` | Birds with no neighbors get zero separation force |
| `test_separation_force_inactive_ignored` | Inactive birds don't contribute to separation |
| `test_alignment_force_parallel` | Two birds with identical velocities produce alignment force colinear with velocity |
| `test_alignment_force_opposite` | Two birds with opposite velocities produce alignment force toward zero |
| `test_alignment_force_no_neighbors` | Birds with no neighbors get zero alignment force |
| `test_cohesion_force_toward_center` | Force points toward the center of mass of neighbors |
| `test_cohesion_force_single_neighbor` | With one neighbor, cohesion force points directly toward that neighbor |
| `test_cohesion_force_no_neighbors` | Birds with no neighbors get zero cohesion force |
| `test_noise_force_unit_scale` | `noise_force(N, 1.0)` produces vectors with roughly unit variance |
| `test_noise_force_zero_scale` | `noise_force(N, 0.0)` produces all-zero array |
| `test_noise_force_shape` | `noise_force(N, s)` returns `(N, 3)` float32 |
| `test_force_primitives_inactive_rows` | All primitives return zero force for inactive birds |

### 0.4 Spatial Index (`test/physics/test_flock.py`)

| Test | What it verifies |
|------|-----------------|
| **SpatialHashGrid** | |
| `test_hash_grid_rebuild` | `rebuild()` runs on (N,3) positions + (N,) active without error |
| `test_hash_grid_query_returns_self` | Query at bird's position includes that bird in results |
| `test_hash_grid_query_empty` | Query with radius=0 returns only the queried bird |
| `test_hash_grid_query_all` | Query with very large radius returns all active birds |
| `test_hash_grid_inactive_excluded` | Inactive birds are not returned in queries |
| `test_hash_grid_cell_wrapping` | Query near domain edge wraps to opposite side (toroidal) |
| `test_hash_grid_performance_10k` | `rebuild()` on 10K birds completes in < 50 ms |
| **KDTreeIndex** | |
| `test_kdtree_build` | `build()` on (N,3) positions completes without error |
| `test_kdtree_query_knn` | `query_knn(pos, k=5)` returns 5 indices |
| `test_kdtree_closest_is_self` | First neighbor of a bird is itself (or a very close bird) |
| `test_kdtree_distance_increases` | `query_knn(pos, k=10)` returns indices in order of increasing distance |
| `test_kdtree_performance_100k` | `build()` + `query_knn()` on 100K birds completes in < 200 ms |

### 0.5 Integration Kernel (`test/physics/test_boid.py`)

| Test | What it verifies |
|------|-----------------|
| `test_integrate_applies_acceleration` | After `integrate()`, `velocities += accelerations` holds |
| `test_integrate_resets_acceleration` | After `integrate()`, all active accelerations are zero |
| `test_integrate_moves_position` | After `integrate()` with non-zero velocity, positions change |
| `test_integrate_stationary_bird_stays` | Bird with zero velocity and zero acceleration stays in place |
| `test_speed_clamp_fast` | Bird exceeding `v0` is clamped to exactly `v0` |
| `test_speed_clamp_slow` | Bird below `0.3*v0` is boosted to exactly `0.3*v0` |
| `test_speed_clamp_within_band` | Bird within `[0.3*v0, v0]` keeps its speed unchanged |
| `test_speed_clamp_vectorised` | All N birds clamped correctly in a single call |
| `test_integrate_inactive_unchanged` | Inactive birds' positions and velocities are unchanged |
| `test_boundary_toroidal_x` | Bird crossing `x > width` wraps to `x = 0` |
| `test_boundary_toroidal_negative` | Bird crossing `x < 0` wraps to `x = width` |
| `test_boundary_toroidal_all_axes` | Wrapping works in X, Y, and Z independently |
| `test_boundary_toroidal_velocity_preserved` | Velocity direction unchanged after wrapping |
| `test_boundary_open` | Bird can leave domain freely (no position clamp) |
| `test_boundary_margin_nudge` | Bird near wall gets velocity nudge away from wall |
| `test_boundary_sphere_soft` | Bird outside sphere radius is projected back |
| `test_integrate_dt_scaling` | Doubling `dt` doubles position change for same velocity |

### 0.6 Occlusion Math (`test/physics/test_occlusion.py`)

| Test | What it verifies |
|------|-----------------|
| `test_occlusion_no_neighbors` | Empty neighbor list returns `delta=(0,0,0)`, empty visible, `theta=0` |
| `test_occlusion_single_neighbor` | One neighbor: visible includes it, theta > 0, delta points toward it |
| `test_occlusion_delta_magnitude` | `|delta|` ∈ [0,1] for all test cases |
| `test_occlusion_interior_bird_surrounded` | Bird surrounded by neighbors has `theta → 1` and `|delta| → 0` |
| `test_occlusion_edge_bird` | Bird at edge of flock has `|delta| ≈ 1` (boundary-length-weighted mean points one way) |
| `test_occlusion_theta_never_negative` | `theta >= 0` always |
| `test_occlusion_theta_never_exceeds_one` | `theta <= 1` always |
| `test_occlusion_closest_first` | Visible list is sorted by distance (closest first) |
| `test_occlusion_occluded_behind` | Bird directly behind a nearer bird is excluded from visible |
| `test_occlusion_self_excluded` | Observer is not in neighbor list |
| `test_blind_angle_excludes_rear` | Neighbor directly behind observer excluded when `blind_cos` set |
| `test_blind_angle_forward_visible` | Neighbor in front of observer still visible with blind angle |
| `test_anisotropic_body_broadside` | Bird seen broadside has larger cap radius than end-on |
| `test_marginal_opacity_emerges` | With 150 birds and Pearce defaults, Θ ≈ 0.30 ± 0.10 |
| **SoA Adapter** | |
| `test_occlusion_soa_adapter_same_result` | `spherical_cap_occlusion_soa` gives same result as original with BoidView |
| `test_occlusion_soa_adapter_performance` | Processes σ=6 neighbors in < 1 ms per bird |

### 0.7 Steric Repulsion (`test/physics/test_steric.py`)

| Test | What it verifies |
|------|-----------------|
| `test_steric_zero_strength` | `steric_force(strength=0)` returns zero vector |
| `test_steric_direction_away` | Force points away from neighbor |
| `test_steric_falls_with_distance` | Force magnitude ∝ 1/d² |
| `test_steric_no_neighbors` | Empty neighbor list returns zero vector |
| `test_steric_close_range_only` | Neighbor at distance > threshold produces no force |

### 0.8 Array Helpers (`test/physics/test_boid.py`)

| Test | What it verifies |
|------|-----------------|
| `test_random_positions_shape` | Returns `(N, 3)` float32 |
| `test_random_positions_in_domain` | All positions within `[0, width] × [0, height] × [0, depth]` |
| `test_random_positions_seeded` | Same seed → same positions (reproducible) |
| `test_random_positions_different` | Different seeds → different positions |
| `test_random_unit_sphere_shape` | Returns `(N, 3)` float32 |
| `test_random_unit_sphere_unit_norm` | All vectors have norm approximately 1.0 (± 1e-6) |
| `test_random_unit_sphere_uniform` | Distribution of θ, φ is visually uniform (chi-squared test) |
| `test_random_unit_sphere_seeded` | Same seed → same directions |

---

## Composition Tests — Component Assembly

Tests that verify Level 0 components compose correctly into Level 1 assemblies.

### 1.1 PhysicsFlock (`test/physics/test_flock.py`)

| Test | What it verifies |
|------|-----------------|
| `test_flock_init_creates_birds` | `PhysicsFlock(config)` has `N_active == config.num_boids` |
| `test_flock_init_positions_in_domain` | All positions within domain bounds |
| `test_flock_init_velocities_nonzero` | All velocities have non-zero norm |
| `test_flock_init_accelerations_zero` | All accelerations are zero |
| `test_flock_step_runs` | `flock.step(config, dt)` completes without error |
| `test_flock_step_positions_change` | Positions change after `step()` with non-zero forces |
| `test_flock_step_metrics_accessible` | After `step()`, flock state arrays are in valid state |
| `test_flock_add_boids` | `add_boids(5)` increases `N_active` by 5 |
| `test_flock_add_boids_initializes` | Added birds have non-zero positions and velocities |
| `test_flock_remove_boids` | `remove_boids(5)` decreases `N_active` by 5 |
| `test_flock_remove_boids_deactivates` | Removed birds have `active[i] = False` |
| `test_flock_add_beyond_capacity` | `add_boids()` when all inactive slots are filled raises or extends |
| `test_flock_remove_all` | Removing all birds leaves `N_active = 0` |
| `test_flock_spatial_index_auto_select` | N < 5000 uses SpatialHashGrid, N ≥ 5000 uses KDTreeIndex |
| `test_flock_seeded_reproducible` | Same seed → identical state after N steps |

### 1.2 Force Dispatch (`test/physics/forces/`)

| Test | What it verifies |
|------|-----------------|
| `test_mode_dispatch_projection` | `compute_all_forces()` with `mode="projection"` calls `projection_forces` |
| `test_mode_dispatch_spatial` | `mode="spatial"` dispatches correctly |
| `test_mode_dispatch_field` | `mode="field"` dispatches correctly |
| `test_mode_dispatch_vicsek` | `mode="vicsek"` dispatches correctly |
| `test_mode_dispatch_influencer` | `mode="influencer"` dispatches correctly |
| `test_mode_dispatch_invalid` | Unknown mode raises `KeyError` or `ValueError` |
| `test_mode_switch_mid_simulation` | Changing `config.mode` between steps switches force computation |
| **Spatial Mode** | |
| `test_spatial_mode_all_weights_zero` | All weights=0 → birds move in straight lines (no steering) |
| `test_spatial_mode_separation_only` | Only separation → birds spread apart over time |
| `test_spatial_mode_alignment_only` | Only alignment → velocity directions converge |
| `test_spatial_mode_cohesion_only` | Only cohesion → birds cluster together |
| `test_spatial_mode_noise_only` | Only noise → random walk (positions drift, no convergence) |
| `test_spatial_mode_force_clamped` | No bird's acceleration exceeds `config.max_force` |
| `test_spatial_mode_numba_fallback` | When numba unavailable, numpy path produces identical results |
| **Projection Mode** | |
| `test_projection_mode_delta_computed` | `delta` is non-zero when neighbors present |
| `test_projection_mode_theta_cached` | `last_theta[i]` updated after `projection_forces()` |
| `test_projection_mode_blind_angle_effect` | `blind_deg > 0` changes delta vs. `blind_deg = 0` |
| `test_projection_mode_anisotropy_effect` | `anisotropy > 1` changes cap sizes and resulting delta |
| `test_projection_mode_sigma_effect` | Increasing `sigma` changes the visible neighbor set |
| **Field Mode** (planned — Phase 8) | |
| `test_field_mode_no_errors_16k` | Field mode runs on 16K birds without error |
| `test_field_mode_o1_complexity` | Time scales linearly with N (no neighbor loop) |
| **Vicsek Mode** (planned — Phase 8) | |
| `test_vicsek_constant_speed` | All birds maintain exactly `config.velocity` speed |
| `test_vicsek_order_transition` | High couplage + low noise → high order parameter (> 0.8) |
| `test_vicsek_disorder` | Low couplage + high noise → low order parameter (< 0.3) |
| **Influencer Mode** (planned — Phase 8) | |
| `test_influencer_no_neighbor_queries` | Spatial index is never queried |
| `test_influencer_follows_target` | Flock center of mass tracks the Lissajous target |

### 1.3 Extensions (`test/physics/extensions/`)

| Test | What it verifies |
|------|-----------------|
| `test_extension_manager_empty` | All extensions disabled → `pre_step()` is a no-op |
| `test_predator_spawns` | `predator_enabled=True` → Predator instance created |
| `test_predator_threat_force` | Birds near predator experience non-zero threat force |
| `test_predator_approach_phase` | Predator moves toward flock center initially |
| `test_predator_pass_through` | Predator passes through flock (doesn't circle indefinitely) |
| `test_ecology_day_length` | `Ecology.day_length(day=172)` ≈ 16.5 hours (summer solstice) |
| `test_ecology_day_length_winter` | `Ecology.day_length(day=355)` ≈ 7.5 hours (winter solstice) |
| `test_ecology_dusk_roost_pull` | Near dusk, birds experience downward pull toward roost |
| `test_ecology_critical_mass` | Below critical mass, roosting pull is dampened |
| `test_wander_bounded` | Attractor stays within configured radius |
| `test_ripple_envelope_decay` | Ripple intensity decays with distance from pulse center |

### 1.4 Metrics (`test/analysis/test_metrics.py`)

| Test | What it verifies |
|------|-----------------|
| **Fast Metrics (O(N))** | |
| `test_order_parameter_perfect` | All identical velocities → `alpha = 1.0` |
| `test_order_parameter_random` | Random velocities → `alpha ≈ 0` for large N |
| `test_order_parameter_opposite` | Half up, half down → `alpha = 0` |
| `test_internal_opacity_range` | Θ always in [0, 1] |
| `test_external_opacity_empty` | Empty flock → `theta_prime = 0` |
| `test_external_opacity_dense` | Dense flock → `theta_prime > 0` |
| `test_dispersion_concentrated` | All birds at same point → `dispersion = 0` |
| `test_dispersion_spread` | Birds at corners of domain → high dispersion |
| `test_angular_momentum_linear` | All straight-line motion → `L ≈ 0` |
| `test_angular_momentum_circular` | Circular motion → `L > 0` |
| `test_speed_avg` | `speed_avg` matches `np.mean(np.linalg.norm(velocities, axis=1))` |
| `test_local_spacing` | Known positions → known k=7 neighbor distance |
| **Expensive Metrics (gated)** | |
| `test_h2_complete_graph` | Fully connected flock → H₂ finite |
| `test_h2_smaller_than_N` | H₂ < N always |
| `test_msd_diffusion` | Random walk → MSD grows linearly with time |
| `test_msd_ballistic` | Constant velocity → MSD grows quadratically |
| `test_gyration_radius_trimmed` | Outliers don't dominate R_g (15% tail trim) |
| `test_aspect_ratio_line` | All birds on a line → aspect ≫ 1 |
| `test_aspect_ratio_sphere` | Uniform sphere → aspect ≈ 1 |
| `test_metrics_gating` | `detail_level=0` → no metrics computed |
| `test_metrics_gating_fast` | `detail_level=1` → only fast metrics computed |
| `test_metrics_gating_full` | `detail_level=2` → all metrics computed every `interval` frames |

### 1.5 Presets (`test/analysis/test_presets.py`)

| Test | What it verifies |
|------|-----------------|
| `test_all_presets_valid` | Every preset dict contains only valid `SimConfig` field names |
| `test_preset_apply_changes_mode` | Applying "ball" preset sets `mode = "projection"` |
| `test_preset_apply_changes_weights` | Applying "acro" preset changes `separation_weight` and `noise_scale` |
| `test_preset_does_not_mutate_default` | Applying a preset doesn't modify the `PRESETS` dict entries |
| `test_all_presets_run` | Every preset produces a working simulation (no crash in 10 steps) |

---

## Integration Tests — Subsystem Wiring

Tests that verify Level 1 assemblies wire together into Level 2 subsystems.

### INT.1 SimulationEngine (`test/simulation/test_engine.py`)

| Test | What it verifies |
|------|-----------------|
| `test_engine_init_creates_flock` | Engine has non-null `flock`, `extensions`, `metrics` |
| `test_engine_step_increments_frame` | `engine.frame` increments by 1 per `step()` |
| `test_engine_step_order` | Extensions run before flock.step, metrics after (spy on calls) |
| `test_engine_run_headless_n_steps` | `run_headless(steps=100)` runs exactly 100 steps |
| `test_engine_run_headless_callback` | Callback is called once per step |
| `test_engine_run_headless_forever` | `run_headless(steps=None)` runs until externally stopped |
| `test_engine_reset` | `reset()` creates new flock with same config, frame=0 |
| `test_engine_reset_metrics` | `reset()` clears metrics history |
| `test_engine_no_viz_imports` | `simulation.py` does not import pygame, moderngl, or PyGLM |
| `test_engine_config_live_mutation` | Mutating `config.phi_p` between steps affects next step's forces |
| `test_engine_deterministic_with_seed` | Same seed → identical flock state after 100 steps |

### INT.2 Renderer (`test/viz/test_renderer.py`)

| Test | What it verifies |
|------|-----------------|
| `test_renderer_init` | `Renderer3D(width, height)` creates context without error |
| `test_renderer_headless_init` | `Renderer3D(width, height, headless=True)` creates FBO |
| `test_renderer_update_instances` | `update_instances()` returns correct active count |
| `test_renderer_begin_frame` | `begin_frame(camera)` clears and computes matrices |
| `test_renderer_draw_birds_no_error` | `draw_birds(flock)` completes without GL error |
| `test_renderer_draw_grid_no_error` | `draw_grid()` completes without GL error |
| `test_renderer_capture_frame` | `capture_frame()` returns a PIL Image with correct dimensions |
| `test_renderer_buffer_growth` | Adding more birds than `max_instances` triggers growth |
| `test_renderer_single_memcpy` | `update_instances()` uses `vbo.write()` with a single call |
| `test_renderer_zero_birds` | Rendering with 0 active birds doesn't crash |

> **Note:** Renderer tests require a GPU context. Use `pytest.mark.skipif` with `moderngl` availability check. Headless FBO tests work without a window but still need a GPU.

### INT.3 Camera (`test/viz/test_camera.py`)

| Test | What it verifies |
|------|-----------------|
| `test_camera_default_position` | Default camera position is at expected spherical coordinates |
| `test_camera_rotate_azimuth` | `rotate(pi/2, 0)` rotates azimuth by 90° |
| `test_camera_rotate_elevation` | `rotate(0, pi/4)` rotates elevation by 45° |
| `test_camera_elevation_clamped` | Elevation never exceeds `[-pi/2+0.05, pi/2-0.05]` |
| `test_camera_zoom_in` | `zoom(1)` decreases distance |
| `test_camera_zoom_out` | `zoom(-1)` increases distance |
| `test_camera_distance_clamped` | Distance clamped to `[min_distance, max_distance]` |
| `test_camera_auto_rotate` | `step_auto_rotate(dt)` advances azimuth by `AUTO_ROTATE_SPEED * dt` |
| `test_camera_auto_rotate_off` | When disabled, `step_auto_rotate()` does nothing |
| `test_camera_reset` | `reset()` restores default azimuth, elevation, distance |
| `test_camera_view_matrix_is_mat4` | `view_matrix()` returns `glm.mat4` |
| `test_camera_projection_matrix` | `projection_matrix(aspect)` returns `glm.mat4` |
| `test_camera_no_gpu_dependency` | No `moderngl` import in `camera.py` |

> **Note:** Camera tests use PyGLM math only — no GPU required. Fully testable in CI.

### INT.4 Visualizer (`test/viz/test_renderer.py`)

| Test | What it verifies |
|------|-----------------|
| `test_visualizer_init` | `Visualizer(sim, vis_config)` creates renderer, camera, trails |
| `test_visualizer_headless_frame` | `headless_frame()` returns PIL Image |
| `test_visualizer_run_one_frame` | `run()` processes at least one frame without crash |
| `test_visualizer_paused_skips_step` | Paused visualizer calls `step()` zero times |

> **Note:** Visualizer tests that call `run()` need a mock or headless mode since `run()` is an infinite loop.

### INT.5 InputControl (`test/viz/test_input.py`)

| Test | What it verifies |
|------|-----------------|
| `test_input_quit_event` | `pygame.QUIT` event → `handle_events()` returns `False` |
| `test_input_escape_key` | `K_ESCAPE` → returns `False` |
| `test_input_space_pause` | `K_SPACE` toggles `paused` |
| `test_input_r_reset` | `K_r` sets `pending_reset = True` |
| `test_input_m_cycle_mode` | `K_m` cycles through 5 modes |
| `test_input_up_phi_p` | `K_UP` increases `config.phi_p` by 0.01 |
| `test_input_down_phi_p` | `K_DOWN` decreases `config.phi_p` by 0.01 |
| `test_input_phi_p_clamped` | `phi_p` never goes below 0 or above 1 |
| `test_input_right_phi_a` | `K_RIGHT` increases `config.phi_a` by 0.01 |
| `test_input_left_phi_a` | `K_LEFT` decreases `config.phi_a` by 0.01 |
| `test_input_brackets_sigma` | `K_RIGHTBRACKET`/`K_LEFTBRACKET` change `sigma` ±1 |
| `test_input_sigma_clamped` | `sigma` never goes below 1 or above 20 |
| `test_input_plus_add_birds` | `K_EQUALS` adds 10 birds |
| `test_input_minus_remove_birds` | `K_MINUS` removes 10 birds |
| `test_input_g_toggle_grid` | `K_g` toggles `show_grid` |
| `test_input_v_reset_camera` | `K_v` calls `camera.reset()` |
| `test_input_o_toggle_auto_rotate` | `K_o` toggles `camera.auto_rotate` |
| `test_input_t_predator` | `K_t` spawns/removes predator |
| `test_input_mouse_drag_orbit` | Mouse drag → `camera.rotate()` called |
| `test_input_scroll_zoom` | Scroll → `camera.zoom()` called |
| `test_input_never_imports_simulation` | `input_control.py` has no `import simulation` |

> **Note:** Input tests use `pygame.event.post()` to inject synthetic events, or mock the event queue.

### INT.6 Recorder (`test/capture/test_recorder.py`)

| Test | What it verifies |
|------|-----------------|
| `test_recorder_on_frame_captures` | `on_frame()` appends to `frames` and `metrics_history` |
| `test_recorder_capture_every_n` | `on_frame()` only captures when `frame % capture_every == 0` |
| `test_recorder_save_gif` | `save_gif()` creates a non-empty .gif file |
| `test_recorder_save_metrics_csv` | `save_metrics_csv()` creates a CSV with correct columns |
| `test_recorder_save_metrics_json` | `save_metrics_json()` creates valid JSON with metadata |
| `test_recorder_no_viz` | `with_viz=False` → no frames captured, only metrics |
| `test_recorder_empty_run` | Zero frames → `save_gif()` handles gracefully (no crash) |

---

## System Assembly Tests (arch.md Level 3 — Module Interfaces)

### SYS_ASSEMBLY.1 Module Import Rules (`test/test_imports.py` + `test/test_architecture.py`)

| Test | What it verifies |
|------|-----------------|
| `test_cli_no_args` | `python -m pymurmur` runs with built-in defaults (projection, N=150) |
| `test_cli_config_name` | `--config murmuration_spatial` loads `conf/murmuration_spatial.yaml` |
| `test_cli_config_path` | `--config /path/to/custom.yaml` loads from absolute path |
| `test_cli_config_not_found` | `--config nonexistent` raises `FileNotFoundError` |
| `test_cli_list_configs` | `--list-configs` prints all 7 config files |
| `test_cli_no_viz` | `--no-viz` runs without importing pygame/ModernGL |
| `test_cli_capture` | `--capture` creates `output/` directory with .gif and .csv |
| `test_cli_help` | `--help` prints usage and exits |

### SYS_ASSEMBLY.2 Config (`test/core/test_config.py`)

| Test | What it verifies |
|------|-----------------|
| `test_config_defaults` | `SimConfig()` has all documented default values |
| `test_config_from_file_flattens_nested` | `domain.width` in YAML → `config.width` |
| `test_config_from_file_mode_specific` | `projection.phi_p` in YAML → `config.phi_p` |
| `test_config_from_file_unknown_keys_ignored` | Extra YAML keys don't raise errors |
| `test_config_to_file_roundtrip` | `config.to_file()` → `from_file()` produces identical config |
| `test_config_live_mutable_list` | All mutable params are in documented list |
| `test_config_static_params_unchanged` | Changing static params at runtime has no effect on running sim |
| `test_config_seed_reproducibility` | Same seed → identical `random_positions()` output |
| `test_config_all_presets_loadable` | All 7 `conf/*.yaml` files parse without error |
| `test_config_performance_fields` | `use_numba`, `spatial_index`, `metrics_detail_level`, `metrics_interval`, `instance_buffer_chunk` have correct defaults |

### SYS_ASSEMBLY.3 Module Import Rules (`test/test_imports.py`)

| Test | What it verifies |
|------|-----------------|
| `test_physics_boid_no_flock_import` | `physics_boid` does not import `physics_flock` |
| `test_physics_boid_no_forces_import` | `physics_boid` does not import `forces` |
| `test_simulation_no_visualization_import` | `simulation` does not import `visualization`, `camera`, or `shaders` |
| `test_input_control_no_simulation_import` | `input_control` does not import `simulation` |
| `test_occlusion_no_project_imports` | `occlusion` imports only `numpy` and `types` |
| `test_steric_no_project_imports` | `steric` imports only `numpy` and `types` |
| `test_camera_no_moderngl_import` | `camera` does not import `moderngl` |
| `test_shaders_no_moderngl_import` | `shaders` does not import `moderngl` |
| `test_config_no_pygame_import` | `config` imports only stdlib + `PyYAML` + `numpy` |

---

## Cross-Cutting Tests (arch.md Level 4 — Implementation Details)

### 4.1 Performance Tests (`test/test_performance.py`)

| Test | What it verifies | Threshold |
|------|-----------------|:---------:|
| `test_bench_150_projection` | Projection mode at N=150 within budget | < 16 ms |
| `test_bench_200_spatial` | Spatial mode at N=200 within budget | < 16 ms |
| `test_bench_16k_field` | Field mode at N=16K within budget | < 16 ms |
| `test_bench_100_vicsek` | Vicsek mode at N=100 within budget | < 16 ms |
| `test_bench_200_influencer` | Influencer mode at N=200 within budget | < 16 ms |
| `test_bench_50k_spatial_numba` | Spatial mode at 50K with numba | < 33 ms |
| `test_bench_300k_spatial_numba` | Spatial mode at 300K with numba | < 50 ms |
| `test_bench_300k_field` | Field mode at 300K | < 10 ms |
| `test_bench_300k_influencer` | Influencer mode at 300K | < 5 ms |
| `test_memory_150` | Memory at N=150 | < 10 MB |
| `test_memory_16k` | Memory at N=16K | < 50 MB |
| `test_memory_300k` | Memory at N=300K | < 30 MB |
| `test_bit_reproducibility` | Same seed + same config → identical metrics after 1000 steps |

> **Note:** Performance tests use approximate thresholds. CI may need relaxed bounds. Use `pytest.mark.slow` and `pytest.mark.benchmark`.

### 4.2 Scaling Validation (`test/test_scaling.py`)

| Test | What it verifies |
|------|-----------------|
| `test_o1_scaling_field` | Field mode time ∝ N (linear fit R² > 0.95) |
| `test_nlogn_scaling_kdtree` | KDTree build time ∝ N log N |
| `test_topological_scaling_projection` | Projection mode time independent of flock density (topological σ) |
| `test_active_mask_scaling` | `add_boids()`/`remove_boids()` time is O(1) (no reallocation) |
| `test_buffer_growth_amortized` | Instance buffer growth amortized to O(1) per addition |
| `test_metrics_gating_scaling` | Expensive metrics trigger on `interval` boundary, not every frame |

### 4.3 Config File Validation (`test/test_config_files.py`)

| Test | What it verifies |
|------|-----------------|
| `test_all_7_configs_parse` | All `conf/*.yaml` files parse without error |
| `test_config_required_fields_present` | Each config has `domain`, `flock`, `mode`, `boundary_mode` |
| `test_config_performance_fields_present` | All 7 configs have `performance.use_numba` and `performance.spatial_index` |
| `test_config_metrics_fields_present` | All 7 configs have `metrics.detail_level` and `metrics.interval` |
| `test_config_modes_valid` | Config `mode` is one of the 5 valid values |
| `test_config_boundary_valid` | Config `boundary_mode` is one of the 4 valid values |
| `test_spatial_config_has_predator` | `murmuration_spatial.yaml` has `extensions.predator: true` |
| `test_field_config_has_wander` | `murmuration_field.yaml` has `extensions.wander: true` |
| `test_300k_config_numba_enabled` | `murmuration_300k.yaml` has `performance.use_numba: true` |
| `test_300k_config_kdtree` | `murmuration_300k.yaml` has `performance.spatial_index: kdtree` |

### 4.4 End-to-End Tests (`test/test_e2e.py`)

| Test | What it verifies |
|------|-----------------|
| `test_e2e_headless_default` | `python -m pymurmur --no-viz` runs 10 steps without error |
| `test_e2e_headless_capture` | `--no-viz --capture` produces output files |
| `test_e2e_all_modes_headless` | All 5 modes run 10 headless steps |
| `test_e2e_config_switch` | Switching config mid-simulation doesn't crash |
| `test_e2e_predator_spawn_mid_run` | Enabling predator mid-simulation adds threat forces |
| `test_e2e_add_remove_birds` | Adding then removing 50 birds leaves flock in valid state |
| `test_e2e_seed_reproducibility` | Two runs with same seed produce identical metrics CSV |
| `test_e2e_boundary_toroidal` | No bird leaves domain in toroidal mode after 1000 steps |
| `test_e2e_gif_output_valid` | Output .gif is viewable (correct magic bytes, non-zero size) |
| `test_e2e_csv_columns_match` | CSV column count matches `FlockMetrics` field count |

### 4.5 Reuse Validation (planned — `test/test_reuse.py`)

> ⚠️ Not yet implemented. Will verify code reused from the existing murmuration codebase produces identical results.

| Test | What it verifies |
|------|-----------------|
| `test_occlusion_matches_original` | New `occlusion.py` gives same δ̂, Θ as `occlusion_3d.py` for identical inputs |
| `test_steric_matches_original` | New `steric.py` gives same force as `steric_3d.py` |
| `test_camera_matches_original` | New `camera.py` gives same matrices as `camera_3d.py` |
| `test_ecology_matches_original` | New `Ecology` class gives same outputs as original `ecology.py` |

---

## Test Execution Strategy

```
Phase 0  (Level 0 unit tests)     →  pytest test/core/ test/physics/test_boid.py test/physics/test_occlusion.py test/physics/test_steric.py
Phase 1  (Level 0 spatial tests)  →  pytest test/physics/forces/ -k "force_primitive" -k "spatial_index"
Phase 2  (Level 1 assembly tests) →  pytest test/physics/test_flock.py test/physics/forces/ test/physics/extensions/
Phase 3  (Level 1 metrics)        →  pytest test/analysis/test_metrics.py test/analysis/test_presets.py
Phase 4  (Level 2 subsystems)     →  pytest test/simulation/ test/viz/test_camera.py test/capture/
Phase 5  (Level 2 renderer)       →  pytest test/viz/test_renderer.py  (GPU required)
Phase 6  (Level 2 input)          →  pytest test/viz/test_input.py    (pygame required)
Phase 7  (Level 3 system)         →  pytest test/test_subsystem_a.py test/core/test_config.py test/test_imports.py
Phase 8  (Level 3 config files)   →  pytest test/test_config_files.py
Phase 9  (Level 4 performance)    →  pytest test/test_performance.py -m slow
Phase 10 (Level 4 E2E)            →  pytest test/test_e2e.py
```

**CI pipeline:** Phases 0–4 and 7–8 run on every commit. Phase 5 (renderer), Phase 6 (input), and Phases 9–10 (performance/E2E) run on PR merge or nightly.

---

## Test Fixtures

Shared pytest fixtures for all test files:

```python
# conftest.py

@pytest.fixture
def default_config():
    """SimConfig with default projection mode parameters."""
    return SimConfig()

@pytest.fixture
def spatial_config():
    """SimConfig with spatial mode, N=200."""
    cfg = SimConfig()
    cfg.mode = "spatial"
    cfg.num_boids = 200
    return cfg

@pytest.fixture
def small_flock(default_config):
    """PhysicsFlock with N=50 for fast unit tests."""
    cfg = copy(default_config)
    cfg.num_boids = 50
    return PhysicsFlock(cfg)

@pytest.fixture
def two_bird_flock(default_config):
    """PhysicsFlock with exactly 2 birds for neighbor tests."""
    cfg = copy(default_config)
    cfg.num_boids = 2
    return PhysicsFlock(cfg)

@pytest.fixture
def known_positions():
    """Return (N, 3) array with known positions for deterministic tests."""
    return np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [-10, 0, 0]], dtype=np.float32)

@pytest.fixture
def known_velocities():
    """Return (N, 3) array with known velocities."""
    return np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)

@pytest.fixture
def neighbor_idx():
    """Pre-computed neighbor indices: bird 0 sees [1,2,3], bird 1 sees [0,2,3], etc."""
    N = 4
    idx = np.empty((N, N - 1), dtype=np.int32)
    for i in range(N):
        idx[i] = [j for j in range(N) if j != i]
    return idx

@pytest.fixture(scope="session")
def gpu_available():
    """Check if ModernGL can create a standalone context."""
    try:
        import moderngl
        moderngl.create_context(standalone=True, require=330)
        return True
    except Exception:
        return False

@pytest.fixture(scope="session")
def numba_available():
    """Check if numba is importable."""
    try:
        import numba
        return True
    except ImportError:
        return False
```

---

*Test plan derived from `arch.md` (both design views, §2). July 2026.*
