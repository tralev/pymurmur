# roadmap_deepseek.md — Self-Contained Implementation Roadmap

> **Where to start** (this document is ~3,000 lines — pick your entry point):
>
> | You want to… | Read this |
> |---|---|
> | **Understand the plan in 5 minutes** | [Summary Table](#summary-table) at the bottom → then the [Phase tree](#part-ii--phases) |
> | **Implement Phase 0 today** | [P0.1](#p01--golden-trajectory-harness-l3) through [P0.15](#p015--position-init-variants-l0), then run `pytest test/test_golden.py test/test_architecture.py` |
> | **Find which phases touch a module** | [Appendix E — Module→Phase Index](#appendix-e--module--phase-reverse-index) |
> | **Check acceptance criteria for a phase** | [Appendix F — Phase Boundary Checklists](#appendix-f--phase-boundary-checklists) |
> | **Look up a term** | [Appendix D — Glossary](#appendix-d--glossary) |
> | **Understand the target architecture** | [`arch.md`](arch.md) §2–§5 (both design views in one document) |
> | **See what changed from the old roadmap** | [`#appendix-g--oldnew-identifier-mapping`](#appendix-g--oldnew-identifier-mapping) |
> | **See the execution schedule** | [`.github/gantt-schedule.md`](.github/gantt-schedule.md) |
> | **Understand CI guard rails** | [`.github/workflows/guard-rails.yml`](.github/workflows/guard-rails.yml) |
>
> **Key principles:** Strictly 3D (z-up, `(N,3) float32`, no 2D arrays in `physics/`).
> Every phase ships **independently testable modules** — `pytest test/` green at each boundary.
> Composition is a DAG: `core(L0) → physics atoms(L0) → assemblies(L1) → subsystems(L2) → system(L3)`.
>



## Design Guide — How To Read This Roadmap

### Two Complementary Views (from arch.md §2)

**Macro→Micro (Top-Down):** The system decomposes into 7 functional subsystems:
A (Entry/Config), B (Simulation Engine), C (Viz/Input), D (Capture/Export),
E (Physics), F1 (Observables), F2 (Drivers). Every step below targets exactly
**one** subsystem; cross-subsystem steps are broken into separate sub-steps.

**Micro→Macro (Bottom-Up):** Components build from atoms upward. Each
step heading declares its level with a `[LX]` badge:

| Level | Badge | What lives here | Depends on |
|---|---|---|---|
| **L0** | `[L0]` | Pure atoms: math helpers, min_image, SDF primitives, force primitives (sep/align/coh/noise), occlusion, steric, numba kernels, integrate() variants | numpy/stdlib only |
| **L1** | `[L1]` | Assemblies: PhysicsFlock columns (rng/center/species), 7 ForceMode classes, ExtensionManager, MetricsCollector, rewards, presets, obstacle scenes | L0 atoms only |
| **L2** | `[L2]` | Subsystems: SimulationEngine, Visualizer, Renderer3D, Recorder, QualityGovernor, MurmurationEnv, EvoFlock | L1 assemblies |
| **L3** | `[L3]` | System: `__main__` CLI, `pymurmur` facade, scripts/ | L2 subsystems |

**Note:** Most L0 atoms live in `core/` (numpy/stdlib only). SDF primitives
are an exception — they live in `physics/obstacles.py` (new file, zero
project imports, pure numpy) because they conceptually belong to the physics
layer. The L0 rule is **import discipline**, not file location.

**The golden rule:** A component at level *n* depends only on levels < *n*.
No L1 assembly imports another L1 assembly. No L0 atom imports anything from
`pymurmur` except `core/`. This is enforced by `test/test_architecture.py`
which is extended at every phase boundary.

### Composer Convention

Every L0 atom lists its **composers** — the L1 assemblies that consume it.
No atom is shipped without at least one composer test proving it is actually
used. Dead atoms (no composers) are deleted. This is enforced by
`test/test_composers.py`.

### 3D Compliance

Every formula in this roadmap is written in **three-dimensional form** —
vectors are `(N,3) float32`, cross products use the z-up convention, and
all spatial operations (distance, normalisation, rotation) operate in ℝ³.

**Hard constraints (enforced by P14.3 + `test/test_architecture.py`):**
- No `(N,2)` or `(…, 2)`-shaped spatial arrays in `physics/` — AST scan fails CI.
- `SimConfig.validate()` enforces `domain.depth > 0`.
- Invariance tests use random SO(3) rotations, not z-only (2D-equivalent) rotations.
- Boundary modes (toroidal, sphere, sphere_soft) are computed in all three axes.
- Neighbour queries (cKDTree, SpatialHashGrid) use 3D boxsize `(W, H, D)`.
- The silhouette metric Θ′ (P9.4) is the **only** 2D projection in the codebase —
  it lives in `analysis/metrics.py` (F1 tier, not physics), exists purely for
  diagnostic comparison against the 3D voxel Θ′, and never feeds back into simulation.

**Visualization:** All rendering is 3D — ModernGL instanced meshes with Blinn-Phong
lighting, orbit camera with azimuth+elevation, dual-viewports, and orthographic presets
(top/side/perspective). No 2D sprite sheets, no flat projections. The GPU-free
matplotlib fallback (P8.9) renders dual-view 3D scatter → GIF.


### Per-Phase Structure

Each phase step follows this pattern:
- `[LX]` level badge + **File:** + **Tests:** header
- **Verbal idea** — what problem it solves
- **Math** — the exact formula in 3D form
- **Code sketch** — the target file and skeleton
- **Composers** — who consumes this (L0 atoms only)
- **Test** — concrete assertions with expected values
- **Migration** — what old code is deleted or refactored
- **Citation** — provenance in `sci/`

### Phase Acceptance

Each phase boundary has a **structured checklist** in [Appendix F](#appendix-f--phase-boundary-checklists)
— 229 CI-auditable `- [ ]` items across all 15 phases. Every item maps to a
specific test or AST assertion. A phase is accepted when every checkbox in its
section passes.


### DAG Enforcement Strategy

The architecture test (`test/test_architecture.py`) ships with `FORBIDDEN_EDGES`
from day one. `ALLOWED_EDGES` grows at each phase acceptance boundary — the
new edges introduced by that phase are added to the matrix. After P14, the
full matrix matches arch.md §5 exactly. No edge is ever removed from
`ALLOWED_EDGES` once added (modules don't lose dependencies).

### Conventions

Domain `[0,W)×[0,H)×[0,D)`, centre `C=(W/2,H/2,D/2)`, z-up.
All arrays `(N,3) float32`. `idx = np.where(flock.active)[0]` for active-bird
indexing. `flock.rng = np.random.default_rng(seed)` — the single randomness
source. `hash01(x) = fract(sin(x·12.9898)·43758.5453)` in `core/types.py`.

**File-location conventions.** Code under `pymurmur/<subsystem>/…`; tests
mirror the package: code in `pymurmur/physics/forces/field.py` → tests in
`test/physics/forces/test_field.py`; golden data in `test/data/`;
dependency-gated examples in `scripts/`; presets in `conf/`.

---

## Part I — Current State Audit

### What exists (July 2026)

| Layer | Inventory | Lines |
|---|---|---|
| Python modules | 31 files across 8 subpackages | 4,274 |
| Test suite | 42 files, 547 fast tests, 0 failures, 4 skipped | 9,039 |
| Slow tests | 26 (`@slow`) | — |
| GPU-gated | 37 (`@gpu`) | — |
| Config presets | 7 YAML files | — |
| Sci docs | 24 files (~12,000 lines) | — |

### What works

- 7 force modes (projection, spatial, field, vicsek, influencer, angle, marl)
- 4 boundary modes (toroidal, open, margin, sphere/soft)
- 4 behavioural extensions (predator FSM, ecology, wander, ripple)
- Instanced GPU rendering (ModernGL tetrahedra, Blinn-Phong, themes)
- Orbit camera, headless FBO→GIF/CSV/JSON capture, metrics collector
- Phase diagram sweep, density scaling sweep, EvoFlock GA (SSGA, 10 params, 4 objectives)
- Perf diagnostics (EMA timing) — adaptive ladder unwired

### Structural gaps (highest impact first)

1. **Flat SimConfig** — 90+ fields, cross-section key collisions silently corrupt `domain.width`
2. **No ForceMode protocol** — modes are stateless functions, no per-mode time/state
3. **`flock ↔ forces` import cycle** — composition is not a DAG
4. **Occlusion doesn't occlude** — all neighbours marked visible, Θ is linear sum not probabilistic union
5. **Determinism broken** — 8+ module-level `np.random.*` calls ignore `config.seed`
6. **No `flock.rng`, `flock.center`, `flock.prev_positions`, `flock.max_speed`** — flock state missing 4 columns
7. **Toroidal distance not min-image** — `use_toroidal_distance` field dead, flocks tear at seam
8. **Renderer VAO stale after buffer growth** — headless FBO has no depth attachment
9. **Field mode implements ~4 of 13 terms** — missing blob anchors, phase weights, leader/chaser, shell force, slot repulsion, flow, fold, buoyancy, drag

---

## Part II — Phases

```
P0 foundations ──► P1 correctness ──► P2 contracts
                                       ├─► P3 field/blob + threat
                                       ├─► P4 reynolds + ecology
                                       ├─► P5 angle mode
                                       ├─► P6 vicsek predator–prey
                                       └─► P7 influencer parity
P2 contracts ──► P8 rendering & capture
P2 contracts ──► P9 metrics & analysis
P2 + P4 ──► P10 UX & tooling
P2 + P4 ──► P11 evoflock
P2 + P9 ──► P12 MARL bridge
P3–P8 ──► P13 scaling & performance
P0–P13 ──► P14 guard rails
```

---

### Phase 0 — Foundations & Safety Net

**Ships:** golden-trajectory regression suite; PhysicsFlock with 5 new columns (rng, center, is_predator, prev_positions, last_accelerations); 10 math helpers in core/types.py; 5 SDF primitives in physics/obstacles.py; 5 position-init strategies; capability probing

**Subsystems:** A, B, E, F1 | **Levels:** L0 → L3 | **Est. effort:** 3 days

**Depends on:** nothing. **Produces:** golden-trajectory regression suite,
flock state contract (5 new columns), deterministic RNG, integration variants,
math helpers, SDF primitives, capability probing, safety rails.

#### P0.1 — Golden trajectory harness `[L3]`

**File:** `test/regenerate_golden.py`, `test/test_golden.py`. **Data:** `test/data/golden_<mode>.npz`.

**Verbal idea.** Before any refactor, pin precise trajectory snapshots per mode
so accidental physics regressions are caught immediately. Run seeded 15-bird ×
30-frame sims, save positions+velocities to `.npz`, and assert future runs match
within `atol=1e-3`. Deliberate physics changes re-pin in the same commit.

**Code sketch** (`test/regenerate_golden.py`):
```python
import numpy as np
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

def generate_golden(mode: str, seed: int = 77, frames: int = 30, n: int = 15):
    cfg = SimConfig(); cfg.mode = mode; cfg.flock.num_boids = n; cfg.flock.seed = seed
    cfg.use_numba = False
    engine = SimulationEngine(cfg)
    positions, velocities = [], []
    for _ in range(frames):
        engine.step(1/60)
        positions.append(engine.flock.positions.copy())
        velocities.append(engine.flock.velocities.copy())
    np.savez(f"test/data/golden_{mode}.npz",
             pos=np.stack(positions), vel=np.stack(velocities))
```

**Test** (`test/test_golden.py`):
```python
@pytest.mark.parametrize("mode", ["projection", "spatial"])
def test_matches_golden(mode):
    cfg = SimConfig(); cfg.mode = mode; cfg.flock.num_boids = 15; cfg.flock.seed = 77
    cfg.use_numba = False
    engine = SimulationEngine(cfg)
    for _ in range(30): engine.step(1/60)
    golden = np.load(f"test/data/golden_{mode}.npz")
    np.testing.assert_allclose(engine.flock.positions, golden["pos"][-1], atol=1e-3)
```

**Citation:** `todo_claude.md` T14.

#### P0.2 — Architecture test skeleton `[L3]`

**File:** `test/test_architecture.py`.

**Code sketch:**
```python
FORBIDDEN_EDGES = [
    ("pymurmur.physics.forces", "pymurmur.physics.flock"),
    ("pymurmur.viz", "pymurmur.simulation"),
]
ALLOWED_EDGES = [
    ("pymurmur.core", ()),
    ("pymurmur.physics.boid", ("pymurmur.core",)),
]
# AST-walk every .py file; assert every import edge ∈ ALLOWED_EDGES
```

**Citation:** `roadmap.md` D0.2.

#### P0.3 — Physics invariant fuzz `[L0]`

**File:** `test/physics/test_boid.py`.

**Math.** For `integrate(..., "toroidal")`: after step, `0 ≤ pos < (W,H,D)` elementwise,
`|v| ≤ v0 + ε` for band mode, `|v| ≡ v0` for fixed mode. No NaN. Inactive rows bit-identical.

**Test:**
```python
def test_speed_band_respected():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vel = rng.uniform(-2*v0, 2*v0, (50,3)).astype(np.float32)
        pos = rng.uniform(0, W, (50,3)).astype(np.float32)
        integrate(pos, vel, np.zeros((50,3)), np.ones(50,bool), W, H, D, v0, "toroidal")
        assert np.all(np.linalg.norm(vel, axis=1) <= v0 + 1e-4)
        assert np.all((0 <= pos) & (pos < [W,H,D]))
```

**Citation:** `todo_claude.md` T13.

#### P0.4 — Single seeded RNG `[L1]`

**File:** `physics/flock.py`.

**Verbal idea.** Add `flock.rng = np.random.default_rng(cfg.flock.seed)` on `PhysicsFlock`.
Every stochastic site draws from it. Module-level `np.random.*` calls are deleted.
Same seed → bit-identical after 100 steps.

**Code sketch** (`physics/flock.py`):
```python
class PhysicsFlock:
    def __init__(self, config):
        self.rng = np.random.default_rng(config.flock.seed if config.flock.seed else 0)
```

**Test** (`test/physics/test_flock.py`):
```python
def test_same_seed_bit_identical():
    cfg = SimConfig(); cfg.flock.seed = 42; cfg.flock.num_boids = 30
    e1 = SimulationEngine(cfg); e2 = SimulationEngine(cfg)
    for _ in range(100): e1.step(1/60); e2.step(1/60)
    np.testing.assert_array_equal(e1.flock.positions, e2.flock.positions)
```

**Citation:** `git0` F1.

#### P0.5 — Smoothed swarm centre `[L1]`

**File:** `physics/flock.py`.

**Math.**
```
centroid = (1/N_active) · Σ_{i∈active} p_i
center ← center + 0.5 · (centroid − center)
```

**Code sketch:**
```python
centroid = self.positions[self.active].mean(axis=0)
if self._center is None: self._center = centroid.copy()
else: self._center += 0.5 * (centroid - self._center)
```

**Test:**
```python
def test_center_smoothed():
    # Teleport flock → center lags; converges within 20 frames
```

**Citation:** `git6` R1.

#### P0.6 — Species column `[L1]`

**File:** `physics/flock.py`.

**Code sketch:**
```python
self.is_predator: np.ndarray = np.zeros(N, dtype=bool)  # (N,) bool
```
Carried through `_extend()`, `add_boids(is_predator=False)`, `remove_boids()`.

**Test:**
```python
def test_species_carried_through_lifecycle():
    flock.add_boids(5, is_predator=True)
    assert flock.is_predator[-5:].all()
    flock.remove_boids(3)
    assert flock.is_predator.sum() == 2
```

**Citation:** `git0` F6.

#### P0.7 — Previous positions + acceleration stash `[L1]`

**File:** `physics/flock.py`.

**Code sketch:**
```python
self.prev_positions: np.ndarray = np.zeros((N,3), dtype=np.float32)
self.last_accelerations: np.ndarray = np.zeros((N,3), dtype=np.float32)
# Before integrate():
self.last_accelerations[:] = self.accelerations
self.prev_positions[:] = self.positions.copy()
```

**Composers:** MSD unwrap (P9.2), ring trails (P8.3), render interpolation (P8.12), physical metrics (P4.4/P9.8).

**Citation:** `git0` F8.

#### P0.8 — Per-bird max_speed array `[L1]`

**File:** `physics/flock.py`.

**Code sketch:**
```python
self.max_speed: np.ndarray | None = None  # None → scalar v0
```
In `integrate()`: `cap = max_speed if max_speed is not None else v0`.

**Citation:** `git6` R8.

#### P0.9 — Integration variants `[L0]`

**File:** `physics/boid.py::integrate`.

**Math.**
```
v += a·dt
speed_mode "band":    clamp |v| to [0.3·v0, v0]  (per-bird max_speed)
speed_mode "fixed":   v = v̂ · v0                   (exact renormalisation; 0-safe)
speed_mode "ceiling": limit |v| ≤ v0 only
speed_mode "none":    no clamp
if inertia > 0:       v = lerp(v_raw, v_clamped, 1−inertia)
if move:              p += v·dt
boundary_enforce(p, v, mode, W, H, D)
```

**Test** (`test/physics/test_boid.py`):
```python
@pytest.mark.parametrize("speed_mode,expect", [("fixed", v0), ("band", v0*0.3)])
def test_speed_contract(speed_mode, expect): ...
```


**Zero-speed fallback.** When |v| falls below `0.3·v0` (or exactly zero),
use the deterministic fallback `(minSpeed, 0, 0)` instead of a random direction.
This prevents NaN propagation in `normalize()` and keeps replay bit-identical
when a bird stalls. Config: `flock.speed_min_factor: 0.3`.

**Citation:** `git0` F3.

#### P0.10 — Safety rails: dt clamp + NaN guard `[L0]`

**File:** `physics/boid.py::integrate`.

**Math.** `dt = clip(dt, 0, 0.05)`. `if ~isfinite(p[i]): p[i] = center; v[i] = 0`.

**Citation:** `sci.md` §22.

#### P0.11 — Capability probing `[L3]`

**File:** `pymurmur/__main__.py`.

**Code sketch:**
```python
def probe_capabilities():
    caps = {}
    try: import moderngl; caps["moderngl"] = moderngl.__version__
    except: caps["moderngl"] = None
    try: import numba; caps["numba"] = numba.__version__
    except: caps["numba"] = None
    return caps
```

**Citation:** `sci/todo_claude_sci2.md`.

#### P0.12 — Math helpers in core/types.py `[L0]`

**File:** `core/types.py`. **Tests:** `test/core/test_types.py`.

**Verbal idea.** The single canonical location for all vector math. Every L0 helper
is unit-tested against its documented formula before any assembly consumes it.

**Functions:** `safe_normalize(v)` · `limit3(v, max_mag)` · `lerp(a, b, t)` ·
`rotate_about(v, k, angle)` (Rodrigues) · `smoothstep(e0, e1, x)` ·
`hash01(x)` (fract(sin·12.9898)·43758.5453) · `min_image(Δ, box)` ·
`min_image_distance(Δ, box)` (per-axis toroidal distance) ·
`fibonacci_sphere(n)` (golden-angle spiral on S²) ·
`seed_noise3(seeds, t)` (deterministic sinusoidal noise, [-0.18, 0.18]/axis).

**Test** (`test/core/test_types.py`):
```python
def test_rotate_about_exact():
    v = np.array([1.0,0,0]); k = np.array([0,0,1.0])
    assert np.allclose(rotate_about(v, k, np.pi/2), [0,1,0])
def test_min_image_wraps():
    d = np.array([[90.0, 0.0, 0.0]]); box = np.array([100.0, 100.0, 100.0])
    assert min_image(d, box)[0,0] == -10.0
def test_fibonacci_sphere_count():
    assert fibonacci_sphere(256).shape == (256, 3)
def test_seed_noise3_range():
    n = seed_noise3(np.arange(1000), 0.5)
    assert n.min() >= -0.18 and n.max() <= 0.18
```

**Citation:** `todo_claude.md` §3 (Rodrigues), `sci/todo_claude_sci2.md` §5.3, `sci/todo_claude_git6.md`.

#### P0.13 — H₂ disconnected → inf fix `[L0]`

**File:** `analysis/metrics.py::compute_h2`.

**Verbal idea.** When the k-NN graph is disconnected, return `(inf, inf)` instead
of `(0.0, 0.0)`. Skip non-finite values in `find_optimal_m`.

**Test:**
```python
def test_h2_inf_when_disconnected():
    pts = np.array([[0,0,0],[1,0,0],[1000,0,0],[1001,0,0]], np.float32)
    _, h2 = compute_h2(pts, m=1)
    assert math.isinf(h2)
```

**Citation:** `todo_claude.md` §10.

#### P0.14 — SDF primitives `[L0]`

**File:** `physics/obstacles.py` (new). **Tests:** `test/physics/test_obstacles.py`.

**Math.**
```
Sphere:   ‖p−c‖ − r
Box:      max(|p−c| − b)                [b = half-extents]
Cylinder: max(√(dx²+dz²) − r, |dy−c_y| − h/2)   [z-up]
Union:    min(a, b)
Subtract: max(a, −b)
Collision: sign(SDF(p_old)) ≠ sign(SDF(p_new))
Kinematic correction: p ← p − SDF(p)·∇SDF/‖∇SDF‖
```

**Composers:** ObstacleScene (P11.5) — CSG tree builder for EvoFlock obstacle courses.

**Citation:** `new10_sci.md` §5.


#### P0.15 — Position init variants `[L0]`

**File:** `physics/boid.py`. **Tests:** `test/physics/test_boid.py`.

**Verbal idea.** Beyond the 4 velocity-init variants (P4.9), add position
initialisation strategies that produce better-conditioned starting states.

**Math.**
```
"box" (legacy):     p = U(0,1)³ · (W,H,D)                    [uniform box]
"sphere_shell":     p = C + R · dir̂_uniform                   [exact spherical shell]
                    R = 0.4 · min(W,H,D)                     [prevents steric pairs]
"gaussian":         p = C + N(0, σ²I₃)                       [compact cloud]
                    σ = N^(1/3) · separation                  [P7.4 density-scaled form]
"grid":             p = deterministic 3D grid layout          [even spacing, zero overlaps]
                    spacing = (W·H·D/N)^(1/3)                [cubic lattice fill]
"blob":             p = 5-centre ∛-uniform shell + jitter     [P3.10 field mode init]
```
Selector: `cfg.flock.position_init: str = "box"`.

**Composers:** All ForceMode classes (init called by `PhysicsFlock.__init__` and `add_boids`).

**Test:**
```python
@pytest.mark.parametrize("init", ["box", "sphere_shell", "gaussian", "grid"])
def test_init_no_overlaps(init):
    cfg.flock.position_init = init; cfg.flock.num_boids = 100
    flock = PhysicsFlock(cfg)
    # No pair closer than 0.5·boid_size after init
def test_grid_spacing_even():
    cfg.flock.position_init = "grid"
    flock = PhysicsFlock(cfg)
    min_sep = cdist(flock.positions, flock.positions).min(axis=1).min()
    assert min_sep > 0.8 * expected_spacing
```

**Citation:** `sci/todo_claude3.md` position init, `sci/todo_claude_sci2.md` §12,
`sci/todo_claude_git6.md` R12.

---

**Phase 0 acceptance:** Golden suite green across all modes. Same-seed → bit-identical
after 100 steps per mode. `flock.center` lags teleported centroid. `flock.is_predator`
survives add/remove. All 10 math helpers round-trip correct (exact angle for Rodrigues,
correct wrap for min_image, boundary endpoints for smoothstep, range for hash01).
`compute_h2` returns `inf` for disconnected graph. Five SDF primitives round-trip correct.

**Architecture test:** `ALLOWED_EDGES` contains core + physics/boid. No module-level `np.random.*`.

**Migration (P0):** No code deleted. Golden `.npz` files are new fixtures. Module-level
`np.random.*` calls replaced with `flock.rng.*`. PhysicsFlock gains 5 new columns.
`core/types.py` gains 10 math helpers. `physics/obstacles.py` created (5 SDF functions).
Five position-init strategies (`box`, `sphere_shell`, `gaussian`, `grid`, `blob`) shipped;
`cfg.flock.position_init` selector functional.

---

### Phase 1 — Scientific Correctness

**Ships:** true occlusion culling with visibility test; probabilistic-union Θ; boundary-weighted δ̂; steric clamp; corrected force primitives (sep 1/d², coh bounded, noise ×scale); vicsek memory term + tangent-plane noise; thickness ratio fix; Θ N/A guard

**Subsystems:** E, F1 | **Levels:** L0 | **Est. effort:** 3 days

**Depends on:** P0. **Produces:** true occlusion culling, probabilistic-union Θ,
boundary-weighted δ̂, steric clamp, force-kernel fixes (sep 1/d², coh bounded, noise ×scale),
vicsek memory term + tangent-plane noise, thickness fix, Θ N/A in non-projection modes.

#### P1.1 — Occlusion culling: visibility test `[L0]`

**File:** `physics/occlusion.py::spherical_cap_occlusion`. **Tests:** `test/physics/test_occlusion.py`.

**Math.** For neighbour j (closest-first order):
```
d̂_j = (p_j − p_obs) / d_j
α_j = arcsin(min(b_eff / d_j, 1))
j visible iff: ¬blind(j) AND ∀ k<n_vis: d̂_j·d̂_k < cos α_k
```
Candidates capped at nearest 64 (`config.projection.max_occlusion_neighbors`).

**Test:**
```python
def test_occlusion_culls_behind_nearer_cap():
    # 3 collinear birds at (10,0,0),(30,0,0),(80,0,0) → visible=[0] only
    nbr_pos = np.array([[10,0,0],[30,0,0],[80,0,0]], np.float32)
    _, visible, _ = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel, boid_size=9.0)
    assert list(visible) == [0]
def test_occlusion_separated_all_visible():
    # 3 birds on orthogonal axes → all 3 visible
    nbr_pos = np.array([[60,0,0],[0,60,0],[0,0,60]], np.float32)
    _, visible, _ = spherical_cap_occlusion(obs_pos, obs_vel, nbr_pos, nbr_vel)
    assert len(visible) == 3
```

**Citation:** `todo_claude.md` §1, T1.

#### P1.2 — Θ as probabilistic union `[L0]`

**Math.**
```
Ω_j = 2π (1 − cos α_j)          [solid angle, steradians]
Θ = 1 − ∏_{j∈visible} (1 − Ω_j / 4π)
```

**Test:**
```python
def test_theta_sub_additive():
    # Θ₁ < Θ₁₂ < Θ₁+Θ₂, Θ ∈ [0,1]
    assert th1 < th2 < 2*th1 and 0.0 <= th2 <= 1.0
```

**Citation:** `todo_claude.md` §2, T2.

#### P1.3 — δ̂ boundary-length weighted `[L0]`

**Math.**
```
δ̂ = (Σ_{j∈visible} sin α_j · d̂_j) / (Σ_{j∈visible} sin α_j)
```
`|δ̂| ∈ [0,1]`: →1 at edge, →0 deep inside. Do NOT normalise to unit magnitude.

**Test:**
```python
def test_delta_edge_vs_surrounded():
    # Octahedral surround (6 neighbours on ±axes) → |δ̂| < 1e-2
    # Single neighbour → |δ̂| ≈ 1.0
```

**Citation:** `todo_claude.md` §3, T3.

#### P1.4 — Exact α = asin(min(b/d, 1)) `[L0]`

Replace all `b_eff/d` (small-angle approx) with `math.asin(min(b_eff / d, 1.0))`.

**Citation:** `todo_claude.md` §1.

#### P1.5 — Candidate cutoff at 64 `[L0]`

`order = order[:64]` after argsort. Config: `projection.max_occlusion_neighbors = 64`.

**Citation:** `todo_claude.md` §4.

#### P1.6 — Steric clamp to max_force `[L0]`

**File:** `physics/steric.py::steric_force`.

**Math.** After summing `strength · Σ r̂/d²`, if `‖F‖ > max_force`: `F ← F·max_force/‖F‖`.

**Test:**
```python
def test_steric_clamped():
    f = steric_force(np.zeros(3), np.array([[0.01,0,0]]), strength=0.6, max_force=0.15)
    assert np.linalg.norm(f) == pytest.approx(0.15, abs=1e-6)
```

**Citation:** `todo_claude.md` §14, T6.

#### P1.7 — Force kernel fixes `[L0]`

**File:** `physics/forces/_base.py`. **Tests:** `test/physics/forces/test_kernels.py`.

**Math.**
```
F_sep = −α · Σ r̂_{ij} / d_{ij}²    [unit/d² — currently 1/d]
F_coh =  γ · normalize(p̄ − p_i)     [bounded — currently unbounded]
F_noise = δ · û_rand                 [scale applied — currently discarded]
```

**Test:**
```python
def test_separation_inverse_square():
    # One neighbour at d=2 → |F| = 1/4
def test_cohesion_bounded():
    # Large offset → |F| = weight (not unbounded)
def test_noise_scale_live():
    # scale=0 → zero force; scale=0.5 → mean|F|≈0.5
```

**Citation:** `git0` W3.1, `sci/todo_claude_sci3.md` §1.

#### P1.8 — Vicsek update: memory term + tangent-plane noise `[L0]`

**File:** `physics/forces/vicsek.py`. **Tests:** `test/physics/forces/test_vicsek_core.py`.

**Math.**
```
û_noisy = normalize(û_old + √(2·D·Δt) · n_⊥)
n_⊥ = g − (g·û_old)·û_old,  g ~ N(0, I₃)    [tangent-plane projection]
û_new = normalize(η · û_target + (1−η) · û_noisy)
```
D lives — currently normalised away. `Δt = cfg.vicsek.time_step`.

**Test:**
```python
def test_vicsek_memory():
    # Lone bird, D=0, no neighbours → autocorr(û_t, û_{t+1}) > 0.999
def test_vicsek_D_live():
    # D=0.01 → high autocorr; D=4 → < 0.5 at lag 1
def test_noise_in_tangent_plane():
    # |n_⊥·û| < 1e-6
```

**Citation:** `git2` R1, `sci/todo_claude_sci8.md` §1.

#### P1.9 — Thickness ratio fix `[L0]`

**File:** `analysis/metrics.py::compute_shape`.

**Math.**
```
C = cov(positions); λ₁ ≥ λ₂ ≥ λ₃
thickness = √(λ₃ / λ₁) ∈ (0,1]
```
Currently returns `√(λ₂/λ₃) ≥ 1` — wrong formula.

**Test:**
```python
def test_thickness_thin_line():
    line = np.array([[x*10,0,0] for x in range(20)], np.float32)
    _, thickness = compute_shape(line)
    assert thickness < 0.2 and thickness > 0
```

**Citation:** `todo_claude.md` E13.

#### P1.10 — Θ reports N/A in non-projection modes `[L0]`

**File:** `analysis/metrics.py`. **Tests:** `test/analysis/test_metrics.py`.

**Verbal idea.** In modes other than projection, report `last_theta = NaN` so analysis
pipelines can filter correctly, instead of displaying stale zeros from initialisation.

**Also fix:** Move lazy `from ..steric import steric_force` out of the per-bird loop
in `projection.py` — import at module top (L0 atom, no cycle risk).

**Tests:**
```python
def test_theta_nan_in_spatial():
    cfg.mode = "spatial"; ...; assert math.isnan(metrics.last_theta)
```

**Citation:** `todo_claude3.md` §4, `todo_claude2.md` §2.

---

**Phase 1 acceptance:** Collinear birds → only nearest visible. Θ sub-additive, ∈ [0,1].
`|δ̂| < 1e-2` surrounded, `≈1` at edge. Steric at d=0.01 returns exactly `max_force`.
Separation 1/d² unit-vector. Cohesion bounded. Noise scale live. D axis of phase diagram
active (D=0.01 vs D=4 → different α). Thickness ∈ (0,1]. Θ is NaN in non-projection modes.
**Re-pin projection + spatial + vicsek goldens.**

**Architecture test:** `ALLOWED_EDGES` extended: `physics/occlusion → core`,
`physics/steric → core`, `physics/forces/_base → core`, `physics/forces/vicsek → core`,
`analysis/metrics → core + physics/flock(read)`.

**Migration:** `physics/occlusion.py` rewritten (culling logic). `physics/steric.py` adds clamp.
`physics/forces/_base.py` kernel formulas corrected. `physics/forces/vicsek.py` update rewritten.
`analysis/metrics.py` thickness formula + Θ N/A logic corrected.

---

### Phase 2 — Contracts & Protocols

**Ships:** nested SimConfig (17 sub-config dataclasses); ForceMode ABC + MODE_REGISTRY (7 modes); ForceTerm protocol + composeForces; SpatialIndex Protocol (2 implementations); StepContext + ExtensionManager; KDTreeIndex global indices + ghost-cell replication; InstanceSchema + VAO discipline; flock→forces import cycle broken (orchestration moves to engine)

**Subsystems:** A, B, C, E | **Levels:** L1 → L2 | **Est. effort:** 5.5 days

**Depends on:** P1. **Produces:** nested SimConfig (per-subsystem dataclasses),
ForceMode protocol + registry (7 modes as classes), SpatialIndex protocol (2 impls),
KDTreeIndex global indices + boxsize toroidal, ghost-cell replication, StepContext +
Extension widening, engine orchestration (DAG), InstanceSchema + VAO discipline,
PyGLM matrix upload, holey-mask contract suite.

#### P2.1 — Nested SimConfig `[L1]`

**File:** `core/config.py`, `conf/*.yaml`. **Tests:** `test/core/test_config.py`.

**Verbal idea.** Replace the flat 90-field dataclass with per-subsystem dataclasses.
YAML section names match SimConfig field names 1:1. The cross-section collision
(`capture.width` overwriting `domain.width`) becomes structurally impossible.
Unknown keys warn. `validate()` clamps ranges.

**Config fields:**
```python
@dataclass
class DomainConfig:
    width: float = 1000.0; height: float = 700.0; depth: float = 400.0

@dataclass
class BoundaryConfig:
    mode: str = "toroidal"    # toroidal | open | margin | sphere | sphere_soft
    sphere_radius: float = 300.0; avoidance_factor: float = 0.05
    margin: float = 42.0; use_toroidal_distance: bool = True

@dataclass
class FlockConfig:
    num_boids: int = 150; boid_size: float = 9.0; v0: float = 4.0
    max_force: float = 0.15; visual_range: float = 70.0
    seed: int | None = None; velocity_init: str = "fixed"
    speed_min_factor: float = 0.3; n_predators: int = 0
    position_init: str = "box"  # box | sphere_shell | gaussian | grid | blob

@dataclass
class ProjectionConfig:
    phi_p: float = 0.04; phi_a: float = 0.80; sigma: int = 6
    refinements: bool = False; steric: float = 0.0; blind_deg: float = 60.0
    anisotropy: float = 1.0; max_visibility: float = 200.0
    max_occlusion_neighbors: int = 64

@dataclass
class SpatialConfig:
    separation_weight: float = 1.5; alignment_weight: float = 0.65
    cohesion_weight: float = 0.75; noise_scale: float = 0.02
    acceleration_scale: float = 1.0; separation_distance: float = 20.0
    neighbor_filter: str = "knn"; influence_count: int = 7
    alignment_radius_ratio: float = 0.75; noise_mode: str = "force"
    speed_mode: str = "band"; parameter_jitter: bool = False
    jitter_separation: float = 0.5; jitter_cohesion: float = 0.1
    jitter_alignment: float = 0.005
    predator_escape_factor: float = 10_000_000.0
    predator_speed_boost: float = 1.8; predator_perception_boost: float = 1.5
    predator_accel_boost: float = 1.4

@dataclass
class FieldConfig:
    unit_scale: float = 0.0; chase_strength: float = 0.72
    shell_influence: float = 1.0; target_pull: float = 0.35
    drift_pull: float = 0.55; tangent_pull: float = 1.0
    flow_pull: float = 1.0; wave_gain: float = 0.5
    inertia: float = 0.8; separation: float = 0.85
    alignment: float = 0.65; cohesion: float = 1.85; flow: float = 0.18

@dataclass
class VicsekConfig:
    couplage: float = 0.5; diffusion: float = 0.1; time_step: float = 0.1
    velocity: float = 1.0; radius_influence: float = 5.0
    radius_avoid: float = 1.0; radius_predators: float = 10.0
    weight_afraid: float = 3.0; predator_noise_ratio: float = 0.2
    detect_ratio: float = 1.5; velocity_predator: float = 2.0

@dataclass
class InfluencerConfig:
    rank_exponent: float = 1.8; substeps: int = 5; scale: float = 1.0
    influence_mode: str = "rank"; near_dist_sq: float = 100.0
    init_separation: float = 0.5

@dataclass
class AngleConfig:
    turn_rate: float = 120.0; max_turn_rate: float = 360.0
    turn_threshold: float = 0.8; jitter_deg: float = 4.0
    margin: float = 50.0; speed_mode: str = "linear"
    base_speed: float = 4.0; neighbors: int = 7
    sep_radius_bodies: float = 1.0; align_radius_bodies: float = 5.0
    range_radius_bodies: float = 12.0

@dataclass
class MarlConfig:
    action_scale: float = 0.1; velocity_cap: float = 1.0
    rule_weight: float = 0.01; separation_radius: float = 0.2
    episode_steps: int = 500

@dataclass
class ThreatConfig:
    mode: str = "off"; radius: float = 0.15; strength: float = 0.5
    momentum: float = 0.3; acceleration: float = 0.8
    split_gain: float = 0.5; vacuole_strength: float = 0.0
    blackening_gain: float = 0.6

@dataclass
class EcologyConfig:
    roost: float = 1.0; critical_mass: int = 500
    roosting_enabled: bool = True; seasonal_size: bool = False
    peak_size: int = 500; wander_speed: float = 0.2; ripple_enabled: bool = True

@dataclass
class MetricsConfig:
    detail_level: int = 1; interval: int = 10
    bird_mass_kg: float = 0.075; cruise_speed_ms: float = 8.94
    acc_peak_ms2: float = 40.0

@dataclass
class VizConfig:
    fps: int = 60; window_width: int = 1200; window_height: int = 800
    theme: str = "dark"; trails: str = "off"; trail_length: int = 30
    point_sprites: bool = False; per_bird_color: bool = False
    dual_view: bool = False; background: str = "flat"
    show_grid: bool = False; auto_rotate: bool = False; hud: bool = True

@dataclass
class CaptureConfig:
    width: int = 800; height: int = 600; frames: int = 120
    every: int = 1; fps: int = 30; output: str = "capture.gif"
    metrics_csv: bool = False; metrics_json: bool = False
    with_viz: bool = False; sweep: bool = True; prewarm: int = 60

@dataclass
class PerfConfig:
    use_numba: bool = True; fastmath: bool = False; num_threads: int = -1
    spatial_index: str = "kd_tree"; instance_buffer_chunk: int = 4096
    adaptive_quality: bool = True; target_fps: int = 60

@dataclass
class SimConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    flock: FlockConfig = field(default_factory=FlockConfig)
    mode: str = "projection"
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    field: FieldConfig = field(default_factory=FieldConfig)
    vicsek: VicsekConfig = field(default_factory=VicsekConfig)
    influencer: InfluencerConfig = field(default_factory=InfluencerConfig)
    angle: AngleConfig = field(default_factory=AngleConfig)
    marl: MarlConfig = field(default_factory=MarlConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    ecology: EcologyConfig = field(default_factory=EcologyConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
```

**Loader** (`from_file`): YAML section name = SimConfig field name; build each sub-config
with `{k:v for k,v in section.items() if k in sub_fields}`; `warnings.warn` for unknown keys.
`validate()` clamps: σ ≥ 1, v0 > 0, depth > 0 (strictly-3D).

**Migration:** Property shims → dotted access → shim retirement. Rewrite `conf/*.yaml`.

**Tests** (`test/core/test_config.py`):
```python
def test_round_trip():
    cfg = SimConfig(); cfg.to_file(tmp); cfg2 = SimConfig.from_file(tmp)
    assert cfg == cfg2

@pytest.mark.parametrize("preset", glob("conf/*.yaml"))
def test_preset_domain_survival(preset):
    cfg = SimConfig.from_file(preset)
    assert cfg.domain.width > 0  # not overwritten by capture.width

def test_unknown_key_warns():
    with pytest.warns(UserWarning): SimConfig.from_file(yaml_with_unknown)

def test_validate_depth_positive():
    with pytest.raises(AssertionError):
        SimConfig(domain=DomainConfig(depth=0)).validate()
```

**Citation:** `roadmap.md` D1, `git0` F0, `todo_claude1.md` §1.

#### P2.2 — ForceMode protocol + registry `[L1]`

**File:** `physics/forces/_mode.py` (new), `simulation/engine.py`. **Tests:** `test/physics/forces/test_mode_contract.py`.

**Verbal idea.** Modes become stateful classes behind a single ABC with declared flags
(`needs_index`, `speed_mode`, `owns_positions`). `@register` → `MODE_REGISTRY`.
Orchestration moves to `SimulationEngine`. `PhysicsFlock` no longer imports forces —
the `flock ↔ forces` cycle is dead.

**Code sketch** (`physics/forces/_mode.py`):
```python
class ForceMode(ABC):
    name: ClassVar[str]
    needs_index: ClassVar[bool] = False
    speed_mode: ClassVar[str] = "band"     # band | fixed | ceiling | none
    owns_positions: ClassVar[bool] = False # True → integrate(move=False)
    def reset(self, flock, config): ...
    @abstractmethod
    def step(self, flock, config, dt) -> None: ...

MODE_REGISTRY: dict[str, type[ForceMode]] = {}
def register(cls): MODE_REGISTRY[cls.name] = cls; return cls

@register class ProjectionMode(ForceMode): name="projection"; needs_index=True
@register class SpatialMode(ForceMode): name="spatial"; needs_index=True; speed_mode="band"
@register class FieldMode(ForceMode): name="field"
@register class VicsekMode(ForceMode): name="vicsek"; needs_index=True; speed_mode="fixed"
@register class InfluencerMode(ForceMode): name="influencer"; owns_positions=True; speed_mode="fixed"
@register class AngleMode(ForceMode): name="angle"; speed_mode="fixed"
@register class MarlMode(ForceMode): name="marl"; needs_index=True; speed_mode="none"
```

**Engine orchestration** (`simulation/engine.py::step`):
```python
def step(self, dt, control=None):
    self._drain_commands()
    ctx = StepContext(frame=self.frame, dt=dt, rng=self.flock.rng,
                      center=self.flock.center, config=self.config)
    self.extensions.pre_step(self.flock, ctx)
    if control is not None: self._apply_control(control)
    if self._mode.needs_index: self.flock.rebuild_index(self.config)
    self._mode.step(self.flock, self.config, dt)
    self.flock.stash_accelerations()
    integrate(self.flock, self.config, dt,
              speed_mode=self._mode.speed_mode,
              move=not self._mode.owns_positions)
    self.flock.update_center()
    self.metrics.collect(self.flock, self.frame, ctx)
    self.frame += 1
```

**Tests:**
```python
@pytest.mark.parametrize("mode_cls", MODE_REGISTRY.values())
def test_registered_and_instantiable(mode_cls): ...
def test_step_respects_active_mask(mode_cls, holey_flock, cfg): ...
def test_determinism(mode_cls, cfg): ...
def test_import_cycle_dead():
    # AST check: physics/flock.py does not import anything from physics/forces/
```

**Citation:** `roadmap.md` D2, `git0` F2, `todo_claude2.md` §5.

#### P2.3 — SpatialIndex protocol `[L1]`

**File:** `core/types.py`. **Tests:** `test/physics/test_spatial_index_contract.py`.

**Code sketch:**
```python
class SpatialIndex(Protocol):
    def rebuild(self, positions, active, box: tuple | None) -> None: ...
    def query_knn(self, pos, k) -> np.ndarray: ...       # GLOBAL indices, closest-first
    def query_radius(self, pos, r) -> np.ndarray: ...    # GLOBAL indices
    def query_knn_batch(self, positions, k, workers=-1) -> np.ndarray: ...
```

**Citation:** `roadmap.md` D5, `todo_claude1.md` §3.

#### P2.4 — KDTreeIndex returns global indices `[L1]`

**File:** `physics/flock.py::KDTreeIndex`. **Tests:** `test/physics/test_flock.py`.

**Verbal idea.** Map compacted→global via `np.where(active)[0][compacted_idx]`.
Pass `boxsize=(W,H,D)` when toroidal.

**Citation:** `roadmap.md` D5.

#### P2.5 — Ghost-cell replication + modulo cells `[L1]`

**File:** `physics/flock.py::SpatialHashGrid`.

**Math.** Cell keys: `(cx % cols, cy % rows, cz % slices)`. Ghost replication: for birds
within search_radius of boundary, add modulo-offset copies at the opposite side.

**Citation:** `git3` R2, `sci/todo_claude_sci9.md` §13.

#### P2.6 — StepContext + Extension widening `[L1]`

**File:** `physics/extensions/_base.py`. **Tests:** `test/physics/extensions/test_extensions.py`.

**Code sketch:**
```python
@dataclass
class StepContext:
    frame: int; dt: float; rng: np.random.Generator
    center: np.ndarray; config: SimConfig
    threat_prox: np.ndarray | None = None

class Extension(ABC):
    def apply(self, flock: PhysicsFlock, ctx: StepContext) -> None: ...

class ExtensionManager:
    def pre_step(self, flock, ctx):
        for ext in self._extensions:
            if getattr(ctx.config, f"{ext.name}_enabled", True):
                ext.apply(flock, ctx)
```

**Citation:** `roadmap.md` D6.

#### P2.7 — InstanceSchema + VAO discipline `[L2]`

**File:** `viz/renderer.py`. **Tests:** `test/viz/test_renderer.py`.

**Code sketch:**
```python
@dataclass
class InstanceSchema:
    floats: int = 8   # pos.xyz, vel.xyz, flag, hue
    layout: str = "3f 3f 1f 1f/i"

class Renderer3D:
    def _build_bird_vao(self): ...    # called in __init__ AND after every buffer growth
    def __init__(self, config):
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((w,h), components=3)],
            depth_attachment=self.ctx.depth_renderbuffer((w,h)))
```

**Citation:** `todo_claude.md` E1, E2.

#### P2.8 — PyGLM matrix upload `[L2]`

**File:** `viz/renderer.py`.

**Code sketch:**
```python
def _mat4_bytes(m):
    """PyGLM builds differ in memory layout — explicit float32 array is safe."""
    return np.array(m.to_list(), np.float32).tobytes()
```

**Citation:** `todo_claude.md` E3.

#### P2.9 — Holey-mask contract tests `[L1→L2]`

**File:** `test/physics/test_composition.py`.

**Test:**
```python
@pytest.mark.parametrize("mode_name", MODE_REGISTRY.keys())
def test_holey_mask_no_exception(mode_name, holey_flock, cfg): ...
def test_holey_mask_inactive_unchanged(mode_name, holey_flock, cfg): ...
```

**Citation:** `roadmap.md` T4.1.


#### P2.10 — Functional ForceTerm composition `[L1]`

**File:** `physics/forces/_base.py` (extend). **Tests:** `test/physics/forces/test_force_terms.py`.

**Verbal idea.** The 13 field-mode terms and the 4 Reynolds primitives should be
independently testable, runtime-togglable, and swappable — not hardcoded in one
monolithic `step()`. A `ForceTerm` protocol and a `composeForces` reducer make
every term a named, typed, independently-benchmarked function.

**Math.**
```python
@dataclass
class ForceTerm:
    name: str                          # e.g. "shell", "tangential", "drag"
    enabled: bool = True               # runtime toggle
    gain: float = 1.0                  # per-term intensity multiplier
    fn: Callable = None                # (flock, ctx, cfg) → (N,3) force array

composeForces(flock, ctx, cfg, terms: list[ForceTerm]) → (N,3) float32:
    total = zeros(N,3)
    for term in terms:
        if term.enabled:
            total += term.gain * term.fn(flock, ctx, cfg)
    return total
```

**Code sketch:**
```python
# Each term is a pure function — no side effects, no internal state:
def shell_term(flock, ctx, cfg):
    """−d̂(d−R_blob)·coh·1.35·(1−chase)·shell_influence"""
    ...
    return F_shell  # (N,3)

# FieldMode assembles its term list once in __init__:
self.terms = [
    ForceTerm("shell", gain=1.0, fn=shell_term),
    ForceTerm("expand", gain=1.0, fn=expand_term),
    ForceTerm("target", gain=cfg.field.target_pull, fn=target_term),
    # ... 13 terms total
]
# Per frame:
F_total = composeForces(flock, ctx, cfg, self.terms)
```

**Test:**
```python
@pytest.mark.parametrize("term_name", ["shell","tangential","buoyancy","drag","flow","fold"])
def test_each_term_independently(term_name):
    # Single term active → output matches hand-computed expected force
def test_composeForces_linear():
    # composeForces(a+b) = composeForces(a) + composeForces(b)
def test_runtime_toggle():
    # Flip term.enabled = False mid-run → that term contributes zero
def test_all_terms_nonzero():
    # Random config → every term produces nonzero output (no dead terms)
```

**Composers:** `FieldMode`, `SpatialMode`, `VicsekMode` — any mode that accumulates
multiple independent force contributions.

**Citation:** `sci/todo_claude.md` §8 (composeForces pattern),
`sci/todo_claude_sci1.md` §15 (functional ForceTerm).

---

**Phase 2 acceptance:** Architecture test green — import cycle dead, `physics/flock !→ physics/forces`.
All presets load with correct domains (domain.width not overwritten by capture.width).
`MODE_REGISTRY` has ≥7 entries. `KDTreeIndex` + `SpatialHashGrid` conformance suite green
(global indices, toroidal cross-seam). Holey-mask matrix green (all modes × 20 steps).
VAO rebuilt after buffer growth. Headless FBO has depth attachment.
Shim retirement: `not hasattr(SimConfig(), "phi_p")`.

**Architecture test:** `ALLOWED_EDGES` extended:
`physics/flock → core`, `physics/forces/* → core + physics/occlusion + physics/steric + physics/flock(read)`,
`physics/extensions → core + physics/flock`, `simulation/engine → physics/* + analysis`,
`viz/ → core + physics/flock(read) + analysis/presets` (never simulation import),
`capture/ → simulation + viz + core`, `analysis/{metrics,rewards,presets} → core + physics/flock(read)`.
**New conformance suites:** `test_force_mode_contract.py`, `test_spatial_index_contract.py`,
`test_extension_contract.py` — every registered class passes these suites.

**Migration:** Flat `SimConfig` → nested dataclasses. Property shims for backward compat;
retire shims after P3. `physics/flock` no longer imports from `physics/forces/`
(imports moved to engine). Private `cKDTree` calls in `forces/` deleted — all through
`flock.index`. Flat `conf/*.yaml` files rewritten to nested schema.

**Dead-code cleanup (deferred from P0):** `FlockArrays` composed into `PhysicsFlock.arrays`;
`ForceKernel` deleted; `BoidView` deleted. These are safe to remove now that the import
cycle is broken and all consumers use the new protocols.

**ForceTerm conformance:** `composeForces` linearity test green; each ForceTerm
independently testable and runtime-togglable.

---

### Phase 3 — Field/Blob Mode + Threat FSM

**Ships:** complete crs48 field mode (13 force terms: 5 blob anchors, leader/chaser, shell/cavity, slot repulsion, 6 remaining terms); Threat FSM with force bundle (push/wake/split/wave); boundedUnitTravel wander; 3-train ripple envelopes; 7 field presets; grid-mode separation normalization; 1.45·R_blob floating boundary

**Est. effort:** 5.5 days

**Depends on:** P2. **Produces:** complete crs48 field mode — all 13 force terms
(5 blob anchors, cyclic phase weights, leader/chaser groups, shell force + cavity,
slot repulsion, buoyancy, tangential orbital, curl flow, fold noise, drag, drift,
ripple envelopes), bounded panic + blackening, full Threat FSM with wake/split/wave
force bundle, bounded-unit-travel wander, 7 field presets.

**Files:** `physics/forces/field.py` (rewrite), `physics/extensions/predator.py` (rewrite),
`physics/extensions/wander.py` (rewrite), `physics/extensions/ripple.py` (update).
**Tests:** `test/physics/forces/test_field.py`, `test/physics/extensions/test_threat.py`.
**Presets:** `conf/field_quiet_roost.yaml`, `conf/field_lava_lamp.yaml`,
`conf/field_ink_cloud.yaml`, `conf/field_predator_ripple.yaml`,
`conf/field_vacuole.yaml`, `conf/field_silk_sheet.yaml`, `conf/field_storm_turn.yaml`.

**Shared unit scale:** `U = cfg.field.unit_scale or 0.4 · min(W, H, D)`.

#### P3.1 — Wander path (boundedUnitTravel) `[L1]`

**Math.**
```
raw_x = sin(t·0.47 + sin(t·0.19)·1.15)·0.82 + sin(t·1.07+1.4)·0.38 + cos(t·0.23+2.1)·0.22
raw_y = cos(t·0.43+0.6 + sin(t·0.13)·0.9)·0.78 + sin(t·0.91+2.8)·0.42 + cos(t·0.29+0.4)·0.24
raw_z = sin(t·0.39+1.1 + cos(t·0.17)·1.05)·0.80 + cos(t·0.97+0.2)·0.40 + sin(t·0.21+2.6)·0.22
pulse = 0.72 + 0.28·(0.5 + 0.5·sin(t·0.41 + cos(t·0.17)))
path(t) = raw · pulse / max(1, ‖raw‖)              ⇒ ‖path‖ ≤ 1 guaranteed
wander_center(t) = C + path(t·speed)·radius·U
heading(t) = normalize(path(t+0.75) − path(t))
```

**Test:** `‖path‖ ≤ 1` for 10⁶ fuzzed t; heading continuous (‖h(t+ε)−h(t)‖ < 0.05).

**Citation:** `git6` R2, `sci/todo_claude_sci1.md` §8.

#### P3.2 — Five Lissajous blob anchors + cyclic phase weights `[L1]`

**File:** `physics/forces/field.py::FieldMode`.

**Math (anchors).**
```
B₀ = C + (sin(t·0.19)·0.74,          sin(t·0.31+0.8)·0.48,  cos(t·0.23)·0.62)·U
B₁ = C + (cos(t·0.17+1.6)·0.68,      sin(t·0.37+2.1)·0.54,  sin(t·0.29+0.4)·0.72)·U
B₂ = C + (sin(t·0.27+2.7)·0.58,      cos(t·0.21+1.2)·0.42,  cos(t·0.33+2.5)·0.68)·U
B₃ = C + (cos(t·0.24+3.4)·0.70,      sin(t·0.33+0.6)·0.50,  sin(t·0.18+1.4)·0.58)·U
B₄ = C + (sin(t·0.14+4.4)·0.48,      sin(t·0.47+2.3)·0.62,  cos(t·0.26+4.0)·0.70)·U
```

**Math (weights).**
```
φ_i = fract(seed_i · 3.71 + t · 0.022 + sin(seed_i · 19 + t · 0.11) · 0.09)
c_k ∈ {0, 0.2, 0.4, 0.6, 0.8}
wrap_dist(φ, c) = min(|φ−c|, 1−|φ−c|)
w_k = max(0, 1 − wrap_dist(φ, c_k) · 7.5)²
T_legacy_i = (Σ_k B_k · w_k) / Σ_k w_k    [Σw > 0 provably]
```

**Test:** 2K birds → k-means finds ≥4 clusters at t=30s; per-bird target variance > 0.

**Citation:** `git6` R3, `sci/todo_claude_sci1.md` §2.1–2.2.

#### P3.3 — Leader/chaser groups `[L1]`

**Math.**
```
group_seed = floor(seed_i · 7) / 7                              [7 groups]
lag_i = hash01(seed_i + 9.17) · (1.1 + chaseStrength · 2.4)
leader_i = hash01(seed_i + 5.91) ≥ 0.84            [~16% leaders]

Golden-angle shell offset (per group):
    ga = 2.39996323
    y = 1 − 2·fract((slot+0.5)·0.618034 + gs·0.13)
    shell = fract((slot+1)·0.754877)^(1/3)
    radius = (0.16 + shell·0.34)·(0.68 + cs·0.34)·(0.92 + sep·0.045)·U
    breath = 1 + sin(t·0.13 + gs·12)·0.035

T_i = lerp(T_legacy, chase_target, chaseStrength)
```

**Test:** leader fraction 0.16±0.02; chase=0 ≡ P3.2 targets; chase=0.8 → 7-cluster structure.

**Citation:** `git6` R4, `sci/todo_claude_sci1.md` §2.2.

#### P3.4 — Shell force + inner cavity `[L1]`

**Math.**
```
R_blob,i = (0.24 + (0.5+0.5·sin(seed_i·41 + t·0.29))·0.16
                + sin(φ_i·2π + t·0.17)·0.05) · U
F_shell = −d̂ · (d − R_blob) · cohesion · 1.35 · (1 − chaseStrength) · shell_influence

inner_i = R_blob,i · (0.28 + (1−chaseStrength)·0.18 + separation·0.012)
if d < inner_i: F_expand = d̂ · (inner_i − d) · separation · 1.4
```

**Test:** Settled 5K blob — centre voxel density < 0.3× shell band density.

**Citation:** `git6` R5, `sci/todo_claude_sci1.md` §2.3.

#### P3.5 — Slot repulsion `[L1]`

**Math.** Offsets `{±1, ±7, ±31}` with modulo wrap:
```
r_slot = (0.07 + separation·0.02)·U
if d < r_slot: F_slot += (away/d) · ((r_slot−d)/r_slot)²
gain = separation · (0.14 + chaseStrength·0.05)
```

**Test:** kernel zero at r_slot and beyond; continuous at boundary.

**Citation:** `git6` R6, `sci/todo_claude_sci2.md` §3.5.

#### P3.6 — Remaining 6 field terms `[L1]`

**Math (per bird, vectorised).**
```
Tangential orbital:
    axis_i = normalize(sin(t·0.13+seed·7), 0.72+sin(t·0.19+seed·3)·0.28, cos(t·0.17+seed·5))
    F_tan = normalize(axis × (p−T)) · alignment · 0.035 · (1−chase) · tangent_pull

Buoyancy (z-up):
    F_z += (sin(d·8/U − t·1.1 + seed·17)·0.09 + (T_z−p_z)/U·0.24) · (0.75 + flow·0.25)

Curl flow (q = (p−C)/U):
    flow_vec = (sin(q_y·2.8 + t·0.24) + cos(q_z·2.1 − t·0.17), ...)
    F_flow = normalize(flow_vec) · flow · 0.08 · flow_pull

Fold noise: F_fold = fold_vec · flow · flow_pull · ripple_envelope_sum

Viscous drag: F_drag = −v · chaseStrength · (0.08 + flow·0.02)

Drift alignment: F_drift = (wander_heading·v0 − v) · alignment · drift_pull
```

**Test:** 10⁴-frame NaN/speed fuzz all-terms-on passes. Tangential on → nonzero L about blob axes.
Buoyancy z-only. Drag anti-parallel to velocity.

**Citation:** `git6` R6, `sci/todo_claude_sci1.md` §2.6, §3.4–3.7.

#### P3.7 — Ripple envelopes (vectorised) `[L1]`

**Math.** Three trains at offsets `{0, 9.33, 18.67}`:
```
env(τ) = smoothstep(0.6, 1.7, τ) · (1 − smoothstep(6.2, 8.8, τ))
radius(τ) = (0.16 + τ·0.16)·U;  width(τ) = (0.11 + τ·0.012)·U
origin(τ) = C + (sin(t·0.17+o)·0.46, cos(t·0.13+o·1.7)·0.25, cos(t·0.19+o·0.6)·0.42)·U
amount = exp(−|(r−radius)/width|²) · env(local_t)
F_radial = (p−origin)/r · amount;  F_twist = heading × F_radial
F_ripple = (F_radial + F_twist·0.28) · flow · (0.13 + waveGain·0.04)
```

**Test:** env zero outside [0.6,8.8]; paused-flock radial histogram shows 3 rings; <5ms at N=100K.

**Citation:** `git6` R7, `sci/todo_claude_sci1.md` §2.5.

#### P3.8 — Bounded panic + blackening `[L1]`

**Math.**
```
panic = clamp(prox_i, 0, 1) · threat_strength
boost = panic · (0.72 + wave_gain·0.18 + vacuole_strength·0.12)
max_speed_i = v0 · (1 + min(1.35, boost))    [ceiling raise, NOT compound multiply]

black = 1 + blackening_gain · prox_i · 0.85
sep_eff = separation · (2 − black)           [weaker near threat]
coh_eff = cohesion · black                   [stronger near threat]
```

**Test:** panicked speed ≤ 2.35·v0 always; near-threat pair separation decreases.

**Citation:** `git6` R8, `sci/todo_claude_sci1.md` §3.1–3.3.

#### P3.9 — Threat FSM + force bundle `[L1]`

**Math.**
```
capture = max(0.18, threat_radius·0.72)·U
pass_dist = (0.92 + threat_radius·2.6 + momentum·1.32)·U
clear = pass_dist·(0.72 + momentum·0.16)
turn_rate = (0.54 + acceleration·0.025)·(1 − momentum·0.24)    [chase]
          = 0.42·(1 − momentum·0.24)                           [orbit]

Approach→egress:  ‖p_threat−center‖ ≤ capture
Egress→approach:  ‖p_threat−center‖ > clear AND dot(dir, to_center̂) < −0.12

Force bundle (prox = 1−d/(threat_radius·U·2); broad = √prox; â = away/d):
    push  = â · strength · (2.5+vacuole_strength·1.7) · broad
    wake  = (â−dir·0.35) · min(1.8,‖v_t‖/v0) · strength · broad · 0.42
    split = (−â_y·1.45, â_x·1.45, â_z·0.28) · splitGain · broad    [XZ tear, z-up]
    wave  = v̂_i · waveGain · broad · 0.22
```
Modes: `off | cursor | orbit | autonomous`. Publishes `ctx.threat_prox`.

**Test:** predator passes through and exits (>clear distance); evacuated region
horizontally biased (XY-extent > Z-extent); `threat_prox ∈ [0,1]`.
*(The red predator marker rendering is deferred to P8.4 — threat force bundle is fully functional in headless mode at P3.)*

**Citation:** `git6` R9, `sci/todo_claude_sci1.md` §4.

#### P3.10 — Blob init + 7 field presets `[L1]`

**Init math.**
```
centres = C + (−0.48,0.18,0.12) (0.36,−0.20,−0.28) (0.12,0.34,0.42)
              (−0.16,−0.30,0.34) (0.48,0.16,0.18) · U
r = cbrt(U(0,1))·(0.22 + U(0,1)·0.28)·U;  jitter = U(−1,1)·0.045·U
v = ((0.34±0.08), ±0.16, (0.08±0.08))·v0·0.5    [drift-biased tangential]
```

**Presets** (`conf/field_*.yaml`, 7 files):

| Preset | N | v0 | sep | align | coh | chase | notable |
|---|---|---|---|---|---|---|---|
| quiet_roost | 3K | 0.48 | 0.85 | 0.65 | 1.85 | 0.72 | Slow, cohesive |
| lava_lamp | 16K | — | — | — | — | 0 | Pure blob dynamics |
| ink_cloud | 18K | 0.62 | 0.92 | 0.90 | 1.80 | 0.82 | Dense, accumulation trails |
| predator_ripple | 12K | 0.78 | 1.05 | 1.05 | 1.15 | 0.64 | Orbit threat + waves |
| vacuole | 10K | 0.68 | 1.12 | 0.92 | 1.25 | 0.76 | vacuole_strength=0.9 |
| silk_sheet | 14K | 0.46 | 0.92 | 1.10 | 1.10 | 0.68 | Thin planar |
| storm_turn | 16K | 0.90 | 1.10 | 1.15 | 1.25 | 0.42 | Fast, autonomous threat |

**Test:** All 7 presets load with documented values; frame-0 lobes visible.

**Citation:** `git6` R12, `sci/todo_claude_sci2.md` §12.


#### P3.11 — Grid-mode separation normalization `[L1]`

**File:** `physics/forces/field.py`.

**Math.** The separation term currently sums raw 1/d² contributions without dividing
by neighbour count. Dense regions get disproportionately large kicks:
```
F_sep_grid = −(separation / max(1, neighbour_count)) · Σ_{j∈N_i} r̂_{ij} / d_{ij}²
```
This is the **averaged** steering vector, not the raw sum — the sci spec's
documented `· separation / max(1, found)` normalization that prevents density
hotspots from self-amplifying.

**Test:**
```python
def test_grid_sep_normalized_by_count():
    # Bird with 10 equidistant neighbours vs bird with 2 → same |F_sep|
    # (within 5% — slight difference from geometry, not neighbour count)
```

**Citation:** `sci/todo_claude_sci2.md` §2.3 ("Separation normalised by neighbour count").


#### P3.12 — Field-mode 1.45·R_blob floating boundary `[L1]`

**File:** `physics/forces/field.py::FieldMode`.

**Math.** Field mode needs a dynamic soft boundary that floats with the blob radius
rather than a fixed sphere centred on C. Without it, the generic sphere boundary
(P4.7) cuts into the wave dynamics during blob expansion:
```
R_boundary = 1.45 · max_i(R_blob,i)    [floats with the blob]
if |p_i − C| > R_boundary:
    Δv = −μ · r̂ / max(R_boundary − |p_i−C|, 0.05·R_boundary)
```
Applied as the final force term in `composeForces` (P2.10) — a soft push-back
that scales with the current blob extent.

**Test:**
```python
def test_floating_boundary_scales_with_blob():
    # Small blob (chase=0) → small R_boundary
    # Large blob (chase=0.8, leader expansion) → R_boundary grows
    # No bird exceeds 1.02·R_boundary over 10⁴ frames
```

**Citation:** `sci/todo_claude.md` §8, `sci/todo_claude_git6.md`.

---

**Phase 3 acceptance:** Golden trajectory pinned for field mode. 10⁴-frame NaN/speed fuzz
all-terms-on passes. Seven presets load and run. Grid-mode separation normalized by
neighbour count (dense vs sparse regions → same |F_sep|). 1.45·R_blob floating
boundary contains birds without cutting wave dynamics. Threat passes through flock
and exits. Wander path stays in-domain over 10⁴ frames. `‖path‖ ≤ 1` for 10⁶ fuzzed t.
Field mode runs at ≤3ms at N=16K.

**Architecture test:** New edges: `field.py → core + physics/flock(read)`,
`extensions/predator.py → core + physics/flock(read) + physics/forces`,
`extensions/wander.py → core`, `extensions/ripple.py → core + physics/flock(read)`.

**New files:** `conf/field_*.yaml` (7 presets).

---

### Phase 4 — Reynolds Variants + Ecology

**Ships:** hybrid metric+topological filter; correct force accumulation order; predator boids (species-based); physical metrics (watts/newtons/joules); per-frame parameter jitter; parallel two-phase update; sphere/sphere_soft boundary modes; ecology (logistic dusk, coherence gate, seasonal model); 4 velocity-init variants; numba force kernels

**Subsystem:** E, F1 | **Level:** L1 | **Est. effort:** 3.5 days

**Depends on:** P2. **Produces:** hybrid metric+topological filter, correct force
accumulation order, predator boids (species-based), physical metrics (watts/newtons),
per-frame parameter jitter, parallel two-phase update, sphere centring + asymptotic
wall, ecology completion (logistic dusk, coherence gate, seasonal model),
velocity-init variants, numba force kernels.

**Files:** `physics/forces/spatial.py` (rewrite), `physics/forces/_kernels.py` (new),
`physics/extensions/ecology.py` (rewrite), `physics/boid.py` (boundary additions),
`analysis/metrics.py` (physical units).
**Tests:** `test/physics/forces/test_spatial_variants.py`,
`test/physics/extensions/test_ecology.py`, `test/physics/forces/test_kernels.py`.

#### P4.1 — Hybrid metric+topological filter `[L1]`

**Math.** Neighbour iff `d < visual_range` AND among first `influence_count` (7) accepted.
Alignment subset: `d < alignment_radius_ratio · visual_range` (0.75).
Separation gate: `d < separation_distance` (20). Global-neighbourhood fallback for MARL.

**Config:** `neighbor_filter: "hybrid" | "knn" | "global"`, `influence_count: 7`,
`alignment_radius_ratio: 0.75`, `separation_distance: 20.0`.

**Test:** Neighbour set respects radius AND cap; alignment set ⊆ cohesion set.

**Citation:** `git5` R1, `sci/todo_claude_sci3.md` §4.

#### P4.2 — Force accumulation order `[L1]`

**Math.**
```
forces → predator_boost(×1.4) → acceleration_scale(0.3) → clamp(max_force)
→ v += a → velocity_noise (when noise_mode="velocity") → speed_clamp → move → wrap
```

**Config:** `acceleration_scale: 1.0`, `noise_mode: "force" | "velocity"`,
`speed_mode: "band" | "fixed" | "ceiling"`.

**Test:** Monkeypatched stage-order recording matches doc; ceiling allows |v| < 0.3v0;
fixed → |v| ≡ v0 ± 1e-5.

**Citation:** `git3` R1, `sci/todo_claude_sci3.md` §1, `sci/todo_claude_sci5.md` §5.

#### P4.3 — Predator boids (species) `[L1]`

**Math.**
```
Boosts: speed ×1.8, perception ×1.5, acceleration ×1.4
Escape: F_esc = normalize(p_prey − p_pred) · predator_escape_factor (10⁷)
        replaces separation entirely
Hard zero: align + coh = 0 when any predator perceived
Predators flock among themselves (normal sep/align/coh)
```

**Test:** Hand neighbourhood with predator → align+coh exactly zero; escape >> separation;
flash-expansion (mean NN distance doubles in 30 frames).

**Citation:** `git3` R3, `sci/todo_claude_sci5.md` §2.

#### P4.4 — Physical metrics `[L1]`

**Math.**
```
k_v = cruise_speed_ms / v0       [default: 8.94 / v0]
k_a = acc_peak_ms2 / max_force   [default: 40.0 / max_force]
m   = bird_mass_kg               [default: 0.075 kg]
|v|_real = k_v · |v|_sim          [m/s]
|a|_real = k_a · |a|_sim          [m/s²]
F_avg = (m/N) · Σ k_a·|a_i|      [newtons]
P_avg = (m/N) · Σ |(k_a·a_i)·(k_v·v_i)|  [watts]
E = Σ P_avg(t) · Δt               [joules]
L = m · (r−CoM) × (k_v·v)        [kg·m²/s, about CoM]
```

**File:** `analysis/metrics.py` — add `speed_real, accel_real, force_real_N, power_real_W, energy_J, angular_momentum_real`.

**Test:** Hand-set v → `speed_real = k_v·|v|` exactly; E ≈ P̄·elapsed ± 1%; L about CoM.

**Citation:** `git5` R3, `sci/todo_claude_sci3.md` §3.

#### P4.5 — Per-frame parameter jitter `[L1]`

**Math.**
```
sep_eff   = separation_weight   + U(0, jitter_separation)      [default +U(0,0.5)]
coh_eff   = cohesion_weight     + U(0, jitter_cohesion)        [default +U(0,0.1)]
align_eff = alignment_weight    + U(0, jitter_alignment)       [default +U(0,0.005)]
```
Config never mutated. Seeded RNG.

**Test:** spacing_std(on) > spacing_std(off), same seed; config unchanged after run.

**Citation:** `git5` R4, `sci/todo_claude_sci3.md` §5.

#### P4.6 — Parallel two-phase update `[L1]`

**Code sketch:** batched `index.query_knn_batch(pos, k, workers=cfg.perf.num_threads)` +
fully vectorised gather/reduce force pass (no per-bird Python loops).

**Test:** ≥3× at N=20K vs recorded loop baseline (`@slow`); identical results across worker counts.

**Citation:** `git3` R4, `sci/todo_claude_sci5.md` §4.

#### P4.7 — Sphere centring + asymptotic wall `[L1]`

**Math.**
```
Sphere (impulsive):   if r > R: v ← v − μ·r̂
Sphere soft (asymptotic): Δv = −μ·r̂ / max(R−r, 0.05R)
```
Both centred on `C = (W/2, H/2, D/2)`. Config: `boundary.mode: "sphere" | "sphere_soft"`.

**Test:** Centre-initialised flock ‖CoM−C‖ < 0.1R over 5000 frames; soft mode never crosses R.

**Citation:** `git5` R2, `sci/todo_claude_sci3.md` §2.

#### P4.8 — Ecology completion `[L1]`

**Math.**
```
sunset = 12 + day_length/2
dusk_factor = 1/(1+e^{−z}), z = (hour−sunset)/(DUSK_WIDTH/4), clamp |z|>60
roost_strength = base · dusk_factor · max(0, 1+0.2·(T_mean−T(day))/T_amp)
coherence(N) = smoothstep(0.4·N_crit, 1.2·N_crit, N)
gated_weight(w, N) = w · coherence(N)    [applied to flocking weights]
seasonal_size_factor(day) = cos(2π·(day−15)/365), peak=1.0, trough=0.25
is_murmuration_season(day) = Oct–Mar
```

**Test:** `seasonal_size_factor(15)≈1.0, (197)≈0.25`; Jan-in/Jul-out; `dusk_factor(0)=0, (40)=1`;
`gated_weight(0.8,10)≈0, (0.8,600)>0.7`; colder days → stronger roost.

**Citation:** `todo_claude.md` §5–8, `sci/todo_claude_sci3.md` §6.

#### P4.9 — Velocity-init variants `[L1]`

**Math.**
```
"cube":         v = (U³−0.5) · 2·v0      [E|v| ≈ 0.816·v0]
"speed_uniform":v = dir̂_uniform · U(min(1, 0.3v0), v0)
"tangential":   v = normalize(p−C) × random_unit · U(1, v0)
"fixed":        v = dir̂_uniform · 0.8·v0  [legacy]
```

**Test:** cube E|v|≈0.816·v0 ±5%; speed_uniform in-band, non-constant; tangential ⊥ radial.

**Citation:** `todo_claude.md` E12, `sci/todo_claude_sci5.md` §10.

#### P4.10 — Numba force kernels `[L1]`

**File:** `physics/forces/_kernels.py` (new).

**Math.** Batched index query (Python/scipy) → `@njit(parallel=True)` kernel.
`cfg.perf.use_numba` gates; `fastmath` allowed only when `metrics.detail_level == 0`
(visual runs) — IEEE kernels whenever observables are exported.

**Test:** numba ≡ numpy within `atol=1e-5` (fastmath off), N=2K; `@slow` N=50K within budget.

**Citation:** `arch.md` §9, `sci/todo_claude_sci9.md` §17.

---

**Phase 4 acceptance:** Golden pinned for spatial mode. Physical metrics report
plausible watts/newtons. Ecology seasonal/coherence/dusk tests green. Jitter-increased
variance verified. Sphere centred on domain centre. Numba-numpy equivalence green.
Predator escapes dominate forces. Flash-expansion visible within 30 frames.

**Architecture test:** New edges: `spatial.py → core + physics/flock(read) + physics/forces/_base`,
`_kernels.py → core (numba optional)`, `ecology.py → core + physics/flock(read)`.
**New files:** `physics/forces/_kernels.py`, `test/physics/forces/test_kernels.py`.

---

### Phase 5 — Angle Mode (PyNBoids Paradigm)

**Ships:** standalone angle force mode: Rodrigues steering core with dead-zone, unified neighbour modes (flee/align+coh/coh-only), adaptive speed, edge handling, heading jitter, incremental spatial grid, body-unit scale invariance

**Subsystem:** E | **Level:** L1 | **Est. effort:** 2 days

**Depends on:** P2. **Produces:** new `"angle"` force mode — axis-angle heading steering
with turn-rate cap (Rodrigues rotation in 3D), unified neighbour modes, adaptive speed,
cardinal-axis edge avoidance, per-frame heading jitter, incremental spatial grid,
body-unit scale invariance.

**Files:** `physics/forces/angle.py` (new). **Tests:** `test/physics/forces/test_angle.py`.

#### P5.1 — Steering core `[L1]`

**Math.**
```
φ = acos(clamp(ĥ·t̂, −1, 1))
k̂ = normalize(ĥ × t̂)    [any ⊥ axis when parallel/anti-parallel]
ĥ ← rotate_about(ĥ, k̂, min(φ, turnRate·dt))          [Rodrigues, never overshoot]
```
Dead zone: no turn if `φ < turn_threshold°` (anti-oscillation).

**Test:** 180° turn completes in π/rate seconds ±1 frame; per-frame heading change ≤ rate·dt + jitter.

**Citation:** `git4` R1, `sci/todo_claude_sci4.md` §1.

#### P5.2 — Unified neighbour modes `[L1]`

**Math.** 7 closest within `boid_size·12`:
```
nearest < b·1:  steer away from nearest (exclusive flee state)
nearest < b·5:  toward normalize(ĉ + m̂)  (centroid + mean heading in 3D)
else:           toward ĉ only
```

**Citation:** `git4` R2, `sci/todo_claude_sci4.md` §2.

#### P5.3 — Adaptive speed `[L1]`

**Math.**
```
s = base_speed + (7 − neighbor_count)·5          [linear]
  = base_speed + min(49, (7−m)²)                 [quadratic]
  = base_speed + min(49, (7−m)²·0.5)             [softened]
```

**Test:** m=0 → base+35 (linear); m≥7 → base; median 7th-NN distance converges.

**Citation:** `git4` R3, `sci/todo_claude_sci4.md` §3.

#### P5.4 — Edge handling `[L1]`

**Math.**
```
Cube: inside margin, target = nearest face inward normal (±x/±y/±z, sequence priority)
Sphere: t̂ = normalize(C−p)
turnRate += (1 − dist/margin) · (maxTurnRate − turnRate)
```

**Test:** 10⁴ frames at max speeds → zero escapes; birds arc (tangential speed at wall > 0).

**Citation:** `git4` R4, `sci/todo_claude_sci4.md` §4.

#### P5.5 — Heading jitter `[L1]`

**Math.** `ĥ ← rotate_about(ĥ, random_axis, ±jitter_deg°)` before steering. From `flock.rng`.

**Test:** Steering-off distribution bounded ±4°; net track endpoint within 2% of jitter-off run.

**Citation:** `git4` R5, `sci/todo_claude_sci4.md` §5.

#### P5.6 — Incremental spatial grid `[L1]`

Per-bird `last_cell`; re-file only on cell crossing. Behind `SpatialIndex` protocol.

**Test:** Neighbour sets == full-rebuild sets over 500 random-walk frames; touches <10% of birds/frame.

**Citation:** `git4` R6, `sci/todo_claude_sci4.md` §6.

#### P5.7 — Body-unit scale invariance `[L1]`

**Math.** All radii as `boid_size` multiples: `sep/align/range_radius_bodies`.

**Test:** Doubling `boid_size` doubles all three thresholds; 2×-scale behavioural smoke.

**Citation:** `git4` R8, `sci/todo_claude_sci4.md` §10.

---

**Phase 5 acceptance:** `"angle"` mode loads from registry. Dead-zone hold exact.
Birds arcing at walls. Speed self-regulates. Incremental grid equivalent to full rebuild.
Doubling boid_size doubles all radii.

**Architecture test:** New edges: `angle.py → core + physics/flock(read)`.
**New files:** `physics/forces/angle.py`, `test/physics/forces/test_angle.py`.

---

### Phase 6 — Vicsek Predator–Prey

**Ships:** fear-weighted alignment blending; predator hunting strategy; asymmetric position collisions (same-type symmetric, prey-predator asymmetric, seam-crossing correct)

**Subsystem:** E | **Level:** L1 | **Est. effort:** 1.5 days

**Depends on:** P2 (P1.8 for core vicsek fix). **Produces:** fear-weighted alignment
blending, predator hunting strategy, asymmetric position collisions, prey-only metrics.

**Files:** `physics/forces/vicsek.py` (extend VicsekMode).
**Tests:** `test/physics/forces/test_vicsek_species.py`.

#### P6.1 — Fear-weighted alignment `[L1]`

**Math.**
```
fear = clamp((R_pred − d̄_pred) / R_pred, 0, 1)     [d̄ = min-image mean distance]
û_combined = normalize((1−fear)·û_align + fear·û_flee)
û_flee = normalize(Σ(p_prey−p_k)/|P|)              [random unit if no predators]
Neighbour weights ×weight_afraid (3.0) while afraid
```

**Test:** Stationary predator at centre → prey ⟨û·r̂⟩ > 0.8 within 5 steps.

**Citation:** `git2` R2, `sci/todo_claude_sci8.md` §2.

#### P6.2 — Predator agent `[L1]`

**Math.**
```
Hunting: nearest prey within detect_ratio·R_pred (1.5×)
Update: û = normalize(û_target + predator_noise_ratio·η̂)   [no couplage]
Fallback: random walk if no prey in range
All-predators: early-out, skip all interaction
```

**Test:** Monotone pursuit (≥90% of steps close distance); n_prey=0 → α ≈ 1/√N for all η,D.

**Citation:** `git2` R3, `sci/todo_claude_sci8.md` §2.

#### P6.3 — Asymmetric position collisions `[L1]`

**Math.**
```
Same-type at d < R_avoid:     each moves (R_avoid−d)/2 along min-image n̂
Prey–predator at d < R_pred:  prey takes FULL (R_pred−d), predator unmoved
Applied after move, before wrap, via np.add.at accumulation
```

**Test:** Hand pair corrections exact (both cases); seam-crossing corrected;
100 steps → no same-type pair < 0.5·R_avoid; predator trace unaffected.

**Citation:** `git2` R5, `sci/todo_claude_sci8.md` §3.

---

**Phase 6 acceptance:** Fear-weighted alignment active. Predator hunts nearest prey.
Asymmetric collisions correct at seam. Prey-only α=1.0 with one orthogonal predator.

**Architecture test:** New edges: `vicsek.py → core + physics/flock(read)` (extended).
**New tests:** `test/physics/forces/test_vicsek_species.py`.

---

### Phase 7 — Influencer Parity (MurmuratR)

**Ships:** persistent tick-driven Lissajous target; move-then-steer at unit speed; rank-by-target-distance influence; density-scaled Gaussian init; distance diagnostics; desktop pilotable-flock mode (WASD attractor + shell force)

**Subsystem:** E | **Level:** L1 | **Est. effort:** 1.5 days

**Depends on:** P2. **Produces:** persistent tick-driven Lissajous trajectory,
move-then-steer at unit speed, rank-by-target-distance influence, density-scaled
Gaussian init, per-frame distance diagnostics, desktop pilotable-flock mode
(keyboard-steered attractor with shell forces).

**Files:** `physics/forces/influencer.py` (rewrite).
**Tests:** `test/physics/forces/test_influencer.py`.

#### P7.1 — Persistent tick + Lissajous target `[L1]`

**Math.**
```
T_raw(t) = (sin(t/97)·200 + cos(t/217)·30,
            cos((t+53)/29)·200 + sin((47−t)/13)·30,
            cos((t+61)/41)·100 + sin((t+13)/7)·27 + 40)
s = scale · min(W/460, H/460, D/254)
T(t) = C + (T_raw(t) − (0,0,40))·s + (0, 0, 40s)
```
Tick increments once per substep (not random teleport).

**Test:** `T(t)` at t∈{0,970,2170} equals hand values; in-domain for scale≤1.

**Citation:** `git1` R1–R2, `sci/todo_claude_sci6.md` §1.

#### P7.2 — Move-then-steer at unit speed `[L1]`

**Math.**
```
p += d̂_old · v0 · dt                [move using OLD direction]
d̂ ← normalize(d̂(1−inf) + t̂·inf)    [then blend toward target]
```
Guard `x += (x == 0)` for zero-speed safety. `owns_positions=True, speed_mode="fixed"`.

**Test:** Frozen target → convergence to hover/orbit; one-step lag after target jump; |v| ≡ v0.

**Citation:** `git1` R3, `sci/todo_claude_sci6.md` §1.3.

#### P7.3 — Rank-by-target-distance influence `[L1]`

**Math.**
```
Distance-based: inf_A = clamp(near_dist_sq·s²/d², 0.2, 0.8)
Rank-based: inf_sorted[i] = (1 − (i/(N−1))·0.8)^rank_exponent  [floor 0.2^1.8 ≈ 0.055]
```
Ranks by distance to **target** (not CoM). Config: `influencer.influence_mode`.

**Test:** Exactly one bird at 1.0; min ≈ 0.055±1e-3; monotone non-increasing in target distance.

**Citation:** `git1` R4, `sci/todo_claude_sci6.md` §3.

#### P7.4 — Density-scaled init `[L1]`

**Math.**
```
σ = N^(1/3) · separation · s
positions = rnorm(N,3)·σ + C + U(0,10s)³
zero initial directions (first blend heads all birds at target)
```

**Test:** Init density equal across N∈{100,1000,8000} (±10%); frame-0 headings ∝ influence.

**Citation:** `git1` R5, `sci/todo_claude_sci6.md` §5.

#### P7.5 — Distance diagnostics `[L1]`

Per-frame `min/max ‖p−T‖` → `FlockMetrics.target_dist_min/max` + window title.

**Citation:** `git1` R6, `sci/todo_claude_sci6.md` §6.3.

#### P7.6 — Desktop pilotable-flock mode `[L1]`

**File:** `physics/forces/influencer.py::InfluencerMode` (force math);
keyboard bindings wired in `viz/input_control.py` (reusing P10 UX infrastructure).

**Verbal idea.** Desktop adaptation of the VR SwarmPilot: the user steers a
keyboard-controlled attractor with an enclosing shell force, giving direct
"conducting" control over flock movement. Distinct from the Lissajous influencer
(P7.1)—here the target is user-driven, not scripted. Birds follow with per-bird
influence weights and the same move-then-steer as P7.2.

**Math.**
```
F_pilot,i = heading_force + core_follow + shell_pull
heading_force = pilot_heading · alignment · 0.12
core_follow   = (pilot_pos − p_i) · cohesion · 0.22    [unbounded — intentional strong attractor]
shell_pull    = (pilot_pos − p_i) / d · (d − shell_radius) · 0.42
```
The pilot position evolves per frame:
```
pilot_pos  += pilot_heading · pilot_speed · dt
pilot_heading = WASD + QE roll + arrow pitch/yaw input (normalized)
shell_radius  = clamp(0.42, 2.2, radius + (scatter − gather) · dt · 1.35)
```

**Keyboard bindings.** W/S thrust, A/D yaw, arrows pitch/yaw, Q/E roll,
Shift gather (shrink shell), Alt scatter (expand shell), Space pulse.

**Test:**
```python
def test_pilot_stationary_flock_converges():
    # Static pilot at C → all birds converge to shell_radius sphere within 60 frames
def test_pilot_moving_flock_follows():
    # Pilot moves +X at constant speed → flock CoM tracks pilot within 2 · shell_radius
def test_shell_radius_expands_on_scatter():
    # Alt held → shell_radius increases monotonically up to 2.2 cap
```

**Citation:** `sci/todo_claude_sci1.md` §7.2 (pilot-aware forces),
`sci/todo_claude_git0.md` W2 (pilotable flock mode).

---

**Phase 7 acceptance:** Persistent tick → deterministic trajectory. Move-then-steer lag visible.
Influence rank-based monotone. Density-scaled init equal across N. CSV contains target_dist columns.
Pilot mode: WASD steers attractor, flock follows within 2·shell_radius, Shift/Alt shrink/expand shell.

**Architecture test:** New edges: `influencer.py → core + physics/flock(read)` (extended).
**New tests:** `test/physics/forces/test_influencer.py`.

---

### Phase 8 — Rendering & Capture

**Ships:** sphere impostors + speed-stretched ellipsoids; depth cues + Fresnel rim; 4 trail modes (velocity/accumulation/ring/lines); winged flapping mesh + gradient sky; theme wiring (ambient+diffuse material tables); adaptive quality governor (EMA, degradation ladder); cinematic capture sweep (GIF/CSV/JSON); dual-view + orthographic presets; GPU-free matplotlib fallback; fixed-timestep accumulator; alpha-accumulation density mode

**Subsystem:** C, D | **Level:** L2 | **Est. effort:** 5 days

**Depends on:** P2 (P2.7 InstanceSchema, P2.8 mat4). **Produces:** sphere impostors,
depth cues, 4 trail modes (velocity/accumulation/ring/lines), winged flapping mesh,
gradient sky, theme wiring, adaptive quality wired, cinematic capture sweep,
dual-view rendering, GPU-free matplotlib fallback, fixed-timestep accumulator,
orthographic camera presets, per-bird colour channels, alpha-accumulation density.

**Files:** `viz/renderer.py`, `viz/shaders.py`, `viz/camera.py`, `viz/hud.py`,
`viz/visualizer.py`, `capture/recorder.py`, `capture/mpl_recorder.py` (new),
`physics/flock.py` (position_history buffer).
**Tests:** `test/viz/test_renderer.py`, `test/viz/test_trails.py`, `test/viz/test_camera.py`,
`test/capture/test_recorder.py`, `test/analysis/test_perf.py`.

#### P8.1 — Sphere impostors + speed-stretched ellipsoid `[L2]`

**GLSL fragment:**
```glsl
vec2 p = v_uv * 2.0 - 1.0;
float r2 = dot(p, p);
if (r2 > 1.0) discard;
float z = sqrt(1.0 - r2);
float edge = smoothstep(1.0, 0.72, r2);
float shade = 0.55 + 0.45 * z;
vec3 color = mix(u_Paper, u_Ink, shade * (1.0 - edge * 0.22));
```
Ellipsoid: scale quad along `project(velocity)` by `1 + speed_ratio·0.3`.
Config: `viz.point_sprites: bool`.

**Test:** Centre pixel brighter than rim; corners = background.

**Citation:** `git6` R10, `sci/todo_claude_sci1.md` §6.

#### P8.2 — Depth cues + Fresnel rim `[L2]`

**Math.**
```
gl_PointSize ∝ 1/depth^k
alpha × mix(1, 1−depth01, fade) · mix(0.65, 1, speed01) · mix(1, 0.76, smoothstep(0.72,1,r²))
rim = pow(1 − N·V, k)
```

**Test:** Near bird renders larger & more opaque than far.

**Citation:** `git6` R10, `sci/todo_claude_sci9.md` §4.

#### P8.3 — Trail rendering ×4 `[L2]`

**File:** `viz/trails.py` (new), `physics/flock.py` (position_history).

**Velocity-stretched impostors:** stretch along `proj(p) − proj(p−v·len·0.12)` with head/tail radius.

**Screen-space accumulation:** fade quad at `clamp(0.24−persist·0.19−vis·0.09, 0.018, 0.32)`;
depth-only clear, then draw particles.

**Ring trails:** K past positions from rolling buffer `flock.position_history: (N, K, 3)` —
push current `positions` each frame, discard oldest. Render as shrinking/fading sprites.

**CPU trail lines:** 5 `LineSegments` per bird traced backward along velocity with
sinusoidal ribbon wave, `depthTest: false`.

**Config:** `viz.trails: "off" | "velocity" | "accumulation" | "ring" | "lines"`.

**Test:** Velocity trails extend along motion > perpendicular; accumulation persists
≈1/fadeOpacity frames; ring K sprites monotone size/alpha; lines cost <2ms at N=20K.

**Citation:** `git6` R10, `sci/todo_claude_sci2.md` §9.4, `sci/todo_claude_sci4.md` §8.

#### P8.4 — Winged flapping mesh + gradient sky `[L2]`

**Mesh geometry:** 6-triangle body+wings+tail. Flap: vertex.y += (±0.5)·flap_weight every
⌊frame/100⌋, applied to mesh-y **before** LookAt rotation (local-up flap).

**Gradient sky:** fullscreen quad, top (0.60,1,1)→bottom (0.686,0.933,0.933), theme-overridable.

**Citation:** `git5` R6–R7, `sci/todo_claude_sci3.md` §7.

#### P8.5 — Colour channels + theme wiring `[L2]`

Per-bird hue from `seeds` (HSV h=seed·360). Predator flag → red, ×1.3–1.5 scale.
Theme forwarding: `self._bird_prog["u_Paper"].value = THEME.paper` from `analysis/presets.py`.



**Theme material tables.** Each theme provides paired **ambient + diffuse** materials
for proper Blinn-Phong interaction (not just a single `u_Paper`/`u_Ink` uniform):
```python
THEMES = {
    "dark":  {"ambient": (0,0.2,0), "diffuse": (0,0.8,0), "paper": (0.95,0.95,0.90), "ink": (0.05,0.05,0.08)},
    "light": {"ambient": (0.3,0.3,0.3), "diffuse": (0.1,0.3,0.1), "paper": (0.98,0.97,0.95), "ink": (0.15,0.15,0.18)},
    # ... per sci/todo_claude_sci9.md §4 tables
}
```
Shader receives `u_Ambient` + `u_Diffuse` + `u_Paper` + `u_Ink` from the theme dict.
**Citation:** `git6` R10, `sci/todo_claude_sci4.md` §13.

#### P8.6 — Adaptive quality wired `[L2]`

**File:** `analysis/perf.py` (governor logic), `viz/visualizer.py` (consumer).

**Math.**
```
avg = 0.92·avg + 0.08·min(250, frame_ms);  budget = 1000/max(24, target_fps)
healthy if avg ≤ 1.12·budget
Ladder (fps < 78% target for ≥1.8s, one step per 1.8s):
    1. trails off  2. render scale −0.15 (floor 0.75)  3. N −18% (floor 512)
Recovery when avg ≤ 0.85·budget for 3.6s.
```

**Test:** Synthetic frame-times → actions fire in correct order, spaced ≥1.8s, recovery stops.

**Citation:** `git6` R11, `sci/todo_claude_sci1.md` §11.

#### P8.7 — Cinematic capture sweep `[L2]`

**File:** `capture/recorder.py`.

**Math.**
```
t = frame / total_frames
azim = 45° + t·180°;  elev = 25° + sin(t·2π)·0.15
dist = (650 + sin(t·1.5π)·100) · scale
```
Config: `capture.prewarm: 60`, `capture.sweep: True`.
Env overrides: `CAPTURE_W/H/FRAMES/OUT` (env > YAML, CLI > env).
GIF: `optimize=True, disposal=2`.

**Citation:** `todo_claude.md` E7–E10.

#### P8.8 — Dual-view + orthographic presets `[L2]`

Two `(camera, viewport)` passes: elev/azim 15°/15° + 45°/45°. Config: `viz.dual_view: False`.
Keys 7/8/9: ortho-top, ortho-side, perspective. Camera presets in `viz/camera.py`.

**Citation:** `sci/todo_claude_sci7.md` §5.

#### P8.9 — GPU-free matplotlib fallback `[L2]`

**File:** `capture/mpl_recorder.py` (new).

Dual-view scatter → GIF. Replaces silent `except: pass` frame loss. Warn on fallback activation.

**Citation:** `sci/todo_claude_sci7.md` §5.

#### P8.10 — Fixed-timestep accumulator + interpolation `[L2]`

**Math.**
```
acc += clamp(frame_dt, 0, 1/20)
while acc ≥ dt_phys: step(dt_phys); acc -= dt_phys
Render lerp: p_render = lerp(prev_positions, positions, acc/dt_phys)
```

**Test:** 30fps vs 60fps → identical physics after same elapsed time.

**Citation:** `sci/todo_claude_sci5.md` §11, `sci/todo_claude_sci9.md` §6.

#### P8.11 — Alpha-accumulation density mode `[L2]`

α ≈ 0.2 sprites, blending on, depth-write off (murmuratR aesthetic).

**Test:** Cluster centre darker than single bird.

**Citation:** `sci/todo_claude_sci6.md` §8.

---

**Phase 8 acceptance:** All four trail modes render. Impostors at 20K keep 60fps.
Capture GIFs depth-correct, pre-warmed, swept. Adaptive ladder fires under throttle.
Dual-view halves differ. Ortho presets show equal pixel sizes at different depths.
GPU-free fallback produces ≥1 GIF frame with warning.

**Architecture test:** New edges: `viz/shaders.py`, `viz/trails.py`, `viz/hud.py`,
`capture/mpl_recorder.py`. `viz/visualizer.py` holds engine reference but imports
no `simulation/` modules.

---

### Phase 9 — Metrics & Analysis

**Ships:** nematic order parameter S (Q-tensor, SO(3) invariant); MSD(τ) curve with crossover detection; hull-volume τρ; silhouette Θ′; shape→m*; η(m) marginal efficiency; robust gyration + ideal exponent −0.5; motion metrics (velocity deviation, boundary overshoot, normalized L); weighted composite rewards module; export schema (JSON round-trip)

**Subsystem:** F1 | **Level:** L1 → L2 | **Est. effort:** 3 days

**Depends on:** P2. **Produces:** nematic order parameter, MSD(τ) curve with crossover,
hull-volume density τρ, silhouette Θ′, shape→m*, H₂ disconnected→inf (P0.13 verified),
η(m) marginal efficiency, robust gyration + number density + ideal exponent,
motion metrics (velocity deviation, boundary overshoot, normalized L),
rewards module, export schema.

**Files:** `analysis/metrics.py`, `analysis/rewards.py`, `analysis/phase_diagram.py`,
`analysis/density_scaling.py`.
**Tests:** `test/analysis/test_metrics.py`, `test/analysis/test_metrics_invariance.py`,
`test/analysis/test_rewards.py`.

#### P9.1 — Nematic order parameter `[L1]`

**Math.**
```
Q_αβ = (1/N) Σ_i ((3/2)·û_i^α·û_i^β − (1/2)·δ_αβ)     [3×3 traceless]
S = λ_max(Q) ∈ [0,1]
```
Polar α = |Σ û_i|/N. `order: polar|nematic` option in phase-diagram sweep.

**Test:** Two anti-parallel half-flocks → α<0.05, S>0.95; isotropic 500 birds → both<0.15;
S invariant under û→−û and SO(3).

**Citation:** `git2` R7, `sci/todo_claude_sci8.md` §4.

#### P9.2 — MSD(τ) curve `[L1]`

**Math.**
```
p_unwrap(t) = p_unwrap(t−1) + min_image(p(t) − p(t−1))
MSD[l] = (1/(T−l))·Σ_t ‖p_unwrap(t+l) − p_unwrap(t)‖²
```
Log-spaced lags {1,2,4,…,64}. Slope: ballistic≈2, diffusive≈1, crossover τ_cross.

**Test:** D=0 aligned flight → slope 2.0±0.1; strong-noise walkers → 1.0±0.2 for τ≥4;
seam crossing contributes MSD(1)=(v·dt)²±1e-4.

**Citation:** `git2` R8, `sci/todo_claude_sci8.md` §4.

#### P9.3 — Hull-volume τρ `[L1]`

**Math.**
```
ρ(t) = N / ConvexHull(positions).volume    [0 if degenerate]
τ = interval · (0.5 + Σ_{lag≥1} r(lag))     [stop at first r(lag) ≤ 0]
```
Ring buffer: sample every 10 frames, 500 slots.

**Test:** Cube hull = edge³±1e-3; coplanar→0; constant series→τ=0; period-P→τ∈[P/6,P].

**Citation:** `todo_claude.md` §11.

#### P9.4 — Silhouette Θ′ `[L1]`

**Math.** Project positions ⊥ observer axis, rasterize disks of radius `boid_size`,
coverage = union fraction (overlaps count once). Additional field beside voxel Θ′.

**Test:** Flat wall ⊥ axis → silhouette≈1 while voxel Θ′≪1; two co-projected birds = one.

**Citation:** `todo_claude.md` §12.

#### P9.5 — Shape→m* `[L1]`

**Math.**
```
m* = 9.78 + clamp((aspect−1)/2, 0, 1)·(6.05 − 9.78)
```
`suggested_m` field on `FlockMetrics`.

**Test:** aspect 1→9.78, ≥3→6.05; monotone; thin flock≤7, round≥8.

**Citation:** `todo_claude.md` §9.

#### P9.6 — η(m) marginal efficiency `[L1]`

**Math.** `η(m) = (H₂(m₀)−H₂(m))/(m−m₀)`, +∞ when m first connects graph, 0.0 when both
disconnected (requires P0.13's inf fix).

**Test:** Connectivity transition → `math.isinf`; both disconnected → 0.0.

**Citation:** `todo_claude.md` §10.

#### P9.7 — Robust gyration + ideal exponent `[L1]`

**Math.** **Median** centroid; one-sided top-15% trim; `R_g = √mean(r²_kept)`;
`ρ = N_kept / ((4/3)πR_g³)`; density-scaling sweep reports `ideal_density_exponent = −0.5`
beside fitted β.

**Test:** One 10K-unit outlier moves R_g <5%; degenerate flock density→0; sweep carries −0.5.

**Citation:** `todo_claude.md` §13.

#### P9.8 — Motion metrics `[L1]`

**Math.**
```
velocity_deviation = (1/N)Σ‖v̄ − v_i‖    [catches speed dispersion α misses]
boundary_overshoot = Σ max(0, ‖p−C‖ − R_dom)
normalized_angular_momentum = ‖⟨r×v⟩‖ / (v0·R_g)    [O(1), about CoM]

altitude_deviation = (1/N)·Σ |z_i − z_target|    [roosting altitude error; z_target from roost config]
```

**Test:** Equal headings + mixed speeds → deviation>0 while α=1; overshoot 0 inside,>0 outside;
normalized L O(1) across ×10 domain scale.

**Citation:** `todo_claude.md` §15, `sci/todo_claude_sci7.md` §2.

#### P9.9 — Rewards module `[L1]`

**File:** `analysis/rewards.py`.

Weighted composite over named metric terms; `reward_faithful_signs` flag;
shared by MARL (P12) and EvoFlock scalarization (P11).

**Test:** Perfect flock → corrected reward 0 (max); faithful flag flips alignment sign;
per-term weight linearity.

**Citation:** `sci/todo_claude_sci7.md` §2.

#### P9.10 — Export schema `[L1]`

**File:** `analysis/metrics.py::FlockMetrics.to_dict()`.

`to_dict()` adopted end-to-end — ndarray→list, numpy scalars→python.
New fields included: `suggested_m, nematic, msd_curve, target_dist_*, *_real`.

**Test:** JSON round-trip; pinned key set; Recorder CSV headers == schema.

**Citation:** `roadmap.md` D9.2.

---

**Phase 9 acceptance:** Nematic S distinguishes anti-parallel from isotropic.
MSD slope crossover detected. τρ constant=0, period-P bounded. Silhouette ≠ voxel Θ′.
Robust gyration trims outliers. Velocity deviation catches speed dispersion α misses.
Rewards linear in weights. Export schema JSON round-trips.

**Architecture test:** New edges: `analysis/rewards.py → core + physics/flock(read)`,
`analysis/phase_diagram.py → core + physics/flock(read)`,
`analysis/density_scaling.py → core + physics/flock(read)`.

---

### Phase 10 — UX & Tooling

**Ships:** 8 preset keys (a–h,w); full window-title readout; 5-slider HUD with drag-lock; cursor-ray spawning (bird + predator); CLI --set/--print-config/--fullscreen; pymurmur facade (Simulation, benchmark); φp+φa ≤ 1 constraint

**Subsystem:** A, C | **Level:** L2 → L3 | **Est. effort:** 2 days

**Depends on:** P2 (nested config). P10.3 (slider HUD) additionally requires P4 (spatial mode).
**Produces:** preset keys a–h,w, full window-title readout, slider HUD (5 knobs),
cursor-ray spawning (birds + predators), CLI (--set, --print-config, --fullscreen),
package facade (`pymurmur.Simulation`), φp+φa≤1 constraint, benchmark API.

**Files:** `analysis/presets.py`, `viz/input_control.py`, `viz/hud.py`,
`viz/visualizer.py`, `pymurmur/__main__.py`, `pymurmur/__init__.py`.
**Tests:** `test/viz/test_input.py`, `test/test_cli.py`, `test/test_facade.py`.

#### P10.1 — Preset keys a–h,w `[L2]`

8 presets wired to keys (a: 0.04/0.80/6/proj · b: 0.18/0.70/7/proj ·
c: 0.06/0.45/3/proj · d: 0.25/0.55/8/spatial · e: 0.10/0.75/6/proj ·
f: 0.02/0.85/3/proj · w: 0.08/0.82/10/spatial · h: 0.35/0.58/9/spatial),
each prints label + description. Key `g` skipped (grid toggle).

**Citation:** `todo_claude.md` E4.

#### P10.2 — Full title readout `[L2]`

Title: `mode | N | φp/φa/σ | α Θ Θ′ L σr | τρ | FPS` (+ physical units).
Rebuilt every 20th frame. Via `FlockMetrics.summary()`.

**Citation:** `todo_claude.md` E6.

#### P10.3 — Slider HUD `[L2]`

5 sliders: sep 1–5→`spatial.separation_weight`; coh 0–2; align 0–0.5;
avoid 0–1→`boundary.avoidance_factor`; noise 0–0.5. Ortho-pass track+knob quads.
Drag locks (suppresses orbit); TAB toggles panel.

**Citation:** `sci/todo_claude_sci3.md` §6.

#### P10.4 — Cursor-ray spawning `[L2]`

Mouse spawn via cursor-ray unprojection: left-click→bird, right-click→predator.
`C` clear, `Q` quit, PageUp/Dn: `flock.v0 ± 0.1` (floor 0.3).

**Citation:** `sci/todo_claude_sci5.md` §7.

#### P10.5 — CLI + facade `[L3]`

`--set key.subkey=value` (repeatable, nested schema) + `--print-config` + `--fullscreen`.
`pymurmur.Simulation(**params)` + `benchmark(flock_size, num_steps)→list[float]`.
`pymurmur/__init__.py` exports `SimConfig, SimulationEngine, Simulation, Recorder`.

**Test:** `--set spatial.separation_weight=6 --set flock.num_boids=500` reflected in
`--print-config`; unknown key exits with field list; facade benchmark returns 20 positive floats.

**Citation:** `sci/todo_claude_sci5.md` §6–8.

#### P10.6 — φp+φa ≤ 1 constraint `[L2]`

After φp/φa increments: `if φp+φa > 1.0: other = 1.0 − changed`. Input handler enforces.

**Citation:** `todo_claude.md` E5.

---

**Phase 10 acceptance:** Presets a–h,w apply with printed descriptions.
Title contains all tokens at correct cadence. Sliders write nested config fields.
Cursor spawn via ray unprojection. CLI --set + --print-config works. Facade benchmark API works.
φp+φa ≤ 1 always.

**Architecture test:** New edges: `pymurmur/__init__.py → core + simulation + capture`,
`__main__.py → everything`. `viz/input_control → analysis/presets`.

---

### Phase 11 — EvoFlock

**Ships:** SSGA (worst-of-3 negative selection, uniform crossover, fitness cache); worst-of-4 evaluation; 4 objectives (separation trapezoid, speed band, curvature, hypervolume); SDF obstacle scene with collision detection + kinematic correction; expanded gene set (12+ evolvable parameters); emergent-alignment experiment

**Subsystem:** F2 | **Level:** L2 | **Est. effort:** 3 days

**Depends on:** P2, P4 (spatial mode), P0.14 (SDF primitives).
**Produces:** SSGA with uniform crossover + worst-of-3 negative selection,
worst-of-4 evaluation, 4 objectives (separation trapezoid, speed band, curvature,
hypervolume), SDF obstacle scene (sphere/box/cylinder CSG), collision detection +
kinematic correction, expanded gene set (forward force, perception cones, fly-away,
k-neighbours, σ integer, speed_min_factor), best-genome persistence.

**Files:** `analysis/evoflock.py`, `physics/obstacles.py` (ObstacleScene builder).
**Tests:** `test/analysis/test_evoflock.py`.
**Presets:** `conf/murmuration_evo.yaml`, `conf/evo_open.yaml`.

#### P11.1 — SSGA fidelity + uniform crossover `[L2]`

**Algorithm.** Per update: select 3 → evaluate all 3 (fitness cache keyed on genome) →
sort → delete worst of 3 (negative selection) → uniform crossover of best two
(each gene from random parent) + per-gene Gaussian mutation → insert in freed slot.
Founders evaluated.

**Test:** Worst-of-3 gone; child mixes genes from both parents (disjoint-value parents);
all three finite fitness; cache prevents re-simulation.

**Citation:** `sci/todo_claude_git0.md` W8.

#### P11.2 — Worst-of-4 evaluation `[L2]`

4 sims per candidate, fixed per-sim seeds, min-reduction (`eval_parallel` live;
deterministic order).

**Test:** Monkeypatched objectives [0.9,0.8,0.95,0.7] → fitness 0.7; seeds recorded.

**Citation:** `sci/todo_claude_git0.md` W8.

#### P11.3 — Objectives `[L2]`

**Math.**
- **Separation:** trapezoid over body diameters (0 below 2, ramp 2→2.5, plateau≤4, ramp 4→5, 0 above)
  on nearest-neighbour distance per boid-step.
- **Speed:** band [19,21] m/s on `speed_real`, ramps [18,22].
- **Curvature:** `κ = |v×a|/|v|³`, `score = clamp(0.8+(κ_avg/0.1)·0.2, 0.8, 1.0)`.
- **Hypervolume:** `F = Π max(o_k, 0.01)`.

**Test:** Trapezoid pinned at d/body∈{1.9→0, 2.5→1, 4→1, 5→0}; helix κ matches analytic ±2%.

**Citation:** `sci/todo_claude_git0.md` W8.

#### P11.4 — SDF obstacle layer `[L1→L2]`

**File:** `physics/obstacles.py::ObstacleScene`.

Composes P0.14 primitives into a CSG scene tree. Collision when
`sign(SDF(p_old)) ≠ sign(SDF(p_new))`. Kinematic correction to surface.
Per-step collision counter feeds `(f_cf)^500` as an objective term.

**Test:** Obstacle course: collisions>0 with zero avoidance, ≈0 with evolved weights (`@slow`).

**Citation:** `sci/todo_claude_git0.md` W8.

#### P11.5 — Expanded gene set `[L2]`

Additional evolvable parameters: `w_fwd` (forward force), `max_dist_{sep,align,coh}`,
`angle_{sep,align,coh}` (perception cones, cos α∈[−1,1]), `fly_away_max_dist`,
`min_time_to_collide` (predictive avoidance), fixed k=7 topological neighbours,
integer gene for σ, `flock.speed_min_factor` as a gene.

**Test:** Forward force sign flips around v*; cones exclude behind-cone birds;
k enforced; σ integer after decode.

**Citation:** `sci/todo_claude_git0.md` W8.

#### P11.6 — Protocol `[L2]`

Persist best genome + Pareto front + per-run seeds + objective scores to
`output/evolved.yaml`. Ship confined (enclosure+obstacles) and open (`boundary: open`)
evaluation configs. **Experiment (`@slow`):** evolve with NO alignment objective
on the confined config → best genome's settled α > 0.5 (emergent alignment).

**Citation:** `sci/todo_claude_git0.md` W8.

---

**Phase 11 acceptance:** SSGA worst-of-3 deleted. Child mixes parent genes.
Worst-of-4 min-reduction correct. Objectives trapzoid/speed/curvature correct.
SDF collision detection works. Evolved weights reduce collisions to ≈0.
**Experiment:** emergent alignment α>0.5.

**Architecture test:** New edges: `analysis/evoflock → simulation + core` (F2 tier).
`physics/obstacles::ObstacleScene → core + physics/obstacles (primitives)`.

---

### Phase 12 — MARL Bridge

**Ships:** standalone 'marl' force mode (deferred global rules); MurmurationEnv gymnasium wrapper (6N obs, 3N action); dependency-gated PPO training + rollout scripts

**Subsystem:** E, F2 | **Level:** L2 | **Est. effort:** 2 days

**Depends on:** P2 (engine control hook), P9 (rewards module).
**Produces:** `"marl"` force mode (deferred global rules under external control),
Gymnasium wrapper (`MurmurationEnv`), dependency-gated training + rollout scripts.

**Files:** `physics/forces/marl.py` (new), `analysis/gym_env.py` (new),
`scripts/train_marl.py`, `scripts/rollout_marl.py`.
**Tests:** `test/analysis/test_marl.py` (`pytest.importorskip("gymnasium")`).

**Unit map:** `U = min(W,H,D)/6`, `v_cap = marl.velocity_cap·U`.

#### P12.1 — "marl" mode: deferred global rules `[L1]`

**Engine order for this mode:** control applies first
(`v += a_ext·action_scale·v_cap`, component clip ±v_cap), **move**, then rules prep
the *next* step: `v += rule_weight·(F_sep(d<separation_radius·U) + (v̄−v) + (CoM−p))`
with `rule_weight=0.01` (global neighbourhood — no radius on align/cohere).

**Test:** Two-step hand trace: positions at step k depend on rules from k−1 only;
0.01 scaling; clip bounds.

**Citation:** `sci/todo_claude_sci7.md` §1.

#### P12.2 — Gymnasium wrapper `[L2]`

**File:** `analysis/gym_env.py::MurmurationEnv`.

```python
observation_space = Box(−1, 1, (6N,))  # concat((p−C)/3U, v/v_cap)
action_space = Box(−1, 1, (3N,))
```
Lazy import (gymnasium optional). Seeded reset. Truncate at `marl.episode_steps` (500).
Reward from P9.9 rewards module.

**Test:** `gymnasium.utils.env_checker.check_env` passes; obs∈[−1,1] over 500 random steps;
same seed+actions → identical obs; truncation at 500.

**Citation:** `sci/todo_claude_sci7.md` §1.

#### P12.3 — Scripts (dependency-gated) `[L3]`

`train_marl.py`: PPO("MlpPolicy"), 5000 timesteps, save `output/marl_ppo`.
`rollout_marl.py`: 500 deterministic-predict steps → dual-view GIF.
Docstring notes centralized-MLP quadratic scaling and points to IPPO for large N.

**Test (`@slow`, skip without sbl3):** 200-timestep learn() smoke; rollout GIF ≥1 frame;
**experiment:** trained policy's mean dispersion < random policy's by ≥20%.

**Citation:** `sci/todo_claude_sci7.md` §A.

---

**Phase 12 acceptance:** MARL mode defers rules to next step. Gymnasium checker passes.
Trained policy beats random on cohesion. Scripts dependency-gated.

**Architecture test:** New edges: `physics/forces/marl.py → core + physics/flock(read)`,
`analysis/gym_env.py → simulation + core` (F2 tier).
**New files:** `physics/forces/marl.py`, `analysis/gym_env.py`, `scripts/train_marl.py`,
`scripts/rollout_marl.py`.

---

### Phase 13 — Scaling & Performance

**Ships:** per-mode step-time regression suite; 5 scaling checkpoints (150→300K birds); memory audit (≤25 MB at 300K); 24-hour soak test; determinism matrix (mode × threads × jitter × numba)

**Subsystem:** B, F2 | **Level:** L2 → L3 | **Est. effort:** 2 days

**Depends on:** P3–P8 (all modes landed), P10 (benchmark API).
**Produces:** per-mode step-time regression suite, scaling checkpoints
(150→300K birds), memory audit, 24-hour soak test, golden consistency
across num_threads/numba/jitter.

**Files:** `test/test_performance.py` (extend), `test/test_scaling.py` (new).

#### P13.1 — Step-time regression suite `[L2]`

**Test:** `@slow` per-mode step-time ≤ budget × headroom at N=2000.
Arch.md scaling checkpoints: N=150 hash grid 60fps, N=1500 SoA 60fps,
N=16000 cKDTree 60fps, N=50000 numba 45fps, N=300000 full 30fps.

**Citation:** `arch.md` §13.

#### P13.2 — Scaling checkpoints `[L2]`

**Test (`@slow`):** `benchmark(flock_size, 100)` at each checkpoint →
per-step time ≤ target (60fps→16.7ms, 45fps→22.2ms, 30fps→33.3ms).

**Citation:** `arch.md` §13.

#### P13.3 — Memory audit `[L2]`

**Test:** `sys.getsizeof` on SoA arrays at N=300K → total ≤25 MB
(positions+velocities+accelerations+prev+last_accel+seeds+max_speed+active+is_predator
+index+instance buffer).

**Citation:** `arch.md` §8.

#### P13.4 — 24-hour soak test `[L2]`

**Test (`@slow`):** 24h of headless steps → no NaN, no memory growth trend,
positions in-bounds, speed within contract.

**Citation:** `sci/todo_claude_sci2.md` §21.

#### P13.5 — Determinism matrix `[L2]`

**Test:** Same seed → bit-identical after 100 steps for all
(mode × num_threads∈{1,−1} × jitter on/off × numba on/off with fastmath off).
Two in-process runs + one subprocess run.

**Citation:** `roadmap.md` T4.3.

---

**Phase 13 acceptance:** All mode budgets within headroom. Scaling checkpoints pass.
Memory ≤25 MB at 300K. 24h soak clean. Determinism matrix green.

**Architecture test:** Test-only phase — no new production edges.

---

### Phase 14 — Guard Rails

**Ships:** full architecture DAG enforcement (ALLOWED_EDGES = arch.md §5); config-usage drift detection; strictly-3D AST guard; doc-link bidirectional sync; collection-count guard; CI summary gate (blocks merge on any guard failure)

**Subsystem:** A | **Level:** L3 | **Est. effort:** 1.5 days

**Depends on:** P0–P13. **Produces:** full architecture DAG enforcement,
config-usage drift detection, strictly-3D guard, doc-link test, collection-count guard.

**Files:** `test/test_architecture.py` (final matrix), `test/test_docs.py`.

#### P14.1 — DAG matrix finalization `[L3]`

`ALLOWED_EDGES` matches arch.md §5 exactly. AST-walk every `.py` file
(function-level imports included). Fail on any edge outside the matrix.
Named regression edges: `physics.flock !→ physics.forces`,
`viz.input_control !→ simulation`, no `cKDTree(` in `forces/`,
no module-level `np.random.*`.

**Citation:** `roadmap.md` T1.1.

#### P14.2 — Config-usage drift `[L3]`

Every `SimConfig` leaf field (recursed) must be read by ≥1 non-config module
(AST attribute-access scan). Fail with orphan list.

**Citation:** `roadmap.md` T1.2.

#### P14.3 — Strictly-3D guard `[L3]`

AST scan for `(…, 2)`-shaped spatial arrays in `physics/` → fail.
`validate()` enforces `depth > 0`. Invariance tests use random SO(3), not z-only.

**Citation:** `roadmap.md` T1.3.

#### P14.4 — Doc-drift test `[L3]`

**File:** `test/test_docs.py` (shipped and operational).

Every module path named in `arch.md` exists. Every intra-repo markdown link
in `arch.md` and `roadmap_deepseek.md` resolves. The arch.md force-mode table
lists exactly `sorted(MODE_REGISTRY)`. Also verifies that `arch.md` references
`roadmap_deepseek.md` with the P0-P14 phase scheme and that the retired
D0-D9 / S1-S7 / T0-T6 scheme is absent.

**Test** (`test/test_docs.py`, 4 tests passing):
```python
def test_arch_md_links_resolve():
    # Every intra-repo link in arch.md points to an existing target
def test_arch_md_references_roadmap():
    # arch.md links to roadmap_deepseek.md with P0-P14, no stale D0-D9/S1-S7/T0-T6
def test_roadmap_md_links_resolve():
    # Every intra-repo link in roadmap_deepseek.md points to an existing target
def test_roadmap_references_arch():
    # roadmap_deepseek.md links back to arch.md (bidirectional sync)
```

**CI:** Enforced by `.github/workflows/guard-rails.yml` job `guard-rail-doc-links`.

**Citation:** `roadmap.md` T1.4.

#### P14.5 — Collection-count guard `[L3]`

Collected-test count pinned per subpackage (update deliberately).

**Citation:** `roadmap.md` T1.5.

---

**Phase 14 acceptance:** Architecture test enforces full matrix. Zero orphan config fields.
No (…,2) arrays in physics/. All doc links resolve. Collection counts pinned.

---

## Summary Table

| Phase | Subsystem(s) | Levels | Days | Key deliverable |
|---|---|---|---|---|
| P0 | A, B, E, F1 | L0→L3 | 3 | Golden suite, RNG, flock state, math helpers, SDF, position init |
| P1 | E, F1 | L0 | 3 | True occlusion, prob-union Θ, steric, force fixes, vicsek memory |
| P2 | A, B, C, E | L1→L2 | 5.5 | Nested config, ForceMode registry, ForceTerm composeForces, index protocol, engine DAG |
| P3 | E | L1 | 5.5 | Field/blob 13 terms + Threat FSM + 7 presets, grid sep norm, 1.45·R floating boundary |
| P4 | E, F1 | L1 | 3.5 | Reynolds hybrid filter, predator boids, ecology, numba |
| P5 | E | L1 | 2 | Angle mode: steering, adaptive speed, incremental grid |
| P6 | E | L1 | 1.5 | Vicsek predator-prey, asymmetric collisions |
| P7 | E | L1 | 1.5 | Influencer: Lissajous, move-then-steer, density init, pilot mode |
| P8 | C, D | L2 | 5 | Impostors, 4 trail modes, winged mesh, capture, adaptive quality |
| P9 | F1 | L1→L2 | 3 | Nematic, MSD, hull-τρ, silhouette, rewards, export schema |
| P10 | A, C | L2→L3 | 2 | Presets, slider HUD, CLI, facade, benchmark API |
| P11 | F2 | L2 | 3 | EvoFlock SSGA, SDF obstacles, emergent alignment |
| P12 | E, F2 | L2 | 2 | MARL mode, gym wrapper, PPO scripts |
| P13 | B, F2 | L2→L3 | 2 | Scaling checkpoints, memory audit, soak, determinism |
| P14 | A | L3 | 1.5 | DAG enforcement, config drift, 3D guard, doc links |

**Total:** ≈42 working days single-track. With two parallel streams
(physics tracks P3–P7 parallel with rendering P8 + metrics P9)
≈ **6–7 calendar weeks.**

**Highest value-per-day cut:** P0 → P1 → P2 → P3 → P8.1–8.4 + P8.7.

**Definition of done:** P14.1 matrix green; P14.2 zero orphan fields;
P13.5 determinism matrix green; golden set covers all 7 modes;
full non-GL suite runs headless in CI; two `@slow` experiments
(P11.6 emergent alignment, P12.3 trained-beats-random) pass nightly.

---

## Appendix A — Excluded Scope

Screensaver + desktop-overlay modes; GPU-compute simulation backends;
Hildenbrandt–Hemelrijk flight physics; CMA-ES benchmark and GP model evolution;
VR/XR. Each stays documented in its source spec


**Reconsidered — multi-flock support.** The predator-prey dynamics in P4.3 and
P6 use a species column within a single `PhysicsFlock`. This works but is fragile
(one array, shared index, shared rng). A cleaner future design: allow the engine
to hold multiple independent `PhysicsFlock` instances sharing one domain, with
per-flock ForceMode, per-flock metrics, and separate instance buffers. This is
the natural architecture for true multi-species, multi-flock parallax, and
separately-configured prey+predator populations. Not in the current phase plan
but documented here as a known upgrade path. See `sci/todo_claude_sci4.md` §9.
under `sci/` if scope changes.

---

## Appendix B — Implementation Conventions

- **Golden re-pinning.** Any deliberate physics change re-pins goldens in the
  same commit. CI fails if goldens are stale.
- **Test mirroring.** Code in `pymurmur/<sub>/module.py` → tests in
  `test/<sub>/test_module.py`.
- **Markers.** `@pytest.mark.gl` (skip without GL context), `@pytest.mark.slow`,
  `@pytest.mark.golden`.
- **Dependency-gated imports.** `gymnasium`, `stable-baselines3`, `numba` are
  optional — lazy import with `pytest.importorskip` in tests.
- **Commit strategy.** One commit per step where practical. Every commit leaves
  `pytest test/ -m "not slow"` green.

---

## Appendix C — Cross-Cutting Concerns

### CC1 — Error Handling

- All YAML load failures produce actionable messages with the offending key/section.
- GPU context loss triggers graceful fallback to headless, not a raw crash.
- NaN positions self-heal to `flock.center` (P0.10).
- `dt` clamped to [0, 1/20] behind fixed-timestep accumulator (P8.10).

### CC2 — Logging

- Config validation warnings via `warnings.warn` (P2.1).
- Capability report at startup: moderngl/numba/scipy versions (P0.11).
- GPU-free capture activates with an explicit warning (P8.9).
- Fastmath + metrics export raises Warning (P4.10).

### CC3 — Documentation Sync

- P14.4 doc-drift test enforced. arch.md module map, force-mode table, and
dependency matrix kept in sync with code at every phase boundary.

### CC4 — Migration Strategy

- P2.1: Property shims on SimConfig for backward compat → dotted access → shim retirement.
- P2.2: Lazy imports in forces/ moved to module top (no per-bird import calls).
- P2.5: Dead `use_toroidal_distance` field replaced by `boundary.use_toroidal_distance`.
- Dead code (FlockArrays compose, ForceKernel delete, BoidView delete) in P2.

### CC5 — Risk Register

| Risk | Mitigation | Phase |
|---|---|---|
| Golden re-pin forgotten | CI fails on stale goldens | P0 |
| Import cycle resurfaces | Architecture test forbids `physics/flock→forces` | P2 |
| Config key collision | Nested dataclasses make collision structurally impossible | P2 |
| Determinism broken by new feature | Determinism matrix (mode×threads×jitter×numba) | P13 |
| Performance regression | Step-time budgets at scaling checkpoints | P13 |
| 2D assumption creeps in | AST scan for (…,2) arrays; depth>0 validator | P14 |

## Appendix D — Glossary

Terms used throughout this roadmap, with the definition that applies in the
context of a 3D murmuration simulation. Grouped by domain.

### Mathematics & Physics

| Term | Definition |
|---|---|
| **SoA** (Structure of Arrays) | Data layout where each property is a contiguous `(N,)` or `(N,3)` array — all positions together, all velocities together — rather than an array of per-bird structs. Enables vectorised numpy/numba operations. |
| **SDF** (Signed Distance Function) | `sdf(p)`: negative inside the shape, zero on the surface, positive outside. Used for obstacle collision detection (`physics/obstacles.py`). |
| **CSG** (Constructive Solid Geometry) | Building complex shapes by combining SDF primitives with `min` (union) and `max` (subtraction) operators. Powers `ObstacleScene` in P11.4. |
| **Rodrigues rotation** | `rotate_about(v, k, angle)` — rotates vector `v` around axis `k` by `angle` radians. The core steering mechanism in angle mode (P5.1). |
| **Lissajous** | A parametric curve built from multiple sine/cosine terms at incommensurate frequencies. Used for blob anchor trajectories (P3.2), wander paths (P3.1), and influencer targets (P7.1). |
| **min-image** | Per-axis toroidal distance: `Δx − W·round(Δx/W)`. The shortest vector between two points across a periodic boundary, used wherever toroidal domains are active. |
| **Θ** (theta, internal opacity) | The fraction of the visual sphere occluded by nearer neighbours — a probabilistic union of solid angles. `Θ ∈ [0,1]`. Drives density-regulation in projection mode. |
| **δ̂** (delta-hat) | Boundary-length-weighted mean of visible-neighbour directions. `|δ̂| ∈ [0,1]`: →1 at the flock edge, →0 deep inside. |
| **Ω** (solid angle) | The angular area a neighbour subtends on the observer's visual sphere, in steradians. `Ω = 2π(1 − cos α)`. |
| **Q-tensor** | A 3×3 traceless symmetric matrix `Q_{αβ} = (1/N) Σ (3/2·û^α·û^β − 1/2·δ_{αβ})`. Its largest eigenvalue `S` is the **nematic order parameter** — distinguishes anti-parallel from isotropic. |
| **MSD(τ)** (Mean Squared Displacement) | `MSD[l] = (1/(T−l))·Σ_t ‖p(t+l) − p(t)‖²` over log-spaced lags. Slope: 2 = ballistic, 1 = diffusive. Requires unwrapped positions via min-image. |
| **hull-τρ** | Convex-hull volume density `ρ = N / Volume(ConvexHull)` with autocorrelation time `τ`. Constant series → τ=0; periodic → τ bounded. |
| **H₂** (Horvitz-Thompson) | A manifold-dimension estimator over the k-NN graph. Returns `inf` when the graph is disconnected (P0.13). |
| **η(m)** (marginal efficiency) | `η(m) = (H₂(m₀)−H₂(m))/(m−m₀)` — the rate at which increasing neighbourhood size improves the dimension estimate. |
| **α** (polar order) | `α = ‖Σ û_i‖ / N ∈ [0,1]`. Measures velocity alignment: 1 = perfect parallel, 0 = isotropic. Cannot distinguish anti-parallel from isotropic — use `S` for that. |
| **S** (nematic order) | The largest eigenvalue of the Q-tensor. `S ∈ [0,1]`. 1 = all axes parallel (including anti-parallel), 0 = isotropic. Invariant under `û → −û` and SO(3) rotations. |
| **Θ′** (theta-prime, silhouette opacity) | Projection of positions onto a 2D observer plane, with disks of radius `boid_size` rasterised and union-counted. An external-view analogue of Θ. |
| **R_g** (gyration radius) | `R_g = √(median_trimmed_mean(r²))`. Top-15% tail trim for robustness against outliers. |
| **CoM** (Center of Mass) | `(1/N) Σ p_i` — the instantaneous centroid of active bird positions. Not to be confused with `flock.center`, which is an exponentially smoothed version. |
| **m*** (suggested m) | `m* = 9.78 + clamp((aspect−1)/2, 0, 1)·(6.05−9.78)` — a shape-driven recommendation for the optimal k-NN neighbourhood size. |

### Rendering & Graphics

| Term | Definition |
|---|---|
| **FBO** (Framebuffer Object) | An off-screen render target in OpenGL. Used for headless capture — render to FBO, then read pixels back to CPU for GIF assembly. Must have a **depth attachment** for correct z-ordering. |
| **VAO** (Vertex Array Object) | Container for GPU buffer state (VBO bindings, attribute layouts). Must be **rebuilt** after any buffer growth (instance count increase) to avoid stale attribute pointers. Enforced by P2.7. |
| **VBO** (Vertex Buffer Object) | A GPU-side array of vertex or instance data. Instance VBOs hold `(pos.xyz, vel.xyz, flag, hue)` = 8 floats per bird. |
| **ModernGL** | A lightweight Python OpenGL binding (no GLUT/freeglut dependency). The project's only GPU library — `viz/` depends on it; all other packages do not. |
| **Blinn-Phong** | A per-pixel lighting model: ambient + diffuse·(N·L) + specular·(N·H)^k. The theme system provides `u_Ambient` + `u_Diffuse` + `u_Paper` + `u_Ink` uniforms. |
| **GLSL** (OpenGL Shading Language) | C-like GPU code in `viz/shaders.py`. Vertex shader: instance→clip-space transforms + flap animation. Fragment shader: impostor disc, depth cues, Fresnel rim. |
| **Impostor** | A camera-facing quad (two triangles) rendered per bird, with a fragment-shader disc that looks like a sphere. Cheaper than a full mesh — key to rendering 20K+ birds at 60 fps. |
| **Speed-stretched ellipsoid** | An impostor quad scaled along the projected velocity direction by `1 + speed_ratio·0.3`, giving the illusion of motion blur without a separate pass. |
| **Fresnel rim** | Edge-lighting effect: `rim = pow(1 − N·V, k)`. Birds appear rim-lit at grazing angles, improving depth perception. |
| **Depth cues** | `gl_PointSize ∝ 1/depth^k` + `alpha × mix(1, 1−depth01, fade)`. Near birds render larger and more opaque than far birds. |
| **Alpha-accumulation** | A rendering mode where sprites use `α ≈ 0.2`, blending on, depth-write off. Dense regions appear darker (cumulative absorption) — the murmuratR aesthetic. |
| **Trail modes** | Four rendering techniques: **velocity**-stretched impostors, screen-space **accumulation**, **ring** trails from `position_history` rolling buffer, and CPU **lines** (5 ribbon segments). |
| **Dual-view** | Two `(camera, viewport)` render passes in one frame — typically elev/azim 15°/15° and 45°/45°. Useful for capture sweeps and debugging. |
| **Xvfb** (X Virtual Framebuffer) | A headless X11 server used in CI to provide a GL context without a physical display. Combined with Mesa llvmpipe for software rendering. |

### Algorithms & Protocols

| Term | Definition |
|---|---|
| **SSGA** (Steady-State Genetic Algorithm) | An evolutionary algorithm where each update: selects 3 genomes → evaluates all 3 (fitness cache) → deletes the worst → uniform crossover of the best two + Gaussian mutation → inserts in freed slot. No generations — continuous turnover. |
| **EMA** (Exponential Moving Average) | `avg = 0.92·avg + 0.08·min(250, frame_ms)`. A low-pass filter on frame times, used by `QualityGovernor` to detect sustained performance degradation. |
| **FSM** (Finite State Machine) | The threat predator's behavioural controller: **approach** (close on flock), **egress** (exit after pass-through), with `capture` and `clear` distance gates. |
| **cKDTree** | scipy's k-d tree implementation. The primary spatial index for N ≥ 5000. `boxsize=(W,H,D)` enables correct toroidal queries. |
| **SpatialHashGrid** | A uniform 3D cell grid with modulo-wrapped cell keys. O(N) rebuild, O(1) query. Used for N < 5000. Ghost-cell replication handles toroidal boundary queries. |
| **k-NN** (k-Nearest Neighbours) | The `k` closest birds to a query point, used in topological neighbourhood modes (projection σ, spatial influence_count). |
| **PPO** (Proximal Policy Optimization) | A policy-gradient RL algorithm used in `scripts/train_marl.py`. External to pymurmur — the gym environment only provides observations and accepts actions. |
| **IPPO** (Independent PPO) | Decentralised PPO where each bird is an independent learner sharing parameters. Scales better than centralised PPO for large N. |
| **Hypervolume** | `F = Π max(o_k, 0.01)` — the product of all objective scores. A multi-objective quality metric that rewards balanced improvement. Used in EvoFlock. |
| **Pareto front** | The set of non-dominated solutions in multi-objective optimisation. Persisted to `output/evolved.yaml` alongside the best single genome. |

### Project-Specific Concepts

| Term | Definition |
|---|---|
| **Composer** | The L1 assembly that consumes an L0 atom. Every L0 atom must list its composers; no atom ships without at least one composer test proving it is actually used. Dead atoms (no composers) are deleted. |
| **Holey-mask** | The `flock.active` boolean array, which may have arbitrary `False` holes at any time (birds removed mid-simulation). Every assembly must be correct under holey masks — inactive rows must return zero force and remain unchanged. |
| **Ghost-cell** | When a bird is near a toroidal boundary, its position is replicated at the opposite side for spatial index queries, so neighbours across the seam are found. Ghost copies are query-only — never integrated. |
| **Golden trajectory** | A pinned reference simulation (positions + velocities saved to `test/data/golden_<mode>.npz`) that future runs must match within `atol=1e-3`. Catches accidental physics regressions. Deliberate changes re-pin in the same commit. |
| **ForceMode** | The ABC in `physics/forces/_mode.py`. Every physics mode is a class that implements `step(flock, config, dt)` and declares flags: `needs_index`, `speed_mode`, `owns_positions`. Registered via `@register` → `MODE_REGISTRY`. |
| **ForceTerm** | A composable force function `(flock, ctx, cfg) → (N,3)` with a `name`, `enabled` toggle, and `gain` multiplier. `composeForces` sums enabled terms linearly (P2.10). |
| **StepContext** | A per-frame dataclass: `frame, dt, rng, center, config, threat_prox`. Built by the engine, passed to extensions and modes — the single source of per-step state. |
| **Extension** | A behavioural addon with `apply(flock, ctx)` — live-toggleable each frame via config. Threat FSM, ecology, wander, and ripple are extensions. |
| **PhysicsFlock** | The main SoA data structure: `positions, velocities, accelerations, prev_positions, last_accelerations, seeds, max_speed, active, is_predator` + `rng, center, index`. |
| **FlockArrays** | The low-level SoA container (`positions`, `velocities`, `accelerations`, `seeds`, `active`, `last_theta`). Does NOT include `rng` or `index` — those live on `PhysicsFlock`. |
| **SimConfig** | The root configuration dataclass — a composition of 17 per-subsystem dataclasses (`DomainConfig`, `FlockConfig`, `ProjectionConfig`, …). YAML sections map 1:1 to sub-configs. |
| **MODE_REGISTRY** | A `dict[str, type[ForceMode]]` populated by the `@register` decorator. `sorted(MODE_REGISTRY)` drives mode cycling. |
| **SpatialIndex** (Protocol) | A structural interface: `rebuild(positions, active, box)`, `query_knn(pos, k) → global_indices`, `query_radius(pos, r) → global_indices`, `query_knn_batch(positions, k, workers)`. |
| **InstanceSchema** | GPU buffer layout descriptor: `floats=8`, `layout="3f 3f 1f 1f/i"`. The 8 floats are `pos.xyz, vel.xyz, flag, hue`. |
| **QualityGovernor** | Adaptive quality ladder: degrade (trails off → render scale −0.15 → N −18%) when fps < 78% target for ≥1.8s. Recovery when healthy for 3.6s. |
| **MurmurationEnv** | A `gymnasium.Env` wrapper: `observation_space = Box(−1, 1, (6N,))`, `action_space = Box(−1, 1, (3N,))`. Lazy import — gymnasium is optional. |
| **EvoFlock** | The evolutionary inverse-design driver: SSGA over 10+ evolvable parameters vs 4 objectives, evaluated in an SDF obstacle world. |
| **MARL** (Multi-Agent Reinforcement Learning) | The RL bridge: an external policy outputs per-bird control actions, applied before the physics step. Supported by the `"marl"` force mode and `MurmurationEnv`. |
| **boundedUnitTravel** | A wander-path guarantee: `‖path(t)‖ ≤ 1` for all `t`. Keeps the wander attractor inside its configured radius. |
| **Toroidal** (boundary mode) | Periodic boundary: positions wrap around domain edges. All distance calculations use min-image. The default boundary condition. |
| **Sphere soft** (boundary mode) | Asymptotic sphere: `Δv = −μ·r̂ / max(R−r, 0.05R)`. Birds feel a gentle push-back that grows as they approach the sphere surface. |
| **DAG** (Directed Acyclic Graph) | The import dependency structure: edges go from higher-level to lower-level (L3→L2→L1→L0). The historical `flock ↔ forces` cycle is the primary architectural debt. |

### Config & Infrastructure

| Term | Definition |
|---|---|
| **Golden re-pinning** | Updating `test/data/golden_<mode>.npz` after a deliberate physics change. Must happen in the same commit — CI fails if goldens are stale. |
| **dt** | Simulation timestep in seconds. Clamped to `[0, 1/20]` by the accumulator (P8.10). A fixed-step accumulator decouples physics from render framerate. |
| **N** | Number of birds (active individuals). The scaling dimension: 150 (default) → 300,000 (target). |
| **v0** | Preferred/cruise speed in simulation units. Default 4.0. Birds are clamped to `[0.3·v0, v0]` in band mode. |
| **σ** (sigma) | Topological neighbourhood size in projection mode — the number of nearest neighbours considered. Default 6. |
| **φp / φa** | Projection-mode weights: directional (alignment) and positional (cohesion) influence. Enforced constraint: `φp + φa ≤ 1`. |
| **CI** (Continuous Integration) | GitHub Actions workflows (`.github/workflows/`): `test.yml` (fast/E2E/slow/GPU/lint) + `guard-rails.yml` (P14 guards). |
| **Preset** | A shipped YAML file in `conf/` that instantiates a specific configuration scenario. Loaded with `--config <name>` or keypress a–h,w. |
| **Marker** (pytest) | Decorators that gate test execution: `@pytest.mark.slow` (performance), `@pytest.mark.gl` (GPU required), `@pytest.mark.golden`. |



## Appendix E — Module → Phase Reverse Index

Which phases touch each file. Use this to find all work affecting a module
without scanning 119 step headings. Listed in the same order as `arch.md` §4.

### Production Code (`pymurmur/`)

| Module | Phases | What happens |
|---|---|---|
| `core/types.py` | P0.12, P2.3 | Math helpers (Rodrigues, min_image, hash01, smoothstep, fibonacci_sphere, seed_noise3, safe_normalize, limit3, lerp, rotate_about); SpatialIndex Protocol |
| `core/config.py` | P2.1 | Flat SimConfig → 17 nested dataclasses; YAML I/O rewrite; validate() with range clamps |
| `simulation/engine.py` | P2.2 | Engine orchestration: drain commands → StepContext → extensions → index → mode → integrate → metrics |
| `physics/boid.py` | P0.3, P0.9, P0.10, P0.15, P4.7 | Invariant fuzz test; 4 integration variants (band/fixed/ceiling/none); safety rails (dt clamp, NaN guard); 5 position-init strategies; sphere centring + asymptotic wall |
| `physics/flock.py` | P0.4, P0.5, P0.6, P0.7, P0.8, P2.4, P2.5, P8.3 | Single seeded RNG; smoothed swarm centre; species column (is_predator); prev_positions + last_accelerations stash; per-bird max_speed array; KDTreeIndex global indices; ghost-cell replication; position_history rolling buffer for ring trails |
| `physics/occlusion.py` | P1.1, P1.2, P1.3, P1.4, P1.5 | True occlusion culling (closest-first, capped 64); probabilistic-union Θ; boundary-length-weighted δ̂; exact α = asin(min(b/d,1)); candidate cutoff at 64 |
| `physics/steric.py` | P1.6 | Clamped 1/d² repulsion — force clamped to max_force when exceeded |
| `physics/obstacles.py` | P0.14, P11.4 | 5 SDF primitives (sphere/box/cylinder/union/subtract) + collision detection + kinematic correction; ObstacleScene CSG tree builder for EvoFlock obstacle courses |
| `physics/forces/_mode.py` | P2.2 | ForceMode ABC (needs_index, speed_mode, owns_positions, reset/step); MODE_REGISTRY; @register decorator |
| `physics/forces/_base.py` | P1.7, P2.10 | Force kernel fixes (sep 1/d², coh bounded, noise ×scale); ForceTerm protocol + composeForces reducer |
| `physics/forces/_kernels.py` | P4.10 | Numba JIT force kernels (use_numba; fastmath policy — off when metrics exported) |
| `physics/forces/projection.py` | P1.10 | Move lazy steric import to module top (L0 atom, no cycle risk) |
| `physics/forces/spatial.py` | P4.1, P4.2, P4.3, P4.5, P4.6 | Hybrid metric+topological filter; correct force accumulation order; predator boids (species-based); per-frame parameter jitter; parallel two-phase update |
| `physics/forces/field.py` | P3.1–P3.12 | Full field-mode rewrite: wander path, 5 Lissajous blob anchors + cyclic phase weights, leader/chaser groups, shell force + inner cavity, slot repulsion, 6 remaining terms (tangential, buoyancy, curl flow, fold noise, drag, drift), ripple envelopes, bounded panic + blackening, Threat FSM + force bundle, blob init + 7 field presets, grid-mode separation normalization, 1.45·R_blob floating boundary |
| `physics/forces/vicsek.py` | P1.8, P6.1–P6.3 | Vicsek memory term + tangent-plane noise (D lives); fear-weighted alignment blending; predator agent (hunt nearest); asymmetric position collisions |
| `physics/forces/influencer.py` | P7.1–P7.6 | Persistent tick + Lissajous target; move-then-steer at unit speed; rank-by-target-distance influence; density-scaled Gaussian init; distance diagnostics; desktop pilotable-flock mode (WASD attractor + shell force) |
| `physics/forces/angle.py` | P5.1–P5.7 | Steering core (Rodrigues, dead zone, never overshoot); unified neighbour modes (flee/align+coh/coh-only); adaptive speed (linear/quadratic/softened); edge handling (cube margin, sphere); heading jitter; incremental spatial grid; body-unit scale invariance |
| `physics/forces/marl.py` | P12.1 | Deferred global rules under external control — control applies first, then move, then rules prep next step |
| `physics/extensions/_base.py` | P2.6 | Extension ABC (apply); ExtensionManager; StepContext dataclass (frame, dt, rng, center, config, threat_prox) |
| `physics/extensions/predator.py` | P3.8, P3.9 | Bounded panic (ceiling raise, NOT compound) + blackening (sep↓, coh↑ near threat); Threat FSM (approach/egress, capture/clear gates) + force bundle (push/wake/split/wave); publishes ctx.threat_prox |
| `physics/extensions/ecology.py` | P4.8 | Logistic dusk roost (temperature-boosted); coherence gate on weights; seasonal flock-size model (cos, Oct–Mar); per-day predator presence |
| `physics/extensions/wander.py` | P3.1 | boundedUnitTravel attractor (‖path‖ ≤ 1 guaranteed) + flock heading |
| `physics/extensions/ripple.py` | P3.7 | 3-train enveloped travelling pulses (env(τ) smoothstep); radial + twist force; fold-noise coupling |
| `viz/renderer.py` | P2.7, P2.8, P8.1, P8.2, P8.4, P8.5 | InstanceSchema + VAO rebuild discipline; PyGLM _mat4_bytes upload; sphere impostors + speed-stretched ellipsoid; depth cues + Fresnel rim; winged flapping mesh + gradient sky; colour channels (per-bird hue, predator red) + theme wiring (ambient+diffuse material tables) |
| `viz/shaders.py` | P8.1, P8.2, P8.4, P8.5 | GLSL: sphere impostor fragment (discard r²>1, edge smoothstep, shade); depth cues (PointSize ∝ 1/depth^k); winged flap (vertex.y += flap_weight before LookAt); theme uniforms (u_Ambient, u_Diffuse, u_Paper, u_Ink) |
| `viz/trails.py` | P8.3 | 4 trail rendering modes: velocity-stretched impostors, screen-space accumulation (depth-only clear), ring trails from position_history rolling buffer, CPU trail lines (5 ribbon segments) |
| `viz/camera.py` | P8.7, P8.8 | Cinematic capture sweep (azim 45°+t·180°, elev 25°+sin·0.15, dist pulse); dual-view + orthographic presets (keys 7/8/9) |
| `viz/hud.py` | P10.3 | Slider HUD (5 knobs: sep/coh/align/avoid/noise with ortho-pass track+knob quads, TAB toggle) |
| `viz/input_control.py` | P7.6, P10.1, P10.4, P10.6 | Pilot mode keyboard bindings (WASD thrust/yaw, QE roll, arrows pitch/yaw, Shift gather, Alt scatter); preset keys a–h,w; cursor-ray spawning (left-click bird, right-click predator); φp+φa ≤ 1 enforcement |
| `viz/visualizer.py` | P8.6, P8.10 | Adaptive quality wired (ladder consumer); fixed-timestep accumulator + render interpolation (lerp prev→current) |
| `capture/recorder.py` | P8.7 | Cinematic capture sweep (pre-warm 60, sweep, GIF optimize+disposal=2); env overrides (CAPTURE_W/H/FRAMES/OUT) |
| `capture/mpl_recorder.py` | P8.9 | GPU-free dual-view matplotlib fallback (warns on activation, never silent except:pass) |
| `analysis/metrics.py` | P0.13, P1.9, P1.10, P4.4, P9.1–P9.8, P9.10 | H₂ disconnected→inf fix; thickness ratio fix (λ₃/λ₁); Θ reports NaN in non-projection modes; physical metrics (watts/newtons/joules, k_v/k_a/m calibration); nematic order parameter S; MSD(τ) curve with crossover; hull-volume τρ; silhouette Θ′; shape→m*; η(m) marginal efficiency; robust gyration + ideal exponent −0.5; motion metrics (velocity deviation, boundary overshoot, normalized L, altitude deviation); export schema (to_dict JSON round-trip) |
| `analysis/rewards.py` | P9.9 | Weighted composite reward terms; faithful_signs flag; shared by MARL (P12) and EvoFlock (P11) |
| `analysis/presets.py` | P8.5, P10.1 | Theme definitions (dark/light) + material tables; preset keys a–h,w with labels and descriptions |
| `analysis/perf.py` | P8.6 | PerfDiagnostics (EMA frame stats, 250ms spike cap); QualityGovernor (budget→classify→degrade ladder: trails→scale→count; 78%-for-1.8s hysteresis) |
| `analysis/evoflock.py` | P11.1–P11.6 | SSGA (worst-of-3 negative selection, uniform crossover, Gaussian mutation, fitness cache); worst-of-4 evaluation (min-reduction); 4 objectives (separation trapezoid, speed band, curvature, hypervolume); expanded gene set (forward force, perception cones, fly-away, k=7, σ integer, speed_min_factor); protocol (persist Pareto front, emergent-alignment experiment) |
| `analysis/phase_diagram.py` | P9.1 | Nematic order parameter option (polar|nematic) in (η, D) sweep |
| `analysis/density_scaling.py` | P9.7 | N-sweep with ideal_density_exponent = −0.5 reported alongside fitted β |
| `analysis/gym_env.py` | P12.2 | MurmurationEnv: Box obs (6N,), Box action (3N,); lazy gymnasium import; seeded reset; truncate at episode_steps; reward from rewards module |

### Entry Points

| Module | Phases | What happens |
|---|---|---|
| `pymurmur/__init__.py` | P10.5 | Facade exports: SimConfig, SimulationEngine, Simulation(**kw), Recorder, benchmark() |
| `pymurmur/__main__.py` | P0.11, P10.5 | Capability probing (detect moderngl/numba, degrade gracefully); CLI: --set key.subkey=value, --print-config, --fullscreen, --list-configs |

### Test Files

| Module | Phases | What happens |
|---|---|---|
| `test/test_architecture.py` | P0.2, P2–P14 | Skeleton with FORBIDDEN_EDGES (P0.2); ALLOWED_EDGES extended at every phase boundary; final matrix matches arch.md §5 (P14.1) |
| `test/test_docs.py` | P14.4 | 4 tests: arch.md links resolve, arch.md references roadmap with P0-P14, roadmap links resolve, bidirectional sync |
| `test/test_golden.py` | P0.1, P1, P3, P5 | Golden trajectory harness (atol=1e-3); re-pinned after each physics change (P1, P3, P5) |
| `test/regenerate_golden.py` | P0.1 | One-shot script: run seeded 15-bird × 30-frame sim, save positions+velocities to .npz |
| `test/core/test_types.py` | P0.12 | Math helper unit tests (rotate_about exact, min_image wrap, fibonacci_sphere count, seed_noise3 range) |
| `test/core/test_config.py` | P2.1 | Nested config round-trip, preset domain survival, unknown key warns, depth>0 validation |
| `test/physics/test_boid.py` | P0.3, P0.9, P0.15 | Speed contract per mode; invariant fuzz (200 seeds, no NaN, in-bounds); position-init no-overlaps + grid spacing |
| `test/physics/test_flock.py` | P0.4, P2.4 | Same-seed bit-identical (100 steps); KDTreeIndex global indices + boxsize toroidal |
| `test/physics/test_occlusion.py` | P1.1, P1.2, P1.3 | Collinear culling, separated all-visible, Θ sub-additive, δ̂ edge vs surrounded |
| `test/physics/test_steric.py` | P1.6 | Steric clamp to max_force at d=0.01 |
| `test/physics/test_obstacles.py` | P0.14 | 5 SDF primitives round-trip correct; collision detection sign-change |
| `test/physics/forces/test_kernels.py` | P1.7, P4.10 | Separation 1/d², cohesion bounded, noise scale live; numba ≡ numpy atol=1e-5 |
| `test/physics/forces/test_mode_contract.py` | P2.2 | All registered modes instantiable; step respects active mask; determinism; import cycle dead |
| `test/physics/forces/test_force_terms.py` | P2.10 | Each ForceTerm independently testable; composeForces linearity; runtime toggle |
| `test/physics/forces/test_field.py` | P3 | Golden pinned; 10⁴-frame NaN/speed fuzz; 7 presets load and run; grid sep normalized; floating boundary contains birds |
| `test/physics/forces/test_spatial_variants.py` | P4 | Hybrid filter neighbour sets; force order stage recording; predator escape >> separation; jitter variance verified |
| `test/physics/forces/test_vicsek_core.py` | P1.8 | D=0 aligned flight autocorr > 0.999; D=4 → < 0.5; noise in tangent plane |
| `test/physics/forces/test_vicsek_species.py` | P6 | Fear-weighted alignment; predator monotone pursuit; asymmetric collisions at seam |
| `test/physics/forces/test_angle.py` | P5 | 180° turn in π/rate seconds; dead-zone hold; birds arcing at walls; incremental grid ≡ full rebuild |
| `test/physics/forces/test_influencer.py` | P7 | T(t) at known t; move-then-steer lag; rank influence monotone; density-scaled init equal across N; pilot converge/follow/shell |
| `test/physics/extensions/test_extensions.py` | P2.6 | ExtensionManager empty no-op; PreStepOrder verified |
| `test/physics/extensions/test_ecology.py` | P4.8 | seasonal_size_factor(15)≈1.0, (197)≈0.25; dusk_factor(0)=0, (40)=1; gated_weight 0→1 transition |
| `test/physics/extensions/test_threat.py` | P3.8, P3.9 | Panic speed ≤ 2.35·v0; predator pass-through and exit; threat_prox ∈ [0,1] |
| `test/physics/test_spatial_index_contract.py` | P2.3, P2.4, P2.5 | KDTreeIndex + SpatialHashGrid conformance: global indices, toroidal cross-seam, ghost-cell equivalence |
| `test/physics/test_composition.py` | P2.9 | Holey-mask matrix: all modes × 20 steps, no exceptions, inactive rows unchanged |
| `test/simulation/test_engine.py` | P2.2 | Engine step order (spy); headless N steps; reset restores initial state |
| `test/viz/test_renderer.py` | P2.7, P8.1 | VAO rebuilt after buffer growth; headless FBO has depth attachment; impostor centre > rim |
| `test/viz/test_trails.py` | P8.3 | 4 trail modes: velocity extends along motion, accumulation persists, ring K sprites, lines <2ms at 20K |
| `test/viz/test_trails.py` | P8.3 | 4 trail modes: velocity extends along motion, accumulation persists, ring K sprites, lines <2ms at 20K |
| `test/viz/test_camera.py` | P8.8 | Ortho presets (keys 7/8/9); dual-view halves differ; sweep params in range |
| `test/capture/test_recorder.py` | P8.7 | GIF depth-correct, pre-warmed, swept; env overrides active |
| `test/analysis/test_metrics.py` | P0.13, P1.9, P1.10, P4.4, P9.1–P9.8 | H₂ inf when disconnected; thickness ∈ (0,1]; Θ NaN in non-projection; physical units calibration; nematic S anti-parallel vs isotropic; MSD slope crossover; τρ period-P bounded; silhouette ≠ voxel; m* monotone; η(m) inf at transition; robust gyration trims outliers; motion metrics O(1) |
| `test/analysis/test_metrics_invariance.py` | P9.1 | Nematic S invariant under û→−û and SO(3); polar α = 0 for anti-parallel |
| `test/analysis/test_rewards.py` | P9.9 | Perfect flock → reward 0 (max); faithful flag flips sign; per-term weight linearity |
| `test/analysis/test_evoflock.py` | P11.1–P11.6 | Worst-of-3 deleted; child mixes parents; worst-of-4 min-reduction; trapezoid pinned; SDF collisions → ≈0 after evolution; emergent alignment α > 0.5 |
| `test/analysis/test_marl.py` | P12.2, P12.3 | env_checker passes; obs ∈ [−1,1] over 500 steps; trained beats random by ≥20% |
| `test/analysis/test_perf.py` | P8.6 | Adaptive ladder fires in correct order; synthetic frame-times trigger actions; recovery stops |
| `test/test_performance.py` | P13.1, P13.2 | Per-mode step-time budgets; scaling checkpoints (150→300K birds); memory audit ≤25 MB |
| `test/test_scaling.py` | P13.3, P13.4 | 24-hour soak (no NaN, no memory growth); determinism matrix (mode × threads × jitter × numba) |

### Config & Scripts

| Module | Phases | What happens |
|---|---|---|
| `conf/*.yaml` (7 presets) | P2.1 | Rewrite flat YAML → nested schema (section names = sub-config field names) |
| `conf/field_*.yaml` (7 presets) | P3.10 | quiet_roost, lava_lamp, ink_cloud, predator_ripple, vacuole, silk_sheet, storm_turn |
| `conf/murmuration_evo.yaml` | P11.6 | Confined evaluation config (enclosure + SDF obstacles) |
| `conf/evo_open.yaml` | P11.6 | Open evaluation config (boundary: open) |
| `scripts/train_marl.py` | P12.3 | PPO("MlpPolicy"), 5000 timesteps, save output/marl_ppo (dependency-gated) |
| `scripts/rollout_marl.py` | P12.3 | 500 deterministic-predict steps → dual-view GIF (dependency-gated) |

### CI & Infrastructure

| Module | Phases | What happens |
|---|---|---|
| `.github/workflows/guard-rails.yml` | P14 | 7 jobs: DAG matrix, golden trajectory, config-usage drift, strictly-3D, doc-links, collection-count, summary gate |
| `.github/workflows/test.yml` | P0–P14 | Fast/E2E/slow/GPU/lint matrix; phase-by-phase test commands |
| `.github/gantt-schedule.md` | — | Two-stream parallel execution schedule (P3–P7 ∥ P8+P9) |
| `#appendix-g--oldnew-identifier-mapping` | — | 108 old→new identifier mappings (D0–D9, S1–S7, T0–T6 → P0–P14) |
| `#appendix-h--step-index-119-steps` | — | 119 step headings extracted: Phase, Step, Title, Level, Files, Test Files, Citations |



## Appendix F — Phase Boundary Checklists

Consolidated acceptance criteria from each phase's prose block, formatted as
a CI-auditable checklist. Every item maps to a specific test or AST assertion
that can be run mechanically. Checkboxes are unchecked `[ ]` by default;
mark `[x]` when the phase is accepted.

---

### Phase 0 — Foundations & Safety Net

- [ ] **Golden suite** — All modes have golden trajectories at `test/data/golden_<mode>.npz`; CI asserts match within `atol=1e-3`
- [ ] **Determinism** — Same seed → bit-identical after 100 steps per mode
- [ ] **flock.center** — Lags teleported centroid (`center += 0.5·(centroid − center)`)
- [ ] **flock.is_predator** — Survives `add_boids(is_predator=True)` and `remove_boids()`
- [ ] **10 math helpers** — Round-trip correct: Rodrigues exact at π/2, min_image wraps to −10, smoothstep endpoints, hash01 range
- [ ] **H₂ inf** — `compute_h2` returns `inf` for disconnected graph (`test_h2_inf_when_disconnected`)
- [ ] **5 SDF primitives** — Round-trip correct (sphere, box, cylinder, union, subtract)
- [ ] **5 position-init strategies** — `box`, `sphere_shell`, `gaussian`, `grid`, `blob` all functional; `cfg.flock.position_init` selector works
- [ ] **Safety rails** — `dt` clamped to `[0, 0.05]`; non-finite positions self-heal to `flock.center`
- [ ] **Capability probing** — `--print-config` reports moderngl/numba availability
- [ ] **Architecture test** — `ALLOWED_EDGES` contains `core` + `physics/boid`; no module-level `np.random.*` in production code
- [ ] **Migration** — Module-level `np.random.*` → `flock.rng.*`; PhysicsFlock gains 5 new columns; `core/types.py` gains 10 helpers; `physics/obstacles.py` created
- [ ] **Test command** — `pytest test/core/ test/physics/test_boid.py test/physics/test_obstacles.py test/test_golden.py test/test_architecture.py -v`

### Phase 1 — Scientific Correctness

- [ ] **Occlusion culling** — Collinear birds → only nearest visible (`test_occlusion_culls_behind_nearer_cap`)
- [ ] **Θ sub-additive** — `Θ₁ < Θ₁₂ < Θ₁+Θ₂`, `Θ ∈ [0,1]` (`test_theta_sub_additive`)
- [ ] **|δ̂| contract** — `< 1e-2` when surrounded, `≈1` at edge (`test_delta_edge_vs_surrounded`)
- [ ] **Exact α** — All `b_eff/d` replaced with `asin(min(b_eff/d, 1.0))`
- [ ] **Candidate cutoff** — Neighbours capped at nearest 64 in occlusion
- [ ] **Steric clamp** — At `d=0.01`, force equals exactly `max_force` (`test_steric_clamped`)
- [ ] **Separation 1/d²** — Force kernel uses inverse-square law (`test_separation_inverse_square`)
- [ ] **Cohesion bounded** — Large offset → `|F| = weight` (not unbounded)
- [ ] **Noise scale live** — `scale=0 → zero force`; `scale=0.5 → mean|F|≈0.5`
- [ ] **Vicsek memory** — D=0 aligned flight → autocorrelation > 0.999 at lag 1
- [ ] **Vicsek D live** — D=4 → autocorrelation < 0.5 (`test_vicsek_D_live`)
- [ ] **Vicsek tangent-plane** — Noise perpendicular to heading: `|n⊥·û| < 1e-6`
- [ ] **Thickness fix** — Thin line → `thickness < 0.2`; formula = `√(λ₃/λ₁)`
- [ ] **Θ NaN** — Non-projection modes report `last_theta = NaN` (not stale zero)
- [ ] **Projection lazy import** — `from ..steric import steric_force` moved to module top
- [ ] **Golden re-pin** — projection + spatial + vicsek goldens regenerated
- [ ] **Architecture test** — `ALLOWED_EDGES` extended: occlusion, steric, forces/_base, vicsek, metrics → core
- [ ] **Test command** — `pytest test/physics/test_occlusion.py test/physics/test_steric.py test/physics/forces/test_kernels.py test/physics/forces/test_vicsek_core.py test/analysis/test_metrics.py -v`

### Phase 2 — Contracts & Protocols

- [ ] **Import cycle dead** — `physics/flock` does not import anything from `physics/forces/` (AST check)
- [ ] **Preset domains** — All `conf/*.yaml` load with correct `domain.width` not overwritten by `capture.width`
- [ ] **Nested config round-trip** — `SimConfig.from_file(p) → to_file → from_file` produces identical config
- [ ] **Unknown keys warn** — Extra YAML keys produce `UserWarning`
- [ ] **Depth validation** — `validate()` rejects `depth=0`
- [ ] **MODE_REGISTRY** — ≥7 entries; `sorted(MODE_REGISTRY)` matches `arch.md` force-mode table
- [ ] **Registered modes instantiable** — All 7 mode classes pass `test_registered_and_instantiable`
- [ ] **Mode step respects active mask** — Holey-flock test: inactive rows unchanged
- [ ] **Mode determinism** — Same seed → identical flock state after 100 steps per mode
- [ ] **KDTreeIndex global indices** — Query returns capacity-space row numbers (not compacted)
- [ ] **KDTreeIndex boxsize** — Toroidal queries correct across seam
- [ ] **SpatialHashGrid ghost-cell** — Cross-seam neighbour matches full brute-force
- [ ] **ExtensionManager** — Empty → no-op; pre_step order verified (extensions before mode)
- [ ] **Holey-mask matrix** — All modes × 20 steps, no exceptions, inactive rows unchanged
- [ ] **VAO rebuild** — After buffer growth (`add_boids` beyond chunk), VAO rebuilt and renders correctly
- [ ] **Headless FBO depth** — `depth_attachment` present on FBO
- [ ] **Dead-code cleanup** — `FlockArrays` composed into `PhysicsFlock.arrays`; `ForceKernel` deleted; `BoidView` deleted
- [ ] **Shim retirement** — `not hasattr(SimConfig(), "phi_p")` (flat fields gone)
- [ ] **ForceTerm conformance** — `composeForces(a+b) = composeForces(a) + composeForces(b)`
- [ ] **ForceTerm toggles** — Flipping `term.enabled = False` mid-run → that term contributes zero
- [ ] **Architecture test** — `ALLOWED_EDGES` extended: flock, forces/*, extensions, engine, viz, capture, analysis
- [ ] **Test command** — `pytest test/core/test_config.py test/physics/forces/test_mode_contract.py test/physics/test_spatial_index_contract.py test/physics/extensions/test_extensions.py test/physics/test_composition.py test/simulation/test_engine.py test/viz/test_renderer.py -v`

### Phase 3 — Field/Blob Mode + Threat FSM

- [ ] **Golden trajectory** — Pinned for field mode at `test/data/golden_field.npz`
- [ ] **10⁴-frame NaN/speed fuzz** — All 13 terms on, no NaN, all speeds in contract
- [ ] **7 field presets** — All load with documented values; frame-0 lobes visible
- [ ] **Grid-mode separation** — Bird with 10 neighbours vs bird with 2 → same `|F_sep|` within 5%
- [ ] **1.45·R_blob floating boundary** — No bird exceeds `1.02·R_boundary` over 10⁴ frames
- [ ] **Wander path** — `‖path‖ ≤ 1` for 10⁶ fuzzed t; heading continuous (`‖h(t+ε)−h(t)‖ < 0.05`)
- [ ] **5 anchor clusters** — 2K birds → k-means finds ≥4 clusters at t=30s
- [ ] **Leader fraction** — 0.16 ± 0.02 (`hash01(seed+5.91) ≥ 0.84`)
- [ ] **Shell cavity** — 5K blob: centre voxel density < 0.3× shell-band density
- [ ] **Slot repulsion** — Kernel zero at `r_slot` and beyond; continuous at boundary
- [ ] **13 terms all non-zero** — Random config → every term produces non-zero output
- [ ] **Ripple envelope** — Zero outside [0.6, 8.8]; paused-flock shows 3 concentric rings
- [ ] **Panic ceiling** — Panicked speed ≤ 2.35·v0 (ceiling raise, not compound multiply)
- [ ] **Threat pass-through** — Predator enters flock, crosses capture distance, exits beyond clear distance
- [ ] **Threat horizontal bias** — Evacuated region XY-extent > Z-extent (split force design)
- [ ] **threat_prox range** — `ctx.threat_prox ∈ [0,1]` every frame
- [ ] **Field mode perf** — ≤3ms per step at N=16K
- [ ] **Architecture test** — New edges: field.py, predator.py, wander.py, ripple.py → core + flock(read)
- [ ] **New files** — `conf/field_*.yaml` (7 presets) created and validated
- [ ] **Test command** — `pytest test/physics/forces/test_field.py test/physics/extensions/test_threat.py -v`

### Phase 4 — Reynolds Variants + Ecology

- [ ] **Golden trajectory** — Pinned for spatial mode at `test/data/golden_spatial.npz`
- [ ] **Hybrid filter** — Neighbour set respects visual_range AND influence_count cap
- [ ] **Alignment subset** — alignment neighbours ⊆ cohesion neighbours (dual radii)
- [ ] **Force order** — Stage-order recording matches documented pipeline (forces → predator boost → accel_scale → clamp → v+=a → noise → speed → move → wrap)
- [ ] **Predator escape** — `|F_escape| >> |F_sep|`; align+coh exactly zero when predator perceived
- [ ] **Flash-expansion** — Mean NN distance doubles within 30 frames after predator spawn
- [ ] **Physical metrics** — `speed_real = k_v·|v|` exactly; `E ≈ P̄·elapsed ± 1%`; L about CoM
- [ ] **Jitter variance** — `spacing_std(on) > spacing_std(off)`, same seed; config unchanged after run
- [ ] **Parallel two-phase** — ≥3× speedup at N=20K vs recorded loop baseline; identical results across worker counts
- [ ] **Sphere centring** — Centre-initialised flock → `‖CoM−C‖ < 0.1R` over 5000 frames
- [ ] **Soft sphere** — No bird crosses `R` in sphere_soft mode
- [ ] **Ecology dusk** — `dusk_factor(0)=0`, `dusk_factor(40)=1`; `seasonal_size_factor(15)≈1.0`, `(197)≈0.25`
- [ ] **Ecology coherence** — `gated_weight(0.8, 10)≈0`, `gated_weight(0.8, 600)>0.7`
- [ ] **Ecology seasonal** — Jan in-season, Jul out-of-season; Oct–Mar window
- [ ] **Velocity init** — Cube `E|v|≈0.816·v0 ±5%`; speed_uniform in-band; tangential ⊥ radial
- [ ] **Numba equivalence** — numba ≡ numpy within `atol=1e-5` (fastmath off), N=2K
- [ ] **Architecture test** — New edges: spatial.py, _kernels.py, ecology.py, boid.py
- [ ] **New files** — `physics/forces/_kernels.py`, `test/physics/forces/test_kernels.py`
- [ ] **Test command** — `pytest test/physics/forces/test_spatial_variants.py test/physics/forces/test_kernels.py test/physics/extensions/test_ecology.py test/analysis/test_metrics.py test/physics/test_boid.py -v`

### Phase 5 — Angle Mode

- [ ] **Mode loads** — `"angle"` key in `MODE_REGISTRY`; `AngleMode` instantiable
- [ ] **180° turn** — Completes in `π/rate` seconds ± 1 frame (`test_steering_180_turn`)
- [ ] **Dead-zone hold** — No turn when `φ < turn_threshold°` (anti-oscillation)
- [ ] **Never overshoot** — Per-frame heading change ≤ `rate·dt + jitter`
- [ ] **Birds arcing** — Tangential speed at wall > 0; zero escapes over 10⁴ frames at max speed
- [ ] **Speed self-regulates** — m=0 → base+35 (linear); m≥7 → base; median 7th-NN distance converges
- [ ] **Incremental grid ≡ full rebuild** — Neighbour sets match over 500 random-walk frames; touches <10% of birds/frame
- [ ] **Scale invariance** — Doubling `boid_size` doubles all three radii thresholds
- [ ] **Jitter bounded** — Steering-off distribution bounded `±4°`
- [ ] **Architecture test** — New edge: `angle.py → core + physics/flock(read)`
- [ ] **New files** — `physics/forces/angle.py`, `test/physics/forces/test_angle.py`
- [ ] **Test command** — `pytest test/physics/forces/test_angle.py -v`

### Phase 6 — Vicsek Predator–Prey

- [ ] **Fear-weighted alignment** — Stationary predator at centre → prey `⟨û·r̂⟩ > 0.8` within 5 steps
- [ ] **Predator hunts** — Nearest prey within `detect_ratio·R_pred`; ≥90% of steps close distance
- [ ] **Predator fallback** — `n_prey=0` → `α ≈ 1/√N` for all η, D (random walk)
- [ ] **Asymmetric same-type** — Each moves `(R_avoid−d)/2` along min-image n̂
- [ ] **Asymmetric prey-predator** — Prey takes full `(R_pred−d)`, predator unmoved
- [ ] **Seam-crossing** — Collision correction works across toroidal boundary
- [ ] **No same-type overlaps** — 100 steps → no pair < `0.5·R_avoid`
- [ ] **Prey-only α** — `α=1.0` with one orthogonal predator (polar alignment preserved)
- [ ] **Architecture test** — New edges: `vicsek.py → core + physics/flock(read)` (extended)
- [ ] **New tests** — `test/physics/forces/test_vicsek_species.py`
- [ ] **Test command** — `pytest test/physics/forces/test_vicsek_species.py -v`

### Phase 7 — Influencer Parity

- [ ] **Persistent tick** — `T(t)` at t∈{0, 970, 2170} equals hand-computed values
- [ ] **In-domain** — Target stays inside domain for `scale ≤ 1`
- [ ] **Move-then-steer lag** — One-step lag visible after target jump; `|v| ≡ v0`
- [ ] **Frozen target convergence** — Birds converge to hover/orbit around stationary target
- [ ] **Rank influence monotone** — Exactly one bird at 1.0; min ≈ 0.055 ± 1e-3; non-increasing in target distance
- [ ] **Density-scaled init** — Init density equal across N∈{100, 1000, 8000} (±10%)
- [ ] **Distance diagnostics** — `FlockMetrics.target_dist_min/max` populated; CSV contains target_dist columns
- [ ] **Pilot converge** — Static pilot at C → all birds converge to shell_radius within 60 frames
- [ ] **Pilot follow** — Pilot moves +X → flock CoM tracks within `2·shell_radius`
- [ ] **Shell expand/contract** — Shift/Alt → shell_radius changes monotonically, capped at [0.42, 2.2]
- [ ] **Architecture test** — New edges: `influencer.py → core + physics/flock(read)` (extended)
- [ ] **New tests** — `test/physics/forces/test_influencer.py`
- [ ] **Test command** — `pytest test/physics/forces/test_influencer.py -v`

### Phase 8 — Rendering & Capture

- [ ] **Sphere impostors** — Centre pixel brighter than rim; corners = background (`test_impostor_centre_vs_rim`)
- [ ] **Speed-stretched ellipsoid** — Quad scales along projected velocity by `1 + speed_ratio·0.3`
- [ ] **Depth cues** — Near bird renders larger and more opaque than far bird
- [ ] **Fresnel rim** — Rim-lighting visible at grazing angles
- [ ] **4 trail modes** — Velocity extends along motion; accumulation persists ~1/fadeOpacity frames; ring K sprites monotone; lines <2ms at N=20K
- [ ] **Winged mesh** — Flap animation visible; vertex.y oscillates with flap_weight before LookAt
- [ ] **Gradient sky** — Top→bottom colour gradient renders; theme-overridable
- [ ] **Theme wiring** — `u_Ambient`, `u_Diffuse`, `u_Paper`, `u_Ink` uniforms all set from theme dict
- [ ] **Per-bird hue** — `h = seed·360` for HSV; predator flag → red, ×1.3–1.5 scale
- [ ] **Adaptive ladder** — Synthetic frame-times → actions fire in correct order, spaced ≥1.8s, recovery stops
- [ ] **Capture sweep** — GIF: `optimize=True, disposal=2`; pre-warmed 60 frames; cinematic azim/elev curve applied
- [ ] **Capture env overrides** — `CAPTURE_W/H/FRAMES/OUT` take precedence
- [ ] **Dual-view** — Two halves render with distinct camera angles (15°/15° vs 45°/45°)
- [ ] **Ortho presets** — Keys 7/8/9 → ortho-top/ortho-side/perspective; equal pixel sizes at different depths
- [ ] **GPU-free fallback** — Matplotlib fallback produces ≥1 GIF frame; warning emitted on activation
- [ ] **Fixed-timestep** — 30fps vs 60fps → identical physics after same elapsed time
- [ ] **Alpha-accumulation** — Cluster centre darker than single bird (cumulative absorption)
- [ ] **Impostor perf** — 20K birds maintain 60fps
- [ ] **Architecture test** — New edges: shaders.py, trails.py, hud.py, mpl_recorder.py; `viz/visualizer` holds engine ref but no simulation imports
- [ ] **New files** — `viz/trails.py`, `capture/mpl_recorder.py`
- [ ] **Test command** — `pytest test/viz/ test/capture/ test/analysis/test_perf.py -v -m gl`

### Phase 9 — Metrics & Analysis

- [ ] **Nematic S** — Two anti-parallel half-flocks: `α<0.05`, `S>0.95`
- [ ] **Nematic isotropic** — 500 random birds: both `α<0.15`, `S<0.15`
- [ ] **Nematic invariance** — `S` invariant under `û→−û` and SO(3) rotations
- [ ] **MSD slope** — D=0 aligned → slope `2.0±0.1`; strong-noise walkers → `1.0±0.2` for τ≥4
- [ ] **MSD seam** — Seam crossing contributes `MSD(1) = (v·dt)² ± 1e-4`
- [ ] **Hull-τρ** — Cube hull = edge³ ± 1e-3; coplanar → 0; constant series → τ=0; period-P → τ∈[P/6,P]
- [ ] **Silhouette Θ′** — Flat wall ⊥ axis → silhouette≈1 while voxel Θ′≪1; two co-projected = one
- [ ] **Shape→m*** — aspect 1→9.78, ≥3→6.05; monotone; thin flock ≤7, round ≥8
- [ ] **η(m)** — Connectivity transition → `math.isinf`; both disconnected → 0.0
- [ ] **Robust gyration** — One 10K-unit outlier moves `R_g` <5%; degenerate → density=0
- [ ] **Ideal exponent** — Density-scaling sweep carries `ideal_density_exponent = −0.5`
- [ ] **Velocity deviation** — Equal headings + mixed speeds → deviation>0 while α=1
- [ ] **Boundary overshoot** — 0 inside, >0 outside domain
- [ ] **Normalized L** — O(1) across ×10 domain scale change
- [ ] **Rewards linearity** — Per-term weights produce proportional contribution
- [ ] **Rewards faithful flag** — Flips alignment sign when `reward_faithful_signs=True`
- [ ] **Export schema** — `to_dict()` JSON round-trips; ndarray→list, numpy scalar→python
- [ ] **Architecture test** — New edges: rewards.py, phase_diagram.py, density_scaling.py
- [ ] **Test command** — `pytest test/analysis/test_metrics.py test/analysis/test_metrics_invariance.py test/analysis/test_rewards.py -v`

### Phase 10 — UX & Tooling

- [ ] **Presets a–h,w** — All 8 keys apply correct preset with printed label + description
- [ ] **Title readout** — Contains `mode | N | φp/φa/σ | α Θ Θ′ L σr | τρ | FPS` at correct cadence (every 20th frame)
- [ ] **Slider HUD** — 5 sliders (sep/coh/align/avoid/noise) write nested config fields; drag locks orbit; TAB toggles
- [ ] **Cursor-ray spawn** — Left-click → bird at unprojected world position; right-click → predator
- [ ] **CLI --set** — `--set spatial.separation_weight=6 --set flock.num_boids=500` reflected in `--print-config`
- [ ] **CLI unknown key** — Unknown key exits with field list
- [ ] **CLI --fullscreen** — Toggles fullscreen mode
- [ ] **Facade benchmark** — `pymurmur.Simulation(num_boids=200).benchmark(1000)` returns 20 positive floats
- [ ] **φp+φa ≤ 1** — Input handler enforces: incrementing one reduces the other if sum would exceed 1
- [ ] **Architecture test** — New edges: `pymurmur/__init__.py`, `__main__.py`, `viz/input_control → analysis/presets`
- [ ] **Test command** — `pytest test/viz/test_input.py test/test_cli.py test/test_facade.py -v`

### Phase 11 — EvoFlock

- [ ] **SSGA worst-of-3** — Deleted genome is worst of 3 evaluated candidates
- [ ] **Child mixes parents** — Disjoint-value parents → child has genes from both (uniform crossover)
- [ ] **Fitness cache** — Cache prevents re-simulation of identical genomes
- [ ] **Worst-of-4** — `eval_parallel` with min-reduction; `fitness = min([0.9, 0.8, 0.95, 0.7]) = 0.7`
- [ ] **Separation trapezoid** — Pinned at d/body∈{1.9→0, 2.5→1, 4→1, 5→0}
- [ ] **Speed band** — [19,21] m/s on speed_real; ramps [18,22]
- [ ] **Curvature** — Helix κ matches analytic ±2%
- [ ] **Hypervolume** — `Π max(o_k, 0.01)`, all objectives finite
- [ ] **SDF collision detection** — sign(SDF(p_old)) ≠ sign(SDF(p_new)) → collision counted
- [ ] **Kinematic correction** — Colliding bird moved to SDF surface along gradient
- [ ] **Obstacle course** — Collisions > 0 with zero avoidance, ≈ 0 with evolved weights
- [ ] **Expanded genes** — Forward force, perception cones, fly-away, k=7, σ integer, speed_min_factor all evolvable
- [ ] **Pareto front** — Persisted to `output/evolved.yaml` alongside best genome + seeds + scores
- [ ] **Emergent alignment** — Evolve with NO alignment objective on confined config → settled α > 0.5
- [ ] **Architecture test** — New edges: `evoflock → simulation + core` (F2 tier); `obstacles::ObstacleScene → core + obstacles`
- [ ] **Test command** — `pytest test/analysis/test_evoflock.py -v -m slow`

### Phase 12 — MARL Bridge

- [ ] **MARL deferral** — Two-step hand trace: positions at step k depend on rules from k−1 only
- [ ] **MARL scaling** — Rule weight = 0.01; control clip bounds at ±v_cap
- [ ] **Gym checker** — `gymnasium.utils.env_checker.check_env(MurmurationEnv)` passes
- [ ] **Obs ∈ [−1,1]** — Over 500 random steps, all observations in range
- [ ] **Identical reset** — Same seed + same actions → identical observations
- [ ] **Truncation** — Episode ends at `marl.episode_steps` (500)
- [ ] **Train smoke** — 200-timestep `learn()` completes (`@slow`, skip without sb3)
- [ ] **Rollout GIF** — `rollout_marl.py` produces ≥1 frame GIF (`@slow`)
- [ ] **Trained beats random** — Trained policy's mean dispersion < random policy's by ≥20%
- [ ] **Scripts gated** — `train_marl.py` and `rollout_marl.py` skip gracefully without gymnasium/sb3
- [ ] **Architecture test** — New edges: `marl.py → core + flock(read)`; `gym_env.py → simulation + core`
- [ ] **New files** — `physics/forces/marl.py`, `analysis/gym_env.py`, `scripts/train_marl.py`, `scripts/rollout_marl.py`
- [ ] **Test command** — `pytest test/analysis/test_marl.py -v -m slow`

### Phase 13 — Scaling & Performance

- [ ] **Per-mode budgets** — Each mode's step-time ≤ its documented budget at N=2000 (field ≤3ms, influencer ≤1ms, projection ≤13ms, spatial ≤17ms, vicsek ≤17ms; `@slow`)
- [ ] **Scaling checkpoint 1** — N=150 hash grid → 60fps (≤16.7ms)
- [ ] **Scaling checkpoint 2** — N=1500 SoA → 60fps
- [ ] **Scaling checkpoint 3** — N=16000 cKDTree → 60fps
- [ ] **Scaling checkpoint 4** — N=50000 numba → 45fps (≤22.2ms)
- [ ] **Scaling checkpoint 5** — N=300000 full → 30fps (≤33.3ms)
- [ ] **Memory audit** — `sys.getsizeof` on SoA arrays at N=300K → total ≤25 MB
- [ ] **24h soak** — No NaN, no memory growth trend, positions in-bounds, speed within contract
- [ ] **Determinism matrix** — Same seed → bit-identical for all (mode × num_threads∈{1,−1} × jitter on/off × numba on/off with fastmath off)
- [ ] **Two in-process** — Two `SimulationEngine` instances with same seed → identical
- [ ] **Subprocess** — Separate Python process with same seed → identical to in-process
- [ ] **Test command** — `pytest test/test_performance.py test/test_scaling.py -v -m slow`

### Phase 14 — Guard Rails

- [ ] **DAG matrix** — `ALLOWED_EDGES` matches `arch.md` §5 exactly; zero edges outside matrix
- [ ] **Named regression edges** — `physics.flock !→ physics.forces`, `viz.input_control !→ simulation`, no `cKDTree(` in `forces/`, no module-level `np.random.*`
- [ ] **Config-usage drift** — Every `SimConfig` leaf field read by ≥1 non-config module; zero orphan fields
- [ ] **Strictly-3D** — No `(…, 2)`-shaped spatial arrays in `physics/`; `validate()` enforces `depth > 0`
- [ ] **Doc-links** — All intra-repo markdown links resolve in `arch.md` and `roadmap_deepseek.md`
- [ ] **Bidirectional sync** — `arch.md` references `roadmap_deepseek.md` with P0-P14; no stale D0-D9/S1-S7/T0-T6
- [ ] **Collection-count** — Test count pinned per subpackage; deliberate changes update the count
- [ ] **CI gate (meta)** — `.github/workflows/guard-rails.yml` summary job blocks merge if any of the 6 guard jobs fails; green when all checkboxes above are satisfied
- [ ] **Test command** — `pytest test/test_architecture.py test/test_docs.py -v`



## Appendix G — Old→New Identifier Mapping

> Merged from `#appendix-g--oldnew-identifier-mapping` (now removed). Maps every
> D0-D9, S1-S7, T0-T6 identifier from the retired `roadmap.md` to its
> corresponding P-phase step(s) in this document.

# Migration Reference — Old → New Phase Mapping

> **Purpose:** Anyone reading the retired `roadmap.md` (D0–D9, T0–T6, S1–S7) can
> use this table to find the equivalent content in `roadmap_deepseek.md` (P0–P14).
>
> **Supersedes:** `roadmap.md` (retired) → `roadmap_deepseek.md` (current).
> Target architecture: [`arch.md`](arch.md).

---

## Quick Lookup

```
D0  → P0.1–P0.3         S1.x  → P1.x             T0   → P0.1–P0.3
D1  → P2.1              S2.Ax → P3.x             T1   → P14.x
D2  → P2.2, P2.9        S2.Bx → P4.x             T2   → P2.1
D3  → P0.4–P0.8         S2.Cx → P5.x             T3   → P2.2–2.9
D4  → P0.9–P0.10        S2.Dx → P6.x             T4   → P0.4, P13.5
D5  → P2.3–P2.5         S2.Ex → P7.1–P7.5        T5   → P8
D6  → P2.6              S3.x  → P9.x             T6   → P13
D7  → P2.7–P2.8         S4.x  → P8.x
D8  → P2.2, P10         S5.x  → P10.x
D9  → P14, Appx B       S6.x  → P11.x
                         S7.x  → P12.x
```

---

## Architecture Foundation (D0–D9)

| Old | Description | New | Notes |
|---|---|---|---|
| **D0** | Safety net before refactor | **P0.1–P0.3** | Golden trajectory harness, architecture test skeleton, physics invariant fuzz |
| **D1** | Nested configuration layer | **P2.1** | Nested `SimConfig` — per-subsystem dataclasses, YAML section=field, unknown-key warnings |
| **D2** | ForceMode protocol + registry | **P2.2, P2.9** | `ForceMode` ABC, `MODE_REGISTRY`, `@register`, engine orchestration, holey-mask contract tests |
| **D3** | Flock state contract | **P0.4–P0.8** | `flock.rng` (P0.4), `flock.center` (P0.5), `flock.is_predator` (P0.6), `prev_positions` + `last_accel` (P0.7), `max_speed` (P0.8) |
| **D4** | Integration contract | **P0.9–P0.10** | Integration variants (speed_mode band/fixed/ceiling/none), zero-speed fallback, dt clamp, NaN guard |
| **D5** | SpatialIndex protocol | **P2.3–P2.5** | `SpatialIndex` protocol, `KDTreeIndex` global indices + boxsize toroidal, `SpatialHashGrid` ghost-cell replication + modulo cells |
| **D6** | Extension protocol widening | **P2.6** | `StepContext` dataclass (frame/dt/rng/center/config/threat_prox), `Extension` ABC, `ExtensionManager` |
| **D7** | Renderer contract | **P2.7–P2.8** | `InstanceSchema` + VAO discipline (rebuild on growth), FBO with depth attachment, PyGLM `_mat4_bytes` |
| **D8** | Engine seams: control, commands, quality | **P2.2, P10** | Control hook → P2.2 engine orchestration; Command queue → P10.4 cursor-ray spawning; Quality governor → P8.6 adaptive quality |
| **D9** | Analysis split, facade, exports, cleanup, doc sync | **P14.1–P14.5, Appx B** | DAG matrix finalization (P14.1), config-usage drift (P14.2), 3D guard (P14.3), doc-drift test (P14.4), collection-count (P14.5), dead-code retirement in Appendix B migration notes |

---

## Test Infrastructure (T0–T6)

| Old | Description | New | Notes |
|---|---|---|---|
| **T0** | Harness & fixtures | **P0.1–P0.3** | Shared fixtures (`conftest.py`), golden harness (`test_golden.py`), invariant fuzz (`test_boid.py`) |
| **T1** | Architecture & drift guards | **P14.1–P14.5** | All five T1 sub-items promoted to their own P14 steps |
| **T1.1** | Import-rule matrix | **P14.1** | AST-walk DAG enforcement, named regression edges |
| **T1.2** | Config-usage drift | **P14.2** | Every `SimConfig` leaf field read by ≥1 non-config module |
| **T1.3** | Strictly-3D guard | **P14.3** | No `(…, 2)` spatial arrays in `physics/` |
| **T1.4** | Doc-drift test | **P14.4** | Every intra-repo markdown link in `arch.md` + `roadmap_deepseek.md` resolves |
| **T1.5** | Collection-count guard | **P14.5** | Test count pinned per subpackage |
| **T2** | Config suite | **P2.1** | Round-trip, preset survival, unknown-key warnings, validation, live-mutation smoke, shim retirement — folded into nested SimConfig tests |
| **T3.1** | SpatialIndex contract | **P2.3–P2.4** | Global indices, toroidal cross-seam, implementations interchangeable |
| **T3.2** | ForceMode contract | **P2.2** | Registered + instantiable, active-mask respect, determinism, speed_mode honored |
| **T3.3** | Integration contract | **P0.9–P0.10** | All speed_mode×move×inertia combinations, per-bird max_speed, toroidal wrap exactness |
| **T3.4** | Extension contract | **P2.6** | Live toggles take effect next frame, `threat_prox` published |
| **T3.5** | Engine/commands | **P2.2, P10.4** | Commands drained at step start, control clipping bounds, queue survives interleaving |
| **T3.6** | Metrics export schema | **P9.10** | `to_dict()` JSON round-trip, pinned key set, Recorder CSV headers match |
| **T4.1** | Holey-mask matrix | **P2.9** | Every mode × `holey_flock` fixture × 20 steps — no exception, invariants hold |
| **T4.2** | Lifecycle | **P0.6** | `add_boids`/`remove_boids` carry species column, index rebuilt, metrics survive N_active==0 |
| **T4.3** | Determinism matrix | **P13.5** | Same seed → bit-identical across (mode × num_threads × jitter × numba) |
| **T4.4** | Metamorphic invariances | **P9** | Order parameter & nematic S invariant under SO(3), disp/gyration translation-invariant, permutation invariance |
| **T5** | Viz/capture suites | **P8** | FBO depth attachment, VAO rebuilt after growth, render purity, GPU-free matplotlib fallback, capture pipeline |
| **T6** | Perf/quality guards | **P13** | `benchmark()` returns positive floats, per-mode step-time budgets, scaling checkpoints, 24h soak |

---

## Science Portfolio (S1–S7)

### S1 — Scientific Correctness

| Old | Description | New |
|---|---|---|
| **S1.1** | Occlusion culling | **P1.1** (exact `α = asin(min(b/d,1))`, closest-first sweep, visible iff no nearer cap) |
| **S1.1a** | Anisotropy identity | **P1.1** (`anisotropy=1.0` vs default → identical `δ̂, visible, Θ`) |
| **S1.2** | Θ probabilistic union | **P1.2** (`Θ = 1 − Π(1−Ω_j/4π)`, sub-additive, ∈ [0,1]) |
| **S1.3** | δ̂ boundary-length weighted | **P1.3** (`δ̂ = Σ sin α_j d̂_j / Σ sin α_j`, no magnitude clamp) |
| **S1.4** | Pearce noise term φn + weight constraint | **P10.6** (`φn = 1−φp−φa`, input handler renormalises, φp+φa≤1) |
| **S1.5** | Force-kernel corrections | **P1.7** (separation 1/d² unit-vector, cohesion bounded, noise ×scale) |
| **S1.6** | Steric clamp | **P1.6** (`‖F‖ ≤ max_force` after 1/d² sum) |
| **S1.7** | Vicsek update corrected | **P1.8** (memory term `û_old`, tangent-plane noise `n_⊥ = g−(g·û)û`, η interpolation, D live) |
| **S1.8** | Metrics formula fixes | **P0.13, P1.9** (thickness = `√(λ₃/λ₁)`, `compute_h2` → inf when disconnected, symmetrization) |

### S2.A — Field/Blob + Threat (Track A)

| Old | Description | New |
|---|---|---|
| **S2.A1** | Wander path | **P3.1** (`boundedUnitTravel`, `‖path‖≤1` guaranteed, domain-corner bug fix) |
| **S2.A2** | Blob anchors + phase weights | **P3.2** (5 Lissajous anchors, cyclic phase weights `w_k = max(0, 1−wrap_dist·7.5)²`) |
| **S2.A3** | Leader/chaser | **P3.3** (7 seed groups, lagged anchor, golden-angle shells, `T = lerp(T_legacy, chase_target, chase)`) |
| **S2.A4** | Shell + cavity | **P3.4** (`R_blob` oscillating, `F = −d̂(d−R_blob)·coh·1.35`, inner floor push-out) |
| **S2.A5** | Remaining 13 terms | **P3.5–P3.6** (slot repulsion ±{1,7,31}, tangential orbital, buoyancy z-only, curl flow, fold noise, drag, drift) |
| **S2.A6** | Ripples | **P3.7** (3 trains, `env=smoothstep(0.6,1.7)·(1−ss(6.2,8.8))`, moving Lissajous origins, twist) |
| **S2.A7** | Inertia / panic / blackening | **P3.8** (ceiling raise, NOT compound ×; `sep_eff = sep·(2−black)`, `coh_eff = coh·black`) |
| **S2.A8** | Threat FSM + force bundle | **P3.9** (capture/pass/clear phases, Rodrigues `rotate_toward`, push/wake/split/wave bundle) |
| **S2.A9** | Blob init + presets | **P3.10** (5 fixed centres, ∛-uniform shells, 7 field presets `conf/field_*.yaml`) |

### S2.B — Reynolds Variants + Ecology (Track B)

| Old | Description | New |
|---|---|---|
| **S2.B1** | Hybrid filter + dual radii | **P4.1** (metric+topological: `d<visual_range` AND among first `influence_count` (7)) |
| **S2.B2** | Update-order fidelity | **P4.2** (predator boost → accel_scale → clamp → v+=a → noise → ceiling → move) |
| **S2.B3** | Predator boids (species) | **P4.3** (1.8× speed, hard-zero align+coh when predator perceived, escape replaces separation) |
| **S2.B4** | Physical metrics | **P4.4** (`F=m·k_a·⟨|a|⟩` (N), `P=m·⟨|k_a a·k_v v|⟩` (W), `L=m·(r−CoM)×(k_v·v)`) |
| **S2.B5** | Parameter jitter | **P4.5** (effective weights per frame from `flock.rng`, config never mutated) |
| **S2.B6** | Parallel two-phase | **P4.6** (batched `query_knn_batch` + vectorised gather/reduce, no per-bird Python loops) |
| **S2.B7** | Sphere centring + asymptotic wall | **P4.7** (centre on C, soft: `Δv = −μ·r̂/max(R−r,0.05R)`) |
| **S2.B8** | Ecology completion | **P4.8** (logistic dusk, coherence gate, seasonal model `cos(2π·(day−15)/365)`, Oct–Mar) |
| **S2.B9** | Velocity-init variants | **P4.9** (`cube` / `speed_uniform` / `tangential` / `fixed`, selector `cfg.flock.velocity_init`) |
| **S2.B10** | Numba force kernels | **P4.10** (`@njit(parallel=True)`, `fastmath` only when `detail_level==0`, IEEE otherwise) |

### S2.C — Angle Mode (Track C)

| Old | Description | New |
|---|---|---|
| **S2.C1** | Steering core | **P5.1** (axis-angle `φ=acos(clamp(ĥ·t̂,−1,1))`, Rodrigues rotation, dead zone) |
| **S2.C2** | Unified neighbour modes | **P5.2** (7 closest within `b·12`: flee/mid-range/far, 3D `normalize(ĉ+m̂)`) |
| **S2.C3** | Adaptive speed | **P5.3** (linear / quadratic / softened, self-regulating density) |
| **S2.C4** | Edge handling | **P5.4** (cube: nearest face inward normal; sphere: `normalize(C−p)`; rate ramp) |
| **S2.C5** | Heading jitter | **P5.5** (±`jitter_deg`° about random axis before steering) |
| **S2.C6** | Incremental grid | **P5.6** (per-bird `last_cell`, re-file on crossing, behind `SpatialIndex` protocol) |
| **S2.C7** | Body-unit radii | **P5.7** (`sep/align/range_radius_bodies` scale with `boid_size`) |

### S2.D — Vicsek Predator–Prey (Track D)

| Old | Description | New |
|---|---|---|
| **S2.D1** | Species dynamics | **P6.1–P6.2** (fear-weighted alignment, predator hunting, `weight_afraid` (3.0), all-predators early-out) |
| **S2.D2** | Asymmetric collisions | **P6.3** (same-type: each moves `(R_avoid−d)/2`; prey–predator: prey takes full correction) |
| **S2.D3** | Prey-only metrics | **P6** (α of aligned prey + one orthogonal predator == 1.0) |

### S2.E — Influencer (Track E)

| Old | Description | New |
|---|---|---|
| **S2.E1** | Trajectory | **P7.1** (persistent tick, one per substep, verbatim `target_pos` from murmuratR) |
| **S2.E2** | Move-then-steer | **P7.2** (`p += d̂_old·v0·dt` → recompute `t̂` → blend, `owns_positions=True`) |
| **S2.E3** | Influence | **P7.3** (rank-by-target-distance: `(1−(i/(N−1))·0.8)^rank_exponent`, floor 0.055) |
| **S2.E4** | Density-scaled init | **P7.4** (Gaussian `σ=N^(1/3)·sep·s`, zero initial directions) |
| **S2.E5** | Diagnostics | **P7.5** (per-frame `min/max ‖p−T‖` → `FlockMetrics.target_dist_min/max`) |

### S3 — Metrics & Analysis

| Old | Description | New |
|---|---|---|
| **S3.1** | Nematic order | **P9.1** (`Q = (3/2)(ûᵀû)/N − ½I`, `S = λ_max(Q)`, `order: polar|nematic`) |
| **S3.2** | MSD(τ) curve | **P9.2** (unwrapped accumulation, log-spaced lags, ballistic→diffusive crossover) |
| **S3.3** | Shape→m\* | **P9.5** (`m*=9.78 + clamp((aspect−1)/2,0,1)·(6.05−9.78)`) |
| **S3.4** | η(m) marginal efficiency | **P9.6** (`η(m) = (H₂(m₀)−H₂(m))/(m−m₀)`, +∞ at connectivity transition) |
| **S3.5** | Hull-volume τρ | **P9.3** (`ρ=N/ConvexHull.volume`, ring buffer, autocorrelation τ) |
| **S3.6** | Θ′ silhouette | **P9.4** (project positions ⊥ observer axis, rasterize disks, union fraction) |
| **S3.7** | Robust gyration + ideal exponent | **P9.7** (median centroid, top-15% trim, `R_g = √mean(r²_kept)`, `ideal_exponent = −0.5`) |
| **S3.8** | Motion metrics | **P9.8** (velocity deviation, boundary overshoot, normalized L, altitude deviation) |
| **S3.9** | Rewards module | **P9.9** (weighted composite, `reward_faithful_signs` flag) |
| **S3.10** | Export schema | **P9.10** (`FlockMetrics.to_dict()` adopted end-to-end, JSON round-trip, pinned key set) |

### S4 — Rendering & Capture

| Old | Description | New |
|---|---|---|
| **S4.1** | Sphere impostors | **P8.1** (billboard quads, fragment `discard > 1`, Blinn-Phong shade, speed-stretched ellipsoid) |
| **S4.2** | Depth cues + Fresnel rim | **P8.2** (size ∝ 1/depth^k, alpha × depth × speed × rim factors) |
| **S4.3** | Trails ×4 | **P8.3** (velocity-stretched, screen-space accumulation, ring via position_history buffer, CPU lines) |
| **S4.4** | Winged mesh + flap | **P8.4** (6-triangle body+wings+tail, `u_Flap = ±0.5` toggled every 100 frames) |
| **S4.5** | Gradient sky | **P8.4** (fullscreen quad, top→bottom theme-overridable, depth off) |
| **S4.6** | Colour channels | **P8.5** (per-bird hue from `seeds` (HSV), predator→red, theme material tables (ambient+diffuse pairs)) |
| **S4.7** | Alpha-accumulation density | **P8.11** (α≈0.2 sprites, blending on, depth-write off) |
| **S4.8** | Views | **P8.8** (dual-viewport, orthographic top/side presets, keys 7/8/9) |
| **S4.9** | Capture pipeline | **P8.7** (cinematic sweep, prewarm, env overrides, GIF `optimize=True, disposal=2`) |
| **S4.10** | Adaptive quality | **P8.6** (EMA averaging, budget, risk classification, hysteresis ladder, recovery) |
| **S4.11** | Fixed-timestep accumulator | **P8.10** (`acc += clamp(frame_dt,0,1/20)`, render lerp, 30fps≡60fps physics) |

### S5 — UX & Tooling

| Old | Description | New |
|---|---|---|
| **S5.1** | Preset keys a–h,w | **P10.1** (8 presets, printed labels, key-g skipped for grid toggle) |
| **S5.2** | Full title readout | **P10.2** (mode, N, φp/φa/σ, α, Θ, Θ′, L, σr, τρ, FPS — rebuilt every 20th frame) |
| **S5.3** | Slider HUD | **P10.3** (5 sliders: sep/coh/align/avoid/noise, ortho-pass track+knob quads, TAB toggle) |
| **S5.4** | Cursor-ray spawning | **P10.4** (mouse unprojection, left-click→bird, right-click→predator, PageUp/Dn) |
| **S5.5** | CLI + facade | **P10.5** (`--set key.subkey=value`, `--print-config`, `pymurmur.Simulation(**params)`, `benchmark()`) |
| **S5.6** | Run logging | *(deferred)* (structured log to `output/` — covered by Appendix C Cross-Cutting Concerns) |

### S6 — EvoFlock

| Old | Description | New |
|---|---|---|
| **S6.1** | SSGA fidelity | **P11.1** (uniform crossover of best 2, Gaussian mutation, worst-of-3 negative selection) |
| **S6.2** | Worst-of-4 evaluation | **P11.2** (4 sims per candidate, fixed per-sim seeds, min-reduction) |
| **S6.3** | Objectives | **P11.3** (separation trapezoid, speed band [19,21] m/s, curvature κ, hypervolume) |
| **S6.4** | SDF obstacle layer | **P0.14, P11.4** (5 primitives in P0.14, `ObstacleScene` CSG tree in P11.4, collision detection + correction) |
| **S6.5** | Expanded gene set | **P11.5** (forward force, perception cones, fly-away, predictive avoidance, k=7, σ integer, speed_min_factor) |
| **S6.6** | Protocol | **P11.6** (best genome + Pareto front persistence, emergent-alignment experiment `α>0.5`) |

### S7 — MARL Bridge

| Old | Description | New |
|---|---|---|
| **S7.1** | MARL mode: deferred global rules | **P12.1** (control applies first, move, then rules prep next step; `rule_weight=0.01`) |
| **S7.2** | Gymnasium wrapper | **P12.2** (`Box(−1,1,(6N,))` obs, `Box(−1,1,(3N,))` action, seeded reset, truncate at 500) |
| **S7.3** | Scripts | **P12.3** (`train_marl.py` PPO 5000 steps, `rollout_marl.py` 500 steps, trained > random by ≥20%) |

---

## Items Added From Gap Analysis (No Direct Old Mapping)

These were identified by cross-referencing roadmap_deepseek.md against all 24 `sci/` audit files:

| # | Item | New Location | Source |
|---|---|---|---|
| 1 | Grid-mode separation normalization | **P3.11** | `sci/todo_claude_sci2.md` §2.3 |
| 2 | 1.45·R_blob floating boundary for field mode | **P3.12** | `sci/todo_claude.md` §8 |
| 3 | Functional ForceTerm + composeForces composition | **P2.10** | `sci/todo_claude.md` §8, `sci/todo_claude_sci1.md` §15 |
| 4 | Position init variants (sphere_shell, grid, gaussian, box, blob) | **P0.15** | `sci/todo_claude3.md`, `sci/todo_claude_sci2.md` §12 |
| 5 | Theme material tables (ambient+diffuse pairs) | **P8.5** | `sci/todo_claude_sci9.md` §4 |
| 6 | Altitude-deviation reward metric | **P9.8** | `sci/todo_claude_sci7.md` §2 |
| 7 | Zero-speed deterministic fallback `(minSpeed, 0, 0)` | **P0.9** | `sci/todo_claude3.md` §4.4 |
| 8 | Desktop pilotable-flock mode | **P7.6** | `sci/todo_claude_sci1.md` §7.2 |
| 9 | Multi-flock support (reconsidered in Appendix A) | **Appx A** | `sci/todo_claude_sci4.md` §9 |

---

## Phase Summary

| Old Scheme | New Scheme |
|---|---|
| D0–D9 foundation (10½ days) | P0–P2 (foundations + correctness + contracts, 11.5 days) |
| T0–T6 infrastructure (interleaved) | Distributed across P0–P14 (tests inline with feature phases) |
| S1 correctness (2 days) | P1 (scientific correctness, 3 days) |
| S2 tracks A/B/C/D/E (4/3½/2/2/1½ days) | P3/P4/P5/P6/P7 (field/reynolds/angle/vicsek/influencer, 5.5+3.5+2+1.5+1.5 days) |
| S3 metrics (2 days) | P9 (metrics & analysis, 3 days) |
| S4 rendering/capture (4 days) | P8 (rendering & capture, 5 days) |
| S5 UX/tooling (2 days) | P10 (UX & tooling, 2 days) |
| S6 EvoFlock (3 days) | P11 (EvoFlock, 3 days) |
| S7 MARL bridge (2 days) | P12 (MARL bridge, 2 days) |
| *(none — gap items)* | P13 scaling (2 days) + P14 guard rails (1.5 days) |

**Total:** ~38 working days old scheme → ~42 working days new scheme (new phases P13–P14 + gap items account for the difference).

---

*Generated from a full cross-reference of `roadmap.md` and `roadmap_deepseek.md` against all 24 `sci/` audit files, July 2026.*


## Sub-Item Mappings (implicit coverage)

The following sub-items appear in `roadmap.md` Appendix A but roll up into
their parent's mapping. Added for 100% cross-check coverage.

| Old ID | Parent | Maps to | Description |
|---|---|---|---|
| D0.1 | D0 | P0.1 | Project structure setup |
| D0.2 | D0 | P0.2 | Architecture test skeleton |
| D0.4 | D0 | P0.1 | Conftest fixtures |
| D9.1 | D9 | P13.1 | Step-time regression suite |
| D9.2 | D9 | P13.2 | Scaling checkpoints |
| D9.3 | D9 | P13.3 | Memory audit |
| D9.4 | D9 | P13.4 | 24-hour soak test |
| D9.6 | D9 | P13.5 | Determinism matrix |
| S3.11 | S3 | P9.1 | Nematic order parameter (was implicit in S3 scope) |
| S3.6a | S3 | P9.5 | Shape→m* variant (was S3.6 sub-clause) |
| S4.4a | S4 | P10.3 | Slider HUD variant (was S4.4 sub-clause) |
| S5.6 | S5 | P11.3 | Curvature objective (was S5 sub-item) |
| T0.1 | T0 | P0.2 | FORBIDDEN_EDGES declaration |
| T0.2 | T0 | P0.2 | ALLOWED_EDGES skeleton |
| T0.3 | T0 | P0.1 | Golden test harness |
| T3 | T3 | P4 | Spatial mode conformance suite (phase-level) |
| T4 | T4 | P5 | Angle mode conformance suite (phase-level) |
| T5.4 | T5 | P8.9 | GPU-free capture fallback test |
| T6.1 | T6 | P13.1 | Per-mode budget test |
| T6.2 | T6 | P13.5 | Determinism matrix sub-item |



---


## Appendix H — Step Index (119 Steps)

> Merged from `#appendix-h--step-index-119-steps` (now removed). Flat table of all 119 step
> headings with their phase, level, file paths, test paths, and citations.
> Useful for CI integration and coverage tracking.

| Phase | Step | Title | Level | Files | Test Files | Citations |
|---|---|---|---|---|---|---|
| P0 | P0.1 | Golden trajectory harness | L3 | `test/regenerate_golden.py`, `test/test_golden.py`. **Data:*... | — | `todo_claude.md` T14. |
| P0 | P0.2 | Architecture test skeleton | L3 | `test/test_architecture.py` | — | `roadmap.md` D0.2. |
| P0 | P0.3 | Physics invariant fuzz | L0 | `test/physics/test_boid.py` | — | `todo_claude.md` T13. |
| P0 | P0.4 | Single seeded RNG | L1 | `physics/flock.py` | — | `git0` F1. |
| P0 | P0.5 | Smoothed swarm centre | L1 | `physics/flock.py` | — | `git6` R1. |
| P0 | P0.6 | Species column | L1 | `physics/flock.py` | — | `git0` F6. |
| P0 | P0.7 | Previous positions + acceleration stash | L1 | `physics/flock.py` | — | `git0` F8. |
| P0 | P0.8 | Per-bird max_speed array | L1 | `physics/flock.py` | — | `git6` R8. |
| P0 | P0.9 | Integration variants | L0 | `physics/boid.py::integrate` | — | `git0` F3. |
| P0 | P0.10 | Safety rails: dt clamp + NaN guard | L0 | `physics/boid.py::integrate` | — | `sci.md` §22. |
| P0 | P0.11 | Capability probing | L3 | `pymurmur/__main__.py` | — | `sci/todo_claude_sci2.md`. |
| P0 | P0.12 | Math helpers in core/types.py | L0 | `core/types.py`. **Tests:** `test/core/test_types.py` | `test/core/test_types.py` | `todo_claude.md` §3 (Rodrigues), `sci/to |
| P0 | P0.13 | H₂ disconnected → inf fix | L0 | `analysis/metrics.py::compute_h2` | — | `todo_claude.md` §10. |
| P0 | P0.14 | SDF primitives | L0 | `physics/obstacles.py` (new). **Tests:** `test/physics/test_... | `test/physics/test_obstacles.py` | `new10_sci.md` §5. |
| P0 | P0.15 | Position init variants | L0 | `physics/boid.py`. **Tests:** `test/physics/test_boid.py` | `test/physics/test_boid.py` | `sci/todo_claude3.md` position init, `sc |
| P1 | P1.1 | Occlusion culling: visibility test | L0 | `physics/occlusion.py::spherical_cap_occlusion`. **Tests:** ... | `test/physics/test_occlusion.py` | `todo_claude.md` §1, T1. |
| P1 | P1.2 | Θ as probabilistic union | L0 | physics/occlusion.py::spherical_cap_occlusion; test/physics/... | — | `todo_claude.md` §2, T2. |
| P1 | P1.3 | δ̂ boundary-length weighted | L0 | physics/occlusion.py::spherical_cap_occlusion; test/physics/... | — | `todo_claude.md` §3, T3. |
| P1 | P1.4 | Exact α = asin(min(b/d, 1)) | L0 | physics/occlusion.py::spherical_cap_occlusion; test/physics/... | — | `todo_claude.md` §1. |
| P1 | P1.5 | Candidate cutoff at 64 | L0 | physics/occlusion.py::spherical_cap_occlusion; test/physics/... | — | `todo_claude.md` §4. |
| P1 | P1.6 | Steric clamp to max_force | L0 | `physics/steric.py::steric_force` | — | `todo_claude.md` §14, T6. |
| P1 | P1.7 | Force kernel fixes | L0 | `physics/forces/_base.py`. **Tests:** `test/physics/forces/t... | `test/physics/forces/test_kernels.py` | `git0` W3.1, `sci/todo_claude_sci3.md` § |
| P1 | P1.8 | Vicsek update: memory term + tangent-plane noise | L0 | `physics/forces/vicsek.py`. **Tests:** `test/physics/forces/... | `test/physics/forces/test_vicsek_core.py` | `git2` R1, `sci/todo_claude_sci8.md` §1. |
| P1 | P1.9 | Thickness ratio fix | L0 | `analysis/metrics.py::compute_shape` | — | `todo_claude.md` E13. |
| P1 | P1.10 | Θ reports N/A in non-projection modes | L0 | `analysis/metrics.py`. **Tests:** `test/analysis/test_metric... | `test/analysis/test_metrics.py` | `todo_claude3.md` §4, `todo_claude2.md`  |
| P2 | P2.1 | Nested SimConfig | L1 | `core/config.py`, `conf/*.yaml`. **Tests:** `test/core/test_... | `test/core/test_config.py` | `roadmap.md` D1, `git0` F0, `todo_claude |
| P2 | P2.2 | ForceMode protocol + registry | L1 | `physics/forces/_mode.py` (new), `simulation/engine.py`. **T... | `test/physics/forces/test_mode_contract.py` | `roadmap.md` D2, `git0` F2, `todo_claude |
| P2 | P2.3 | SpatialIndex protocol | L1 | `core/types.py`. **Tests:** `test/physics/test_spatial_index... | `test/physics/test_spatial_index_contract.py` | `roadmap.md` D5, `todo_claude1.md` §3. |
| P2 | P2.4 | KDTreeIndex returns global indices | L1 | `physics/flock.py::KDTreeIndex`. **Tests:** `test/physics/te... | `test/physics/test_flock.py` | `roadmap.md` D5. |
| P2 | P2.5 | Ghost-cell replication + modulo cells | L1 | `physics/flock.py::SpatialHashGrid` | — | `git3` R2, `sci/todo_claude_sci9.md` §13 |
| P2 | P2.6 | StepContext + Extension widening | L1 | `physics/extensions/_base.py`. **Tests:** `test/physics/exte... | `test/physics/extensions/test_extensions.py` | `roadmap.md` D6. |
| P2 | P2.7 | InstanceSchema + VAO discipline | L2 | `viz/renderer.py`. **Tests:** `test/viz/test_renderer.py` | `test/viz/test_renderer.py` | `todo_claude.md` E1, E2. |
| P2 | P2.8 | PyGLM matrix upload | L2 | `viz/renderer.py` | — | `todo_claude.md` E3. |
| P2 | P2.9 | Holey-mask contract tests | L1 | `test/physics/test_composition.py` | — | `roadmap.md` T4.1. |
| P2 | P2.10 | Functional ForceTerm composition | L1 | `physics/forces/_base.py` (extend). **Tests:** `test/physics... | `test/physics/forces/test_force_terms.py` | `sci/todo_claude.md` §8 (composeForces p |
| P3 | P3.1 | Wander path (boundedUnitTravel) | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | `‖path‖ ≤ 1` for 10⁶ fuzzed t; heading continuous (‖h(t+ε)−h... | `git6` R2, `sci/todo_claude_sci1.md` §8. |
| P3 | P3.2 | Five Lissajous blob anchors + cyclic phase weights | L1 | `physics/forces/field.py::FieldMode` | 2K birds → k-means finds ≥4 clusters at t=30s; per-bird targ... | `git6` R3, `sci/todo_claude_sci1.md` §2. |
| P3 | P3.3 | Leader/chaser groups | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | test/physics/forces/test_field.py; test/physics/extensions/t... | `git6` R4, `sci/todo_claude_sci1.md` §2. |
| P3 | P3.4 | Shell force + inner cavity | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | Settled 5K blob — centre voxel density < 0.3× shell band den... | `git6` R5, `sci/todo_claude_sci1.md` §2. |
| P3 | P3.5 | Slot repulsion | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | kernel zero at r_slot and beyond; continuous at boundary | `git6` R6, `sci/todo_claude_sci2.md` §3. |
| P3 | P3.6 | Remaining 6 field terms | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | test/physics/forces/test_field.py; test/physics/extensions/t... | `git6` R6, `sci/todo_claude_sci1.md` §2. |
| P3 | P3.7 | Ripple envelopes (vectorised) | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | test/physics/forces/test_field.py; test/physics/extensions/t... | `git6` R7, `sci/todo_claude_sci1.md` §2. |
| P3 | P3.8 | Bounded panic + blackening | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | panicked speed ≤ 2.35·v0 always; near-threat pair separation... | `git6` R8, `sci/todo_claude_sci1.md` §3. |
| P3 | P3.9 | Threat FSM + force bundle | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | test/physics/forces/test_field.py; test/physics/extensions/t... | `git6` R9, `sci/todo_claude_sci1.md` §4. |
| P3 | P3.10 | Blob init + 7 field presets | L1 | physics/forces/field.py; physics/extensions/predator.py; phy... | All 7 presets load with documented values; frame-0 lobes vis... | `git6` R12, `sci/todo_claude_sci2.md` §1 |
| P3 | P3.11 | Grid-mode separation normalization | L1 | `physics/forces/field.py` | test/physics/forces/test_field.py; test/physics/extensions/t... | `sci/todo_claude_sci2.md` §2.3 ("Separat |
| P3 | P3.12 | Field-mode 1.45·R_blob floating boundary | L1 | `physics/forces/field.py::FieldMode` | test/physics/forces/test_field.py; test/physics/extensions/t... | `sci/todo_claude.md` §8, `sci/todo_claud |
| P4 | P4.1 | Hybrid metric+topological filter | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | Neighbour set respects radius AND cap; alignment set ⊆ cohes... | `git5` R1, `sci/todo_claude_sci3.md` §4. |
| P4 | P4.2 | Force accumulation order | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | test/physics/forces/test_spatial_variants.py; speed_real = k... | `git3` R1, `sci/todo_claude_sci3.md` §1, |
| P4 | P4.3 | Predator boids (species) | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | test/physics/forces/test_spatial_variants.py; speed_real = k... | `git3` R3, `sci/todo_claude_sci5.md` §2. |
| P4 | P4.4 | Physical metrics | L1 | `analysis/metrics.py` — add `speed_real, accel_real, force_r... | Hand-set v → `speed_real = k_v·|v|` exactly; E ≈ P̄·elapsed ... | `git5` R3, `sci/todo_claude_sci3.md` §3. |
| P4 | P4.5 | Per-frame parameter jitter | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | spacing_std(on) > spacing_std(off), same seed; config unchan... | `git5` R4, `sci/todo_claude_sci3.md` §5. |
| P4 | P4.6 | Parallel two-phase update | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | test/physics/forces/test_spatial_variants.py; speed_real = k... | `git3` R4, `sci/todo_claude_sci5.md` §4. |
| P4 | P4.7 | Sphere centring + asymptotic wall | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | test/physics/forces/test_spatial_variants.py; speed_real = k... | `git5` R2, `sci/todo_claude_sci3.md` §2. |
| P4 | P4.8 | Ecology completion | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | test/physics/forces/test_spatial_variants.py; speed_real = k... | `todo_claude.md` §5–8, `sci/todo_claude_ |
| P4 | P4.9 | Velocity-init variants | L1 | physics/forces/spatial.py; physics/forces/_kernels.py; analy... | cube E|v|≈0.816·v0 ±5%; speed_uniform in-band, non-constant;... | `todo_claude.md` E12, `sci/todo_claude_s |
| P4 | P4.10 | Numba force kernels | L1 | `physics/forces/_kernels.py` (new) | test/physics/forces/test_spatial_variants.py; speed_real = k... | `arch.md` §9, `sci/todo_claude_sci9.md`  |
| P5 | P5.1 | Steering core | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | 180° turn completes in π/rate seconds ±1 frame; per-frame he... | `git4` R1, `sci/todo_claude_sci4.md` §1. |
| P5 | P5.2 | Unified neighbour modes | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | boid_size | `git4` R2, `sci/todo_claude_sci4.md` §2. |
| P5 | P5.3 | Adaptive speed | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | m=0 → base+35 (linear); m≥7 → base; median 7th-NN distance c... | `git4` R3, `sci/todo_claude_sci4.md` §3. |
| P5 | P5.4 | Edge handling | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | boid_size | `git4` R4, `sci/todo_claude_sci4.md` §4. |
| P5 | P5.5 | Heading jitter | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | boid_size | `git4` R5, `sci/todo_claude_sci4.md` §5. |
| P5 | P5.6 | Incremental spatial grid | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | Neighbour sets == full-rebuild sets over 500 random-walk fra... | `git4` R6, `sci/todo_claude_sci4.md` §6. |
| P5 | P5.7 | Body-unit scale invariance | L1 | physics/forces/angle.py; test/physics/forces/test_angle.py | Doubling `boid_size` doubles all three thresholds; 2×-scale ... | `git4` R8, `sci/todo_claude_sci4.md` §10 |
| P6 | P6.1 | Fear-weighted alignment | L1 | physics/forces/vicsek.py | Stationary predator at centre → prey ⟨û·r̂⟩ > 0.8 within 5 s... | `git2` R2, `sci/todo_claude_sci8.md` §2. |
| P6 | P6.2 | Predator agent | L1 | physics/forces/vicsek.py | Monotone pursuit (≥90% of steps close distance); n_prey=0 → ... | `git2` R3, `sci/todo_claude_sci8.md` §2. |
| P6 | P6.3 | Asymmetric position collisions | L1 | physics/forces/vicsek.py | test/physics/forces/test_vicsek_species.py | `git2` R5, `sci/todo_claude_sci8.md` §3. |
| P7 | P7.1 | Persistent tick + Lissajous target | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | `T(t)` at t∈{0,970,2170} equals hand values; in-domain for s... | `git1` R1–R2, `sci/todo_claude_sci6.md`  |
| P7 | P7.2 | Move-then-steer at unit speed | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | Frozen target → convergence to hover/orbit; one-step lag aft... | `git1` R3, `sci/todo_claude_sci6.md` §1. |
| P7 | P7.3 | Rank-by-target-distance influence | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | test/physics/forces/test_influencer.py; T(t) | `git1` R4, `sci/todo_claude_sci6.md` §3. |
| P7 | P7.4 | Density-scaled init | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | Init density equal across N∈{100,1000,8000} (±10%); frame-0 ... | `git1` R5, `sci/todo_claude_sci6.md` §5. |
| P7 | P7.5 | Distance diagnostics | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | test/physics/forces/test_influencer.py; T(t) | `git1` R6, `sci/todo_claude_sci6.md` §6. |
| P7 | P7.6 | Desktop pilotable-flock mode | L1 | physics/forces/influencer.py; physics/forces/influencer.py::... | test/physics/forces/test_influencer.py; T(t) | `sci/todo_claude_sci1.md` §7.2 (pilot-aw |
| P8 | P8.1 | Sphere impostors + speed-stretched ellipsoid | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | Centre pixel brighter than rim; corners = background | `git6` R10, `sci/todo_claude_sci1.md` §6 |
| P8 | P8.2 | Depth cues + Fresnel rim | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | Near bird renders larger & more opaque than far | `git6` R10, `sci/todo_claude_sci9.md` §4 |
| P8 | P8.3 | Trail rendering ×4 | L2 | `viz/trails.py` (new), `physics/flock.py` (position_history) | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `git6` R10, `sci/todo_claude_sci2.md` §9 |
| P8 | P8.4 | Winged flapping mesh + gradient sky | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `git5` R6–R7, `sci/todo_claude_sci3.md`  |
| P8 | P8.5 | Colour channels + theme wiring | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `git6` R10, `sci/todo_claude_sci4.md` §1 |
| P8 | P8.6 | Adaptive quality wired | L2 | `analysis/perf.py` (governor logic), `viz/visualizer.py` (co... | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `git6` R11, `sci/todo_claude_sci1.md` §1 |
| P8 | P8.7 | Cinematic capture sweep | L2 | `capture/recorder.py` | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `todo_claude.md` E7–E10. |
| P8 | P8.8 | Dual-view + orthographic presets | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `sci/todo_claude_sci7.md` §5. |
| P8 | P8.9 | GPU-free matplotlib fallback | L2 | `capture/mpl_recorder.py` (new) | test/viz/test_renderer.py; test/viz/test_trails.py; test/viz... | `sci/todo_claude_sci7.md` §5. |
| P8 | P8.10 | Fixed-timestep accumulator + interpolation | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | 30fps vs 60fps → identical physics after same elapsed time | `sci/todo_claude_sci5.md` §11, `sci/todo |
| P8 | P8.11 | Alpha-accumulation density mode | L2 | viz/renderer.py; viz/shaders.py; viz/camera.py; viz/hud.py; ... | Cluster centre darker than single bird | `sci/todo_claude_sci6.md` §8. |
| P9 | P9.1 | Nematic order parameter | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | test/analysis/test_metrics.py; test/analysis/test_metrics_in... | `git2` R7, `sci/todo_claude_sci8.md` §4. |
| P9 | P9.2 | MSD(τ) curve | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | test/analysis/test_metrics.py; test/analysis/test_metrics_in... | `git2` R8, `sci/todo_claude_sci8.md` §4. |
| P9 | P9.3 | Hull-volume τρ | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | Cube hull = edge³±1e-3; coplanar→0; constant series→τ=0; per... | `todo_claude.md` §11. |
| P9 | P9.4 | Silhouette Θ′ | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | Flat wall ⊥ axis → silhouette≈1 while voxel Θ′≪1; two co-pro... | `todo_claude.md` §12. |
| P9 | P9.5 | Shape→m* | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | aspect 1→9.78, ≥3→6.05; monotone; thin flock≤7, round≥8 | `todo_claude.md` §9. |
| P9 | P9.6 | η(m) marginal efficiency | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | Connectivity transition → `math.isinf`; both disconnected → ... | `todo_claude.md` §10. |
| P9 | P9.7 | Robust gyration + ideal exponent | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | test/analysis/test_metrics.py; test/analysis/test_metrics_in... | `todo_claude.md` §13. |
| P9 | P9.8 | Motion metrics | L1 | analysis/metrics.py; analysis/rewards.py; analysis/phase_dia... | test/analysis/test_metrics.py; test/analysis/test_metrics_in... | `todo_claude.md` §15, `sci/todo_claude_s |
| P9 | P9.9 | Rewards module | L1 | `analysis/rewards.py` | test/analysis/test_metrics.py; test/analysis/test_metrics_in... | `sci/todo_claude_sci7.md` §2. |
| P9 | P9.10 | Export schema | L1 | `analysis/metrics.py::FlockMetrics.to_dict()` | JSON round-trip; pinned key set; Recorder CSV headers == sch... | `roadmap.md` D9.2. |
| P10 | P10.1 | Preset keys a–h,w | L2 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `todo_claude.md` E4. |
| P10 | P10.2 | Full title readout | L2 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `todo_claude.md` E6. |
| P10 | P10.3 | Slider HUD | L2 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `sci/todo_claude_sci3.md` §6. |
| P10 | P10.4 | Cursor-ray spawning | L2 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `sci/todo_claude_sci5.md` §7. |
| P10 | P10.5 | CLI + facade | L3 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `sci/todo_claude_sci5.md` §6–8. |
| P10 | P10.6 | φp+φa ≤ 1 constraint | L2 | analysis/presets.py; viz/input_control.py; viz/hud.py | test/viz/test_input.py; test/test_cli.py; test/test_facade.p... | `todo_claude.md` E5. |
| P11 | P11.1 | SSGA fidelity + uniform crossover | L2 | analysis/evoflock.py; physics/obstacles.py; physics/obstacle... | test/analysis/test_evoflock.py; @slow | `sci/todo_claude_git0.md` W8. |
| P11 | P11.2 | Worst-of-4 evaluation | L2 | analysis/evoflock.py; physics/obstacles.py; physics/obstacle... | Monkeypatched objectives [0.9,0.8,0.95,0.7] → fitness 0.7; s... | `sci/todo_claude_git0.md` W8. |
| P11 | P11.3 | Objectives | L2 | analysis/evoflock.py; physics/obstacles.py; physics/obstacle... | Trapezoid pinned at d/body∈{1.9→0, 2.5→1, 4→1, 5→0}; helix κ... | `sci/todo_claude_git0.md` W8. |
| P11 | P11.4 | SDF obstacle layer | L1 | `physics/obstacles.py::ObstacleScene` | test/analysis/test_evoflock.py; @slow | `sci/todo_claude_git0.md` W8. |
| P11 | P11.5 | Expanded gene set | L2 | analysis/evoflock.py; physics/obstacles.py; physics/obstacle... | test/analysis/test_evoflock.py; @slow | `sci/todo_claude_git0.md` W8. |
| P11 | P11.6 | Protocol | L2 | analysis/evoflock.py; physics/obstacles.py; physics/obstacle... | test/analysis/test_evoflock.py; @slow | `sci/todo_claude_git0.md` W8. |
| P12 | P12.1 | "marl" mode: deferred global rules | L1 | physics/forces/marl.py; analysis/gym_env.py; analysis/gym_en... | test/analysis/test_marl.py; pytest.importorskip("gymnasium")... | `sci/todo_claude_sci7.md` §1. |
| P12 | P12.2 | Gymnasium wrapper | L2 | `analysis/gym_env.py::MurmurationEnv` | test/analysis/test_marl.py; pytest.importorskip("gymnasium")... | `sci/todo_claude_sci7.md` §1. |
| P12 | P12.3 | Scripts (dependency-gated) | L3 | physics/forces/marl.py; analysis/gym_env.py; analysis/gym_en... | test/analysis/test_marl.py; pytest.importorskip("gymnasium")... | `sci/todo_claude_sci7.md` §A. |
| P13 | P13.1 | Step-time regression suite | L2 | test/test_performance.py; test/test_scaling.py | @slow; sys.getsizeof | `arch.md` §13. |
| P13 | P13.2 | Scaling checkpoints | L2 | test/test_performance.py; test/test_scaling.py | @slow; sys.getsizeof | `arch.md` §13. |
| P13 | P13.3 | Memory audit | L2 | test/test_performance.py; test/test_scaling.py | @slow; sys.getsizeof | `arch.md` §8. |
| P13 | P13.4 | 24-hour soak test | L2 | test/test_performance.py; test/test_scaling.py | @slow; sys.getsizeof | `sci/todo_claude_sci2.md` §21. |
| P13 | P13.5 | Determinism matrix | L2 | test/test_performance.py; test/test_scaling.py | @slow; sys.getsizeof | `roadmap.md` T4.3. |
| P14 | P14.1 | DAG matrix finalization | L3 | test/test_architecture.py; test/test_docs.py; test/test_docs... | — | `roadmap.md` T1.1. |
| P14 | P14.2 | Config-usage drift | L3 | test/test_architecture.py; test/test_docs.py; test/test_docs... | — | `roadmap.md` T1.2. |
| P14 | P14.3 | Strictly-3D guard | L3 | test/test_architecture.py; test/test_docs.py; test/test_docs... | — | `roadmap.md` T1.3. |
| P14 | P14.4 | Doc-drift test | L3 | test/test_architecture.py; test/test_docs.py; test/test_docs... | — | `roadmap.md` T1.4. |
| P14 | P14.5 | Collection-count guard | L3 | test/test_architecture.py; test/test_docs.py; test/test_docs... | — | `roadmap.md` T1.5. |




---

*Self-contained implementation roadmap, July 2026. Every phase above contains
the idea, math formulas (3D form), code sketches, target files, config fields,
test specifications, and acceptance criteria needed to implement it.
Provenance: the 24-file audit corpus in `sci/`.*
