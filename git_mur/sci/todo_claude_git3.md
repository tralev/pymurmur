# todo_claude_git3.md — Porting `rystrauss/boids` to pymurmur (3D)

**Source:** https://github.com/rystrauss/boids — a C++/SFML Reynolds boids
simulation with k-d tree spatial partitioning, OpenMP parallelism, predator
boids, toroidal distance, pybind11 Python bindings, and a CLI. Source files
verified: `src/Boid.{h,cpp}`, `src/Flock.cpp`, `src/Vector2D.cpp`,
`src/KDTree.{h,cpp}`, `src/Simulation.{h,cpp}`, `src/main.cpp`,
`src/pyboids.cpp`.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`), whose `spatial`
force mode copied this project's default weights but not its mechanics.

**What this file is.** Every idea and piece of math in the source that
pymurmur does not implement (or implements differently in a way that loses
the source's behaviour), re-derived for **3D**, with formulas, constants,
file paths, config fields, code sketches, and acceptance tests — so each
roadmap item is implementable from this file alone.

**Already implemented in pymurmur (do not redo):** the three-term Reynolds
decomposition as separate functions, a k-d tree spatial index
(`scipy.spatial.cKDTree` — superior to the hand-rolled tree; keep it),
per-frame tree rebuild, toroidal *position* wrapping with velocity preserved,
max-force clamping, +/− keyboard flock resizing, frame-rate limiting via
`pygame.time.Clock`, and this repo's exact default weight vector
(`sep 4.5 / align 0.65 / coh 0.75 / accel_scale 0.3 / max_speed 6 /
max_force 1 / perception 100 / separation_distance 20`) already sitting in
`SimConfig` — mostly as **dead fields**. The theme palettes exist but are
unwired.

**Source constants (verified in `Boid.h`):**

```
PREDATOR_ESCAPE_FACTOR       = 10_000_000
PREDATOR_SPEED_BOOST         = 1.8
PREDATOR_PERCEPTION_BOOST    = 1.5
PREDATOR_ACCELERATION_BOOST  = 1.4
```

---

## Conventions used throughout

- All vectors become 3-vectors; `normalize(v) = v/‖v‖ (0 if ‖v‖=0)` and
  `limit(v, m) = v·m/‖v‖ if ‖v‖>m else v` are the source's two primitives —
  add both as helpers in `pymurmur/core/types.py` (`normalize3`, `limit3`)
  and use them everywhere below.
- **RNG:** all randomness from the flock-owned seeded generator
  (`PhysicsFlock.rng = np.random.default_rng(config.seed)`).
- `idx = np.where(flock.active)[0]`; per-bird arrays gather/scatter through
  it. Domain `[0,W)×[0,H)×[0,D)`.

---

## R0 — Prerequisite: config fields and loading

**Idea.** Wire the already-present-but-dead fields and add the missing ones.
Also: pymurmur's YAML loader flattens `section: key:` to `key` without
prefixing, silently dropping keys whose dataclass field is prefixed and
letting `capture: width:` overwrite the domain `width`. If that loader fix
is not yet applied from another work stream, apply it now (code below);
otherwise skip step 1.

**Implementation.**

1. Loader fix in `SimConfig.from_file`
   ([pymurmur/core/config.py:149-164](pymurmur/core/config.py#L149-L164)):

```python
valid = {f.name for f in fields(cls)}
flat: dict[str, Any] = {}
for section, data in raw.items():
    if isinstance(data, dict):
        for k, v in data.items():
            pk = f"{section}_{k}"
            if pk in valid:   flat[pk] = v
            elif k in valid:  flat[k] = v
            else: warnings.warn(f"config: unknown key {section}.{k}")
    else:
        flat[section] = data
```

2. New/activated dataclass fields (defaults = source defaults):

```python
# ── Reynolds/rystrauss spatial mode ───────────────────────────
separation_distance: float = 20.0      # NEW: metric separation radius
speed_mode: str = "band"               # NEW: "band" (legacy) | "ceiling" (source)
noise_mode: str = "force"              # NEW: "force" (legacy) | "velocity" (source)
separation_kernel: str = "unit"        # NEW: "unit" (source) | "inverse_square" (legacy)
num_threads: int = -1                  # NEW: -1 = auto (all cores)
# acceleration_scale (0.3), visual_range→perception (100): exist, currently dead
# ── Predator boids ────────────────────────────────────────────
n_predator_boids: int = 0
predator_speed_boost: float = 1.8
predator_perception_boost: float = 1.5
predator_acceleration_boost: float = 1.4
predator_escape_factor: float = 1.0e7
```

**Accept:** round-trip `to_file→from_file` equality; loading any shipped
preset leaves domain dimensions intact; the new fields appear in
`--list-configs`-loadable YAML under a `spatial:`/`predator:` section.

---

## R1 — The faithful Reynolds update pipeline (3D)

**Idea (verbal).** The source's per-boid update is a strict pipeline whose
*order* is the behaviour: forces are normalized **directions** scaled by
weights (not magnitude-weighted sums), acceleration is damped by a scale
factor then capped, noise perturbs **velocity directly** (after integration,
so the flocking forces must fight it next frame), and speed has a **ceiling
only** — birds may fly arbitrarily slowly. pymurmur's spatial mode deviates
on each point: separation is 1/d-weighted and unnormalized, cohesion is
unbounded, `noise_scale`'s magnitude is discarded, `acceleration_scale`
is never applied, and the speed band forbids slow flight.

**Math (3D; per boid, neighbours = all boids within `perception` radius).**

```
v̄  = Σ_{j∈N, ¬pred} v_j / |N|                     alignment target
F_align = normalize(v̄ − v_i)

p̄  = Σ_{j∈N, ¬pred} p_j / |N|                     cohesion target
F_coh   = normalize(p̄ − p_i)

F_sep   = normalize( Σ_{j∈N, d_ij < separation_distance} −(p_j − p_i) )
          (raw displacement sum, normalized once at the end — NOT 1/d² weighted;
           keep pymurmur's 1/d² as separation_kernel="inverse_square")

a  = F_align·w_a + F_coh·w_c + F_sep·w_s
a *= predator_acceleration_boost      (predators only — BEFORE the scale)
a *= acceleration_scale                            (0.3)
a  = limit(a, max_force)                           (1.0)
v += a · dt
v += (U³[0,1] − 0.5) · noise_scale                 (velocity noise, if scale ≠ 0)
v  = limit(v, max_speed)                           (ceiling only; no floor)
p += v · dt
a  = 0                                             (reset every frame)
p  = p mod (W, H, D)                               (toroidal wrap, velocity untouched)
```

Effective predator damping = 1.4 × 0.3 = 0.42 vs prey 0.30 (boost applies
before the scale — order matters).

**Implementation.** Rewrite `spatial_forces` in
`pymurmur/physics/forces/spatial.py` to:

1. Query neighbours by **metric radius** `config.visual_range`
   (`tree.query_ball_point(pos, r, workers=...)`; see R2/R4) instead of pure
   kNN. Keep `topological_cap` as an optional max-count truncation.
2. Compute the three normalized steering terms above (fix
   `_base.py::separation_force`'s unnormalized 1/d form by adding the
   `"unit"` kernel; fix `cohesion_force` to `normalize(p̄ − p_i)`).
3. Apply `acceleration_scale` before the max-force clamp.
4. Velocity noise: when `noise_mode == "velocity"`, skip the noise force and
   instead have `integrate()` add
   `(rng.uniform(0, 1, (n,3)) − 0.5) · noise_scale` to velocities **after**
   `v += a` and **before** the speed limit.
5. Speed ceiling: `integrate(..., speed_mode="ceiling")` skips the
   `0.3·v0` floor and zero-speed re-seed, keeping only
   `v = limit(v, v0)` (add the parameter to
   [pymurmur/physics/boid.py::integrate](pymurmur/physics/boid.py#L19)).

**Accept:** with source defaults and N=150, the flock forms cohesive
non-overlapping groups within 300 frames (order parameter α > 0.6); a bird
released at zero speed *stays* slow until forces accelerate it (no floor
kick); doubling `noise_scale` visibly increases heading jitter (its
magnitude is no longer discarded).

---

## R2 — Toroidal distance in neighbour queries (with the source's bug fixed)

**Idea (verbal).** On a wrapping domain, the shortest path between two boids
may cross a boundary; neighbour queries must use that wrapped distance or
flocks tear at the seam. The source implements this — with a known **sign
bug**: its `toroidal_distance2` tests `dx > width/2` on a *signed* dx, so
negative differences never wrap (a bird at x=10 vs x=90 in a width-100 world
reports 80 instead of 20). Implement the corrected form.

**Math (3D, corrected — apply `abs` before the wrap test):**

```
dx = |x₁ − x₂|;  if dx > W/2: dx = W − dx        (same for dy/H, dz/D)
d² = dx² + dy² + dz²
equivalently per axis:  Δ_mi = Δ − L·round(Δ/L)
```

**Implementation.**

1. Neighbour queries: `cKDTree(pos, boxsize=(W, H, D))` gives torus-correct
   `query` / `query_ball_point` natively (positions are already wrapped into
   `[0, L)` by the integrator). Gate on
   `config.boundary_mode == "toroidal" and config.use_toroidal_distance`
   (the latter exists and is currently dead — this makes it live).
2. Pair vectors (separation displacements, R3 escape direction): helper
   `def min_image(delta, box): return delta - box * np.round(delta / box)`
   in `core/types.py`; use it wherever `p_j − p_i` feeds a force on a
   toroidal domain.
3. Add the regression test that the *source* would fail:
   birds at x=1 and x=W−1 must be mutual neighbours at radius 3.

**Accept:** cross-seam neighbour test passes; a flock crossing the boundary
stays visually contiguous (no splitting at the seam); disabling
`use_toroidal_distance` restores raw behaviour.

---

## R3 — Predator boids (species inside the flock)

**Idea (verbal).** Predators are boids in the same arrays with three
multiplicative advantages (faster ×1.8, see farther ×1.5, turn harder ×1.4)
and three interaction rules on prey: (1) any perceived predator **hard-zeros
alignment and cohesion** — the boid instantly drops out of collective motion;
(2) separation is replaced by an **overwhelming escape force** away from the
predator (factor 10⁷ — it must dominate everything, that is the design);
(3) predators themselves flock normally with each other and ignore prey
separation. The emergent result is flash-expansion: all social steering
vanishes near the threat and the flock scatters coherently because every
bird flees the same point.

**Math (3D).**

```
prey i, any predator k ∈ N_i (perception radius):
    F_align = 0;  F_coh = 0                       (hard zero, not reduction)
    F_sep   = normalize(p_i − p_k) · predator_escape_factor
              (nearest such k; min-image difference on toroidal domains)

predator k:  standard R1 pipeline over predator neighbours only, with
    max_speed·1.8, perception·1.5, acceleration·1.4 (before accel_scale)
```

**Implementation.**

1. `PhysicsFlock` gains `self.is_predator = np.zeros(N, bool)`; carried
   through `_extend`/`add_boids`/`remove_boids`; last
   `config.n_predator_boids` slots are predators at init.
2. Vectorised prey handling in `spatial_forces`: build
   `tree_pred = cKDTree(pos[pred], boxsize=box)`;
   `d_pred, j_pred = tree_pred.query(pos[prey], k=1)`;
   `threatened = d_pred < perception`. For threatened rows: zero the
   align/coh contributions, set separation row to
   `normalize(min_image(p_prey − p_pred))·escape_factor`. The subsequent
   `limit(a, max_force)` keeps the 10⁷ from exploding the integrator —
   its job is to *win the sum*, not to set the magnitude (source semantics:
   the limit caps it too).
3. Per-species parameter arrays where the pipeline needs them:
   `max_speed_i = v0 · where(pred, 1.8, 1.0)`,
   `perception_i = visual_range · where(pred, 1.5, 1.0)` (two radius queries:
   one per species, since cKDTree takes a scalar radius per call),
   boost factor in the acceleration step.
4. Metrics: compute α/dispersion over prey only when predators exist.

**Accept:** planting one predator in a settled flock produces a visible hole
within 30 frames and mean prey speed spikes; threatened prey have exactly
zero alignment/cohesion contribution (assert in a unit test with a hand-built
neighbourhood); two predators flock with each other (their pair distance
stabilises).

---

## R4 — Two-phase parallel update

**Idea (verbal).** The source splits each frame into (1) a parallel
neighbour-search phase over an immutable tree and (2) a parallel per-boid
update phase where each boid writes only its own state — lock-free by
construction, with dynamic scheduling because boids in dense regions cost
more. The Python translation gets phase 1 from scipy and phase 2 from
vectorisation (or numba), controlled by one `num_threads` knob.

**Implementation.**

1. Phase 1: replace the current per-bird Python loop
   ([spatial.py:78-81](pymurmur/physics/forces/spatial.py#L78-L81)) with one
   batched call — `tree.query(active_pos, k=k+1, workers=config.num_threads)`
   (kNN path) and
   `tree.query_ball_point(active_pos, r, workers=config.num_threads)`
   (radius path). `workers=-1` = all cores = the source's
   `num_threads=-1` auto mode.
2. Phase 2: the force math in R1 is pure array arithmetic over the padded
   neighbour-index matrix — vectorise with a gather
   (`positions[neighbor_idx]`, shape `(n, k, 3)`, mask padding) and axis-1
   reductions; no Python loop remains. (Optional later: numba
   `@njit(parallel=True)` kernel; keep behind `use_numba`.)
3. Thread the flag: `num_threads` from config → both query calls; document
   that phase 2 relies on NumPy/BLAS threading.

**Accept:** at N=20 000 spatial mode, frame time drops ≥3× vs the loop
version on a multicore machine; results identical (same seed → same
positions) for `num_threads ∈ {1, -1}`.

---

## R5 — Runtime interaction: spawn, predator spawn, clear

**Idea.** The source's mouse interaction: left-click adds a boid at the
cursor, right-click adds a predator there, `C` clears all boids, `Q` quits.
In 3D the cursor is a ray, not a point — spawn at the ray's intersection
with the flock's median-depth plane.

**Math (cursor unprojection).** With view matrix V, projection P, viewport
(w, h), mouse (mx, my):

```
ndc = (2·mx/w − 1, 1 − 2·my/h)
ray_clip = (ndc.x, ndc.y, −1, 1)
ray_eye  = P⁻¹ · ray_clip;  ray_eye = (ray_eye.x, ray_eye.y, −1, 0)
r̂_world  = normalize((V⁻¹ · ray_eye).xyz)
o        = camera position
depth    = median over active birds of (p_i − o)·f̂      (f̂ = camera forward)
spawn    = o + r̂_world · depth / (r̂_world·f̂)
```

**Implementation.** In `pymurmur/viz/input_control.py`: on
`MOUSEBUTTONDOWN` button 1/3 (when not dragging), compute `spawn` via the
camera's matrices (PyGLM `glm.inverse`), then queue
`pending_spawn.append((spawn, is_predator))`; `Visualizer.run` drains the
queue calling a new `flock.add_boid_at(pos, is_predator, config)` (place at
`pos`, random velocity `limit((U³−0.5)·2·v0, v0)`). `C` sets
`pending_clear`, applied as `flock.active[:] = False`; `Q` quits alongside
ESC. Keep left-drag orbit — spawn on *click* (down+up without motion),
orbit on drag: track a small movement threshold (5 px).

**Accept:** clicking an empty region visibly adds a bird near the flock's
depth; right-click adds a red predator (R3 + R8); `C` empties the sim
without crashing metrics (`N_active == 0` guarded).

---

## R6 — CLI parameter system

**Idea.** Every simulation parameter settable from the command line — the
source exposes 16 cxxopts flags with defaults, enabling scripted sweeps
without editing files.

**Implementation.** In `pymurmur/__main__.py` add a repeatable generic
override plus the source's convenience flags:

```
--set KEY=VALUE          (repeatable; KEY must be a SimConfig field;
                          value parsed by the field's type)
--flock-size N           → num_boids
--max-speed X            → v0
--num-threads N          → num_threads
--fullscreen             → pygame.FULLSCREEN in set_mode
--light-scheme           → theme = "ink"    (dark default = "inverse")
```

`--set` implementation: `field_types = {f.name: f.type for f in
fields(SimConfig)}`; parse `int/float/bool/str` accordingly; unknown key →
exit with the valid-field list. Precedence: YAML < env (if present) < CLI.

**Accept:** `python -m pymurmur --config murmuration_spatial --set
separation_weight=6 --set num_boids=500 --no-viz` runs with those values
(assert via a `--print-config` debug flag added alongside).

---

## R7 — Scripting facade + benchmark API (the pyboids equivalent)

**Idea (verbal).** The source ships Python bindings whose two entry points
are a keyword-argument constructor and a **render-free benchmark** returning
nanosecond per-step timings — the tool for parameter sweeps and perf
regression. pymurmur is already Python but exports nothing and has no
benchmark; mirror the API.

**Implementation.** In `pymurmur/__init__.py`:

```python
from .core.config import SimConfig
from .simulation.engine import SimulationEngine

def Simulation(**params) -> SimulationEngine:
    cfg = SimConfig(**params)          # TypeError on unknown kwarg = validation
    return SimulationEngine(cfg)
```

In `SimulationEngine`:

```python
def benchmark(self, flock_size: int | None = None, num_steps: int = 1000) -> list[float]:
    """Render-free per-step wall-clock durations in seconds."""
    if flock_size is not None:
        self.config.num_boids = flock_size
        self.reset()
    out = []
    for _ in range(num_steps):
        t0 = time.perf_counter_ns()
        self.step()
        out.append((time.perf_counter_ns() - t0) / 1e9)
    return out
```

**Accept:** `import pymurmur; pymurmur.Simulation(num_boids=200,
mode="spatial").benchmark(num_steps=100)` returns 100 positive floats; the
scaling table in the architecture docs can be regenerated from it.

---

## R8 — Rendering: predator distinction, colour schemes, init variant, fixed timestep

**Idea.** Four presentation-layer behaviours from the source:

1. **Predators are visually distinct** — 1.3× larger, always red regardless
   of scheme. pymurmur currently renders no species distinction at all.
2. **Dark/light colour schemes** — pymurmur's `THEMES` palettes exist but
   `config.theme` is never passed to the renderer (one-line wire).
3. **Velocity init with dispersion**: `v = (U³[0,1] − 0.5)·2·max_speed`,
   expected magnitude `≈ 0.816·max_speed` (pymurmur fixes all speeds at
   `0.8·v0` — same mean, no spread).
4. **Fixed physics timestep**: the source runs physics at the frame rate
   with `dt = 1/FRAME_RATE`; for variable-rate 3D rendering use the
   accumulator so behaviour is frame-rate independent:

```
acc += frame_dt
while acc >= dt_phys:            # dt_phys = 1/60
    sim.step(dt_phys); acc -= dt_phys
```

**Implementation.**

1. Instance buffer gains a 7th float `flag` (`'3f 3f 1f/i'`, attribute
   `in_bird_flag`) packed from `flock.is_predator`; vertex shader scales the
   mesh by `mix(1.0, 1.3, flag)`; fragment shader
   `color = mix(theme_color, vec3(0.85, 0.08, 0.08), flag)`.
   **Rebuild the VAO wherever the instance buffer is (re)created** —
   including the growth path in
   [renderer.py:104-109](pymurmur/viz/renderer.py#L104-L109), which
   currently reallocates the buffer without rebinding (fix together).
2. `Visualizer.__init__`: pass `theme=config.theme` to `Renderer3D`.
3. `PhysicsFlock.__init__`/`add_boids`:
   `velocities = limit3((rng.uniform(0, 1, (N,3)) − 0.5) * 2 * v0, v0)`.
4. `Visualizer.run`: replace the direct `sim.step(dt)` with the accumulator
   loop (clamp `frame_dt` to 1/20 first so a dragged window can't produce a
   step spiral).

**Accept:** predators render red and larger in every theme; same-seed
headless runs at 30 and 60 fps produce identical trajectories (accumulator);
initial speed histogram is dispersed, mean ≈ 0.82·v0.

---

## R9 — Preset, tests, golden

**Implementation.**

1. **Preset** `conf/murmuration_boids.yaml`: source defaults —
   `mode: spatial`, `num_boids: 150`, `v0: 6`, `max_force: 1`,
   `spatial: {separation_weight: 4.5, alignment_weight: 0.65,
   cohesion_weight: 0.75, acceleration_scale: 0.3, separation_kernel: unit,
   noise_mode: velocity, speed_mode: ceiling, separation_distance: 20}`,
   `visual_range: 100`, `boundary: toroidal`, `use_toroidal_distance: true`,
   `predator: {n_predator_boids: 0}` (spawn via right-click).
2. **Tests** (`test/physics/test_boids_port.py`): the acceptance assertions
   from R1–R5 and R8 above, plus determinism (same seed → identical
   positions after 200 steps, `num_threads ∈ {1, -1}`) and the corrected
   toroidal-distance regression (the case the source's sign bug fails:
   x=10 vs x=90, W=100 → d=20).
3. **Golden re-pin** for spatial mode (R1 deliberately changes dynamics) in
   the same commit as R1.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config fields + loader | — | ½ day | `core/config.py`, tests |
| R2 | Toroidal query distance | R0 | ½ day | `forces/spatial.py`, `core/types.py` |
| R1 | Faithful Reynolds pipeline | R0, R2 | 1 day | `forces/spatial.py`, `forces/_base.py`, `physics/boid.py` |
| R3 | Predator boids | R1 | 1 day | `physics/flock.py`, `forces/spatial.py`, `analysis/metrics.py` |
| R4 | Parallel two-phase update | R1 | ½ day | `forces/spatial.py` |
| R5 | Mouse spawn / clear | R3 | ½ day | `viz/input_control.py`, `viz/visualizer.py`, `physics/flock.py` |
| R6 | CLI overrides | R0 | ¼ day | `__main__.py` |
| R7 | Facade + benchmark | — | ¼ day | `__init__.py`, `simulation/engine.py` |
| R8 | Rendering + init + timestep | R3 | 1 day | `viz/renderer.py`, `viz/shaders.py`, `viz/visualizer.py`, `physics/flock.py` |
| R9 | Preset + tests + golden | all | ½ day | `conf/`, `test/` |

Total ≈ **6 working days**. R6/R7 are independent quick wins; R4 and R8 can
proceed in parallel once their prerequisites land.
