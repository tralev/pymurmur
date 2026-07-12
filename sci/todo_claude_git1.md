# todo_claude_git1.md — Porting `JerBoon/murmuratR` to pymurmur (3D)

**Source:** https://github.com/JerBoon/murmuratR — "Exploring the maths of
starling murmurations in R". An agent-based **cosmic-influencer** model: all
flocking emerges from birds following a single moving 3D target with
individually varying influence weights — **no neighbour interactions at all**.
Source files verified: `R/iterate_flock.R`, `R/target_pos.R`, `R/new_flock.R`,
`R/plot_birds.R`, `R/main.R`, `R/random_bits.R`, `R/xyz.R`.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`), whose
`influencer` force mode
([pymurmur/physics/forces/influencer.py](pymurmur/physics/forces/influencer.py))
is a loose sketch of this model.

**What this file is.** Every idea and piece of math in murmuratR that pymurmur
does not implement (or implements incorrectly), adapted for pymurmur's 3D
simulation and visualization, with formulas, constants, array shapes, file
paths, config fields, and acceptance tests — so each roadmap item is
implementable from this file alone.

**Model status note.** murmuratR is *natively 3D* — positions and directions
are 3-vectors throughout; only its *rendering* is 2D (ggplot projections). So
the simulation math ports directly, and the "adjust to 3D" work concentrates
in R2 (embedding the trajectory in pymurmur's bounded domain) and R7
(replacing 2D projections with 3D rendering equivalents).

**Already implemented in pymurmur (do not redo):** an influencer mode exists
with a rank-based influence gradient (`argsort().argsort()`), the 1.8
exponent as `influencer_rank_exponent`, a substep loop
(`influencer_substeps`), and max-force clamping. Everything else below is
missing or wrong.

---

## Conventions used throughout

- **Bird state:** murmuratR birds have position `(x,y,z)` and a **unit
  direction** `(dx,dy,dz)`; motion is unit-speed (`p += d` each tick).
  pymurmur stores `positions`/`velocities` `(N,3) float32`; this mode treats
  `d̂ = v/|v|` and maintains `|v| = v₀` (see R3).
- **RNG:** all randomness from the flock-owned seeded generator
  (`PhysicsFlock.rng = np.random.default_rng(config.seed)`), never module
  `np.random.*`.
- **Active mask:** `idx = np.where(flock.active)[0]`; all per-bird arrays are
  gathered/scattered through `idx`.
- **Domain:** pymurmur's `[0,width)×[0,height)×[0,depth)`. murmuratR is
  unbounded; the recommended boundary for this mode is `open` (R2 explains
  the embedding). `C = (width/2, height/2, depth/2)` denotes the domain
  centre.

---

## R0 — Prerequisite: config plumbing for the influencer parameters

**Idea.** pymurmur's YAML loader flattens section keys without prefixing:
`influencer: rank_exponent: 1.8` becomes key `rank_exponent`, which does not
match the dataclass field `influencer_rank_exponent`, so the whole
`influencer:` section of any preset is **silently dropped** (the mode runs on
dataclass defaults). Unprefixed keys can also collide with real fields
(`capture: width:` overwrites the domain `width`). Fix loading first or the
new parameters below never reach the simulation.

**Implementation.**

1. In `SimConfig.from_file`
   ([pymurmur/core/config.py:149-164](pymurmur/core/config.py#L149-L164)),
   prefix flattened keys with their section when the prefixed name is a valid
   field, warn on unknown keys:

```python
valid = {f.name for f in fields(cls)}
flat: dict[str, Any] = {}
for section, data in raw.items():
    if isinstance(data, dict):
        for k, v in data.items():
            pk = f"{section}_{k}"              # influencer.substeps -> influencer_substeps
            if pk in valid:   flat[pk] = v
            elif k in valid:  flat[k] = v      # domain.width -> width (legacy sections)
            else: warnings.warn(f"config: unknown key {section}.{k}")
    else:
        flat[section] = data
```

2. Add the new dataclass fields used by R1–R7 (defaults = murmuratR values):

```python
# ── Influencer (murmuratR) ────────────────────────────────────
influencer_rank_exponent: float = 1.8      # exists already
influencer_substeps: int = 5               # exists already (murmuratR main uses 3–10)
influencer_scale: float = 1.0              # R2: trajectory size multiplier
influencer_influence_mode: str = "rank"    # "rank" (source behaviour) | "distance"
influencer_near_dist_sq: float = 100.0     # R4: numerator of the 100/d² term
influencer_init: str = "gauss"             # R5: "gauss" (source) | "box" (legacy)
influencer_separation: float = 0.5         # R5: sep in mult = N^(1/3)·sep
```

3. Round-trip test (`to_file` → `from_file` equal on all fields) and a test
   that `conf/murmuration_influencer.yaml` actually sets the fields it names.

**Accept:** editing `influencer: substeps:` in the YAML changes runtime
behaviour; domain dimensions survive loading every shipped preset.

---

## R1 — Persistent tick counter: give the mode real time

**Idea (verbal).** murmuratR's influencer moves along a deterministic path in
a persistent time variable: the flock carries a `ticks` attribute
(initialised `0L` in `new_flock.R`), and every iteration reads it, computes
the target, and increments it. pymurmur's mode has **no time state at all** —
it draws `t = np.random.uniform(0, 2π)` fresh each substep, so the "moving
target" is a random teleporting point and none of the model's emergent
behaviour (core-follows/periphery-lags stretching, §R2's speed variation) can
occur.

The structural cause: pymurmur's `ForceFunc` protocol is a stateless
`(flock, config) -> None`. The mode needs somewhere to keep `t`.

**Implementation.** Add a generic per-mode state dict to the flock —
smallest change that also serves other modes later:

```python
# physics/flock.py, in PhysicsFlock.__init__
self.mode_state: dict[str, object] = {}
self.rng = np.random.default_rng(config.seed)     # if not already present
```

In the influencer mode:

```python
state = flock.mode_state.setdefault("influencer", {"tick": 0})
tick = state["tick"]
...                                # one tick consumed per SUBSTEP (see R3)
state["tick"] = tick + 1
```

`SimulationEngine.reset()` recreates the flock, which resets ticks — matching
murmuratR's `new_flock` semantics. Do **not** reset on mode switch (source
keeps ticks on the flock object; a fresh mode entry naturally starts at 0
only after reset).

**Accept:** calling the mode twice with the same flock advances the target
deterministically; two same-seed runs produce identical target sequences;
`sim.reset()` restarts the trajectory at t=0.

---

## R2 — The Lissajous target trajectory (exact formulas, 3D domain embedding)

**Idea (verbal).** The influencer's path is a smooth, effectively aperiodic
3D curve built from six mutually prime periods. Two amplitude scales per
axis give large sweeping arcs plus small local flutter; a constant vertical
offset keeps the display airborne. Because the periods share no common
factor, the path never visibly repeats; because it is a sum of sines, it is
C∞ — no jerks. The target's instantaneous speed varies naturally (fast when
component waves align, slow when they oppose), which alternately stretches
and condenses the flock.

**Math (verbatim from `target_pos.R`, t in ticks):**

```
T_raw(t) = ( sin(t/97)·200  + cos(t/217)·30,
             cos((t+53)/29)·200 + sin((47−t)/13)·30,
             cos((t+61)/41)·100 + sin((t+13)/7)·27 + 40 )
```

Component ranges: x ∈ [−230, 230], y ∈ [−230, 230], z ∈ [−87, 167].

**3D domain embedding.** murmuratR is unbounded around the origin; pymurmur
has a finite domain. Centre the trajectory on the domain and scale it to fit:

```
s      = influencer_scale · min(width/460, height/460, depth/254)
T(t)   = C + (T_raw(t) − (0, 0, 40)) · s + (0, 0, 40·s)
```

i.e. remove the source's vertical offset, scale about the origin, re-apply
the offset scaled, and translate to the domain centre `C`. With the default
1000×700×400 domain, `s ≈ 1.52` — the sweep fills the volume. Recommended
`boundary_mode: open` for this mode (the source is unbounded; toroidal
wrapping would teleport laggards across the seam mid-stretch). Per-step
target displacement `‖T(t+1) − T(t)‖` needs no extra code — speed variation
falls out of the formula.

**Implementation.** Pure function in
`pymurmur/physics/forces/influencer.py`:

```python
def _target_pos(t: float, config) -> np.ndarray:
    raw = np.array([
        math.sin(t / 97.0) * 200.0 + math.cos(t / 217.0) * 30.0,
        math.cos((t + 53.0) / 29.0) * 200.0 + math.sin((47.0 - t) / 13.0) * 30.0,
        math.cos((t + 61.0) / 41.0) * 100.0 + math.sin((t + 13.0) / 7.0) * 27.0 + 40.0,
    ], dtype=np.float32)
    s = config.influencer_scale * min(config.width / 460.0,
                                      config.height / 460.0,
                                      config.depth / 254.0)
    C = np.array([config.width / 2, config.height / 2, config.depth / 2], np.float32)
    raw[2] -= 40.0
    return C + raw * s + np.array([0.0, 0.0, 40.0 * s], np.float32)
```

**Accept:** plotting `T(t)` for t = 0…2000·10 reproduces the source's three
orthogonal projections (smooth open loops, no repetition); the target never
leaves the domain box for `influencer_scale ≤ 1`.

---

## R3 — Direction-state dynamics: move-then-steer at unit speed

**Idea (verbal).** murmuratR birds are pure headings: they move one step
along their **old** direction *first*, then blend the direction toward the
target from the **new** position. This one-step control lag is the model's
inertia — birds commit to their heading before reacting, which prevents
instantaneous snapping and produces the flowing, ribbon-like motion. Speed
is constant (unit per tick); all dynamics live in the direction. A substep
runs one *full* move+steer+tick cycle — pymurmur's current substep loop
(five force accumulations onto static positions against five random targets)
must be replaced.

**Math (per substep, verbatim from `iterate_flock.R`, adapted to vectors):**

```
1. p_i ← p_i + d̂_i · v₀·Δt                    (move with OLD direction)
2. T = target(tick)                            (R2)
3. Δ_i = T − p_i ;  dist_i = ‖Δ_i‖ ;  guard dist_i ← dist_i + [dist_i = 0]
   t̂_i = Δ_i / dist_i
4. inf_i = influence(dist ranks)               (R4)
5. d_i ← d̂_i·(1 − inf_i) + t̂_i·inf_i          (component-wise blend)
   len_i = ‖d_i‖ ; guard len_i ← len_i + [len_i = 0]
   d̂_i ← d_i / len_i                           (renormalize to unit)
6. tick ← tick + 1
```

The `x + (x == 0)` guards are the source's branch-free divide-by-zero
protection (adds 1 exactly when the value is 0) — keep them.

**Implementation.** The mode owns the position update (like the source), so
it must **bypass `integrate()`'s move and speed band**. Mirror the vicsek
pattern: mode writes final velocities; `PhysicsFlock.step` passes a
`speed_mode="fixed"` / "mode owns positions" flag. Cleanest concrete shape:

```python
def influencer_forces(flock, config):
    idx = np.where(flock.active)[0]
    n = len(idx)
    if n == 0: return
    state = flock.mode_state.setdefault("influencer", {"tick": 0})
    v0, dt = config.v0, 1.0 / 60.0

    pos = flock.positions[idx]
    vel = flock.velocities[idx]
    spd = np.linalg.norm(vel, axis=1, keepdims=True)
    d = np.where(spd > 1e-9, vel / np.maximum(spd, 1e-9), 0.0)   # unit dirs; zero stays zero (R5)

    for _ in range(config.influencer_substeps):
        pos += d * (v0 * dt)                                     # 1. move (old dir)
        T = _target_pos(float(state["tick"]), config)            # 2. target
        delta = T - pos
        dist = np.linalg.norm(delta, axis=1)
        dist = dist + (dist == 0.0)                              # 3. guard
        t_hat = delta / dist[:, None]
        inf = _influence(dist, config)                           # 4. R4
        d = d * (1.0 - inf[:, None]) + t_hat * inf[:, None]      # 5. blend
        ln = np.linalg.norm(d, axis=1)
        ln = ln + (ln == 0.0)
        d = d / ln[:, None]
        state["tick"] += 1                                       # 6. tick

    flock.positions[idx] = pos
    flock.velocities[idx] = d * v0        # heading carried in velocity at |v| = v0
```

`PhysicsFlock.step` for `mode == "influencer"`: skip force clearing pipeline
subtleties by calling `integrate` with a `move=False, speed_mode="fixed"`
variant (add both parameters; `move=False` skips step 3 "positions +=" and
the speed clamps, keeping only boundary enforcement so `open`/`margin`
still apply). Accelerations stay zero.

**Accept:** all speeds exactly `v0` after any run; freezing the target
(monkeypatch `_target_pos` to a constant) makes every bird converge to
orbit/hover near it; one-step lag is observable — after an instantaneous
target jump, birds' headings change only on the *next* substep (positions
move along old headings first).

---

## R4 — The influence model: rank gradient with a guaranteed floor (+ distance mode)

**Idea (verbal).** Which birds follow the target hard? In the source, the
answer is decided by **rank**: sort birds by distance *to the target* (not
to the flock centre — pymurmur currently ranks by CoM distance, which makes
the "core" the centre of the blob rather than the followers of the
influencer). The closest bird gets influence 1.0; values fall along a fixed
sequence to 0.2, raised to the 1.8 power — so the floor is 0.2^1.8 ≈ 0.055:
even the farthest bird *keeps following faintly* instead of decoupling.
The concave (>1) exponent makes a small tight core and a long diffuse tail —
real murmuration morphology. The source also computes a distance-based
influence `clamp(100/d², 0.2, 0.8)` and then **overrides it** with the rank
values; implement rank as the default (`influence_mode="rank"`) and keep the
distance form as the documented alternative (`"distance"`).

**Math.**

Rank mode (source behaviour). For birds sorted by distance ascending,
sorted-position `i ∈ {0 … N−1}`:

```
inf_sorted[i] = (1 − (i / (N−1)) · 0.8) ^ 1.8         (N>1; N=1 → inf = 1)
inf[original bird] = inf_sorted[rank(bird)]
```

Endpoints: i=0 → 1.0; i=N−1 → 0.2^1.8 ≈ 0.055; median → 0.37.

Distance mode (source's computed-then-overridden alternative):

```
inf_i = clamp( influencer_near_dist_sq / dist_i², 0.2, 0.8 )
```

(with default 100: full-ish influence inside d ≈ 11.2, floor beyond
d ≈ 22.4 — scale `influencer_near_dist_sq` with `influencer_scale²` so the
knee tracks the trajectory embedding: use `100·s²`.)

**Implementation.**

```python
def _influence(dist: np.ndarray, config) -> np.ndarray:
    n = len(dist)
    if config.influencer_influence_mode == "distance":
        s = _embed_scale(config)                       # same s as in _target_pos
        return np.clip(config.influencer_near_dist_sq * s * s / (dist * dist), 0.2, 0.8)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    ranks = np.argsort(np.argsort(dist))               # 0 = closest
    seq = (1.0 - (ranks / (n - 1)) * 0.8) ** config.influencer_rank_exponent
    return seq.astype(np.float32)
```

(This replaces the current `(1 − rank/n)^1.8`-of-CoM-distance in
[influencer.py:42-47](pymurmur/physics/forces/influencer.py#L42-L47).)

**Accept:** exactly one bird has influence 1.0 each substep; minimum
influence ≈ 0.055 (never below); influence is monotonically decreasing in
distance-to-target; setting `influencer_rank_exponent: 1.0` produces a
linear gradient (visible as an evenly stretched flock).

---

## R5 — Density-scaled initialization

**Idea (verbal).** Initial positions are a Gaussian cloud whose width grows
as N^(1/3), so the **starting density is the same for any flock size**
(volume ∝ σ³ and density ∝ N/σ³ ⇒ σ ∝ N^(1/3)). A single shared random
offset shifts the whole cloud so runs differ in placement. Initial
directions are **zero**: the first blend then yields `d = t̂·inf` — every
bird's first heading points at the target, weighted by its influence, which
is exactly how the flock "wakes up toward" the influencer.

**Math (verbatim from `new_flock.R`, embedded in the domain):**

```
mult = N^(1/3) · influencer_separation · s        (sep = 0.5; s = embed scale, R2)
offset = C + U(0,1)·10·s  per axis                (source: 10 + runif(1)·10, origin-relative)
p_i = N(0,1)·mult + offset                        (per axis, per bird)
d_i = (0, 0, 0)
```

**Implementation.** Mode-specific init hook in `PhysicsFlock.__init__`
(and `SimulationEngine.reset` path — it recreates the flock, so one place):

```python
if config.mode == "influencer" and config.influencer_init == "gauss":
    s = _embed_scale(config)
    mult = (N ** (1.0/3.0)) * config.influencer_separation * s
    C = np.array([config.width/2, config.height/2, config.depth/2], np.float32)
    offset = C + rng.uniform(0.0, 10.0 * s, size=3).astype(np.float32)
    self.positions = (rng.normal(size=(N, 3)).astype(np.float32) * mult) + offset
    self.velocities = np.zeros((N, 3), dtype=np.float32)   # zero dirs (source)
```

R3's `d` computation maps zero velocity to zero direction (not a random
unit) so the first-blend semantics hold — that is why the fallback in R3
uses `0.0`, not a random vector, when speed is zero **in this mode**.

**Accept:** for N ∈ {100, 1000, 8000}, initial local density (median
7th-neighbour distance) is equal within 10%; frame-0 headings all point at
the target with magnitudes ordered by influence.

---

## R6 — Distance diagnostics

**Idea.** Every iteration, the source records and prints the min and max
bird-to-target distance — a two-number health readout: min ≈ how tightly the
core tracks; max ≈ the flock's spatial extent along the pursuit.

**Math.** `d_min(t) = min_i ‖p_i − T(t)‖`, `d_max(t) = max_i ‖p_i − T(t)‖`.

**Implementation.** The mode already has `dist` per substep (R3); stash the
last substep's values on the state dict:

```python
state["d_min"], state["d_max"] = float(dist.min()), float(dist.max())
```

Add `target_dist_min: float | None` and `target_dist_max: float | None` to
`FlockMetrics` (`pymurmur/analysis/metrics.py`); `MetricsCollector.collect`
copies them from `flock.mode_state.get("influencer", {})` when present.
Surface in the window title when in influencer mode
(`viz/visualizer.py` title builder): `dT=[{min:.0f},{max:.0f}]`.

**Accept:** metrics CSV from a headless capture contains both columns with
plausible monotone-ish envelopes; title shows the pair live.

---

## R7 — 3D visualization equivalents of the ggplot rendering

**Idea (verbal).** The source renders the flock as **semi-transparent points
(alpha = 0.2) on a void background**, x–z projection — overlap builds
visible density gradients, which is most of the murmuration aesthetic at
N = 1000. pymurmur's opaque instanced tetrahedra lose this. Port the *ideas*,
not the ggplot: (a) density-accumulating translucent rendering, (b) the
source's orthogonal projections as camera presets, (c) draw the influencer
itself for debugging (the source never sees it — in 3D it's invaluable).

**Implementation** (all in `pymurmur/viz/`):

1. **Alpha-accumulation mode** (`config.point_sprites: true` +
   `config.theme: "ink"`): render birds as camera-facing point sprites with
   `gl.PROGRAM_POINT_SIZE`, fragment alpha 0.2, blending
   `ctx.enable(moderngl.BLEND); ctx.blend_func = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`,
   **depth write off** (`ctx.depth_mask = False`) while drawing sprites so
   translucency accumulates instead of z-fighting; restore after. Fragment
   shader: discard outside unit disc (`p = uv*2-1; if dot(p,p) > 1 discard;`),
   colour = theme ink at alpha 0.2.
2. **Projection camera presets**: keys `7` = top `(x,z)` view
   (azimuth 0, elevation 89.9°... note pymurmur is z-up, so the source's x–z
   plot = front view: azimuth −90°, elevation 0°+ε), `8` = side `(y,z)`
   (azimuth 0°, elevation 0°+ε), `9` = default perspective. Implement as
   `OrbitCamera.set_view(azimuth, elevation, distance)` presets in
   `viz/camera.py` + three bindings in `viz/input_control.py`. For true
   orthographic parity add `OrbitCamera.orthographic: bool` and an ortho
   projection matrix (`glm.ortho(-w/2, w/2, -h/2, h/2, near, far)` scaled by
   `distance`) — perspective at long distance is an acceptable first cut.
3. **Influencer marker**: one extra instance appended to the packed buffer
   when `config.mode == "influencer"` — position `T(tick)`, velocity = its
   finite-difference direction, and (if the per-instance flag column from the
   species work exists) flag = 1 to render it red/larger; otherwise render a
   second 1-instance draw call with a distinct uniform colour.

**Accept:** N=1000 alpha mode shows visibly darker core than fringe (density
gradient); pressing `7`/`8` reproduces the source's two projection plots;
the red marker traces a smooth curve the flock chases.

---

## R8 — Preset, tests, and regression pinning

**Implementation.**

1. **Preset** `conf/murmuration_influencer.yaml` (after R0 the section
   loads): `mode: influencer`, `flock: num_boids: 1000, v0: 4.0`,
   `boundary: open`, `influencer: {substeps: 3, rank_exponent: 1.8,
   scale: 1.0, influence_mode: rank, init: gauss, separation: 0.5}`,
   `visual: theme: ink, point_sprites: true`.
2. **Tests** (`test/physics/test_influencer.py`):
   - determinism: same seed → identical positions after 100 steps;
   - unit speed: `|v| == v0` for all birds, all frames;
   - trajectory: `_target_pos` matches the R formulas at t = 0, 970, 2170
     (hand-computed values, `s=1` domain 460×460×254);
   - influence: floor ≈ 0.2^1.8, max = 1.0, monotone in target distance;
   - init: density invariance across N (R5 acceptance);
   - **emergent stretching** (the model's signature): after 500 settled
     steps, flock extent along the target's velocity direction exceeds the
     mean transverse extent (`λ₁` eigenvector of the position covariance
     roughly parallel to `T'(t)`, dot > 0.7) — the core-leads/tail-lags
     morphology.
3. **Golden re-pin**: influencer-mode dynamics change deliberately; re-pin
   the golden trajectory snapshot in the same commit as R3.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config loading + fields | — | ½ day | `core/config.py`, tests |
| R1 | Persistent tick state | R0 | ¼ day | `physics/flock.py`, `forces/influencer.py` |
| R2 | Lissajous target + embedding | R1 | ¼ day | `forces/influencer.py` |
| R3 | Move-then-steer unit-speed dynamics | R1, R2 | 1 day | `forces/influencer.py`, `physics/flock.py`, `physics/boid.py` |
| R4 | Rank/distance influence | R3 | ¼ day | `forces/influencer.py` |
| R5 | Density-scaled init | R0 | ¼ day | `physics/flock.py` |
| R6 | Distance diagnostics | R3 | ¼ day | `forces/influencer.py`, `analysis/metrics.py`, `viz/visualizer.py` |
| R7 | 3D visualization (alpha, projections, marker) | R2 | 1 day | `viz/renderer.py`, `viz/shaders.py`, `viz/camera.py`, `viz/input_control.py` |
| R8 | Preset + tests + golden | all | ½ day | `conf/`, `test/` |

Total ≈ **4 working days**. R5 and R7 are independent of the R3 dynamics
chain and can proceed in parallel once R0/R2 land.
