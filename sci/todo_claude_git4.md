# todo_claude_git4.md — Porting `Nikorasu/PyNBoids` to pymurmur (3D)

**Source:** https://github.com/Nikorasu/PyNBoids — a Python/Pygame boids
family built on **angle-based steering**: scalar headings, turn-rate-limited
rotation, distance-gated behaviour modes, adaptive speed, and pixel-fade
trails. Source files verified: `pynboids_sp.py` (spatial grid, recommended),
`pynboids.py`, `pynboids2.py` (numpy), `pixelboids.py` (trails),
`example_scene.py`, `nboids_ss.py`/`run_ss.py`, `pynboids_desktop.py`.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`).

**What this file is.** Every idea and piece of math in PyNBoids that pymurmur
does not implement, re-derived for **3D simulation and visualization**, with
formulas, constants, file paths, config fields, code sketches, and acceptance
tests — each roadmap item implementable from this file alone.

**Why this port is different from the other git specs.** PyNBoids is a
*third steering paradigm*: pymurmur's modes all write forces or velocities;
PyNBoids birds own a **heading** that rotates toward targets at a capped
rate, never faster. None of it exists in pymurmur — this spec adds a new
`"angle"` force mode plus five supporting mechanics. The 2D scalar angle
generalises to a 3D unit heading rotated about computed axes (Rodrigues).

**Already implemented in pymurmur (do not redo):** the 27-cell spatial-hash
query, vectorised k-nearest selection (cKDTree / argpartition), `boid_size`
as a config field (though not as the interaction unit — R8), pause/reset/
add-remove keyboard interaction, ESC to quit.

---

## Conventions used throughout

- **Heading state:** the bird's heading is the unit vector
  `ĥ = v/‖v‖`; the mode rewrites `v = ĥ·s` each step (per-bird speed `s`,
  R3). Positions/velocities are pymurmur's `(N,3) float32` SoA arrays,
  masked by `idx = np.where(flock.active)[0]`.
- **RNG:** all randomness from the flock-owned seeded generator
  (`PhysicsFlock.rng = np.random.default_rng(config.seed)`).
- **Rotation utility (used by R1, R4, R5)** — Rodrigues rotation of `v`
  about unit axis `k̂` by angle θ:

```
rot(v, k̂, θ) = v·cosθ + (k̂ × v)·sinθ + k̂·(k̂·v)·(1 − cosθ)
```

  Add to `pymurmur/core/types.py` as `rotate_about(v, axis, angle)` —
  vectorised over rows (`(N,3)` inputs, per-row axis/angle).
- **Body size unit:** `b = config.boid_size`. PyNBoids expresses every
  interaction distance as a multiple of the sprite size; R8 adopts that.

---

## R0 — Prerequisite: config fields for the angle mode

**Implementation.** New `SimConfig` fields (defaults = `pynboids_sp.py`
values, distances in body units per R8):

```python
# ── Angle mode (PyNBoids) ─────────────────────────────────────
angle_turn_rate: float = 120.0        # deg/sec base turn rate
angle_max_turn_rate: float = 200.0    # deg/sec at the wall (source: ~20 deg/frame @60fps)
angle_turn_threshold: float = 0.5     # deg dead-zone: no turn below this error
angle_jitter_deg: float = 4.0         # per-frame heading jitter amplitude (R5)
angle_margin: float = 42.0            # world-units edge margin (R4)
angle_speed_mode: str = "linear"      # "linear" (+5/missing) | "quadratic" (+(7-N)²) | "off"
angle_base_speed: float = 150.0       # world-units/sec (source sp variant: 150)
angle_neighbors: int = 7              # ideal/maximum neighbour count
sep_radius_bodies: float = 1.0        # mode-A threshold  = b·1
align_radius_bodies: float = 5.0      # mode-B threshold  = b·5
range_radius_bodies: float = 12.0     # neighbour cutoff  = b·12
```

Register the mode: add `"angle": angle_forces` to `_DISPATCH` in
`pymurmur/physics/forces/__init__.py`; add `"angle"` to the `_MODES` cycle
in `viz/input_control.py`. If the YAML section-prefix loader fix (flattening
`angle: turn_rate:` → `angle_turn_rate`; warn on unknown keys; stop
`capture: width:` overwriting the domain) is not yet applied from another
work stream, apply it in `SimConfig.from_file` first — without it these
fields never load from presets.

**Accept:** `--config` presets can set every field above; `M` cycles into
the angle mode.

---

## R1 — Angle-based steering core (3D)

**Idea (verbal).** The bird never sets its direction — it *turns toward* a
target direction at a capped angular rate, with a small dead zone so it
doesn't oscillate around the goal. In 2D the source computes the shortest
signed angle; in 3D the equivalent is the axis-angle rotation from current
heading to target: rotate about `ĥ × t̂` by at most `turnRate·dt`.

**Math.**

2D source (for reference — the wrap trick maps any difference into
(−180°, 180°]):

```
angleDiff = (targetAngle − currentAngle) + 180
turnDir   = (angleDiff/360 − ⌊angleDiff/360⌋)·360 − 180
if |turnDir| > threshold:  θ += turnRate·dt·sign(turnDir)
```

3D generalisation (per bird, target direction `t̂`):

```
c = clamp(ĥ·t̂, −1, 1);   φ = acos(c)                 (angle between, ∈ [0, π])
if φ ≤ threshold_rad: keep ĥ                          (dead zone)
axis = ĥ × t̂
if ‖axis‖ < 1e-6:                                     (parallel/anti-parallel)
    axis = any unit vector ⊥ ĥ                        (e.g. normalize(ĥ × x̂), fallback ĥ × ŷ)
k̂ = axis/‖axis‖
ĥ ← rot(ĥ, k̂, min(φ, turnRate_rad·dt))               (never overshoot: cap at φ)
```

**Implementation.** New file `pymurmur/physics/forces/angle.py`:

```python
def angle_forces(flock, config):
    idx = np.where(flock.active)[0]
    n = len(idx)
    if n == 0: return
    dt = 1.0 / 60.0
    pos, vel = flock.positions[idx], flock.velocities[idx]
    spd = np.linalg.norm(vel, axis=1, keepdims=True)
    h = np.where(spd > 1e-9, vel / np.maximum(spd, 1e-9),
                 _random_units(flock.rng, n))

    t_hat, rate = _steering_targets(flock, config, idx, pos, h, dt)  # R2 + R4
    h = _apply_jitter(h, config, flock.rng, dt)                      # R5 (before steering)
    h = _turn_toward(h, t_hat, rate * dt, math.radians(config.angle_turn_threshold))

    speed = _adaptive_speed(config, neighbor_counts)                 # R3
    flock.velocities[idx] = h * speed[:, None]
```

`_turn_toward` implements the boxed math with `rotate_about` from
`core/types.py`, fully vectorised (mask rows inside the dead zone; batch the
Rodrigues formula). Like the vicsek/influencer modes, this mode owns
velocity: `integrate()` must be called with `speed_mode="fixed"` (no band
clamp) for `mode == "angle"` — the mode sets exact speeds.

**Accept:** a bird with target 180° behind it turns smoothly through
`π / (turnRate_rad)` seconds, never snapping; heading change per frame never
exceeds `turnRate·dt + jitter`; a bird within the dead zone of its target
holds heading exactly.

---

## R2 — Unified neighbour behaviour (distance-gated modes)

**Idea (verbal).** Instead of summing three forces, the source builds **one
sorted neighbour list** (the 7 closest within `b·12`) and picks ONE
behaviour by the *nearest* neighbour's distance: too close → steer away from
that neighbour; close enough → steer toward the group centroid *while
matching the group's average heading*; otherwise → just head for the
centroid. Separation is an exclusive state, not a blended term.

**Math (3D).** For bird i with sorted neighbours `j₁…j_m` (m ≤ 7, all
within `R = b·12`), nearest distance `d₁`:

```
centroid  ĉ = normalize( (1/m)·Σ p_j − p_i )
mean dir  m̂ = normalize( Σ ĥ_j )                       (3D replaces atan2(Σsin, Σcos))

Mode A (d₁ < b·1):      t̂ = normalize(p_i − p_{j₁})     (away from nearest)
Mode B (d₁ < b·5):      t̂ = normalize( ĉ + m̂ )          (cohere at the flock heading)
Mode C (otherwise):     t̂ = ĉ                            (cohere only)
m == 0:                  t̂ = ĥ                            (no neighbours: hold course; R4 may override)
```

(The source's Mode B "angle matching overrides positional steering" maps to
weighting m̂ into the target; the equal-weight sum above reproduces its
visual behaviour — expose `angle_modeB_align_weight: float = 1.0` if tuning
is wanted.)

**Implementation.** In `_steering_targets`: one batched
`tree.query(pos, k=8)` (cKDTree, `workers=-1`; drop self column), mask
columns with `d > b·12`, `m = per-row count`. Vectorise the three modes with
`np.select` on `d₁` thresholds. Return `t̂` rows plus `m` (R3 needs the
counts) and the per-row turn rate (R4 adjusts it).

**Accept:** two birds forced within `b` steer apart (their pair distance
grows next frames); a loose cluster (spacing ≈ `b·8`) contracts toward its
centroid; a mid-range cluster (spacing ≈ `b·3`) both contracts *and* aligns
(polar α rises).

---

## R3 — Adaptive speed (self-regulating density)

**Idea (verbal).** Isolated birds hurry to rejoin; crowded birds ease off.
Speed is a function of how many of the ideal 7 neighbours are present —
the source's linear variant adds 5 units per missing neighbour; the original
uses a quadratic boost (stragglers *sprint*, max +49).

**Math.**

```
linear    : s_i = base + (7 − m_i)·5
quadratic : s_i = base + (7 − m_i)²
softened 3D (recommended default for shell-structured flocks):
            s_i = base + min(49, (7 − m_i)²·0.5)
m_i ≥ 7 → s_i = base
```

**Implementation.** `_adaptive_speed(config, m)` returns `(n,)` speeds per
`config.angle_speed_mode`; `"off"` → constant `angle_base_speed`. Applied as
the per-bird magnitude in R1's final write. Note: this is the first per-bird
speed in pymurmur — no `integrate` change needed beyond the
`speed_mode="fixed"` bypass already required by R1.

**Accept:** a bird separated from the flock by > `b·12` moves at
`base + 35` (linear) and visibly slingshots back; birds inside a 7+
neighbourhood cruise at exactly `base`; flock density self-stabilises
(median 7th-neighbour distance converges).

---

## R4 — Edge avoidance: cardinal targets + turn-rate scaling

**Idea (verbal).** Near a wall the source does two things: (1) it *replaces*
the steering target with the inward cardinal direction of the nearest edge
(a direction override, not a force), first-triggering edge wins; (2) it
raises the turn rate the deeper the bird is into the margin, so escape is
always geometrically possible.

**Math (3D, cuboid domain, margin M = `angle_margin`).**

```
if p.x < M:      t̂ = (+1, 0, 0)        elif p.x > W−M: t̂ = (−1, 0, 0)
elif p.y < M:    t̂ = (0, +1, 0)        elif p.y > H−M: t̂ = (0, −1, 0)
elif p.z < M:    t̂ = (0, 0, +1)        elif p.z > D−M: t̂ = (0, 0, −1)
(sequence priority exactly as ordered; overrides R2's target)

edgeDist = min(p.x, W−p.x, p.y, H−p.y, p.z, D−p.z)
if edgeDist < M:
    rate = turnRate + (1 − edgeDist/M)·(maxTurnRate − turnRate)
```

Spherical-domain variant (for `boundary_mode: "sphere"`): continuous inward
normal `t̂ = normalize(C − p)` when `‖p − C‖ > R − M`, same rate scaling with
`edgeDist = R − ‖p − C‖`.

**Implementation.** In `_steering_targets`, after R2: compute the six margin
tests vectorised (`np.select` in the listed priority order), override `t̂`
rows where any fires, and return the scaled per-row `rate`. Gate on
`config.boundary_mode in ("margin", "open")` — under `toroidal` there is no
edge and this stage is skipped (the source's `wrap` setting is the same
switch).

**Accept:** with `boundary_mode: margin`, no bird exits the domain over
10 000 frames even at `maxTurnRate` speeds; birds enter the margin, arc, and
leave — no sticking to walls (the cardinal override plus rate boost
guarantees the turning circle fits: require
`base_speed/√(maxTurnRate_rad·…) < M` in the preset).

---

## R5 — Per-frame heading jitter

**Idea (verbal).** Every frame, before steering, each heading gets a small
random rotation (±4°). Steering then compensates — the flock stays on
course, but individuals constantly micro-correct, which reads as organic
nervousness instead of robotic smoothness.

**Math (3D).** Random rotation axis ⊥ nothing in particular — uniform on the
sphere works; the source jitters a scalar angle:

```
θ_j ~ U(−jitter_rad, +jitter_rad)
k̂_j = random unit vector (uniform on S²)
ĥ ← rot(ĥ, k̂_j, θ_j)
```

(A random axis with a signed angle double-counts nothing: the effective
angular perturbation magnitude is |θ_j|·sin(angle between k̂ and ĥ) ≤ |θ_j| —
matching the ±4° cap.)

**Implementation.** `_apply_jitter(h, config, rng, dt)`: draw
`(n,)` angles and `(n,3)` unit axes from the seeded RNG, one batched
`rotate_about`. Applied **before** `_turn_toward` (source order) so steering
absorbs it. Skip when `angle_jitter_deg == 0`.

**Accept:** with steering disabled and jitter on, per-step heading change
distribution is bounded by ±4° and roughly symmetric; with steering on, the
flock's net track is unchanged (same endpoint ±2% vs jitter-off run) while
per-bird heading variance is higher.

---

## R6 — Incremental spatial grid

**Idea (verbal).** The source's grid is maintained, not rebuilt: each boid
remembers its cell and re-files itself only when it crosses a cell boundary.
pymurmur's `SpatialHashGrid.rebuild`
([physics/flock.py:152-163](pymurmur/physics/flock.py#L152-L163)) clears and
re-inserts every bird every frame in a Python loop — cost paid even when
nobody moved cells.

**Math/logic (3D).**

```
cell(p) = (⌊p.x/cs⌋, ⌊p.y/cs⌋, ⌊p.z/cs⌋)         cs = cell size (source: 100)
per bird: if cell(p) ≠ last_cell:
    bins[last_cell].remove(i); bins[cell(p)].append(i); last_cell = cell(p)
query: union of the 27 cells around cell(p)       (already implemented)
```

**Implementation.** In `SpatialHashGrid`: add
`self._last_cell = np.full((N,), -1, dtype=np.int64)` storing a packed key
(`cx + cy·P + cz·P²`, P large prime or per-axis shift); `update(positions,
active)` computes all keys vectorised (`np.floor_divide`), finds
`moved = keys != self._last_cell`, and re-files only those rows (Python loop
over `np.where(moved)[0]` — typically a few % of N per frame). Keep
`rebuild()` for the first frame and post-`reset`. Micro-fix while there:
`query_knn` should select on **squared** distances and take the sqrt only
for the k winners (the source's `argsort`-on-squared pattern; current code
sqrt's every candidate).

**Accept:** identical neighbour sets to full rebuild (property test over
random walks); at N=5 000 with typical speeds, `update()` touches < 10% of
birds per frame and beats `rebuild()` wall-clock.

---

## R7 — Fading trails (pixel-fade idea, two 3D forms)

**Idea (verbal).** `pixelboids.py` never stores trail history: birds stamp
pixels into a buffer that **decays every frame**, so trails emerge from
persistence of vision. In 3D, two faithful translations: (a) screen-space —
fade the previous frame instead of clearing it; (b) world-space — a short
ring buffer of past positions rendered as shrinking, fading sprites
(true parallax; recommended).

**Math.**

Source fade (per frame, 8-bit buffer, FADE = 30):

```
img[img > 0] −= FADE · (60/FPS/1.5) · ((dt/10)·FPS)      then clip to [0, 255]
(≈ 20 intensity units/frame at 60 fps → ~12-frame persistence)
```

(a) Screen-space accumulation: draw a full-screen quad of the background
colour at `alpha_fade = FADE_frame/255 ≈ 0.08` over the previous frame
(**clear depth only, keep colour**), then draw birds.

(b) Ring buffer: keep K = 12 past positions per bird;
trail sprite `k` frames old gets `scale = 1 − k/K`, `alpha = (1 − k/K)·0.6`.

**Implementation.**

(a) `viz/renderer.py`: `begin_frame(fade=True)` path — skip
`ctx.clear(color)`; clear depth via `ctx.clear(depth=1.0, color=False)`
equivalents (moderngl: render the fade quad with blending
`(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)`, depth test off, then
`ctx.clear(depth=...)` is not separable — instead disable depth write for
the quad and clear only the depth renderbuffer via
`fbo.depth_attachment` re-clear). Wire to `config.trails: "accumulation"`.

(b) `PhysicsFlock` gains `self.trail = np.zeros((K, N, 3), float32)` written
round-robin each step (`self.trail[frame % K] = positions`); renderer packs
`K·N` extra instances with per-instance scale/alpha columns. Wire to
`config.trails: "velocity"`→rename `"ring"`; K = `trail_length: int = 12`.

Both make `config.trails` — currently a dead field — live.

**Accept:** (a) shows ~12-frame streaks at 60 fps that fully clear when
paused for a second; (b) shows 3D trails with parallax under camera orbit;
`trails: off` is pixel-identical to today.

---

## R8 — Body-size-relative radii + per-bird colour

**Idea (verbal).** Two presentation/tuning principles from the source:
(1) **scale invariance** — every interaction distance is a multiple of the
bird's rendered size, so resizing birds retunes the whole simulation
coherently (R0's `*_bodies` fields implement this for the angle mode;
extend the principle: derive spatial-mode `visual_range` and steric range
from `boid_size` when a `radii_in_bodies: bool = False` flag is on);
(2) **per-bird colour** — HSV with random hue, S=V=90%
(`hue ~ U(0°,360°)`), giving individual identity at zero cost.

**Math.** HSV→RGB with S=V=0.9: standard conversion; per bird
`hue = rng.uniform(0, 360)` fixed at spawn (store in `flock.seeds` — it
already exists per bird and is unused by most modes: `hue = seeds·360`).

**Implementation.** Renderer instance layout gains a colour source: either a
7th float (hue; convert in the vertex shader) or reuse the existing
per-instance velocity for speed-tint plus a `u_per_bird_hue` toggle. Minimal:
add `in_bird_hue` (`'3f 3f 1f/i'`), shader
`base_color = mix(theme_color, hsv2rgb(vec3(hue, 0.9, 0.9)), u_hue_mix)`;
`config.per_bird_color: bool = False`. (If the species-flag column from the
predator work exists, hue shares the same float: flag ≥ 2.0 encodes hue as
`flag − 2.0`; simpler to keep separate columns.)

**Accept:** `per_bird_color: true` renders a stable rainbow flock (colours
don't flicker frame to frame — they derive from `seeds`); setting
`boid_size: 18` with `radii_in_bodies: true` scales all angle-mode
thresholds proportionally (behaviour visually similar at double scale).

---

## R9 — Scenes & application modes (optional tier)

**Idea.** Three source features that are packaging, not physics:

1. **Dual-layer parallax scene** (`example_scene.py`): two *independent*
   flocks sharing one window — background layer smaller/darker, foreground
   larger/brighter. In true 3D this is two flocks at different depths; the
   blocker in pymurmur is architectural: `SimulationEngine` owns exactly one
   `PhysicsFlock`. Minimal viable: a `MultiScene` driver in
   `analysis/`/`scripts/` that owns two `SimulationEngine`s and one
   renderer, drawing both instance buffers per frame (second draw call);
   per-flock tint via R8.
2. **Screensaver mode** (`run_ss.py` idea): idle-watch + **fresh subprocess
   per activation** (memory isolated by process exit). macOS idle time:
   `ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1e9; exit}'`.
   Ship `scripts/screensaver.py`: poll every 60 s, above threshold spawn
   `subprocess.call([sys.executable, "-m", "pymurmur", "--config", ...])`,
   kill on activity.
3. **Desktop overlay** (`pynboids_desktop.py`): render over a desktop
   screenshot. 3D translation: grab the desktop once
   (`PIL.ImageGrab.grab()`), draw it as a fullscreen textured quad behind
   the flock. Honest note: a *live transparent* overlay needs per-OS window
   compositing beyond pygame — the screenshot-background variant is what the
   source actually does; implement that.

**Accept:** the two-flock scene runs both simulations independently (pausing
one leaves the other moving); the screensaver launcher starts and cleanly
kills the sim; the desktop variant shows birds over a frozen desktop image.

---

## R10 — Preset, tests, golden

**Implementation.**

1. **Preset** `conf/murmuration_angle.yaml`: `mode: angle`,
   `num_boids: 200`, `boid_size: 9`, `boundary: margin`,
   `angle: {turn_rate: 120, max_turn_rate: 200, turn_threshold: 0.5,
   jitter_deg: 4, margin: 42, speed_mode: linear, base_speed: 150,
   neighbors: 7}`, `visual: {theme: inverse, per_bird_color: true,
   trails: ring}`.
2. **Tests** (`test/physics/test_angle_mode.py`): the acceptance assertions
   from R1–R6 above, plus determinism (same seed → identical positions after
   200 steps) and a containment test (margin boundary, 10 000 steps, zero
   escapes).
3. **Golden**: the angle mode is new (no re-pin of existing modes needed);
   pin its own golden trajectory alongside.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config fields + mode registration | — | ¼ day | `core/config.py`, `forces/__init__.py`, `viz/input_control.py` |
| R1 | Angle-steering core + `rotate_about` | R0 | 1 day | `core/types.py`, `forces/angle.py` (new), `physics/boid.py`, `physics/flock.py` |
| R2 | Unified neighbour modes | R1 | ½ day | `forces/angle.py` |
| R3 | Adaptive speed | R2 | ¼ day | `forces/angle.py` |
| R4 | Edge avoidance + turn-rate scaling | R1 | ½ day | `forces/angle.py` |
| R5 | Heading jitter | R1 | ¼ day | `forces/angle.py` |
| R6 | Incremental grid | — | ½ day | `physics/flock.py` |
| R7 | Fading trails (both forms) | — | 1 day | `viz/renderer.py`, `viz/shaders.py`, `physics/flock.py`, `core/config.py` |
| R8 | Body-unit radii + per-bird colour | R0 | ½ day | `viz/renderer.py`, `viz/shaders.py`, `core/config.py` |
| R9 | Scenes / screensaver / overlay | R8 | 1 day (optional) | `scripts/`, `viz/` |
| R10 | Preset + tests + golden | R1–R5 | ½ day | `conf/`, `test/` |

Total ≈ **5 working days** core (R0–R8, R10) + 1 optional (R9). R6 and R7
are independent of the angle-mode chain and can be developed in parallel at
any point.
