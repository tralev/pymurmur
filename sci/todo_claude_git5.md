# todo_claude_git5.md — Porting `djin31/Starlings` to pymurmur (3D)

**Source:** https://github.com/djin31/Starlings — a Processing 3 (Java) boids
simulation of starling murmurations, already **natively 3D**, built on Craig
Reynolds' model with Hildenbrandt & Hemelrijk influences. Its distinctive
contributions: a hybrid metric+topological neighbour filter, spherical
confinement, **physically calibrated energy/power metrics** (kg, m/s, m/s²),
per-frame parameter jitter, live sliders, and a winged, flapping bird mesh.
Source files verified: `src/starlings/boid.pde`, `flock.pde`,
`starlings.pde`, `sliders.pde`, `button.pde`.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`).

**What this file is.** Every idea and piece of math in Starlings that
pymurmur does not implement (or implements with different math), adapted to
pymurmur's 3D pipeline, with formulas, constants, file paths, config fields,
code sketches, and acceptance tests — each roadmap item implementable from
this file alone.

**Already implemented in pymurmur (do not redo):** the three-term Reynolds
decomposition as separate primitives, a max-force clamp, an impulsive
spherical-wall correction (but mis-centred — R2), runtime boid injection
(+/− keys ≈ the source's add-boid button), fast metrics
`⟨|a·v|⟩ / ⟨|a|⟩ / ⟨|v|⟩ / dispersion` (unscaled — R3), and GPU LookAt
orientation from velocity (the source's yaw/pitch pair — only the *mesh* is
missing, R6).

**Source constants (verified):**

```
INFLUENCE          = 7            (max neighbour count)
INFLUENCE_CIRCLE   = 80.0         (metric radius; alignment uses ×0.75)
RADIUS_OF_CONFINEMENT = 300.0
mass               = 0.075 kg
acc_peak           = 40 m/s²
speed calibration  = 8.94 m/s  (≈ 20 mph starling cruise) / FLIGHT_SPEED
sliders            = sep 1–5 (3.0) · coh 0–2 (0.2) · align 0–0.5 (0.02)
                     · avoid 0–1 (0.05) · noise 0–0.5 (0.05)
per-frame jitter   = sep +U(0,0.5) · coh +U(0,0.1) · align +U(0,0.005)
metrics cadence    = every 20th frame
wing flap          = (−1)^⌊frameCount/100⌋ / 2
```

---

## Conventions used throughout

- All vectors are 3-vectors in pymurmur's `(N,3) float32` SoA arrays, masked
  by `idx = np.where(flock.active)[0]`. Domain `[0,W)×[0,H)×[0,D)`, centre
  `C = (W/2, H/2, D/2)`.
- **RNG:** every draw from the flock-owned seeded generator
  (`PhysicsFlock.rng = np.random.default_rng(config.seed)`).
- Helpers `normalize3(v)` (0-safe) and `limit3(v, m)` in `core/types.py`
  (add if absent).

---

## R0 — Prerequisite: config fields

**Implementation.** New/activated `SimConfig` fields (defaults = source):

```python
# ── Starlings spatial-mode variant ────────────────────────────
neighbor_filter: str = "knn"          # "knn" (legacy) | "hybrid" (source: radius AND count cap)
influence_count: int = 7              # topological cap in hybrid mode
alignment_radius_ratio: float = 0.75  # EXISTS, currently dead — R1 wires it
separation_kernel: str = "inverse_square"  # source uses true r̂/d² (R1 fixes the current 1/d)
speed_mode: str = "band"              # "band" | "fixed" (source: |v| = v0 exactly)
parameter_jitter: bool = False        # R4
jitter_separation: float = 0.5
jitter_cohesion: float = 0.1
jitter_alignment: float = 0.005
# ── Physical calibration (R3) ─────────────────────────────────
bird_mass_kg: float = 0.075
cruise_speed_ms: float = 8.94         # real-world m/s mapped to v0
acc_peak_ms2: float = 40.0            # real-world m/s² mapped to max_force
# ── Rendering (R6/R7) ─────────────────────────────────────────
bird_mesh: str = "tetra"              # "tetra" (legacy) | "winged" (source)
flap_period_frames: int = 100
background: str = "flat"              # "flat" | "gradient"
```

If the YAML section-prefix loader fix (flatten `spatial: x:` → try
`spatial_x` first; warn on unknown keys; stop `capture: width:` overwriting
the domain `width`) has not been applied from another work stream, apply it
in `SimConfig.from_file` first — otherwise sectioned presets never deliver
these fields.

**Accept:** round-trip `to_file→from_file` equality including new fields.

---

## R1 — The Starlings force model (hybrid filter, dual radii, exact kernels)

**Idea (verbal).** Four mechanics distinguish the source's spatial model from
pymurmur's:

1. **Hybrid metric-topological neighbour selection** — a neighbour counts
   only if it is *both* within the influence circle *and* among the first
   `INFLUENCE = 7` accepted (a sequential filter: O(N), accepts the first 7
   in range rather than sorting for the exact 7 closest — Young's optimum
   as a cap, metric perception as a gate).
2. **Dual radii** — cohesion uses the full circle; alignment only the inner
   75% (`0.75 × INFLUENCE_CIRCLE`). Velocity-matching is a more local
   phenomenon than positional attraction; the two nested spheres produce
   richer layering.
3. **Exact force kernels** — separation is a true inverse-square over
   *unit* vectors (pymurmur's current code divides an unnormalised
   difference by d², yielding a 1/d falloff); alignment and cohesion are
   Reynolds *steering* forms (desired − current, normalised); noise is a
   δ-scaled random unit vector (pymurmur currently normalises the noise and
   discards δ entirely).
4. **Fixed-speed renormalisation** — after integrating, velocity is
   rescaled to exactly `FLIGHT_SPEED`: birds never slow down or stall; all
   dynamics live in direction (contrast pymurmur's `[0.3·v0, v0]` band).

**Math (3D; per bird i).**

```
Neighbour filter (hybrid mode):
  N_i = first ≤ 7 birds j (any order the index yields) with d_ij < R      (R = visual_range)
  A_i = { j ∈ N_i : d_ij < 0.75·R }                                        (alignment subset)

Separation (over j with d_ij < MIN_SEP ≡ separation_distance):
  Δv_sep = −α · Σ r̂_ij / d_ij²            r̂_ij = (p_j − p_i)/d_ij         (unit vector / d²)

Alignment (over A_i):        v̄ = Σ v_j / |A_i|
  Δv_align = β · normalize(v̄ − v_i)

Cohesion (over N_i):         p̄ = Σ p_j / |N_i|
  Δv_coh = γ · normalize(p̄ − p_i)

Noise:  Δv_noise = δ · û_rand              (û_rand uniform on S²; δ = noise_scale)

Total:  a = Δv_sep + Δv_align + Δv_coh + Δv_noise [+ Δv_wall (R2)]
        if ‖a‖ > MAX_FORCE:  a ← a·MAX_FORCE/‖a‖
        v ← v + a·dt
        v ← v̂ · v0                          (speed_mode="fixed": exact renormalisation)
        p ← p + v·dt
```

**Implementation.**

1. `pymurmur/physics/forces/spatial.py`: when
   `config.neighbor_filter == "hybrid"`, build neighbour lists via
   `tree.query_ball_point(pos, config.visual_range, workers=-1)` and truncate
   each list to `influence_count` (sequential-accept parity; the order scipy
   returns is fine — the source never sorts either). Build the alignment
   mask per row from `d < 0.75·visual_range` (wires the dead
   `alignment_radius_ratio`).
2. `forces/_base.py`: fix `separation_force` to normalise the difference
   before dividing (`(p_i−p_j)/d · 1/d² = unit/d²`) under
   `separation_kernel="inverse_square"`; fix `cohesion_force` to
   `normalize(p̄ − p_i)`; fix `noise_force` to **multiply the unit vectors by
   `scale`** after normalising (the current code returns unit vectors and
   throws δ away).
3. `physics/boid.py::integrate`: add `speed_mode="fixed"` — replace both
   clamps with `v = v̂·v0` (0-speed guard: re-seed a random unit from the
   flock RNG × v0).

**Accept:** with source defaults (`sep 3.0, coh 0.2, align 0.02, noise 0.05,
R=80, cap 7, fixed speed`), a 150-bird flock forms cohesive rotating groups
inside the confinement sphere (R2) within 500 frames; every bird's speed is
exactly `v0` every frame; halving `noise_scale` halves the measured heading
jitter (δ is live again); alignment neighbour counts ≤ cohesion neighbour
counts (dual radii active).

---

## R2 — Spherical confinement, correctly centred (+ the asymptotic variant)

**Idea (verbal).** The flock lives inside an invisible sphere. The source
ships the **impulsive** form — once outside the radius, subtract a fraction
of the radial direction from velocity (`v −= μ·r̂` once per breach). Its
paper/`doc` describes the smooth **asymptotic** form, a `1/(R−r)` repulsion
that hardens near the shell. pymurmur has a sphere mode but measures `‖p‖`
from the **origin** — while the domain spans `[0,1000]×[0,700]×[0,400]`, so
every bird is permanently "outside" and gets dragged toward the corner. Fix
the centre; ship both force forms.

**Math (3D; centre C, radius R = `boundary_sphere_radius`, μ =
`boundary_avoidance_factor`).**

```
r_vec = p − C;   r = ‖r_vec‖;   r̂ = r_vec/r

impulsive (source code):     if r > R:  v ← v − μ·r̂
asymptotic (source theory):  Δv_wall = −μ·r̂ / (R − r)      applied while r < R
                             (diverges as r → R; add ε: 1/max(R − r, 0.05·R))
```

**Implementation.** `physics/boid.py::_sphere_soft`: replace
`np.linalg.norm(positions)` with `np.linalg.norm(positions − C)` (pass
W,H,D — already parameters). Add `boundary_mode: "sphere_soft"` implementing
the asymptotic form as a per-frame velocity increment for birds *inside* the
shell margin, keeping `"sphere"` as the impulsive/projection form. Both
centred on C.

**Accept:** a flock initialised at the domain centre stays statistically
centred on C (‖CoM − C‖ < 0.1·R over 5 000 frames) in both modes; no drift
toward the origin corner; in `sphere_soft`, birds decelerate radially
*before* crossing R (max ‖p−C‖ < R for μ ≥ 0.05).

---

## R3 — Physically calibrated metrics: power, energy, force, angular momentum

**Idea (verbal).** The source's headline feature: metrics reported in
**real-world units**, calibrated by three constants — starling mass
0.075 kg, cruise speed 8.94 m/s (mapped to whatever `FLIGHT_SPEED` is in sim
units), and peak acceleration 40 m/s² (mapped to `MAX_FORCE`). Power is the
steering work rate `P = m·a·v` (source omits the mass in code; include it —
the doc flags the omission), and energy is power integrated over the run.
pymurmur computes the dimensionless cousins only.

**Math (per frame; sim quantities on the right).**

```
k_v = cruise_speed_ms / v0                    (m/s per sim-speed-unit)
k_a = acc_peak_ms2 / max_force                (m/s² per sim-accel-unit)
m   = bird_mass_kg

|v|_real   = k_v · |v|_sim                                   (m/s)
|a|_real   = k_a · |a|_sim                                   (m/s²)
F_avg      = (m/N) · Σ k_a·|a_i|                             (newtons)
P_avg      = (m/N) · Σ | (k_a·a_i) · (k_v·v_i) |             (watts; |dot product|)
E_avg      = Σ_frames P_avg(t) · Δt_real                     (joules/bird; Δt_real = 1/fps s)
L_i        = m · (r_i − CoM) × (k_v·v_i)                     (kg·m²/s)
L_avg      = (1/N) · Σ ‖L_i‖
```

3D adjustment on L: the source uses raw positions (origin-relative — its
world is origin-centred); pymurmur's domain is not, so compute **about the
CoM** (translation-invariant, physically the spin of the flock). Positions
also need the length scale: if a metres-per-unit calibration is wanted,
add `k_x = k_v · Δt_real` (consistent kinematics); otherwise report L in
mixed units and document it — the source does the latter.

**Implementation.** `pymurmur/analysis/metrics.py`:

1. `FlockMetrics` gains `speed_real, accel_real, force_real_N, power_real_W,
   energy_J, angular_momentum_real` fields.
2. `MetricsCollector.__init__` reads the three calibration constants from
   config; `collect()` fills the fields (note: sample **accelerations before
   `integrate` zeroes them** — collect() already runs after `flock.step`, so
   either buffer `accelerations.copy()` in the engine pre-integrate, or move
   the acceleration-based metrics to read a stashed
   `flock.last_accelerations` written by `PhysicsFlock.step` before
   integration — do the stash, it is two lines).
3. Energy: running accumulator on the collector
   (`self._energy += power_real * (1/config.fps)`), reset with the engine.
4. Export in CSV/JSON automatically (they serialise `FlockMetrics`).

**Accept:** with `v0=4, max_force=0.15` and defaults, a settled flock
reports `speed_real ≈ 8.9 m/s`; `power_real_W` is positive and spikes during
predator passes; `energy_J` is monotonically increasing and ≈ mean power ×
elapsed seconds (±1%).

---

## R4 — Per-frame parameter jitter (environmental stochasticity)

**Idea (verbal).** Every frame, the *global* steering weights wander upward
by a small uniform random amount — not per-bird noise but slow environmental
variation (gusts, collective mood). The separation weight breathes hardest
(+U(0, 0.5) on a 1–5 scale), producing visible density pulsing; alignment
barely (+U(0, 0.005)). Avoidance and noise stay deterministic.

**Math (each frame, applied to the *effective* weights only — sliders/config
keep the base value).**

```
sep_eff   = separation_weight + U(0, jitter_separation)      (default 0.5)
coh_eff   = cohesion_weight   + U(0, jitter_cohesion)        (default 0.1)
align_eff = alignment_weight  + U(0, jitter_alignment)       (default 0.005)
```

**Implementation.** Top of `spatial_forces` (and the hybrid path), when
`config.parameter_jitter`:

```python
r = flock.rng
w_sep   = config.separation_weight + r.uniform(0, config.jitter_separation)
w_coh   = config.cohesion_weight   + r.uniform(0, config.jitter_cohesion)
w_align = config.alignment_weight  + r.uniform(0, config.jitter_alignment)
```

Use the `w_*` locals in the accumulation; never write back to config (the
UI reads config as the base value). Seeded RNG keeps runs reproducible.

**Accept:** with jitter on, the flock's local spacing time-series shows
higher variance than jitter-off at identical seeds (measure std of median
7th-neighbour distance over 1 000 frames); config values are unchanged after
the run; same-seed determinism still holds.

---

## R5 — Live control surface: sliders, speed keys, metrics readout

**Idea (verbal).** The source's interactivity: five draggable sliders on a
side panel bound to the steering weights, arrow keys stepping the global
flight speed, an add-boid button, and a metrics readout refreshed every 20th
frame. pymurmur has keys for φp/φa/σ only — the spatial weights have **no
runtime controls at all** — and its title bar shows a subset of metrics.

**Math (slider widget).** A horizontal slider at screen rect
`(x0, y0, w, h)` maps knob position to value and back:

```
value(mx)  = low + (high − low) · clamp((mx − x0)/w, 0, 1)
knob_x(v)  = x0 + w · (v − low)/(high − low)
hit test   = x0−6 ≤ mx ≤ x0+w+6  and  y0−6 ≤ my ≤ y0+h+6
```

Slider set (label, low, high, initial → config field):

```
Separation  1.0  5.0  3.0   → separation_weight
Cohesion    0.0  2.0  0.2   → cohesion_weight
Alignment   0.0  0.5  0.02  → alignment_weight
Avoidance   0.0  1.0  0.05  → boundary_avoidance_factor
Noise       0.0  0.5  0.05  → noise_scale
```

**Implementation.**

1. **Slider panel** (`pymurmur/viz/hud.py`, new): pymurmur draws through a
   moderngl context, so render sliders as flat quads/lines with the existing
   grid shader in an orthographic pixel-space pass at frame end
   (`u_View = I`, `u_Projection = glm.ortho(0, w, h, 0, -1, 1)`): one track
   line + one 8×16 knob quad per slider, right-edge column. Mouse handling
   in `viz/input_control.py`: on MOUSEBUTTONDOWN inside a slider's hit rect,
   lock it (and suppress camera orbit); on MOUSEMOTION while locked, write
   `value(mx)` to the bound config field; unlock on MOUSEBUTTONUP.
   Toggle panel with `TAB`.
2. **Flight-speed keys**: `PageUp/PageDown` (arrows are taken by φp/φa):
   `cfg.v0 = max(0.3, cfg.v0 ± 0.1)` — works live because `integrate` reads
   `v0` per step; under `speed_mode="fixed"` the whole flock retunes
   instantly (the source's exact behaviour).
3. **Metrics readout, 20-frame cadence**: extend the window-title builder to
   include the R3 physical values, refreshed when `frame % 20 == 0`
   (`f"v={speed_real:.1f}m/s a={accel_real:.1f}m/s² P={power_real_W:.2f}W
   L={L_avg:.2f} σr={dispersion:.0f}"`), and mirror the same string to
   stdout at the same cadence when `--no-viz` is off. (On-canvas text needs
   a font atlas — out of scope here; the title+console pair covers the
   source's information content.)

**Accept:** dragging Separation from 1→5 visibly loosens the flock within a
second; PageUp raises measured `speed_real` by ≈ `0.1·k_v` per press;
title/console update exactly every 20 frames.

---

## R6 — Winged bird mesh with flapping

**Idea (verbal).** The source's bird is not a bare tetrahedron: a pointed
body **plus two wing triangles and a tail cap** (6 triangles), with wings
that flap by toggling their tip offset every 100 frames — a slow ~1.7 s
cycle at 60 fps that makes individuals read as birds at close zoom. The
source flaps along global z; the correct 3D form flaps along the bird's
**local up**, which pymurmur's per-instance LookAt basis provides for free.

**Math / geometry.** Mesh-space: forward = +Z (matches pymurmur's existing
tetra), up = +Y, wingspan along X. Body 6 units long, wingspan 16, per the
source's proportions (`SCALE = 1.2` overall):

```
T  = ( 0.0,  0.0,  3.0)      body tip
B1 = (−1.0,  0.0, −3.0)      back left
B2 = ( 1.0,  0.0, −3.0)      back right
B3 = ( 0.0,  1.0, −3.0)      back top
WL = (−8.0,  0.0, −1.0)      left wing tip   (flap weight 1)
WR = ( 8.0,  0.0, −1.0)      right wing tip  (flap weight 1)
RL = (−0.8,  0.1,  0.8)      left wing root
RR = ( 0.8,  0.1,  0.8)      right wing root

faces: (T,B1,B2) (T,B2,B3) (T,B3,B1)      body
       (B1,B3,B2)                          tail cap
       (RL,WL,B1) (RR,B2,WR)               wings

flap(t) = (−1)^⌊frame / flap_period_frames⌋ · 0.5          (period 100 → ±0.5 toggle)
vertex:  y += flap(t) · flap_weight        (BEFORE the LookAt rotation ⇒ local-up flap)
```

**Implementation.**

1. `pymurmur/viz/shaders.py`: add `WINGED_VERTS / WINGED_NORMALS /
   WINGED_INDICES / WINGED_FLAP` (per-vertex flap weight: 1.0 for WL/WR,
   else 0.0). Vertex shader gains `in_FlapWeight` attribute and `u_Flap`
   uniform; apply `pos.y += u_Flap * in_FlapWeight` before rotation.
2. `viz/renderer.py`: select mesh by `config.bird_mesh`; per frame set
   `u_Flap = 0.5 if (frame // config.flap_period_frames) % 2 == 0 else -0.5`
   (renderer needs the frame counter — pass `sim.frame` into `draw_birds`
   or set via `begin_frame`).
3. Normals for the wing triangles: face normals are fine (flat shading on
   wings reads well); recompute per-vertex if smooth shading is preferred.

**Accept:** `bird_mesh: winged` renders visibly winged birds oriented along
velocity; wings toggle position every `flap_period_frames` frames; the flap
displaces wing tips along each bird's local up (verify: a bird flying
straight up flaps horizontally).

---

## R7 — Gradient sky background

**Idea.** The source clears to a vertical gradient — pale cyan sky
(top `RGB(153,255,255)` → bottom `RGB(175,238,238)`) — instead of a flat
colour; a cheap but strong depth cue.

**Implementation.** `viz/renderer.py`: when `config.background ==
"gradient"`, after `ctx.clear()` draw a fullscreen quad (depth test/write
off, drawn first) with a two-colour vertex-interpolated shader:

```glsl
// vertex: pass uv;  fragment:
vec3 top = u_TopColor, bottom = u_BottomColor;      // (0.60,1.0,1.0) / (0.686,0.933,0.933)
fragColor = vec4(mix(bottom, top, v_uv.y), 1.0);
```

Expose `background_top` / `background_bottom` as config colour triples with
the source's defaults; themes may override.

**Accept:** gradient renders behind birds and grid; capture GIFs show it;
`background: flat` is pixel-identical to today.

---

## R8 — Optional tier: Hildenbrandt & Hemelrijk flight physics

**Idea.** The source's README names H&H (2011) as an inspiration but its
code implements none of the aerodynamic terms. Recorded here as the natural
extension tier, with starting math, explicitly **not** part of source
parity:

```
gravity:   a += (0, 0, −g)                                  (z-up world; g in sim units)
lift:      a += (0, 0,  k_L·|v|²)  with k_L tuned so lift = g at |v| = v0
drag:      a += −k_D·(|v| − v0)·v̂                           (relaxes speed toward cruise —
                                                             replaces the hard clamp)
banking:   turn rate ∝ roll angle: cap lateral acceleration by a_lat ≤ g·tan(φ_max)
roost:     a += k_R·normalize(roost − p) gated by dusk factor (Ecology extension exists)
```

Implement, if desired, as a `physics/extensions/flight.py` extension
(gravity+lift+drag) applied in `pre_step`, config-gated
(`flight_physics: bool = False`). Acceptance: level flight at `v0` is
self-sustaining; slowed birds descend; the speed distribution becomes
unimodal around `v0` without a hard clamp.

---

## R9 — Preset, tests, golden

**Implementation.**

1. **Preset** `conf/murmuration_starlings.yaml`: `mode: spatial`,
   `num_boids: 150`, `v0: 4.0`, `visual_range: 80`,
   `spatial: {separation_weight: 3.0, cohesion_weight: 0.2,
   alignment_weight: 0.02, noise_scale: 0.05, neighbor_filter: hybrid,
   influence_count: 7, alignment_radius_ratio: 0.75,
   speed_mode: fixed, parameter_jitter: true}`,
   `boundary: {mode: sphere, sphere_radius: 300, avoidance_factor: 0.05}`,
   `visual: {bird_mesh: winged, background: gradient}`,
   physical constants at defaults.
2. **Tests** (`test/physics/test_starlings_port.py`): the acceptance
   assertions from R1–R6, plus determinism (same seed → identical positions,
   jitter on) and unit-calibration spot checks
   (`speed_real == v0·k_v` exactly for a hand-set velocity).
3. **Golden**: R1/R2 deliberately change spatial-mode dynamics — re-pin the
   golden trajectory in the same commit.

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config fields (+ loader fix if pending) | — | ¼ day | `core/config.py`, tests |
| R1 | Starlings force model (hybrid filter, kernels, fixed speed) | R0 | 1 day | `forces/spatial.py`, `forces/_base.py`, `physics/boid.py` |
| R2 | Sphere confinement centring + asymptotic form | R0 | ¼ day | `physics/boid.py` |
| R3 | Physical metrics (P, E, F, L in real units) | R0 | ½ day | `analysis/metrics.py`, `physics/flock.py`, `simulation/engine.py` |
| R4 | Parameter jitter | R1 | ¼ day | `forces/spatial.py` |
| R5 | Sliders + speed keys + 20-frame readout | R3 | 1 day | `viz/hud.py` (new), `viz/input_control.py`, `viz/visualizer.py` |
| R6 | Winged flapping mesh | — | ½ day | `viz/shaders.py`, `viz/renderer.py` |
| R7 | Gradient background | — | ¼ day | `viz/renderer.py`, `viz/shaders.py`, `core/config.py` |
| R8 | H&H flight physics (optional) | R1 | 1 day (opt) | `physics/extensions/flight.py` (new) |
| R9 | Preset + tests + golden | R1–R7 | ½ day | `conf/`, `test/` |

Total ≈ **4½ working days** core (R0–R7, R9) + 1 optional (R8). R6 and R7
are independent of the physics chain and can be developed in parallel from
day one.
