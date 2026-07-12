# todo_claude_git6.md — Porting `crs48/murmuration` to pymurmur (3D)

**Source:** https://github.com/crs48/murmuration — a browser 3D murmuration
(Three.js/TypeScript) with a field/blob "lava-lamp" simulation, an autonomous
predator FSM, GPU trails, sphere impostors, monochrome themes, adaptive
quality, and experimental XR. Source verified: `src/simulation/`
(`CpuMurmurationSimulation.ts`, `rules.ts`, `threat.ts`, `flockWander.ts`,
`swarmCenter.ts`, `cpuSpatialHash.ts`, `particleInitialization.ts`),
`src/rendering/`, `src/diagnostics/`.
**Target:** `/Users/tralev/Developer/git_mur` (`pymurmur/`), whose `field`
force mode ([pymurmur/physics/forces/field.py](pymurmur/physics/forces/field.py))
implements ~4 of the source's 13 force terms, and whose Wander/Ripple/Predator
extensions are simplified sketches of this project's systems.

**What this file is.** Every idea and piece of math in crs48/murmuration that
pymurmur lacks, adapted to pymurmur's 3D pipeline (the source is already 3D —
adaptation means mapping its unit-scale, origin-centred world onto pymurmur's
`[0,W)×[0,H)×[0,D)` domain and moderngl renderer), with formulas, constants,
file paths, config fields, and acceptance tests. Each roadmap item is
implementable from this file alone.

**Already implemented in pymurmur (do not redo):** SoA float32 arrays, a
27-cell spatial hash + cKDTree, a field mode skeleton (CoM pull, drift
alignment, a sine flow, forward-only slot repulsion), a Ripple extension with
the three train offsets {0, 9.33, 18.67}, a Wander extension (different
math — replaced in R2), a minimal predator FSM, four theme palettes
(unwired), and an EMA perf tracker (unconsumed).

**Scale mapping convention.** The source world is roughly a unit sphere
(`R_pilot ≈ 1`). Define once:

```
U = 0.4 · min(W, H, D)        # world units per source-unit (≈160 for 1000×700×400)
C = (W/2, H/2, D/2)           # domain centre = source origin
```

All source lengths below are written in source units; multiply by `U` when a
formula produces a position/radius (config: `field_unit_scale: float = 0.0`
→ 0 means auto `0.4·min(W,H,D)`).

---

## Conventions

- `idx = np.where(flock.active)[0]`; per-bird arrays `(N,3) float32`.
- **RNG:** the flock-owned seeded generator (`flock.rng`), never module
  `np.random.*`. (The source uses seedable Mulberry32 for the same reason —
  numpy's `default_rng(seed)` is the Python equivalent; no port needed.)
- `t` = simulation time in seconds (`frame / fps`), kept per mode in
  `flock.mode_state["field"]["t"]` (add `mode_state: dict` to `PhysicsFlock`
  if absent).
- `seeds = flock.seeds` — the existing per-bird `(N,) float32 ∈ [0,1)` array
  (currently unused by field mode; nearly every formula below consumes it).
- `hash01(x) = fract(sin(x·12.9898)·43758.5453)` — deterministic per-seed
  hash used where the source calls `hash(seed + k)`.

---

## R0 — Prerequisite: config fields

New `SimConfig` fields (defaults = source defaults). Apply the YAML
section-prefix loader fix first if still pending (flatten `field: x:` →
`field_x`, warn on unknown keys, stop `capture: width:` clobbering the
domain — see `SimConfig.from_file`).

```python
# ── Field mode (crs48) ────────────────────────────────────────
field_unit_scale: float = 0.0          # 0 = auto 0.4·min(W,H,D)
field_chase_strength: float = 0.72     # EXISTS, dead — R4 wires it
field_shell_influence: float = 1.0
field_target_pull: float = 0.35
field_drift_pull: float = 0.55
field_tangent_pull: float = 1.0
field_flow_pull: float = 1.0
field_wave_gain: float = 0.5           # ripple/wave coupling
field_inertia: float = 0.8             # R8: 0 = crisp clamp, 1 = pure direction
# ── Threat (crs48 predator) ───────────────────────────────────
threat_mode: str = "off"               # off | cursor | orbit | autonomous
threat_radius: float = 0.5             # source units
threat_strength: float = 1.0
threat_momentum: float = 0.5
threat_acceleration: float = 1.0
threat_split_gain: float = 0.8
threat_vacuole_strength: float = 0.4
threat_blackening_gain: float = 0.6
# ── Wander (R2) ───────────────────────────────────────────────
wander_speed: float = 1.0
wander_radius: float = 0.6             # fraction of U
```

---

## R1 — Smoothed swarm centre (shared foundation)

**Idea.** One exponentially smoothed centroid feeds every consumer (blob
anchors, wander, predator target, ripple origins) instead of each computing
a raw jittery per-frame CoM.

**Math.** `centroid = (1/N)·Σ p_i`;
`center ← center + β·(centroid − center)`, β ∈ [0.4, 0.7] (source blends per
frame at 60 fps).

**Implementation.** `PhysicsFlock.step` computes it once:
`self.center = self.center + 0.5*(centroid - self.center)` (init to first
centroid). All R2–R9 systems read `flock.center`.

**Accept:** `flock.center` lags a teleported flock by a few frames (smooth),
and all consumers below reference it (grep test: no other `positions.mean`
in field/threat/wander paths).

---

## R2 — Flock wander: bounded unit travel (replaces the Wander extension)

**Idea (verbal).** The flock's *attractor* drifts along a multiscale
sinusoidal path that is **provably confined to the unit sphere**: compose
three frequency bands per axis (with nested phase modulation so the path
never visibly loops), then scale by a radial pulse divided by
`max(1, ‖raw‖)` — the output radius is always ≤ 1. The flock's headline
direction is the forward difference of the path.

**Math (verified verbatim from `flockWander.ts`).**

```
raw_x = sin(t·0.47 + sin(t·0.19)·1.15)·0.82 + sin(t·1.07+1.4)·0.38 + cos(t·0.23+2.1)·0.22
raw_y = cos(t·0.43+0.6 + sin(t·0.13)·0.9)·0.78 + sin(t·0.91+2.8)·0.42 + cos(t·0.29+0.4)·0.24
raw_z = sin(t·0.39+1.1 + cos(t·0.17)·1.05)·0.80 + cos(t·0.97+0.2)·0.40 + sin(t·0.21+2.6)·0.22
pulse = 0.72 + 0.28·(0.5 + 0.5·sin(t·0.41 + cos(t·0.17)))
path(t) = raw · pulse / max(1, ‖raw‖)                       (‖path‖ ≤ 1 guaranteed)

wander_center(t) = C + path(t · wander_speed) · wander_radius · U
heading(t)       = normalize(wander_center(t + 0.75) − wander_center(t))
```

**Implementation.** Rewrite
[pymurmur/physics/extensions/wander.py](pymurmur/physics/extensions/wander.py)
with the exact formulas (fixing its current bug: it orbits the *origin* —
the domain corner). The extension stores `t`, exposes
`wander_center()`/`heading()`; its `apply()` pulls birds gently toward the
centre as today but from the new path. Field mode (R3) uses
`wander_center` as its `C` when wander is enabled, else `flock.center`.

**Accept:** `‖path(t)‖ ≤ 1` for 10⁶ random t (property test); the attractor
stays inside the domain; heading is a unit vector continuous in t.

---

## R3 — Field mode I: five blob anchors + cyclic phase weights

**Idea (verbal).** Above ~1 200 birds, neighbour queries are replaced by a
**field**: five anchor points orbit the flock centre on independent
Lissajous paths, and each bird follows a *phase-weighted blend* of them. A
bird's phase drifts with time and its seed; the cyclic weight function makes
it "belong" to one or two adjacent blobs at a time, smoothly handing over as
phase drifts — the source of the lava-lamp lobe dynamics. All O(N).

**Math.** Anchors (source units → ×U, around centre C):

```
B₀ = C + ( sin(t·0.19)·0.74,      sin(t·0.31+0.8)·0.48,  cos(t·0.23)·0.62 )·U
B₁ = C + ( cos(t·0.17+1.6)·0.68,  sin(t·0.37+2.1)·0.54,  sin(t·0.29+0.4)·0.72 )·U
B₂ = C + ( sin(t·0.27+2.7)·0.58,  cos(t·0.21+1.2)·0.42,  cos(t·0.33+2.5)·0.68 )·U
B₃ = C + ( cos(t·0.24+3.4)·0.70,  sin(t·0.33+0.6)·0.50,  sin(t·0.18+1.4)·0.58 )·U
B₄ = C + ( sin(t·0.14+4.4)·0.48,  sin(t·0.47+2.3)·0.62,  cos(t·0.26+4.0)·0.70 )·U

φ_i = fract( seed_i·3.71 + t·0.022 + sin(seed_i·19 + t·0.11)·0.09 )
w_k(φ) = max(0, 1 − min(|φ−c_k|, 1−|φ−c_k|)·7.5)²        c_k ∈ {0, .2, .4, .6, .8}
T_legacy,i = Σ_k B_k·w_k(φ_i) / Σ_k w_k(φ_i)              (Σw > 0 always: 7.5·0.1 < 1)
```

**Implementation.** In `field.py`: `_blob_anchors(t, C, U) -> (5,3)`;
phases and the five weights fully vectorised
(`w = np.maximum(0, 1 - wrapdist*7.5)**2`, shape `(n,5)`;
`T = (w @ B) / w.sum(1, keepdims=True)`). `T_legacy` replaces the current
single-CoM target.

**Accept:** with uniform seeds, birds partition into ~5 visible lobes that
exchange members over ~45 s (`0.022·t` full phase cycle); variance of
per-bird target positions ≫ 0 (no longer one shared target).

---

## R4 — Field mode II: leader/chaser groups and the chase blend

**Idea (verbal).** A second target system: birds belong to one of 7 groups,
each chasing a group-specific **leader anchor** that orbits the centre; each
bird trails it by a seed-dependent time lag. ~16 % are "leaders" with high
lag (they pull ahead); followers trail. Within a group, birds hold
**golden-angle stratified shell offsets** so the group is a layered blob,
not a point. The final target lerps between the R3 blob field and this
chase system by `chaseStrength` — the single most important character knob.

**Math.**

```
group_seed_i = floor(seed_i·7)/7;   phase = group_seed·2π
anchor(t, gs) = C + ( cos(phase + t·0.21)·0.50 + sin(t·0.13 + phase·2.3)·0.16,
                      sin(phase·1.7 + t·0.19)·0.34 + cos(t·0.11 + phase)·0.12,
                      sin(phase + t·0.16)·0.46 + cos(t·0.23 + phase·1.4)·0.14 )·U

lag_i    = hash01(seed_i + 9.17) · (1.1 + chaseStrength·2.4)
leader_i = hash01(seed_i + 5.91) ≥ 0.84                       (~16 %)
primary  = anchor(t − lag_i, group_seed_i)
secondary= anchor(t − lag_i, fract(group_seed_i + 1/7))
sec_mix  = hash01(seed_i + 3.33)·0.5

Stratified offset (slot = per-bird index within its group):
  ga = 2.39996323
  y     = 1 − 2·fract((slot+0.5)·0.618034 + group_seed·0.13)
  ring  = sqrt(max(0, 1 − y²))
  θ     = slot·ga + group_seed·2π
  shell = fract((slot+1)·0.754877)^(1/3)
  radius= (0.16 + shell·0.34)·(0.68 + chaseStrength·0.34)·(0.92 + separation·0.045)·U
  breath= 1 + sin(t·0.13 + group_seed·12)·0.035
  offset= (cosθ·ring, y, sinθ·ring)·radius·breath

follower_target = lerp(primary, secondary, sec_mix) + offset
leader_target   = C + wander_heading(t)·(0.18 + hash01(seed_i+7.1)·0.18)·U
chase_target    = where(leader_i, leader_target, follower_target)

T_i = lerp(T_legacy,i , chase_target_i , chaseStrength)
```

**Implementation.** All hash/lag/role/offset quantities depend only on
`seeds` — precompute once per flock (cache in `mode_state["field"]`),
recompute only radii terms that depend on t/chaseStrength per frame. `slot`
= rank of the bird within its group by seed order (stable; compute once).

**Accept:** at `chaseStrength=0.8`, 7 staggered lobes orbit with visible
leaders ahead of trailing streams; at `chaseStrength=0`, behaviour is
identical to R3 alone; group membership is stable across frames.

---

## R5 — Field mode III: shell force and inner cavity

**Idea (verbal).** Birds are pulled to a **breathing shell radius** around
their target — not the target point — so blobs stay hollow and layered:
outside the shell pulls in, inside pushes out, and a second, harder inner
radius floor prevents core collapse.

**Math.**

```
Δ = p_i − T_i;  d = ‖Δ‖;  d̂ = Δ/d
R_blob,i = (0.24 + (0.5+0.5·sin(seed_i·41 + t·0.29))·0.16 + sin(φ_i·2π + t·0.17)·0.05)·U
F_shell = −d̂ · (d − R_blob) · field_cohesion · 1.35 · (1 − chaseStrength) · shell_influence

inner_i = R_blob,i · (0.28 + (1−chaseStrength)·0.18 + field_separation·0.012)
if d < inner_i:  F_expand = d̂ · (inner_i − d) · field_separation · 1.4
```

**Implementation.** Replaces the current unconditional CoM pull
([field.py:31-37](pymurmur/physics/forces/field.py#L31-L37)). Both terms
vectorised; guard `d > 1e-6`.

**Accept:** a settled 5 000-bird blob shows a hollow core (voxel density at
the lobe centre < 30 % of the shell density); the shell radius visibly
breathes with ~20 s and ~35 s periods.

---

## R6 — Field mode IV: the remaining force terms + full composition

**Idea.** Six more cheap per-bird terms complete the source's field
composition; each is one vectorised expression.

**Math (per bird; all forces added to acceleration).**

```
Slot repulsion (fix the existing one): offsets o ∈ {±1, ±7, ±31}, wrapped:
  other = positions[(i+o) mod n_active]        (active-compacted order)
  away = p_i − other;  d = ‖away‖;  r_slot = (0.07 + field_separation·0.02)·U
  if 1e-4 < d < r_slot:  F += (away/d)·((r_slot−d)/r_slot)² · field_separation·(0.14 + chaseStrength·0.05)

Tangential orbital:
  axis_i = normalize( sin(t·0.13+seed_i·7), 0.72 + sin(t·0.19+seed_i·3)·0.28, cos(t·0.17+seed_i·5) )
  F_tan  = normalize(axis_i × (p_i − T_i)) · field_alignment · 0.035 · (1−chaseStrength) · tangent_pull

Buoyancy (z-up in pymurmur — apply to z, not y):
  F_z += ( sin(d·8/U − t·1.1 + seed_i·17)·0.09 + (T_z − p_z)/U·0.24 ) · (0.75 + field_flow·0.25)

Curl-style flow (replaces the current plain sines; q = (p − C)/U):
  flow = ( sin(q_y·2.8 + t·0.24 + seed_i) + cos(q_z·2.1 − t·0.17),
           sin(q_z·2.3 + t·0.20) − cos(q_x·1.9 + t·0.24),
           sin(q_x·2.6 − t·0.16) + cos(q_y·2.2 + t·0.24) )
  F_flow = normalize(flow) · field_flow · 0.08 · flow_pull

Fold noise (second, finer band):
  fold = ( sin(q_y·3.7 + t·0.73 + seed_i) + cos(q_z·2.9 − t·0.51),
           sin(q_z·3.1 − t·0.67 + seed_i) − cos(q_x·2.4 + t·0.43),
           sin(q_x·3.3 + t·0.59 + seed_i) + cos(q_y·2.6 − t·0.47) )
  F_fold = fold · field_flow · flow_pull · ripple_envelope_sum   (couples to R7)

Viscous drag:  F_drag = −v_i · chaseStrength · (0.08 + field_flow·0.02)

Drift alignment (keep, retargeted):  F = (v_wander − v_i)·field_alignment·drift_pull
   where v_wander = heading(t)·v0 when wander enabled, else flock mean velocity

Target pull (keep):  F = (T_i − p_i)/U · field_cohesion · target_pull
```

Composition = shell + expand + target-pull + drift + drag + tangent + slot +
flow + fold + ripple(R7) + buoyancy + threat(R9) + boundary (existing).

**Accept:** with all terms on, the flock exhibits laminar swirl (nonzero
mean angular momentum about the local blob axes), vertical undulation
(z-variance of lobe centroids > 0), and no term produces NaN over 10⁴ frames
(invariant fuzz).

---

## R7 — Ripple envelopes done right

**Idea (verbal).** Three staggered pulse trains radiate from *moving
origins* near the centre; each pulse has a finite life (smoothstep rise and
fall), an expanding radius, a widening front, a radial push and a **twist**
component, all modulated by flow and wave gain.

**Math (per train, offset o ∈ {0, 9.33, 18.67} s; local_t = (t − o) mod 28).**

```
env(τ)    = smoothstep(0.6, 1.7, τ) · (1 − smoothstep(6.2, 8.8, τ))
radius(τ) = (0.16 + τ·0.16)·U ;  width(τ) = (0.11 + τ·0.012)·U
origin(τ) = C + ( sin(t·0.17+o)·0.46, cos(t·0.13+o·1.7)·0.25, cos(t·0.19+o·0.6)·0.42 )·U
r = ‖p − origin‖;  δ = |r − radius|/width;  amount = exp(−δ²)·env(local_t)
F_radial = (p − origin)/r · amount
F_twist  = wander_heading(t) × F_radial
F_ripple = (F_radial + F_twist·0.28) · field_flow · (0.13 + wave_gain·0.04)
ripple_envelope_sum = Σ_trains amount                       (feeds R6 fold noise)
```

**Implementation.** Rewrite
[pymurmur/physics/extensions/ripple.py](pymurmur/physics/extensions/ripple.py)
vectorised (current version loops per bird, never decays, and uses the CoM
as origin); or move it into field mode and keep the extension as a thin
wrapper for other modes. `smoothstep(a,b,x) = t²(3−2t), t=clamp((x−a)/(b−a))`.

**Accept:** pulses visibly die out (~9 s life), fronts expand and widen, and
a paused-flock density plot shows three interleaved rings; O(N) vectorised
(< 1 ms at 100 k birds).

---

## R8 — Steering refinements: inertia, bounded panic, blackening

**Idea (verbal).** Three globally applicable dynamics fixes from the source:
(1) **inertial speed smoothing** — instead of hard-clamping speed, lerp
between the raw and clamped velocity by `inertia` (0.6–0.9 feels natural;
1.0 = pure direction steering); (2) **bounded panic boost** — threatened
birds may exceed max speed by a *capped* factor (pymurmur's current predator
multiplies velocity ×1.5 *every frame* — unbounded compounding); (3)
**blackening** — near threats, separation weakens and cohesion strengthens,
so the flock *compresses* (darkens) around danger rather than only fleeing.

**Math.**

```
Inertia (in integrate, replacing the hard clamp when inertia > 0):
  v_raw = v + a·dt;  v_clamped = clamp_speed(v_raw)
  v     = lerp(v_raw, v_clamped, 1 − inertia)      # inertia=0 → instant clamp

Panic (threat proximity prox_i ∈ [0,1] from R9):
  panic = clamp(prox_i, 0, 1)·threat_strength
  boost = panic·(0.72 + wave_gain·0.18 + vacuole_strength·0.12)
  local_max_speed_i = v0·(1 + min(1.35, boost))     # ceiling raise, not a multiply

Blackening:
  black = 1 + blackening_gain·prox_i·0.85
  sep_eff_i = separation·(2 − black);   coh_eff_i = cohesion·black
```

**Implementation.** `integrate()` gains `inertia: float = 0.0` and an
optional per-bird `max_speed` array (panic writes it; default scalar `v0`).
The threat system (R9) exports `prox` per bird via
`mode_state["threat"]["prox"]`; field/spatial modes read it to compute
`sep_eff/coh_eff` row-wise. Delete the compounding `velocities *= 1.5` in
the current predator.

**Accept:** with `inertia=0.8` velocity magnitude changes smoothly (no
frame-to-frame steps > 20 % of v0); panicked birds' speed ≤ `2.35·v0`
always; a predator pass shows local density *increase* around the threat
wake (blackening) alongside the hole (split, R9).

---

## R9 — The autonomous predator FSM (threat system)

**Idea (verbal).** A predator with smoothed *intent*: it keeps a persistent
attack direction and preferred turn axis (both exponentially smoothed, axis
sign-aligned to avoid flips), turns at a capped rate, aims **through** the
swarm to a pass-through point (not at the centre), arcs gently on egress,
and cycles approach ↔ egress by distance thresholds. Its influence on birds
is a four-part bundle: radial push, velocity **wake** (birds are dragged
along the predator's motion), an XZ-biased tangential **split** that tears
the hole, and a **wave** amplification along each bird's own heading.

**Math (verified from `threat.ts`; lengths ×U).**

```
capture   = max(0.18, threat_radius·0.72)·U
pass_dist = (0.92 + threat_radius·2.6 + threat_momentum·1.32)·U
clear     = pass_dist·(0.72 + threat_momentum·0.16)
turn_rate = (0.54 + threat_acceleration·0.025)·(1 − threat_momentum·0.24)   rad/s (chase)
          = 0.42·(1 − threat_momentum·0.24)                                  (orbit mode)

approach→egress:  dist_to_center ≤ capture
egress→approach:  dist_to_center > clear  AND  dot(dir, to_center̂) < −0.12

rotate_toward(f̂, t̂, θmax):  φ = acos(clamp(f̂·t̂,−1,1));  if φ ≤ θmax → t̂
   else rotate f̂ about normalize(f̂×t̂) (fallback: any ⊥ axis) by θmax   (Rodrigues)
smooth_axis: desired = normalize(dir × to_center̂); flip if dot(prev, desired) < 0;
   axis ← normalize(lerp(prev, desired, amount))

Arc offset (egress):  broad = threat_radius·(0.36 chase | 0.24 orbit)·U
  arc = turn_axis·sin(t·0.18+0.7)·broad + normalize(turn_axis×dir)·cos(t·0.13+1.4)·broad·0.72
target = approach ? swarm_center : swarm_center + dir·pass_dist + arc
steer response: approach 1.86 + (1−momentum)·0.48 ; egress 0.34 + (1−momentum)·0.44

Force bundle on bird i (away = p_i − p_threat, d = ‖away‖ < threat_radius·U·2):
  prox = 1 − d/(threat_radius·U·2);  broad = sqrt(prox);  â = away/d
  push  = â · threat_strength·(2.5 + vacuole_strength·1.7) · broad
  wake  = (â − dir·0.35) · min(1.8, ‖v_threat‖/v0) · threat_strength · broad · 0.42
  split = (−â_z·1.45, â_y·0.28, â_x·1.45) · split_gain · broad      (XZ-biased tear; z-up:
           swap so the tear is horizontal: (−â_y·1.45, â_x·1.45, â_z·0.28))
  wave  = v̂_i · wave_gain · broad · 0.22
  F_threat,i = push + wake + split + wave;   prox_i exported for R8
```

Threat modes: `off`; `cursor` (threat at the mouse ray's median-depth point
— reuse the unprojection math from the interaction work if present, else
orbit); `orbit`; `autonomous` (full FSM).

**Implementation.** Rewrite
[pymurmur/physics/extensions/predator.py](pymurmur/physics/extensions/predator.py)
as `Threat` with the state `{pos, vel, dir, turn_axis, phase}`; per frame:
compute target → `rotate_toward` with `turn_rate·dt` → move
`pos += dir·speed·dt` (speed = `2·v0·(1+momentum·0.5)`); apply the force
bundle vectorised over birds in range; export `prox`. Render it (a single
extra red instance — see the species/flag column if implemented; else a
second 1-instance draw call).

**Accept:** the predator flies *through* and out (no teleport reset —
pymurmur's current one resets instantly); egress paths curve; a pass leaves
a horizontal-biased hole that heals; birds behind the predator stream along
its wake direction.

---

## R10 — Rendering: impostors, depth cues, trails, themes

**Idea.** Four source rendering systems, all absent:

**(a) Sphere impostors + monochrome shading.** Camera-facing quads shaded as
spheres:

```
p = uv·2−1;  r² = p·p;  if r² > 1 discard
z = sqrt(1−r²);  edge = smoothstep(1.0, 0.72, r²)
shade = 0.55 + 0.45·z;  color = mix(paper, ink, shade·(1 − rim·0.22))
```

**(b) Depth cues** (monochrome legibility): size `∝ 1/depth^k`; alpha ×
`mix(1, 1−depth01, depthFade)` × `mix(0.65, 1, speed01)` ×
`mix(1, 0.76, smoothstep(0.72, 1, r²))`.

**(c) Trails.** Two modes: *velocity* — stretch the impostor along
screen-space motion `project(p) − project(p − v·trailLen·0.12)`, head radius
`max(0.28, 1/(1+stretch·2.8))`, tail `0.22 + stretch·1.35` with wave
`sin(prog·(5.4+speed01·3.4)+seed)·waviness·stretch·0.18`; *accumulation* —
fade the previous frame with a fullscreen quad at
`fadeOpacity = clamp(0.24 − persistence·0.19 − visibility·0.09, 0.018, 0.32)`
(`persistence = clamp(trailLen/5)`, `visibility = clamp(trailOpacity)`),
clear depth only, then draw.

**(d) Themes**: wire `config.theme` into the renderer (palettes exist,
never passed), and route the impostor `paper/ink` pair from the theme.

**Implementation.** `viz/shaders.py` gains the impostor program (billboard
quad expansion in the vertex shader from per-instance pos/vel) and the trail
uniforms; `viz/renderer.py` gains `render_mode: "tetra" | "impostor"`,
the accumulation path (`begin_frame(fade=...)`: skip colour clear, draw fade
quad depth-write-off, re-clear depth), and `theme=config.theme` from the
Visualizer. Makes the dead `point_sprites` and `trails` fields live.

**Accept:** impostor mode at 20 k birds keeps 60 fps with visible
near/far size+alpha gradients; velocity trails stretch with speed;
accumulation trails persist ~1/fadeOpacity frames and clear when paused;
all four themes render with correct paper/ink pairs.

---

## R11 — Performance diagnostics + adaptive quality (wired end to end)

**Idea.** The source's loop: EMA frame stats (spike-capped), a frame budget,
a risk-based bottleneck classifier, and a **degradation ladder** that
actually changes settings when fps stays low.

**Math/logic.**

```
frame_ms = min(250, max(0.01, now − prev));  avg ← avg·0.92 + frame_ms·0.08
budget_ms = 1000 / max(24, target_fps)
healthy if avg ≤ budget·1.12
risks: cpu = mode is field/spatial at high N and python path
       vertex = N > 30 000 ; fragment = trails on or window very large
classification: >1 risk → mixed; else the single risk; else mixed
degrade when fps < 0.78·target for ≥ 1.8 s continuously, one step per 1.8 s:
  1) trails off   2) capture/window scale −0.15 (floor 0.75)   3) N −18 % (floor 512)
```

**Implementation.** Extend
[pymurmur/analysis/perf.py](pymurmur/analysis/perf.py) with the cap, budget,
classifier, and a `QualityGovernor` holding the hysteresis timer and the
ladder; `Visualizer.run` creates `PerfDiagnostics`, feeds it physics/render
timings (the engine already supports `sim.perf`), and applies governor
actions: set `config.trails="off"`, reduce render scale (render to a smaller
FBO, blit up — or reduce `capture` size headless), call
`flock.remove_boids(int(0.18·N))`. Add the source's data-oriented rails
while here: clamp `dt` to `[0, 1/20]` before `sim.step`, and an
`np.isfinite` guard on positions after integrate (reset offending birds to
`flock.center`).

**Accept:** artificially throttling (sleep in the render loop) triggers the
ladder in order with ≥1.8 s spacing; removing the throttle stops
degradation; a NaN injected into one bird's position self-heals within a
frame.

---

## R12 — Initialization, presets, tests

**Idea.** (a) The source starts birds around **five fixed blob centres** on
∛-uniform shells with drift-biased tangential velocities — visual interest
from frame 0; (b) seven desktop presets with full parameter vectors define
the product's character range.

**Math (init; centres in source units → `C + c·U`).**

```
centres = (−0.48,0.18,0.12) (0.36,−0.20,−0.28) (0.12,0.34,0.42)
          (−0.16,−0.30,0.34) (0.48,0.16,0.18);   centre_i = centres[i mod 5]
θ = U(0,2π);  y = U(−1,1);  ring = sqrt(1−y²)
r = cbrt(U(0,1))·(0.22 + U(0,1)·0.28)·U;  jitter = U(−1,1)·0.045·U per axis
p = C + centre·U + (cosθ·ring, y, sinθ·ring)·r + jitter
v = ( (0.34 + U(−1,1)·0.08), U(−1,1)·0.16, (0.08 + U(−1,1)·0.08) )·v0·0.5 + jitter(0.05·v0)
```

**Presets** (`conf/field_*.yaml`; count, speed, sep, align, coh, chase,
inertia, noise, flow, trail, threat):

```
quiet_roost     3000  0.48 0.85 0.65 1.85 0.72 0.82 0.03  0.18 velocity off
lava_lamp      16000  defaults (pure R3 blob dynamics, chase 0)
ink_cloud      18000  0.62 0.92 0.90 1.80 0.82 0.84 0.035 0.30 accumulation autonomous
predator_ripple 12000 0.78 1.05 1.05 1.15 0.64 0.70 0.08  0.48 velocity orbit
vacuole        10000  0.68 1.12 0.92 1.25 0.76 —    —     0.42 accumulation autonomous (vacuole_strength 0.9)
silk_sheet     14000  0.46 0.92 1.10 1.10 0.68 0.88 0.025 0.24 velocity off
storm_turn     16000  0.90 1.10 1.15 1.25 0.42 0.58 0.10  0.72 velocity autonomous
```

(speed column scales `v0`; sep/align/coh map to `field_*` weights.)

**Tests** (`test/physics/test_field_port.py`, `test/viz/`): acceptance
assertions from R1–R11; determinism with seeds; invariant fuzz (no NaN,
bounded speeds with panic cap); golden trajectory pinned for the new field
mode (R3–R6 change dynamics deliberately — pin once complete).

---

## Out of scope (recorded, not roadmapped)

- **WebGL2-GPGPU / WebGPU tiers** — pymurmur is desktop moderngl; the
  ping-pong texture pattern (`S = ⌈√N⌉` packing) is the blueprint if a GPU
  compute tier is ever attempted (moderngl compute shaders, GL 4.3).
- **XR/VR (SwarmPilot, medium modes, haptics)** — excluded by
  `functional_decomposition.md`. Portable pieces already covered elsewhere:
  the pilot-aware shell forces (a desktop "pilotable flock" mode) and the
  data-oriented rails (dt clamp, NaN guard — included in R11).

---

## Roadmap summary (dependency order)

| # | Item | Depends on | Size | Files touched |
|---|------|-----------|------|---------------|
| R0 | Config fields (+ loader fix if pending) | — | ½ day | `core/config.py` |
| R1 | Smoothed swarm centre | R0 | ¼ day | `physics/flock.py` |
| R2 | Bounded-unit-travel wander | R1 | ½ day | `physics/extensions/wander.py` |
| R3 | Blob anchors + phase weights | R1 | ½ day | `forces/field.py` |
| R4 | Leader/chaser + chase blend | R2, R3 | 1 day | `forces/field.py` |
| R5 | Shell force + inner cavity | R3 | ½ day | `forces/field.py` |
| R6 | Remaining field terms + composition | R5 | 1 day | `forces/field.py` |
| R7 | Ripple envelopes | R2 | ½ day | `physics/extensions/ripple.py`, `forces/field.py` |
| R8 | Inertia / panic / blackening | R0 | ½ day | `physics/boid.py`, `forces/*.py` |
| R9 | Threat FSM + force bundle | R1, R8 | 1½ days | `physics/extensions/predator.py`, `core/types.py` |
| R10 | Impostors, depth cues, trails, themes | — | 2 days | `viz/shaders.py`, `viz/renderer.py`, `viz/visualizer.py` |
| R11 | Perf + adaptive quality + rails | — | 1 day | `analysis/perf.py`, `viz/visualizer.py` |
| R12 | Init + presets + tests + golden | R3–R9 | 1 day | `physics/flock.py`, `conf/`, `test/` |

Total ≈ **10–11 working days**. Three independent tracks can run in
parallel: physics (R1–R9), rendering (R10), performance (R11).
