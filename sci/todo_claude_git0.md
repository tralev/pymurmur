# todo_claude_git0.md — Master consolidation: everything in `sci/*.md` not yet implemented

**Sources:** all eleven documents in `/Users/tralev/Developer/git_mur/sci/`
(`new1_sci.md` … `new11_sci.md`), each derived from an external reference
implementation and cross-verified against it:

| Doc | Reference | Paradigm |
|-----|-----------|----------|
| new1, new2 | crs48/murmuration | field/blob "lava-lamp", threat FSM, trails, adaptive quality |
| new3 | djin31/Starlings | Reynolds + physical energy metrics, sliders, winged birds |
| new4 | Nikorasu/PyNBoids | angle-based steering, adaptive speed, pixel trails |
| new5, new9 | rystrauss/boids | k-d tree, toroidal distance, predator boids, OpenMP, CLI |
| new6 | JerBoon/murmuratR | cosmic-influencer target attraction |
| new7 | TheAmirHK/BirdMurmuration | MARL/PPO two-layer control |
| new8 | corentinpradier/collective-motion | Vicsek predator–prey |
| new10 | Reynolds EvoFlock (arXiv 2026) | evolutionary inverse design |
| new11 | `/Users/tralev/Developer/murmuration` | the predecessor codebase (features lost in the port) |

**Target:** the `pymurmur/` codebase. Everything below is **absent or
implemented with different math**, stated with the defining formulas
(3D form, matched to pymurmur's z-up `[0,W)×[0,H)×[0,D)` domain and moderngl
renderer), target files, and acceptance criteria. Expanded per-source specs
with full code sketches exist as `todo_claude_git1.md` … `todo_claude_git7.md`
(cited per item as *[→gitN]*); this file is the complete inventory and the
one roadmap.

**Global conventions.** `C = (W/2, H/2, D/2)`; `U` = per-source unit scale
(defined per workstream); `idx = np.where(flock.active)[0]`; all arrays
`(N,3) float32`; `hash01(x) = fract(sin(x·12.9898)·43758.5453)`;
`smoothstep(a,b,x) = t²(3−2t), t = clamp((x−a)/(b−a), 0, 1)`.

---

# Part A — Shared infrastructure (implement once; nearly everything depends on it)

### F0. Config loader fix + field inventory  **[blocker]**

`SimConfig.from_file` ([core/config.py:149-164](pymurmur/core/config.py#L149-L164))
flattens `section: key:` to bare `key`: prefixed fields (`vicsek_couplage`,
`influencer_substeps`, …) are **silently dropped**, and colliding keys are
worse — `capture: width:` overwrites the **domain width** in every shipped
preset. Fix:

```python
valid = {f.name for f in fields(cls)}
flat = {}
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

Add a `to_file→from_file` round-trip equality test and a per-preset
domain-survival test. Every workstream's new fields (listed inline below)
ride on this. *[→git1 R0, git2 R0]*

### F1. Seeded flock RNG  **[blocker for reproducibility]**

`PhysicsFlock` owns `self.rng = np.random.default_rng(config.seed)`; every
stochastic site (vicsek noise, influencer, predator, `add_boids`,
`noise_force`, jitter, init) draws from it. Today `config.seed` affects only
frame 0 — vicsek/influencer/predator use module `np.random.*`. Acceptance:
same seed → bit-identical positions after 100 steps, per mode.

### F2. Per-mode state (`mode_state`) 

`PhysicsFlock.mode_state: dict[str, dict] = {}` — force modes are currently
stateless `(flock, config)` functions with nowhere to keep time. Needed by:
influencer tick counter, field-mode `t`, threat `prox` export, wander cache.
*[→git1 R1]*

### F3. `integrate()` variants

Add parameters to [physics/boid.py::integrate](pymurmur/physics/boid.py#L19):

```
speed_mode: "band" (legacy [0.3v0, v0]) | "fixed" (|v| = v0 exact)
          | "ceiling" (limit only)      | "none" (no clamp)
move: bool = True            (False → boundary enforcement only; mode owns positions)
inertia: float = 0.0         (v = lerp(v_raw, v_clamped, 1−inertia))
max_speed_per_bird: ndarray | None    (panic boost ceiling, F7/W2)
```

Also: clamp visualizer `dt` to `[0, 1/20]` and add an `np.isfinite` position
guard (reset offenders to `flock.center`) — the data-oriented rails every
source insists on.

### F4. Minimum-image toroidal distance

Under the default toroidal boundary, no query is wrap-aware — flocks tear at
the seam. `cKDTree(pos, boxsize=(W,H,D))` for queries;
`min_image(Δ, box) = Δ − box·round(Δ/box)` in `core/types.py` for pair
vectors (forces, collisions, MSD). Corrected form (the rystrauss source has
a sign bug — take `|Δ|` before testing): `d_axis = min(|Δ|, L−|Δ|)`.
Activates the dead `use_toroidal_distance` field. Hash-grid alternative:
modulo-wrapped cell keys (the predecessor did this; the port dropped it).
*[→git3 R2]*

### F5. Renderer instance-attribute channel + VAO discipline

One extension serves many features: grow the instance layout
`'3f 3f/i'` → `'3f 3f 1f 1f/i'` (`in_bird_flag`, `in_bird_hue`), packed from
`flock.is_predator` and `flock.seeds`. **Rebuild the VAO wherever the
instance buffer is (re)created** — the growth path at
[viz/renderer.py:104-109](pymurmur/viz/renderer.py#L104-L109) currently
reallocates without rebinding (stale-buffer bug; the predecessor rebuilt).
Also from new11: give the headless FBO a **depth attachment**
(`depth_attachment=ctx.depth_renderbuffer((w,h))` — captures currently
render in draw order), use the predecessor's `_mat4_bytes` (`np.array(
m.to_list(), np.float32).tobytes()`) for matrix uploads (macOS PyGLM layout
hazard), and pass `theme=config.theme` into the renderer (palettes exist,
never wired).

### F6. Species column

`flock.is_predator: (N,) bool`, carried through `_extend/add_boids/
remove_boids`; per-species masks and per-species speed arrays. Needed by
Vicsek predator–prey (W5), rystrauss predator boids (W3), rendering (W10).
*[→git3 R3, git2 R2]*

### F7. Per-bird external control hook

`SimulationEngine.step(dt, control: ndarray | None)` — apply
`v += control·scale`, clip. No API today lets caller code inject per-bird
control. Serves MARL (W7), pilot mode, choreography. *[→git7 R1]*

### F8. Smoothed swarm centre

`flock.center ← center + 0.5·(centroid − center)` computed once per step;
consumed by wander, blob anchors, threat targeting, ripple origins (each
currently recomputes a raw jittery CoM). *[→git6 R1]*

---

# Part B — Workstreams (the inventory)

## W1 — Pearce/predecessor fidelity (new11 + core model)

1. **Occlusion actually occludes.** The visibility test is missing from
   [physics/occlusion.py](pymurmur/physics/occlusion.py): drop neighbour j if
   `d̂_j·d̂_k ≥ cos α_k` for any nearer accepted k (closest-first sweep;
   cap candidates at 64). Exact cap half-angle `α = asin(min(b_eff/d, 1))`.
2. **Θ as probabilistic union**, not linear sum:
   `Ω_j = 2π(1−cos α_j)`, `Θ = 1 − Π_visible (1 − Ω_j/4π)`.
3. **δ̂ boundary-length weighted**: `δ̂ = Σ sin α_j·d̂_j / Σ sin α_j`
   (no magnitude clamp; `|δ̂|` *is* the density signal: →1 at the edge,
   →0 surrounded).
4. **Pearce noise term**: `v ∝ φp·δ̂ + φa·⟨v̂⟩_σ + φn·η̂`, `φn = 1−φp−φa`;
   enforce the constraint in input handling (raising φa reduces φp).
5. **Steric clamp**: `steric_force` is unbounded 1/d² — cap at `max_force`.
6. **Occlusion candidate cutoff**: filter σ-neighbour candidates by a max
   visibility range (predecessor: 200) before ranking.
7. **Ecology completion**: logistic dusk
   `1/(1+e^{−z})`, `z = (hour−sunset)/(width/4)` (overflow-clamped);
   cold-weather roost boost `1 + 0.2·(T_mean−T(day))/T_amp` (temperature()
   exists, unused); critical-mass smoothstep over `[0.4, 1.2]·N_crit` as a
   reusable `gated_weight(w, N)`; seasonal size factor (cosine, peak day 15,
   trough 0.25) and `is_murmuration_season(day)`; read the dead
   `ecology_roost`/`ecology_critical_mass` fields.
8. **Sphere boundary centring**: `_sphere_soft` measures from the **origin**
   — centre on `C`; add the asymptotic wall `Δv = −μ·r̂/(R−r)` variant.

## W2 — Field/blob mode + threat system (new1, new2)  *[→git6, expanded]*

Unit scale `U = 0.4·min(W,H,D)`. The current field mode implements ~4 of 13
terms; the full system:

1. **Five blob anchors** `B₀…B₄` on independent Lissajous orbits around
   `flock.center` (coefficients in git6 R3), **cyclic phase weights**
   `φᵢ = fract(seedᵢ·3.71 + t·0.022 + sin(seedᵢ·19+t·0.11)·0.09)`,
   `w_k = max(0, 1 − wrapdist(φ, c_k)·7.5)²`, `T_legacy = Σ B_k w_k / Σ w_k`.
2. **Leader/chaser**: 7 seed groups, lagged leader anchors
   (`lag = hash01(seed+9.17)·(1.1+chase·2.4)`), ~16 % leaders
   (`hash01(seed+5.91) ≥ 0.84`), **golden-angle stratified shells**
   (`ga = 2.39996323`, `shell = fract((slot+1)·0.754877)^{1/3}`, breathing
   ×`(1+sin(t·0.13+gs·12)·0.035)`), final
   `T = lerp(T_legacy, chase_target, chaseStrength)` — wires the dead
   `field_chase_strength`.
3. **Shell force + inner cavity**:
   `R_blob = (0.24 + (0.5+0.5·sin(seed·41+t·0.29))·0.16 + sin(φ·2π+t·0.17)·0.05)·U`;
   `F = −d̂(d−R_blob)·coh·1.35·(1−chase)`; inner floor
   `R_blob·(0.28+(1−chase)·0.18+sep·0.012)` pushing out ×`sep·1.4`.
4. **Slot repulsion fixed**: offsets **±{1,7,31}** with modulo wrap, bounded
   kernel `((r_slot−d)/r_slot)²` inside `r_slot = (0.07+sep·0.02)·U`, gain
   `sep·(0.14+chase·0.05)` (current code: forward-only, unbounded 1/d²).
5. **Tangential orbital** `normalize(axis×(p−T))·align·0.035·(1−chase)`
   with drifting seed axis; **buoyancy** (z-up)
   `F_z += (sin(d·8/U − t·1.1 + seed·17)·0.09 + (T_z−p_z)/U·0.24)·(0.75+flow·0.25)`;
   **curl flow** (normalized sin+cos pairs, gain 0.08) and the finer
   **fold noise** band (spatial 2.4–3.7, temporal 0.43–0.73, coupled to
   ripple activity); **viscous drag** `−v·chase·(0.08+flow·0.02)`.
6. **Ripple envelopes done right**: three trains, offsets {0, 9.33, 18.67};
   `env = smoothstep(0.6,1.7,τ)·(1−smoothstep(6.2,8.8,τ))`;
   `radius = (0.16+τ·0.16)·U`, `width = (0.11+τ·0.012)·U`; **moving
   origins** (Lissajous about C); twist `+ (heading×F_radial)·0.28`; gain
   `flow·(0.13+waveGain·0.04)`. Current extension never decays, uses the
   CoM as origin, loops per bird.
7. **Flock wander** (verified `boundedUnitTravel`): three-band nested
   sinusoids per axis, radial pulse `0.72+0.28·(…)`, scale
   `pulse/max(1,‖raw‖)` ⇒ `‖path‖ ≤ 1` guaranteed;
   `heading(t) = normalize(path(t+0.75)−path(t))`. Replaces the current
   Wander (wrong formula, orbits the domain **corner**).
8. **Steering refinements**: inertia lerp (F3); **bounded panic**
   `v0·(1+min(1.35, panic·(0.72+wave·0.18+vacuole·0.12)))` — replaces the
   compounding `velocities *= 1.5` bug; **blackening**
   `sep_eff = sep·(2−black)`, `coh_eff = coh·black`,
   `black = 1+gain·prox·0.85`.
9. **Threat FSM** (verified): smoothed `dir`/`turn_axis`
   (sign-aligned EMA), Rodrigues `rotate_toward` capped at
   `turn_rate·dt` with `turn_rate = (0.54+accel·0.025)·(1−mom·0.24)`
   (orbit: 0.42·…); `capture = max(0.18, R·0.72)·U`,
   `pass = (0.92+R·2.6+mom·1.32)·U`, `clear = pass·(0.72+mom·0.16)` with
   the `dot < −0.12` heading check; arc offset
   (`broad = R·(0.36|0.24)·U`, lift `sin(t·0.18+0.7)`, drift
   `cos(t·0.13+1.4)·0.72`); force bundle with `broad = √prox`:
   push `â·strength·(2.5+vacuole·1.7)·broad`, wake
   `(â−dir·0.35)·min(1.8,‖v_t‖/v0)·strength·broad·0.42`, split
   `(−â_y, â_x, â_z·0.28/1.45)·splitGain·broad·1.45` (horizontal tear,
   z-up), wave `v̂·waveGain·broad·0.22`. Modes off/cursor/orbit/autonomous.
   Replaces the teleport-reset predator; exports `prox` for item 8; **render
   the threat** (F5 flag channel — it is currently invisible).
10. **Blob-cluster init**: 5 fixed centres, ∛-uniform shells
    (`r = cbrt(u)·(0.22+u'·0.28)·U`, jitter 0.045·U), drift-biased
    tangential velocities. **Seven presets** with full parameter vectors
    (quiet_roost … storm_turn — table in git6 R12).

## W3 — Reynolds variants: Starlings + rystrauss (new3, new5, new9)  *[→git5, git3]*

1. **Force-kernel corrections** in `forces/_base.py`: separation must be
   `Σ r̂/d²` (unit vector — current code yields 1/d falloff); cohesion
   `normalize(p̄−p_i)` (currently unbounded); `noise_force` must multiply by
   its scale (currently normalised away — δ does nothing).
2. **Hybrid metric+topological filter**: neighbour iff `d < R` **and**
   among first 7 accepted; **dual radii** — alignment within `0.75·R` only
   (wires dead `alignment_radius_ratio`); **`separation_distance`** (20) as
   a metric gate (new field); `visual_range` as a true perception radius.
3. **Update-order fidelity (rystrauss)**: predator boost(×1.4) →
   `acceleration_scale`(0.3, dead field) → limit(max_force) → `v += a` →
   **velocity noise** `(U³−0.5)·noise_scale` → limit(max_speed,
   ceiling-only) → move → reset → wrap. `speed_mode="ceiling"` +
   `noise_mode="velocity"` config switches.
4. **Fixed-speed mode (Starlings)**: `v = v̂·v0` exactly (F3 "fixed").
5. **Predator boids** (species, F6): boosts 1.8/1.5/1.4; escape
   `normalize(p_prey−p_pred)·10⁷` **replacing** separation; **hard-zero**
   alignment+cohesion when any predator is perceived; predators flock among
   themselves.
6. **Physical metrics (Starlings)**: `k_v = 8.94/v0`, `k_a = 40/max_force`,
   `m = 0.075 kg`; `F_avg = m·k_a·⟨|a|⟩` (N), `P_avg = m·⟨|k_a a·k_v v|⟩`
   (W), `E = Σ P·Δt` (J), `L = m·(r−CoM)×(k_v v)`; stash accelerations
   pre-integrate; report `speed_real` m/s. 
7. **Per-frame parameter jitter**: `sep += U(0,0.5)`, `coh += U(0,0.1)`,
   `align += U(0,0.005)` on the *effective* weights, from `flock.rng`.
8. **Two-phase parallel update**: batched
   `tree.query(pos, k, workers=num_threads)` + fully vectorised force pass
   (`positions[neighbor_idx]` gather, axis-1 reductions — removes the
   per-bird Python loops); `num_threads: int = -1` config.
9. **Interaction/tooling**: mouse spawn via cursor-ray unprojection to the
   flock's median-depth plane (math in git3 R5), right-click predator, `C`
   clear; CLI `--set KEY=VALUE` (+ `--fullscreen`, `--light-scheme`);
   `pymurmur.Simulation(**params)` facade +
   `benchmark(flock_size, num_steps) → per-step seconds`;
   **slider HUD** (5 sliders: sep 1–5/3.0, coh 0–2/0.2, align 0–0.5/0.02,
   avoid 0–1/0.05, noise 0–0.5/0.05; moderngl ortho quads + hit-test math in
   git5 R5); `PageUp/Dn` flight-speed ±0.1 (floor 0.3); metrics readout
   every 20th frame.
10. **Velocity-init variants**: `(U³−0.5)·2·v0` (E|v|≈0.816·v0, rystrauss);
    uniform-direction × `U(1, v0)` speed (predecessor); tangential-orbital
    (crs48). Config `velocity_init` enum.
11. **Winged flapping mesh (Starlings)**: 6-triangle body+wings+tail
    (vertex table in git5 R6), per-vertex flap weight, uniform
    `u_Flap = ±0.5` toggled every `⌊frame/100⌋`, applied along local up
    *before* the LookAt rotation; **gradient sky** background
    (top (0.60,1,1) → bottom (0.686,0.933,0.933)).

## W4 — Angle-based steering mode (new4)  *[→git4, expanded]*

New `"angle"` force mode — a third paradigm (bounded rotation, not forces):

1. **Steering core**: dead zone (no turn if error < 0.5–0.8°); 3D
   axis-angle: `φ = acos(ĥ·t̂)`, axis `= normalize(ĥ×t̂)` (⊥ fallback when
   parallel), rotate by `min(φ, turnRate·dt)` via Rodrigues
   (`rotate_about` helper in `core/types.py` — shared with W2 threat).
2. **Unified neighbour modes** (7 closest within `b·12`): nearest < `b·1`
   → steer away from nearest; nearest < `b·5` → toward `normalize(ĉ + m̂)`
   (centroid + mean heading); else → centroid only.
3. **Adaptive speed**: `s = base + (7−m)·5` (linear) / `+(7−m)²`
   (quadratic, cap 49) / softened `+min(49, (7−m)²·0.5)` — first per-bird
   speed (F3 array ceiling).
4. **Edge handling**: cardinal inward-normal target override inside margin
   (6 faces, sequence priority; spherical variant `t̂ = normalize(C−p)`);
   turn-rate scaling `rate += (1−edgeDist/M)·(maxRate−rate)`.
5. **±4° per-frame heading jitter** *before* steering (random axis, seeded).
6. **Incremental spatial grid**: per-bird `last_cell`, re-file only on cell
   crossing (current grid clears + reinserts everything every frame);
   select k-NN on squared distances.
7. **Scale invariance**: all radii as `boid_size` multiples
   (`radii_in_bodies` flag); **per-bird HSV colour** from `seeds`
   (`hue = seed·360`, S=V=0.9) via the F5 hue channel.
8. Optional: dual-flock scene driver, screensaver subprocess launcher
   (macOS `ioreg` idle), desktop-screenshot background mode.

## W5 — Vicsek predator–prey (new8)  *[→git2, expanded]*

1. **Correct update**: the (1−η) branch must be
   `û_noisy = normalize(û_old + √(2DΔt)·n_⊥)` — the **memory term** is
   missing (code blends against a fresh random unit) and the **diffusion
   amplitude is normalised away** (D does nothing); noise in the tangent
   plane of `û_old`; `vicsek_time_step` field. Constant speed must bypass
   the integrator band (F3 "fixed" — defaults currently rescale 1.0→1.2
   every frame).
2. **Species dynamics** (F6): fear
   `= clamp((R_pred−d̄_pred)/R_pred, 0, 1)`;
   `û = normalize((1−fear)·û_align + fear·û_flee)`,
   `û_flee = normalize(Σ(p_prey−p_k)/|P|)`; neighbour weights ×3
   (`weight_afraid`) while afraid; predator hunting: nearest prey within
   `1.5·R_pred`, update `normalize(û_target + 0.2·η̂)` — **no couplage**;
   random walk fallback; all-predator early-out.
3. **Asymmetric position collisions**: prey–prey/pred–pred split
   `(R_avoid−d)/2` each along `n̂` (min-image); prey–predator: prey absorbs
   the **full** `(R_pred−d)` correction, predator unmoved; run after move,
   before wrap; `np.add.at` accumulation. Wires dead `vicsek_radius_avoid`.
4. **Nematic order parameter** (O(N) Q-tensor):
   `Q = (3/2)(ûᵀû)/N − ½I`, `S = λ_max(Q)` — distinguishes two-lane states
   α cannot; `order: polar|nematic` option in the phase-diagram sweep.
5. **MSD(τ) curve** on **unwrapped** trajectories
   (`p_unwrap += min_image(step)`), log-spaced lags, log-log slope
   ballistic(2)→diffusive(1) crossover diagnostic — replaces the single
   wrap-corrupted first-vs-last scalar.

## W6 — Cosmic influencer (new6)  *[→git1, expanded]*

1. **Persistent tick** (F2) — currently `t ~ U(0,2π)` per substep: the
   target teleports; no trajectory exists.
2. **Exact Lissajous target** (periods 97/217/29/13/41/7, two amplitude
   scales, +40 z-offset), embedded:
   `s = scale·min(W/460, H/460, D/254)`, `T = C + (T_raw−(0,0,40))·s +
   (0,0,40s)`.
3. **Move-then-steer at unit speed** per substep
   (`p += d̂·v0·dt` *then* re-blend `d̂ ← normalize(d̂(1−inf) + t̂·inf)`;
   branch-free zero guards `x += (x==0)`); one tick per substep.
4. **Influence**: rank by distance **to the target** (not CoM) with
   `inf_sorted[i] = (1 − (i/(N−1))·0.8)^1.8` (floor 0.2^1.8 ≈ 0.055;
   current code lacks the ·0.8 compression and ranks by CoM); distance
   alternative `clamp(100·s²/d², 0.2, 0.8)`.
5. **Density-scaled init**: Gaussian `σ = N^{1/3}·0.5·s` + shared random
   offset; **zero initial directions** (first blend points every bird at
   the target, influence-weighted).
6. **Diagnostics**: per-frame `min/max ‖p−T‖` into `FlockMetrics` + title.

## W7 — MARL bridge (new7)  *[→git7, expanded]*

1. Control hook = F7. **Deferred global-rule "marl" mode**: move first, then
   `v += 0.01·(F_sep(r<0.2U) + (v̄−v) + (CoM−p))` — rules prep the *next*
   step; agent action (0.1× scale, component clip ±0.1U) dominates the
   current one.
2. **Gymnasium wrapper** (lazy import): obs `(6N,)` = normalized positions
   `(p−C)/3U` + velocities `v/v_cap`; action `(3N,) ∈ [−1,1]`; episode 500
   steps; seeded reset `p ~ C + U(−1,1)³U`, `v ~ U(−0.1,0.1)³U`.
3. **Reward module** + two new metrics: `velocity_deviation =
   (1/N)Σ‖v̄−v_i‖` (catches speed dispersion α misses) and
   `boundary_overshoot = Σmax(0, ‖p−C‖−3U)`; reward = ±w_a·align_dev −
   w_c·dispersion (faithful sign quirk behind a flag) + optional angular/
   boundary/altitude terms.
4. **Gated scripts**: PPO MlpPolicy 5 000 steps train; 500-step rollout.
5. **Dual-view rendering** (two viewports, elev/azim 15/15 + 45/45) and the
   **matplotlib GPU-free capture fallback** — replaces the Recorder's
   silent `except Exception: pass` frame loss.

## W8 — EvoFlock corrections (new10)

In [analysis/evoflock.py](pymurmur/analysis/evoflock.py):

1. **Uniform crossover** (each gene from a random parent of the tournament's
   best two) — currently mutation-only; **evaluate all 3 tournament members**
   (with a fitness cache; founders are currently never evaluated —
   `fitness=−inf` forever); **delete the worst of the 3** (negative
   selection) instead of elitist island-worst replacement.
2. **Worst-of-4 evaluation**: 4 sims per candidate (fixed per-sim seeds,
   min-reduction), `eval_parallel` finally used.
3. **Separation objective on nearest-neighbour distance** per boid-step
   (currently median 7th-neighbour at intervals — hides collisions); ramps
   2→2.5 up, 4→5 down.
4. **True curvature** `κ = |v×a|/|v|³` per boid-step (arrays exist;
   pre-integrate acceleration stash), score
   `clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)`.
5. **SDF obstacle layer** (`physics/obstacles.py`): sphere `‖p−c‖−r`, box,
   cylinder, union=min, subtract=max(a,−b); zero-crossing detection
   `sign(SDF(p_old)) ≠ sign(SDF(p_new))`; kinematic correction
   `p ← p − SDF(p)·∇SDF/‖∇SDF‖`; collision counting → makes `(f_cf)^500`
   a live objective (currently hardwired 1.0).
6. **Missing genes/behaviours**: `forward_weight` speed-control force
   `w·sign(v_target−|v|)·û`; per-behaviour `max_dist_*` and `angle_*`
   perception cones; `fly_away_max_dist`, `min_time_to_collide`
   (predictive avoidance); fixed k=7 topological neighbours; integer-gene
   handling for σ; promote hardcoded `speed_min_factor` (0.3) to config.
7. **Protocol**: persist best genome + Pareto front to
   `output/evolved.yaml` with per-run seeds; confined
   (enclosure + obstacles) vs open evaluation configs; reproduce the
   headline result (alignment emerges with no alignment objective).

## W9 — Metrics & analysis completions (new11 + cross-source)

1. **Thickness ratio fix**: export `√(λ₃/λ₁) ∈ (0,1]` (code returns
   `√(λ₂/λ₃) ≥ 1` under the documented name).
2. **Shape→m\***: `m* = 9.78 + clamp((aspect−1)/2, 0, 1)·(6.05−9.78)`;
   add `suggested_m` to `FlockMetrics`.
3. **H₂**: disconnected graph → `inf` (not 0.0); marginal efficiency
   `η(m) = H₂(m−1) − H₂(m)` (+∞ at the connectivity transition);
   symmetrization `max(A, Aᵀ)` to match Young.
4. **τρ reference method**: density = `N/ConvexHull(positions).volume`
   (0 if degenerate), ring buffer (sample every 10 frames, 500 slots),
   integrated autocorrelation `τ = interval·(0.5 + Σ_{lag} r(lag))`
   stopping at the first `r ≤ 0` (ConvexHull is a declared dependency used
   nowhere).
5. **Θ′ silhouette**: project ⊥ an observer axis, rasterize `boid_size`
   disks, union coverage — the Pearce distant-observer quantity (keep the
   voxel metric separately).
6. **Robust gyration**: **median** centroid + one-sided top-15 % trim
   (code uses mean + two-sided); **number density**
   `ρ = N_kept/((4/3)πR_g³)`; density-scaling sweep reports the ideal
   exponent `−0.5` alongside fitted β.
7. **Angular momentum**: about the CoM, with mass (W3.6), plus normalized
   `‖⟨r×v⟩‖/(v0·R_g)` for cross-run comparability.
8. Plus: nematic S (W5.4), MSD(τ) (W5.5), physical units (W3.6),
   `velocity_deviation`/`boundary_overshoot` (W7.3), influencer target
   distances (W6.6).

## W10 — Rendering & capture (new1/2/11 + cross-source)

1. **Sphere impostors** (`point_sprites` is a dead field): camera-facing
   quads; fragment `r² = ‖uv·2−1‖²`, discard > 1, `z = √(1−r²)`,
   `shade = 0.55+0.45z`, `color = mix(paper, ink, shade·(1−rim·0.22))`.
2. **Depth cues**: size ∝ `1/depth^k`; alpha × `mix(1, 1−depth01, fade)` ×
   `mix(0.65, 1, speed01)` × `mix(1, 0.76, smoothstep(0.72, 1, r²))`.
3. **Trails** (`trails` is a dead field): *velocity* — impostor stretched
   along `project(p)−project(p−v·len·0.12)` with head/tail/wave math (git6
   R10); *accumulation* — fade quad at
   `clamp(0.24−persist·0.19−vis·0.09, 0.018, 0.32)`, depth-only clear;
   *ring* — K≈12 past positions as shrinking/fading sprites (needs the
   previous-positions buffer, shared with MSD unwrapping and interpolated
   rendering).
4. **Capture pipeline** (predecessor parity): cinematic sweep
   (`azim = 45°+t·180°`, `elev = 25°+sin(2πt)·0.15`,
   `dist = (650+sin(1.5πt)·100)·scale`), 60-frame pre-warm,
   `CAPTURE_W/H/FRAMES/OUT` env overrides, GIF `optimize=True, disposal=2`,
   Recorder using `capture_width/height` (currently ignored) and reusing a
   render-only `Visualizer.headless_frame` (currently `frame()` *steps the
   sim* as a side effect — make render pure).
5. **Alpha-accumulation density mode** (murmuratR): translucent sprites
   (α≈0.2), blend on, depth-write off. **Orthographic top/side camera
   presets** (keys 7/8/9). **Fixed capture framing** option.
6. **Dual-view** (W7.5), **winged mesh + gradient** (W3.11), **per-bird
   hue + predator flag rendering** (F5/F6), **heading-hue debug theme**.
7. **Adaptive quality wired end to end**: EMA with 250 ms spike cap →
   budget `1000/max(24, fps)` → healthy if `≤ budget·1.12` → risk classifier
   (cpu/vertex/fragment/mixed) → ladder (trails off → render scale −0.15
   floor 0.75 → N −18 % floor 512) with 78 %/1.8 s hysteresis; consumed by
   the Visualizer (the current flags go nowhere).
8. **Fixed-timestep accumulator** with optional render interpolation
   (`render(interp = acc/dt_phys)`).

## W11 — UX & presets

1. **Preset keys a–h,w** with the predecessor's exact (φp, φa, σ, mode)
   table (git5-adjacent; table in todo_claude.md Part 3 E4), descriptions
   printed on apply; reconcile `analysis/presets.py` (wrong values, no
   consumer).
2. **Full title readout**: mode, N, φp/φa/σ, `α Θ Θ' L σr`, τρ, FPS —
   refreshed every 20th frame (+ physical units once W3.6 lands).
3. Sliders/speed keys (W3.9), CLI/facade/benchmark (W3.9), mouse spawn
   (W3.9), threat-mode & preset cycling keys.

---

# Part C — Unified roadmap

Phases are independently shippable; `pytest test/` green at each boundary.
Deliberate physics changes end with a golden-trajectory re-pin in the same
commit (pin the golden harness first — P0).

```
P0 foundations ─► P1 correctness ─► P2 physics workstreams (parallel tracks)
                                 ├─► P3 metrics & analysis
                                 ├─► P4 rendering & capture
                                 └─► P5 UX & tooling
P6 EvoFlock • P7 MARL bridge • P8 optional tier      (after their P2 deps)
```

**P0 — Foundations (≈2 days).** F0 loader fix → F1 seeded RNG → F2
`mode_state` → F3 integrate variants + dt clamp + NaN guard → F4 min-image →
F5 renderer channel/VAO/depth-FBO/mat4/theme → F6 species column → F7
control hook → F8 swarm centre. Golden harness + invariant fuzz tests
pinned at the start.
*Accept:* every shipped preset loads with its domain intact; same-seed
determinism per mode; headless captures depth-correct.

**P1 — Scientific correctness (≈2 days).** W1.1–6 (occlusion culling, union
Θ, δ̂, φn, steric clamp, candidate cutoff), W3.1 kernel fixes, W5.1 Vicsek
update, W9.1/9.3 (thickness, H₂ inf). Re-pin golden.
*Accept:* collinear birds occlude; Θ sub-additive; δ̂ edge≈1/surrounded≈0;
noise scales live; D axis of the phase diagram does something.

**P2 — Physics workstreams (parallel, ≈8 days across tracks).**
- *Track A — field/threat (git6):* W2.1–10 in order R1→R12.
- *Track B — Reynolds variants (git3/git5):* W3.2–5, 7–8, 10; sphere
  centring (W1.8); ecology completion (W1.7).
- *Track C — angle mode (git4):* W4.1–6.
- *Track D — Vicsek species (git2):* W5.2–3.
- *Track E — influencer (git1):* W6.1–6.
Each track re-pins its mode's golden on completion.

**P3 — Metrics & analysis (≈2 days).** W9.2, 9.4–9.8; W5.4–5; W3.6; W6.6;
W7.3 metrics. *Accept:* per-item assertions (τ of constant density = 0;
nematic distinguishes two-lane; MSD slope 2→1 crossover; watts/joules
plausible).

**P4 — Rendering & capture (≈4 days).** W10.1–8 and W3.11. *Accept:* trails
in all three modes; impostor mode 60 fps at 20 k; adaptive ladder fires
under throttle; capture GIFs depth-correct, pre-warmed, swept.

**P5 — UX & tooling (≈2 days).** W11.1–3; W3.9 interaction set. *Accept:*
preset keys live; sliders drag; `--set` works; benchmark API returns
timings.

**P6 — EvoFlock (≈3 days, after P2-B and the SDF layer).** W8.1–7.
*Accept:* GA runs with crossover + worst-of-4; obstacle objective live;
evolved params persisted; emergent-alignment experiment reproduced.

**P7 — MARL bridge (≈2 days, after F7).** W7.1–5 (env, rewards, scripts,
dual-view/mpl fallback if not already in P4). *Accept:* `check_env` passes;
trained rollout beats random on cohesion.

**P8 — Optional tier.** W4.8 scenes/screensaver/overlay; GPU-compute
backend (ping-pong `⌈√N⌉²` texture packing blueprint); H&H flight physics
(gravity/lift/drag/banking — git5 R8); CMA-ES benchmark vs SSGA.

**Total ≈ 25–27 working days** (single track; P2–P5 parallelise to ~3
calendar weeks for two streams). If time is short, the highest
value-per-day cut is **P0 → P1 → P2-Track A → P4.1–4.4** — foundations,
correctness, the flagship field mode, and visible rendering.
