# roadmap2.md — Scientific correctness (S1) + mode workstreams (S2)

All simulation is strictly 3D ([roadmap0.md](roadmap0.md) §1). Each item:
*math* (3D form) → *impl* (module / class / config path / linkage) →
*tests* (concrete assertions; test files per the mirror convention in
[roadmap0.md](roadmap0.md) §5) → **Status** vs the current codebase
(bug/divergence register: [roadmap5.md](roadmap5.md)).

Gates: S1 needs D0–D2; S2 needs D1–D6 ([roadmap1.md](roadmap1.md)).
Metrics/rendering: [roadmap3.md](roadmap3.md). UX/EvoFlock/MARL:
[roadmap4.md](roadmap4.md).

---

## S1 — Scientific correctness cluster  *(≈2 days; re-pin goldens)*

**S1.1 Occlusion culling** — *math:* closest-first sweep; j visible iff
no accepted nearer k has `d̂_j·d̂_k ≥ cos α_k`; exact
`α = asin(min(b_eff/d, 1))`; candidates pre-filtered by
`cfg.projection.max_visibility`, capped at nearest
`cfg.projection.max_occlusion_neighbors` (64). *impl:*
`physics/occlusion.py::spherical_cap_occlusion` (signature stable;
array-native hot path — no per-neighbour object allocations; steric
import at module top). *tests* (`test/physics/test_occlusion.py`):
collinear (30,0,0),(60,0,0),(90,0,0) → `visible == [0]` (a legacy test
asserting `[2,1,0]` enshrines the bug — fix it); separated axes → all 3
visible; property (100 random configs): closest-first, duplicate-free,
no bird inside a nearer accepted cap, none in the blind cone, count ≤
cap.
**S1.1a Anisotropy identity**: `anisotropy=1.0` vs default → identical
`(δ̂, visible, Θ)`.
**Status: MOSTLY DONE.** Closest-first culling with the exact-asin cap,
64-candidate cap, blind cone, anisotropy, and a batched zero-allocation
path are implemented. Missing: the `max_visibility` distance pre-filter
and its config field (candidates are currently only the σ topological
neighbours passed in).

**S1.2 Θ probabilistic union** — *math:* `Ω_j = 2π(1−cos α_j)`,
`Θ = 1 − Π_visible(1 − Ω_j/4π)` (running product). *tests:*
`Θ(two separated caps) = 1−(1−Ω/4π)²` to 1e-6; metamorphic
sub-additivity `Θ₁ < Θ₁₂ < Θ₁+Θ₂`, monotone in neighbour count;
Θ ∈ [0,1] always.
**Status: DONE.**

**S1.3 δ̂ boundary-length weighting** — *math:*
`δ̂ = Σ sin α_j d̂_j / Σ sin α_j`, no magnitude clamp; |δ̂| *is* the
density signal. *tests:* octahedral surround → |δ̂| < 1e-2; single
neighbour → |δ̂| = 1 ± 1e-6; property |δ̂| ≤ 1; rotation-equivariance
δ̂(Rp) = R δ̂(p) for random SO(3) R.
**Status: DONE.**

**S1.4 Pearce noise term φn + weight constraint** — *math:*
`v ∝ φp δ̂ + φa ⟨v̂⟩_σ + φn η̂`, `φn = 1 − φp − φa`, η̂ uniform on S²
from `flock.rng`; the input handler renormalises the pair. *impl:*
`ProjectionMode.step`; `viz/input_control.py`. *tests:* hammer φp↑ 100×
→ `φp+φa ≤ 1` always (and symmetric); behavioural: φn = 0.2 keeps
residual heading variance > φn = 0 (same seed).
**Status: PARTIAL.** The φp+φa ≤ 1 input constraint is implemented
(`_enforce_phi_constraint`). The **φn noise term itself is missing** —
`ProjectionMode` blends only φp·δ̂ + φa·align.

**S1.5 Force-kernel corrections** — *math:* separation `Σ r̂/d²` (unit
direction over squared distance), with kernel selector
`cfg.spatial.separation_kernel: "sum" | "mean" | "unit"` — `mean`
divides the summed force by `max(1, n_found)` (dense regions *average*
instead of accumulate, so kick magnitude stays bounded as neighbour
count grows); `unit` is `normalize(Σ −(p_j − p_i))` over
`d < separation_distance` — the raw displacement sum normalised **once
at the end** (direction-only steering, no distance weighting); `sum`
keeps current semantics and stays the default. Cohesion
`limit3(p̄−p_i, 1)` — capped at unit length, sub-unit approach vectors
pass through unscaled (never inflate short vectors). Alignment in the
Reynolds *steering* form `normalize(v̄ − v_i)` (desired-minus-current
then normalise — `normalize(v̄) − normalize(v_i)` is a different vector
whenever speeds differ). `noise_force` output ×scale (the scale must
affect magnitude, not just direction draws). *impl:*
`physics/forces/_base.py`. *tests*
(`test/physics/forces/test_kernels.py`): one neighbour at d = 2 →
separation magnitude 1/4; mean kernel with k equidistant neighbours ==
sum kernel / k; cohesion magnitude ≤ 1 always AND == |p̄−p_i| when that
is < 1 (no inflation of short vectors); alignment unit-length and
parallel to `v̄ − v_i` for a hand pair with unequal speeds; noise mean
magnitude ≈ scale (±10 %, 10⁴ draws), zero at scale 0.
**Status: MISSING (all four corrections).** Current `_base.py`:
separation magnitude is 1/d (`−Δ/|Δ|²`), no kernel selector; cohesion
always normalises to unit length (inflates short vectors); alignment is
`normalize(v̄) − normalize(v_i)`; `noise_force` normalises its output so
`noise_scale` never changes the magnitude. A `test_kernels.py` exists —
**verify** what it pins; it must not enshrine the old forms.

**S1.6 Steric clamp** — *math:* `‖F‖ ≤ max_force` after the 1/d² sum.
*impl:* `physics/steric.py` (`max_force` param; caller passes
`cfg.flock.max_force`). *test:* pair at d = 0.01, strength 0.6, cap 0.15
→ ‖F‖ == 0.15 exactly.
**Status: PARTIAL.** The clamp parameter exists in `steric_force`, but
the projection-mode caller never passes `max_force` — the clamp is
inert in production. One-line fix + test.

**S1.7 Vicsek update corrected** — *math:*
`û_noisy = normalize(û_old + √(2DΔt)·n_⊥)` — the **memory term**;
tangent-plane noise `n_⊥ = g − (g·û)û, g ~ N(0, I₃)` (in 3D, isotropic
noise biases angular diffusion — project it out);
`û_new = normalize(η û_target + (1−η) û_noisy)`; constant speed via
`speed_mode="fixed"`; `cfg.vicsek.time_step` live. *impl:*
`physics/forces/vicsek.py::VicsekMode`. *tests*
(`test/physics/forces/test_vicsek_core.py`): lone-bird heading
autocorrelation `⟨û_t·û_{t+1}⟩` > 0.99 at D = 0.01 and < 0.5 at D = 4
(memory live); D ∈ {0.1, 2.0} at η = 0.7 → settled α differs > 0.3
(D live); |v| == `cfg.vicsek.velocity` ± 1e-5 every frame; property
`|n_⊥·û| < 1e-6`.
**Status: DONE** (memory term, √(2DΔt), tangent-plane projection,
per-species constant speeds). Note: fixed speed is self-enforced inside
the mode rather than via `speed_mode="fixed"` through `integrate()` —
converges once [roadmap1.md](roadmap1.md) D2 wiring lands.

**S1.8 Metrics formula fixes** — thickness `= √(λ₃/λ₁) ∈ (0,1]`;
`compute_h2` disconnected → `inf` (0.0 conflates "disconnected" with
"perfectly robust"); `find_optimal_m` skips non-finite J;
symmetrization `max(A, Aᵀ)`. *impl:* `analysis/metrics.py`. *tests*
(`test/analysis/test_metric_fixes.py`): thin-line flock thickness < 0.2
and ∈ (0,1], round cloud ≈ 1; two far pairs at m=1 → `math.isinf`; hand
3-node directed graph → max-form Laplacian.
**Status: MOSTLY DONE.** Thickness, `inf` on disconnect, and
non-finite-skip are implemented. **DIVERGES:** symmetrization uses
`(A + Aᵀ)/2`, not `max(A, Aᵀ)` — decide, pin with the 3-node hand test,
and note the H₂ values shift.

---

## S2 — Mode workstreams  *(parallel tracks; ≈11½ days)*

### Track A — Field/blob + threat  *(≈4 days)*
Files: `physics/forces/field.py::FieldMode`,
`physics/extensions/{wander,ripple,predator}.py`; tests
`test/physics/forces/test_field.py`,
`test/physics/extensions/test_threat.py`. Unit scale
`U = cfg.field.unit_scale or 0.4·min(W,H,D)`; `C` = domain centre;
seed-derived quantities cached in `reset()`.

**S2.A1 Wander path** — *math* (verified bounded-unit-travel):
```
raw_x = sin(t·0.47 + sin(t·0.19)·1.15)·0.82 + sin(t·1.07+1.4)·0.38 + cos(t·0.23+2.1)·0.22
raw_y = cos(t·0.43+0.6 + sin(t·0.13)·0.9)·0.78 + sin(t·0.91+2.8)·0.42 + cos(t·0.29+0.4)·0.24
raw_z = sin(t·0.39+1.1 + cos(t·0.17)·1.05)·0.80 + cos(t·0.97+0.2)·0.40 + sin(t·0.21+2.6)·0.22
pulse = 0.72 + 0.28·(0.5 + 0.5·sin(t·0.41 + cos(t·0.17)))
path(t) = raw · pulse / max(1, ‖raw‖)              ⇒ ‖path‖ ≤ 1 guaranteed
wander_center(t) = C + path(t·speed)·radius·U
heading(t) = normalize(path(t+0.75) − path(t))
```
*impl:* `extensions/wander.py` (fixes the domain-corner bug). *tests:*
`‖path‖ ≤ 1` for 10⁶ fuzzed t; heading unit & continuous
(‖h(t+ε)−h(t)‖ < 0.05); attractor in-domain over 10⁴ frames.
**Status: DONE with a wiring BUG** — the path/heading math matches, but
the extension reads non-existent config keys (`wander_speed`,
`attractor_radius`) so `wander_attractor_speed/radius` are dead and the
defaults differ (1.0 vs 0.10). Also the roadmap centre is C (or the
flock's smoothed centre) with `radius·U` scaling — current code uses
`radius` in raw domain units by design; pin one convention.

**S2.A2 Blob anchors + phase weights** — *math:* five Lissajous anchors
about `flock.center` (×U about C):
```
B₀ = C + ( sin(t·0.19)·0.74,      sin(t·0.31+0.8)·0.48,  cos(t·0.23)·0.62 )·U
B₁ = C + ( cos(t·0.17+1.6)·0.68,  sin(t·0.37+2.1)·0.54,  sin(t·0.29+0.4)·0.72 )·U
B₂ = C + ( sin(t·0.27+2.7)·0.58,  cos(t·0.21+1.2)·0.42,  cos(t·0.33+2.5)·0.68 )·U
B₃ = C + ( cos(t·0.24+3.4)·0.70,  sin(t·0.33+0.6)·0.50,  sin(t·0.18+1.4)·0.58 )·U
B₄ = C + ( sin(t·0.14+4.4)·0.48,  sin(t·0.47+2.3)·0.62,  cos(t·0.26+4.0)·0.70 )·U
```
`φᵢ = fract(seedᵢ·3.71 + t·0.022 + sin(seedᵢ·19 + t·0.11)·0.09)`;
`w_k = max(0, 1 − wrap(φ, c_k)·7.5)²`, c_k ∈ {0,.2,.4,.6,.8}, cyclic
`wrap(φ, c) = min(|φ−c|, 1−|φ−c|)`;
`T_legacy = Σ B_k w_k / Σ w_k` (Σw > 0 provably: 7.5·0.1 < 1).
Vectorised: weights `(N,5)`, `T = (w @ B)/w.sum(1, keepdims=True)`.
*tests:* `Σ_k w_k(φ) > 0` fuzzed; anchors at fixed t match hand values;
2 000 birds → k-means finds ≥ 4 clusters at t = 30 s; per-bird target
variance > 0.
**Status: DONE** (coefficients match). Note: `seeds` in FieldMode are
`arange(n_active)` rather than the flock's random `seeds` column —
decide which is canonical and test it.

**S2.A3 Leader/chaser** — *math:* 7 seed groups, `gs = floor(seed·7)/7`,
`phase = gs·2π`; the group's lagged anchor:
```
anchor(t, gs) = C + ( cos(phase + t·0.21)·0.50 + sin(t·0.13 + phase·2.3)·0.16,
                      sin(phase·1.7 + t·0.19)·0.34 + cos(t·0.11 + phase)·0.12,
                      sin(phase + t·0.16)·0.46 + cos(t·0.23 + phase·1.4)·0.14 )·U
lag = hash01(seed+9.17)·(1.1+2.4·chase)     → primary  = anchor(t−lag, gs)
secondary = anchor(t−lag, fract(gs + 1/7));   sec_mix = hash01(seed+3.33)·0.5
```
leaders `hash01(seed+5.91) ≥ 0.84` (~16 %),
`leader_target = C + wander_heading(t)·(0.18 + hash01(seed+7.1)·0.18)·U`;
golden-angle stratified shells (`slot` = the bird's stable seed-order
rank within its group): `ga = 2.39996323`,
`y = 1−2·fract((slot+0.5)·0.618034 + gs·0.13)`, `ring = √max(0, 1−y²)`,
`θ = slot·ga + gs·2π`, `shell = fract((slot+1)·0.754877)^{1/3}`,
`radius = (0.16+shell·0.34)(0.68+chase·0.34)(0.92+sep·0.045)·U`,
breath `1+sin(t·0.13+gs·12)·0.035`,
`offset = (cos θ·ring, y, sin θ·ring)·radius·breath`;
`follower_target = lerp(primary, secondary, sec_mix) + offset`;
`chase_target = leader ? leader_target : follower_target`;
`T = lerp(T_legacy, chase_target, chase)` — activates
`cfg.field.chase_strength`. Seed-only quantities (lag, role, group,
slot, sec_mix) cached in `reset()`. *tests:* leader fraction 0.16 ± 0.02
over 10⁴ seeds; group membership stable; chase = 0 ≡ S2.A2 targets
(allclose); chase = 0.8 → 7-cluster structure, leaders' anchor-distance
< followers'.
**Status: VERIFY / DIVERGES.** An implementation exists but deviates
from the spec in several load-bearing places: it has **no dedicated
`anchor(t, gs)` formula** (reuses the S2.A2 anchors indexed `gid % 5`),
**no secondary anchor / `sec_mix` blend**, uses a **per-group mean
lagged time** instead of per-bird lags, and the leader target is an
admitted approximation (group-phase direction instead of
`wander_heading`). The stratified-shell offsets broadly match. Either
implement the spec formulas above, or bless the current variant and
rewrite its tests — currently it is unclear which behaviour is intended.
Also `field_num_groups` / `field_leader_fraction` config fields exist
but the code hardcodes 7 and 0.84.

**S2.A4 Shell + cavity** — *math:* with `Δ = p − T`, `d = ‖Δ‖`,
`d̂ = Δ/d` (guard d > 1e-6):
`R_blob = (0.24 + (0.5+0.5 sin(seed·41+t·0.29))·0.16 + sin(φ·2π+t·0.17)·0.05)U`;
`F = −d̂(d−R_blob)·coh·1.35(1−chase)·shell_influence`; inner floor
`R_blob(0.28+(1−chase)·0.18+sep·0.012)`, push-out `d̂(inner−d)·sep·1.4`
when `d < inner`. *tests:* settled blob — central voxel density < 0.3×
shell band; R_blob FFT shows both documented oscillation frequencies.
**Status: DONE** (the implemented `0.32 + 0.08·sin + 0.05·sin` expansion
is algebraically identical; blackening modulation of sep/coh wired).

**S2.A5 Remaining terms (full 13-term composition)** — *math:* slot
repulsion offsets **±{1,7,31}** mod-wrapped
(`other = positions[(i+o) mod n_active]`, active-compacted order),
kernel `((r_slot−d)/r_slot)²` inside `r_slot = (0.07+sep·0.02)U`, gain
`sep(0.14+chase·0.05)`; tangential
`normalize(axis×(p−T))·align·0.035(1−chase)·tangent_pull` with drifting
seed axis `axis = normalize(sin(t·0.13+seed·7),
0.72+sin(t·0.19+seed·3)·0.28, cos(t·0.17+seed·5))`; buoyancy (z-up)
`F_z += (sin(8d/U−1.1t+17·seed)·0.09 + 0.24(T_z−p_z)/U)(0.75+0.25·flow)`;
curl flow, with `q = (p−C)/U`:
```
flow = ( sin(q_y·2.8 + t·0.24 + seed) + cos(q_z·2.1 − t·0.17),
         sin(q_z·2.3 + t·0.20)        − cos(q_x·1.9 + t·0.24),
         sin(q_x·2.6 − t·0.16)        + cos(q_y·2.2 + t·0.24) )
F_flow = normalize(flow)·flow_w·0.08·flow_pull      (flow_w = cfg.field.flow)
```
fold noise (the finer second band, coupled to ripple activity):
```
fold = ( sin(q_y·3.7 + t·0.73 + seed) + cos(q_z·2.9 − t·0.51),
         sin(q_z·3.1 − t·0.67 + seed) − cos(q_x·2.4 + t·0.43),
         sin(q_x·3.3 + t·0.59 + seed) + cos(q_y·2.6 − t·0.47) )
F_fold = fold·flow_w·flow_pull·ripple_envelope_sum        (S2.A6 export)
```
drag `−v·chase(0.08+0.02·flow)`; drift alignment to `heading(t)·v0`;
**target pull** `(T−p)/U·coh·target_pull`; **boundary containment**
(the composition's own term — distinct from the S2.B7 asymptotic wall):
for `d = ‖p−C‖ > 1.45U`, `F −= r̂·(d−1.45U)·1.6` — a *linear* overshoot
spring, zero inside `1.45U` (keeps the blob free-floating rather than
hard-projected). **Composition contract**: the 13 terms are pure, named
functions `(flock, cfg, cache) → (N,3)` registered in an ordered
`FIELD_TERMS` table; `FieldMode.step` composes them by reduction, and
`cfg.field.disabled_terms: list[str]` (default `[]`, live-mutable) skips
entries at runtime — per-term isolation for benchmarks/A-B comparison
without mode forks, each term individually unit-testable. *tests:* each
term unit-pinned on hand inputs (slot kernel zero outside r_slot &
continuous at it; buoyancy z-only; drag anti-parallel; flow/fold
normalized pre-gain; boundary zero at d = 1.44U, linear in overshoot at
2U vs 3U — slope 1.6); full step force == Σ of individual term outputs
on a frozen state; disabling one term changes the sum by exactly that
term's contribution; unknown name in `disabled_terms` warns and is
ignored; 10⁴-frame NaN/speed fuzz all-terms-on; tangential on → nonzero
sign-stable angular momentum about blob axes.
**Status: PARTIAL / DIVERGES.**
- Implemented and matching: tangential, buoyancy, curl flow, fold noise,
  drag, drift alignment, slot-repulsion kernel/gain.
- Slot repulsion pairs are **not mod-wrapped** (`active_idx[:-o]` vs
  `(i+o) mod n_active` — the wrap pairs are skipped).
- **Target pull term MISSING** (`field_target_pull` is a dead config
  field).
- **Boundary containment DIVERGES**: implemented as
  `−μ·r̂/max(overshoot, 0.05·R_boundary)` with
  `R_boundary = 1.45·max(R_blob)` (inverse-overshoot, μ from the
  boundary config) instead of the linear `(d−1.45U)·1.6` spring — decide
  which law is canonical and pin its slope test.
- **Composition contract MISSING**: `ForceTerm`/`composeForces` exist in
  `_base.py` but `FieldMode` hardcodes the term sequence — the atoms are
  dead; no `disabled_terms`.
- Fold noise consumes a **scalar** ripple export (see S2.A6 bug).

**S2.A6 Ripples** — *math:* three trains, offsets o ∈ {0, 9.33, 18.67}
s, per-train `τ = (t − o) mod 28` (28 s cycle);
`env = ss(0.6,1.7,τ)(1−ss(6.2,8.8,τ))`; `radius = (0.16+0.16τ)U`;
`width = (0.11+0.012τ)U`; moving origin
`origin = C + (sin(t·0.17+o)·0.46, cos(t·0.13+o·1.7)·0.25,
cos(t·0.19+o·0.6)·0.42)·U`; with `r = ‖p−origin‖`,
`δ = |r−radius|/width`, `amount = exp(−δ²)·env`:
`F_radial = (p−origin)/r·amount`, twist `+(heading×F_radial)·0.28`,
total `F_ripple = (F_radial + twist)·flow_w·(0.13+0.04·waveGain)`;
export `ripple_envelope_sum = Σ_trains amount` (per-bird; consumed by
S2.A5's fold term). *impl:* vectorised in `FieldMode`;
`extensions/ripple.py` a thin wrapper for other modes. *tests:* env zero
outside [0.6, 8.8], peak in [1.7, 6.2]; origins move; paused-flock
radial histogram shows 3 rings; envelope sum matches a hand-computed
3-train value at fixed t; < 5 ms at N = 100 k.
**Status: DONE with one BUG.** Formulas match, but the export collapses
the per-bird `(N,)` envelope to a **scalar `np.sum` over all birds** —
the fold-noise magnitude then scales with N (thousands× too large for
big flocks). Export the per-bird array (Σ over trains only).

**S2.A7 Inertia / bounded panic / blackening** — *math:* inertia lerp
(integration contract, [roadmap0.md](roadmap0.md) §4.5); panic ceiling
`v0(1+min(1.35, panic(0.72+0.18·wave+0.12·vacuole)))` via per-bird
`max_speed` (a ceiling *raise*, never a compounding multiply);
blackening `sep_eff = sep(2−black)`, `coh_eff = coh·black`,
`black = 1+gain·prox·0.85` (prox from S2.A8 via `ctx.threat_prox`).
*tests:* max speed ≤ 2.35·v0 across 10⁴ panic frames; wake-region pair
distance drops during a pass; inertia 0.8 → per-frame |Δ‖v‖| < 0.2·v0.
**Status: PARTIAL.** Panic ceiling (max-raise) and blackening are
implemented. **Inertia is dead**: `integrate()` accepts `inertia` but
the engine/flock never passes `cfg.field.inertia` — every field preset's
`inertia` value is ignored.

**S2.A8 Threat FSM + force bundle** — *math (source-verified):*
persistent state `{pos, vel, dir, turn_axis, phase ∈ {approach,
egress}}`; speed `2·v0·(1+0.5·mom)`, moved `pos += dir·speed·dt`;
`capture = max(0.18, 0.72R)U`; `pass = (0.92+2.6R+1.32·mom)U`;
`clear = pass(0.72+0.16·mom)`; approach→egress at
`dist_to_center ≤ capture`; egress→approach at `dist > clear` AND
heading gate `dot(dir, to_center̂) < −0.12`; target = approach ?
`flock.center` : `center + dir·pass + arc`; steer response ×
`1.86+(1−mom)·0.48` (approach) / `0.34+(1−mom)·0.44` (egress); turn
rate `(0.54+0.025·accel)(1−0.24·mom)` rad/s (orbit 0.42·…);
sign-aligned EMA turn axis — `desired = normalize(dir × to_center̂)`,
negate if `dot(prev, desired) < 0`,
`axis ← normalize(lerp(prev, desired, amt))`; Rodrigues `rotate_toward`
capped at `rate·dt` (any ⊥ axis when parallel/anti-parallel); egress arc
`broad = R·(0.36 chase | 0.24 orbit)·U`, lift
`turn_axis·sin(0.18t+0.7)·broad`, drift
`normalize(turn_axis×dir)·cos(0.13t+1.4)·broad·0.72`; force bundle on
birds within `d < 2R·U`, `prox = 1 − d/(2R·U)`, `broad = √prox`,
`â = (p−p_threat)/d`: push `â·strength(2.5+1.7·vacuole)·broad`; wake
`(â−dir·0.35)·min(1.8, ‖v_t‖/v0)·strength·broad·0.42`; split
`(−â_y, â_x, 0.28/1.45·â_z)·splitGain·broad·1.45` (horizontal tear,
z-up); wave `v̂·waveGain·broad·0.22`. Modes off/cursor/orbit/autonomous
(`cursor` = threat at the mouse-ray median-depth point,
[roadmap4.md](roadmap4.md) S5.4; falls back to orbit headless). *impl:*
`extensions/predator.py → Threat(Extension)`; publishes
`ctx.threat_prox`; **rendered** via the flag channel (an invisible
predator is undebuggable). *tests:* `rotate_toward` Rodrigues-exact and
capped; phase transitions at exact distances with the dot gate; trace
continuous (max step < 3·speed·dt), crosses and exits ≥ clear;
evacuated region horizontally biased (xy-extent > z-extent);
`threat_prox ∈ [0,1]` shape (N,); *(gl)* red marker visible in all
themes.
**Status: PARTIAL.** Implemented: FSM state + capture/pass/clear
distances, dot-gated egress→approach, speed law, Rodrigues
`_rotate_toward` with cap, egress arc (lift+drift), the four-force
bundle with `√prox`, panic ceiling, blackening publication,
`ctx.threat_prox`. Missing/diverging: **steer-response multipliers**
(1.86+…/0.34+…) absent; **sign-aligned EMA turn axis** absent (fixed
axis (0,1,0)); **threat modes** (off/cursor/orbit/autonomous) absent —
always autonomous; `ThreatConfig.mode/acceleration/vacuole_strength/
blackening_gain` not real config fields (getattr defaults); **no
rendered marker**.

**S2.A9 Blob init + presets** — *math:* 5 fixed centres
`(−0.48,0.18,0.12) (0.36,−0.20,−0.28) (0.12,0.34,0.42) (−0.16,−0.30,0.34)
(0.48,0.16,0.18)` (×U about C, assigned `i mod 5`), ∛-uniform shells
`r = cbrt(u)(0.22+u'·0.28)U`, jitter `U(−1,1)·0.045U` per axis; drift
velocities
`v = ((0.34+U(−1,1)·0.08), U(−1,1)·0.16, (0.08+U(−1,1)·0.08))·v0·0.5 +
jitter(0.05·v0)` (coherent initial flow, wired to
`velocity_init: "drift"`); presets `conf/field_*.yaml`
(columns: N, speed×v0, sep, align, coh, chase, inertia, noise, flow,
trail, threat):
```
quiet_roost      3000  0.48 0.85 0.65 1.85 0.72 0.82 0.03  0.18 velocity     off
lava_lamp       16000  defaults (pure blob dynamics, chase 0)
ink_cloud       18000  0.62 0.92 0.90 1.80 0.82 0.84 0.035 0.30 accumulation autonomous
predator_ripple 12000  0.78 1.05 1.05 1.15 0.64 0.70 0.08  0.48 velocity     orbit
vacuole         10000  0.68 1.12 0.92 1.25 0.76 —    —     0.42 accumulation autonomous (vacuole_strength 0.9)
silk_sheet      14000  0.46 0.92 1.10 1.10 0.68 0.88 0.025 0.24 velocity     off
storm_turn      16000  0.90 1.10 1.15 1.25 0.42 0.58 0.10  0.72 velocity     autonomous
```
*tests:* frame-0 lobes; init densities equal across N (±10 %); drift
init mean velocity within 5 % of `(0.34, 0, 0.08)·v0·0.5` over 10⁴
birds; presets load with the tabled values.
**Status: MOSTLY DONE.** Blob position init and the drift-velocity
vector are implemented (`init_positions("blob")`,
`init_velocities_blob` — wired as `velocity_init: "blob"`; add the
`"drift"` alias and the `jitter(0.05·v0)` term). The seven
`conf/field_*.yaml` presets **exist** — **verify** their values match
the table, and note their `inertia`/`threat mode` columns are currently
inert (S2.A7 inertia dead; threat modes missing).

### Track B — Reynolds variants  *(≈3½ days)*
Files: `physics/forces/spatial.py`, `physics/forces/_base.py`,
`physics/forces/_kernels.py`, `physics/extensions/ecology.py`,
`physics/boid.py`; tests
`test/physics/forces/test_spatial_variants.py`,
`test/physics/extensions/test_ecology.py`.
(Hildenbrandt–Hemelrijk flight physics is deliberately excluded from
this track; its starting math is recorded in
[roadmap5.md](roadmap5.md) Appendix A.)

**S2.B1 Hybrid filter + dual radii** — neighbour iff `d < visual_range`
AND among first `cfg.spatial.influence_count` (7); alignment subset
`d < alignment_radius_ratio·R` (0.75); `separation_distance` (20) as a
metric gate. Extend `cfg.spatial.neighbor_filter` to
`knn | hybrid | global`: the `global` degenerate case steers
alignment/cohesion toward the **whole-flock** mean velocity / CoM (no
radius, no kNN) — the same behaviour the marl mode's embedded rules use
([roadmap4.md](roadmap4.md) S7.1), exposed as a general spatial-mode
option for studying global vs local coupling. Ship
`conf/murmuration_starlings.yaml` — the source-parity preset exercising
the whole track: `mode: spatial`, `flock: {num_boids: 150, v0: 4.0,
visual_range: 80}`, `spatial: {separation_weight: 3.0,
cohesion_weight: 0.2, alignment_weight: 0.02, noise_scale: 0.05,
neighbor_filter: hybrid, influence_count: 7,
alignment_radius_ratio: 0.75, speed_mode: fixed, parameter_jitter:
true}`, `boundary: {mode: sphere, sphere_radius: 300,
avoidance_factor: 0.05}`, `viz: {bird_mesh: winged, background:
gradient}`. *tests:* hand cluster — neighbour set respects radius AND
cap; alignment set ⊆ cohesion set; `global` → every bird's cohesion
target equals the flock CoM; preset loads with the listed values and
settles into cohesive rotating groups inside the sphere within 500
frames (`@slow` behavioural smoke).
**Status: PARTIAL.** The hybrid metric+topological filter is
implemented (numba kernel + numpy fallback). Missing: the
alignment-radius subset (`alignment_radius_ratio` is a dead field),
`separation_distance`, the `neighbor_filter` selector with `global`,
and the starlings preset.

**S2.B2 Update-order fidelity** — order: predator boost(×1.4) →
`acceleration_scale`(0.3) → limit(max_force) → `v += a` → velocity
noise `(U³−0.5)·noise_scale` (when `noise_mode="velocity"`) → ceiling
limit → move; `speed_mode ∈ {band, ceiling, fixed}` (effective predator
damping 1.4×0.3 = 0.42 vs prey 0.30 — boost before scale, order
matters). Ship `conf/murmuration_boids.yaml`: `mode: spatial`,
`flock: {num_boids: 150, v0: 6, max_force: 1, visual_range: 100}`,
`spatial: {separation_weight: 4.5, alignment_weight: 0.65,
cohesion_weight: 0.75, acceleration_scale: 0.3, separation_kernel:
unit, noise_mode: velocity, speed_mode: ceiling,
separation_distance: 20}`, `boundary: {mode: toroidal,
use_toroidal_distance: true}` (predators spawned via right-click).
*tests:* monkeypatched-stage order recording; "ceiling" allows
|v| < 0.3v0; "fixed" → |v| ≡ v0; preset loads with the listed values
and reaches α > 0.6 within 300 frames (`@slow` behavioural smoke).
**Status: PARTIAL.** The accumulate → acceleration_scale → clamp →
noise-after-clamp pipeline exists. Missing: velocity-noise mode, the
per-config `speed_mode` selection actually reaching `integrate()`
([roadmap1.md](roadmap1.md) D2 wiring), the boost-before-scale ordering
test, and the boids preset.

**S2.B3 Predator boids (species)** — boosts 1.8× speed / 1.5×
perception / 1.4× acceleration; escape
`normalize(p_prey−p_pred)·cfg.spatial.predator_escape_factor (10⁷)`
**replacing** separation (min-image difference on toroidal domains; the
subsequent max-force limit caps it — its job is to *win the sum*, not
set the magnitude); **hard-zero** align+cohesion when any predator is
perceived; predators flock among themselves; whenever the species
column is populated, α/dispersion (and the other flock observables) are
computed over **prey only** in every mode. *tests:* hand neighbourhood →
align/coh contributions exactly zero; escape wins the sum pre-limit;
flash-expansion (mean NN distance doubles in 30 frames); two predators'
pair distance stabilises; α of an aligned prey flock is unchanged by
adding one orthogonal predator (prey-only metrics).
**Status: PARTIAL.** Detection, escape-replaces-separation, and
hard-zero align/coh are implemented (numba kernels). Missing: the
speed/perception/accel boost application path (config fields exist —
verify they are actually consumed), min-image escape on toroidal
domains (current escape uses raw differences), and **prey-only metrics
everywhere** (currently metrics include predators).

**S2.B4 Physical metrics** — `k_v = cruise_ms/v0` (8.94 m/s default),
`k_a = acc_peak/max_force` (40 m/s²), m = 0.075 kg;
`F = m·k_a⟨|a|⟩` (N); `P = m⟨|k_a a · k_v v|⟩` (W); `E = Σ P·Δt` (J);
`L = m(r−CoM)×(k_v v)`; reads `flock.last_accelerations`. *impl:*
`analysis/metrics.py` + `cfg.metrics.*`. *tests:* hand-set v →
`speed_real == k_v·|v|` exactly; E ≈ P̄·elapsed ± 1 %; stash test —
metrics see nonzero pre-reset accelerations.
**Status: DIVERGES.** Conversions for speed/accel/force exist; but
`power_real_W` multiplies *means* (`F̄·v̄`) instead of the mean of the
per-bird `|a·v|` product, and `energy_J` is instantaneous kinetic
`½mv²`, not the accumulated work `Σ P·Δt`. Metrics read live
`accelerations` — **verify** they see pre-reset values (the engine
collects after `integrate()`, which zeroes accelerations; the stash
`last_accelerations` exists but metrics do not read it — likely reading
zeros; fix to read the stash).

**S2.B5 Parameter jitter** — effective weights per frame from
`flock.rng`: `sep + U(0, 0.5)`, `coh + U(0, 0.1)`,
`align + U(0, 0.005)`; config never mutated. *tests:* spacing-series
std(on) > std(off), same seed; config unchanged after run; determinism
holds.
**Status: DIVERGES.** Implemented as multiplicative
`weight·(1 + U(0, jitter_*))` with per-field jitter amplitudes, not the
additive absolute ranges. Decide (additive spec vs multiplicative
current), pin, and re-check the starlings preset semantics
(`parameter_jitter: true` flag absent).

**S2.B6 Parallel two-phase** — batched
`index.query_knn_batch(pos, k, workers=cfg.perf.num_threads)` + fully
vectorised gather/reduce force pass (`positions[neighbor_idx]` shape
(n,k,3), masked padding, axis-1 reductions — no per-bird Python loops).
*tests:* ≥3× at N = 20 k vs recorded loop baseline (`@slow`); identical
results across worker counts (T4.3).
**Status: MOSTLY DONE** for spatial mode (batch `tree.query(workers=-1)`
+ vectorised primitives). `workers` is hardcoded −1
(`perf.num_threads` field missing); vicsek fear/hunt paths and angle
mode remain per-bird Python loops.

**S2.B7 Sphere centring + asymptotic wall** — centre on C
([roadmap1.md](roadmap1.md) D4); add `boundary.mode = "sphere_soft"`:
`Δv = −μ r̂ / max(R−r, 0.05R)` applied inside the shell margin. *tests:*
centre-initialised flock ‖CoM−C‖ < 0.1R over 5 000 frames, both modes;
soft mode never crosses R.
**Status: MISSING** (and the centring bug is live — see
[roadmap1.md](roadmap1.md) D4).

**S2.B8 Ecology completion** — *math:* logistic dusk `1/(1+e^{−z})`,
`z = (hour−sunset(day))/(width/4)` clamped |z| > 60,
`sunset = 12 + day_length/2`; `is_roosting_time(hour, day, 0.5)`; cold
boost `roost_strength = base·dusk·max(0, 1+0.2(T_mean − T(day))/T_amp)`;
`roost_force = unit(roost−p)·roost_strength`; **coherence gate**
`coherence(N) = smoothstep over [0.4, 1.2]·N_crit`,
`gated_weight(w, N) = w·coherence(N)` **applied to the flocking
weights** (φa/φp or spatial weights); **seasonal model**
`seasonal_size_factor(day)` = cosine, 1.0 at PEAK_DAY = 15,
MIN_FACTOR = 0.25 at +182; `flock_size_for_day(day, peak_size,
min_size=0) → int` driving N via the command queue when
`cfg.ecology.seasonal_size`; `is_murmuration_season(day)` (Oct–Mar)
gating roost/predator behaviour; **stochastic predator presence**
`predator_present(day, rng=None)` — empirical rate
`PREDATOR_RATE = 0.296`; deterministic per-day Knuth-hash branch
(`((day·2654435761) mod 1000)/1000 < RATE`, reproducible) OR a true
draw `rng.random() < RATE` when an rng is supplied (from `ctx.rng`),
selected by `cfg.ecology.predator_presence:
deterministic|stochastic` (a `77/256` shortcut drifts from the cited
rate — use `RATE`); `cfg.ecology.roost / critical_mass` live. *impl:*
free functions + `Ecology(Extension)` in
`physics/extensions/ecology.py`. *tests:*
`seasonal_size_factor(PEAK_DAY) ≈ 1.0`, `(+182) ≈ MIN_FACTOR`;
`flock_size_for_day` ints in range, curve-shaped; season
Jan-in/Jul-out; `dusk_factor(0)=0, (40)=1`, 0.5 at sunset; colder →
stronger (day 20 > day 200, same hour); `gated_weight(0.8, 10) ≈ 0`,
`(0.8, 600) > 0.7`; behavioural — gate on: α(N=50) < α(N=800) identical
params; seasonal N tracks the curve over a simulated year;
`predator_present` deterministic same-day-same-result and yearly
frequency 0.296 ± 0.03, seeded-rng frequency 0.296 ± 0.01 over 10⁴
draws.
**Status: PARTIAL / DIVERGES.** Implemented: logistic dusk (different
parameterisation — minutes-before-dusk with `_DUSK_CENTER = 20`,
`dusk_width` in minutes), seasonal amplitude (different formula:
`1 − a·0.75·(1−cos)`), smoothstep coherence gate (over `[0, N_crit]`,
not `[0.4, 1.2]·N_crit`), temperature boost (different form),
day-boundary predator presence via the **77/256 hash shortcut the spec
explicitly replaces**. Missing: `is_roosting_time`,
`is_murmuration_season`, seasonal flock-size driving N through the
command queue, the stochastic-rng option, `unit(roost−p)` force form
(current is linear `to_roost·0.01·ramp` in a per-bird Python loop —
also a perf smell), gating applied to φa/φp (only spatial weights are
gated). Decide formula-by-formula: adopt the spec forms (re-pin tests)
or bless the current ones (rewrite this item).

**S2.B9 Init variants (velocity + position)** — *velocity* — `cube`:
`(U³−0.5)·2v0` (E|v| ≈ 0.816·v0); `speed_uniform`: uniform direction ×
`U(min(1, 0.3v0), v0)`; `tangential`: `normalize(p−C)×random_unit ·
U(1, v0)`; selector `cfg.flock.velocity_init`. *position* — selector
`cfg.flock.position_init`; `box | sphere_shell | gaussian | grid |
blob` implemented; add the filled **`sphere`** variant —
volume-uniform ball about C: `r = ∛u·0.88·R_dom`
(`R_dom = 0.4·min(W,H,D)`), direction uniform on S² — the ∛ law gives
constant density in 3D (a shell-free single-cloud start;
`sphere_shell` is surface-only). *tests:* cube mean ≈ 0.816·v0 (±5 %,
10⁴ birds); speed_uniform in-band, non-constant; tangential ⊥ radial
(dot < 1e-5); sphere init — radial-bin counts ∝ r² (±15 %, 10⁴ birds),
max r ≤ 0.88·R_dom, all in-domain; each position_init value produces n
in-domain points and is seed-reproducible.
**Status: PARTIAL, one BUG.** cube/speed_uniform/tangential/fixed/blob
velocity modes exist (speed_uniform lower bound is 0, not
`min(1, 0.3v0)`; tangential speed is constant v0, not `U(1, v0)`).
**BUG:** `position_init: "sphere"` passes validation but has **no
branch** in `init_positions` — it silently falls through to `"box"`.
Implement the filled-sphere variant.

**S2.B10 Numba force kernels + precision policy** — two-pass: batched
index query (Python/scipy) → `@njit(parallel=True)` kernel over
`(positions, velocities, accelerations, active, neighbor_idx,
weights…)`. `cfg.perf.use_numba` gates; `cfg.perf.fastmath` allowed
**only** when `metrics.detail_level == 0` (visual runs) — IEEE kernels
whenever observables are exported; lazy import; numpy path stays the
reference. *impl:* `physics/forces/_kernels.py`, consumed by
SpatialMode/VicsekMode. *tests:* numba ≡ numpy within `atol=1e-5`
(fastmath off), same seeds, N = 2 000; exporting metrics with fastmath
on raises/warns; `@slow` N = 50 k step within budget ×2.
**Status: PARTIAL.** `_kernels.py` exists (hybrid filter, predator
detect/escape) with numpy fallbacks and `cache=True`. Missing:
`use_numba`/`fastmath` config gates + the fastmath-vs-metrics policy,
`parallel=True`, vicsek kernels, and the equivalence test at N = 2 000.
(GPU-compute simulation backends remain excluded —
[roadmap5.md](roadmap5.md) Appendix A; numba is the accepted
acceleration tier.)

**S2.B11 Grid-tier flow + deterministic seed noise** — *math:* the
**same curl-flow primitive** S2.A5 uses (per-axis normalized sin+cos
pairs, cyclic in (x,y,z), `F = normalize(f)·gain`) offered to **spatial
mode** at the documented grid gain **0.22** behind
`cfg.spatial.flow_weight` (default 0 = off — one shared Level-0
function, two composers, no duplicate math); **seed-sinusoidal noise**
as `cfg.spatial.noise_mode: "seed_sinusoidal"` — wire the existing but
unconsumed `core/types.py::seed_noise3(seeds, t)` atom (deterministic
per-bird sinusoids, bounded ±0.18/axis), output ×`noise_scale/0.18` so
`noise_scale` keeps its magnitude meaning; deterministic ⇒ same-seed
replay covers the noise term too (T4.3). *impl:* curl-flow atom in
`physics/forces/_base.py` (FieldMode imports it); `SpatialMode.step`
adds both terms; no dead atom remains. *tests*
(`test/physics/forces/test_spatial_variants.py`): flow output
unit-length pre-gain, varies with p, t, and seed; `flow_weight = 0` →
bit-identical to baseline; seed_sinusoidal — per-axis bound
`noise_scale` respected, same (seeds, t) → identical output, two
same-seed runs bit-identical with noise on; FieldMode and SpatialMode
flow terms agree on identical inputs up to their gains.
**Status: MISSING.** `seed_noise3` exists and is a dead atom; the curl
flow lives only inside `field.py` (not a shared `_base.py` atom); no
`flow_weight`/`noise_mode` fields.

### Track C — Angle mode  *(≈2 days)*
Files: `physics/forces/angle.py::AngleMode` (`speed_mode="fixed"`,
per-bird speeds); tests `test/physics/forces/test_angle.py`.

**S2.C1 Steering core** — dead zone (no turn if error <
`turn_threshold`°, anti-oscillation); 3D axis-angle:
`φ = acos(clamp(ĥ·t̂, −1, 1))`, axis `normalize(ĥ×t̂)` (any ⊥ axis
when parallel/anti-parallel), rotate
`rotate_about(ĥ, k̂, min(φ, rate·dt))` — never overshoot. *tests:*
180°-behind target turns through π/rate seconds ± 1 frame; per-frame
heading change ≤ rate·dt + jitter; dead-zone hold exact.
**Status: DONE** (implementation matches; per-bird Python loop —
acceptable initially).

**S2.C2 Unified neighbour modes** — 7 closest within `b·12`
(b = boid_size): nearest < `b·1` → steer away from nearest (exclusive
flee state); nearest < `b·5` → toward `normalize(ĉ + m̂)` (centroid +
mean heading; full 3D — no planar angle averaging); else → ĉ only.
*tests:* forced-close pair separates; mid-range cluster contracts AND α
rises; far cluster contracts only.
**Status: DONE** (min-image handling included for toroidal).

**S2.C3 Adaptive speed** — `s = base + (7−m)·5` (linear) | `+(7−m)²`
(quadratic, cap 49) | `+min(49, (7−m)²/2)` (softened), per
`cfg.angle.speed_mode`. *tests:* m = 0 → base+35 (linear); m ≥ 7 →
base; median 7th-NN distance converges (self-regulating density).
**Status: PARTIAL** — only the linear law exists; no selector.

**S2.C4 Edge handling** — inside `margin`: steering target overridden
by the nearest face's inward normal (±x/±y/±z, sequence priority);
spherical variant `t̂ = normalize(C−p)`; turn rate
`rate += (1 − edgeDist/M)(maxRate − rate)`. *tests:* margin boundary,
10⁴ frames, zero escapes at max speeds; birds arc (mean tangential
speed at the wall > 0 — no sticking).
**Status: DONE** (margin faces + sphere variant + turn-rate ramp).

**S2.C5 Heading jitter** — ±`jitter_deg`° rotation about a random axis
**before** steering (steering compensates), from `flock.rng`. *tests:*
steering-off distribution bounded ±4°, ~symmetric; net track endpoint
within 2 % of jitter-off run.
**Status: DONE.**

**S2.C6 Incremental grid** — per-bird `last_cell`; re-file only on cell
crossing (vs full rebuild); behind the SpatialIndex protocol. *tests:*
neighbour sets == full-rebuild sets over 500 random-walk frames;
touches < 10 % of birds/frame at N = 5 k.
**Status: DONE with a defect** — `incremental_rebuild` exists on the
hash grid, but `AngleMode._last_cell` is **class-level state** shared
by every engine in the process (two engines clobber each other; mode
switches leak it). Move to instance state
([roadmap1.md](roadmap1.md) D2).

**S2.C7 Body-unit radii** — `sep/align/range_radius_bodies` scale with
`boid_size` (scale invariance; also offered to spatial mode as
`radii_in_bodies`). *tests:* doubling boid_size doubles all three
thresholds; 2×-scale behavioural smoke.
**Status: PARTIAL** — the body-unit scaling is coded, but **none of the
angle knobs are config fields** (all read via `getattr` defaults, so no
AngleConfig section exists and nothing is tunable). Spatial-mode
`radii_in_bodies` missing.

**S2.C8 Angle-mode preset** — ship `conf/murmuration_angle.yaml` with
the source-parity values: `mode: angle`, `flock: {num_boids: 200,
boid_size: 9}`, `boundary: {mode: margin}`, `angle: {turn_rate: 120,
max_turn_rate: 200, turn_threshold: 0.5, jitter_deg: 4, margin: 42,
speed_mode: linear, base_speed: 150, neighbors: 7,
sep_radius_bodies: 1, align_radius_bodies: 5, range_radius_bodies:
12}`, `viz: {per_bird_color: true, trails: ring}` (these double as the
AngleConfig defaults). *tests:* preset loads with the listed values;
the mode golden pins its trajectory; margin containment at these speeds
(10⁴ frames, zero escapes — S2.C4's test run on the shipped preset).
**Status: MISSING** (blocked on AngleConfig; note current in-code
defaults differ — `max_turn_rate: 360`, `turn_threshold: 0.8`,
`base_speed: 4.0`).

### Track D — Vicsek predator–prey  *(≈2 days)*
Files: `physics/forces/vicsek.py`; tests
`test/physics/forces/test_vicsek_species.py`.

**S2.D1 Species dynamics** — *math:* fear
`= clamp((R_pred − d̄_pred)/R_pred, 0, 1)` (d̄ = mean distance to
predators within R_pred, min-image);
`û_combined = normalize((1−fear)·û_align + fear·û_flee)`,
`û_flee = normalize(Σ (p_prey − p_k)_mi / |P|)` (random unit if none);
neighbour weights ×`weight_afraid` (3.0) while afraid; predator update:
nearest prey within `detect_ratio·R_pred` (1.5×), then
`û = normalize(û_target + predator_noise_ratio·η̂)` — **no couplage**;
random-walk fallback; all-predators early-out (pure random walk, skip
all interaction). *tests:* stationary predator at flock centre → prey
inside R_pred have ⟨û·r̂⟩ > 0.8 within 5 steps; monotone pursuit
(≥ 90 % of steps close distance); n_prey = 0 → α ≈ 1/√N for all η, D;
afraid birds align to neighbours more strongly than calm (two-group
setup).
**Status: MOSTLY DONE.** Fear blending (with `weight_afraid` in the
blend), solo-prey flee, predator hunting with noise + random-walk
fallback, and the all-predator early-out are implemented (per-bird
Python loops — vectorise later). **Verify:** the early-out `return`
leaves predator velocities unchanged instead of random-walking them
(spec: pure random walk); the fear blend folds `(1−η)·û_noisy` into the
combined vector — confirm against the spec's two-stage form.

**S2.D2 Asymmetric position collisions** — *math:* same-type pairs at
`d < R_avoid`: each moves `(R_avoid − d)/2` along min-image n̂;
prey–predator at `d < R_pred`: **prey takes the full** `(R_pred − d)`
correction, predator unmoved; applied after move, before wrap;
`np.add.at` accumulation. Activates `cfg.vicsek.radius_avoid`. *tests:*
hand pair corrections exact (both cases); seam-crossing pair corrected;
100 steps → no same-type pair < 0.5·R_avoid; predator trace unaffected
by contacts.
**Status: DONE** (O(N²) brute force with min-image, applied post-move
with re-wrap in the engine; sequential accumulation instead of
`np.add.at` — fine for small N, note for scaling).

**S2.D3 Prey-only metrics in vicsek mode** — *test:* α of aligned prey
+ one orthogonal predator == 1.0.
**Status: MISSING** (metrics include predators in every mode;
generalise per S2.B3).

**S2.D4 Preset parity + order transition** —
`conf/murmuration_vicsek.yaml` carries the source-parity vector:
`n_preys = 100, n_predators = 1, R_inf = 5, R_avoid = 1, R_pred = 5,
v = v_pred = 1, Δt = 1, D = 0.8, η = 0.8, w_afraid = 3,
detect_ratio = 1.5, predator_noise_ratio = 0.2`, domain 40³ (source
half-width W = 20). *tests:* preset sentinel values load as written;
order transition — settled α(η = 0.95, D = 0.05) > 0.8 AND
α(η = 0.05, D = 2.0) < 0.3 at N = 200 after 300 settle steps (both
phase-diagram corners behave; complements S1.7's D-liveness test);
vicsek golden re-pinned with S1.7 in the same commit.
**Status: PARTIAL** — the preset file exists; **verify** its vector
matches (defaults differ: `R_pred = 80`, `v_pred = 2`,
`predator_noise_ratio = 0.1`, `Δt = 0.1`); no `n_predators` mechanism
exists in FlockConfig yet, so the 1-predator population cannot be
declared from YAML.

### Track E — Influencer  *(≈1½ days)*
Files: `physics/forces/influencer.py::InfluencerMode`
(`owns_positions=True`, `speed_mode="fixed"`, instance `tick`); tests
`test/physics/forces/test_influencer.py`.

**S2.E1 Trajectory** — *math (verbatim):*
```
T_raw(t) = ( sin(t/97)·200 + cos(t/217)·30,
             cos((t+53)/29)·200 + sin((47−t)/13)·30,
             cos((t+61)/41)·100 + sin((t+13)/7)·27 + 40 )
s = cfg.influencer.scale · min(W/460, H/460, D/254)
T(t) = C + (T_raw(t) − (0,0,40))·s + (0, 0, 40s)
```
Persistent tick (one per substep) replaces any random-t teleport.
*Optional path-shaping:* lift the hardcoded coefficients to
`cfg.influencer` fields (defaulting to the verbatim values) —
`traj_primary_amp` (200,200,100 → range of motion),
`traj_secondary_amp` (30,30,27 → local flutter), `traj_periods`
(97,217,29,13,41,7 → looping-vs-wandering character; keep mutually
prime for aperiodicity), `traj_phase` (53,47,61,13 → starting
position), `traj_z_bias` (40 → mean altitude). Each optional; unset =
the canonical path. *tests:* `_target_pos` at t ∈ {0, 970, 2170} equals
hand values (s = 1, 460×460×254 domain, default coefficients);
in-domain for scale ≤ 1; step distance varies; overriding
`traj_primary_amp` scales the path extent proportionally.
**Status: DONE** for the verbatim formula + persistent tick (engine
recomputes the tick from the frame counter each step *and* compute
increments it — redundant double bookkeeping, consolidate on the mode
instance per [roadmap1.md](roadmap1.md) D2). `traj_*` shaping fields
MISSING.

**S2.E2 Move-then-steer at unit speed** — per substep:
`p += d̂·v0·dt` (OLD direction) → recompute `t̂, dist` (guard
`x += (x==0)`) → `d̂ ← normalize(d̂(1−inf) + t̂·inf)` → `tick += 1`.
*tests:* frozen target → convergence to hover/orbit; one-step lag —
after a target jump, headings change only on the following substep;
|v| ≡ v0.
**Status: DIVERGES — flag untruthful.** The current mode runs
`substeps` direction-blend iterations **without moving positions**;
positions advance once per frame in `integrate()` at the final
velocity. That is steer-then-move with collapsed substeps, not
move-then-steer, and `owns_positions=True` is false in practice. Either
implement per-substep movement with `integrate(move=False)`
([roadmap1.md](roadmap1.md) D2/D4 wiring) or re-declare the contract;
the one-step-lag and hover/orbit tests only make sense for the spec
form.

**S2.E3 Influence** — rank by **distance-to-target** (not CoM):
`inf_sorted[i] = (1 − (i/(N−1))·0.8)^rank_exponent` (1.8 → floor
0.2^1.8 ≈ 0.055); distance alternative
`clamp(near_dist_sq·s²/d², 0.2, 0.8)` behind
`cfg.influencer.influence_mode`. *tests:* exactly one bird at 1.0; min
≈ 0.055 ± 1e-3; monotone non-increasing in target distance; exponent
1.0 → linear.
**Status: DONE.**

**S2.E4 Density-scaled init** — Gaussian `σ = N^{1/3}·separation·s`
(sep 0.5) + shared random offset `C + U(0, 10s)³`; **zero initial
directions** (first blend heads every bird at the target, weighted).
*tests:* init density equal across N ∈ {100, 1 000, 8 000} (±10 %);
frame-0 headings ∝ influence toward target.
**Status: PARTIAL — dead code.** `influencer_density_init` exists with
the right law but **nothing calls it** (`PhysicsFlock` uses the generic
`position_init` selector); note the spec adds a *shared* `U(0,10s)³`
offset while the implementation jitters **per-bird** — fix when wiring.
Zero-initial-directions not wired.

**S2.E5 Diagnostics + influencer marker** — per-frame `min/max ‖p − T‖`
→ `FlockMetrics.target_dist_min/max` + window title
(`dT=[{min:.0f},{max:.0f}]` in influencer mode). **Render the
influencer target** (in 3D it is invaluable for debugging): one extra
instance appended to the packed buffer at `T(tick)` with velocity = its
finite-difference direction, flagged through the renderer flag channel
(red/larger, same mechanism as the threat marker). *tests:* CSV
contains both columns, min ≤ max, finite; *(gl)* marker visible and
tracing a smooth curve the flock chases.
**Status: PARTIAL.** The distances are computed onto config privates
(`_target_dist_min/max`) but never reach `FlockMetrics`, the export
schema, or the title; no marker rendering (needs `draw_layer`,
[roadmap1.md](roadmap1.md) D7).

**Track-E signature test (emergent stretching)** — after 500 settled
steps on the shipped preset, the flock's extent along the target's
velocity direction exceeds its mean transverse extent: the leading
eigenvector of the position covariance is roughly parallel to `T'(t)`
(|dot| > 0.7) — the core-leads/tail-lags morphology that is this
model's headline behaviour (`@slow`,
`test/physics/forces/test_influencer.py`).
**Status: MISSING.**

**S2.E6 Pilotable flock** — *math:* a user-steered **pilot point** `P`
with heading `ĥ` replaces the trajectory target when
`cfg.influencer.pilot_enabled`; per bird, with `Δ = P − p`, `d = ‖Δ‖`:
`F = ĥ·align·0.12 + Δ·coh·0.22 + (Δ/d)·(d − shell_radius)·0.42`
(third term is signed — pulls in beyond the shell, pushes out inside
it, so birds orbit a sphere of radius `cfg.influencer.shell_radius·U`
around the pilot rather than collapsing onto it). *impl:* pilot state
on `InfluencerMode` (position + heading), driven through the command
queue from `input_control` keys (arrows/WASD move `P` at
`pilot_speed·U` per second in the camera frame; headless path can
enqueue the same commands — scriptable choreography). *tests*: force
zero-crossing exactly at `d = shell_radius·U` for the radial term;
settled flock's median `‖p − P‖` within ±20 % of the shell radius;
pilot displacement commands move the settled flock centroid in the
commanded direction; disabled ⇒ S2.E1 trajectory unchanged (allclose).
**Status: PARTIAL.** `PilotTarget` with the exact 0.12/0.22/0.42 force
law and shell expand/contract exists on the mode; **no input wiring**
(no WASD/command-queue path, no `pilot_enabled/pilot_speed` config
fields) — currently unreachable in normal runs.
