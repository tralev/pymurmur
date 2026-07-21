# roadmap3.md — Metrics & analysis (S3) + 3D rendering & capture (S4)

All visualization is strictly 3D ([roadmap0.md](roadmap0.md) §1). Gates:
S3 needs D3, S4 needs D7 ([roadmap1.md](roadmap1.md)). Physics
prerequisites: [roadmap2.md](roadmap2.md). UX/EvoFlock/MARL:
[roadmap4.md](roadmap4.md). Bug/divergence register:
[roadmap5.md](roadmap5.md).

---

## S3 — Metrics & analysis suite  *(≈2 days; after D3)*
Files: `analysis/metrics.py`, `analysis/rewards.py`,
`analysis/phase_diagram.py`, `analysis/density_scaling.py`; tests
`test/analysis/`.

**S3.1 Nematic order** — `Q = (3/2)(ûᵀû)/N − ½I` (3×3, traceless),
`S = λ_max(Q)`; O(N); `order: polar|nematic` option in the
phase-diagram sweep. *tests:* two anti-parallel half-flocks → α < 0.05,
S > 0.95; isotropic 500 birds → both < 0.15; S invariant under per-bird
û → −û and SO(3).
Also add a `quick=True` **snapshot mode** to `phase_diagram.sweep`: one
single-step angle update per (η, D) grid cell on the current
configuration instead of a full settled run — ~200× cheaper, for
interactive parameter-space exploration (the settled-run path stays the
default and the scientifically correct one). *tests:* `quick=True`
returns the same grid shape as the settled sweep and runs in a fraction
of the time.
**Status: MOSTLY DONE** — nematic S and the `order_type` sweep option
are implemented; the `quick=True` snapshot mode is MISSING.

**S3.2 MSD(τ) curve** — unwrapped accumulation
`p_unwrap += min_image(p_t − p_{t−1})`; per-lag
`MSD(τ) = ⟨‖p_unwrap(t+τ) − p_unwrap(t)‖²⟩` at log-spaced lags
{1,2,4,…,64}; crossover `τ_cross: d log MSD/d log τ ≈ 1.5`. *tests:*
D = 0 aligned flight → slope 2.0 ± 0.1 all lags; strong-noise walkers →
1.0 ± 0.2 for τ ≥ 4; seam crossing contributes MSD(1) = (v dt)² ± 1e-4.
**Status: DONE** (`compute_msd_curve` with unwrapping, log-spaced lags,
slope fit, crossover detection).

**S3.3 Shape→m\*** —
`m* = 9.78 + clamp((aspect−1)/2, 0, 1)·(6.05 − 9.78)`; `suggested_m`
field on `FlockMetrics`. *tests:* endpoints (aspect 1 → 9.78, ≥ 3 →
6.05); monotone; thin flock ≤ 7, round ≥ 8.
**Status: DONE.**

**S3.4 η(m)** — `η(m) = (H₂(m₀) − H₂(m))/(m − m₀)`; +∞ when m first
connects the graph; 0.0 when both disconnected (needs S1.8's `inf`,
[roadmap2.md](roadmap2.md)). *tests:* connectivity transition →
`math.isinf`; both disconnected → 0.0; telescoping sum property.
**Status: DONE** (`_compute_eta_m` with the m₀ = max(2, m*−2)
baseline).

**S3.5 Hull-volume τρ** —
`ρ = N / ConvexHull(positions).volume` (0 if degenerate/< 4 points);
ring buffer (sample every 10 frames, 500 slots);
`τ = interval·(0.5 + Σ_{lag≥1} r(lag))`, stop at first `r ≤ 0` **or at
lag = 0.25·buffer, whichever comes first** (keeps τ finite on
slowly-varying series that never cross zero). *tests:* cube hull =
edge³ ± 1e-3; coplanar → 0; constant series → τ == 0 (not NaN);
period-P oscillation → τ ∈ [P/6, P].
**Status: MOSTLY DONE** — hull density, 500-slot ring at 10-frame
cadence, and the trapezoid-style τ sum are implemented. DIVERGES on the
stop cap: current `max_lag = min(n−1, 20)` vs the spec's
`0.25·buffer = 125` — pick one and pin the period-P test.

**S3.6 Θ′ silhouette** — project positions ⊥ an observer axis,
rasterize disks of radius `boid_size`, coverage = union fraction
(overlaps counted once); **additional field** beside the voxel Θ′.
*tests:* flat wall ⊥ axis → silhouette ≈ 1 while voxel Θ′ ≪ 1; two
co-projected birds == one.
**Status: DONE** (`compute_silhouette_2d`; note it uses a fixed default
`boid_size = 5.0` rather than `cfg.flock.boid_size` — wire the config
value).

**S3.6a Marginal-opacity validation** — the projection model's
*raison d'être*: a flock steering on δ̂ self-regulates its density to
**marginal opacity**. Reference (documented, not runtime-enforced):
silhouette `Θ′ ~ N(µ=0.30, σ²=0.059)` fitted across 118 real flocks.
*impl:* module constants `MARGINAL_OPACITY_MEAN = 0.30`,
`MARGINAL_OPACITY_STD = 0.243` in `analysis/metrics.py`; **no new
physics** — consumes the S3.6 *silhouette* Θ′ (not the voxel metric).
*tests* (`test/analysis/test_marginal_opacity.py`, `@slow`):
**scientific regression** — a seeded projection-mode run (N≈150, ~300
settle frames) has time-averaged silhouette Θ′ ∈ [0.05, 0.55] (a loose
µ±3σ band; the claim is "marginally opaque", not exactly 0.30 —
domain/N/φ shift the operating point); if a future physics change
breaks self-regulation, δ̂ or the φ weights are wrong. Plus a
constants-documented guard.
**Status: MISSING.**

**S3.7 Robust gyration + number density + ideal exponent** — **median**
centroid; one-sided top-15 % trim (`keep = 0.85`);
`R_g = √mean(r²_kept)`; `ρ = N_kept / ((4/3)πR_g³)`; the
density-scaling sweep reports `ideal_density_exponent = −0.5` beside
fitted β (keep toroidal-vs-open + R² framing). *tests:* one 10 000-unit
outlier moves R_g < 5 %; degenerate flock density → 0; sweep carries
`−0.5` and finite β.
**Status: DONE** (`compute_gyration`, `compute_robust_density`,
`DensityScalingResult.ideal_density_exponent = −0.5`).

**S3.8 Motion metrics** —
`velocity_deviation = (1/N)Σ‖v̄ − v_i‖` (catches speed dispersion α
misses); `boundary_overshoot = Σ max(0, ‖p−C‖ − R_dom)`;
`altitude_deviation = (1/N)Σ|z_i − z_target|`
(`z_target` = `cfg.metrics.altitude_target`, default domain-centre z) —
a strictly-3D observable that pairs with roosting/ecology and feeds the
reward module's altitude term (S3.9); normalized angular momentum
`‖⟨r×v⟩‖/(v0·R_g)`; L about CoM with mass
([roadmap2.md](roadmap2.md) S2.B4). *tests:* equal headings + mixed
speeds → deviation > 0 while α == 1; overshoot 0 inside, > 0 for
planted outliers; altitude_deviation 0 for a flat sheet at `z_target`,
grows with vertical spread; L translation-invariant; normalized L O(1)
across ×10 domain scale (±10 %).
**Status: MOSTLY DONE.** All four metrics exist. Fix: `altitude_target`
is not a config field — the code falls back to a hardcoded 500.0 (via a
non-existent `config.roost.z_target` read); default should be the
domain-centre z.

**S3.9 Rewards module** — `analysis/rewards.py`:
`compute_reward(flock, config) → float`, pure numpy, no gym dependency
— the weighted composite (weights = `cfg.marl.reward_*`, defaults
w_a = w_c = 1, extension terms 0):
```
R = ±w_a·velocity_deviation − w_c·dispersion          (core two terms)
    − w_L·‖Σᵢ (pᵢ−CoM)×vᵢ‖/N                          (excess-rotation penalty)
    − w_b·boundary_overshoot − w_z·altitude_deviation  (containment, altitude)
```
core sign: `+w_a` under `reward_faithful_signs=True` (the source's
quirk — the agent trades deviation against compactness), `−w_a`
corrected (both negative, maximum 0 at perfect order); every term is an
S3.8/existing metric — no new physics. Shared by MARL
([roadmap4.md](roadmap4.md) S7) and EvoFlock scalarization. *tests:*
perfect flock → corrected reward 0 (max); faithful flag flips the
alignment sign; per-term weight linearity (doubling one weight doubles
exactly that term's contribution).
**Status: DIVERGES.** A rewards module exists but implements a
different contract: nine bonus-shaped terms normalised via `1/(1+x)`
transforms, `faithful_signs` negating the *whole* reward, plus
baseline/clip. The spec's five-term penalty composite (max 0 at
perfection, per-term sign semantics) is required by the MARL bridge
tests. Decide: either replace with the spec formula, or keep the
current shaping and rewrite S7's reward expectations — do not leave
both undefined. The weight-linearity test exists conceptually
(`reward_linearity_check`) and carries over either way.

**S3.10 Export schema** — `FlockMetrics.to_dict()` adopted end-to-end;
new fields (suggested_m, nematic, msd_curve, target_dist_*, *_real)
included. **Mode-gated observables export honestly**: fields only a
specific mode populates (Θ/Θ′ from projection's `last_theta`,
`target_dist_*` from influencer) are `None`/absent in other modes —
never stale zeros masquerading as measurements (0.0 reads as "perfectly
transparent", not "not measured"). *tests:* JSON round-trip; pinned key
set; Recorder CSV headers == schema; a spatial-mode run exports
`theta = None` (CSV: empty cell), a projection-mode run exports a
float.
**Status: MOSTLY DONE.** `to_dict()` (ndarray→list, NaN/inf→None) is
consumed by the Recorder; Θ is NaN→None outside projection mode ✓.
Missing: `target_dist_min/max` fields on `FlockMetrics`
([roadmap2.md](roadmap2.md) S2.E5) and the pinned-key-set test.

**S3.11 EMA-smoothed readout** — the HUD/title/console readout smooths
the fast fields (α, Θ, Θ′, L, σ_r) with a one-pole EMA so it is stable
frame-to-frame: `ema ← ema + s·(raw − ema)`,
`cfg.metrics.readout_smooth: float = 0.04` (0 = raw). **Strict
display/analysis separation** (modularity): smoothing is *display-only*
— `to_dict()`, CSV/JSON export, and every science/validation path keep
**raw** per-frame values; never smooth what you later analyse. *impl:*
a `MetricsReadout` view (or `smoothed` property) on `MetricsCollector`
updated in `collect()`; consumed only by the title
([roadmap4.md](roadmap4.md) S5.2) and run log
([roadmap4.md](roadmap4.md) S5.6); Recorder stays on raw. *tests*
(`test/analysis/test_metrics_readout.py`): EMA converges to a constant
raw stream and approaches a step monotonically without overshoot;
`readout_smooth=0` → passthrough; `to_dict()` values equal the raw
snapshot even while the readout lags (raw/display separation asserted).
**Status: MISSING.**

---

## S4 — Rendering & capture  *(≈4 days; after D7)*
Files: `viz/{renderer,shaders,camera,hud,visualizer}.py`,
`capture/{recorder,mpl_recorder}.py`; tests `test/viz/`,
`test/capture/` (`@gl` unless noted). Everything here is 3D: instanced
meshes/impostors under a 3D camera, depth-tested, dual 3D viewports,
3D capture.

**S4.1 Sphere impostors** — billboard quads; fragment: `p = uv·2−1`,
`r² = p·p`, discard > 1, `z = √(1−r²)`, `shade = 0.55+0.45z`,
`color = mix(paper, ink, shade(1−0.22·rim))`; `cfg.viz.point_sprites`.
*tests:* shader compiles; centre pixel brighter than rim; corners =
background.
**Status: DONE.**

**S4.2 Depth cues** — size ∝ 1/depth^k; alpha ×
`mix(1, 1−depth01, fade) · mix(0.65, 1, speed01) ·
mix(1, 0.76, ss(0.72, 1, r²))`. Plus **Fresnel rim lighting** (distinct
from the impostor-disc rim) — the 3D generalisation of a 1-px outline:
`rim = pow(1 − max(N·V, 0), k)` (k ≈ 2–3), added to the **mesh**
fragment shader as a view-angle silhouette highlight (a depth/shape cue
that reads on solid meshes where the disc-`r²` rim does not apply).
*tests:* near bird renders larger & more opaque than far
(pixel-area/alpha probe); edge-on mesh pixels (low N·V) brighter than
face-on.
**Status: PARTIAL** — impostor depth scaling + the three-factor alpha
and an impostor Fresnel rim are implemented; the **mesh-shader** Fresnel
rim is MISSING.

**S4.3 Trails ×4** (`cfg.viz.trails`) — *velocity:* impostor stretched
along `proj(p) − proj(p − v·len·0.12)`; head `max(0.28, 1/(1+2.8s))`;
tail `0.22+1.35s`; wave `sin(prog(5.4+3.4·speed01)+seed)·wav·s·0.18`.
*accumulation:* fade quad at
`clamp(0.24 − 0.19·persist − 0.09·vis, 0.018, 0.32)`
(persist = clamp(len/5), vis = clamp(opacity)); depth-only clear, then
draw. *ring:* K = `trail_length` past positions (from `prev_positions`
lineage) as shrinking/fading sprites. *lines (CPU fallback —
shader-free, the natural first implementation and the S4.10 ladder's
cheap tier):* 5 segments per bird traced backward along velocity,
segment span `trailScale = 0.1·trail_length` (vertex k at
`p − v̂·trailScale·prog`, `prog = k/5`); ribbon wave displaces vertices
along the camera-plane perpendicular `(−v_y, v_x, 0)/√(v_x²+v_y²)`
(z-up; fall back to `(1,0,0)` when v_x = v_y = 0) by
`sin(prog·2π·2.6 + seed)·waveScale·prog²` — amplitude vanishing at the
head; one `LINES` VAO of 2·5 vertices per bird, CPU-filled each frame
from positions/velocities, drawn depth-test-off with uniform alpha
through the flat-colour pipeline. *tests:* velocity — lit extent along
motion > ⊥; accumulation — persists ≈ 1/fadeOpacity frames, clears when
paused; ring — K sprites monotone size/alpha; lines — buffer holds
exactly 10 vertices/bird, head-vertex wave displacement zero, segment
extent anti-parallel to v, degenerate vertical-v bird produces finite
vertices; `off` pixel-identical baseline.
**Status: MOSTLY DONE / VERIFY.** All four modes exist
(`viz/trails.py`). Divergences to reconcile with the spec: velocity
trails lack the head/tail/wave shaping factors; accumulation decay is a
fixed 0.97 texture fade (no persist/vis-driven fade-alpha formula);
lines mode uses up to 20 segments/bird with `sin(t·3π)` wave and no
`prog²` head-vanishing — the spec's 5-segment/10-vertex layout and its
tests don't match. Ring mode: size/alpha are uniform (0.5), not
monotone-fading. Bless or align, then pin.

**S4.4 Winged mesh + flap** — 6-triangle body+wings+tail; mesh space
forward = +Z, up = +Y, wingspan ±8 on X (source-verified vertex table):
```
T  = ( 0.0, 0.0,  3.0)  body tip        WL = (−8.0, 0.0, −1.0)  L wing tip (flap 1.0)
B1 = (−1.0, 0.0, −3.0)  back left       WR = ( 8.0, 0.0, −1.0)  R wing tip (flap 1.0)
B2 = ( 1.0, 0.0, −3.0)  back right      RL = (−0.8, 0.1,  0.8)  L wing root
B3 = ( 0.0, 1.0, −3.0)  back top        RR = ( 0.8, 0.1,  0.8)  R wing root
faces: (T,B1,B2) (T,B2,B3) (T,B3,B1) body · (B1,B3,B2) tail cap ·
       (RL,WL,B1) (RR,B2,WR) wings          (flat face normals suffice)
```
per-vertex flap weight (1.0 at WL/WR, else 0);
`u_Flap = ±0.5` toggled every `⌊frame/flap_period⌋` (period 100);
applied to mesh-y **before** the LookAt rotation (local-up flap; the
renderer receives `sim.frame` via `begin_frame`). *tests:* geometry
counts; tip y toggles at exact frame boundaries; bird flying +z flaps
in world-xy.
**Status: DIVERGES.** A winged mesh + flap exists but with a
**different 7-vertex table** (nose/body/wings/tail, wingspan ±0.65) and
a **continuous sine** flap (`sin(frame/100·2π)`, weights ±0.5) instead
of the ±0.5 square-wave toggle at `⌊frame/flap_period⌋`; no
`flap_period` config. Decide: adopt the source-verified table + toggle
(then the exact-boundary test applies) or bless the current mesh and
rewrite the tests.

**S4.4a Mesh-registry entries + theme material sets** — the
`cfg.viz.bird_mesh` selector + `shaders.py` mesh table is the extension
seam:
- **Speed-stretched ellipsoid**: an ellipsoid mesh scaled along the
  velocity axis by the speed ratio — one extra per-instance factor
  `(1, 1, clamp(|v|/v0, lo, hi))` applied in the existing LookAt vertex
  shader before rotation; a cheap motion cue that reads even with
  trails off.
- **Cone / arrow procedural meshes**: registry entries beside
  `tetra | winged | impostor | ellipsoid` — proves the mesh table is
  truly pluggable (no shader branching; one entry each).
- **`points` render tier + count-based recommendation**: a raw
  `GL_POINTS` registry entry (per-instance position + hue only,
  `gl_PointSize ∝ boid_size/depth`, flat colour — the cheapest tier for
  very large N), and a pure function
  `recommend_render_mode(n) → "instanced" | "impostor" | "points"`
  (thresholds ≈ n ≤ 10 k instanced meshes, ≤ 60 k impostors, above →
  points) logged at startup and available to the S4.10 governor as an
  optional ladder step after count reduction. (These CPU render tiers
  are the accepted scaling path — GPU-compute *simulation* backends
  stay excluded, [roadmap5.md](roadmap5.md) Appendix A.)
- **Theme-driven material sets**: promote the Blinn-Phong
  `ambient`/`diffuse` pair from hardcoded constants to a per-theme
  table, driven by `cfg.viz.theme`, so mesh shading matches the scheme.
*tests:* each registered mesh
(`tetra|winged|impostor|ellipsoid|cone|arrow|points`) renders one frame
without GL error (`@gl` smoke); a bird at 2·v0 renders longer along its
heading than at 0.5·v0 (ellipsoid stretch); switching `theme` changes
the sampled mesh ambient/diffuse (pixel probe); `recommend_render_mode`
pinned at the three thresholds and monotone (pure logic, no GL).
**Status: PARTIAL** — theme material tables (per-theme
ambient/diffuse) are DONE; ellipsoid/cone/arrow/points registry entries,
the `bird_mesh` selector (current selection is two booleans
`winged_mesh`/`point_sprites`), and `recommend_render_mode` are
MISSING.

**S4.5 Gradient sky** — fullscreen quad, top (0.60, 1, 1) → bottom
(0.686, 0.933, 0.933), theme-overridable
(`cfg.viz.background_top/bottom`); drawn first, depth off. *tests:*
top/bottom row pixel colours; flat mode unchanged.
**Status: PARTIAL** — a gradient sky quad is drawn (depth off, first),
but colours are derived from the theme clear colour (×1.3) — the
documented default colours and the `background_top/bottom` override
fields are MISSING.

**S4.6 Colour channels** — per-bird hue from `seeds` (HSV h = seed·360,
S = V = 0.9) via the schema hue float; predator flag → red, ×1.3–1.5
scale; heading-hue debug theme (azimuth → hue). *tests:* hue stable
across frames; predator red in all themes; +x vs −x flight differs ≈
180° in hue.
**Status: MOSTLY DONE** — seed hue + predator red/×1.35 scale are in
the shaders (S=0.7, V=1.0 — minor constant drift); the heading-hue
debug theme is MISSING.

**S4.7 Alpha-accumulation density mode** — α ≈ 0.2 sprites, blending
on, depth-write off. *tests:* cluster centre darker than single bird.
*Optional exotic variant (low priority):* a **64³ volumetric
accumulation texture** — splat bird positions into a low-res 3D density
grid, apply a per-frame 3D blur+fade, and raymarch it as a volumetric
overlay for a true smoke-like density field. Behind
`cfg.viz.trails: "volumetric"`; GL 4.3 compute or a slice-stack
fallback. *tests:* occupied voxel fraction tracks flock compactness;
fade decays a static splat over ~N frames.
**Status: DONE** (density mode with alpha/depth-write handling);
volumetric variant not started (optional).

**S4.8 Views** — dual-viewport (elev/azim 15°/15° + 45°/45°, two
camera/viewport passes — exposes planar flocks); orthographic top/side
presets (keys 7/8/9); fixed capture framing option. *tests:* halves
differ and both contain birds; ortho — equal pixel size at different
depths.
**Status: DONE** (dual view + ortho presets); fixed capture framing
option not explicit (sweep off ≈ fixed — verify).

**S4.9 Capture pipeline** — cinematic sweep (`azim = 45°+180°t`,
`elev = 25°+0.15 sin 2πt`, `dist = (650+100 sin 1.5πt)·scale`);
pre-warm (`capture.prewarm = 60` un-captured settle frames);
`CAPTURE_W/H/FRAMES/OUT` env overrides (YAML < env < CLI); GIF
`optimize=True, disposal=2`; matplotlib GPU-free fallback
(`capture/mpl_recorder.py`, dual-view 3D scatter → GIF) replacing
silent frame loss. *tests:* folded into
[roadmap1.md](roadmap1.md) D7-T5.4/5.5 + first-captured-frame
dispersion < unwarmed frame-0.
**Status: DONE with one defect** — sweep formula, pre-warm, env
overrides, GIF flags, and the mpl dual-view fallback (with warning) are
all implemented; the override precedence is **env > CLI** (spec:
YAML < env < CLI) — flip it.

**S4.10 Adaptive quality** — EMA
`avg = 0.92·avg + 0.08·min(250, frame_ms)` (spike-capped, floor 0.01);
budget `1000/max(24, target_fps)`; healthy if `avg ≤ 1.12·budget`; risk
rules → classification: *cpu* — python force path at high N
(field/spatial modes, numba off); *vertex* — N > 30 000; *fragment* —
trails on or very large window/pixel ratio; more than one risk (or
none) → *mixed*; ladder (trails off → render scale −0.15 floor 0.75 →
N −18 % floor 512) when fps < 78 % of target for ≥ 1.8 s, one step per
1.8 s. *tests:* synthetic series → ladder order, spacing, per-action
effects, recovery stop; classifier pinned on the four rule combinations
(pure logic, no GL).
**Status: PARTIAL, unwired.** `QualityGovernor` implements the EMA,
spike cap, budget, 0.78/1.8 s degrade, 3.6 s recovery, and the 3-step
ladder; `Visualizer._apply_quality_actions` applies them. But the
render loop **never feeds the governor** — the whole feature is inert
([roadmap1.md](roadmap1.md) D8) — and the cpu/vertex/fragment/mixed
risk classifier is MISSING (only a cpu/gpu bottleneck ratio exists in
`PerfDiagnostics`).

**S4.11 Fixed-timestep accumulator + interpolation** —
`acc += clamp(frame_dt, 0, 1/20); while acc ≥ dt_phys: step(dt_phys)`;
optional render lerp `prev_positions → positions` by `acc/dt_phys`.
*tests:* same seed at simulated 30 vs 60 fps → identical physics;
rendered position between the two states.
**Status: DONE** (engine accumulator + `render_positions` lerp consumed
by the visualizer).
