# roadmap4.md — UX & tooling (S5), EvoFlock (S6), MARL bridge (S7), sequencing

Gates: S5 needs D8; S6 needs S2-B + S6.4; S7 needs D8 + S3.9
([roadmap1.md](roadmap1.md), [roadmap2.md](roadmap2.md),
[roadmap3.md](roadmap3.md)). Bug/divergence register:
[roadmap5.md](roadmap5.md).

---

## S5 — UX & tooling  *(≈2 days; after D8)*
Files: `analysis/presets.py`, `viz/{input_control,hud,visualizer}.py`,
`__main__.py`, `__init__.py`; tests `test/viz/test_input.py`,
`test/test_cli.py` (no GL).

**S5.1 Preset keys a–h,w** — the exact table
(key: φp/φa/σ/mode — label):
a: 0.04/0.80/6/proj — *3D Pearce Default* · b: 0.18/0.70/7/proj —
*Ball of Birds* · c: 0.06/0.45/3/proj — *Storm Cloud* ·
d: 0.25/0.55/8/spatial — *3D Stream* · e: 0.10/0.75/6/proj —
*Vertical Column* · f: 0.02/0.85/3/proj — *3D Acro* ·
w: 0.08/0.82/10/spatial — *Spiral Vortex* · h: 0.35/0.58/9/spatial —
*3D Void*. `analysis/presets.py` holds this table (fields: label,
phi_p, phi_a, sigma, mode, description) +
`apply_preset(config, key) → label`, printed on apply; key range skips
`g` (grid). *tests:* synthetic KEYDOWN 'b' → config equals the row;
'g' still toggles grid; label + description printed (capsys).
**Status: ✅ DONE (verified Phase 5, S5 track — roadmap status was
stale).** `LETTER_PRESETS` labels already match the spec table exactly,
and `apply_preset(config, key)` was already extracted as a standalone
function. The `c`-key/clear-all collision was already fixed (Phase 1).

**S5.2 Full title readout** — mode, N, φp/φa/σ, `α Θ Θ' L σr`, τρ, FPS
(+ physical units), rebuilt every 20th frame, consuming the S3.11
smoothed readout ([roadmap3.md](roadmap3.md)). *tests:* token presence;
cadence.
**Status: ✅ DONE (verified Phase 5, S5 track — roadmap status was
stale).** Already rebuilt every 20th frame, using
`MetricsCollector.smoothed()` (S3.11), with φp/φa/σ and physical units
included.

**S5.3 Slider HUD** — 5 sliders: sep 1–5 (3.0) →
`spatial.separation_weight`; coh 0–2 (0.2); align 0–0.5 (0.02); avoid
0–1 (0.05) → `boundary.avoidance_factor`; noise 0–0.5 (0.05).
Ortho-pass track + knob quads;
`value(mx) = low + (high−low)·clamp((mx−x0)/w, 0, 1)`; hit-rect ±6 px;
drag locks (suppresses orbit); TAB toggles panel. *tests:* mapping
endpoints/midpoint pinned; drag writes the bound nested field; TAB
restores orbit-drag.
**Status: ✅ DONE (verified Phase 5, S5 track — roadmap status was
stale).** `pymurmur/viz/hud.py::SliderHUD` already implements the exact
5 sliders (sep/coh/align/avoid/noise) with matching ranges/defaults,
ortho-pass track+knob quads, hit-rect, drag-lock (suppresses orbit),
and TAB toggle — fully wired into `Visualizer`.

**S5.4 Interaction** — mouse spawn via cursor-ray unprojection:
`ndc = (2mx/w−1, 1−2my/h)`; `ray_eye = P⁻¹·(ndc, −1, 1)` with
`(x, y, −1, 0)`; `r̂ = normalize((V⁻¹·ray_eye).xyz)`;
`depth = median((p_i − o)·f̂)`; `spawn = o + r̂·depth/(r̂·f̂)` →
`SpawnAt` command; right-click → predator; `C` clear; `Q` quit;
click-vs-drag disambiguation — spawn on click (down+up within a 5 px
movement threshold), orbit on drag, so left-drag camera control and
left-click spawning coexist; spawned birds get the S2.B9 `cube`
velocity `limit3((U³−0.5)·2v0, v0)` ([roadmap2.md](roadmap2.md));
PageUp/Dn: `flock.v0 ± 0.1` floor 0.3 (live). *tests:*
synthetic-camera unprojection hand-computed; right-click sets
is_predator; C survives metrics; a 4 px down-move-up sequence spawns, a
20 px one orbits; PageUp floor respected.
**Status: ✅ DONE (Phase 5, S5 track).** `C`-clear branch and `K_c`
collision already fixed (Phase 1). Spawn velocity already implements
the cube-law `limit3((U³−0.5)·2v0, v0)` with `v0` threaded from config
(tagged D20) — not a hardcoded ×4.0 as this entry described. **Fixed:**
`OrbitCamera.screen_to_world` now intersects the median-flock-depth
plane (`depth = median((p_i−o)·f̂)`) when live positions are available,
instead of the fixed `Z=target.z` plane — falls back to the old
behaviour when no positions are given (back-compat preserved for
callers without flock context). `Q` is deliberately bound to camera
roll, not quit (explicit "quit via ESC only" comment, tagged S2.E6) — a
blessed intentional divergence from this item's "Q quit" ask, left
as-is.

**S5.5 CLI + facade** — repeatable `--set key.subkey=value` typed
against the nested schema + `--print-config`; `--fullscreen`;
`--light-scheme` → theme; `pymurmur.Simulation(**params)`;
`benchmark(flock_size, num_steps) → list[float]` (perf_counter).
*tests:* `--set spatial.separation_weight=6 --set flock.num_boids=500`
reflected in `--print-config`; unknown key exits with the field list;
facade benchmark returns 20 positive floats.
**Status: ✅ DONE (Phase 1 + Phase 5, S5 track).** `--set` (dotted +
flat), `--print-config`, `--probe`, `--list-configs`, facade + benchmark,
and `--fullscreen` application were already correct (Phase 1). **Fixed:**
`--light-scheme` crashed on use (`cfg.theme = "light"` isn't a valid
theme) — now maps to `"paper"`. The flat (non-dotted) `--set key=value`
path had zero validation (a typo'd key silently became a stray unused
attribute); now validates against the full known-field set and exits(1)
with the complete field list, matching the dotted-key path's existing
behaviour.

**S5.6 Run logging to `output/`** — every run (visual or `--no-viz`)
writes a structured log `output/run-<UTC-timestamp>.log` via stdlib
`logging`: run header (resolved config echo = `--print-config` content,
seed, mode, N, package version), one metrics line every
`metrics.interval` frames (the `FlockMetrics.to_dict()` payload of the
fast fields), lifecycle events (commands drained, mode switches,
governor actions, golden-relevant resets), and a run footer (frames,
wall time, mean step ms). `--log-level {debug,info,warning}` CLI flag;
viz-only console echo at warning+. **No `print()` calls anywhere in
`pymurmur/`** (architecture-test rule — presets/input currently print;
route through `logging`). *impl:* `core/logging.py` (setup helper),
wired in `__main__.py`; engine/extension call sites swap `print` →
`logger`. *tests* (`test/test_cli.py`): a 30-frame headless run creates
the file with header/footer and ≥ 1 metrics line; `--log-level debug`
increases line count; AST guard — no `print(` in package sources.
**Status: ✅ DONE (verified Phase 5, S5 track — roadmap status was
stale).** `core/logging.py` already fully implements header/per-interval-
metrics/lifecycle/footer logging, `--log-level`, and `print()` →
`cli_out`/`cli_err` replacement with an existing AST guard test — zero
real `print(` calls remain anywhere in `pymurmur/`.

---

## S6 — EvoFlock  *(≈3 days; after S2-B + S6.4)*
Files: `analysis/evoflock.py`, `physics/obstacles.py`; tests
`test/analysis/test_evoflock.py`.

**S6.1 SSGA fidelity** — per update: select 3 → **evaluate all 3**
(fitness cache keyed on genome) → sort → **delete the worst of the 3**
(negative selection) → **uniform crossover** of the best two (each gene
from a random parent) + per-gene Gaussian mutation → insert in the
freed slot. Founders evaluated. *tests:* worst-of-3 gone; child mixes
genes from both parents (disjoint-value parents); all three finite
fitness; cache prevents re-simulation (call counter).
**Status: DIVERGES.** The current SSGA selects a tournament winner,
mutates it (no crossover), and replaces the island's global worst if
the child is better; candidates are not (re-)evaluated at selection,
founders start at fitness −inf unevaluated, and there is no fitness
cache. Implement the spec update rule (or formally re-scope — but the
crossover and cache tests are the point of this item).

**S6.2 Worst-of-4 evaluation** — 4 sims per candidate, fixed per-sim
seeds, min-reduction (`eval_parallel` live; deterministic order).
*tests:* monkeypatched objectives [0.9, 0.8, 0.95, 0.7] → fitness 0.7;
seeds recorded.
**Status: MISSING** (single evaluation per genome; `eval_parallel`
config exists but unused).

**S6.3 Objectives** — separation on **nearest-neighbour** distance per
boid-step, trapezoid over body diameters (0 below 2, ramp 2→2.5,
plateau ≤ 4, ramp 4→5, 0 above); speed on `speed_real` band [19, 21]
m/s (ramps [18, 22]); curvature `κ = |v×a|/|v|³` per boid-step,
`score = clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)`; hypervolume
`F = Π max(o_k, 0.01)`. *tests:* trapezoid pinned at d/body ∈ {1.9→0,
2.5→1, 4→1, 5→0}; helix trajectory κ matches analytic ± 2 %; speed uses
`speed_real`; *ramp ablation (`@slow`, optional experiment):* a short
evolution with binary-threshold objectives (1 inside each band, 0
outside) vs the trapezoids, same seeds — trapezoid run reaches ≥ the
binary run's best fitness with lower variance across islands (the
ramps supply selection gradient).
**Status: DIVERGES.** Hypervolume `Π max(o, ε)` matches. But:
separation uses median `local_spacing` with ramp knees (2→4 plateau,
floor 1/ceiling 8) instead of per-boid-step NN distance with the
2/2.5/4/5 trapezoid; speed scores the simulation-unit band [3, 5]
instead of `speed_real` [19, 21] m/s; curvature is approximated by
`dispersion/α` instead of `|v×a|/|v|³` (needs the acceleration stash,
[roadmap1.md](roadmap1.md) D3); the obstacle objective is a
constant-1.0 placeholder. Align to spec.

**S6.4 SDF obstacle layer** — `physics/obstacles.py`: sphere
`‖p−c‖−r`; box `max(|p|−b)` (componentwise); cylinder; union = min,
subtract = max(a, −b); collision when
`sign(SDF(p_old)) ≠ sign(SDF(p_new))`; kinematic correction
`p ← p − SDF(p)·∇SDF/‖∇SDF‖` (numeric gradient ok); per-step collision
counter feeds `(f_cf)^500`. *tests:* SDF signs & surface zeros;
composition; zero-crossing on a straight path; correction lands
|SDF| < 1e-4; behavioural — obstacle course: collisions > 0 with zero
avoidance, ≈ 0 with evolved weights (`@slow`).
**Status: PARTIAL** — the SDF layer itself (primitives, CSG, gradient,
collision detection, kinematic correction) is DONE in
`physics/obstacles.py`; the **engine integration is missing**: no
obstacle scene in the step loop, no per-step collision counter, so
`(f_cf)^500` never gets real data.

**S6.5 Missing behaviours/genes** *(further research directions —
CMA-ES, GP evolution, non-uniform agents, non-reciprocal interactions,
stigmergy — are deliberately excluded; recorded in
[roadmap5.md](roadmap5.md) Appendix A)* — forward force
`w_fwd·sign(v* − |v|)·û`; per-behaviour `max_dist_{sep,align,coh}` and
`angle_{sep,align,coh}` perception cones (cos α ∈ [−1, 1]);
`fly_away_max_dist`; predictive avoidance (`min_time_to_collide`
look-ahead); **static SDF-gradient avoidance** — steer
`w_static·(−∇SDF(p))·max(0, 1 − |SDF(p)|/fly_away_max_dist)` when
within `fly_away_max_dist` of a surface (the reader for the
currently-dead `static_avoid_weight` gene; predictive + static together
retire both dead genes); fixed k = 7 topological neighbours; integer
gene for σ; `flock.speed_min_factor` as a gene; GA-range entries for
pymurmur-native params (`σ` integer, `blind_deg`, `anisotropy`,
`speed_min_factor` — tuning the projection model with the same
harness). *tests:* forward force sign flips around v*; cones exclude
behind-cone birds (hand geometry); static avoidance zero beyond
`fly_away_max_dist`, anti-parallel to ∇SDF inside it; k enforced; σ
integer after decode; every gene in the range table is read by physics
(no dead genes — AST/attribute check mirroring T1.2).
**Status: MISSING** — and the dead-gene problem is live:
`EVOLVABLE_PARAMS` already contains `predictive_avoid_weight` and
`static_avoid_weight`, which **no physics code reads** (they are
`setattr`-ed onto the config and ignored). Until the readers land,
those genes silently waste GA dimensions; the no-dead-genes test is the
guard.

**S6.6 Protocol** — persist best genome + Pareto front + per-run seeds
+ objective scores to `output/evolved.yaml`; ship confined
(enclosure + obstacles) and open (`boundary: open`) evaluation configs.
*tests:* run(n_runs=2) writes the file; **experiment (`@slow`):**
evolve with NO alignment objective on the confined config → best
genome's settled α > 0.5 (the emergent-alignment headline).
**Status: PARTIAL.** Pareto extraction, islands ×4 with 0.05 migration,
hypervolume ε, and `run(n_runs)` exist; **`output/evolved.yaml` is
promised in the docstring but never written** — implement the
persistence (genome + front + seeds + scores). `conf/murmuration_evo.yaml`
exists — **verify** it matches the confined spec; the open variant is
missing.

---

## S7 — MARL bridge  *(≈2 days; after D8 + S3.9)*
Files: `physics/forces/marl.py::MarlMode`, `analysis/gym_env.py`,
`scripts/{train_marl,rollout_marl}.py`; tests
`test/analysis/test_marl.py` (`pytest.importorskip("gymnasium")` where
needed). Unit map: `U = min(W,H,D)/6`, `v_cap = marl.velocity_cap·U`.

**S7.1 "marl" mode (deferred global rules)** — engine order for this
mode: control applies first (D8: `v += a_ext·action_scale·v_cap`,
component clip ±v_cap), **move**, then rules prep the *next* step:
`v += rule_weight·(F_sep(d < separation_radius·U) + (v̄ − v) +
(CoM − p))` with rule_weight 0.01 (global neighbourhood — no radius on
align/cohere). `MarlMode.speed_mode = "none"` — the source dynamics
have no speed floor/ceiling beyond the component clip, so integrate
applies move + boundary only. Conveniences:
`run_headless(controller=...)` accepts a
`Callable[[SimulationEngine], np.ndarray]` supplying `control` per
step; ship `conf/murmuration_marl.yaml` (`mode: marl`,
`flock: {num_boids: 200, seed: 42}`, `boundary: {mode: open}`,
`marl: {action_scale: 0.1, velocity_cap: 0.1, rule_weight: 0.01,
separation_radius: 0.2, episode_steps: 500}` — the source-verified
constants — and `viz: {dual_view: true}`). *tests:* two-step hand trace
shows positions at step k depend on rules from k−1 only; 0.01 scaling;
clip bounds; no band clamp applied (a bird at 0.05·v0 keeps its speed);
preset loads with the documented values.
**Status: MISSING** (no MarlMode, no MarlConfig, no preset; blocked on
the D8 control hook, [roadmap1.md](roadmap1.md)).

**S7.2 Gymnasium wrapper** — lazy import; `MurmurationEnv(config)`:
`observation_space = Box(−1, 1, (6N,))` — `concat((p−C)/3U, v/v_cap)`;
`action_space = Box(−1, 1, (3N,))`; seeded `reset` →
`p ~ C + U(−1,1)³·U`, `v ~ U(−0.1, 0.1)³·U`; truncate at
`marl.episode_steps` (500); reward from S3.9
([roadmap3.md](roadmap3.md)). *tests:*
`gymnasium.utils.env_checker.check_env` passes; obs ∈ [−1,1] over 500
random steps; same seed + same actions → identical obs; truncation at
500.
**Status: MISSING.**

**S7.3 Scripts (dependency-gated)** — `train_marl.py`:
PPO("MlpPolicy"), 5 000 timesteps, save `output/marl_ppo`;
`rollout_marl.py`: 500 deterministic-predict steps → dual-view GIF;
docstring notes the centralized-MLP quadratic scaling and points to
IPPO for large N. *tests:* `@slow`, skip without stable-baselines3 —
200-timestep learn() smoke; rollout GIF ≥ 1 frame; **experiment:**
trained policy's mean dispersion < random policy's by ≥ 20 %.
**Status: MISSING.**

---

## Unified sequencing

```
D0 ─► D1 ─► D2 ─► D3 ─► D4 ─┬─► D5 ─┐
                            ├─► D6 ─┤
                            ├─► D7 ─┼─► D8 ─► D9
                            └───────┘
S1 (after D2) ─► S2 tracks A–E (parallel, after their gates)
S3 (after D3) · S4 (after D7) · S5 (after D8)
S6 (after S2-B + S6.4) · S7 (after D8 + S3.9)
```

| Phase | Days | Notes |
|-------|------|-------|
| D0–D9 foundation | 10½ | D2 and D5-golden re-pins; D5–D8 parallel after D4 |
| T0–T6 infrastructure | ≈6½ | interleaved with D-phases (not additive serial) |
| S1 correctness | 2 | re-pin projection/spatial/vicsek goldens |
| S2 tracks A/B/C/D/E | 4 / 3½ / 2 / 2 / 1½ | parallel; golden per track |
| S3 metrics | 2 | |
| S4 rendering/capture | 4 | independent of S2 physics tracks |
| S5 UX/tooling | 2 | |
| S6 EvoFlock | 3 | |
| S7 MARL bridge | 2 | |

Original estimate ≈ 38 working days single-track; with two parallel
streams (physics vs rendering/UX/tooling) ≈ 5–6 calendar weeks. **The
current codebase has already landed a substantial fraction** — see the
per-item Status lines and [roadmap5.md](roadmap5.md); the remaining
critical path is: fix the confirmed bugs → resolve the DIVERGES
decisions → land the missing wiring (D2 speed/owns flags, D8 control
hook, governor feed) → complete the MISSING S-items in gate order.
Every feature PR carries its inline test block — the *accept* criterion
**is** the test. If time is short, the highest value-per-day cut:
**bug fixes ([roadmap5.md](roadmap5.md) §1) → D-phase gaps → S1 →
S2-A → S4.9/S4.10 wiring**.

**Definition of done:** T1.2 reports zero orphan config fields;
T4.1/T4.3 matrices green across all registered modes; the golden set
covers every mode; the full non-GL suite runs headless in CI; the two
`@slow` experiments (S6.6 emergent alignment, S7.3
trained-beats-random) pass nightly.

Deliberately excluded scope (screensaver/overlay modes, GPU-compute
backends, flight physics, EvoFlock research directions, multi-flock
scenes, VR/XR) is recorded with its mechanics and math in
[roadmap5.md](roadmap5.md) Appendix A; the documentation-change
checklist that accompanies each landing phase is
[roadmap5.md](roadmap5.md) Appendix B.
