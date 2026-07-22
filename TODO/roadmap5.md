# roadmap5.md — Codebase audit, improvement register, appendixes

Findings from reading the full current `pymurmur/` package against the
contracts in [roadmap0.md](roadmap0.md) and the item specs in
[roadmap1.md](roadmap1.md) / [roadmap2.md](roadmap2.md) /
[roadmap3.md](roadmap3.md) / [roadmap4.md](roadmap4.md). This file
contains no implementation phases: it is the audit bookend of the set —
confirmed defects (§1), implemented-but-unclear items that need a
correctness decision or verification (§2), dead config fields and dead
atoms (§3), a prioritized improvement sequence (§4), the
deliberately-excluded scope with its recorded mechanics and math
(Appendix A), the documentation-change checklist (Appendix B), and a
conclusion (§5).

---

## 1. Confirmed defects (fix first — small, high value)

1. ✅ **FIXED (Phase 1). Sphere boundary is origin-centred.**
   `physics/boid.py::_sphere_soft` measures `‖p‖`, not `‖p − C‖`. With
   the default domain `[0,1000]×[0,700]×[0,400]` most birds are
   permanently "outside" and get hard-projected. Fix, then re-pin the
   `test/data/golden_*_sphere.npz` goldens in the same commit.
   → [roadmap1.md](roadmap1.md) D4.
2. ✅ **FIXED (Phase 1). Wander extension reads non-existent config keys.**
   `physics/extensions/wander.py` reads `cfg.wander_speed` /
   `cfg.attractor_radius`; the config defines `wander_attractor_speed`
   / `wander_attractor_radius`. Both fields are dead; wander silently
   runs at speed 1.0 (configured default: 0.10).
   → [roadmap2.md](roadmap2.md) S2.A1.
3. ✅ **FIXED (Phase 1). `C` (clear-all) key is dead code; `Q` unbound.**
   `viz/input_control.py::_handle_keydown` has two `elif key ==
   pygame.K_c:` branches — the letter-preset "c" branch wins, the
   clear branch is unreachable. Rebind clear (e.g. Shift+C or another
   key) or reorder deliberately; add `Q` quit.
   → [roadmap4.md](roadmap4.md) S5.4.
4. ✅ **FIXED (Phase 1). `--fullscreen` parsed but never applied**
   (`__main__.py` creates the window without checking
   `args.fullscreen`). → [roadmap4.md](roadmap4.md) S5.5.
5. ✅ **FIXED (Phase 1). QualityGovernor is never fed.** `Visualizer.run()`
   neither calls `self._governor.feed(frame_ms)` nor
   `_apply_quality_actions()` — adaptive quality is entirely inert (only
   the `[Qn]` title suffix exists). → [roadmap1.md](roadmap1.md) D8,
   [roadmap3.md](roadmap3.md) S4.10.
6. ✅ **FIXED (Phase 1). Seed semantics broken for 0/None.**
   `PhysicsFlock.__init__`: `np.random.default_rng(config.seed if
   config.seed else 0)` — `seed=0` is conflated with `seed=None`, and
   unseeded runs are silently deterministic with seed 0. Contract:
   integer seeds honored (0 included), `None` → fresh entropy
   ([roadmap0.md](roadmap0.md) §3.2). → [roadmap1.md](roadmap1.md) D3.
7. ✅ **FIXED (Phase 1). `position_init: "sphere"` silently degrades to
   `"box"`.** The validator accepts `"sphere"` but
   `boid.py::init_positions` has no branch for it. Implement the filled
   ∛-law sphere. → [roadmap2.md](roadmap2.md) S2.B9.
8. ✅ **FIXED (Phase 1). Steric clamp inert in production.**
   `steric_force` supports `max_force` but `ProjectionMode` never
   passes it. → [roadmap2.md](roadmap2.md) S1.6.
9. ✅ **FIXED (Phase 1). `noise_force` discards its scale.** The output
   is normalised to unit magnitude, so `noise_scale` only toggles
   on/off instead of setting the kick size.
   → [roadmap2.md](roadmap2.md) S1.5.
10. ✅ **ALREADY CORRECT (verified Phase 3).** ~~Ripple envelope exported
    as a scalar summed over all birds.~~ `extensions/ripple.py` already
    exports the per-bird `(N,)` array (sum over trains only) by the
    time Track A checked — the scalar-sum bug described here was no
    longer present. → [roadmap2.md](roadmap2.md) S2.A5/S2.A6.
11. **PARTIAL — narrower fix landed (Phase 2).** Engine ignores mode
    `speed_mode`/`owns_positions`. `flock.integrate()` always ran
    band-clamped, `move=True`; `InfluencerMode.owns_positions=True` was
    untruthful. Fixed: each mode now declares `speed_mode`/`owns_positions`
    `ClassVar`s, honored by `engine._step_physics`. **Not done:** the
    full `compute()`→stateful `step()` instance-method refactor this
    item originally implied — the narrower fix was sufficient for every
    downstream item that depended on it. → [roadmap1.md](roadmap1.md) D2.
12. ✅ **ALREADY CORRECT (verified Phase 3).** ~~`field_inertia` dead.~~
    `integrate()`'s `inertia` parameter is wired end-to-end (tagged D12
    in the code) — every field preset's inertia column is live.
    → [roadmap2.md](roadmap2.md) S2.A7.
13. ⏳ **OPEN — Phase 6 (EvoFlock) scope, not yet reached.**
    `output/evolved.yaml` never written despite the EvoFlock docstring
    promising it. → [roadmap4.md](roadmap4.md) S6.6.
14. ✅ **FIXED (Phase 2). AngleMode state is class-level.**
    `AngleMode._last_cell` was shared across all engines in the
    process — moved to per-index instance state.
    → [roadmap2.md](roadmap2.md) S2.C6, [roadmap1.md](roadmap1.md) D2.
15. ✅ **FIXED (Phase 2 + Phase 3, Track C). Angle mode has no config
    section.** A full `AngleConfig` dataclass now exists with defaults
    matching the S2.C8 spec table exactly (`max_turn_rate: 200`,
    `turn_threshold: 0.5`, `base_speed: 150`) — the divergent in-code
    defaults this item flagged are gone.
    → [roadmap2.md](roadmap2.md) S2.C7/S2.C8.
16. ⏳ **OPEN — Phase 4 (S4 rendering/capture) scope.** Capture override
    precedence is env > CLI (env vars are read after CLI flags mutate
    the config); contract is YAML < env < CLI.
    → [roadmap3.md](roadmap3.md) S4.9.
17. ✅ **FIXED (Phase 2, D7). Headless FBO has no depth attachment.**
    Headless captures now resolve overlap by depth
    (`depth_attachment=self._depth_rb`), not draw order.
    → [roadmap1.md](roadmap1.md) D7.
18. ✅ **FIXED (tagged D18 in the code, verified Phase 3). Metrics likely
    read zeroed accelerations.** `engine._step_physics` calls
    `metrics.collect()` after `integrate()` resets `accelerations`;
    `force_avg`/`power_avg` now read the `last_accelerations` stash
    instead. → [roadmap2.md](roadmap2.md) S2.B4.
19. ✅ **FIXED (Phase 2). Unbounded accumulators.**
    `MetricsCollector.history` and the Recorder frame list are now
    capacity-capped ring buffers (`cfg.metrics.history_cap`); soak-tested
    to ≥20,000 frames. → [roadmap1.md](roadmap1.md) T6.3.
20. ✅ **FIXED (Phase 1). T4.4 invariance coverage incomplete after
    dedup.** α rotation-invariance, dispersion/gyration
    translation-invariance, permutation invariance, and the
    `[0,1]`-bounds sweep landed in
    `test/l0_modules/analysis/test_metrics_invariance.py`.
    → [roadmap1.md](roadmap1.md) T4.4.
21. ✅ **FIXED (Phase 1). `spawn_at` hardcodes speed 4.0** (ignores
    `config.v0` and `velocity_init`) — the engine now passes
    `v0=self.config.v0`; the `4.0` default remains only as a safety net
    for direct callers that bypass the engine.
    → [roadmap1.md](roadmap1.md) D3, [roadmap4.md](roadmap4.md) S5.4.

## 2. Implemented but unclear — verify or decide, then pin with tests

Where the code deviates from the spec it is *not obvious which is
intended*; each needs an explicit decision (keep the code → rewrite the
spec item and its tests; keep the spec → fix the code and re-pin
goldens). Do not leave both ambiguous.

1. ✅ **DECIDED — adopted spec, fixed (Phase 3). S1.5 force kernels**
   ([roadmap2.md](roadmap2.md)) — separation is now `Σ r̂/d²` with the
   `sum|mean|unit` kernel selector; cohesion is `limit3(·, 1)`;
   alignment is `normalize(v̄−v_i)`.
2. ✅ **DECIDED — adopted spec, fixed (Phase 3, Track A). S2.A3
   leader/chaser** — now has a dedicated `anchor(t,gs)` formula,
   secondary anchor/`sec_mix` blend, per-bird lag, and a real
   `wander_heading(t)` leader target.
3. ✅ **DECIDED — blessed the code (confirmed Phase 3, Track A). S2.A5
   boundary containment** — the inverse-overshoot form
   `−μ r̂/max(d−R, 0.05R)` is kept; the code's deliberate-deviation
   comment was verified present and this was treated as the one
   confirmed intentional divergence in the whole plan.
4. ✅ **DECIDED — code already matched spec (tagged D11, verified Phase 3,
   Track E). S2.E2 influencer substeps** — per-substep move-then-steer
   with `integrate(move=False)` was already implemented; not the
   collapsed single-move form this entry described.
5. ⏳ **OPEN — Phase 4 (S3 metrics) scope.** S3.9 rewards — nine
   `1/(1+x)` bonus terms vs the five-term penalty composite.
6. ✅ **DECIDED — adopted spec, fixed (Phase 3). S1.8 H₂ symmetrization**
   — now `max(A, Aᵀ)`, pinned with a hand-computed 3-node graph test.
7. ✅ **DECIDED — code already matched spec (verified Phase 3, Track B).
   S2.B4 physical power/energy** — `power_real_W`/`energy_J` already
   used the mean-of-per-bird-dot-product and accumulated-work forms by
   the time this pass checked, not the product-of-means/kinetic forms
   this entry described.
8. ✅ **DECIDED — blessed the code (Phase 3, Track B). S2.B5 jitter** —
   kept the multiplicative `w·(1+U(0,j))` form; no functional
   deficiency motivated switching to the additive spec form.
9. ✅ **MOSTLY DECIDED (Phase 3, Track B).** S2.B8 ecology formulas —
   predator-presence hash replaced with the real `PREDATOR_RATE=0.296`
   (deterministic + stochastic-rng paths), coherence-gate window fixed
   to spec, roost force fixed to the `unit(roost−p)` form. Dusk-sigmoid
   parameterization (minutes-before-dusk) kept as a blessed divergence.
   Seasonal-amplitude curve not independently reconciled.
10. ⏳ **OPEN — Phase 4 (S4 rendering) scope.** S4.4 winged mesh —
    different vertex table and continuous-sine flap.
11. ⏳ **OPEN — Phase 4 (S4 rendering) scope.** S4.3 trail shaping.
12. ⏳ **OPEN — not addressed in Phases 1-3.** S2.A6/S2.A1 field-time
    bases — `config._field_time` vs Wander/Ripple's own `self._t`
    clocks can still desynchronize after a reset/mode switch.
13. ✅ **DECIDED (Phase 3, Track D).** S2.D1 vicsek species blend —
    all-predator early-out fixed to a pure random walk (was freezing
    velocities). Fear-blend two-stage form blessed as-is: the spec
    prose doesn't fully pin down the composition, and the current form
    is already pinned by an exact-value test.
14. ⏳ **OPEN — not addressed in Phases 1-3.** S2.A2 seeds — field mode
    still uses `arange(n_active)` rather than the flock's random
    `seeds` column.
15. ⏳ **OPEN — Phase 5 (S5 UX) scope.** S5.4 spawn ray.
16. ⏳ **OPEN — not independently re-verified.** Recorder→MPL fallback
    frame merge.
17. ⏳ **OPEN — Phase 4 (S3 metrics) scope.** S3.5 τρ stop cap.
18. ⏳ **OPEN — Phase 5 (S5 UX) scope.** Preset labels (S5.1).
19. ✅ **DECIDED — adopted the D7 single-schema contract (Phase 2).**
    Instance-buffer memcpy count — `InstanceSchema` merged to the
    8-float single schema; `test_renderer_single_memcpy` re-tightened
    to assert exactly 1 `vbo.write()` call per frame.

## 3. Dead config fields and dead atoms (drift-guard targets)

Fields defined but never read (or read under wrong names) — each needs
a reader or removal; the T1.2 drift guard ([roadmap1.md](roadmap1.md))
should fail on all of these today:

- ✅ **FIXED (Phase 1).** `wander_attractor_speed`, `wander_attractor_radius`
  (wrong-name reads — defect §1.2)
- ✅ **RESOLVED — all consumed (verified Phase 3; tagged C3 in the code,
  meaning these were already wired by the time this pass checked).**
  `field_target_pull`, `field_inertia` (§1.12), `field_drift_direction`,
  `field_ripple_trains`, `field_shell_radius_base`,
  `field_inner_radius_factor` are all read. `field_num_groups`/
  `field_leader_fraction` fixed to be actually consumed (Phase 3, Track
  A, S2.A3 — were hardcoded 7/0.84).
- ✅ **FIXED (Phase 3, Track B).** `alignment_radius_ratio` — now a real
  alignment-radius subset in spatial mode.
- ✅ **FIXED (Phase 2, D5).** `use_toroidal_distance` for the KDTree
  path — `KDTreeIndex` now accepts a `box` param.
- ✅ **RESOLVED — consumed (verified Phase 3; tagged C3).**
  `boundary_radius_factor` — scales the effective sphere radius in
  `flock.py`.
- ✅ **FIXED (Phase 3, Track E).** `influencer_init_separation` — density
  init now auto-triggers via `influencer_density_scaled_init` and
  actually calls it.
- ✅ **VERIFIED CONSUMED (Phase 3, Track B).** `predator_speed_boost` /
  `predator_perception_boost` — confirmed wired in `flock.py::integrate`
  and `spatial.py::_query_neighbors`.
- ⏳ **OPEN — Phase 6 (EvoFlock) scope, not yet reached.** EvoFlock
  genes `predictive_avoid_weight`, `static_avoid_weight` (no physics
  reader — [roadmap4.md](roadmap4.md) S6.5)

Dead atoms (defined, never composed — violates the Micro→Macro rule,
[roadmap0.md](roadmap0.md) §3.3):

- ✅ **FIXED (Phase 3, Track B).** `core/types.py::seed_noise3` →
  composed via `spatial.noise_mode: "seed_sinusoidal"` (S2.B11).
- ✅ **FIXED (verified Phase 3, Track A — already composed by the time
  this pass checked).** `physics/forces/_base.py::ForceTerm` +
  `composeForces` → the S2.A5 composition contract.
- ✅ **FIXED (Phase 3, Track E).** `influencer.py::influencer_density_init`
  / `density_init_positions` → wired via S2.E4.
- ⏳ **OPEN — Phase 6 (EvoFlock) scope, not yet reached.**
  `physics/obstacles.py` (whole layer) → engine integration in
  [roadmap4.md](roadmap4.md) S6.4

## 4. Improvement steps (recommended order)

1. **Land the §1 defect fixes** (1–2 days). Most are one-to-five-line
   changes with an obvious test; re-pin affected goldens in the same
   commits (sphere boundary, noise scale, steric clamp are
   physics-visible).
2. **Tighten the guards so regressions stay caught**: make T1.2 fail on
   every §3 orphan (remove allowlists), restore the deleted invariance
   suite (T4.4), add the soak caps + test (T6.3), and point T1.4 at
   roadmap0–roadmap5. ([roadmap1.md](roadmap1.md))
3. **Resolve the §2 decisions** one by one — each decision is: pick
   spec or code, write/fix the pinned test, re-pin goldens if
   physics-visible, and update the item's Status in
   [roadmap2.md](roadmap2.md)/[roadmap3.md](roadmap3.md)/
   [roadmap4.md](roadmap4.md). Highest-impact first: S1.5 kernels,
   S3.9 rewards (blocks S7), S2.B4 stash (metrics currently
   near-meaningless for force/power), S2.E2 influencer semantics.
4. **Finish the D-phase wiring** ([roadmap1.md](roadmap1.md)):
   D2 stateful modes + speed/owns flags through integrate; D5 protocol
   completion (boxsize, radius-honoring, `query_knn_batch`, delete
   private tree builds); D7 depth attachment + `draw_layer` (threat and
   influencer markers become visible); D8 control hook + governor feed.
5. **Config completion** (D1 remainder): add the missing sections
   (AngleConfig, MarlConfig, ThreatConfig fields), the missing leaves
   listed in [roadmap1.md](roadmap1.md) D1, and decide the flat-map
   retirement question; then rewrite presets to the settled schema and
   add the four missing source-parity presets
   (`murmuration_starlings/boids/angle/marl`).
6. **Complete the MISSING science items in gate order**
   ([roadmap4.md](roadmap4.md) §Sequencing): S1.4 φn → S2 track gaps
   (B7 sphere_soft, B11 flow/noise, C3 speed laws, D3 prey-only
   metrics, E4/E5/E6 wiring) → S3.6a/S3.11 → S4.4a/S4.10 classifier →
   S5.3/S5.6 → S6 SSGA fidelity + obstacle integration → S7 MARL
   bridge.
7. **Performance passes after correctness**: vectorise the per-bird
   Python loops that remain (vicsek fear/hunt, ecology roost pull,
   angle steering at large N), add the numba gates + equivalence tests
   (S2.B10), and only then chase the scaling budgets.
8. **Test-suite and tooling hygiene** (cheap, parallel to any step):
   - Drive `mypy` on `pymurmur/` to zero errors and keep it a CI gate.
   - Run coverage with the existing `[tool.coverage]` config —
     `pytest --cov=pymurmur/analysis --cov=pymurmur/viz
     --cov=pymurmur/capture` — and add tests for the thin branches it
     reveals (likely: `compute_msd_curve`, `compute_tau_rho`,
     `_compute_eta_m` edge paths; renderer/trails/recorder fallback
     branches).
   - Split the oversized test modules along their existing class
     seams: `test/viz/test_renderer.py` (~1 100 lines — move the
     GPU-gated impostor/depth-cue classes to
     `test/viz/test_renderer_impostor.py`) and
     `test/analysis/test_metrics.py` (~1 000 lines — move
     motion/export tests beside the existing
     `test_metrics_motion.py`/`test_metrics_schema.py`); update the
     T1.5 collection-count pins in the same commit.
   - Periodically run the gated suites **together**, not only
     per-file: `pytest -m gl` (alias `gpu`) for the GPU set and
     `pytest -m slow` for the phase-diagram sweeps, density scaling,
     and EvoFlock runs — these sit outside the fast suite and can
     bit-rot silently after metrics/renderer changes.
9. **Keep the discipline that is already working**: goldens re-pinned
   in the same commit as any physics change; every new feature lands
   with its inline test block; no new flat config names; no new
   `print()`; every new atom ships with its composer.

---

## Appendix A — Deliberately excluded scope (recorded for a future scope change)

These capabilities are **out of scope by decision**
([roadmap0.md](roadmap0.md) §0, decision 3). Their mechanics and math
are recorded here so a future scope change can start from a verified
sketch instead of re-research. Nothing below blocks anything in
[roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md).

1. **Screensaver mode** — `scripts/screensaver.py` polling idle every
   60 s (macOS: `ioreg -c IOHIDSystem` HIDIdleTime) and
   spawning/killing a fresh `python -m pymurmur` subprocess.
2. **Desktop-overlay mode** — one `PIL.ImageGrab.grab()` screenshot
   drawn as a fullscreen textured quad behind the flock; a *live*
   transparent overlay needs per-OS compositing that no source ever
   implemented.
3. **GPU-compute simulation backends** — WebGPU/GPGPU ping-pong
   blueprints (texture-arithmetic force passes). The CPU tiers plus
   numba ([roadmap2.md](roadmap2.md) S2.B10) and the render-tier ladder
   ([roadmap3.md](roadmap3.md) S4.4a/S4.10) cover the target scales.
4. **Hildenbrandt–Hemelrijk flight physics** — starting math for a
   future `physics/extensions/flight.py`, gated
   `flight_physics: bool`:
   - gravity `a += (0, 0, −g)`;
   - lift `a += (0, 0, k_L·|v|²)` with `k_L` tuned so lift = g at
     `|v| = v0`;
   - drag `a += −k_D·(|v|−v0)·v̂` — relaxes speed toward cruise,
     replacing the hard clamp;
   - banking — cap lateral acceleration at `a_lat ≤ g·tan(φ_max)`;
   - roost pull gated by the S2.B8 dusk factor
     ([roadmap2.md](roadmap2.md));
   - acceptance: level flight at v0 self-sustains, slowed birds
     descend, speed distribution unimodal around v0 with no clamp.
5. **EvoFlock research directions** — CMA-ES benchmark, GP model
   evolution, non-uniform agents (per-bird ±10 % perturbation of
   mass/max_force/speed — would ride on the per-bird columns of
   [roadmap0.md](roadmap0.md) §4.4), non-reciprocal interactions,
   stigmergy (spatial trail markers read by later birds). All kept
   feasible by S6's black-box params-in → scalar-out evaluator
   interface ([roadmap4.md](roadmap4.md)), which survives a model swap.
6. **Multi-flock parallax scenes** — a `MultiScene` driver in
   `scripts/` owning two `SimulationEngine`s and one renderer, two
   instance draws per frame, per-flock tint via the S4.6 hue channel
   ([roadmap3.md](roadmap3.md)); blocked only by the
   one-engine-one-flock architecture, deliberately unchanged.
7. **VR/XR** — out of scope entirely (the pilotable-flock desktop port
   lives in [roadmap2.md](roadmap2.md) S2.E6).

## Appendix B — Documentation-change checklist

When a phase or track lands, keep the document set and repo artifacts
in sync (the doc-drift test T1.4, [roadmap1.md](roadmap1.md), enforces
the mechanical parts):

| When | Where | Change |
|------|-------|--------|
| D1 lands | [roadmap0.md](roadmap0.md) §4.1 | finalize the nested-config contract; live-vs-static tables into sub-config docstrings |
| D1 lands | `conf/*.yaml` | rewrite to the nested schema (documented-intent values) |
| D2 lands | [roadmap0.md](roadmap0.md) §4.2, §6 | ForceMode protocol final; force-mode list kept equal to `sorted(MODE_REGISTRY)` |
| D5/D6/D7 land | [roadmap0.md](roadmap0.md) §4.6/§4.7/§4.8 | SpatialIndex, StepContext/extensions, InstanceSchema/renderer contracts final |
| D9 lands | [roadmap0.md](roadmap0.md) §2 | module map + dependency matrix final; doc-drift test T1.4 enabled |
| S2.A9 / S6.6 land | `conf/` | field_* presets verified against the table; evo confined/open configs |
| each S-track lands | `test/data/` | golden re-pins in the same commits as physics changes |
| any item completes | [roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md) | update the item's Status line; keep IDs stable (tests reference them) |
| any defect fixed / decision taken | this file | strike the §1/§2/§3 entry in the same commit |

## 5. Conclusion

The current codebase is a genuinely advanced partial implementation of
this plan: the config composition, mode registry, flock state columns,
integration kernel, occlusion/Θ/δ̂ math, Vicsek memory term, most of
the field-mode term set, the threat FSM core, the metrics suite, the
renderer/trails/capture stack, and the CLI/facade all exist. What
remains is characteristic of a mid-migration system: a handful of
**small confirmed defects** (§1) that silently disable whole features
(wander config, adaptive quality, sphere boundary, acceleration
metrics), a set of **spec-vs-code ambiguities** (§2) that must be
decided rather than left dual, **dead fields and dead atoms** (§3)
that the drift guards exist to catch, and the **missing wiring** that
turns declared contracts into enforced ones (speed/owns flags, control
hook, depth attachment, governor feed). Follow §4 in order — defects,
guards, decisions, wiring, config completion, then the remaining
science items by gate — and every step stays small, tested, and
traceable to an item ID in
[roadmap1.md](roadmap1.md)–[roadmap4.md](roadmap4.md), with
[roadmap0.md](roadmap0.md) as the stable contract reference.
