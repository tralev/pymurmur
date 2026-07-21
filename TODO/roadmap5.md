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

1. **Sphere boundary is origin-centred.**
   `physics/boid.py::_sphere_soft` measures `‖p‖`, not `‖p − C‖`. With
   the default domain `[0,1000]×[0,700]×[0,400]` most birds are
   permanently "outside" and get hard-projected. Fix, then re-pin the
   `test/data/golden_*_sphere.npz` goldens in the same commit.
   → [roadmap1.md](roadmap1.md) D4.
2. **Wander extension reads non-existent config keys.**
   `physics/extensions/wander.py` reads `cfg.wander_speed` /
   `cfg.attractor_radius`; the config defines `wander_attractor_speed`
   / `wander_attractor_radius`. Both fields are dead; wander silently
   runs at speed 1.0 (configured default: 0.10).
   → [roadmap2.md](roadmap2.md) S2.A1.
3. **`C` (clear-all) key is dead code; `Q` unbound.**
   `viz/input_control.py::_handle_keydown` has two `elif key ==
   pygame.K_c:` branches — the letter-preset "c" branch wins, the
   clear branch is unreachable. Rebind clear (e.g. Shift+C or another
   key) or reorder deliberately; add `Q` quit.
   → [roadmap4.md](roadmap4.md) S5.4.
4. **`--fullscreen` parsed but never applied**
   (`__main__.py` creates the window without checking
   `args.fullscreen`). → [roadmap4.md](roadmap4.md) S5.5.
5. **QualityGovernor is never fed.** `Visualizer.run()` neither calls
   `self._governor.feed(frame_ms)` nor `_apply_quality_actions()` —
   adaptive quality is entirely inert (only the `[Qn]` title suffix
   exists). → [roadmap1.md](roadmap1.md) D8,
   [roadmap3.md](roadmap3.md) S4.10.
6. **Seed semantics broken for 0/None.**
   `PhysicsFlock.__init__`: `np.random.default_rng(config.seed if
   config.seed else 0)` — `seed=0` is conflated with `seed=None`, and
   unseeded runs are silently deterministic with seed 0. Contract:
   integer seeds honored (0 included), `None` → fresh entropy
   ([roadmap0.md](roadmap0.md) §3.2). → [roadmap1.md](roadmap1.md) D3.
7. **`position_init: "sphere"` silently degrades to `"box"`.** The
   validator accepts `"sphere"` but `boid.py::init_positions` has no
   branch for it. Implement the filled ∛-law sphere.
   → [roadmap2.md](roadmap2.md) S2.B9.
8. **Steric clamp inert in production.** `steric_force` supports
   `max_force` but `ProjectionMode` never passes it.
   → [roadmap2.md](roadmap2.md) S1.6.
9. **`noise_force` discards its scale.** The output is normalised to
   unit magnitude, so `noise_scale` only toggles on/off instead of
   setting the kick size. → [roadmap2.md](roadmap2.md) S1.5.
10. **Ripple envelope exported as a scalar summed over all birds.**
    `extensions/ripple.py` sets `config._ripple_envelope_sum =
    float(np.sum(...))` — the field-mode fold-noise term then scales
    with N (huge for large flocks). Export the per-bird `(N,)` array
    (sum over trains only). → [roadmap2.md](roadmap2.md) S2.A5/S2.A6.
11. **Engine ignores mode `speed_mode`/`owns_positions`.**
    `flock.integrate()` always runs band-clamped, `move=True`;
    `InfluencerMode.owns_positions=True` is untruthful; per-mode fixed
    speed is hand-rolled inside modes. Wire the flags through
    `engine._step_physics`. → [roadmap1.md](roadmap1.md) D2.
12. **`field_inertia` dead** — `integrate()` supports `inertia` but no
    caller passes `cfg.field_inertia`; every field preset's inertia
    column is ignored. → [roadmap2.md](roadmap2.md) S2.A7.
13. **`output/evolved.yaml` never written** despite the EvoFlock
    docstring promising it. → [roadmap4.md](roadmap4.md) S6.6.
14. **AngleMode state is class-level.** `AngleMode._last_cell` is
    shared across all engines in the process — parallel engines or
    mode switches corrupt each other's incremental-grid state.
    → [roadmap2.md](roadmap2.md) S2.C6, [roadmap1.md](roadmap1.md) D2.
15. **Angle mode has no config section.** Every knob (`turn_rate`,
    `margin`, `base_speed`, radii …) is a `getattr` default — nothing
    is tunable from YAML, and the values silently differ from the
    documented preset (`max_turn_rate` 360 vs 200, `turn_threshold`
    0.8 vs 0.5, `base_speed` 4 vs 150).
    → [roadmap2.md](roadmap2.md) S2.C7/S2.C8.
16. **Capture override precedence is env > CLI** (env vars are read
    after CLI flags mutate the config); contract is YAML < env < CLI.
    → [roadmap3.md](roadmap3.md) S4.9.
17. **Headless FBO has no depth attachment** — headless captures
    resolve overlap by draw order, not depth.
    → [roadmap1.md](roadmap1.md) D7.
18. **Metrics likely read zeroed accelerations.**
    `engine._step_physics` calls `metrics.collect()` *after*
    `integrate()` has reset `accelerations`; `force_avg`/`power_avg`
    and the physical conversions probably read zeros every frame. The
    stash (`last_accelerations`) exists — make metrics read it, and add
    the stash test. → [roadmap2.md](roadmap2.md) S2.B4.
19. **Unbounded accumulators.** `MetricsCollector.history`,
    `_position_snapshots`, `_density_history`, and the Recorder frame
    list grow without caps — long runs leak. Add
    `cfg.metrics.history_cap` / frame caps + the soak test.
    → [roadmap1.md](roadmap1.md) T6.3.
20. **T4.4 invariance coverage incomplete after dedup.** The deleted
    `test/analysis/test_metrics_invariance.py` was a partial
    deduplication — the nematic sign-flip and SO(3) invariance tests
    survive in `test_metrics.py`, but α rotation-invariance,
    dispersion/gyration translation-invariance, permutation invariance,
    and the `[0,1]`-bounds sweep now have **no test anywhere**. Land
    the missing cases. → [roadmap1.md](roadmap1.md) T4.4.
21. **`spawn_at` hardcodes speed 4.0** (ignores `config.v0` and
    `velocity_init`); interactive spawns misbehave at other cruise
    speeds. → [roadmap1.md](roadmap1.md) D3,
    [roadmap4.md](roadmap4.md) S5.4.

## 2. Implemented but unclear — verify or decide, then pin with tests

Where the code deviates from the spec it is *not obvious which is
intended*; each needs an explicit decision (keep the code → rewrite the
spec item and its tests; keep the spec → fix the code and re-pin
goldens). Do not leave both ambiguous.

1. **S1.5 force kernels** ([roadmap2.md](roadmap2.md)) — separation
   1/d vs spec 1/d²; cohesion always-normalised vs `limit3(·, 1)`;
   alignment `normalize(v̄)−normalize(v_i)` vs `normalize(v̄−v_i)`.
   These change flock behaviour materially; the existing
   `test_kernels.py` may be pinning the old forms.
2. **S2.A3 leader/chaser** — no dedicated group-anchor formula, no
   secondary anchor/`sec_mix`, per-group mean lag, approximated leader
   target; `field_num_groups`/`field_leader_fraction` ignored
   (hardcoded 7 / 0.84).
3. **S2.A5 boundary containment** — inverse-overshoot
   `−μ r̂/max(d−R, 0.05R)` vs the spec's linear `(d−1.45U)·1.6`
   spring (the code comments it deliberately deviates).
4. **S2.E2 influencer substeps** — direction-only blending with a
   single end-of-frame move vs per-substep move-then-steer with
   one-step lag; also the engine and the mode both maintain the tick.
5. **S3.9 rewards** — nine `1/(1+x)` bonus terms vs the five-term
   penalty composite; `faithful_signs` flips everything vs only the
   alignment term. The MARL bridge tests assume the spec form.
6. **S1.8 H₂ symmetrization** — `(A+Aᵀ)/2` vs `max(A, Aᵀ)`.
7. **S2.B4 physical power/energy** — product-of-means power and
   kinetic `½mv²` energy vs `m⟨|k_a a · k_v v|⟩` and accumulated work
   `Σ P·Δt`.
8. **S2.B5 jitter** — multiplicative `w·(1+U(0,j))` vs additive
   `w + U(0, range)` with fixed ranges.
9. **S2.B8 ecology formulas** — dusk-sigmoid parameterization,
   seasonal-amplitude curve, coherence-gate window, temperature-boost
   form, and the 77/256 predator hash (spec: 0.296 rate via
   `(day·2654435761 mod 1000)/1000`) all deviate.
10. **S4.4 winged mesh** — different vertex table and continuous-sine
    flap vs the source-verified 8-vertex table with square-wave
    `⌊frame/flap_period⌋` toggle.
11. **S4.3 trail shaping** — velocity head/tail/wave factors,
    accumulation fade-alpha formula, lines-mode 5-segment layout and
    `prog²` head-vanishing wave are all simplified/different.
12. **S2.A6/S2.A1 field-time bases** — field terms run on
    `config._field_time = frame·dt` while Wander/Ripple advance their
    own `self._t` clocks; after a reset or mode switch these
    desynchronize. Decide a single time authority (mode instance state
    per [roadmap1.md](roadmap1.md) D2).
13. **S2.D1 vicsek species blend** — fear blend folds `(1−η)·û_noisy`
    into one normalisation; all-predator early-out returns without
    randomizing predator headings. Confirm against the spec's staged
    form.
14. **S2.A2 seeds** — field mode uses `arange(n_active)` as seeds
    rather than the flock's random `seeds` column; targets are
    index-dependent, so bird removal reshuffles targets.
15. **S5.4 spawn ray** — intersects the camera-target Z-plane vs the
    spec's median-flock-depth point.
16. **Recorder→MPL fallback frame merge** (`recorder.py::
    _fallback_to_mpl` slices `mpl.frames[existing:]`) — index-offset
    bookkeeping across two recorders; verify no duplication/loss when
    the GPU fails mid-run.
17. **S3.5 τρ stop cap** — 20-lag cap vs `0.25·buffer` (=125).
18. **Preset labels (S5.1)** — parameter values match the table but
    labels/descriptions are shuffled relative to it.
19. **Instance-buffer memcpy count** — `test/viz/test_renderer.py::
    test_renderer_single_memcpy` was loosened to assert **2**
    `vbo.write` calls (instance VBO + separate colour VBO). That is a
    symptom of the D7 schema divergence, not an independent fact:
    merging to the contract's 8-float single schema
    ([roadmap0.md](roadmap0.md) §4.8) restores one memcpy per frame and
    the test re-tightens to 1; blessing the two-VBO layout instead
    means amending §4.8. Resolve together with the D7 decision
    ([roadmap1.md](roadmap1.md)).

## 3. Dead config fields and dead atoms (drift-guard targets)

Fields defined but never read (or read under wrong names) — each needs
a reader or removal; the T1.2 drift guard ([roadmap1.md](roadmap1.md))
should fail on all of these today:

- `wander_attractor_speed`, `wander_attractor_radius` (wrong-name
  reads — defect §1.2)
- `field_target_pull` (term missing), `field_inertia` (§1.12),
  `field_drift_direction`, `field_ripple_trains`,
  `field_shell_radius_base`, `field_inner_radius_factor`,
  `field_num_groups`, `field_leader_fraction` (hardcoded constants)
- `alignment_radius_ratio` (no alignment-radius subset in spatial mode)
- `use_toroidal_distance` for the KDTree path (no `boxsize`)
- `boundary_radius_factor` (no reader found — verify)
- `influencer_init_separation` (density init never called)
- `predator_speed_boost` / `predator_perception_boost` (verify actual
  consumption in the spatial pipeline)
- EvoFlock genes `predictive_avoid_weight`, `static_avoid_weight`
  (no physics reader — [roadmap4.md](roadmap4.md) S6.5)

Dead atoms (defined, never composed — violates the Micro→Macro rule,
[roadmap0.md](roadmap0.md) §3.3):

- `core/types.py::seed_noise3` → composer is
  [roadmap2.md](roadmap2.md) S2.B11
- `physics/forces/_base.py::ForceTerm` + `composeForces` → composer is
  the S2.A5 composition contract
- `influencer.py::influencer_density_init` /
  `density_init_positions` → wire via S2.E4
- `physics/obstacles.py` (whole layer) → engine integration in
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
