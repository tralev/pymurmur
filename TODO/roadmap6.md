# roadmap6.md — Delta from roadmap_deepseek.md not covered by roadmap0–5 or the codebase

This file extends the [TODO/roadmap0.md](roadmap0.md)–[TODO/roadmap5.md](roadmap5.md)
set with everything in `roadmap_deepseek.md` (historical — completed and
removed 2026-07-21, preserved in git history) that is
**not** already (a) implemented in the current `pymurmur/` codebase or
(b) specified as a work item in roadmap0–roadmap5. Each item follows the
set's convention: *idea* → *math/spec* → *impl* → *tests* → **Status**
(legend in [TODO/roadmap0.md](roadmap0.md) §0: DONE · PARTIAL ·
DIVERGES · MISSING · VERIFY).

**Audit basis (July 2026, current working tree).** The codebase has been
executing the deepseek P-scheme: P0–P12 are landed (MARL bridge included:
`physics/forces/marl.py`, `analysis/gym_env.py`,
`scripts/{train_marl,rollout_marl}.py` all exist), the test tree is
restructured into `test/{l0_system,l1_subsystems,l2_integration,
l3_modules,l4_crosscutting}`, and `.github/workflows/guard-rails.yml`
ships 9 jobs (dag, golden, config-drift, doc-links, 3d, collection-count,
mypy, evolved, summary gate). What remains from roadmap_deepseek.md is
therefore concentrated in **P13 (scaling & performance)**, a few **P14
guard-rail refinements**, and the **contract/appendix material** below.

**Disposition of the rest of roadmap_deepseek.md** (excluded here because
it is already covered):

| roadmap_deepseek.md section | Where it lives now |
|---|---|
| Part I current-state audit | superseded by [TODO/roadmap5.md](roadmap5.md) §1–§3 |
| P0–P2 foundations/correctness/contracts | D0–D9, T0–T4, S1 ([TODO/roadmap1.md](roadmap1.md), [TODO/roadmap2.md](roadmap2.md)) + implemented |
| P3–P7 mode phases | S2 tracks A–E ([TODO/roadmap2.md](roadmap2.md)) + implemented (incl. P3.11 grid-sep normalization — `field.py::_compute_grid_sep_normalized` exists) |
| P8–P10 rendering/metrics/UX | S3–S5 ([TODO/roadmap3.md](roadmap3.md), [TODO/roadmap4.md](roadmap4.md)) |
| P11–P12 EvoFlock/MARL | S6–S7 ([TODO/roadmap4.md](roadmap4.md)) + implemented |
| P13 scaling & performance | **S8 below** (partially implemented; gaps itemized) |
| P14 guard rails | T1 + **T7 below** (largely implemented; deltas itemized) |
| Appendix C cross-cutting concerns | **§4 below** (unrecorded items only) |
| CC5 risk register | **§5 below** (roadmap0 §6 promises it in roadmap5, where it is absent) |
| Appendix A multi-flock reconsideration | **§6 below** (a different design than roadmap5 App. A.6) |
| Appendix D glossary | **§7 below** (delta entries roadmap0 §7 lacks) |
| Appendices E/F/G (module index, ID mappings, step index) | navigation aids over the retired P-scheme — not work items; keep roadmap_deepseek.md as the historical reference |

---

## 1. Contract addenda (adopt into [TODO/roadmap0.md](roadmap0.md))

**R6.1 Level-badge system `[L0]`–`[L3]`.** roadmap0 §3.3 states "each
level only composes the level below"; deepseek makes it operational with
explicit level badges and a placement table worth adopting verbatim:

| Level | What lives here | Depends on |
|---|---|---|
| **L0** | Pure atoms: math helpers, min_image, SDF primitives, force primitives (sep/align/coh/noise), occlusion, steric, numba kernels, `integrate()` variants | numpy/stdlib only |
| **L1** | Assemblies: PhysicsFlock columns, the ForceMode classes, ExtensionManager, MetricsCollector, rewards, presets, obstacle scenes | L0 only |
| **L2** | Subsystems: SimulationEngine, Visualizer, Renderer3D, Recorder, QualityGovernor, MurmurationEnv, EvoFlock | L1 |
| **L3** | System: `__main__` CLI, `pymurmur` facade, `scripts/` | L2 |

Two rules deepseek states that roadmap0 leaves implicit — record them:
(1) **No L1 assembly imports another L1 assembly** (the golden rule is
"level *n* depends only on levels < *n*", not "≤ *n*"). (2) **The L0
rule is import discipline, not file location** — SDF primitives live in
`physics/obstacles.py` (zero project imports, pure numpy) yet are L0;
an atom's level is defined by what it imports, not by its directory.
*impl:* text amendment to roadmap0 §3.3 + the architecture test's edge
matrix already enforces the import side.
**Status: PARTIAL** — the rules are enforced de facto by
`test/l4_crosscutting/guards/test_architecture.py`; the contract text
and the L1↛L1 named rule are unrecorded.

**R6.2 Composer-enforcement test (`test/test_composers.py`).** Deepseek:
"No atom is shipped without at least one composer test proving it is
actually used. Dead atoms (no composers) are deleted. This is enforced
by `test/test_composers.py`." The TODO set states the rule
(roadmap0 §3.3) and catches *some* dead atoms via the T1.2 config-drift
guard, but no test enumerates L0 atoms and asserts each has a consumer.
*impl:* `test/l4_crosscutting/guards/test_composers.py` — collect the
public functions of the L0 surfaces (`core/types.py`,
`physics/{occlusion,steric,obstacles}.py`, `physics/forces/_base.py`,
`physics/forces/_kernels.py`, `physics/boid.py`) and AST-scan the rest
of the package for at least one call site of each; fail with the
dead-atom list. Allowlist only entries with a named roadmap item
attached (mirror of T1.2's rule for config fields). *tests:* the guard
itself; seeded regression — temporarily adding an unused helper makes it
fail.
**Status: MISSING.** Known current would-be failures (from
[TODO/roadmap5.md](roadmap5.md) §3, re-verify against the working
tree): `seed_noise3`, `ForceTerm`/`composeForces` (if S2.A5's
composition contract has not landed), `influencer_density_init`.

**R6.3 Per-step documentation template.** Deepseek's step structure —
**Verbal idea / Math / Code sketch / Composers / Test / Migration /
Source** — is a good authoring convention for future roadmap items (the
TODO set uses *math/impl/tests/Status*; the **Composers** and
**Migration** fields are the additions worth keeping).
**Status:** doc convention only; adopt when writing new items.

---

## 2. S8 — Scaling & performance  *(from P13; ≈2 days; after S2.B10 numba gates)*

Files: `test/l4_crosscutting/perf/{test_performance,test_scaling}.py`
(extend), `test/l4_crosscutting/guards/test_determinism.py` (extend),
`test/test_soak.py` (new). All heavy tests `@slow` (nightly CI lane).

**S8.1 Per-mode step-time budget table.** One data-driven budget table
(mode → ms budget at N = 2 000, headroom ×3 for CI jitter) instead of
scattered per-test constants; parametrized over `sorted(MODE_REGISTRY)`
so a newly registered mode without a budget entry fails collection.
*tests:* `benchmark(2000, 100)` per mode → mean step ≤ budget×headroom.
**Status: PARTIAL** — `test_performance.py` has hand-rolled per-mode
benches at N ∈ {150, 200, 16 000, 100, 200} with inline ms constants;
not table-driven, not registry-parametrized (marl/projection coverage —
verify), N = 2 000 point absent.

**S8.2 Scaling checkpoint ladder** (arch.md §13). The five checkpoints
tie flock size to the index/kernel tier and an explicit frame budget:

| N | Tier exercised | Target | Budget/step |
|---|---|---|---|
| 150 | SpatialHashGrid | 60 fps | 16.7 ms |
| 1 500 | SoA vectorised path | 60 fps | 16.7 ms |
| 16 000 | cKDTree batch | 60 fps | 16.7 ms |
| 50 000 | numba kernels | 45 fps | 22.2 ms |
| 300 000 | full stack, metrics off | 30 fps | 33.3 ms |

*tests (`@slow`):* `benchmark(N, 100)` at each checkpoint → per-step
time ≤ budget; assert the intended tier is actually selected at each N
(index choice + numba path), so a silent fallback to a slower tier
fails the checkpoint rather than just the timing.
**Status: PARTIAL** — benches exist at 150/16 000/300 000; the 1 500
and 50 000 checkpoints, the fps-budget framing, and the
tier-selection assertions are MISSING.

**S8.3 Memory audit at N = 300 000 — complete inventory.** Budget
≤ 25 MB over the **full** SoA inventory: `positions, velocities,
accelerations, prev_positions, last_accelerations, seeds, max_speed,
active, is_predator` + the spatial-index footprint + the packed GPU
instance buffer (when viz is active). *tests:* sum of `arr.nbytes` over
the full list ≤ 25 MB; the audited-array list is derived from the
flock-state contract (roadmap0 §4.4), not hand-maintained — adding a
new per-bird column without adding it to the audit fails.
**Status: DIVERGES** — `test_300k_allocation_and_step` audits only 6
arrays (`positions, velocities, accelerations, seeds, last_theta,
active`) against a 30 MB budget; `prev_positions`,
`last_accelerations`, `max_speed`, `is_predator`, index, and instance
buffer are uncounted. Decide 25 vs 30 MB when the full inventory is in
(the spec's 25 MB assumed the full list).

**S8.4 Long-soak escalation (24 h).** T6.3
([TODO/roadmap1.md](roadmap1.md)) specifies the ≥ 20 000-frame
nightly soak; deepseek adds the release-gate tier: 24 hours of headless
stepping (N ≈ 500, metrics + Recorder attached, ring-buffer caps live)
→ no NaN, no monotone memory-growth trend (linear fit on tracemalloc
samples ≈ 0 slope after warm-up), positions in-bounds, speed contract
held throughout. *impl:* same `test/test_soak.py` with a
`--soak-hours` knob (default the T6.3 frame count; 24 h in the release
lane only).
**Status: MISSING** — no soak test of either tier exists in the tree
(the T6.3 prerequisite `history_cap` / Recorder caps must land first —
[TODO/roadmap5.md](roadmap5.md) §1.19).

**S8.5 Determinism matrix — full breadth.** Same seed → bit-identical
positions after 100 steps for every combination of
`mode × num_threads ∈ {1, −1} × parameter_jitter on/off × numba on/off`
(fastmath **off** whenever numba is on), verified by two in-process
runs **plus one subprocess run** (catches import-order and
thread-pool nondeterminism the in-process pair cannot). *tests:*
extend `guards/test_determinism.py` with the parametrized grid; the
subprocess leg runs one representative cell per mode.
**Status: PARTIAL** — per-mode same-seed bit-identity and a
numba≡numpy equivalence test exist; the threads/jitter axes and the
subprocess leg are MISSING.

---

## 3. T7 — Guard-rail & doc-sync deltas  *(from P14 + CI appendix)*

**T7.1 Reconcile the two doc-drift targets.** The shipped
`guards/test_docs.py` implements deepseek P14.4 exactly (4 tests:
arch.md links resolve, arch.md ↔ roadmap_deepseek.md bidirectional,
roadmap links resolve) — while T1.4
([TODO/roadmap1.md](roadmap1.md)) specifies the same guard over
**roadmap0–roadmap5**. Both document sets now coexist in the repo.
Decide the doc-of-record and extend the guard accordingly: keep
arch.md bidirectional sync (already enforced), add link-resolution +
module-path-existence coverage for `TODO/roadmap0–5.md` **and this
file**, and keep the force-mode-table == `sorted(MODE_REGISTRY)` check
pointed at whichever file carries the table. *tests:* extend
`test_docs.py`; a dangling link in any covered file fails CI
(`guard-rail-doc-links`).
**Status: PARTIAL** — arch.md/roadmap_deepseek.md legs DONE; the
TODO-set and roadmap6 legs MISSING.

**T7.2 Record the CI guard topology in the doc set.** Deepseek's CI
appendix (7 jobs + summary gate) is implemented — and exceeded — by
`.github/workflows/guard-rails.yml`: `guard-rail-{dag, golden,
config-drift, doc-links, 3d, collection-count, mypy, evolved}` +
`guard-rails-summary` (merge-blocking). Neither the TODO set nor
arch.md documents the `mypy` and `evolved` jobs. *impl:* one paragraph
in the doc-of-record listing the jobs and what each enforces, covered
by T7.1's drift guard so the list cannot rot.
**Status: MISSING (docs only; CI itself DONE).**

**T7.3 Named regression edges — completeness check.** Deepseek P14.1
pins four named edges (`physics.flock !→ physics.forces`,
`viz.input_control !→ simulation`, no `cKDTree(` in `forces/`, no
module-level `np.random.*`); roadmap0 §2 adds a fifth: **no `print(`
in package sources**. Verify all five are asserted in
`guards/test_architecture.py` (the print rule gates S5.6 logging,
[TODO/roadmap4.md](roadmap4.md)).
**Status: VERIFY.**

---

## 4. Cross-cutting contracts not recorded in the TODO set  *(from Appendix C)*

1. **Actionable YAML errors (CC1).** All config-load failures name the
   offending key *and section* (not a bare `TypeError` from a dataclass
   constructor). *test:* malformed preset → exception message contains
   the section and key. **Status: VERIFY** (unknown-key *warnings* are
   specified in D1; the failure-path message contract is not).
2. **GPU context loss → graceful fallback (CC1).** Losing the GL
   context mid-run degrades to headless (or the mpl capture fallback)
   instead of a raw crash. Related but distinct from the
   startup-probe fallback (implemented) and the Recorder mid-run
   fallback seam flagged in [TODO/roadmap5.md](roadmap5.md) §2.16.
   *test:* monkeypatch the context to raise on `render` mid-run →
   clean degradation, warning logged, no frame loss in the recorder.
   **Status: MISSING.**
3. **Fastmath × metrics-export warning (CC2).** Exporting observables
   with `perf.fastmath` on raises a Warning (IEEE kernels whenever
   science is exported) — specified in S2.B10
   ([TODO/roadmap2.md](roadmap2.md)); listed here because it is
   also a cross-cutting policy: any *future* accelerated path inherits
   it. **Status: per S2.B10 (gates missing at last audit — re-verify).**

(The other CC items — NaN self-heal to `flock.center`, dt clamp behind
the accumulator, startup capability report, GPU-free-capture warning —
are implemented and already specified in D4/S4.9/S4.11/S5.5.)

---

## 5. Risk register  *(from CC5 — the Appendix C that roadmap0 §6 promises in roadmap5 but which is absent there)*

| Risk | Mitigation | Enforced by |
|---|---|---|
| Golden re-pin forgotten after a physics change | CI fails on stale goldens | `guard-rail-golden` |
| Import cycle resurfaces (`flock ↔ forces`) | named forbidden edge | `guards/test_architecture.py` |
| Config key collision corrupts sections | nested dataclasses make collision structural­ly impossible | D1 / T2 |
| Determinism broken by a new feature | full determinism matrix | S8.5 |
| Performance regression | budget table + checkpoint ladder | S8.1/S8.2 |
| 2D assumption creeps into physics | AST `(…, 2)` scan; `depth > 0` validator; random-SO(3) invariance | `guard-rail-3d` |
| Doc set drifts from code | link/module/mode-table drift tests | T7.1 |
| Dead atoms / dead config fields accumulate | composer test + config-usage drift | R6.2 / T1.2 |
| Unbounded accumulators leak on long runs | ring-buffer caps + soak tiers | T6.3 / S8.4 |

---

## 6. Multi-flock upgrade path  *(from Appendix A "Reconsidered" — recorded design, not scheduled)*

The predator–prey mechanics (S2.B3, Track D) use a **species column
inside one `PhysicsFlock`** — one array, shared index, shared rng.
Deepseek records the cleaner future design, which differs from the
`MultiScene` two-engine driver already recorded in
[TODO/roadmap5.md](roadmap5.md) Appendix A.6 — keep both sketches:

- **In-engine multi-flock (deepseek):** the engine holds multiple
  independent `PhysicsFlock` instances sharing one domain, with
  **per-flock ForceMode, per-flock metrics, and separate instance
  buffers**. The natural architecture for true multi-species,
  separately configured prey+predator populations, and per-flock
  parameterisation; requires the engine step and the renderer draw
  path to iterate flocks.
- **Multi-engine scene (roadmap5 A.6):** a `scripts/` driver owning
  two `SimulationEngine`s and one renderer — cheaper, no engine
  change, but flocks cannot interact physically.

Choose per use-case when scope changes: interaction ⇒ in-engine;
parallax/visual layering only ⇒ multi-engine.
**Status:** out of scope by decision (roadmap0 §0 decision 3); recorded.

---

## 7. Glossary delta  *(from Appendix D — entries [TODO/roadmap0.md](roadmap0.md) §7 lacks)*

Rendering & graphics: **FBO** (off-screen render target; headless
capture **must** attach depth or overlap resolves in draw order) ·
**VAO/VBO** (buffer-state container / GPU array; VAO rebuilt after any
buffer growth) · **ModernGL** (the only GPU dependency; confined to
`viz/`) · **Blinn-Phong** (ambient + diffuse·(N·L) + specular·(N·H)^k;
themes supply `u_Ambient/u_Diffuse/u_Paper/u_Ink`) · **impostor**
(camera-facing quad + fragment-shader disc; the 20 k+ birds tier) ·
**speed-stretched ellipsoid** (impostor scaled along projected velocity
by `1 + speed_ratio·0.3`) · **Fresnel rim** (`pow(1 − N·V, k)` edge
light) · **alpha-accumulation** (α ≈ 0.2, blend on, depth-write off —
density-as-darkness) · **dual-view** (two (camera, viewport) passes per
frame) · **Xvfb** (headless X server + Mesa llvmpipe for GL in CI).

Algorithms & protocols: **SSGA** (steady-state GA: select 3 → evaluate
→ delete worst → uniform crossover + Gaussian mutation → insert; no
generations) · **EMA** (`avg = 0.92·avg + 0.08·min(250, frame_ms)` in
the governor) · **FSM** (threat approach/egress with capture/clear
gates) · **ghost-cell** (boundary birds replicated across the seam for
index queries; query-only, never integrated) · **hypervolume**
(`Π max(o_k, 0.01)`) · **Pareto front** (non-dominated set persisted
to `output/evolved.yaml`) · **PPO / IPPO** (external RL trainers;
IPPO = parameter-shared independent learners, the scaling path for
large N noted in S7.3's docstring requirement).

Project concepts: **boundedUnitTravel** (`‖path(t)‖ ≤ 1` wander
guarantee) · **InstanceSchema** (8 floats `pos.xyz, vel.xyz, flag,
hue`, layout `"3f 3f 1f 1f/i"` — the D7 decision point) ·
**QualityGovernor** (degrade ladder trails → scale −0.15 → N −18 % at
< 78 % target fps for 1.8 s; recover after 3.6 s healthy) ·
**golden re-pinning** (deliberate physics change re-pins
`test/data/golden_*.npz` in the same commit; CI fails otherwise).

---

## Sequencing

S8 slots after the S2.B10 numba gates (its 50 k checkpoint needs them)
and is independent of every S-track otherwise; T7 and R6.2 can land any
time (pure guards); §4.2's context-loss fallback belongs with the D7/S4
rendering work. Adding S8 + T7 to the
[TODO/roadmap4.md](roadmap4.md) sequencing table:

| Phase | Days | Gate |
|---|---|---|
| R6 contract addenda + composer guard | ½ | none |
| T7 doc/CI reconciliation | ½ | none |
| S8 scaling & performance | 2 | S2.B10 (numba); T6.3 caps for S8.4 |

**Definition of done (delta):** composer guard green with an empty
allowlist; doc-drift guard covers arch.md + TODO/roadmap0–5 + roadmap6
with zero dangling links; budget table parametrized over
`MODE_REGISTRY`; all five scaling checkpoints pass with tier
assertions; full-inventory memory audit ≤ budget at 300 k; determinism
matrix green across all four axes incl. the subprocess leg; both soak
tiers pass in their lanes.
