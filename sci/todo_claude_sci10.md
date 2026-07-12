# TODO — Ideas & Math from `sci/new10_sci.md` (Reynolds 2026, EvoFlock) Not Implemented in the Codebase

Comparison of `sci/new10_sci.md` — evolved inverse design of flocking parameters —
against the `pymurmur/` codebase. Maps almost entirely to
[evoflock.py](pymurmur/analysis/evoflock.py) plus the physics behaviours the
15-parameter model presumes. The consolidated audit (`todo_claude3.md` §11) flagged
the headline gaps; this source-level pass adds the exact mechanics and several
deviations only visible against the paper's detail.

Already implemented: the SSGA skeleton (islands ×4, migration at the 0.05/step
rate, per-gene Gaussian mutation), hypervolume scalarization with the ε=0.01 floor,
ε-dominance Pareto-front extraction, population 300 / 30,000-step defaults,
seeded GA-level RNG, and `run(n_runs)` multi-run support.

---

## 1. The SSGA update rule deviates from the paper (§1.2)

The paper's step: select 3 → **evaluate all 3** (cached) → sort → **delete the
worst of the 3** → **crossover the best two** → insert offspring in the freed slot.
The code diverges on every clause:

- [ ] **No crossover.** Reproduction is mutation-of-one-tournament-winner
  ([evoflock.py:177-184](pymurmur/analysis/evoflock.py#L177-L184)); uniform
  crossover (each gene from a random parent) does not exist.
- [ ] **Tournament members are never (re)evaluated.** Selection compares stored
  fitness ([evoflock.py:169-175](pymurmur/analysis/evoflock.py#L169-L175)), and
  the **initial population is created with `fitness = −inf` and never evaluated
  at all** — early tournaments select essentially at random, and founders are
  judged on no data. The paper evaluates each tournament member (with a fitness
  cache — also absent: the doc's §7.1 pseudocode carries `self.fitness_cache`).
- [ ] **Replacement is elitist-global, not negative-selection-local.** The paper
  deletes the worst *of the 3* unconditionally (diversity-preserving negative
  selection); the code replaces the island-wide worst and only if the child
  beats it — a different, convergence-prone dynamic the paper explicitly designs
  against.
- [ ] **§1.6 Conservative evaluation**: 4 independent sims per candidate
  (parallel threads), **worst** fitness returned. `EvoConfig.eval_parallel`
  exists, is 1, and is read by nothing — one sim runs.

## 2. Objective functions (§3)

- [ ] **§3.1 Separation measures the wrong distance.** The paper scores the
  **nearest-neighbour** distance per boid per step, averaged over all
  N·T boid-steps. The code averages `local_spacing` — the median **7th**-
  neighbour distance, sampled at metric intervals
  ([evoflock.py:252-258](pymurmur/analysis/evoflock.py#L252-L258)) — a
  systematically larger quantity that lets collisions hide inside a good score.
  The trapezoid's ramp widths also differ (paper: up-ramp 2→2.5, down-ramp 4→5;
  code: 1→2 and 4→8).
- [ ] **§3.2–3.3 Obstacle infrastructure absent, objective inert.** `f_cf` is
  hardwired to 1.0, so `(f_cf)^500` contributes nothing to selection. Missing
  underneath: **SDF obstacle primitives** (sphere `‖p−c‖−r`, box, cylinder,
  union/subtraction), **zero-crossing collision detection**
  (`sign(SDF(p_old)) ≠ sign(SDF(p_new))`), the **kinematic surface correction**
  `p ← p − SDF(p)·∇SDF/‖∇SDF‖`, and per-step collision counting. This is a
  physics-layer feature (a new `obstacles` module + integrate hook), not just an
  evoflock one.
- [ ] **§3.5 Speed score is unscaled and its mechanism is missing.** Target band
  [19,21] m/s requires the physical-unit layer (`todo_claude_sci3.md` §3); code
  scores [3,5] sim units. Deeper gap: the paper's speed control is a
  **forward-weight behaviour** `F = w_fwd·sign(v_target − |v|)·û` whose strength
  is itself evolved — the codebase has no forward force at all, only the hard
  clamp band.
- [ ] **§3.6 Curvature is a proxy, not curvature.** Paper:
  `κ = |v × a| / |v|³` per boid-step, averaged, then
  `clamp(0.8 + (κ_avg/0.1)·0.2, 0.8, 1.0)`. Code substitutes a
  dispersion/α ratio with an unrelated normalisation. Velocities *and*
  accelerations are already in the SoA arrays — true κ is a three-line
  vectorised computation (note: sample accelerations before the
  integrate-reset zeroes them).
- [ ] **§3.8 The headline experiment is unreproduced**: evolve with **no
  alignment objective** and verify alignment (α) emerges from
  separation + speed + obstacles alone — the paper's central scientific claim,
  and a natural regression test once items above land.

## 3. The 15-parameter black-box model (§4)

The code evolves 10 genes; 2 of them (`predictive_avoid_weight`,
`static_avoid_weight`) are `setattr` onto SimConfig where nothing reads them —
dead genes adding search noise. Missing genes **and their underlying physics
behaviours**:

- [ ] `forward_weight` — needs the §3.5 forward force.
- [ ] `max_dist_separate/align/cohere` — **per-behaviour interaction distances**;
  spatial mode has one shared kNN set and no metric gating at all (cf. the
  hybrid-filter gap, `todo_claude_sci3.md` §4).
- [ ] `angle_separate/align/cohere` — **per-behaviour perception cones**
  (`cos α ∈ [−1,1]`); only projection mode has any angular gating (one global
  blind cone).
- [ ] `fly_away_max_dist`, `min_time_to_collide` — fly-away and **predictive
  obstacle avoidance** (look-ahead time) behaviours; depend on §3.2's obstacle
  layer.
- [ ] **§4.2 k = 7 topological neighbours** as the model's fixed constant
  ("The Seventh Starling") — spatial mode uses `topological_cap = 50`.
- [ ] **§7.3 Tuning pymurmur's own model**: the doc's GA-range table includes
  `sigma` (needs integer-gene handling — all genes are floats today),
  `blind_deg`, `anisotropy`, and `speed_min_factor` — the last is **hardcoded
  0.3** in `integrate()` and not a config field at all; promote it.

## 4. Determinism & run protocol (§5, §11)

- [ ] **Persist evolved parameter sets.** The docstring promises
  `output/evolved.yaml`; `run()` returns a dict and writes nothing. Save the
  best genome (and the Pareto front) with the seed and objective scores —
  §5.2's "store full parameter sets, not just fitness".
- [ ] **Per-run seed logging** in the §11 protocol (`seed = record_and_log()`)
  — `run(n_runs)` exists but reuses one RNG stream with no per-run seed record,
  so a winning run can't be replayed in isolation.
- [ ] **Deterministic parallel evaluation** (§5.2) — a design constraint to
  honour when `eval_parallel` is actually implemented (fixed per-sim seeds,
  order-independent reduction via min).

## 5. Evaluation environments (§6)

- [ ] **Confined vs open-space scenarios.** The paper's key limitation:
  parameters evolved in obstacle-rich confinement fail in open space
  ("subflocks head off in different directions"). Current evaluations run in a
  toroidal box — neither confined nor truly open, and unlike either paper
  regime. Once SDF obstacles exist, ship two evaluation configs
  (`evo_confined.yaml` with an enclosure + obstacles; `evo_open.yaml` with
  `boundary_mode: open`) and score candidates in both; this is also the doc's
  bridge to Pearce-style open-space cohesion.

## 6. Research-direction backlog (§10 — record, low priority)

- [ ] **CMA-ES benchmark** (§10.5) alongside SSGA (optional `cma` dependency;
  same objective interface).
- [ ] **Ramp-ablation test** (§10.6): binary-threshold scoring vs trapezoids —
  cheap experiment once objectives are correct.
- [ ] **Non-uniform agents** (§10.2): per-bird parameter perturbation vectors
  (±10% on mass/max_force/speed) — needs the per-bird attribute columns also
  wanted by the predator-species work (`todo_claude_sci5.md` §2).
- [ ] **Non-reciprocal interactions** (§10.3) and **stigmergy** (§10.4 — spatial
  trail markers influencing later birds): note as future extensions; no
  supporting mechanism exists.
- [ ] **GP model evolution** (§10.1): out of near-term scope; the practical
  takeaway is to keep the objective/evaluator API model-agnostic (black-box:
  params-in, scalar-out) so the GA layer survives a model swap.
