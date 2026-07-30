# Simulation Features — Yet To Be Implemented

This file tracks concrete, verified gaps in pymurmur's simulation
capabilities — as opposed to `TODO/seo.md` (non-simulation
discoverability/polish gaps) and `TODO/sim.md` (reference survey
material of 20 external flocking implementations, not itself a todo
list). `TODO/validation_rigor.md` is a related, separate item: a
not-yet-wired scientific acceptance-test harness, tracked on its own
because it's implementation-ready rather than a design candidate.

---

## 1. Confirmed still missing (re-verified against current code, 2026-07-30)

An earlier gap analysis (formerly `TODO/comparison.md`'s "What's
Missing in §22 vs §01–21" section, now folded into this file) had gone
stale — most of what it listed as "not yet implemented" had in fact
already been built by the time this file was written. Re-checked every
claim directly against the live codebase rather than trusting the old
document. Genuinely still missing:

| Feature | Status | Notes |
|---|---|---|
| Reflective boundary mode (velocity-mirror on contact) | Not implemented | Only 5 boundary strategies exist (toroidal, open, margin, sphere, sphere_soft) — none reflect velocity on contact. See `sci/boundary_strategies.md`'s Beyond-pymurmur section. |
| GPU compute-shader offloading | Not implemented | pymurmur is CPU-only (numpy + joblib parallelism); no compute-shader path for the O(N²)/grid workloads. Raised independently in `sci/field_mode.md`, `sci/spatial_mode.md`, and `sci/spatial_acceleration.md`'s Beyond-pymurmur sections — see §2 below, this is the single most-repeated candidate across the whole survey. |
| Animation-only concerns (wing flap, banking/roll, skeletal trails) | Not implemented, likely out of scope | Rendering-layer concerns, not simulation-core — pymurmur's visualization is a separate optional layer from the physics engine. Not a physics gap. |

**Already implemented, contrary to the old gap analysis's claims** (verified
directly, so this doesn't silently re-inherit the staleness): H₂
consensus robustness, density-scaling analysis, PCA flock-shape
metrics, density correlation time τ_ρ, external opacity Θ′,
exponential-smoothing/noise-modulated speed models (this session's
`velocity_adaptive`/`noise_modulated` speed strategies), and a
priority-ordered force stack (`physics/priority_stack.py`, wired and
tested, opt-in).

---

## 2. Recurring candidate themes across sci/*.md's "Beyond pymurmur" sections

Every one of the 21 `sci/*.md` files now documents its own 2-5
not-implemented candidate techniques from the broader flocking/collective-
motion literature (see each file's own "Beyond pymurmur" section for
full formulas and rationale). Two themes recur independently across
multiple, unrelated files — worth flagging as higher-signal candidates
precisely because several different authors converged on them without
cross-referencing each other:

- **GPU compute-shader evaluation** — raised in `field_mode.md`,
  `spatial_mode.md`, and `spatial_acceleration.md` independently. Only
  relevant above roughly N > 1,000; CPU/numpy is adequate below that.
- **Roosting / fixed-site attraction** — raised in `field_mode.md`,
  `spatial_mode.md`, and `vicsek_mode.md` independently (a gentle,
  decoupled attractor toward a landing site, distinct from the
  existing Wander extension's free-roaming attractor).

## 3. Full cross-reference

| File | Candidates (see file for formulas) |
|---|---|
| `angle_mode.md` | Reynolds wander circle; banking/roll on turns; priority-ordered turn-rate stack |
| `boundary_strategies.md` | Reflective boundary; local-space zone clamp (matrix transform); multi-level edge state machine |
| `conceptual_map.md` | Cucker–Smale flocking; model-predictive/receding-horizon steering; imitation-learned steering |
| `density_and_shape.md` | Voronoi-cell local density; radial distribution function g(r); fractal/box-counting dimension |
| `field_mode.md` | Roosting/fixed-site landing; within-mode priority-ordered term evaluation; GPU compute-shader term evaluation |
| `force_kernels.md` | Exponential-decay cohesion; velocity-differential separation/alignment/cohesion; distance-override "nearest wins" |
| `h2_consensus.md` | Time-varying (dynamic) network robustness; directed (non-symmetrized) consensus; robustness under adversarial node removal; higher-order (hypergraph) consensus |
| `influencer_mode.md` | Multiple simultaneous targets/leaders; obstacle-aware hard heading override; banking/orientation animation |
| `marl_mode.md` | Centralized critic/decentralized actors (CTDE); inter-agent communication channels; curriculum scheduling; per-bird heterogeneous policies; terminal failure conditions |
| `msd_and_order.md` | Spatial correlation function C(r); Van Hove self-correlation function; susceptibility near the ordering transition |
| `neighbor_selection.md` | Dynamic vision-distance feedback; forward-hemisphere-only selection; capped-radius as a first-class strategy |
| `noise_and_speed.md` | True gradient (Perlin/Simplex) noise; persistent per-bird noise seeds; two-rate speed response |
| `obstacle_avoidance.md` | Golden-spiral direction sampling; multi-point see-ahead sampling |
| `predator_prey.md` | Multi-predator inverse-lerp fear weighting; selfish-herd/confusion-effect modelling; visual-looming threat detection; post-attack recovery/cooldown state |
| `projection_model.md` | Full-ellipsoid anisotropic silhouette occlusion; raycast-based occlusion; soft/probabilistic blind cone; multi-frame occlusion memory |
| `spatial_acceleration.md` | Hashed grid + bitonic sort (GPU); parallel prefix-sum reduction (GPU); Verlet/skin-radius neighbour lists |
| `spatial_mode.md` | GPU compute-shader force evaluation; roosting/fixed-site attraction; parallel prefix-sum global reduction |
| `speed_models.md` | Priority-ordered speed override; two-rate acceleration; distance-to-wall speed damping |
| `steering_paradigms.md` | Exponential smoothing of heading (mode-wide, not just speed); force accumulation without a clamp step; cosine-zone weighting as primary paradigm |
| `theta_accel_correlation.md` | Granger causality test; transfer entropy |
| `vicsek_mode.md` | Topological (k-NN) neighbour selection for Vicsek specifically; roosting force; literal original-Vicsek angular noise |

---

## 4. Scientific validation harness (separate, implementation-ready item)

See `TODO/validation_rigor.md` — a 5-hard-gate scientific acceptance
suite (internal opacity Θ̄≈0.30, density-scaling exponent b≈−0.5, no
fragmentation, robustness-optimum m*∈{5,6,7}, m* vs. flock-thickness
monotonicity) is fully specified with tolerances and a runnable
harness structure, but none of the 5 gates are currently wired as
automated tests. Unlike the analysis-module code that document
sketches (which pymurmur already has under different names — see that
file's adaptation notice), the acceptance-gate thresholds and harness
structure are net-new and not duplicated anywhere else in the repo.
