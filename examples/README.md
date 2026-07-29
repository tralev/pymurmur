# Example metrics captures

Curated, **committed** example runs showing what pymurmur's metrics
output actually looks like — unlike `output/` (gitignored, whatever you
last ran locally), these are checked into the repo so anyone can see
real output without running the simulator.

Every pair except `metrics_full` (`metrics_<name>.csv` /
`metrics_<name>.json`) is 180 frames, seed 42, `metrics_detail_level=2`
with `metrics_interval=20` (so the expensive/gated metrics — `h2`,
`tau_rho`, `hull_volume`, `aspect_ratio`, `msd_slope`, etc. — actually
populate 9 times across the run instead of once or never). Field
semantics are documented in `pymurmur/analysis/metrics/flock_metrics.py`
and summarized in [arch.md §8](../arch.md#8-data-representation-level-3).

## What each one demonstrates

| File | Source | Demonstrates |
|---|---|---|
| `metrics_projection.json/csv` | `SimConfig()` defaults (projection mode, 150 birds) | The baseline most users will see first — no extensions, default kernels. |
| `metrics_influencer.json/csv` | [`conf/murmuration_influencer.yaml`](../conf/murmuration_influencer.yaml) | The only mode where `target_dist_min`/`target_dist_max` populate (every frame here) — every other example leaves them `null`. |
| `metrics_showcase.json/csv` | [`conf/murmuration_showcase.yaml`](../conf/murmuration_showcase.yaml) | `sphere_soft` boundary, `bell_zone` kernels, `hash_grid` spatial index, and all 4 of the newer behavioral extensions (SpeedNoise, NeighborAdaptiveSpeed, DynamicVisionRange, BoidStateMachine) plus Predator/Ecology — none of which any other shipped preset exercises. |
| `metrics_angle.json/csv` | [`conf/murmuration_angle.yaml`](../conf/murmuration_angle.yaml) | Axis-angle (Rodrigues) steering mode. |
| `metrics_vicsek.json/csv` | [`conf/murmuration_vicsek.yaml`](../conf/murmuration_vicsek.yaml) | Vicsek constant-speed alignment + predator-prey (101 birds: 100 prey + 1 predator). |
| `metrics_field.json/csv` | [`conf/murmuration_field.yaml`](../conf/murmuration_field.yaml) | Field/blob-anchor mode. `num_boids` capped to 200 for this capture only — the preset's native 16,000-boid scale is intractable at `metrics_detail_level=2` (the H2/consensus-robustness pipeline does a dense `eigh` of an N×N graph Laplacian, O(N³); see `test_performance_budgets_combined.py`). |
| `metrics_marl.json/csv` | [`conf/murmuration_marl.yaml`](../conf/murmuration_marl.yaml) | MARL bridge mode: external control + deferred global rules. |
| `metrics_kernel_velocity_weighted.json/csv` | [`conf/kernels/kernel_velocity_weighted.yaml`](../conf/kernels/kernel_velocity_weighted.yaml) | The one separation kernel weighted by runtime closing speed rather than a static distance/angle formula. |
| `metrics_kernel_exp.json/csv` | [`conf/kernels/kernel_exp.yaml`](../conf/kernels/kernel_exp.yaml) | A closed-form exponential-falloff separation kernel, contrasting with `velocity_weighted`'s runtime-computed one. |
| `metrics_obstacles.json/csv` | [`conf/murmuration_obstacles.yaml`](../conf/murmuration_obstacles.yaml) | `priority_stack_enabled` binary-cutoff force-budget cascade (obstacle avoidance / predator threat / flocking) plus an SDF obstacle course. |
| `metrics_speed_law_quadratic.json/csv` | [`conf/speed_laws/speed_law_quadratic.yaml`](../conf/speed_laws/speed_law_quadratic.yaml) | Angle mode with both `angle_speed_mode` and `neighbor_adaptive_speed.mode` set to `quadratic` — contrast with `metrics_angle.json/csv`'s default `linear` law. |
| `metrics_speed_law_softened.json/csv` | [`conf/speed_laws/speed_law_softened.yaml`](../conf/speed_laws/speed_law_softened.yaml) | Same pairing, `softened` — a sigmoid-shaped deficit-based speed law instead of quadratic/linear. |
| `metrics_filter_global.json/csv` | [`conf/filters/filter_global.yaml`](../conf/filters/filter_global.yaml) | `neighbor_filter=global` — every active bird considers every other active bird, no spatial index or k-NN cap, contrasting with the default `hybrid` filter every other example uses. |
| `metrics_kernel_bell_zone.json/csv` | [`conf/kernels/kernel_bell_zone.yaml`](../conf/kernels/kernel_bell_zone.yaml) | `bell_zone` — the only kernel name valid for separation, alignment, AND cohesion simultaneously (cosine-bell-weighted falloff, peaking at a zone center and tapering over a zone width), in isolation from `metrics_showcase`'s many other simultaneous extensions. |
| `metrics_filter_topological.json/csv` | [`conf/filters/filter_topological.yaml`](../conf/filters/filter_topological.yaml) | `neighbor_filter=topological` — capped at `influence_count` nearest neighbors, no `visual_range` distance filter at all. |
| `metrics_filter_none.json/csv` | [`conf/filters/filter_none.yaml`](../conf/filters/filter_none.yaml) | `neighbor_filter=none` — every k-NN candidate returned completely unfiltered. |
| `metrics_full.json/csv` | `conf/murmuration_showcase.yaml`, 40 frames, `metrics_interval=1` | Every expensive/gated metric computed every frame instead of every 20th — 41 of 47 `FlockMetrics` fields populate at least once. The remaining 6 are structurally impossible here, not a gap: `target_dist_min`/`target_dist_max` are influencer-mode-only, `theta`/`theta_accel_correlation`/`theta_accel_peak_lag` are projection-mode-only, and `msd_crossover` needs more history than 40 frames provides. `hull_volume`/`tau_rho` have their own internal 10-frame ring-buffer warm-up independent of `metrics_interval`, so they only start populating partway through even here. |

## Regenerating

```bash
python scripts/generate_examples.py                # all targets
python scripts/generate_examples.py --name showcase
python scripts/generate_examples.py --dry-run       # preview without writing
```

Deterministic: every target is re-seeded to 42 regardless of what its
source preset specifies, so regeneration reproduces byte-identical
metric values. `test/crosscutting/guards/test_examples.py` asserts
each file's key set still matches the live `FlockMetrics` schema — if
that guard fails after adding a new metric, re-run the command above.
