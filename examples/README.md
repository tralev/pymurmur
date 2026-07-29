# Example metrics captures

Three curated, **committed** example runs showing what pymurmur's
metrics output actually looks like — unlike `output/` (gitignored,
whatever you last ran locally), these are checked into the repo so
anyone can see real output without running the simulator.

Each pair (`metrics_<name>.csv` / `metrics_<name>.json`) is 180 frames,
seed 42, `metrics_detail_level=2` with `metrics_interval=20` (so the
expensive/gated metrics — `h2`, `tau_rho`, `hull_volume`,
`aspect_ratio`, `msd_slope`, etc. — actually populate 9 times across
the run instead of once or never). Field semantics are documented in
`pymurmur/analysis/metrics/flock_metrics.py` and summarized in
[arch.md §8](../arch.md#8-data-representation-level-3).

## What each one demonstrates

| File | Source | Demonstrates |
|---|---|---|
| `metrics_projection.json/csv` | `SimConfig()` defaults (projection mode, 150 birds) | The baseline most users will see first — no extensions, default kernels. |
| `metrics_influencer.json/csv` | [`conf/murmuration_influencer.yaml`](../conf/murmuration_influencer.yaml) | The only mode where `target_dist_min`/`target_dist_max` populate (every frame here) — every other example leaves them `null`. |
| `metrics_showcase.json/csv` | [`conf/murmuration_showcase.yaml`](../conf/murmuration_showcase.yaml) | `sphere_soft` boundary, `bell_zone` kernels, `hash_grid` spatial index, and all 4 of the newer behavioral extensions (SpeedNoise, NeighborAdaptiveSpeed, DynamicVisionRange, BoidStateMachine) plus Predator/Ecology — none of which any other shipped preset exercises. |

## Regenerating

```bash
python scripts/generate_examples.py            # all three
python scripts/generate_examples.py --name showcase
python scripts/generate_examples.py --dry-run   # preview without writing
```

Deterministic: every target is re-seeded to 42 regardless of what its
source preset specifies, so regeneration reproduces byte-identical
metric values. `test/crosscutting/guards/test_examples.py` asserts
each file's key set still matches the live `FlockMetrics` schema — if
that guard fails after adding a new metric, re-run the command above.
