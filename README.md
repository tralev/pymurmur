# pymurmur

A 3D starling-murmuration simulator: seven interchangeable physics
models (Pearce projection, Reynolds boids, blob/field, Vicsek,
cosmic-influencer, angle-steering, MARL), optional predator–prey
dynamics, and behavioral extensions (predator threat FSM, ecology,
wander, ripples, and more). Runs at any scale (150 → 300,000 birds),
real-time visualized (ModernGL, strict z-up 3D) or fully headless with
physically calibrated metrics (watts, joules, newtons, plus
dimensionless order/opacity/robustness observables). Deterministic
under a seed.

## Quickstart

```bash
pip install -e .                                       # core (numpy/scipy/PyYAML)
pip install -r requirements-optional.txt               # + visual/GPU (pygame/moderngl/PyGLM/numba)

python -m pymurmur                                   # defaults (projection, N=150)
python -m pymurmur --config field                    # conf/murmuration_field.yaml
python -m pymurmur --config /path/custom.yaml
python -m pymurmur --set spatial.separation_weight=6 --set flock.num_boids=500
python -m pymurmur --print-config                    # resolved effective config
python -m pymurmur --list-configs                     # discovered presets
python -m pymurmur --probe                             # capability report (GL/numba/scipy)
python -m pymurmur --no-viz --capture                  # headless + GIF/CSV/JSON to output/
```

Interactive controls while running visually: drag to orbit, scroll to
zoom, `M` cycles the 7 force modes, `+`/`-` change flock size,
click/right-click to spawn a bird/predator, `TAB` toggles the slider
HUD. Full control reference and the programmatic (`import pymurmur`)
surface: [arch.md §11](arch.md#11-cli--programmatic-surface-level-3).

## Where to look next

- **[`conf/`](conf/)** — 20 shipped, tested config presets, one per
  force mode plus character variants (field presets, EvoFlock/MARL
  configs, a 300K-bird benchmark). Full table:
  [arch.md §10](arch.md#10-shipped-config-presets-conf-level-3).
  [`conf/examples/murmuration_nested.yaml`](conf/examples/murmuration_nested.yaml)
  is a hand-maintained reference listing every config field (not a
  loadable preset).
- **[`examples/`](examples/)** — curated, committed example metrics
  captures (unlike gitignored `output/`) showing what the full metrics
  schema looks like across a few contrasting runs.
- **[`arch.md`](arch.md)** — the single architecture reference:
  module map, dependency rules, force-mode/plugin taxonomy, data
  representation, determinism/safety/scaling.
- **[`test.md`](test.md)** — test suite organization and how CI/Docker
  run it.

## Metrics & scientific output

Every run (headless or visual, with `--capture`) can export a
per-frame metrics time series — polar/nematic order, opacity Θ,
consensus robustness H₂, MSD, density scaling, physical units, and
more — serialized via `FlockMetrics.to_dict()`
(`pymurmur/analysis/metrics/flock_metrics.py`). See
[`examples/README.md`](examples/README.md) for real sample output and
what each field means.

## Research bridges

- **EvoFlock** — evolutionary inverse design (GA over an SDF obstacle
  world) → `output/evolved.yaml`. See `conf/murmuration_evo.yaml`.
- **MARL** — a gymnasium environment (`pymurmur.analysis.rl.MurmurationEnv`)
  plus a per-bird control hook; training stays external in `scripts/`.
