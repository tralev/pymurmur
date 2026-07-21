# Test Suite & CI — pymurmur 3D Murmuration Simulation

> **STATUS: ACTIVE.** This document describes the test suite's organization,
> conventions, and how CI (including Docker) runs it.
> **3,157 tests collected; 2,759 run in the fast suite** (`-m "not slow and not
> gl and not gpu"`).
> **Organization: Bottom-Up / Micro-to-Macro.** The `test/` tree is layered by
> altitude, mirroring `arch.md` §2.2's bottom-up view: Level 0 (module
> interfaces, micro) up to Level 4 (system goal, macro). Directory names carry
> the level prefix (`l0_` … `l4_`) so the tree lists in micro-to-macro order —
> except `crosscutting/`, which is **deliberately unnumbered**: guards and
> perf budgets are orthogonal to the micro↔macro axis, not a rung on it (see
> [Directory Organization](#directory-organization--micro-to-macro) below).
> **Design docs:** `arch.md` — single architecture reference.
> **Framework:** `pytest` with `numpy` test helpers. GPU tests use `@pytest.mark.gpu`.
> **All Docker/CI files live in `ci/`** (workflows in `.github/workflows/`);
> **all tests live in `test/`.** This document merges the former `docker.md`
> (removed 2026-07-21) — it is now the single reference for both the test
> tree and how CI/Docker run it.

## Test Cheatsheet

### Run Commands

```bash
# All fast tests (marker-based: auto-skips GPU where unavailable)
python3 -m pytest test/ -q -m "not slow"

# CI: fast suite — what runs on every commit
pytest test/ -v -m "not slow and not gl and not gpu" --timeout=60 --tb=short

# Guard rails (architecture, docs, golden, collection count, config drift…)
pytest -m guard -v

# Golden trajectory tests only
pytest -m golden -v

# One altitude level (micro → macro)
pytest test/l0_modules/ -v         # module interfaces
pytest test/l2_integration/ -v     # subsystem wiring
pytest test/l3_subsystems/ -v      # A–F isolation
pytest test/l4_system/ -v          # system goal
pytest test/crosscutting/ -v       # guards + perf (orthogonal)

# One module's tests
pytest test/l0_modules/physics/forces/test_field.py -v

# Match by test name pattern
pytest test/ -k "test_vicsek" -v

# Performance & scaling (slow, nightly/PR-merge only)
pytest -m slow test/crosscutting/perf/ -v

# Coverage (do NOT use the pytest-cov plugin — it reloads numpy and
# breaks ~440 tests; use coverage-run instead):
python3 -m coverage run --branch --include="pymurmur/*" -m pytest -q -m "not slow"
python3 -m coverage report --show-missing
```

### Pytest Markers

| Marker | Purpose |
|--------|---------|
| `slow` | Performance/benchmark tests, run on PR merge or nightly |
| `gpu` / `gl` | Tests requiring a ModernGL GPU context (`gl` is a registered alias — kept for `-m "gl or gpu"` selection expressions; no test currently self-marks `gl` directly) |
| `golden` | Golden trajectory regression tests |
| `guard` | Guard-rail tests (architecture, docs, golden, collection count, drift) |
| `e2e` | End-to-end tests that run the full simulation |
| `acceptance` | Phase-level acceptance tests |
| `phase1`/`phase3`/`phase4` | Phase acceptance-criteria groupings |
| `s6_4` | S6.4 obstacle-engine integration tests |
| `part4_cross`/`part5_cross` | Part IV/V cross-item integration tests (engine pipeline) |

Removed 2026-07-21 as dead declarations (zero `@pytest.mark.X` usages found
anywhere in `test/`): `numba`, `pygame`, `integration`. Numba/pygame
availability is instead handled per-test via `numba_available`/
`PYGAME_AVAILABLE` skipif fixtures — see Conventions below.

### Shared Fixtures (`test/conftest.py`)

| Fixture | Returns | Use for |
|---------|---------|---------|
| `default_config` | `SimConfig()` with projection defaults | Any test needing a config |
| `spatial_config` | `SimConfig(mode="spatial", num_boids=200)` | Spatial mode tests |
| `small_flock` | `PhysicsFlock(config, N=50)` | Fast unit tests |
| `two_bird_flock` | `PhysicsFlock(config, N=2)` | Neighbor/pair tests |
| `known_positions` | `(4,3) float32` array | Deterministic position tests |
| `known_velocities` | `(4,3) float32` array | Deterministic velocity tests |
| `neighbor_idx` | `(4,3) int32` pre-computed neighbor indices | Force primitive tests |
| `gpu_available` | `bool` (session-scoped) | Conditional GPU test skipping |
| `numba_available` | `bool` (session-scoped) | Conditional numba test skipping |

Shared helpers (`_step_flock`, `_call_force`) live in `test/helpers.py` and are
imported as `from test.helpers import …`.

### Conventions & Gotchas

- Python is 3.9 — test files using `X | Y` annotations need
  `from __future__ import annotations`.
- `capture_prewarm` defaults to 60: Recorder frame-capture tests must set
  `cfg.capture_prewarm = 0` or the first 60 frames are silently skipped.
- Neighbor-index arrays use 0 as the padding value (`nbrs > 0` filters), so
  bird index 0 never appears as a neighbor slot value.
- `SimConfig.from_file(path)` is the loader (not `from_yaml`); validation
  requires `capture.frames >= 1`.
- Every new module import edge must be registered in `ALLOWED_EDGES` in
  `test/crosscutting/guards/test_architecture.py`.
- Docker/pygame-gated tests use `pytest.mark.skipif(not PYGAME_AVAILABLE, …)`
  rather than a `pygame` marker — this is why that marker was removed rather
  than kept as a second, unused mechanism.

---

## Directory Organization — Micro to Macro

All input files (YAML configs) MUST be placed within `conf/`. All output files
MUST be placed within `output/`. All test files MUST be placed within `test/`.

The tree descends bottom-up, mirroring `arch.md` §2.2 (Bottom-Up · Component
Assembly · Micro→Macro · Inside-Out): start at one module's interface, wire
modules together, isolate each functional subsystem, then assert system-level
goals. Reading the tree top-to-bottom **is** reading the architecture
bottom-up. `crosscutting/` sits beside this ladder rather than on it — guards
and perf budgets apply at every altitude simultaneously, so giving it a level
number would misstate what it does (this is also why it lists alphabetically
*before* `l0_modules/` in a bare `ls` — a harmless side effect of dropping its
numeric prefix, not a claim that it comes "before" Level 0).

```
test/
├── __init__.py
├── conftest.py                  # shared fixtures (see cheatsheet)
├── helpers.py                   # _step_flock, _call_force
├── regenerate_golden.py         # golden trajectory generator (not collected)
├── data/                        # golden .npz baselines (CI-validated paths)
│
├── l0_modules/                  # ── Level 0: MODULE INTERFACES (micro) ──
│   │                            # one mirror directory per pymurmur package;
│   │                            # test for pymurmur/<pkg>/<mod>.py lives at
│   │                            # l0_modules/<pkg>/test_<mod>*.py
│   ├── core/                    # test_types, test_config, test_config_validation,
│   │                            # test_logging (S5.6 print-guard + structured log)
│   ├── physics/                 # boid, flock, occlusion, steric, obstacles,
│   │   │                        # composition, spatial-index contract,
│   │   │                        # edge cases, field×extensions integration
│   │   ├── forces/              # one file per force mode + kernels, terms,
│   │   │                        # primitives, mode contract
│   │   └── extensions/          # manager, predator, threat, wander, ripple, ecology
│   ├── simulation/               # engine step order, fixed-timestep accumulator
│   ├── viz/                      # renderer (+ impostor split), shaders, camera,
│   │                             # trails (4 modes), wings, colour, density,
│   │                             # dual view, hud, input, mesh registry
│   ├── capture/                  # recorder, mpl_recorder, cinematic sweep
│   └── analysis/                 # metrics (schema/motion/expensive), h2, presets,
│                                  # perf+quality governor, phase diagram, density
│                                  # scaling, rewards, evoflock, evolved-yaml guard,
│                                  # marl (P12 placeholder)
│
├── l2_integration/               # ── Level 2: SUBSYSTEM WIRING (meso) ──
│   ├── test_engine_pipeline.py   # 6-stage engine order under live mutations
│   ├── test_render_contract.py   # frame()/headless_frame() never step the sim
│   ├── test_capture_pipeline.py  # step → on_frame → serialize round-trip
│   ├── test_config_contract.py   # facade, field map, nested↔flat contract
│   ├── test_cross_subsystem.py   # index swap, threat pipeline, instance schema
│   └── test_defect_regressions.py # D1–D21 whole-system defect regression guards
│
├── l3_subsystems/                # ── Level 3: FUNCTIONAL DECOMPOSITION (A–F) ──
│   │                             # each subsystem isolated, deps mocked
│   ├── test_subsystem_a.py       # A — Entry & Configuration
│   ├── test_subsystem_b.py       # B — Simulation Engine
│   ├── test_subsystem_c.py       # C — Visualization & Input
│   ├── test_subsystem_d.py       # D — Capture & Export
│   ├── test_subsystem_e.py       # E — Physics & Forces
│   └── test_subsystem_f.py       # F — Metrics & Analysis
│
├── l4_system/                    # ── Level 4: SYSTEM GOAL (macro) ──
│   │                             # "load config, run modes, viz/capture
│   │                             #  optionally, run headlessly"
│   ├── test_cli.py               # __main__ dispatch: parse_args, load_config
│   ├── test_cli_e2e.py           # CLI flags end-to-end (--set, --print-config…)
│   ├── test_probe.py             # --probe capability probing
│   ├── test_config_resolution.py # name → conf/*.yaml → path resolution
│   ├── test_config_files.py      # every shipped conf/*.yaml is valid
│   ├── test_facade.py            # pymurmur.Simulation public API
│   ├── test_e2e.py               # headless end-to-end scenarios
│   ├── test_mode_switch_no_crash.py  # cycle all modes live (guard)
│   └── acceptance/               # phase gates: whole-system acceptance criteria
│       └── test_phase1.py, test_phase3.py, test_phase4.py
│
└── crosscutting/                 # ── unnumbered: ORTHOGONAL TO ALL LEVELS ──
    ├── guards/                   # repo guard-rails (CI: guard-rails.yml, -m guard)
    │   ├── test_architecture.py       # ALLOWED_EDGES import-DAG enforcement
    │   ├── test_imports.py            # no-upward-import rules
    │   ├── test_docs.py               # arch.md/test.md link + topology validation
    │   ├── test_golden.py             # golden trajectory regression (test/data)
    │   ├── test_determinism.py        # same-seed identity matrix + subprocess leg
    │   ├── test_config_drift.py       # every SimConfig field used in source
    │   ├── test_collection_count.py   # per-level floors — suite never shrinks
    │   ├── test_composers.py          # G1 — every public L0 atom has a call site
    │   ├── test_ci_workflow_integrity.py  # the CI YAML itself, as CI executes it
    │   └── test_strictly_3d.py        # P14.3 — no 2D spatial arrays in physics/
    └── perf/                     # slow benchmarks (nightly / PR merge)
        ├── test_budgets.py       # P1 — per-mode step-time budget table (MODE_REGISTRY-parametrized)
        ├── test_performance.py   # P2/P3/P4 — scaling checkpoints, memory audit, soak
        └── test_scaling.py       # O(N) / O(N log N) scaling fits
```

**Placement rules (decide bottom-up):**

1. Does it exercise **one module's interface** (possibly with direct
   collaborators)? → `l0_modules/<pkg>/`.
2. Does it wire **multiple subsystems** and assert their contract? → `l2_integration/`.
3. Does it isolate **one subsystem A–F** with the rest mocked? → `l3_subsystems/`.
4. Does it verify a **system-level goal** — CLI, facade, an end-to-end run, a
   phase acceptance gate? → `l4_system/` (gates in `l4_system/acceptance/`).
5. Does it pin an **implementation invariant** — imports, goldens, drift,
   collection count, step-time — that applies at every altitude? → `crosscutting/`
   (`guards/` or `perf/`).

File names are module-first (`test_<module>[_<aspect>].py`), not phase-first;
phase provenance lives in docstrings (`P8.10: …`). The `guard` marker — not
the directory — is what CI's guard-rails workflow selects, so module-shaped
guards (e.g. `l0_modules/analysis/test_evolved_yaml.py`,
`l0_modules/physics/forces/test_vicsek_core.py`) stay with their modules.

---

## Level ↔ arch.md Mapping

`arch.md` §2 documents **two complementary views** of the same system — the
test tree draws directory boundaries from both, which is why the mapping
below is not a single clean 1:1 correspondence:

| Test level | Draws on | What's tested |
|------------|----------|----------------|
| `l0_modules/` | §2.2 Bottom-Up Levels 0–1 (Atoms, Assemblies) | Individual modules — force primitives, kernels, SDF, up through PhysicsFlock/ForceMode/ExtensionManager |
| `l2_integration/` | §2.2 Bottom-Up Level 2 (Subsystems) | Engine/capture pipelines wiring SimulationEngine, Visualizer, Renderer3D, Recorder together |
| `l3_subsystems/` | §2.1 Top-Down Level 1 (seven functional subsystems A–F) | Each subsystem in isolation with mocked dependencies |
| `l4_system/` | §2.1 Top-Down Level 0 (Goal) and §2.2 Bottom-Up Level 3 (System) | CLI dispatch, facade, config resolution, E2E, phase gates |
| `crosscutting/` | Neither — orthogonal to both views | Import DAG, goldens, drift, collection floors, perf/scaling |

Subsystem key: A = Entry & Configuration, B = Simulation Engine,
C = Visualization & Input, D = Capture & Export, E = Physics & Forces,
F = Metrics & Analysis.

---

## Guard Rails (`crosscutting/guards/` + `.github/workflows/guard-rails.yml`)

Guards run on every commit via `pytest -m guard`. Each guard fails loudly and
explains how to update itself when a change is intentional:

| Guard | Protects against |
|-------|-----------------|
| `test_architecture.py` | New import edges not registered in `ALLOWED_EDGES`; layering violations |
| `test_imports.py` | Upward imports (physics→viz, engine→pygame, …) |
| `test_golden.py` | Behavioural drift in the 5 force modes (bit-exact vs `test/data/golden_*.npz`; regenerate with `test/regenerate_golden.py`) |
| `test_determinism.py` | Same-seed non-determinism across mode × threads × jitter, plus one subprocess leg per mode |
| `test_docs.py` | Broken intra-repo links anywhere in the repo's `.md` files; arch.md ↔ test.md guard-topology sync; stale retired-scheme references |
| `test_config_drift.py` | Orphan `SimConfig` fields no source file reads |
| `test_collection_count.py` | Silent test loss — floors per level, per module mirror, and in total |
| `test_composers.py` | Dead L0 atoms (public functions with zero call sites) |
| `test_ci_workflow_integrity.py` | Bugs in the CI YAML itself — invalid bash, dangling `needs:`, unrendered `${{ }}`, summary gate not covering every job |
| `test_strictly_3d.py` | 2D spatial arrays creeping into `physics/`; missing `depth > 0` validation |

System-level guards live at their altitude (`l4_system/test_mode_switch_no_crash.py`);
module-level guards live with their module. The marker selects them all.

---

## Continuous Integration & Docker

> Merged from the former `docker.md` (removed 2026-07-21). All tests —
> OpenGL/GPU, gymnasium/MARL, scipy-backed physics, ruff, mypy, and every
> plain CPU test — run **inside Docker** in CI, via `docker compose`. Nothing
> in either workflow installs project dependencies onto the bare GitHub
> Actions runner; every job's actual test/lint/type-check execution happens
> through a container built from `ci/Dockerfile` or `ci/Dockerfile.gpu`.

### File inventory

| File | Role |
|---|---|
| [ci/Dockerfile](ci/Dockerfile) | CPU-only headless image `pymurmur-test` (`python:${PYTHON_VERSION}-slim`, default 3.12) — fast suite, E2E, guards, lint |
| [ci/Dockerfile.gpu](ci/Dockerfile.gpu) | `pymurmur-test-gpu`, extends the CPU image with GL/SDL deps + moderngl/PyGLM/pygame |
| [ci/entrypoint.sh](ci/entrypoint.sh) / [ci/entrypoint-gpu.sh](ci/entrypoint-gpu.sh) | Xvfb wrappers (`:99`, GLX + render) so GL tests run without a display |
| [ci/docker-compose.yml](ci/docker-compose.yml) | Profiles: `fast`, `e2e`, `slow`, `gpu`, `capture`, `lint`, `full` (+ guard/subsystem services in `fast`) |
| [ci/docker-compose.gpu.yml](ci/docker-compose.gpu.yml) | Override adding the nvidia runtime + driver env for real-GPU runs |
| [.github/workflows/test.yml](.github/workflows/test.yml) | Main CI: fast matrix, guards/subsystems, E2E, slow, GPU (nightly), lint — every job runs via `docker compose` |
| [.github/workflows/guard-rails.yml](.github/workflows/guard-rails.yml) | P14 guard rails: `guard-rail-dag`, `guard-rail-golden`, `guard-rail-config-drift`, `guard-rail-3d`, `guard-rail-doc-links`, `guard-rail-collection-count`, `guard-rail-mypy`, `guard-rail-evolved`, `guard-rail-composers` + a merge-blocking `guard-rails-summary` gate — every substantive job runs `docker run pymurmur-test:latest pytest <path>` |
| [requirements.txt](requirements.txt) | Production deps only: numpy, scipy, PyYAML |
| [requirements-optional.txt](requirements-optional.txt) | numba, pygame, moderngl, PyGLM, Pillow, matplotlib, gymnasium (sb3 commented — scripts-only) |
| [requirements-test.txt](requirements-test.txt) | pytest (+cov/xdist/timeout), ruff, mypy |

**Image strategy:** two images, layered. The CPU image installs
production + headless-safe optional deps (numba, pygame, Pillow,
matplotlib, gymnasium — the GPU-free capture fallback and MARL
scaffolding must work *without* GL; that is the point) and bakes
`output/evolved.yaml` at build time (P0.16 — so `guard-rail-evolved` and
`test_evolved_yaml.py` never fail on a fresh checkout for lack of an
artifact). The GPU image extends it with Mesa/EGL/SDL libraries and the
viz stack, and wraps every command in Xvfb via the entrypoint.

### Quick reference

```bash
# Build images
docker build -f ci/Dockerfile -t pymurmur-test .
docker build -f ci/Dockerfile.gpu -t pymurmur-test-gpu .

# Fast suite (every commit; also runs the guard + subsystem services)
docker compose -f ci/docker-compose.yml --profile fast up --build --abort-on-container-exit

# E2E · slow · lint
docker compose -f ci/docker-compose.yml --profile e2e up --build
docker compose -f ci/docker-compose.yml --profile slow up
docker compose -f ci/docker-compose.yml --profile lint up

# GPU tests (software GL by default; add the override for a real NVIDIA GPU)
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile gpu up

# Capture smoke: headless llvmpipe GIF, exercises the CAPTURE_* env overrides
docker compose -f ci/docker-compose.yml --profile capture up

# Everything (nightly-equivalent)
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile full up

# Single test file / interactive debugging
docker run --rm -v $(pwd)/output:/app/output pymurmur-test \
  pytest test/l0_modules/physics/test_steric.py -v
docker run --rm -it -v $(pwd)/output:/app/output pymurmur-test bash

# A single guard rail, exactly as guard-rails.yml runs it
docker run --rm -v $(pwd)/output:/app/output pymurmur-test \
  pytest test/crosscutting/guards/test_architecture.py -v
```

All services mount `./output` (JUnit XML `test-results-*.xml`, coverage
XML, capture artifacts) and `./conf` read-only.

### Compose profiles

| Profile | Service(s) | Selects | Timeout |
|---|---|---|:---:|
| `fast` | `test-fast` + `test-guards` + `test-subsystems` | `-m "not slow and not gl and not gpu and not pygame"` + guard/subsystem files | 60 s |
| `e2e` | `test-e2e` | `-m e2e` (l4_system e2e/CLI/config tests) | 120 s |
| `slow` | `test-slow` | `-m "slow and not gl and not gpu"` (installs stable-baselines3 best-effort for the P11.6/P12.3 experiments) | 600 s |
| `gpu` | `test-gpu` | `-m "gl or gpu"` under Xvfb (llvmpipe; nvidia via override file) | 120 s |
| `capture` | `capture` | `python -m pymurmur --config field --no-viz --capture` at 400×300×120f | — |
| `lint` | `lint` | ruff + mypy | — |
| `full` | all of the above | nightly superset | — |

### Environment variables

| Variable | Purpose | Default |
|---|---|:---:|
| `PYMURMUR_TEST` | enable test-only code paths | `0` |
| `PYMURMUR_CONFIG_DIR` | override config search path | `conf/` |
| `CAPTURE_W` / `CAPTURE_H` / `CAPTURE_FRAMES` / `CAPTURE_OUT` | headless-capture overrides (P8.7/P8.9; precedence YAML < env < CLI) | config |
| `NUMBA_NUM_THREADS` | cap numba parallelism in CI | `4` |
| `LIBGL_ALWAYS_SOFTWARE` | force Mesa llvmpipe even when a GPU exists (deterministic CI rendering) | unset |
| `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` | GPU selection for the nvidia runtime (override file) | `all` / `compute,graphics,utility` |
| `__GL_SYNC_TO_VBLANK` | disable vsync in headless GPU runs (override file) | `1` |

### How CI actually runs (both workflows, Docker end to end)

Two workflows. Both run on push to `main` (docs/conf/output changes
ignored), on PRs to `main`, and nightly.

**1. `test.yml` — "Test pymurmur" (nightly 03:00 UTC).** Every job builds
`ci/Dockerfile` (or `.gpu`) with Buildx GHA layer caching, then drives the
matching Compose profile — no bare-runner `pip install` remains outside a
container:

| Job | When | Compose profile | What it runs |
|---|---|---|---|
| `test-fast` | always; Python **3.11 + 3.12** matrix (`--build-arg PYTHON_VERSION`) | `fast` | `pytest test/ -m "not slow and not gl and not gpu and not pygame"` with coverage |
| `test-imports` | always | `fast` (`test-guards` + `test-subsystems` services) | `crosscutting/guards/test_imports.py` + `l3_subsystems/test_subsystem_a–f.py` |
| `test-e2e` | always | `e2e` | `l4_system/` e2e + config-resolution + config-files (`-m e2e`) |
| `test-slow` | PR / nightly | `slow` | `crosscutting/perf/{test_performance,test_scaling,test_budgets}.py -m slow`, 600 s timeout |
| `test-gpu` | nightly only | `gpu` | Xvfb + Mesa llvmpipe inside `pymurmur-test-gpu`; `-m "gl or gpu"` across `l0_modules/viz/` |
| `lint` | always | `lint` | `ruff check pymurmur/ test/` + `mypy pymurmur/ --ignore-missing-imports` |

**2. `guard-rails.yml` — "P14 Guard Rails" (nightly 04:00 UTC).** Nine
substantive jobs, each `docker run pymurmur-test:latest pytest <file>`
against the already-built fast image (no per-job Compose service needed —
every job is a single `pytest` invocation); `guard-rails-summary` aggregates
and **blocks merge** on any failure:

| Job | Enforces |
|---|---|
| `guard-rail-dag` | P14.1 — architecture DAG matrix (`test_architecture.py`) + stale old-scheme identifier scan (`test_docs.py`) |
| `guard-rail-golden` | P0.1 golden trajectories (`test_golden.py`) + P13.5 determinism runs (`test_determinism.py`) |
| `guard-rail-config-drift` | P14.2 — every config leaf read by ≥ 1 non-config module (`test_config_drift.py`) |
| `guard-rail-3d` | P14.3 — no 2D spatial arrays in `physics/`; `depth > 0` validation (`test_strictly_3d.py`) |
| `guard-rail-doc-links` | P14.4 — every repo `.md` file's intra-repo links resolve; arch.md ↔ test.md topology sync (`test_docs.py`) |
| `guard-rail-collection-count` | P14.5 — collected-test floors per level/module (`EXPECTED_MINIMUMS`, `test_collection_count.py`) |
| `guard-rail-mypy` | type-check gate (`mypy pymurmur/ --ignore-missing-imports`, inside the container) |
| `guard-rail-evolved` | `output/evolved.yaml` artifact validity (`test_evolved_yaml.py`) — the artifact is baked in at image build time, so no separate generation step runs in CI |
| `guard-rail-composers` | G1 — every public L0 atom has ≥ 1 cross-module call site (`test_composers.py`) |
| `guard-rails-summary` | merge-blocking roll-up over the nine jobs above |

### Pipeline matrix & typical durations

| Trigger | Fast | Guards | E2E | Slow | GPU | Lint |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Push to main | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| PR to main | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Nightly | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Fast ≈ 30–90 s · guards ≈ 10–30 s · e2e ≈ 1–2 min · slow ≈ 5–15 min ·
GPU ≈ 1–3 min · lint ≈ 15–30 s, all plus Docker build time (cached via
Buildx GHA `type=gha` cache after the first run on a given layer set).
Push total ≈ 3–4 min; nightly ≈ 10–25 min.

### GPU testing approach

CI uses **software OpenGL** (Mesa llvmpipe under Xvfb) — no paid GPU
runners. The GPU compose profile runs the same way by default; adding
`ci/docker-compose.gpu.yml` switches to the real NVIDIA runtime
(requires the NVIDIA Container Toolkit).

| Environment | Renderer | Notes |
|---|---|---|
| GitHub Actions / local Docker without GPU | Mesa llvmpipe via Xvfb | set `LIBGL_ALWAYS_SOFTWARE=1` for consistency |
| Local Docker + NVIDIA | real GPU via nvidia runtime | use the `.gpu.yml` override |

Limitations of software GL: rendering benchmarks are not
representative (skip `test_bench_*` renderer timings); FBO readback can
differ slightly from hardware drivers — pixel assertions must be
approximate.

### macOS development

macOS has **no native OpenGL context** (no display server, no GPU
visible to ModernGL). The following tests **skip automatically** when
run outside Docker:

| Test / class | Skip message | File |
|---|---|---|
| `TestRenderer3D::test_renderer_windowed_context` | `No display available for windowed context` | `test/l0_modules/viz/test_renderer.py` |
| `TestVisualizerIntegration::test_visualizer_windowed_frame` | `Windowed context creation failed (no display)` | `test/l0_modules/viz/test_renderer.py` |
| `@pytest.mark.gpu`-decorated tests (~80+) | `GPU not available` | `test/l0_modules/viz/test_renderer.py`, `test_renderer_impostor.py`, `test_camera.py`, `test_input.py`, `test_shaders.py`, `test_hud.py`, etc. |

To run the **full GPU test suite** on macOS, use Docker:

```bash
# Build and run GPU tests (software GL via Xvfb)
docker compose -f ci/docker-compose.yml --profile gpu up --build

# Or with real GPU (if you have an eGPU + NVIDIA Container Toolkit)
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile gpu up

# Quick smoke: just the GPU-marked tests
docker run --rm -v $(pwd)/output:/app/output pymurmur-test-gpu \
  pytest -m "gl or gpu" -v
```

This uses the `ci/entrypoint-gpu.sh` Xvfb wrapper (display `:99`,
GLX + software rendering). See [GPU testing approach](#gpu-testing-approach)
above for Mesa llvmpipe vs. real GPU trade-offs.

### Local development (outside Docker)

- `requires-python >= 3.9`; mypy targets 3.9. The dev-machine baseline
  interpreter is `python3` (3.9) — source files need
  `from __future__ import annotations` for `X | Y` annotations. The
  Docker images run 3.12 (matrix-built at 3.11 too, see `test-fast`) and
  CI runs both, so both ends of the supported range are exercised.
- Fast suite: `python3 -m pytest -q -m "not slow"` (~1 min). Guards
  only: `-m guard`. GL tests auto-skip without a context.
- Local coverage: prefer `python3 -m coverage run --branch -m pytest …`
  over the `--cov` plugin if you hit numpy-reload `_NoValueType`
  failures (a known local-environment issue; the in-container `--cov`
  path is unaffected).

```bash
pip install -r requirements.txt -r requirements-test.txt   # + requirements-optional.txt for viz/numba/MARL
python3 -m pytest -q -m "not slow"
```

---

## Index A — Implemented Ideas → Test Files

Where to look when you want the tests for a concept. Paths are relative to
`test/`; `l0m` abbreviates `l0_modules`.

| Idea / invariant | Test files |
|---|---|
| **Determinism** (same seed → bit-identical; numba↔numpy parity; threads/jitter axes; subprocess leg) | `crosscutting/guards/test_determinism.py`, `l0m/physics/test_flock.py` |
| **Golden trajectory regression** (5 modes × 2 boundaries, bit-exact) | `crosscutting/guards/test_golden.py` (+ generator `regenerate_golden.py`) |
| **Import-DAG / layering enforcement** (`ALLOWED_EDGES`) | `crosscutting/guards/test_architecture.py`, `test_imports.py`; waiver-removal in `l2_integration/test_render_contract.py` |
| **Suite never silently shrinks** (collection floors) | `crosscutting/guards/test_collection_count.py` |
| **Docs stay linked & in sync** (arch.md ↔ test.md, retired schemes absent) | `crosscutting/guards/test_docs.py` |
| **No orphan config fields** | `crosscutting/guards/test_config_drift.py` |
| **No dead L0 atoms** (every public atom has a caller) | `crosscutting/guards/test_composers.py` |
| **CI YAML correctness** (valid bash, dangling `needs:`, summary-gate completeness) | `crosscutting/guards/test_ci_workflow_integrity.py` |
| **Strictly-3D invariant** (no 2D spatial arrays; `depth > 0`) | `crosscutting/guards/test_strictly_3d.py` |
| **Config system** (nested dataclasses, YAML I/O, validation, flat↔nested map) | `l0m/core/test_config.py`, `test_config_validation.py`, `l2_integration/test_config_contract.py` |
| **Config loading & resolution** (search path, shipped files) | `l4_system/test_config_resolution.py`, `test_config_files.py` |
| **No `print()` in package sources / structured run logging** | `l0m/core/test_logging.py` |
| **CLI dispatch & flags** (`--set`, `--probe`, `--list-configs`…) | `l4_system/test_cli.py`, `test_cli_e2e.py`, `test_probe.py` |
| **Public facade** (`pymurmur.Simulation`) | `l4_system/test_facade.py`, `l2_integration/test_config_contract.py` |
| **Headless end-to-end runs** | `l4_system/test_e2e.py`, `test_mode_switch_no_crash.py` |
| **Phase acceptance gates** | `l4_system/acceptance/test_phase{1,3,4}.py` |
| **Subsystem isolation (A–F)** | `l3_subsystems/test_subsystem_{a..f}.py` |
| **Whole-system defect regressions (D1–D21)** | `l2_integration/test_defect_regressions.py` |
| **Engine step order & command queue** | `l0m/simulation/test_engine.py`, `l2_integration/test_engine_pipeline.py` |
| **Fixed-timestep accumulator + render lerp** | `l0m/simulation/test_accumulator.py` |
| **Force-mode contract** (registry, needs_index, active mask) | `l0m/physics/forces/test_mode_contract.py` |
| **Force primitives** (sep/align/cohere/noise; property-based) | `l0m/physics/forces/test_base.py`, `test_force_primitives_properties.py`, `test_force_terms.py` |
| **Projection mode** (Pearce occlusion-driven) | `l0m/physics/forces/test_projection.py`, `l0m/physics/test_occlusion.py` |
| **Spatial mode** (+ hybrid filter, variants, numba kernels) | `l0m/physics/forces/test_spatial_variants.py`, `test_forces_hybrid.py`, `test_kernels.py` |
| **Field/blob mode** (anchors, targets, cavity, buoyancy) | `l0m/physics/forces/test_field.py`, `test_field_units.py`, `l0m/physics/test_field_integration.py` |
| **Vicsek mode** (core, memory term, species predator-prey) | `l0m/physics/forces/test_vicsek.py`, `test_vicsek_core.py`, `test_vicsek_species.py` |
| **Influencer mode** | `l0m/physics/forces/test_influencer.py` |
| **Angle mode** | `l0m/physics/forces/test_angle.py` |
| **Force dispatch** | `l0m/physics/forces/test_forces.py` |
| **Flock state & spatial indices** (SoA, RNG, hash grid, KD-tree) | `l0m/physics/test_flock.py`, `test_spatial_index_contract.py` |
| **Integration kernel & boundaries** | `l0m/physics/test_boid.py` |
| **Steric repulsion** | `l0m/physics/test_steric.py` |
| **Obstacles / SDF primitives** | `l0m/physics/test_obstacles.py` |
| **Active-mask (holey array) survival** | `l0m/physics/test_holey_mask_composition.py`, `test_composition.py` |
| **Extensions** (manager, predator FSM, threat, wander, ripple, ecology) | `l0m/physics/extensions/*` |
| **Physics edge cases / branch gaps** | `l0m/physics/test_edge_cases.py` |
| **Metrics** (order params, gating, schema, motion, expensive, H₂) | `l0m/analysis/test_metrics*.py`, `test_h2.py`, `test_cross_element.py` |
| **Presets** | `l0m/analysis/test_presets.py` |
| **Perf diagnostics + adaptive quality governor** | `l0m/analysis/test_perf.py`, `test_quality.py`; ladder actions in `l0m/viz/test_visualizer_quality.py` |
| **Phase diagram / density scaling experiments** | `l0m/analysis/test_phase_diagram.py`, `test_density_scaling.py` |
| **Rewards (shared MARL/Evo scalarization)** | `l0m/analysis/test_rewards.py` |
| **EvoFlock SSGA evolution** (+ evolved artifact guard) | `l0m/analysis/test_evoflock.py`, `test_evolved_yaml.py` |
| **MARL bridge (P12 — pending)** | `l0m/analysis/test_marl.py` (placeholder stub) |
| **Renderer** (impostors, depth cues, buffers, HUD GL) | `l0m/viz/test_renderer.py`, `test_renderer_impostor.py` |
| **Shaders & meshes** (GLSL, tetra/wings, sky, mesh registry) | `l0m/viz/test_shaders.py`, `test_wings.py`, `test_mesh_registry.py` |
| **Camera** (orbit, cinematic sweep) | `l0m/viz/test_camera.py`, `l0m/capture/test_cinematic.py` |
| **Trails (4 modes + growth/degenerate/FBO lifecycle)** | `l0m/viz/test_trails.py` |
| **Colour channels & themes** | `l0m/viz/test_colour.py` |
| **Density (alpha-accumulation) mode** | `l0m/viz/test_density.py` |
| **Dual view** | `l0m/viz/test_dual_view.py`, `l0m/viz/test_cross_element.py` |
| **HUD sliders (logic)** | `l0m/viz/test_hud.py` |
| **Input → config bridge** | `l0m/viz/test_input.py` |
| **Render purity contract** (render never steps sim) | `l2_integration/test_render_contract.py` |
| **Capture** (recorder, GIF/CSV/JSON, prewarm, mpl fallback) | `l0m/capture/test_recorder.py`, `test_mpl_recorder.py`, `l2_integration/test_capture_pipeline.py` |
| **Step-time budget table** (data-driven, MODE_REGISTRY-parametrized) | `crosscutting/perf/test_budgets.py` |
| **Scaling checkpoint ladder** (150/1.5K/16K/50K/300K, tier assertions) | `crosscutting/perf/test_performance.py::TestScalingCheckpoints` |
| **Full-inventory memory audit at N=300K** | `crosscutting/perf/test_performance.py::TestMemoryAtEachCheckpoint` |
| **Long-run stability soak** (20K-frame nightly tier; 24h release-gate tier) | `crosscutting/perf/test_performance.py::TestSoak`; `scripts/soak_release_gate.py` (manual, not CI-wired by design) |
| **Step-time / memory budgets; O(N)/O(N log N) complexity claims** | `crosscutting/perf/test_scaling.py` |

## Index B — Test Files → Implemented Ideas

One line per file: what idea(s) it pins down. Grouped micro → macro.

### l0_modules/core/
| File | Idea |
|---|---|
| `test_types.py` | Math helpers (Rodrigues, min_image, smoothstep…), FlockArrays, protocols |
| `test_config.py` | SimConfig nested dataclasses, YAML I/O, unknown-key warning |
| `test_config_validation.py` | `validate()` cross-field rules and range clamps |
| `test_logging.py` | S5.6: no `print()` in package sources (AST guard), structured log, CLI flag |

### l0_modules/physics/ (+ forces/, extensions/)
| File | Idea |
|---|---|
| `test_boid.py` | `integrate()`: speed clamps, boundary modes, dt scaling; array init helpers |
| `test_flock.py` | PhysicsFlock lifecycle, single RNG, indices, add/remove, determinism |
| `test_occlusion.py` | Spherical-cap occlusion: δ̂/Θ ranges, blind angle, anisotropy, SoA parity |
| `test_steric.py` | 1/d² steric repulsion behaviour |
| `test_obstacles.py` | SDF primitives (5), collision helpers, ObstacleScene |
| `test_composition.py` | Holey-mask contract for composed force pipelines |
| `test_holey_mask_composition.py` | All modes survive inactive slots mid-array |
| `test_spatial_index_contract.py` | KDTreeIndex global indices, boxsize toroidal queries |
| `test_edge_cases.py` | Branch gaps: predator anti-parallel, wander fallbacks, field cavity/clamp |
| `test_field_integration.py` | Field × ripple/wander coupling, egress arc, cycle periodicity |
| `forces/test_base.py` | Force primitive kernels (sep 1/d², bounded cohesion, noise scale) |
| `forces/test_force_primitives_properties.py` | Property-based invariants per primitive |
| `forces/test_force_terms.py` | ForceTerm compose/enable/gain reducer |
| `forces/test_forces.py` | MODE_REGISTRY dispatch, invalid mode, mid-run switch |
| `forces/test_forces_hybrid.py` | Hybrid filter, predator escape, jitter, coherence gate, batch query |
| `forces/test_kernels.py` | Numba kernel ↔ numpy parity (numba-marked) |
| `forces/test_mode_contract.py` | ForceMode ABC: instantiable, active mask, no import cycle |
| `forces/test_projection.py` | Projection mode: δ computation, θ cache, blind/anisotropy effects |
| `forces/test_spatial_variants.py` | Spatial-mode weight isolation (sep/align/cohere/noise only) |
| `forces/test_field.py` / `test_field_units.py` | Field mode L1 behaviour / L0 functions (anchors, targets, phases) |
| `forces/test_vicsek.py` / `test_vicsek_core.py` / `test_vicsek_species.py` | Order transition / memory + tangent noise (guard) / predator-prey (guard) |
| `forces/test_influencer.py` | Influencer parity + Lissajous target tracking |
| `forces/test_angle.py` | Angle-mode steering |
| `extensions/test_extensions.py` | ExtensionManager + all extensions no-op/enable matrix |
| `extensions/test_predator.py` / `test_threat.py` | Threat FSM, panic/blackening / FSM baselines |
| `extensions/test_wander.py` / `test_ripple.py` / `test_ecology.py` | Bounded wander path / ripple envelopes / day-length & roosting |

### l0_modules/simulation/
| File | Idea |
|---|---|
| `test_engine.py` | SimulationEngine: init, step, frame count, reset, no-viz imports |
| `test_accumulator.py` | Fixed-dt accumulator, spike clamp, lerped render_positions |

### l0_modules/viz/
| File | Idea |
|---|---|
| `test_renderer.py` | Renderer3D: headless FBO, instances, depth cues, HUD GL |
| `test_renderer_impostor.py` | Impostor rendering + depth cue tests split out of `test_renderer.py` (~1,000-line file limit) |
| `test_shaders.py` | GLSL sources compile-ready; tetrahedron mesh geometry |
| `test_mesh_registry.py` | S4.4a: mesh data, render-mode recommendation, theme materials |
| `test_camera.py` | OrbitCamera math: rotate/zoom clamps, auto-rotate, matrices |
| `test_trails.py` | 4 trail modes, buffer growth, degenerate inputs, FBO lifecycle, renderer wiring |
| `test_wings.py` | Winged flapping mesh, gradient sky quad |
| `test_colour.py` | Per-bird hue packing, theme material tables |
| `test_density.py` | Alpha-accumulation density rendering |
| `test_dual_view.py` | Dual-view + orthographic presets |
| `test_hud.py` | SliderHUD value mapping, hit-test, config writes (CPU) |
| `test_input.py` | Keyboard/mouse → SimConfig bridge, full key map |
| `test_visualizer_quality.py` | P8.6 degradation ladder + recovery actions (CPU, mocked GL) |
| `test_cross_element.py` | Phase 8 render/capture cross-element wiring |

### l0_modules/capture/
| File | Idea |
|---|---|
| `test_recorder.py` | Recorder: capture_every, prewarm, GIF/CSV/JSON, no-PIL fallback |
| `test_mpl_recorder.py` | Matplotlib fallback recorder |
| `test_cinematic.py` | Cinematic sweep math, prewarm skip, env overrides |

### l0_modules/analysis/
| File | Idea |
|---|---|
| `test_metrics.py` | FlockMetrics + collector: order params, gating, snapshot |
| `test_metrics_schema.py` | `to_dict()` JSON round-trip schema |
| `test_metrics_motion.py` | Silhouette, η(m), robust gyration, motion metrics |
| `test_metrics_expensive.py` | Shape PCA, gyration, MSD, θ′, τ_ρ |
| `test_h2.py` | H₂ k-NN Laplacian robustness, cost-optimal m* |
| `test_cross_element.py` | P9 metric chains working together |
| `test_presets.py` | PRESETS validity + non-mutation |
| `test_perf.py` | EMA timing, bottleneck classification, QualityGovernor internals |
| `test_quality.py` | Governor hysteresis, ladder state machine (unit) |
| `test_phase_diagram.py` | Vicsek η×D sweep |
| `test_density_scaling.py` | N-sweep power-law fits (sweeps are `@slow`), save/load round-trip |
| `test_rewards.py` | Weighted composite reward, linearity |
| `test_evoflock.py` | SSGA: worst-of-4, crossover, objectives, SDF collisions |
| `test_evolved_yaml.py` | Evolved-config artifact validity (guard) |
| `test_marl.py` | **P12 placeholder** — module-skipped until MARL bridge lands |

### l2_integration/ — subsystem wiring
| File | Idea |
|---|---|
| `test_engine_pipeline.py` | 6-stage engine order holds across steps + mutations; only engine imports flock+forces |
| `test_render_contract.py` | `frame()`/`headless_frame()` never step the sim; flock↔forces cycle break |
| `test_capture_pipeline.py` | step → on_frame → serialize round-trip; buffer growth under mutations |
| `test_config_contract.py` | Facade re-exports, `_FIELD_MAP` completeness, nested↔flat integrity, no GL imports in config classes |
| `test_cross_subsystem.py` | Index swap mid-run, threat/evasion pipeline, InstanceSchema packing |
| `test_defect_regressions.py` | D1–D21 whole-system defect regression guards — contracts only visible through the full engine, not at module level |

### l3_subsystems/ — A–F isolation
| File | Idea |
|---|---|
| `test_subsystem_a.py` | Entry & Configuration isolated: defaults, YAML flattening, roundtrip |
| `test_subsystem_b.py` | Engine isolated: step order, headless run, reset, live mutation |
| `test_subsystem_c.py` | Viz & Input isolated: no simulation imports, wiring |
| `test_subsystem_d.py` | Capture isolated: metrics-only mode, GIF/CSV/JSON validity |
| `test_subsystem_e.py` | Physics isolated: two-pass architecture, all modes' force validity |
| `test_subsystem_f.py` | Metrics isolated: O(N) fast metrics, gating levels, intervals |

### l4_system/ — system goal
| File | Idea |
|---|---|
| `test_cli.py` | `__main__` arg parsing, config loading, `--list-configs`, main() dispatch |
| `test_cli_e2e.py` | CLI flags end-to-end: `--set`, `--print-config`, `--fullscreen` (guard) |
| `test_probe.py` | `--probe` capability probing (GPU/numba/pygame detection) |
| `test_config_resolution.py` | `load_config()` search-path order: name → `conf/` → path |
| `test_config_files.py` | Every shipped `conf/*.yaml` parses and has required sections |
| `test_facade.py` | `pymurmur.Simulation` public API: construct, run, metrics, benchmark |
| `test_e2e.py` | Full headless simulation scenarios across modes |
| `test_mode_switch_no_crash.py` | Live-cycling all force modes never crashes (guard) |
| `acceptance/test_phase1.py` | P1 gates: occlusion, steric, boundary δ |
| `acceptance/test_phase3.py` | P3 gates: field mode, wander, ripple, predator, presets |
| `acceptance/test_phase4.py` | P4 gates: spatial golden, fuzz, presets, predator |

### crosscutting/
| File | Idea |
|---|---|
| `guards/test_architecture.py` | Import-DAG matrix (`ALLOWED_EDGES`) enforcement |
| `guards/test_imports.py` | AST-level upward-import bans |
| `guards/test_golden.py` | Bit-exact trajectories vs `test/data/golden_*.npz` |
| `guards/test_determinism.py` | Registry-wide same-seed identity × threads × jitter, plus one subprocess leg per mode |
| `guards/test_docs.py` | Every repo `.md` file's intra-repo links resolve; arch.md ↔ test.md topology sync; retired schemes absent |
| `guards/test_config_drift.py` | Every SimConfig field referenced in source |
| `guards/test_collection_count.py` | Per-level + per-module collection floors |
| `guards/test_composers.py` | G1: every public L0 atom (types/force-primitive/occlusion/steric/boid/config surface) has ≥ 1 call site |
| `guards/test_ci_workflow_integrity.py` | Extracts and executes `.github/workflows/*.yml`'s actual run scripts — bash validity, needs-graph completeness, summary-gate coverage, no bare (non-Docker) test invocations |
| `guards/test_strictly_3d.py` | P14.3: no 2D spatial arrays in `physics/`; `depth > 0` validation |
| `perf/test_budgets.py` | P1: per-mode step-time budget table, parametrized over `sorted(MODE_REGISTRY)` (`@slow`) |
| `perf/test_performance.py` | P2/P3/P4: scaling-checkpoint tier assertions, full-inventory memory audit at 300K, 20K-frame soak (`@slow`) |
| `perf/test_scaling.py` | O(N), O(N log N), O(1) complexity claims (`@slow`) |

---

## Suite Audit — 2026-07-21

Re-audited the suite end to end for this restructure rather than trusting
`TODO/roadmap6.md`'s gap list at face value — several of its "MISSING"/
"PARTIAL" items turned out to already be implemented (its own audit basis
predates the final round of scaling/CI work in the same day). Confirmed
still-implemented-and-correct: `test_composers.py` (roadmap6 R6.2 called this
MISSING), the entire S8 scaling/perf item set (S8.1 budget table → S8.5
determinism-matrix breadth, including both soak tiers — S8.4 called this
MISSING) — all present and passing. Genuinely missing and implemented in this
audit:

- **`guards/test_strictly_3d.py`** — the P14.3 strictly-3D check existed only
  as an inline Python heredoc inside `guard-rails.yml`, unlike every sibling
  P14 guard. Now a real pytest file, dockerized like the rest.
- **Stale-old-scheme-identifier scan** — also only inline shell in
  `guard-rails.yml`'s `guard-rail-dag` job; folded into `guards/test_docs.py`.
- **`test_phase3.py` missing its own `phase3` marker** — `test_phase1.py`/
  `test_phase4.py` self-mark correctly; `test_phase3.py`'s `pytestmark` list
  was missing `pytest.mark.phase3`. Fixed.
- **Three dead pytest markers** (`numba`, `pygame`, `integration`) declared
  in `pytest.ini` with zero actual `@pytest.mark.X` usages anywhere in
  `test/`. Removed rather than left as misleading documentation.
- **`test_docs.py`'s `COVERED_MD_FILES`** was a dead, unused list that still
  named the already-deleted `roadmap_deepseek.md` and the about-to-be-deleted
  `docker.md`. Removed — the real link-resolution mechanism (`rglob("*.md")`)
  already covers every `.md` file in the repo, `TODO/` included, so this
  needed no functional replacement.
- **CI never actually ran through Docker** — both workflows `pip install`ed
  directly on the bare runner despite `ci/Dockerfile{,.gpu}` and
  `ci/docker-compose.yml` existing and being fully documented; the GPU job in
  particular installed Mesa/GL packages onto `ubuntu-latest` and never
  touched `ci/Dockerfile.gpu`. Rewired — see Continuous Integration above.
- **`.dockerignore` was itself untracked** — `.gitignore` ignored it
  (`# Docker (not part of source)`, which was backwards: it's required build
  configuration). Fixed; a fresh clone now gets a working Docker build
  context.
- **`numba` version-pin drift**: `pyproject.toml`'s `numba` extra pinned
  `>=0.60` while `requirements-optional.txt`/`ci/Dockerfile` (the paths
  actually used) pinned `>=0.58`. Aligned to `>=0.58`.
- **Index A/B doc drift**: `test_mesh_registry.py`, `test_renderer_impostor.py`,
  `test_logging.py`, `l2_integration/test_defect_regressions.py`, and
  `crosscutting/perf/test_budgets.py` all existed but were absent from this
  document's tree diagram and indices. Added.

Prior audit (2026-07-19) implemented: registry-parametrized **determinism
guard**, **visualizer degradation-ladder tests**, **trail buffer-growth /
degenerate / FBO-lifecycle tests** and **Renderer3D trail-wiring tests**, and
**HUD GL helper tests**.

Known remaining gaps (acceptable, tracked):

- `l0m/analysis/test_marl.py` is a stub until Phase 12 lands.
- `pymurmur/viz/visualizer.py`'s pygame `run()` loop interaction branches
  (HUD toggle, cursor spawns) are only partially covered — they need a real
  event loop; the extracted logic (`_apply_quality_actions`) is now fully
  tested.
- `_kernels.py` shows low line coverage in fast runs — a tracing artifact:
  numba-jitted bodies bypass the coverage tracer; parity is asserted by
  `forces/test_kernels.py` and `guards/test_determinism.py`.
- `analysis/density_scaling.py` sweep body is `@slow`-only by design.
- `TODO/roadmap6.md`'s Appendix C cross-cutting items (CC1 actionable
  YAML-error messages, CC2 GPU context-loss graceful fallback, CC3
  fastmath×metrics-export warning) were not re-verified in this audit — they
  are `pymurmur/` source-behavior questions, not test-suite gaps, and out of
  this restructure's scope.

---

## Execution Strategy

Execution is bottom-up (micro first — a broken module fails everything above
it), which now also matches the read order of the tree itself:

```
Every commit   →  pytest test/ -m "not slow and not gl and not gpu"   (~55 s)
Every commit   →  pytest -m guard                                     (guard-rails.yml)
PR merge       →  + GPU suites (l0_modules/viz, gpu-marked)            (needs display/Xvfb)
Nightly        →  + pytest -m slow test/crosscutting/perf/             (benchmarks, scaling, soak)
```

Bottom-up debugging order when a wide breakage appears:
`l0_modules/core → l0_modules/physics → l0_modules/simulation →
l0_modules/{analysis,capture,viz} → l2_integration → l3_subsystems →
l4_system → crosscutting`.

---

## History

- **2026-07-21** — Micro-to-Macro renumbering: `l3_modules→l0_modules`,
  `l1_subsystems→l3_subsystems`, `l0_system→l4_system`,
  `l4_crosscutting→crosscutting` (unnumbered, orthogonal); `l2_integration`
  unchanged. Merged `docker.md` into this document (all Docker/CI content
  now lives in the Continuous Integration & Docker section above) and
  deleted it. Rewired both GitHub Actions workflows to run every job through
  `docker build`/`docker compose`/`docker run` instead of installing
  dependencies on the bare runner. Re-audited the suite against
  `TODO/roadmap6.md`'s gap list (see Suite Audit above) — added the two
  genuinely-missing guard tests, fixed a missing `phase3` marker, removed
  three dead marker declarations, fixed `.dockerignore`/numba-pin drift, and
  rebuilt Index A/B against the actual file listing.
- **2026-07-19 (3)** — suite audit: added determinism guard, visualizer
  quality-ladder tests, trail growth/degenerate/FBO tests, renderer trail
  wiring + HUD GL tests; added Index A/B (ideas ↔ files) to this document.
- **2026-07-19 (2)** — reorganized Macro-to-Micro: five altitude levels
  `l0_system` → `l4_crosscutting`; module mirrors nested under `l3_modules/`;
  phase acceptance gates under `l0_system/acceptance/`.
- **2026-07-19 (1)** — root-level files foldered; phase-named files renamed
  module-first; dead stub `test_composers.py` deleted; collection-count guard
  implemented.
- Earlier planning history (per-test tables for P0–P10) is preserved in git
  history of this file and in `roadmap_deepseek.md` Part II.

*Derived from `arch.md` §2. Docker/CI content merged from `docker.md`
(removed 2026-07-21). July 2026.*
