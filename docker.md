# Docker & CI — pymurmur 3D Murmuration Simulation

> **Purpose:** how to build, run, and debug the Dockerised test suites, and what the CI pipelines enforce.
> **Directory rules:** all Docker/Compose/CI files live in `ci/` (workflows in `.github/workflows/`); all tests live in `test/`.
> **References:** [arch.md](arch.md) (module map, dependency rules), [test.md](test.md) (test plan & tree layout).
>
> This document describes the files — it does not duplicate them. The
> canonical sources are the files themselves; when in doubt, read those.

---

## File inventory

| File | Role |
|---|---|
| [ci/Dockerfile](ci/Dockerfile) | CPU-only headless image `pymurmur-test` (`python:3.12-slim`) — fast suite, E2E, guards |
| [ci/Dockerfile.gpu](ci/Dockerfile.gpu) | `pymurmur-test-gpu`, extends the CPU image with GL/SDL deps + moderngl/PyGLM/pygame |
| [ci/entrypoint-gpu.sh](ci/entrypoint-gpu.sh) | Xvfb wrapper (`:99`, GLX + render) so GL tests run without a display |
| [ci/docker-compose.yml](ci/docker-compose.yml) | Profiles: `fast`, `e2e`, `slow`, `gpu`, `capture`, `lint`, `full` (+ guard/subsystem services in `fast`) |
| [ci/docker-compose.gpu.yml](ci/docker-compose.gpu.yml) | Override adding the nvidia runtime + driver env for real-GPU runs |
| [.github/workflows/test.yml](.github/workflows/test.yml) | Main CI: fast matrix, guards/subsystems, E2E, slow, GPU (nightly), lint |
| [.github/workflows/guard-rails.yml](.github/workflows/guard-rails.yml) | P14 guard rails: 9 jobs incl. a merge-blocking summary gate (see below) |
| [requirements.txt](requirements.txt) | Production deps only: numpy, scipy, PyYAML |
| [requirements-optional.txt](requirements-optional.txt) | numba, pygame, moderngl, PyGLM, Pillow, matplotlib, gymnasium (sb3 commented — scripts-only) |
| [requirements-test.txt](requirements-test.txt) | pytest (+cov/xdist/timeout), ruff, mypy |

**Image strategy:** two images, layered. The CPU image installs
production + headless-safe optional deps (numba, pygame, Pillow,
matplotlib, gymnasium — the GPU-free capture fallback and MARL
scaffolding must work *without* GL; that is the point). The GPU image
extends it with Mesa/EGL/SDL libraries and the viz stack, and wraps
every command in Xvfb via the entrypoint.

---

## Test tree & markers (what the suites select)

The test tree is layered macro→micro (see [test.md](test.md)):

```
test/
├── l0_system/        # CLI, facade, e2e, config resolution, acceptance gates
├── l1_subsystems/    # subsystem isolation (A–F)
├── l2_integration/   # cross-module wiring (engine/capture/render/config contracts)
├── l3_modules/       # module mirrors: core, physics, simulation, viz, capture, analysis
└── l4_crosscutting/
    ├── guards/       # architecture DAG, docs, config drift, golden, determinism, collection count
    └── perf/         # step-time budgets, scaling, memory
```

Selection is **marker-based** (`pytest.ini`): `slow` (perf/nightly),
`gl` / `gpu` (alias — needs a GL context; auto-skip without one),
`golden`, `guard`, `numba`, `pygame`, `e2e`, `integration`,
`phase1/3/4`, `acceptance`. The container commands select with markers
(`-m "not slow and not gl and not gpu"`), so they survive tree moves
without `--ignore` lists.

---

## Quick reference

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
  pytest test/l3_modules/physics/test_steric.py -v
docker run --rm -it -v $(pwd)/output:/app/output pymurmur-test bash
```

All services mount `./output` (JUnit XML `test-results-*.xml`, coverage
XML, capture artifacts) and `./conf` read-only.

### Compose profiles

| Profile | Service(s) | Selects | Timeout |
|---|---|---|:---:|
| `fast` | `test-fast` + `test-guards` + `test-imports` | `-m "not slow and not gl and not gpu"` + guard/subsystem files | 60 s |
| `e2e` | `test-e2e` | l0_system e2e/CLI/config tests | 120 s |
| `slow` | `test-slow` | `-m "slow and not gl and not gpu"` (installs gymnasium + sb3 best-effort for the P11.6/P12.3 experiments) | 600 s |
| `gpu` | `test-gpu` | `-m "gl or gpu"` under Xvfb (llvmpipe; nvidia via override file) | 120 s |
| `capture` | `capture` | `python -m pymurmur --config field --no-viz --capture` at 400×300×120f | — |
| `lint` | `lint` | ruff + mypy | — |
| `full` | all of the above | nightly superset | — |

---

## Environment variables

| Variable | Purpose | Default |
|---|---|:---:|
| `PYMURMUR_TEST` | enable test-only code paths | `0` |
| `PYMURMUR_CONFIG_DIR` | override config search path | `conf/` |
| `CAPTURE_W` / `CAPTURE_H` / `CAPTURE_FRAMES` / `CAPTURE_OUT` | headless-capture overrides (P8.7/P8.9; precedence YAML < env < CLI) | config |
| `NUMBA_NUM_THREADS` | cap numba parallelism in CI | `4` |
| `LIBGL_ALWAYS_SOFTWARE` | force Mesa llvmpipe even when a GPU exists (deterministic CI rendering) | unset |
| `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` | GPU selection for the nvidia runtime (override file) | `all` / `compute,graphics,utility` |
| `__GL_SYNC_TO_VBLANK` | disable vsync in headless GPU runs (override file) | `1` |

---

## Continuous integration

Two workflows. Both run on push to `main` (docs/conf/output changes
ignored), on PRs to `main`, and nightly.

### 1. `test.yml` — "Test pymurmur" (nightly 03:00 UTC)

| Job | When | What it runs |
|---|---|---|
| `test-fast` | always; Python **3.11 + 3.12** matrix | `pytest test/ -m "not slow"` with coverage (viz renderer/input excluded via `--ignore` — see Known drift) |
| `test-imports` | always | `l4_crosscutting/guards/` (imports, architecture, docs, golden) + `l1_subsystems/test_subsystem_a–f.py` |
| `test-e2e` | always | `l0_system/` e2e + config-resolution + config-files |
| `test-slow` | PR / nightly | `l4_crosscutting/perf/{test_performance,test_scaling}.py -m slow`, 600 s timeout |
| `test-gpu` | nightly only | Xvfb + Mesa llvmpipe; `l3_modules/viz/{test_renderer,test_input,test_camera,test_shaders}.py` |
| `lint` | always | `ruff check pymurmur/ test/` + `mypy pymurmur/ --ignore-missing-imports` |

### 2. `guard-rails.yml` — "P14 Guard Rails" (nightly 04:00 UTC)

Ten jobs; `guard-rails-summary` aggregates and **blocks merge** on any
failure:

| Job | Enforces |
|---|---|
| `guard-rail-dag` | P14.1 — architecture DAG matrix (AST import walk, named forbidden edges) + stale old-scheme identifier scan |
| `guard-rail-golden` | P0.1 golden trajectories + P13.5 determinism runs |
| `guard-rail-config-drift` | P14.2 — every config leaf read by ≥ 1 non-config module |
| `guard-rail-3d` | P14.3 — no 2D spatial arrays in `physics/`; `depth > 0` validation |
| `guard-rail-doc-links` | P14.4 — arch.md ↔ roadmap links resolve, module paths exist |
| `guard-rail-collection-count` | P14.5 — collected-test floors per level/module (`EXPECTED_MINIMUMS`) |
| `guard-rail-mypy` | type-check gate |
| `guard-rail-evolved` | `output/evolved.yaml` artifact validity |
| `guard-rail-composers` | G1 — every public L0 atom has ≥ 1 cross-module call site (no dead atoms) |
| `guard-rails-summary` | merge-blocking roll-up |

### Pipeline matrix & typical durations

| Trigger | Fast | Guards | E2E | Slow | GPU | Lint |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Push to main | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| PR to main | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Nightly | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Fast ≈ 30–90 s · guards ≈ 10–30 s · e2e ≈ 1–2 min · slow ≈ 5–15 min ·
GPU ≈ 1–3 min · lint ≈ 15–30 s. Push total ≈ 2–3 min; nightly ≈ 8–20 min.

---

## GPU testing approach

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

---

## macOS development

macOS has **no native OpenGL context** (no display server, no GPU
visible to ModernGL). The following tests **skip automatically** when
run outside Docker:

| Test / class | Skip message | File |
|---|---|---|
| `TestRenderer3D::test_renderer_windowed_context` | `No display available for windowed context` | `test/l3_modules/viz/test_renderer.py` |
| `TestVisualizerIntegration::test_visualizer_windowed_frame` | `Windowed context creation failed (no display)` | `test/l3_modules/viz/test_renderer.py` |
| `@pytest.mark.gpu`-decorated tests (~80+) | `GPU not available` | `test/l3_modules/viz/test_renderer.py`, `test_renderer_impostor.py`, `test_camera.py`, `test_input.py`, `test_shaders.py`, `test_hud.py`, etc. |

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

---

## Local development (outside Docker)

- `requires-python >= 3.9`; mypy targets 3.9. The dev-machine baseline
  interpreter is `python3` (3.9) — source files need
  `from __future__ import annotations` for `X | Y` annotations. The
  Docker images run 3.12 and CI runs 3.11 + 3.12, so both ends of the
  supported range are exercised.
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

*Docker & CI documentation for pymurmur. All Docker/CI files belong in `ci/`; all tests belong in `test/`. Rewritten July 2026 against the current tree.*
