# Docker & CI — pymurmur 3D Murmuration Simulation

> **Purpose:** Dockerised testing and continuous integration for pymurmur.
> **Directory rules:** All Docker, Docker Compose, and CI files MUST be placed within the `ci/` directory. All test files MUST be placed within the `test/` directory.
> **References:** `arch.md` (module map, dependency rules, libraries), `test.md` (test plan & execution strategy), `roadmap_deepseek.md` (phased implementation).

---

## Table of Contents

1. [ci/ Directory Layout](#ci-directory-layout)
2. [Docker Setup for Local Testing](#docker-setup-for-local-testing)
3. [Docker Compose Profiles](#docker-compose-profiles)
4. [Running Tests in Docker](#running-tests-in-docker)
5. [Environment Variables](#environment-variables)
6. [Continuous Integration](#continuous-integration)
7. [GPU Testing in CI](#gpu-testing-in-ci)
8. [CI Pipeline Matrix](#ci-pipeline-matrix)

---

## ci/ Directory Layout

```
ci/
├── Dockerfile                 # CPU-only headless image (CI, fast suite)
├── Dockerfile.gpu             # GPU-enabled image (ModernGL renderer tests)
├── docker-compose.yml         # Multi-profile Compose (fast, slow, gpu, e2e, full)
├── docker-compose.gpu.yml     # GPU Compose override (nvidia runtime)
└── entrypoint-gpu.sh          # Xvfb wrapper for headless GPU context

.github/workflows/
└── test.yml                   # GitHub Actions CI workflow (triggers on push/PR/schedule)
```

---

## Docker Setup for Local Testing

### Prerequisites

- Docker Engine 24+ (or Docker Desktop)
- Docker Compose v2
- NVIDIA Container Toolkit (for GPU tests only — see [GPU Testing in CI](#gpu-testing-in-ci))

### Image Strategy

Two images, layered for minimal duplication:

| Image | Base | Purpose | Size target |
|-------|------|---------|:---:|
| `pymurmur-test` | `python:3.12-slim` | Headless fast suite, E2E, config validation | ~400 MB |
| `pymurmur-test-gpu` | `pymurmur-test` + GL deps | Renderer tests, visual tests | ~800 MB |

The GPU image extends the CPU image — all headless tests run in both.

### `ci/Dockerfile` — CPU Headless Image

```dockerfile
# ci/Dockerfile — CPU-only headless test image
# Build: docker build -f ci/Dockerfile -t pymurmur-test .

FROM python:3.12-slim

LABEL org.pymurmur.image=pymurmur-test
LABEL description="Headless test runner for pymurmur 3D murmuration simulation"

# Prevent .pyc files and buffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps for scipy, Pillow, matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layered for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Headless-safe optional deps: Pillow + matplotlib are needed by the
# GPU-free capture fallback tests (roadmap_deepseek.md P8.9 / P8.9), which MUST run
# in the CPU image — that is the point of the fallback.
RUN pip install --no-cache-dir "Pillow>=10.0,<12.0" "matplotlib>=3.7,<4.0"

# Install test-only deps
RUN pip install --no-cache-dir \
    pytest==8.* \
    pytest-cov==6.* \
    pytest-xdist==3.* \
    pytest-timeout==2.*

# Copy project source (conf/, pymurmur/, test/, output/ placeholder)
COPY . .

# Create output directory for capture tests
RUN mkdir -p output

# Default command: run the fast CI suite.
# Test selection is MARKER-based (roadmap_deepseek.md P0): tests needing a GL context
# carry @pytest.mark.gl and auto-skip without one — no --ignore lists,
# so the command survives the test-tree migration to mirrored subpackages.
CMD ["pytest", "test/", "-v", "-m", "not slow and not gl", \
     "--timeout=60", "--tb=short"]
```

### `ci/Dockerfile.gpu` — GPU-Enabled Image

```dockerfile
# ci/Dockerfile.gpu — GPU-enabled test image
# Build: docker build -f ci/Dockerfile.gpu -t pymurmur-test-gpu .
# Requires nvidia-container-runtime.

FROM pymurmur-test:latest

LABEL org.pymurmur.image=pymurmur-test-gpu
LABEL description="GPU test runner for pymurmur (ModernGL renderer tests)"

# System deps for ModernGL (libgl1, libegl1) and pygame (SDL2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libegl1 \
    libsdl2-2.0-0 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# viz-specific dependencies
RUN pip install --no-cache-dir \
    moderngl==5.* \
    PyGLM==2.* \
    pygame==2.* \
    Pillow==11.*

# Xvfb for headless GPU context (software rasterizer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Entrypoint wraps tests with Xvfb for software OpenGL
COPY ci/entrypoint-gpu.sh /usr/local/bin/entrypoint-gpu.sh
RUN chmod +x /usr/local/bin/entrypoint-gpu.sh

ENTRYPOINT ["/usr/local/bin/entrypoint-gpu.sh"]
CMD ["pytest", "test/", "-v", \
     "--timeout=120", "--tb=short"]
```

### `ci/entrypoint-gpu.sh`

```bash
#!/bin/bash
# ci/entrypoint-gpu.sh — start Xvfb for headless GPU context, then run tests
set -e

# Start virtual framebuffer on display :99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render &
XVFB_PID=$!
export DISPLAY=:99

# Wait for Xvfb to be ready
sleep 1

# Run the test command
"$@"
EXIT_CODE=$?

# Cleanup
kill $XVFB_PID 2>/dev/null || true
exit $EXIT_CODE
```

---

## Docker Compose Profiles

### `ci/docker-compose.yml`

```yaml
# ci/docker-compose.yml — multi-profile test orchestration
# Usage:
#   docker compose -f ci/docker-compose.yml --profile fast up --build
#   docker compose -f ci/docker-compose.yml --profile slow up
#   docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile gpu up

services:
  # ── Profile: fast (default — runs on every commit) ──
  test-fast:
    build:
      context: .
      dockerfile: ci/Dockerfile
    image: pymurmur-test:latest
    profiles: [fast]
    command: >
      pytest test/ -v
      -m "not slow and not gl"
      --timeout=60
      --tb=short
      --junitxml=output/test-results-fast.xml
      --cov=pymurmur
      --cov-report=xml:output/coverage-fast.xml
      --cov-report=term
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro       # read-only configs
    environment:
      - PYMURMUR_TEST=1

  # ── Profile: e2e (end-to-end CLI tests) ──
  test-e2e:
    build:
      context: .
      dockerfile: ci/Dockerfile
    image: pymurmur-test:latest
    profiles: [e2e, full]
    command: >
      pytest test/test_e2e.py test/test_config_resolution.py test/test_config_files.py
      -v --timeout=120 --tb=long
      --junitxml=output/test-results-e2e.xml
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro
    environment:
      - PYMURMUR_TEST=1
      - PYMURMUR_CONFIG_DIR=/app/conf

  # ── Profile: slow (performance, scaling, nightly experiments) ──
  # Marker-wide: includes perf/scaling AND the roadmap's @slow experiment
  # tests (P11.6 emergent alignment, P12.3 trained-beats-random). The MARL
  # experiment self-skips unless stable-baselines3 is installed (see the
  # optional pip line below).
  test-slow:
    build:
      context: .
      dockerfile: ci/Dockerfile
    image: pymurmur-test:latest
    profiles: [slow, full]
    command: >
      sh -c "
        pip install --no-cache-dir gymnasium 'stable-baselines3>=2.0' || true;
        pytest test/ -v -m 'slow and not gl'
        --timeout=600
        --tb=long
        --junitxml=output/test-results-slow.xml
        --durations=20
      "
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro
    environment:
      - PYMURMUR_TEST=1
      - NUMBA_NUM_THREADS=4

  # ── Profile: gpu (all @gl-marked tests — renderer/capture/visual) ──
  test-gpu:
    build:
      context: .
      dockerfile: ci/Dockerfile.gpu
    image: pymurmur-test-gpu:latest
    profiles: [gpu, full]
    command: >
      pytest test/ -m gl
      -v --timeout=120 --tb=long
      --junitxml=output/test-results-gpu.xml
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro
    environment:
      - PYMURMUR_TEST=1
      # Allow software rendering if no real GPU
      - LIBGL_ALWAYS_SOFTWARE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ── Profile: guards (architecture / drift / golden — every commit) ──
  # Current flat suite (test_imports.py, test_subsystem_*.py) and the
  # roadmap's guard set (test_architecture.py, test_docs.py,
  # test_golden.py — P14.1/P0.2) as the tree migrates; the glob covers both.
  test-guards:
    build:
      context: .
      dockerfile: ci/Dockerfile
    image: pymurmur-test:latest
    profiles: [fast, full]
    command: >
      sh -c "pytest $(ls test/test_imports.py test/test_architecture.py
      test/test_docs.py test/test_golden.py test/test_subsystem_*.py
      2>/dev/null | tr '\n' ' ')
      -v --timeout=60 --tb=short
      --junitxml=output/test-results-guards.xml"
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro

  # ── Profile: capture (llvmpipe GIF smoke — the env-override consumer) ──
  # Produces a small headless capture on software GL; exercises the
  # CAPTURE_* env overrides (roadmap_deepseek.md P8.9) end to end.
  capture:
    build:
      context: .
      dockerfile: ci/Dockerfile.gpu
    image: pymurmur-test-gpu:latest
    profiles: [capture, full]
    command: python -m pymurmur --config field --no-viz --capture
    volumes:
      - ./output:/app/output
      - ./conf:/app/conf:ro
    environment:
      - LIBGL_ALWAYS_SOFTWARE=1
      - CAPTURE_W=400
      - CAPTURE_H=300
      - CAPTURE_FRAMES=120
      - CAPTURE_OUT=output/ci_capture.gif

  # ── Profile: lint (static analysis) ──
  lint:
    build:
      context: .
      dockerfile: ci/Dockerfile
    image: pymurmur-test:latest
    profiles: [lint, full]
    command: >
      sh -c "
        pip install ruff mypy && 
        ruff check pymurmur/ test/ &&
        mypy pymurmur/ --ignore-missing-imports
      "
    volumes:
      - ./conf:/app/conf:ro
```

### `ci/docker-compose.gpu.yml` — GPU Override

```yaml
# ci/docker-compose.gpu.yml — GPU runtime override
# Usage: docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile gpu up

services:
  test-gpu:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
      - __GL_SYNC_TO_VBLANK=0
```

---

## Running Tests in Docker

### Quick Reference

```bash
# Build images
docker build -f ci/Dockerfile -t pymurmur-test .
docker build -f ci/Dockerfile.gpu -t pymurmur-test-gpu .

# Fast suite (every commit)
docker compose -f ci/docker-compose.yml --profile fast up --build --abort-on-container-exit

# E2E tests
docker compose -f ci/docker-compose.yml --profile e2e up --build

# Slow/performance tests
docker compose -f ci/docker-compose.yml --profile slow up

# GPU tests (requires NVIDIA runtime)
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile gpu up

# Full suite (nightly — all profiles)
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile full up

# Lint only
docker compose -f ci/docker-compose.yml --profile lint up

# Interactive shell inside container for debugging
docker run --rm -it -v $(pwd)/output:/app/output pymurmur-test bash
```

### Running a Single Test File in Docker

```bash
# CPU image — single test file
docker run --rm -v $(pwd)/output:/app/output pymurmur-test \
  pytest test/physics/test_steric.py -v

# GPU image — renderer tests with software GL
docker run --rm --runtime=nvidia -v $(pwd)/output:/app/output pymurmur-test-gpu \
  pytest test/viz/test_renderer.py -v

# With custom config
docker run --rm \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/conf:/app/conf:ro \
  pymurmur-test \
  python -m pymurmur --config spatial --no-viz --capture
```

### Test Result Volumes

All profiles mount `./output` into the container. After running, find results at:

```
output/
├── test-results-fast.xml        # JUnit XML for CI dashboard
├── test-results-e2e.xml
├── test-results-slow.xml
├── test-results-gpu.xml
├── test-results-subsystems.xml
├── coverage-fast.xml            # Coverage XML
└── .coverage                    # Coverage SQLite
```

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|:---:|
| `PYMURMUR_TEST` | Set to `1` to enable test-only code paths | `0` |
| `PYMURMUR_CONFIG_DIR` | Override config search path | `conf/` |
| `CAPTURE_W` / `CAPTURE_H` | Headless capture resolution override (roadmap_deepseek.md P8.9; precedence YAML < env < CLI) | config |
| `CAPTURE_FRAMES` | Headless capture frame-count override | config |
| `CAPTURE_OUT` | Headless capture output path override | config |
| `NUMBA_NUM_THREADS` | Limit numba parallelism in CI | `4` |
| `LIBGL_ALWAYS_SOFTWARE` | Force software OpenGL (no GPU) | unset |
| `NVIDIA_VISIBLE_DEVICES` | GPU selection for nvidia runtime | `all` |
| `__GL_SYNC_TO_VBLANK` | Disable vsync in headless GPU tests | `1` |

---

## Continuous Integration

### GitHub Actions — `.github/workflows/test.yml`

```yaml
# .github/workflows/test.yml — GitHub Actions CI workflow
#
# Triggers:
#   push: runs fast suite on every commit
#   pull_request: runs fast + slow + e2e on PR to main
#   schedule: runs full suite nightly

name: Test pymurmur

on:
  push:
    branches: [main]
    paths-ignore: ['*.md', 'conf/**', 'output/**']
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 3 * * *'  # nightly at 03:00 UTC

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ── Job 1: Fast suite + imports (every commit) ──
  test-fast:
    name: Fast tests (Python ${{ matrix.python }})
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ['3.11', '3.12']
      fail-fast: false
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip

      - name: Install dependencies
        run: |
          pip install numpy scipy PyYAML
          pip install pytest pytest-cov pytest-timeout

      - name: Install optional deps
        run: |
          pip install numba  # test numba fallback paths

      - name: Run fast test suite
        run: |
          mkdir -p output
          pytest test/ -v \
            -m "not slow and not gl" \
            --timeout=60 \
            --tb=short \
            --junitxml=output/test-results-fast.xml \
            --cov=pymurmur \
            --cov-report=xml:output/coverage-fast.xml \
            --cov-report=term

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-fast-py${{ matrix.python }}
          path: output/test-results-fast.xml

      - name: Upload coverage
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-fast-py${{ matrix.python }}
          path: output/coverage-fast.xml

  # ── Job 2: Imports + subsystem isolation ──
  test-imports:
    name: Import rules & subsystems
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install deps
        run: pip install numpy scipy PyYAML pytest

      - name: Run guard tests (imports/architecture/docs/golden + legacy subsystems)
        run: |
          mkdir -p output
          pytest $(ls test/test_imports.py test/test_architecture.py \
            test/test_docs.py test/test_golden.py test/test_subsystem_*.py \
            2>/dev/null | tr '\n' ' ') \
            -v --timeout=60 --tb=short \
            --junitxml=output/test-results-subsystems.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-subsystems
          path: output/test-results-subsystems.xml

  # ── Job 3: E2E + config validation ──
  test-e2e:
    name: E2E & config validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install deps
        run: pip install numpy scipy PyYAML pytest pytest-timeout

      - name: Run E2E & config tests
        run: |
          mkdir -p output
          pytest test/test_e2e.py test/test_config_resolution.py \
            test/test_config_files.py \
            -v --timeout=120 --tb=long \
            --junitxml=output/test-results-e2e.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-e2e
          path: output/test-results-e2e.xml

  # ── Job 4: Slow / performance (PR merge only) ──
  test-slow:
    name: Performance & scaling
    if: github.event_name == 'pull_request' || github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install deps
        run: |
          pip install numpy scipy PyYAML numba
          pip install pytest pytest-timeout

      - name: Run performance tests
        run: |
          mkdir -p output
          pytest test/test_performance.py test/test_scaling.py \
            -v -m slow \
            --timeout=600 --tb=long \
            --durations=20 \
            --junitxml=output/test-results-slow.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-slow
          path: output/test-results-slow.xml

  # ── Job 5: GPU tests (nightly only — software GL) ──
  test-gpu:
    name: GPU tests (software GL)
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install system deps for GL
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb libgl1-mesa-glx libgl1-mesa-dri \
            libegl1-mesa libegl1 libsdl2-2.0-0

      - name: Install Python deps
        run: |
          pip install numpy scipy PyYAML pytest pytest-timeout
          pip install moderngl PyGLM pygame Pillow

      - name: Run GPU tests (software rasterizer)
        run: |
          mkdir -p output
          export DISPLAY=:99
          Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render &
          sleep 2
          pytest test/ -m gl \
            -v --timeout=120 --tb=long \
            --junitxml=output/test-results-gpu.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-gpu
          path: output/test-results-gpu.xml

  # ── Job 6: Lint ──
  lint:
    name: Lint (ruff + mypy)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }

      - name: Install linters
        run: pip install ruff mypy numpy scipy PyYAML

      - name: Run ruff
        run: ruff check pymurmur/ test/

      - name: Run mypy
        run: mypy pymurmur/ --ignore-missing-imports
```

---

## GPU Testing in CI

### Approach

The project uses **software OpenGL** (Mesa llvmpipe) via Xvfb for headless GPU testing in CI. This avoids the cost and complexity of real GPU runners for most tests.

| Environment | GPU | Renderer tests | Input tests |
|-------------|:---:|:---:|:---:|
| GitHub Actions (ubuntu-latest) | Mesa llvmpipe (software) | ✅ via Xvfb | ✅ via Xvfb |
| GitLab CI (gpu-tagged runner) | Real NVIDIA GPU | ✅ via nvidia runtime | ✅ |
| Local Docker (no GPU) | Mesa llvmpipe | ✅ via Xvfb | ✅ via Xvfb |
| Local Docker (NVIDIA GPU) | Real GPU | ✅ native | ✅ native |

### Software GL Limitations

- Performance benchmarks won't match real GPU — skip `test_bench_*` renderer tests in software GL mode
- FBO readback may differ slightly between Mesa and hardware drivers — use approximate pixel comparisons
- Set `LIBGL_ALWAYS_SOFTWARE=1` to force Mesa even on GPU machines for consistent CI behavior

---

## CI Pipeline Matrix

| Trigger | Fast | E2E | Slow/Perf | GPU | Lint |
|---------|:---:|:---:|:---:|:---:|:---:|
| Push to main | ✅ | ✅ | ❌ | ❌ | ✅ |
| PR to main | ✅ | ✅ | ✅ | ❌ | ✅ |
| Nightly (cron) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Manual / workflow_dispatch | ✅ | ✅ | ✅ | ✅ | ✅ |

### Expected Duration

| Job | Time |
|-----|:---:|
| `test-fast` | ~30–90 s |
| `test-imports` | ~10–20 s |
| `test-e2e` | ~60–120 s |
| `test-slow` | ~5–15 min |
| `test-gpu` | ~60–180 s |
| `lint` | ~15–30 s |
| **Total (push)** | **~2–3 min** |
| **Total (nightly)** | **~8–20 min** |

---

## requirements.txt

The `requirements.txt` at the project root should contain only the **required** (non-optional) production dependencies. Test and optional dependencies are installed separately in CI.

```text
# requirements.txt — required production dependencies
numpy>=1.24,<3.0
scipy>=1.10,<2.0
PyYAML>=6.0,<7.0
```

Optional dependencies (installed per test profile):

```text
# requirements-optional.txt — optional dependencies
numba>=0.58,<1.0         # JIT force kernels (N >= 50K)
pygame>=2.5,<3.0         # Window, events, clock
moderngl>=5.8,<6.0       # GPU instanced rendering
PyGLM>=2.7,<3.0          # Camera matrices
Pillow>=10.0,<12.0       # FBO readback, GIF export (also: GPU-free capture fallback)
matplotlib>=3.7,<4.0     # Phase diagrams, plots, GPU-free capture fallback
gymnasium>=0.29,<2.0     # MARL env wrapper (lazy import; roadmap P12.2)
# stable-baselines3>=2.0 # ONLY for scripts/train_marl.py — never a package dep
```

Test dependencies:

```text
# requirements-test.txt — test-only dependencies
pytest>=8.0,<9.0
pytest-cov>=6.0,<7.0
pytest-xdist>=3.0,<4.0   # parallel test execution
pytest-timeout>=2.0,<3.0
ruff>=0.4,<1.0           # linting
mypy>=1.8,<2.0           # type checking
```

---

## Quick Start for Contributors

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd git_mur

# 2. Install production deps
pip install -r requirements.txt

# 3. Install test deps
pip install -r requirements-test.txt

# 4. Run fast test suite locally (gl-marked tests auto-skip without a GL context)
pytest test/ -v -m "not slow and not gl"

# 5. Run in Docker (no Python setup needed)
docker compose -f ci/docker-compose.yml --profile fast up --build

# 6. Run full suite with GPU
docker compose -f ci/docker-compose.yml -f ci/docker-compose.gpu.yml --profile full up
```

---

*Docker & CI documentation for pymurmur. All Docker/CI files belong in `ci/`. All tests belong in `test/`. July 2026.*
