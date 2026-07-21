# ci/Dockerfile — CPU-only headless test image
# Build: docker build -f ci/Dockerfile -t pymurmur-test .
# Build for a specific interpreter (CI matrix): --build-arg PYTHON_VERSION=3.11

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

LABEL org.pymurmur.image=pymurmur-test
LABEL description="Headless test runner for pymurmur 3D murmuration simulation"

# Prevent .pyc files and buffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps for scipy, Pillow, matplotlib, numba (libgomp1), pygame (SDL2)
# build-essential needed for moderngl wheel compilation on ARM
# libegl1-mesa + libegl1: headless EGL for moderngl standalone contexts (no X11)
# xvfb: virtual framebuffer for X11-dependent tests (pygame display init)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-dev \
    libegl-mesa0 \
    libegl1 \
    libxrender1 \
    libxext6 \
    libsdl2-2.0-0 \
    libfreetype6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layered for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optional deps needed by the CPU test suite:
#   numba     — JIT force kernel fallback paths (P4.10, P13)
#   pygame    — headless time.Clock + event scaffolding (imported by conftest)
#   Pillow    — GPU-free capture fallback (P8.9)
#   matplotlib — GPU-free capture fallback + phase-diagram plots
#   moderngl  — needed for viz module imports at test-collection time
#   PyGLM     — needed for viz module imports at test-collection time
#   gymnasium — MARL env wrapper scaffolding (P12.2, lazy import)
RUN pip install --no-cache-dir \
    "numba>=0.58,<1.0" \
    "pygame>=2.5,<3.0" \
    "Pillow>=10.0,<12.0" \
    "matplotlib>=3.7,<4.0" \
    "moderngl>=5.8,<6.0" \
    "PyGLM>=2.7,<3.0" \
    "gymnasium>=0.29,<2.0"

# Install test-only deps
RUN pip install --no-cache-dir \
    pytest==8.* \
    pytest-cov==6.* \
    pytest-xdist==3.* \
    pytest-timeout==2.*

# Copy project source (conf/, pymurmur/, test/)
# Note: output/ is excluded via .dockerignore to prevent stale host artifacts
COPY . .

# Create output directory for capture/quality tests with world-writable permissions.
# The dir is NOT volume-mounted in the default CMD — tests write directly to the
# container filesystem. Users who need to extract artifacts can mount a volume at
# runtime: docker run -v $(pwd)/output:/app/output pymurmur-test ...
RUN mkdir -p output && chmod 777 output

# P0.16: output/evolved.yaml is gitignored (never enters the build
# context via COPY), but test_evolved_yaml.py hard-asserts it exists.
# Without this, any image built from a fresh checkout fails 14 tests
# immediately — generate a schema-valid artifact at build time so the
# image is self-contained (same fix as guard-rail-evolved's CI step).
RUN python3 scripts/generate_evolved_artifact.py

# Write entrypoint that starts Xvfb for tests that probe pygame.display
# or moderngl standalone contexts. The virtual framebuffer provides
# DISPLAY=:99 so that pygame.display.init() and headless moderngl
# contexts work without a physical GPU.
COPY ci/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

ENTRYPOINT ["/usr/local/bin/entrypoint"]

# Default command: marker-based selection — GL/GPU/pygame-marked tests
# auto-skip in headless mode, no --ignore lists needed.
CMD ["pytest", "test/", "-v", "-m", "not slow and not gl and not gpu and not pygame", \
     "--timeout=60", "--tb=short"]
