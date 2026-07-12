# ci/Dockerfile — CPU-only headless test image
# Build: docker build -f ci/Dockerfile -t pymurmur-test .

FROM python:3.12-slim

LABEL org.pymurmur.image=pymurmur-test
LABEL description="Headless test runner for pymurmur 3D murmuration simulation"

# Prevent .pyc files and buffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# System deps for scipy, Pillow, matplotlib, numba (libgomp1), pygame (SDL2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libxext6 \
    libsdl2-2.0-0 \
    libfreetype6 \
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
#   gymnasium — MARL env wrapper scaffolding (P12.2, lazy import)
RUN pip install --no-cache-dir \
    "numba>=0.58,<1.0" \
    "pygame>=2.5,<3.0" \
    "Pillow>=10.0,<12.0" \
    "matplotlib>=3.7,<4.0" \
    "gymnasium>=0.29,<2.0"

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

# Default command: marker-based selection — GL/GPU-marked tests auto-skip
# without a GPU context, no --ignore lists needed.
CMD ["pytest", "test/", "-v", "-m", "not slow and not gl and not gpu", \
     "--timeout=60", "--tb=short"]
