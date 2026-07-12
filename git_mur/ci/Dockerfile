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

# Default command: run the fast CI suite
CMD ["pytest", "test/", "-v", "-m", "not slow", \
     "--ignore=test/viz/test_renderer.py", \
     "--ignore=test/viz/test_input.py", \
     "--ignore=test/viz/test_camera.py", \
     "--timeout=60", "--tb=short"]
