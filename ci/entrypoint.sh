#!/bin/bash
set -e
# Start Xvfb on display :99 so pygame.display.init() and headless
# moderngl contexts work without a physical GPU.
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:99
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render &>/dev/null &
    XVFB_PID=$!
    trap "kill $XVFB_PID 2>/dev/null || true" EXIT
    # Give Xvfb a moment to start
    sleep 0.5
fi

# P0.16: output/evolved.yaml is baked into the image at build time (see
# ci/Dockerfile), but any job that volume-mounts a host output/
# directory over /app/output (needed to extract junit XML results)
# shadows the baked file with an empty host directory — regenerate it
# here if missing so test_evolved_yaml.py sees a real file regardless
# of what got mounted over it. Cheap (~2-3s) and only runs when needed.
if [ ! -f output/evolved.yaml ]; then
    python3 scripts/generate_evolved_artifact.py
fi

exec "$@"
