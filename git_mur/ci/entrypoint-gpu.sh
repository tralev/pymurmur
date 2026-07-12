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
