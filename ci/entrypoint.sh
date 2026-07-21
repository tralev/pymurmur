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
exec "$@"
