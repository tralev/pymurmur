"""Visualizer — composes renderer, camera, and simulation for one frame.

Level 2 — optional GPU layer. Headless-capable for testing and capture.
Per test.md INT.4, provides headless_frame() for PIL snapshots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .renderer import Renderer3D
from .camera import OrbitCamera

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from ..simulation.engine import SimulationEngine
    from ..core.config import SimConfig
    from .input_control import InputControl


class Visualizer:
    """Orchestrates one rendered frame of the simulation.

    Owns the Renderer3D and OrbitCamera. Does NOT own the
    SimulationEngine — it receives it from the caller.
    """

    def __init__(
        self,
        sim: SimulationEngine,
        config: SimConfig,
        *,
        headless: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.sim = sim
        self.config = config
        self.paused = False

        self.renderer = Renderer3D(
            width=width if width is not None else config.window_width,
            height=height if height is not None else config.window_height,
            theme=config.theme,
            headless=headless,
            instance_buffer_chunk=config.instance_buffer_chunk,
        )
        self.camera = OrbitCamera(
            target=(config.width / 2, config.height / 2, config.depth / 2),
        )

    def headless_frame(self) -> PILImage:
        """Render one frame and return a PIL Image (headless only).

        Pure render — does NOT step the simulation. The caller is
        responsible for calling sim.step() before each render frame.
        """
        self.renderer.begin_frame(self.camera)
        self.renderer.draw_birds(self.sim.flock)
        self.renderer.end_frame()
        return self.renderer.capture_frame()

    def frame(self) -> None:
        """Render one frame to the screen (windowed mode).

        Pure render — does NOT step the simulation. The caller is
        responsible for calling sim.step() before each render frame.
        """
        self.renderer.begin_frame(self.camera)
        self.renderer.draw_birds(self.sim.flock)
        self.renderer.end_frame()

    def run(self, input_ctrl: InputControl) -> None:
        """Main visualization loop — input → update → render."""
        import pygame

        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(self.config.fps) / 1000.0

            running = input_ctrl.handle_events()
            self.camera.step_auto_rotate(dt)
            self.paused = input_ctrl.paused

            # Push live-mutation commands into engine queue (I4.3)
            if input_ctrl.pending_add > 0:
                self.sim.enqueue_add(input_ctrl.pending_add)
                input_ctrl.pending_add = 0
            if input_ctrl.pending_remove > 0:
                self.sim.enqueue_remove(input_ctrl.pending_remove)
                input_ctrl.pending_remove = 0
            if input_ctrl.pending_reset:
                self.sim.enqueue_reset()
                input_ctrl.pending_reset = False

            # Drain commands every frame (even when paused) so +/- mutations
            # take effect immediately, then step physics only when unpaused.
            self.sim.drain_commands()

            if not self.paused:
                self.sim.step(dt)

            self.renderer.begin_frame(self.camera)
            self.renderer.draw_birds(self.sim.flock)
            if input_ctrl.show_grid:
                self.renderer.draw_grid()
            self.renderer.end_frame()

            # Live metrics in window title
            snap = self.sim.metrics.snapshot()
            title = (
                f"pymurmur — N={self.sim.flock.N_active} "
                f"φp={self.config.phi_p:.2f} φa={self.config.phi_a:.2f} "
                f"α={snap.alpha:.3f} Θ={snap.theta:.3f}"
            )
            pygame.display.set_caption(title)
            pygame.display.flip()

        pygame.quit()
