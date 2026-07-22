"""Visualizer — composes renderer, camera, and simulation for one frame.

Level 2 — optional GPU layer. Headless-capable for testing and capture.
Per test.md INT.4, provides headless_frame() for PIL snapshots.

P8.6: Adaptive quality governor — measures frame time, feeds
QualityGovernor, and applies degradation ladder actions.
"""

from __future__ import annotations

import time
from math import radians
from typing import TYPE_CHECKING

from ..analysis.perf import QualityGovernor
from .camera import OrbitCamera
from .hud import SliderHUD
from .renderer import Renderer3D

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from ..core.config import SimConfig
    from ..simulation.engine import SimulationEngine
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
            point_sprites=config.point_sprites,
            winged_mesh=config.winged_mesh,
            gradient_sky=config.gradient_sky,
            trails_mode=config.trails,
            trails_length=config.trail_length,
            density_mode=config.density_mode,
            density_alpha=config.density_alpha,
            bird_mesh=config.viz.bird_mesh,
            per_bird_color=config.viz.per_bird_color,
            background_top=config.background_top,
            background_bottom=config.background_bottom,
            flap_period=config.flap_period,
            fps=config.fps,
        )
        self.camera = OrbitCamera(
            target=(config.width / 2, config.height / 2, config.depth / 2),
        )
        # P8.8: Dual-view second camera (15°/15°)
        self._dual_camera = OrbitCamera(
            target=(config.width / 2, config.height / 2, config.depth / 2),
        )
        self._dual_camera.azimuth = radians(15.0)
        self._dual_camera.elevation = radians(15.0)

        # P8.6: Adaptive quality governor (render_scale lives on self.renderer)
        self._governor = QualityGovernor(target_fps=config.target_fps)

        # P10.3: Slider HUD for live parameter tuning
        self._hud = SliderHUD(config)

        # G6: GL context loss tracking — warned once, then skipped
        self._gl_warned: bool = False

    def headless_frame(self) -> PILImage:
        """Render one frame and return a PIL Image (headless only).

        Pure render — does NOT step the simulation.
        P8.8: Supports dual-view rendering.
        P8.10: Passes lerped render_positions for smooth display.
        G6: If GL context is lost, returns a blank PIL Image.
        """
        rpos = self.sim.render_positions
        result = self._render_safe(rpos)
        assert result is not None, "headless_frame requires headless=True"
        return result

    def frame(self) -> None:
        """Render one frame to the screen (windowed mode).

        Pure render — does NOT step the simulation.
        P8.8: Supports dual-view rendering.
        P8.10: Passes lerped render_positions for smooth display.
        G6: If GL context is lost, degrades to no-op (rendering skipped).
        """
        rpos = self.sim.render_positions
        self._render_safe(rpos)

    # ── G6: GL context loss detection + safe degradation ───────

    def _render_safe(self, rpos) -> PILImage | None:
        """G6: Render one frame with GL context loss protection.

        Detects moderngl.Error mid-render, sets the gl_lost flag on the
        renderer, logs a RuntimeWarning once, and falls back to a blank
        PIL Image for headless mode (or None for windowed).

        Tiered degradation:
          1. GL context OK → normal render (FBO or window)
          2. GL lost mid-frame → blank PIL Image / None
             (headless_frame() also tries mpl fallback via Recorder)
        """
        import warnings

        from PIL import Image

        if self.renderer.gl_lost:
            # Context already lost — skip rendering entirely
            return Image.new("RGB", (self.renderer.width, self.renderer.height))

        try:
            self.renderer.begin_frame(self.camera)
            if self.config.dual_view:
                self._render_dual(rpos)
            else:
                self._draw_birds_with_lerp(rpos)
                self._draw_threat_marker()
                self._draw_influencer_marker()
                self.renderer.draw_trails(self.sim.flock)
            self.renderer.end_frame()
            if self.renderer.headless:
                return self.renderer.capture_frame()
            return None
        except Exception as e:
            # G6: Re-raise programming errors — only catch GPU/driver failures
            if isinstance(e, (AttributeError, TypeError,
                              ImportError, NameError)):
                raise
            # G6: Any non-programming exception — potential GL loss
            self.renderer.gl_lost = True
            if not self._gl_warned:
                self._gl_warned = True
                warnings.warn(
                    f"GPU context lost — degrading to headless/mpl fallback. "
                    f"({type(e).__name__}: {e})",
                    RuntimeWarning, stacklevel=2,
                )
            # Fallback: blank PIL Image (caller handles mpl fallback via Recorder)
            return Image.new("RGB", (self.renderer.width, self.renderer.height))

    def run(self, input_ctrl: InputControl) -> None:
        """Main visualization loop — input → update → render."""
        import pygame

        clock = pygame.time.Clock()
        running = True
        while running:
            t0 = time.perf_counter()
            dt = clock.tick(self.config.fps) / 1000.0

            # P8.6: Apply quality decisions from previous frame's governor state.
            # Feeding happens at the END of the loop (after flip) so frame_ms
            # captures the full render+physics wall-clock time, not just the
            # clock.tick() blocking time.  Gated by config.perf.adaptive_quality.
            if not self.paused and self.config.perf.adaptive_quality:
                self._apply_quality_actions()

            # P10.4: Update viewport for cursor-ray unprojection
            input_ctrl.set_viewport(self.config.window_width, self.config.window_height)

            running = input_ctrl.handle_events()

            # P10.3: TAB toggles HUD visibility (gated by config.viz.hud)
            if input_ctrl.pending_hud_toggle and self.config.viz.hud:
                self._hud.toggle()
                input_ctrl.hud_visible = self._hud.visible
                input_ctrl.pending_hud_toggle = False

            self.camera.step_auto_rotate(dt)
            self.paused = input_ctrl.paused

            # P10.4: Suppress cursor-ray spawns when clicking on HUD sliders.
            # handle_events() records spawns on every sub-5px left-click;
            # if that click was over a HUD knob, discard the spawn.
            if self._hud.hit_test_any(input_ctrl.mouse_x, input_ctrl.mouse_y):
                input_ctrl.pending_spawn_bird.clear()
                input_ctrl.pending_spawn_predator.clear()

            # P10.4: Drain cursor-ray spawn commands (S2.E6: cube-law velocity)
            for pos in input_ctrl.pending_spawn_bird:
                self.sim.enqueue_spawn(pos, is_predator=False)
            input_ctrl.pending_spawn_bird.clear()
            for pos in input_ctrl.pending_spawn_predator:
                self.sim.enqueue_spawn(pos, is_predator=True)
            input_ctrl.pending_spawn_predator.clear()

            # S2.E6: Gather/scatter — continuously add/remove birds while
            # Shift (gather) or Alt (scatter) is held.
            if input_ctrl.gathering:
                self.sim.enqueue_add(5)
            if input_ctrl.scattering:
                self.sim.enqueue_remove(5)

            # P10.4: Drain clear + v0 adjustment
            if input_ctrl.pending_clear:
                self.sim.enqueue_clear()
                input_ctrl.pending_clear = False
            if input_ctrl.pending_v0_delta != 0.0:
                self.config.v0 = max(0.3, self.config.v0 + input_ctrl.pending_v0_delta)
                input_ctrl.pending_v0_delta = 0.0

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

            # C1: predator_mode="cursor" — bridge live mouse world position
            # to the Predator extension (see physics/extensions/predator.py).
            if self.config.predator_mode == "cursor":
                world = self.camera.screen_to_world(
                    float(input_ctrl.mouse_x), float(input_ctrl.mouse_y),
                    self.config.window_width, self.config.window_height,
                )
                if world is not None:
                    object.__setattr__(self.config, '_cursor_world_pos', world)

            if not self.paused:
                self.sim.step(dt)

            # P8.10: Use lerped render_positions for smooth display
            rpos = self.sim.render_positions
            self._render_safe(rpos)
            if not self.renderer.gl_lost and input_ctrl.show_grid:
                self.renderer.draw_grid()

            # P10.3: Render slider HUD on top of 3D scene
            hud_lock = self._hud.handle_mouse(
                input_ctrl.mouse_x, input_ctrl.mouse_y,
                input_ctrl.mouse_down,
            )
            if hud_lock:
                # P10.3: Suppress camera orbit while dragging HUD slider
                input_ctrl.suppress_orbit()
            else:
                input_ctrl.release_orbit()
            # G6: Guard HUD rendering — skip if GL context is lost
            if not self.renderer.gl_lost:
                if self._hud.visible and self.config.viz.hud:
                    try:
                        self.renderer.hud_begin()
                        self._hud.render(
                            self.renderer,
                            input_ctrl.mouse_x, input_ctrl.mouse_y,
                        )
                        self.renderer.hud_end()
                    except Exception:
                        self.renderer.gl_lost = True

            # P10.2: Full title readout via FlockMetrics.summary()
            # Rebuilt every 20th frame for performance.
            # S3.11: Uses EMA-smoothed metrics for display (readout_smooth > 0).
            if self.sim.frame % 20 == 0:
                snap = self.sim.metrics.smoothed()
                summary = snap.summary(
                    mode=self.config.mode,
                    N_active=self.sim.flock.N_active,
                    fps=self.config.fps,
                    phi_p=self.config.projection.phi_p,
                    phi_a=self.config.phi_a,
                    sigma=self.config.sigma,
                )
                title = f"pymurmur — {summary}"
                if self._governor.degradation_level > 0:
                    title += f" [Q{self._governor.degradation_level}]"
                try:
                    pygame.display.set_caption(title)
                except Exception:
                    pass  # G6: non-critical — caption update may fail on GL loss
            try:
                pygame.display.flip()
            except Exception:
                self.renderer.gl_lost = True

            # P8.6: Feed governor at END of frame with full wall-clock time.
            # Measured from t0 (top of loop) through render + HUD + flip
            # so the governor sees actual frame cost, not just clock.tick()
            # blocking time.  Quality decisions are applied next iteration.
            # Gated by config.perf.adaptive_quality.
            if not self.paused and self.config.perf.adaptive_quality:
                frame_ms = (time.perf_counter() - t0) * 1000.0
                self._governor.feed(frame_ms)
                # S4.10: Feed bird count to PerfDiagnostics for risk classifier
                if self.sim.perf is not None:
                    self.sim.perf.set_active_count(self.sim.flock.N_active)

        pygame.quit()

    # ── P8.6: Adaptive quality actions ──────────────────────────

    def _render_dual(self, rpos=None) -> None:
        """P8.8: Render dual-view — left half (15°/15°) + right half (45°/45°).

        P8.10: Passes lerped positions for smooth display at low fps.
        """
        w = self.renderer.width
        h = self.renderer.height
        hw = w // 2
        # Left: camera at 15°/15°
        self.renderer.render_pass(self._dual_camera, 0, 0, hw, h)
        self._draw_birds_with_lerp(rpos)
        self._draw_threat_marker()
        self._draw_influencer_marker()
        # Right: main camera at 45°/45°
        self.renderer.render_pass(self.camera, hw, 0, w - hw, h)
        self._draw_birds_with_lerp(rpos)
        self._draw_threat_marker()
        self._draw_influencer_marker()
        # Trails drawn once in main viewport
        self.renderer.draw_trails(self.sim.flock)

    def _draw_birds_with_lerp(self, rpos) -> None:
        """P8.10: Draw birds with lerped positions when available."""
        if rpos is not None:
            self.renderer.draw_birds(self.sim.flock, positions_override=rpos)
        else:
            self.renderer.draw_birds(self.sim.flock)

    def _draw_threat_marker(self) -> None:
        """S2.A8/D7: render the predator/threat as a red, larger marker.

        An invisible predator is undebuggable — draw_layer() (D7) gives
        a non-instanced overlay seam for exactly this. Reuses the
        instanced tetra shader's predator_factor blend (scale > 1.0 →
        red glow, see shaders.py) so no new rendering path is needed.
        """
        pos = self.sim.extensions.predator_position
        if pos is None:
            return
        self.renderer.draw_layer(tuple(pos), hue=0.0, scale=1.5)

    def _draw_influencer_marker(self) -> None:
        """S2.E5/D7: render the influencer's Lissajous/pilot target.

        Same flag-channel mechanism as _draw_threat_marker (spec calls
        for reusing it, not a distinct visual convention) — the target
        is otherwise invisible, making the core-leads/tail-lags
        emergent behaviour hard to debug visually.
        """
        if self.config.mode != "influencer":
            return
        pos = getattr(self.config, "_influencer_target_pos", None)
        if pos is None:
            return
        self.renderer.draw_layer(tuple(pos), hue=0.0, scale=1.5)

    def _apply_quality_actions(self) -> None:
        """P8.6: Apply degradation or recovery from QualityGovernor.

        Degradation ladder:
            1. trails off
            2. render scale −0.15 (floor 0.75)
            3. N −18% (floor 512)
        """
        gov = self._governor
        if gov.should_degrade():
            level = gov.degradation_level
            if level == 1:
                self.renderer.disable_trails()
            elif level == 2:
                self.renderer.render_scale = max(
                    self.renderer.render_scale - gov.RENDER_SCALE_STEP,
                    gov.RENDER_SCALE_FLOOR,
                )
            elif level == 3:
                target = max(
                    int(self.sim.flock.N_active * (1.0 - gov.COUNT_STEP_FRACTION)),
                    gov.COUNT_FLOOR,
                )
                if self.sim.flock.N_active > target:
                    self.sim.enqueue_remove(self.sim.flock.N_active - target)
        elif gov.should_recover():
            level = gov.degradation_level
            if level == 0:  # fully recovered
                self.renderer.render_scale = 1.0
                if self.config.trails != "off":
                    self.renderer.enable_trails(
                        self.config.trails, self.config.trail_length
                    )
            elif level == 1:
                self.renderer.render_scale = min(
                    self.renderer.render_scale + gov.RENDER_SCALE_STEP, 1.0
                )
