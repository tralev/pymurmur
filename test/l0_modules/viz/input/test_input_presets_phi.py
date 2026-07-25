"""Tests for viz.input_control — P10.1 letter-key presets (a-f, h, w), P10.6 phi_p+phi_a <= 1 constraint.

Requires pygame. All tests skip when pygame is unavailable.

Split out of test_input.py (file-size split).
"""

import os

import pytest

# Check pygame availability & initialise once at module level
try:
    import pygame
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    PYGAME_AVAILABLE = True
except (ImportError, pygame.error):
    PYGAME_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed or init failed")


class TestLetterPresets:
    """P10.1: 8 letter-key presets (a–f, h, w) with printed labels."""

    def test_preset_a_applies_projection_params(self):
        """P10.1: Key 'a' → 3D Pearce Default: projection, φp=0.04, φa=0.80, σ=6."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.04)
        assert cfg.phi_a == pytest.approx(0.80)
        assert cfg.sigma == 6

    def test_preset_b_applies_storm_params(self):
        """P10.1: Key 'b' → Ball of Birds: projection, φp=0.18, φa=0.70, σ=7."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_b))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.18)
        assert cfg.phi_a == pytest.approx(0.70)
        assert cfg.sigma == 7

    def test_preset_c_applies_void_params(self):
        """P10.1: Key 'c' → Storm Cloud: projection, φp=0.06, φa=0.45, σ=3."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.06)
        assert cfg.phi_a == pytest.approx(0.45)
        assert cfg.sigma == 3

    def test_preset_d_applies_spatial_params(self):
        """P10.1: Key 'd' → 3D Stream: spatial, sep=0.25, align=0.55."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.25)
        assert cfg.alignment_weight == pytest.approx(0.55)
        assert cfg.cohesion_weight == pytest.approx(0.80)
        assert cfg.influence_count == 8

    def test_preset_e_applies_stream_params(self):
        """P10.1: Preset 'e' → Vertical Column: projection, φp=0.10, φa=0.75, σ=6.

        S2.E6: K_e now rolls camera — preset 'e' is accessed via
        _apply_letter_preset directly rather than a keyboard binding."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        ctrl._apply_letter_preset("e")

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.10)
        assert cfg.phi_a == pytest.approx(0.75)
        assert cfg.sigma == 6

    def test_preset_f_applies_ribbon_params(self):
        """P10.1: Key 'f' → 3D Acro: projection, φp=0.02, φa=0.85, σ=3."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        ctrl.handle_events()

        assert cfg.mode == "projection"
        assert cfg.projection.phi_p == pytest.approx(0.02)
        assert cfg.phi_a == pytest.approx(0.85)
        assert cfg.sigma == 3

    def test_preset_h_applies_huddle_params(self):
        """P10.1: Key 'h' → 3D Void: spatial, sep=0.35, align=0.58."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.35)
        assert cfg.alignment_weight == pytest.approx(0.58)
        assert cfg.cohesion_weight == pytest.approx(0.90)
        assert cfg.influence_count == 9

    def test_preset_w_applies_wander_params(self):
        """P10.1: Key 'w' → Spiral Vortex: spatial, sep=0.08, align=0.82."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        ctrl.handle_events()

        assert cfg.mode == "spatial"
        assert cfg.separation_weight == pytest.approx(0.08)
        assert cfg.alignment_weight == pytest.approx(0.82)
        assert cfg.cohesion_weight == pytest.approx(1.0)
        assert cfg.influence_count == 10


# ── P10.6: φp+φa ≤ 1 constraint ─────────────────────────────

class TestPhiConstraint:
    """P10.6: After φp or φa increments, φp + φa ≤ 1 is enforced."""

    def test_phi_sum_never_exceeds_one(self):
        """P10.6: Repeated φp increases → φa shrinks to keep sum ≤ 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.5
        cfg.phi_a = 0.5
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Bump φp repeatedly — φa should shrink to keep sum ≤ 1
        for _ in range(80):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10, (
            f"φp={cfg.projection.phi_p:.4f} + φa={cfg.phi_a:.4f} = "
            f"{cfg.projection.phi_p + cfg.phi_a:.4f} > 1.0"
        )

    def test_phi_a_increase_shrinks_phi_p(self):
        """P10.6: Increasing φa when at limit (sum=1.0) shrinks φp."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.6
        cfg.phi_a = 0.40  # sum = 1.00 exactly — one more bump triggers constraint
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # One more φa bump → sum would be 1.01 > 1.0, so φp shrinks to 1−φa
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        ctrl.handle_events()

        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, f"sum {total:.4f} > 1.0"
        # φa increased to 0.41, φp decreased to 0.59 to keep sum ≤ 1.0
        assert cfg.phi_a == pytest.approx(0.41)
        assert cfg.projection.phi_p == pytest.approx(0.59)

    def test_phi_p_decrease_never_violates(self):
        """P10.6: Decreasing φp never causes a constraint violation."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.3
        cfg.phi_a = 0.4
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        # φa should be unchanged when there's headroom
        assert cfg.phi_a == pytest.approx(0.4)

    def test_constraint_symmetric(self):
        """P10.6: Constraint is symmetric — pushing either parameter
        to 1.0 forces the other to 0.0."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Push φp to 1.0
        for _ in range(200):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        assert cfg.projection.phi_p >= 0.95  # effectively at limit
        assert cfg.phi_a < 0.05  # pushed to near-zero

        # Reset and push φa to 1.0
        cfg.projection.phi_p = 0.03
        cfg.phi_a = 0.50
        for _ in range(200):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
            ctrl.handle_events()
        assert cfg.projection.phi_p + cfg.phi_a <= 1.0 + 1e-10
        assert cfg.phi_a >= 0.95
        assert cfg.projection.phi_p < 0.05

    def test_phi_p_increase_from_max_no_op(self):
        """P10.6: φp at 1.0 (max clamped) — UP key no-ops, φa stays 0."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 1.0
        cfg.phi_a = 0.0
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        ctrl.handle_events()
        assert cfg.projection.phi_p == pytest.approx(1.0)
        assert cfg.phi_a == pytest.approx(0.0)

    def test_phi_a_decrease_from_zero_no_op(self):
        """P10.6: φa at 0.0 (min clamped) — LEFT key no-ops."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.3
        cfg.phi_a = 0.0
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.0)

    def test_letter_preset_enforces_constraint(self, monkeypatch):
        """P10.6: Letter preset that would violate constraint is corrected."""
        from pymurmur.analysis.presets import LETTER_PRESETS
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Inject a violating preset for key 'z' (unused key)
        monkeypatch.setitem(LETTER_PRESETS, "z", (
            "Test Violator", "φp=0.9 φa=0.9",
            {"mode": "projection", "phi_p": 0.90, "phi_a": 0.90},
        ))
        # Directly call _apply_letter_preset to avoid needing a real pygame key
        ctrl._apply_letter_preset("z")
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, (
            f"Letter preset with sum=1.8 should be corrected, got sum={total:.4f}"
        )

    def test_numbered_preset_enforces_constraint(self, monkeypatch):
        """P10.6: Numbered preset (1-9) that would violate constraint is corrected."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import PRESETS, InputControl
        cfg = SimConfig()
        cfg.mode = "projection"
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Inject a violating preset
        monkeypatch.setitem(PRESETS, "violator", {
            "phi_p": 0.80,
            "phi_a": 0.80,
        })
        monkeypatch.setattr(
            "pymurmur.viz.input_control.PRESETS",
            {"violator": {"phi_p": 0.80, "phi_a": 0.80}},
        )
        # Directly call _apply_preset with index that resolves to "violator"
        ctrl._apply_preset(0)
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, (
            f"Numbered preset with sum=1.6 should be corrected, got sum={total:.4f}"
        )

    def test_interleaved_up_right_never_violates(self):
        """P10.6: Interleaving UP and RIGHT key presses never violates constraint."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.5
        cfg.phi_a = 0.4
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        # Alternate UP and RIGHT 50 times
        for i in range(50):
            key = pygame.K_UP if i % 2 == 0 else pygame.K_RIGHT
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            ctrl.handle_events()
            total = cfg.projection.phi_p + cfg.phi_a
            assert total <= 1.0 + 1e-10, (
                f"Interleaved step {i}: sum={total:.4f} (φp={cfg.projection.phi_p:.4f}, φa={cfg.phi_a:.4f})"
            )

    def test_phi_p_decrease_with_headroom_leaves_phi_a_unchanged(self):
        """P10.6: Decreasing φp when sum < 1 leaves φa untouched (no enforcement triggers)."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.2
        cfg.phi_a = 0.3
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.3)

    def test_sum_exactly_one_no_change(self):
        """P10.6: When phi_p+phi_a=1.0 exactly, no enforcement triggers."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.camera import OrbitCamera
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.4
        cfg.phi_a = 0.6
        cam = OrbitCamera()
        ctrl = InputControl(cfg, cam)
        # Decreasing phi_p should leave phi_a alone (headroom opened)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        ctrl.handle_events()
        assert cfg.phi_a == pytest.approx(0.6)
        assert cfg.projection.phi_p == pytest.approx(0.39)

    def test_enforce_phi_after_preset_standalone(self):
        """P10.6: _enforce_phi_after_preset reduces the smaller value when sum > 1."""
        from pymurmur.core.config import SimConfig
        from pymurmur.viz.input_control import InputControl
        cfg = SimConfig()
        cfg.projection.phi_p = 0.7
        cfg.phi_a = 0.6  # sum=1.3 > 1.0, phi_a is smaller
        InputControl._enforce_phi_after_preset(cfg)
        total = cfg.projection.phi_p + cfg.phi_a
        assert total <= 1.0 + 1e-10, f"Sum should be <= 1.0, got {total}"
        # phi_p=0.7 > phi_a=0.6, so phi_a should be reduced to 0.3
        assert cfg.projection.phi_p == pytest.approx(0.7)
        assert cfg.phi_a == pytest.approx(0.3)


