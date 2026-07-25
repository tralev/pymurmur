"""Extensions — StepContext dataclass (I5.1), Extension ABC (I5.2), predator marker position, lazy extension lifecycle toggles (I5.3).

Split out of test_extensions.py (file-size split).
"""

import numpy as np
import pytest

from pymurmur.physics.extensions import ExtensionManager
from pymurmur.physics.extensions._base import Extension, StepContext
from pymurmur.physics.flock import PhysicsFlock
from test.l0_modules.physics.extensions.test_extensions import _make_ctx

# ── StepContext dataclass (I5.1) ──────────────────────────────────

class TestStepContext:
    """M1-M2: StepContext dataclass contract."""

    def test_step_context_all_fields_present(self, default_config):
        """M1: All 6 fields present with correct types."""
        cfg = default_config
        cfg.num_boids = 10
        flock = PhysicsFlock(cfg)
        # Step once so flock has actual positions → center is computed
        flock.integrate(cfg, 0.016)
        center = flock.center

        ctx = StepContext(
            frame=42, dt=0.016, rng=flock.rng,
            center=center, config=cfg,
        )

        assert isinstance(ctx.frame, int)
        assert isinstance(ctx.dt, float)
        assert hasattr(ctx.rng, 'random')  # numpy Generator
        # center can be None or ndarray — after integrate(), it's an ndarray
        assert ctx.center is not None
        assert hasattr(ctx.center, 'shape')
        assert hasattr(ctx.config, 'num_boids')  # SimConfig
        # threat_prox defaults to None
        assert ctx.threat_prox is None

        # Verify all 6 field names match what extensions expect
        expected_fields = {
            'frame', 'dt', 'rng', 'center', 'config', 'threat_prox'
        }
        actual_fields = set(ctx.__dataclass_fields__.keys())
        assert actual_fields == expected_fields, (
            f"StepContext fields changed: {actual_fields}"
        )

    def test_step_context_threat_prox_defaults_to_none(self, default_config):
        """M2: threat_prox must default to None (not a mutable array).

        A mutable default (e.g. np.zeros(N)) would cause shared-state
        bugs across multiple contexts.
        """
        rng = np.random.default_rng(42)

        ctx1 = StepContext(
            frame=0, dt=0.016, rng=rng, center=None,
            config=default_config,
        )
        # Validate the default explicitly
        assert ctx1.threat_prox is None, (
            f"threat_prox must default to None, got {type(ctx1.threat_prox)}"
        )

        # Also verify via the field default in the class definition
        field_default = StepContext.__dataclass_fields__['threat_prox'].default
        assert field_default is None, (
            f"StepContext.threat_prox field default must be None, "
            f"got {field_default!r}"
        )


# ── Extension ABC (I5.2) ──────────────────────────────────────────

class TestExtensionABC:
    """M3: Extension abstract base class contract."""

    def test_extension_abc_cannot_instantiate(self):
        """M3: Extension() must raise TypeError (abstract class).

        If someone removes ABCMeta or @abstractmethod, the protocol
        silently degrades — concrete extensions can be instantiated
        without implementing apply().
        """
        with pytest.raises(TypeError, match="abstract"):
            Extension()  # type: ignore[abstract]

    def test_extension_subclass_without_apply_cannot_instantiate(self):
        """Extension subclass without apply() is also abstract."""
        class IncompleteExtension(Extension):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteExtension()  # type: ignore[abstract]

    def test_extension_subclass_with_apply_can_instantiate(self):
        """Extension subclass implementing apply() is concrete."""
        class CompleteExtension(Extension):
            def apply(self, flock, ctx):
                pass

        ext = CompleteExtension()
        assert isinstance(ext, Extension)


class TestPredatorMarkerPosition:
    """D7/S2.A8: ExtensionManager.predator_position feeds the threat
    marker draw_layer() call — an invisible predator is undebuggable."""

    def test_none_when_predator_disabled(self, default_config):
        default_config.predator_enabled = False
        mgr = ExtensionManager(default_config)
        assert mgr.predator_position is None

    def test_position_when_predator_enabled(self, default_config):
        default_config.predator_enabled = True
        mgr = ExtensionManager(default_config)
        pos = mgr.predator_position
        assert pos is not None
        assert pos.shape == (3,)

    def test_position_tracks_predator_movement(self, default_config):
        default_config.predator_enabled = True
        default_config.num_boids = 20
        mgr = ExtensionManager(default_config)
        flock = PhysicsFlock(default_config)

        p0 = mgr.predator_position.copy()
        for i in range(1, 30):
            ctx = _make_ctx(flock, default_config, frame=i)
            mgr.pre_step(flock, ctx)
        p1 = mgr.predator_position
        assert not (p0 == p1).all(), "Predator marker position must track FSM movement"


# ── Lazy extension lifecycle (I5.3) ───────────────────────────────

class TestLazyExtensionLifecycle:
    """M4-M14: Extensions are lazily created/dropped on config toggle.

    pre_step() checks cfg.*_enabled each frame and creates or drops
    extensions without requiring a simulation reset.
    """

    @staticmethod
    def _mk_manager(config, predator=False, ecology=False,
                    wander=False, ripple=False):
        """Create an ExtensionManager with specified initial state."""
        config.predator_enabled = predator
        config.roosting_enabled = ecology
        config.wander_enabled = wander
        config.ripple_enabled = ripple
        return ExtensionManager(config)

    @staticmethod
    def _mk_flock_and_ctx(config):
        """Create a flock and StepContext for pre_step calls."""
        flock = PhysicsFlock(config)
        ctx = StepContext(
            frame=0, dt=1.0 / 60.0, rng=flock.rng,
            center=flock.center, config=config,
        )
        return flock, ctx

    # ── Predator lazy create/drop (M4, M5) ──────────────────────

    def test_lazy_create_predator_mid_simulation(self, default_config):
        """M4: predator_enabled False→True creates Predator on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, predator=False)
        assert mgr._predator is None

        # Toggle on
        cfg.predator_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._predator is not None, (
            "Predator must be lazily created when predator_enabled becomes True"
        )

    def test_lazy_drop_predator_mid_simulation(self, default_config):
        """M5: predator_enabled True→False drops Predator on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, predator=True)
        assert mgr._predator is not None

        # Toggle off
        cfg.predator_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._predator is None, (
            "Predator must be dropped when predator_enabled becomes False"
        )

    # ── Ecology lazy create/drop (M6, M7) ───────────────────────

    def test_lazy_create_ecology_mid_simulation(self, default_config):
        """M6: roosting_enabled False→True creates Ecology on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ecology=False)
        assert mgr._ecology is None

        cfg.roosting_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ecology is not None, (
            "Ecology must be lazily created when roosting_enabled becomes True"
        )

    def test_lazy_drop_ecology_mid_simulation(self, default_config):
        """M7: roosting_enabled True→False drops Ecology on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ecology=True)
        assert mgr._ecology is not None

        cfg.roosting_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ecology is None, (
            "Ecology must be dropped when roosting_enabled becomes False"
        )

    # ── Wander lazy create/drop (M8, M9) ────────────────────────

    def test_lazy_create_wander_mid_simulation(self, default_config):
        """M8: wander_enabled False→True creates Wander on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, wander=False)
        assert mgr._wander is None

        cfg.wander_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._wander is not None, (
            "Wander must be lazily created when wander_enabled becomes True"
        )

    def test_lazy_drop_wander_mid_simulation(self, default_config):
        """M9: wander_enabled True→False drops Wander on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, wander=True)
        assert mgr._wander is not None

        cfg.wander_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._wander is None, (
            "Wander must be dropped when wander_enabled becomes False"
        )

    # ── Ripple lazy create/drop (M10, M11) ──────────────────────

    def test_lazy_create_ripple_mid_simulation(self, default_config):
        """M10: ripple_enabled False→True creates Ripple on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ripple=False)
        assert mgr._ripple is None

        cfg.ripple_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ripple is not None, (
            "Ripple must be lazily created when ripple_enabled becomes True"
        )

    def test_lazy_drop_ripple_mid_simulation(self, default_config):
        """M11: ripple_enabled True→False drops Ripple on next pre_step."""
        cfg = default_config
        mgr = self._mk_manager(cfg, ripple=True)
        assert mgr._ripple is not None

        cfg.ripple_enabled = False
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)

        assert mgr._ripple is None, (
            "Ripple must be dropped when ripple_enabled becomes False"
        )

    # ── Count accuracy (M12) ────────────────────────────────────

    def test_lazy_toggle_count_updates(self, default_config):
        """M12: ExtensionManager.count reflects current enabled state.

        Toggling extensions on/off must update the count immediately.
        """
        cfg = default_config
        cfg.num_boids = 10

        # Start with all disabled
        mgr = self._mk_manager(cfg, predator=False, ecology=False,
                               wander=False, ripple=False)
        assert mgr.count == 0

        flock, ctx = self._mk_flock_and_ctx(cfg)

        # Enable each one at a time, verify count
        cfg.predator_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 1, f"After predator: {mgr.count}"

        cfg.roosting_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 2, f"After ecology: {mgr.count}"

        cfg.wander_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 3, f"After wander: {mgr.count}"

        cfg.ripple_enabled = True
        mgr.pre_step(flock, ctx)
        assert mgr.count == 4, f"After ripple: {mgr.count}"

        # Disable all one at a time
        cfg.ripple_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 3, f"After -ripple: {mgr.count}"

        cfg.wander_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 2, f"After -wander: {mgr.count}"

        cfg.roosting_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 1, f"After -ecology: {mgr.count}"

        cfg.predator_enabled = False
        mgr.pre_step(flock, ctx)
        assert mgr.count == 0, f"After -predator: {mgr.count}"

    # ── No recreate if already present (M13) ────────────────────

    def test_lazy_toggle_no_recreate_if_already_present(
        self, default_config
    ):
        """M13: Toggling False→True→False→True creates fresh but no duplicates.

        After the sequence, count must be 1 (not 2), and the extension
        must be functional.
        """
        cfg = default_config
        cfg.num_boids = 10

        # Start with predator disabled
        mgr = self._mk_manager(cfg, predator=False)
        assert mgr._predator is None
        assert mgr.count == 0

        # Toggle on — fresh ctx
        cfg.predator_enabled = True
        flock, ctx = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock, ctx)
        assert mgr._predator is not None
        assert mgr.count == 1
        first_predator = mgr._predator

        # pre_step again with predator still enabled — must NOT recreate
        mgr.pre_step(flock, ctx)
        assert mgr._predator is first_predator, (
            "pre_step must not recreate extension if already present"
        )
        assert mgr.count == 1, (
            f"Count must stay 1 when already-present extension is not recreated. "
            f"Got {mgr.count}"
        )

        # Toggle off — fresh ctx
        cfg.predator_enabled = False
        flock2, ctx2 = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock2, ctx2)
        assert mgr._predator is None
        assert mgr.count == 0

        # Toggle on again — new instance, but still count=1
        cfg.predator_enabled = True
        flock3, ctx3 = self._mk_flock_and_ctx(cfg)
        mgr.pre_step(flock3, ctx3)
        assert mgr._predator is not None
        assert mgr._predator is not first_predator, (
            "After drop+recreate, must be a new instance"
        )
        assert mgr.count == 1, (
            f"After revive, count must be 1, got {mgr.count}"
        )

    # ── Initial state matches config (M14) ──────────────────────

    def test_lazy_toggle_initial_state_matches_config(self, default_config):
        """M14: ExtensionManager.__init__ matches initial config for all 4."""
        # All enabled
        cfg_all = default_config
        mgr_all = self._mk_manager(cfg_all, predator=True, ecology=True,
                                   wander=True, ripple=True)
        assert mgr_all._predator is not None
        assert mgr_all._ecology is not None
        assert mgr_all._wander is not None
        assert mgr_all._ripple is not None
        assert mgr_all.count == 4

        # All disabled
        cfg_none = default_config
        mgr_none = self._mk_manager(cfg_none, predator=False, ecology=False,
                                    wander=False, ripple=False)
        assert mgr_none._predator is None
        assert mgr_none._ecology is None
        assert mgr_none._wander is None
        assert mgr_none._ripple is None
        assert mgr_none.count == 0

        # Mixed: predator + wander only
        cfg_mix = default_config
        mgr_mix = self._mk_manager(cfg_mix, predator=True, ecology=False,
                                   wander=True, ripple=False)
        assert mgr_mix._predator is not None
        assert mgr_mix._ecology is None
        assert mgr_mix._wander is not None
        assert mgr_mix._ripple is None
        assert mgr_mix.count == 2


