"""Holey-mask composition tests — all 7 force modes survive inactive slots mid-array.

I3.4: Verifies that force modes handle holey active masks where
inactive birds are interspersed among active ones (not just at the end).
"""

import numpy as np
import pytest
from pymurmur.core.config import SimConfig
from pymurmur.physics.flock import PhysicsFlock


from test.helpers import _step_flock  # noqa: E402 — shared test helper


ALL_MODES = ["projection", "spatial", "field", "vicsek", "influencer"]


@pytest.fixture
def holey_config():
    """Config with 30 birds — every 3rd bird deactivated mid-array."""
    cfg = SimConfig()
    cfg.num_boids = 30
    cfg.seed = 42
    cfg.spatial_index = "kdtree"  # ensure KDTree for spatial/vicsek modes
    return cfg


@pytest.fixture
def holey_flock(holey_config):
    """Flock with holey active mask: birds at indices 0-9 and 20-29 active,
    birds at 10-19 inactive."""
    flock = PhysicsFlock(holey_config)
    # Deactivate a middle block of 10 birds
    flock.active[10:20] = False
    return flock


@pytest.mark.parametrize("mode", ALL_MODES)
def test_force_mode_survives_holey_mask(holey_flock, holey_config, mode):
    """Each force mode completes a step without error on holey masks."""
    holey_config.mode = mode

    # Some modes need scipy
    if mode in ("spatial", "vicsek", "projection"):
        pytest.importorskip("scipy")

    holey_flock.accelerations[:] = 0.0

    # Should not raise
    _step_flock(holey_flock, holey_config, 1.0 / 60.0)

    # Active birds should have moved
    active = holey_flock.active
    assert holey_flock.N_active > 0
    assert np.isfinite(holey_flock.positions[active]).all()
    assert np.isfinite(holey_flock.velocities[active]).all()


@pytest.mark.parametrize("mode", ALL_MODES)
def test_inactive_birds_unchanged(holey_flock, holey_config, mode):
    """Inactive birds' positions and velocities don't change during step."""
    holey_config.mode = mode

    if mode in ("spatial", "vicsek", "projection"):
        pytest.importorskip("scipy")

    inactive = ~holey_flock.active
    pos_before = holey_flock.positions[inactive].copy()
    vel_before = holey_flock.velocities[inactive].copy()

    _step_flock(holey_flock, holey_config, 1.0 / 60.0)

    # Inactive birds should be unchanged
    assert np.allclose(holey_flock.positions[inactive], pos_before)
    assert np.allclose(holey_flock.velocities[inactive], vel_before)
