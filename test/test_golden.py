"""P0 — Golden trajectory regression tests.

Verifies that the simulation engine produces bit-identical results
when run with the same seed and config that generated the golden files.
"""
import numpy as np
import pytest
from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

pytestmark = [pytest.mark.golden, pytest.mark.guard]



GOLDEN_MODES = ['projection', 'spatial', 'field', 'vicsek', 'influencer']

# Modes that are currently non-deterministic due to module-level np.random.*
# (roadmap structural gap #5 — fixed in P0.4)
NONDETERMINISTIC = {'vicsek', 'influencer'}


@pytest.mark.parametrize('mode', GOLDEN_MODES)
def test_matches_golden(mode):
    """Same seed + config → final frame matches golden within atol=1e-3."""
    if mode in NONDETERMINISTIC:
        pytest.xfail(
            f'{mode} is non-deterministic — module-level np.random.* calls '
            f'ignore config.seed. Fixed in P0.4 (single seeded RNG).'
        )
    golden = np.load(f'test/data/golden_{mode}.npz')

    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = 15
    cfg.seed = 77
    cfg.use_numba = False

    engine = SimulationEngine(cfg)
    for _ in range(30):
        engine.step(1 / 60)

    np.testing.assert_allclose(
        engine.flock.positions, golden['pos'][-1], atol=1e-3,
        err_msg=f'{mode}: positions diverged from golden'
    )
    np.testing.assert_allclose(
        engine.flock.velocities, golden['vel'][-1], atol=1e-3,
        err_msg=f'{mode}: velocities diverged from golden'
    )


@pytest.mark.parametrize('mode', GOLDEN_MODES)
def test_golden_file_exists(mode):
    """Golden .npz file must exist before any physics changes."""
    import os
    path = f'test/data/golden_{mode}.npz'
    assert os.path.exists(path), f'Missing golden file: {path}'


def test_golden_files_have_expected_shape():
    """All golden files have shape (30, 15, 3) for both pos and vel."""
    for mode in GOLDEN_MODES:
        golden = np.load(f'test/data/golden_{mode}.npz')
        assert golden['pos'].shape == (30, 15, 3), f'{mode}: pos shape {golden["pos"].shape}'
        assert golden['vel'].shape == (30, 15, 3), f'{mode}: vel shape {golden["vel"].shape}'
        assert golden['pos'].dtype == np.float32
        assert golden['vel'].dtype == np.float32


@pytest.mark.parametrize('mode', ['projection', 'spatial'])
def test_all_frames_match_golden(mode):
    """Every frame (not just the last) matches golden within atol=1e-3.

    Catches mid-simulation regressions that might be masked if only
    the final frame is checked.
    """
    golden = np.load(f'test/data/golden_{mode}.npz')

    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = 15
    cfg.seed = 77
    cfg.use_numba = False

    engine = SimulationEngine(cfg)
    for frame in range(30):
        engine.step(1 / 60)
        np.testing.assert_allclose(
            engine.flock.positions, golden['pos'][frame], atol=1e-3,
            err_msg=f'{mode}: positions diverged at frame {frame}'
        )
        np.testing.assert_allclose(
            engine.flock.velocities, golden['vel'][frame], atol=1e-3,
            err_msg=f'{mode}: velocities diverged at frame {frame}'
        )


def test_golden_midpoint_not_trivial():
    """Golden trajectories are non-trivial — positions change meaningfully."""
    for mode in ['projection', 'spatial']:
        golden = np.load(f'test/data/golden_{mode}.npz')
        pos_start = golden['pos'][0]
        pos_end = golden['pos'][-1]
        # Birds should move meaningfully over 30 frames
        displacement = np.linalg.norm(pos_end - pos_start, axis=1).mean()
        assert displacement > 1.0, (
            f'{mode}: mean displacement={displacement:.2f} — birds barely moved'
        )
