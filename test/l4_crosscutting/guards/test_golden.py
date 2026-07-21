"""P0 — Golden trajectory regression tests.

Verifies that the simulation engine produces bit-identical results
when run with the same seed and config that generated the golden files.

P2.5: Expanded to cross-product force modes × boundary modes
(toroidal vs sphere) so both boundary paths are regression-tested.
"""
import numpy as np
import pytest

from pymurmur.core.config import SimConfig
from pymurmur.simulation.engine import SimulationEngine

pytestmark = [pytest.mark.golden, pytest.mark.guard]



GOLDEN_MODES = ['projection', 'spatial', 'field', 'vicsek', 'influencer', 'angle']
BOUNDARY_MODES = ['toroidal', 'sphere']

# Parameters used when generating golden files — must match the values
# in golden regeneration scripts exactly.
_GOLDEN_NUM_BOIDS = 15
_GOLDEN_SEED = 77
_GOLDEN_FRAMES = 30
_GOLDEN_DT = 1.0 / 60.0

# Sphere radius used when generating golden files for the sphere boundary.
# The default BoundaryConfig.boundary_sphere_radius is 300.0 — the golden
# files were generated with a smaller radius for more visible boundary effects.
_GOLDEN_SPHERE_RADIUS = 200.0

# All modes are deterministic (P0.4 / I1.5: flock.rng plumbed everywhere).
# Golden files regenerated for all modes.
NONDETERMINISTIC: set = set()


def _golden_path(mode: str, boundary: str) -> str:
    """Resolve golden file path for a (mode, boundary) pair.

    Toroidal is the default — keeps the legacy filename for backward
    compatibility.  Non-toroidal variants append the boundary name.
    """
    if boundary == 'toroidal':
        return f'test/data/golden_{mode}.npz'
    return f'test/data/golden_{mode}_{boundary}.npz'


def _make_config(mode: str, boundary: str) -> SimConfig:
    """Build a SimConfig for the given mode × boundary combination."""
    cfg = SimConfig()
    cfg.mode = mode
    cfg.num_boids = _GOLDEN_NUM_BOIDS
    cfg.seed = _GOLDEN_SEED
    cfg.boundary_mode = boundary
    if boundary == "sphere":
        cfg.boundary_sphere_radius = _GOLDEN_SPHERE_RADIUS
    return cfg


@pytest.mark.parametrize('mode', GOLDEN_MODES)
@pytest.mark.parametrize('boundary', BOUNDARY_MODES)
def test_matches_golden(mode, boundary):
    """Same seed + config → final frame matches golden within atol=1e-3."""
    if mode in NONDETERMINISTIC:
        pytest.xfail(
            f'{mode} is non-deterministic — module-level np.random.* calls '
            f'ignore config.seed. Fixed in P0.4 (single seeded RNG).'
        )
    golden_path = _golden_path(mode, boundary)
    golden = np.load(golden_path)

    cfg = _make_config(mode, boundary)
    engine = SimulationEngine(cfg)
    # Golden frame 0 is initial state, frame N is after N steps.
    # Golden has _GOLDEN_FRAMES frames (0.._GOLDEN_FRAMES-1), so
    # step _GOLDEN_FRAMES-1 times to reach the final frame.
    for _ in range(_GOLDEN_FRAMES - 1):
        engine.step(_GOLDEN_DT)

    np.testing.assert_allclose(
        engine.flock.positions, golden['pos'][-1], atol=1e-3,
        err_msg=f'{mode}/{boundary}: positions diverged from golden'
    )
    np.testing.assert_allclose(
        engine.flock.velocities, golden['vel'][-1], atol=1e-3,
        err_msg=f'{mode}/{boundary}: velocities diverged from golden'
    )


@pytest.mark.parametrize('mode', GOLDEN_MODES)
@pytest.mark.parametrize('boundary', BOUNDARY_MODES)
def test_golden_file_exists(mode, boundary):
    """Golden .npz file must exist before any physics changes."""
    import os
    path = _golden_path(mode, boundary)
    assert os.path.exists(path), f'Missing golden file: {path}'


def test_golden_files_have_expected_shape():
    """All golden files have shape (30, 15, 3) for both pos and vel."""
    for mode in GOLDEN_MODES:
        for boundary in BOUNDARY_MODES:
            golden = np.load(_golden_path(mode, boundary))
            expected_shape = (_GOLDEN_FRAMES, _GOLDEN_NUM_BOIDS, 3)
            assert golden['pos'].shape == expected_shape, (
                f'{mode}/{boundary}: pos shape {golden["pos"].shape}'
            )
            assert golden['vel'].shape == expected_shape, (
                f'{mode}/{boundary}: vel shape {golden["vel"].shape}'
            )
            assert golden['pos'].dtype == np.float32
            assert golden['vel'].dtype == np.float32


@pytest.mark.parametrize('mode', GOLDEN_MODES)
@pytest.mark.parametrize('boundary', BOUNDARY_MODES)
def test_all_frames_match_golden(mode, boundary):
    """Every frame (not just the last) matches golden within atol=1e-3.

    Catches mid-simulation regressions that might be masked if only
    the final frame is checked.  All mode × boundary pairs covered (P2.5).
    """
    golden_path = _golden_path(mode, boundary)
    golden = np.load(golden_path)

    cfg = _make_config(mode, boundary)
    engine = SimulationEngine(cfg)
    for frame in range(_GOLDEN_FRAMES):
        np.testing.assert_allclose(
            engine.flock.positions, golden['pos'][frame], atol=1e-3,
            err_msg=f'{mode}/{boundary}: positions diverged at frame {frame}'
        )
        np.testing.assert_allclose(
            engine.flock.velocities, golden['vel'][frame], atol=1e-3,
            err_msg=f'{mode}/{boundary}: velocities diverged at frame {frame}'
        )
        engine.step(_GOLDEN_DT)


def test_golden_midpoint_not_trivial():
    """Golden trajectories are non-trivial — positions change meaningfully.

    All mode × boundary pairs covered (P2.5).
    """
    for mode in GOLDEN_MODES:
        for boundary in BOUNDARY_MODES:
            golden = np.load(_golden_path(mode, boundary))
            pos_start = golden['pos'][0]
            pos_end = golden['pos'][-1]
            # Birds should move meaningfully over 30 frames
            displacement = np.linalg.norm(pos_end - pos_start, axis=1).mean()
            assert displacement > 0.3, (
                f'{mode}/{boundary}: mean displacement={displacement:.2f} — birds barely moved'
            )
