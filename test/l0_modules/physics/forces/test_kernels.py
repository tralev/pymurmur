"""P4.10 — Numba kernel unit tests.

All kernel tests live in test_forces.py alongside the spatial mode
tests. This file re-exports them and adds any kernel-specific tests.

Tests in test_forces.py:
- test_numba_numpy_hybrid_filter_equivalence
- test_numba_numpy_predator_detect_equivalence
- test_numba_numpy_predator_escape_equivalence
- test_numba_predator_detect_excludes_predators
- test_numba_predator_escape_direction
- test_numba_predator_escape_scattered_zeros
- test_spatial_mode_numba_fallback
"""

import pytest


def test_kernels_file_exists():
    """P4.10: _kernels.py exists and is importable."""
    from pymurmur.physics.forces._kernels import (
        _numba_hybrid_filter,
        _numba_predator_detect,
        _numba_predator_escape,
        _numpy_hybrid_filter,
        _numpy_predator_detect,
        _numpy_predator_escape,
    )
    # All six functions must be importable
    assert callable(_numba_hybrid_filter)
    assert callable(_numpy_hybrid_filter)
    assert callable(_numba_predator_detect)
    assert callable(_numpy_predator_detect)
    assert callable(_numba_predator_escape)
    assert callable(_numpy_predator_escape)


def test_kernels_numpy_fallback_always_works():
    """P4.10: Pure-numpy fallback kernels work even without numba."""
    import numpy as np

    from pymurmur.physics.forces._kernels import (
        _numpy_hybrid_filter,
        _numpy_predator_detect,
        _numpy_predator_escape,
    )

    N, k = 10, 5
    positions = np.random.default_rng(42).uniform(0, 100, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    # Hybrid filter
    n_idx = np.zeros((N, k), dtype=np.int32)
    _numpy_hybrid_filter(n_idx, positions, active, visual_range=100.0, influence_count=3)
    assert np.isfinite(n_idx).all()

    # Predator detect
    is_pred = np.zeros(N, dtype=bool)
    is_pred[0] = True
    n_idx[:] = 0
    n_idx[1, 0] = 0
    threatened = np.zeros(N, dtype=bool)
    _numpy_predator_detect(threatened, n_idx, is_pred, active)
    assert isinstance(threatened, np.ndarray)

    # Predator escape
    escape = np.zeros((N, 3), dtype=np.float32)
    _numpy_predator_escape(escape, positions, n_idx, is_pred, threatened, active, 1e6, 1.0)
    assert np.isfinite(escape).all()


def test_scattered_zeros_all_three_kernels():
    """P4.10: Scattered-zeros regression — all three numba kernels handle
    neighbour arrays with interspersed zero sentinels correctly.

    This is the canonical regression test for the break→continue bug
    (P4.10 fix).  Neighbour arrays use zero as a "no neighbour" sentinel.
    If a kernel's inner loop encounters a zero and breaks instead of
    continuing, all subsequent valid neighbours are lost.

    Every kernel must skip zeros without aborting the scan.
    """
    import numpy as np

    from pymurmur.physics.forces._kernels import (
        _HAS_NUMBA,
        _numba_hybrid_filter,
        _numba_predator_detect,
        _numba_predator_escape,
        _numpy_hybrid_filter,
        _numpy_predator_detect,
        _numpy_predator_escape,
    )
    if not _HAS_NUMBA:
        pytest.skip("numba not available")

    N, k = 10, 10
    positions = np.random.default_rng(42).uniform(0, 200, (N, 3)).astype(np.float32)
    active = np.ones(N, dtype=bool)

    # ── Setup: scattered-zero neighbour array ──
    # Bird 0 has neighbours [1, 0, 2, 0, 3, 4, 0, 5, 6, 0]
    # Zeros at positions 1, 3, 6, 9 — interspersed with valid indices.
    scattered = np.array([1, 0, 2, 0, 3, 4, 0, 5, 6, 0], dtype=np.int32)

    # ── Setup: predator markers ──
    # Birds 3, 5, 6 are predators — scattered among the valid indices
    is_predator = np.zeros(N, dtype=bool)
    is_predator[3] = True   # at scattered[4] (after a zero)
    is_predator[5] = True   # at scattered[7] (after two zeros)
    is_predator[6] = True   # at scattered[8] (adjacent to 5)

    # ── 1. Hybrid filter: scattered zeros must not break neighbour scan ──
    n_idx_numba = np.zeros((N, k), dtype=np.int32)
    n_idx_numpy = np.zeros((N, k), dtype=np.int32)
    n_idx_numba[0, :] = scattered
    n_idx_numpy[0, :] = scattered

    _numba_hybrid_filter(n_idx_numba, positions, active,
                         visual_range=5000.0, influence_count=5)
    _numpy_hybrid_filter(n_idx_numpy, positions, active,
                         visual_range=5000.0, influence_count=5)

    assert np.array_equal(n_idx_numba, n_idx_numpy), \
        "Filter: numba and numpy must match with scattered zeros"
    valid_filter = n_idx_numba[0][n_idx_numba[0] > 0]
    assert len(valid_filter) > 0, \
        "Filter: should retain valid neighbours after zeros"
    assert len(valid_filter) <= 5, \
        f"Filter: should cap at influence_count=5, got {len(valid_filter)}"

    # ── 2. Predator detect: scattered zeros must not hide predators ──
    # Bird 0 sees predators at positions 4, 7, 8 in the scattered array
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, :] = scattered

    threatened_numba = np.zeros(N, dtype=bool)
    threatened_numpy = np.zeros(N, dtype=bool)
    _numba_predator_detect(threatened_numba, n_idx, is_predator, active)
    _numpy_predator_detect(threatened_numpy, n_idx, is_predator, active)

    assert np.array_equal(threatened_numba, threatened_numpy), \
        "Detect: numba and numpy must match with scattered zeros"
    assert threatened_numba[0], \
        "Detect: bird 0 must be threatened (sees predators 3,5,6)"

    # ── 3. Predator escape: scattered zeros must find nearest predator ──
    # Overwrite positions to create a controlled predator/prey layout.
    # Predators at d=2 (nearest), d=50, d=60 — all visible through zeros.
    # Recreate n_idx to avoid coupling with detect test's setup.
    n_idx = np.zeros((N, k), dtype=np.int32)
    n_idx[0, :] = scattered

    positions[1] = positions[0] + np.array([ 3.0, 0, 0], dtype=np.float32)  # near, non-predator
    positions[2] = positions[0] + np.array([ 8.0, 0, 0], dtype=np.float32)  # near, non-predator
    positions[3] = positions[0] + np.array([ 2.0, 0, 0], dtype=np.float32)  # nearest predator
    positions[4] = positions[0] + np.array([15.0, 0, 0], dtype=np.float32)  # far, non-predator
    positions[5] = positions[0] + np.array([50.0, 0, 0], dtype=np.float32)  # far predator
    positions[6] = positions[0] + np.array([60.0, 0, 0], dtype=np.float32)  # farthest predator

    threatened = np.zeros(N, dtype=bool)
    threatened[0] = True  # bird 0 is threatened prey

    escape_numba = np.zeros((N, 3), dtype=np.float32)
    escape_numpy = np.zeros((N, 3), dtype=np.float32)
    _numba_predator_escape(escape_numba, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)
    _numpy_predator_escape(escape_numpy, positions, n_idx, is_predator,
                            threatened, active, escape_factor=1e6, accel_boost=1.0)

    assert np.allclose(escape_numba, escape_numpy), \
        f"Escape: numba and numpy must match, diff max={np.abs(escape_numba - escape_numpy).max():.4f}"
    # Predator 3 is nearest (d=2) → escape force ≈ 1e6/4 = 250000
    # Predator 5 is far (d=50) → escape force ≈ 1e6/2500 = 400
    # If kernel picks predator 5 instead of 3, escape is ~600× weaker
    assert abs(escape_numba[0, 0]) > 10000.0, (\
        "Escape: must find nearest predator (d=2) through scattered zeros, "
        f"got {escape_numba[0, 0]:.1f} (expected ~250000)"
    )
