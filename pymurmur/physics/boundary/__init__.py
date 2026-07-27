"""Boundary strategy dispatch and registry.

Level 0 — pure numpy, zero pymurmur imports beyond this subpackage.

Modularity pass: formalises pymurmur's 5 boundary modes (toroidal, open,
margin, sphere, sphere_soft) behind BOUNDARY_REGISTRY (populated by
@register decorators), mirroring physics/forces/_mode.py's
ForceMode/MODE_REGISTRY pattern. Each strategy module registers its
BoundaryMode subclass at import time.
"""
from __future__ import annotations

from . import strategies  # noqa: F401 — triggers @register decorators
from ._mode import BOUNDARY_REGISTRY, BoundaryMode, register  # noqa: F401 — public API
from .strategies import (  # noqa: F401 — re-exports
    MarginBoundary,
    OpenBoundary,
    SphereBoundary,
    SphereSoftBoundary,
    ToroidalBoundary,
)
