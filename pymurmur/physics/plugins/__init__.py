"""Computational plugin registries — per-strategy dispatch mechanisms
selecting among interchangeable algorithms for a physics computation
(force mode, boundary mode, neighbor selection, obstacle avoidance,
speed model, spatial index strategy, separation/alignment/cohesion
kernels, noise strategy).

Each submodule is independent; import the specific plugin family you
need (e.g. ``from pymurmur.physics.plugins.force_mode import
MODE_REGISTRY``) rather than through this package's ``__init__``,
which intentionally does not re-export anything.

Behavioral extension plugins (predator, ecology, wander, ripple, ...)
are a separate family and live in ``pymurmur.physics.extensions`` —
see arch.md's Plugins section for the full taxonomy.
"""

from __future__ import annotations
