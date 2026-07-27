"""P14.1 — Architecture DAG static data: PHASE_EDGES, PHASE_VIOLATION_REMOVALS.

Split out of test_architecture_edges_data.py (file-size split, pure
extraction) — historical per-phase snapshots (P0-P12), never touched by
new modules added after P14 (see test_architecture_edges_data.py for the
current full-matrix ALLOWED_EDGES those new modules belong in instead).
"""

from __future__ import annotations

# ── Per-Phase Edge Sets ──────────────────────────────────────────────
# Each phase boundary extends ALLOWED_EDGES with the new edges introduced
# by that phase. The full matrix (P14) matches arch.md §5 exactly.
# No edge is ever removed from ALLOWED_EDGES once added.

PHASE_EDGES = {
    "P0": {
        "pymurmur.core.types": set(),
        "pymurmur.physics.boid": {"pymurmur.core.types"},
        "pymurmur.physics.obstacles": {"pymurmur.core.types"},
    },
    "P1": {
        "pymurmur.physics.occlusion": {"pymurmur.core.types"},
        "pymurmur.physics.steric": {"pymurmur.core.types"},
        "pymurmur.physics.forces._base": {"pymurmur.core.types"},
        "pymurmur.physics.forces.vicsek": {"pymurmur.core.types"},
        "pymurmur.analysis.metrics": {"pymurmur.core.types", "pymurmur.physics.flock"},
    },
    "P2": {
        "pymurmur.core.config": {"pymurmur.core.types"},
        "pymurmur.physics.flock": {"pymurmur.core.types"},
        "pymurmur.physics.forces._mode": {"pymurmur.core.types"},
        "pymurmur.physics.forces.projection": {"pymurmur.core.types", "pymurmur.physics.occlusion", "pymurmur.physics.steric", "pymurmur.physics.forces._base", "pymurmur.physics.flock"},
        "pymurmur.physics.forces.spatial": {"pymurmur.core.types", "pymurmur.physics.forces._base", "pymurmur.physics.forces._kernels", "pymurmur.physics.flock"},
        "pymurmur.physics.extensions._base": {"pymurmur.core.types", "pymurmur.physics.flock"},
        "pymurmur.simulation.engine": {"pymurmur.core.types", "pymurmur.physics.flock", "pymurmur.physics.forces._mode", "pymurmur.physics.extensions._base", "pymurmur.analysis.metrics"},
    },
    "P3": {
        "pymurmur.physics.forces.field": {"pymurmur.core.types", "pymurmur.physics.flock"},
        "pymurmur.physics.extensions.predator": {"pymurmur.core.types", "pymurmur.physics.flock", "pymurmur.physics.forces"},
        "pymurmur.physics.extensions.wander": {"pymurmur.core.types"},
        "pymurmur.physics.extensions.ripple": {"pymurmur.core.types", "pymurmur.physics.flock"},
    },
    "P4": {
        "pymurmur.physics.forces._kernels": {"pymurmur.core.types"},
        "pymurmur.physics.extensions.ecology": {"pymurmur.core.types", "pymurmur.physics.flock"},
    },
    "P5": {
        "pymurmur.physics.forces.angle": {"pymurmur.core.types", "pymurmur.physics.flock"},
    },
    "P6": set(),
    "P7": {
        "pymurmur.physics.forces.influencer": {"pymurmur.core.types", "pymurmur.physics.flock"},
        "pymurmur.viz.input_control": {"pymurmur.core.types"},
    },
    "P8": {
        "pymurmur.viz.renderer": {"pymurmur.core.types", "pymurmur.physics.flock", "pymurmur.analysis.presets"},
        "pymurmur.viz.shaders": set(),
        "pymurmur.viz.camera": set(),
        "pymurmur.viz.visualizer": {"pymurmur.core.types", "pymurmur.viz.trails", "pymurmur.analysis.perf"},
        "pymurmur.viz.trails": {"pymurmur.core.types", "pymurmur.physics.flock", "pymurmur.viz.renderer", "pymurmur.viz.shaders", "pymurmur.viz.camera"},
        "pymurmur.capture.recorder": {"pymurmur.simulation.engine", "pymurmur.viz.visualizer", "pymurmur.core.types"},
        "pymurmur.capture.mpl_recorder": {
            "pymurmur.core.config",          # P8.9: TYPE_CHECKING
            "pymurmur.simulation.engine",    # P8.9: TYPE_CHECKING
            "pymurmur.core.types",
        },
        "pymurmur.analysis.perf": {"pymurmur.core.types", "pymurmur.core.config"},
    },
    "P9": {
        "pymurmur.analysis.rewards": {"pymurmur.core.types", "pymurmur.analysis.metrics"},
        "pymurmur.analysis.phase_diagram": {"pymurmur.core.types", "pymurmur.physics.flock"},
        "pymurmur.analysis.density_scaling": {"pymurmur.core.types", "pymurmur.physics.flock"},
    },
    "P10": {
        "pymurmur.viz.hud": {"pymurmur.core.types", "pymurmur.core.config"},
        "pymurmur.viz.visualizer": {"pymurmur.viz.hud"},
        "pymurmur.__init__": {"pymurmur.core.config", "pymurmur.simulation.engine"},
        "pymurmur.__main__": set(),
    },
    "P11": {
        "pymurmur.analysis.evoflock": {"pymurmur.simulation.engine", "pymurmur.core.types"},
    },
    "P12": {
        "pymurmur.physics.forces.marl": {"pymurmur.core.types", "pymurmur.physics.flock"},
        "pymurmur.analysis.gym_env": {"pymurmur.simulation.engine", "pymurmur.core.types"},
    },
}

# ── Phase-Gated Violation Removal ───────────────────────────────────
# At each phase boundary, certain KNOWN_VIOLATIONS are removed because
# the phase resolves the underlying issue.

PHASE_VIOLATION_REMOVALS = {
    "P8": [
        ("pymurmur.viz.visualizer", "pymurmur.simulation.engine"),
    ],
}
