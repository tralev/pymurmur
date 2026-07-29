"""P14.1 — Architecture DAG static data: STDLIB/THIRD_PARTY allowlists,
ALLOWED_EDGES, FORBIDDEN_EDGES, KNOWN_VIOLATIONS, PHASE_EDGES,
PHASE_VIOLATION_REMOVALS.

Split out of test_architecture.py (file-size split) — pure data, no
test functions (0 tests collected from this file is expected).
ALLOWED_EDGES grows at each phase acceptance boundary; FORBIDDEN_EDGES
is permanent. See test_architecture_edges.py for the enforcement
logic/tests that consume this data.

Physics-tier (core/physics) ALLOWED_EDGES entries live here as
ALLOWED_EDGES_CORE; system-tier (simulation/analysis/viz/capture)
entries live in test_architecture_edges_data_system.py (file-size
split of this file) as ALLOWED_EDGES_SYSTEM, dict-spread back into
ALLOWED_EDGES below.
"""

from __future__ import annotations


STDLIB = {
    "abc", "ast", "builtins", "collections", "colorsys", "concurrent", "copy",
    "dataclasses", "datetime", "enum", "functools", "hashlib", "itertools",
    "json", "logging", "math", "operator", "os", "pathlib", "pickle",
    "re", "sys", "time", "typing", "warnings", "weakref", "__future__",
    "inspect", "textwrap", "argparse", "csv", "io", "struct",
    "threading",
}

THIRD_PARTY = {
    "numpy", "scipy", "PyYAML", "yaml", "PIL", "Pillow",
    "pygame", "moderngl", "PyGLM", "glm",
    "numba", "matplotlib", "gymnasium",
    "stable_baselines3", "pytest",
}

# ── ALLOWED_EDGES — subpackage-level rules ────────────────────────

# Format: {module_prefix: set of allowed import prefixes}
# An import from A to B is allowed if B.startswith(any prefix in ALLOWED_EDGES[A]).
# If the module has no entry, only stdlib + third-party imports are allowed.
# TYPE_CHECKING imports ARE subject to ALLOWED_EDGES — they're part of the
# architecture contract (types flow along allowed edges).

ALLOWED_EDGES_CORE: dict[str, set[str]] = {
    # ── Tier 0: core/ — numpy/stdlib only, zero pymurmur imports ──
    "pymurmur.core.types": set(),
    "pymurmur.core.noise": set(),
    # File-size split: config.py is now a slim SimConfig class that
    # composes/re-exports the modules below. Logical-structure split:
    # config.py -> core/config/__init__.py, siblings moved alongside it
    # into core/config/ (dotted path "pymurmur.core.config" unchanged).
    "pymurmur.core.config": {
        "pymurmur.core.config.config_field_map",
        "pymurmur.core.config.config_io",
        "pymurmur.core.config.config_sections",
        "pymurmur.core.config.config_validation",
    },
    "pymurmur.core.config.config_sections": set(),
    "pymurmur.core.config.config_field_map": set(),
    "pymurmur.core.config.config_validation": {
        "pymurmur.core.config.config_validation_extensions",
    },
    "pymurmur.core.config.config_validation_extensions": set(),
    "pymurmur.core.config.config_io": {"pymurmur.core.config.config_field_map"},

    # ── Tier 0: physics/obstacles (L0 atom, P0.14) — core only ──
    "pymurmur.physics.obstacles": {"pymurmur.core.types"},
    # Modularity pass 3: wraps ObstacleScene.avoidance_accel() unchanged
    # behind a registry, mirroring boundary/neighbor_selection. Imports
    # ObstacleScene only under TYPE_CHECKING (for typing, not at runtime).
    "pymurmur.physics.plugins.obstacle_avoidance": {"pymurmur.physics.obstacles"},

    # ── Tier 1: physics L0 atoms — core only ──
    "pymurmur.physics.boid":      {
        "pymurmur.core.types",
        "pymurmur.physics.boid_init",  # re-exports init helpers (file-size split)
        "pymurmur.physics.plugins.boundary",  # boundary-strategy registry dispatch
        "pymurmur.physics.plugins.speed_model",  # modularity pass 4: speed-model registry dispatch
    },
    "pymurmur.physics.boid_init": set(),  # numpy only, zero pymurmur imports
    "pymurmur.physics.plugins.boundary": {
        "pymurmur.physics.plugins.boundary._mode",
        "pymurmur.physics.plugins.boundary.strategies",
    },
    "pymurmur.physics.plugins.boundary._mode": set(),  # numpy only (TYPE_CHECKING), zero pymurmur imports
    "pymurmur.physics.plugins.boundary.strategies": {
        "pymurmur.physics.plugins.boundary._mode",
    },
    "pymurmur.physics.occlusion": {"pymurmur.core.types"},
    "pymurmur.physics.steric":    {"pymurmur.core.types"},
    "pymurmur.physics.priority_stack": set(),  # numpy only, zero pymurmur imports
    "pymurmur.physics.adaptive_speed": set(),  # numpy only, zero pymurmur imports

    # ── Tier 2: physics/flock (L1) — core + boid only, NEVER forces ──
    "pymurmur.physics.flock": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.boid",
        "pymurmur.physics.spatial_index",
        "pymurmur.physics.plugins.spatial_index_strategy",  # modularity pass 5: index selection dispatch
    },

    # ── Tier 1: physics/spatial_index (L1 atom, extracted from flock.py
    # for file-size — SpatialHashGrid/KDTreeIndex) — core only ──
    "pymurmur.physics.spatial_index": {
        "pymurmur.core.types",
        "pymurmur.core.config",  # TYPE_CHECKING only
    },

    # Modularity pass 4: SpeedModel ABC + SPEED_MODEL_REGISTRY (L0 atom).
    "pymurmur.physics.plugins.speed_model": set(),  # numpy only, zero pymurmur imports

    # Modularity pass 5: SpatialIndexStrategy registry (L1 — depends on spatial_index).
    "pymurmur.physics.plugins.spatial_index_strategy": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.spatial_index",
    },

    # ── Tier 2: physics/forces (L1, L0) ──
    "pymurmur.physics.plugins.force_mode": {
        "pymurmur.core.types",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces._base": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.forces.force_kernels",
        "pymurmur.physics.plugins.kernel_registry",  # modularity pass 7: kernel dispatch
    },
    "pymurmur.physics.forces.force_kernels": set(),  # numpy only, zero pymurmur imports
    # Modularity pass 7: kernel registry — imports force_kernels for dispatch.
    "pymurmur.physics.plugins.kernel_registry": {
        "pymurmur.physics.forces.force_kernels",
    },
    "pymurmur.physics.forces._kernels": {
        # S2.B3: min_image for toroidal-aware predator escape distances.
        "pymurmur.core.types",
    },
    "pymurmur.physics.forces.spatial": {
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces._kernels",
        "pymurmur.physics.forces.spatial_helpers",
        "pymurmur.physics.plugins.neighbor_selection",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    # File-size split from spatial.py/projection.py/vicsek.py (modularity
    # pass 2): NeighborSelector ABC + registry, formalising the 3 modes'
    # neighbor-query strategies. Lazy-imports .spatial_helpers/.projection
    # inside select() method bodies (not at module level) to avoid a
    # module-level circular import with spatial.py/projection.py, which
    # both import this module at module level in turn — still counted as
    # edges by the AST walker (it walks function bodies too), so both are
    # listed as allowed targets below. AngleMode's neighbor banding is
    # deliberately NOT included (fused into its own per-bird steering
    # loop, not safely extractable — see the modularity-pass-2 plan).
    "pymurmur.physics.plugins.neighbor_selection": {
        "pymurmur.core.config",
        "pymurmur.core.types",
        "pymurmur.physics.forces.spatial_helpers",
        "pymurmur.physics.forces.projection",
    },
    # File-size split from spatial.py: neighbour-query/filter helpers
    # (_query_neighbors, _apply_hybrid_filter, _maybe_perception_filter,
    # _predator_escape). Imports .spatial locally (inside function
    # bodies, not at module level) for _dispatch_kernels — breaks what
    # would otherwise be a module-level circular import.
    "pymurmur.physics.forces.spatial_helpers": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.forces.spatial",
    },
    "pymurmur.physics.forces.projection": {
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.plugins.neighbor_selection",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.field": {
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
        "pymurmur.physics.extensions.wander",
        "pymurmur.physics.extensions.ripple",
        "pymurmur.physics.forces.field_anchors",
        "pymurmur.physics.forces.field_terms",
    },
    # File-size split from field.py: blob-anchor/leader-chaser setup
    # functions. Pure numpy, zero pymurmur imports.
    "pymurmur.physics.forces.field_anchors": set(),
    # File-size split from field.py: the 14 per-bird force-term
    # implementations. Imports field_anchors for _compute_phases
    # (shared between the shell-force term and the anchor pipeline).
    "pymurmur.physics.forces.field_terms": {
        "pymurmur.core.types",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces.field_anchors",
    },
    "pymurmur.physics.forces.vicsek": {
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces._kernels",
        "pymurmur.physics.plugins.neighbor_selection",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.influencer": {
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.marl": {  # P12.1: MARL force mode
        "pymurmur.core.types",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.angle": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.flock",
    },
    "pymurmur.physics.forces.__init__": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.plugins.force_mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces.spatial",
        "pymurmur.physics.forces.projection",
        "pymurmur.physics.forces.field",
        "pymurmur.physics.forces.vicsek",
        "pymurmur.physics.forces.influencer",
        "pymurmur.physics.forces.angle",
        "pymurmur.physics.forces.marl",  # P12.1
        "pymurmur.physics.plugins.neighbor_selection",
    },

    # ── Tier 2: physics/extensions (L1) — core + read flock ──
    "pymurmur.physics.extensions._base": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
    },
    # Modularity pass 6: Extension registry — TYPE_CHECKING core + _base.
    "pymurmur.physics.extensions.extension_registry": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.predator": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.ecology": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.wander": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.ripple": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.speed_noise": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.core.noise",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.neighbor_adaptive_speed": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.adaptive_speed",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.dynamic_vision_range": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.boid_state_machine": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
    },
    "pymurmur.physics.extensions.__init__": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.extension_registry",  # modularity pass 6
        "pymurmur.physics.extensions.predator",
        "pymurmur.physics.extensions.ecology",
        "pymurmur.physics.extensions.wander",
        "pymurmur.physics.extensions.ripple",
        "pymurmur.physics.extensions.speed_noise",
        "pymurmur.physics.extensions.neighbor_adaptive_speed",
        "pymurmur.physics.extensions.dynamic_vision_range",
        "pymurmur.physics.extensions.boid_state_machine",
    },

}

from .test_architecture_edges_data_system import ALLOWED_EDGES_SYSTEM

ALLOWED_EDGES: dict[str, set[str]] = {**ALLOWED_EDGES_CORE, **ALLOWED_EDGES_SYSTEM}

# ── FORBIDDEN_EDGES — never-allowed import pairs ──────────────────

FORBIDDEN_EDGES: list[tuple[str, str]] = [
    # Import cycle: flock must never import forces
    ("pymurmur.physics.flock", "pymurmur.physics.forces"),
    # Input layer must not couple to the simulation engine
    ("pymurmur.viz.input_control", "pymurmur.simulation"),
    # Core must never import anything from pymurmur
    ("pymurmur.core", "pymurmur.physics"),
    ("pymurmur.core", "pymurmur.simulation"),
    ("pymurmur.core", "pymurmur.analysis"),
    ("pymurmur.core", "pymurmur.viz"),
    ("pymurmur.core", "pymurmur.capture"),
    # Physics L0 atoms must not import flock
    ("pymurmur.physics.occlusion", "pymurmur.physics.flock"),
    ("pymurmur.physics.steric", "pymurmur.physics.flock"),
    ("pymurmur.physics.boid", "pymurmur.physics.flock"),
    ("pymurmur.physics.boid", "pymurmur.physics.forces"),
    # Forces must not import extensions (except via TYPE_CHECKING) or simulation
    ("pymurmur.physics.forces", "pymurmur.physics.extensions"),
    ("pymurmur.physics.forces", "pymurmur.simulation"),
]

# ── KNOWN VIOLATIONS — not yet fixed, scheduled per roadmap phases ──

KNOWN_VIOLATIONS: list[tuple[str, str, str]] = [
    # viz/visualizer.py TYPE_CHECKING-imports simulation.engine (reference only)
    ("pymurmur.viz.visualizer", "pymurmur.simulation.engine", "accepted_ref"),
]

# ── Per-Phase Edge Sets ──────────────────────────────────────────────
# Historical P0-P12 snapshots + phase-gated violation removals now live
# in test_architecture_phase_data.py (file-size split, pure extraction)
# — re-imported/re-exported here so existing
# `from test_architecture_edges_data import PHASE_EDGES` call sites
# keep working.
from .test_architecture_phase_data import (  # noqa: E402, F401
    PHASE_EDGES,
    PHASE_VIOLATION_REMOVALS,
)


