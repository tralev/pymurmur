"""P14.1 — Architecture DAG static data: STDLIB/THIRD_PARTY allowlists,
ALLOWED_EDGES, FORBIDDEN_EDGES, KNOWN_VIOLATIONS, PHASE_EDGES,
PHASE_VIOLATION_REMOVALS.

Split out of test_architecture.py (file-size split) — pure data, no
test functions (0 tests collected from this file is expected).
ALLOWED_EDGES grows at each phase acceptance boundary; FORBIDDEN_EDGES
is permanent. See test_architecture_edges.py for the enforcement
logic/tests that consume this data.
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

ALLOWED_EDGES: dict[str, set[str]] = {
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
    "pymurmur.core.config.config_validation": set(),
    "pymurmur.core.config.config_io": {"pymurmur.core.config.config_field_map"},

    # ── Tier 0: physics/obstacles (L0 atom, P0.14) — core only ──
    "pymurmur.physics.obstacles": {"pymurmur.core.types"},

    # ── Tier 1: physics L0 atoms — core only ──
    "pymurmur.physics.boid":      {"pymurmur.core.types"},
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
    },

    # ── Tier 1: physics/spatial_index (L1 atom, extracted from flock.py
    # for file-size — SpatialHashGrid/KDTreeIndex) — core only ──
    "pymurmur.physics.spatial_index": {
        "pymurmur.core.types",
        "pymurmur.core.config",  # TYPE_CHECKING only
    },

    # ── Tier 2: physics/forces (L1, L0) ──
    "pymurmur.physics.forces._mode": {
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
    },
    "pymurmur.physics.forces.force_kernels": set(),  # numpy only, zero pymurmur imports
    "pymurmur.physics.forces._kernels": {
        # S2.B3: min_image for toroidal-aware predator escape distances.
        "pymurmur.core.types",
    },
    "pymurmur.physics.forces.spatial": {
        "pymurmur.core.types",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces._kernels",
        "pymurmur.physics.forces.spatial_helpers",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
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
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.field": {
        "pymurmur.core.types",
        "pymurmur.physics.forces._mode",
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
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces._kernels",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.influencer": {
        "pymurmur.core.types",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.marl": {  # P12.1: MARL force mode
        "pymurmur.core.types",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
    },
    "pymurmur.physics.forces.angle": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.flock",
    },
    "pymurmur.physics.forces.__init__": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces.spatial",
        "pymurmur.physics.forces.projection",
        "pymurmur.physics.forces.field",
        "pymurmur.physics.forces.vicsek",
        "pymurmur.physics.forces.influencer",
        "pymurmur.physics.forces.angle",
        "pymurmur.physics.forces.marl",  # P12.1
    },

    # ── Tier 2: physics/extensions (L1) — core + read flock ──
    "pymurmur.physics.extensions._base": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
    },
    "pymurmur.physics.extensions.predator": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.ecology": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.wander": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.ripple": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.speed_noise": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.core.noise",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.neighbor_adaptive_speed": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.adaptive_speed",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.dynamic_vision_range": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
    },
    "pymurmur.physics.extensions.__init__": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.predator",
        "pymurmur.physics.extensions.ecology",
        "pymurmur.physics.extensions.wander",
        "pymurmur.physics.extensions.ripple",
        "pymurmur.physics.extensions.speed_noise",
        "pymurmur.physics.extensions.neighbor_adaptive_speed",
        "pymurmur.physics.extensions.dynamic_vision_range",
    },

    # ── Tier 3: simulation/engine (L2) — core + physics + analysis ──
    "pymurmur.simulation.engine": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.forces",
        "pymurmur.physics.extensions",
        "pymurmur.physics.obstacles",  # S6.4: ObstacleScene
        "pymurmur.physics.priority_stack",
        "pymurmur.analysis.metrics",
        "pymurmur.analysis.perf",      # S4.10: PerfDiagnostics
    },

    # ── Tier F1: Observables — core + read flock ──
    # metrics.py is now a thin re-export shim (file-size split), moved
    # into analysis/metrics/__init__.py (logical-structure split) —
    # dotted path "pymurmur.analysis.metrics" unchanged. The
    # implementation lives in the nested sibling modules below.
    "pymurmur.analysis.metrics": {
        "pymurmur.analysis.metrics.collector",
        "pymurmur.analysis.metrics.consensus_robustness",
        "pymurmur.analysis.metrics.dynamics_curves",
        "pymurmur.analysis.metrics.flock_metrics",
        "pymurmur.analysis.metrics.opacity",
        "pymurmur.analysis.metrics.shape_motion",
    },
    "pymurmur.analysis.metrics.flock_metrics": {"pymurmur.core.types"},
    "pymurmur.analysis.metrics.consensus_robustness": set(),
    "pymurmur.analysis.metrics.opacity": set(),
    "pymurmur.analysis.metrics.shape_motion": set(),
    "pymurmur.analysis.metrics.dynamics_curves": set(),
    "pymurmur.analysis.metrics.collector": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
        "pymurmur.analysis.metrics.consensus_robustness",
        "pymurmur.analysis.metrics.dynamics_curves",
        "pymurmur.analysis.metrics.flock_metrics",
        "pymurmur.analysis.metrics.opacity",
        "pymurmur.analysis.metrics.shape_motion",
    },
    "pymurmur.analysis.presets": {
        "pymurmur.core.types",
        "pymurmur.core.config",
    },
    "pymurmur.analysis.perf": {
        "pymurmur.core.types",
        "pymurmur.core.config",          # P8.6: PerfConfig.target_fps
    },

    # ── Tier F2: Drivers — core + simulation ──
    # Logical-structure split: evoflock.py -> analysis/evoflock/__init__.py
    # (dotted path "pymurmur.analysis.evoflock" unchanged, it's still the
    # package root); evoflock_objectives.py moved alongside it.
    "pymurmur.analysis.evoflock": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
        "pymurmur.physics.obstacles",  # P11.4: ObstacleScene evaluation
        "pymurmur.analysis.evoflock.evoflock_objectives",
    },
    # File-size split from evoflock.py: per-step objective collector +
    # objective-function helpers (_ObjectiveCollector, load_obstacle_scene,
    # _trapezoid, _linear_ramp, _pareto_front).
    "pymurmur.analysis.evoflock.evoflock_objectives": {
        "pymurmur.core.types",
        "pymurmur.physics.obstacles",
        "pymurmur.analysis.evoflock",  # TYPE_CHECKING only (Genome)
    },
    # Logical-structure split: phase_diagram.py/density_scaling.py/
    # point_clouds.py/topological_range.py moved from flush in analysis/
    # into a new analysis/research/ subpackage (independent Young-et-al-
    # 2013 validation scripts, no shared API so research/__init__.py is
    # empty). point_clouds.py and topological_range.py have no entries
    # here since they import nothing from pymurmur.
    "pymurmur.analysis.research.phase_diagram": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
    },
    # Logical-structure split: gym_env.py + rewards.py moved from flush in
    # analysis/ into a new analysis/rl/ subpackage (RL bridge). Both
    # dotted paths change; rl/__init__.py re-exports MurmurationEnv/
    # RewardConfig/compute_reward for ergonomics.
    "pymurmur.analysis.rl": {
        "pymurmur.analysis.rl.gym_env",
        "pymurmur.analysis.rl.rewards",
    },
    "pymurmur.analysis.rl.rewards": {
        "pymurmur.core.types",
        "pymurmur.analysis.metrics",
    },
    "pymurmur.analysis.rl.gym_env": {  # P12.2: MurmurationEnv
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
        "pymurmur.analysis.rl.rewards",
        "pymurmur.analysis.metrics",
    },
    "pymurmur.analysis.research.density_scaling": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
    },

    # ── Viz (L2) — core + physics/flock(read) + analysis/presets ──
    "pymurmur.viz.renderer": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.viz.shaders",
        "pymurmur.viz.camera",
        "pymurmur.viz.trails",
        "pymurmur.viz.mesh_registry",  # S4.4a
        "pymurmur.viz.renderer_vao",
        "pymurmur.viz.renderer_draw",
    },
    # File-size split from renderer.py: VAO-building mixin.
    "pymurmur.viz.renderer_vao": {"pymurmur.viz.mesh_registry"},
    # File-size split from renderer.py: drawing mixin.
    "pymurmur.viz.renderer_draw": {
        "pymurmur.core.types",
        "pymurmur.viz.mesh_registry",
        "pymurmur.physics.flock",  # TYPE_CHECKING only
    },
    "pymurmur.viz.shaders": {
        "pymurmur.core.types",
    },
    "pymurmur.viz.camera": {
        "pymurmur.core.types",
        "pymurmur.core.config",
    },
    "pymurmur.viz.visualizer": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.viz.renderer",
        "pymurmur.viz.camera",
        "pymurmur.viz.input_control",
        "pymurmur.viz.hud",              # P10.3: SliderHUD
        "pymurmur.viz.trails",           # P8.6: trail re-creation on recovery
        "pymurmur.analysis.metrics",
        "pymurmur.analysis.perf",        # P8.6: QualityGovernor
    },
    "pymurmur.viz.hud": {
        "pymurmur.core.types",
        "pymurmur.core.config",          # P10.3: TYPE_CHECKING — reads config fields
    },
    "pymurmur.viz.input_control": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.core.logging",        # S5.6: cli_out/cli_err
        "pymurmur.analysis.presets",
        "pymurmur.viz.camera",
    },

    "pymurmur.viz.trails": {
        "pymurmur.core.types",
        "pymurmur.physics.flock",
        "pymurmur.viz.renderer",
        "pymurmur.viz.shaders",
        "pymurmur.viz.camera",
    },

    # ── Viz __init__ (re-exports) ──
    "pymurmur.viz.__init__": {
        "pymurmur.viz.visualizer",
        "pymurmur.viz.renderer",
        "pymurmur.viz.camera",
    },

    # ── Capture (L2) — core + simulation + viz ──
    "pymurmur.capture.recorder": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.analysis.metrics",
        "pymurmur.simulation.engine",
        "pymurmur.viz.visualizer",
        "pymurmur.viz.renderer",
        "pymurmur.capture.mpl_recorder",  # P8.9 fallback
        "pymurmur.viz.camera",
    },
    "pymurmur.capture.mpl_recorder": {       # P8.9: GPU-free fallback
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
    },
}

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


