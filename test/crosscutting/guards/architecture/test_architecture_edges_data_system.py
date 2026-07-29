"""P14.1 — Architecture DAG static data: system-tier ALLOWED_EDGES entries
(simulation/analysis/viz/capture).

Split out of test_architecture_edges_data.py (file-size split) — pure
data, no test functions (0 tests collected from this file is expected).
Physics-tier (core/physics) entries stay in the original as
ALLOWED_EDGES_CORE; this file's ALLOWED_EDGES_SYSTEM is dict-spread
back into ALLOWED_EDGES there, mirroring the existing PHASE_EDGES
re-export precedent already used by the original file.
"""

from __future__ import annotations

ALLOWED_EDGES_SYSTEM: dict[str, set[str]] = {
    # ── Tier 3: simulation/engine (L2) — core + physics + analysis ──
    "pymurmur.simulation.engine": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.forces",
        "pymurmur.physics.extensions",
        "pymurmur.physics.obstacles",  # S6.4: ObstacleScene
        "pymurmur.physics.plugins.obstacle_avoidance",  # modularity pass 3
        "pymurmur.physics.priority_stack",
        "pymurmur.analysis.metrics",
        "pymurmur.analysis.perf",      # S4.10: PerfDiagnostics
        "pymurmur.simulation.command_queue",  # file-size split
    },
    # File-size split from engine.py: CommandQueue + _CommandQueueMixin
    # (enqueue_*/drain_commands). The influencer-pilot-specific draining
    # (needs physics.forces.influencer) deliberately stays on
    # SimulationEngine itself as _drain_pilot_commands() — only
    # simulation.engine may import both physics.flock and physics.forces
    # (I4.2 M3 architecture guard), so this module stays flock-only.
    "pymurmur.simulation.command_queue": {
        "pymurmur.core.config",
        "pymurmur.physics.flock",
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
        "pymurmur.viz.shaders_meshes",  # file-size split from shaders.py
        "pymurmur.viz.shaders_themes",  # file-size split from shaders.py
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
    # File-size splits from shaders.py: mesh vertex/index data (numpy
    # only) and theme palettes (pure dict, zero pymurmur imports).
    "pymurmur.viz.shaders_meshes": set(),
    "pymurmur.viz.shaders_themes": set(),
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
        "pymurmur.viz.trails_modes",  # file-size split
    },
    # File-size split from trails.py: accumulation/lines trail-mode mixin.
    "pymurmur.viz.trails_modes": {
        "pymurmur.physics.flock",  # TYPE_CHECKING only
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
