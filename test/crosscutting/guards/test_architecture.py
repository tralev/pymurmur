"""P14.1 — Architecture DAG matrix enforcement.

AST-walks every .py file under pymurmur/, resolves relative imports
to absolute module paths, and asserts every import edge is within
ALLOWED_EDGES and not in FORBIDDEN_EDGES.

Named regression edges (G3 — from roadmap_deepseek.md P14.1 + Part VII):
  - physics.flock !→ physics.forces    (import cycle kills composition DAG)
  - physics.forces !→ cKDTree          (index building belongs in flock)
  - viz.input_control !→ simulation    (input bridges must not couple to engine)
  - module-level np.random.*           (all randomness through flock.rng)
  - no print( in package sources       (use core/logging.py instead)

ALLOWED_EDGES grows at each phase acceptance boundary; FORBIDDEN_EDGES
is permanent.  KNOWN_VIOLATIONS tracks edges that are not yet fixed but
are scheduled for resolution in specific roadmap phases.

This file is extended at each phase boundary — the target-state matrix
matching arch.md §5 is enforced after P14.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# ── Third-party modules (always allowed) ──────────────────────────

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
    "pymurmur.core.config": {"pymurmur.core.types"},

    # ── Tier 0: physics/obstacles (L0 atom, P0.14) — core only ──
    "pymurmur.physics.obstacles": {"pymurmur.core.types"},

    # ── Tier 1: physics L0 atoms — core only ──
    "pymurmur.physics.boid":      {"pymurmur.core.types"},
    "pymurmur.physics.occlusion": {"pymurmur.core.types"},
    "pymurmur.physics.steric":    {"pymurmur.core.types"},

    # ── Tier 2: physics/flock (L1) — core + boid only, NEVER forces ──
    "pymurmur.physics.flock": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.boid",
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
    },
    "pymurmur.physics.forces._kernels": {
        # S2.B3: min_image for toroidal-aware predator escape distances.
        "pymurmur.core.types",
    },
    "pymurmur.physics.forces.spatial": {
        "pymurmur.core.types",
        "pymurmur.physics.forces._mode",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces._kernels",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.flock",
        "pymurmur.core.config",
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
    "pymurmur.physics.extensions.__init__": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.predator",
        "pymurmur.physics.extensions.ecology",
        "pymurmur.physics.extensions.wander",
        "pymurmur.physics.extensions.ripple",
    },

    # ── Tier 3: simulation/engine (L2) — core + physics + analysis ──
    "pymurmur.simulation.engine": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.forces",
        "pymurmur.physics.extensions",
        "pymurmur.physics.obstacles",  # S6.4: ObstacleScene
        "pymurmur.analysis.metrics",
        "pymurmur.analysis.perf",      # S4.10: PerfDiagnostics
    },

    # ── Tier F1: Observables — core + read flock ──
    "pymurmur.analysis.metrics": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
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
    "pymurmur.analysis.evoflock": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
        "pymurmur.physics.flock",
        "pymurmur.physics.boid",
        "pymurmur.physics.obstacles",  # P11.4: ObstacleScene evaluation
    },
    "pymurmur.analysis.phase_diagram": {
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
    },
    "pymurmur.analysis.rewards": {
        "pymurmur.core.types",
        "pymurmur.analysis.metrics",
    },
    "pymurmur.analysis.gym_env": {  # P12.2: MurmurationEnv
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.simulation.engine",
        "pymurmur.analysis.rewards",
        "pymurmur.analysis.metrics",
    },
    "pymurmur.analysis.density_scaling": {
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


def get_allowed_edges_for_phase(phase: str) -> dict[str, set[str]]:
    """Return the ALLOWED_EDGES active at the given phase boundary.

    Builds incrementally from an empty dict by accumulating PHASE_EDGES
    up to the target phase. P0 returns only core + physics/boid;
    P14 returns the full matrix matching arch.md §5.
    """
    edges: dict[str, set[str]] = {}

    target_num = int(phase[1:])
    for pn in sorted(int(k[1:]) for k in PHASE_EDGES):
        if pn <= target_num:
            ph = f"P{pn}"
            phase_value = PHASE_EDGES.get(ph, {})
            # Empty set means no new edges introduced at this phase (e.g. P6)
            if not isinstance(phase_value, dict):
                continue
            for mod, targets in phase_value.items():
                if mod not in edges:
                    edges[mod] = set()
                edges[mod] |= targets

    return edges


def get_known_violations_for_phase(phase: str) -> set[tuple[str, str]]:
    """Return KNOWN_VIOLATIONS with phase-gated removals applied."""
    violations = {(v[0], v[1]) for v in KNOWN_VIOLATIONS}
    target_num = int(phase[1:])

    for ph_key, removals in PHASE_VIOLATION_REMOVALS.items():
        ph_num = int(ph_key[1:])
        if ph_num <= target_num:
            violations -= set(removals)

    return violations


# ── Import resolution helpers ─────────────────────────────────────

def _resolve_relative_import(
    module_path: str,
    relative_module: str,
    level: int,
) -> str:
    """Resolve a relative import like `from .X import Y` or `from ..X import Y`."""
    parts = module_path.split(".")
    if level > len(parts):
        return ""
    base = parts[:-level] if level > 0 else parts
    if relative_module:
        return ".".join(base + [relative_module])
    return ".".join(base)


def _is_external(module_name: str) -> bool:
    """Return True if the module is stdlib or a known third-party package."""
    if module_name in STDLIB:
        return True
    top = module_name.split(".")[0]
    return top in THIRD_PARTY or top in STDLIB


def _module_is_allowed(source: str, target: str, in_tc: bool = False) -> bool:
    """Check if an import from *source* to *target* is allowed.

    TYPE_CHECKING imports are exempt from forbidden-edge checks — the
    type-flow contract allows type-only references that would be cycles
    at runtime.
    """
    if _is_external(target):
        return True
    if source in ("pymurmur.__init__", "pymurmur.__main__"):
        return True

    if not in_tc:
        for f_src, f_tgt in FORBIDDEN_EDGES:
            if source == f_src or source.startswith(f_src + "."):
                if target == f_tgt or target.startswith(f_tgt + "."):
                    return False

    matched_prefix = None
    for prefix in ALLOWED_EDGES:
        if source == prefix or source.startswith(prefix + "."):
            if matched_prefix is None or len(prefix) > len(matched_prefix):
                matched_prefix = prefix

    if matched_prefix is None:
        return False

    allowed_targets = ALLOWED_EDGES[matched_prefix]
    if not allowed_targets:
        return False

    for prefix in allowed_targets:
        if target == prefix or target.startswith(prefix + "."):
            return True

    return False


def _is_known_violation(source: str, target: str) -> bool:
    """Return True if this edge is a known violation scheduled for a future phase."""
    for v_src, v_tgt, _phase in KNOWN_VIOLATIONS:
        if (source == v_src or source.startswith(v_src + ".")) and \
           (target == v_tgt or target.startswith(v_tgt + ".")):
            return True
    return False


def _find_type_checking_line_ranges(filepath: Path) -> set[tuple[int, int]]:
    """Find line ranges of `if TYPE_CHECKING:` blocks in a file."""
    lines = filepath.read_text().split("\n")
    ranges: set[tuple[int, int]] = set()
    block_start: int | None = None
    block_indent: int = 0

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if block_start is not None:
            if stripped and indent <= block_indent:
                ranges.add((block_start, lineno - 1))
                block_start = None

        if stripped == "if TYPE_CHECKING:":
            block_start = lineno
            block_indent = indent

    if block_start is not None:
        ranges.add((block_start, len(lines)))

    return ranges


def _is_inside_type_checking(lineno: int, ranges: set[tuple[int, int]]) -> bool:
    """Check if a given line number falls within any TYPE_CHECKING block."""
    return any(start <= lineno <= end for start, end in ranges)


def _collect_import_edges() -> list[tuple[str, str, int, bool]]:
    """Walk every .py file under pymurmur/ and extract all import edges."""
    edges: list[tuple[str, str, int, bool]] = []

    for py_file in sorted(Path("pymurmur").rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        rel = str(py_file).replace("/", ".")
        if rel.endswith(".py"):
            source_module = rel[:-len(".py")]

        tree = ast.parse(py_file.read_text())
        tc_ranges = _find_type_checking_line_ranges(py_file)

        for node in ast.walk(tree):
            node_lineno = getattr(node, 'lineno', 0)
            in_tc = _is_inside_type_checking(node_lineno, tc_ranges)

            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if not _is_external(target):
                        edges.append((source_module, target, node_lineno, in_tc))

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                if node.level > 0:
                    target = _resolve_relative_import(
                        source_module, node.module, node.level
                    )
                else:
                    target = node.module
                if target and not _is_external(target):
                    edges.append((source_module, target, node_lineno, in_tc))

    return edges


# ── Tests


pytestmark = pytest.mark.guard

# ── P0.2 Phase-Gated Tests ──────────────────────────────────────
# These validate the incremental phase-gating mechanism itself.
# The full-matrix tests below (test_all_imports_within_allowed_edges,
# test_forbidden_edges_not_present) validate against the P14 target.


def test_p0_allowed_edges_minimal():
    """P0 ALLOWED_EDGES contains exactly core + physics/boid + physics/obstacles.

    Per roadmap P0.2 acceptance: ALLOWED_EDGES contains core + physics/boid.
    P0.14 adds physics/obstacles (L0 atom, core only).
    """
    p0_edges = get_allowed_edges_for_phase("P0")

    # Required entries per P0 acceptance criteria
    assert "pymurmur.core.types" in p0_edges, \
        "P0 must allow pymurmur.core.types (L0, numpy/stdlib only)"
    assert "pymurmur.physics.boid" in p0_edges, \
        "P0 must allow pymurmur.physics.boid (L0, imports core/types)"
    assert "pymurmur.physics.obstacles" in p0_edges, \
        "P0.14 must allow pymurmur.physics.obstacles (L0, imports core/types)"

    # Core types has zero internal pymurmur imports
    assert p0_edges["pymurmur.core.types"] == set(), \
        "pymurmur.core.types must have zero pymurmur imports (L0 atom)"

    # physics/boid may import core/types only
    assert "pymurmur.core.types" in p0_edges["pymurmur.physics.boid"], \
        "physics/boid must be allowed to import core/types"

    # physics/obstacles may import core/types only
    assert "pymurmur.core.types" in p0_edges["pymurmur.physics.obstacles"], \
        "physics/obstacles must be allowed to import core/types"

    print(f"✓ P0 ALLOWED_EDGES: {len(p0_edges)} modules with minimal dependencies")


def test_get_allowed_edges_for_phase_builds_incrementally():
    """get_allowed_edges_for_phase builds strictly additive edge sets.

    P(i) ⊆ P(i+1) for all phases — later phases add edges, never remove.
    """
    phases = sorted(PHASE_EDGES.keys(), key=lambda k: int(k[1:]))

    for i in range(len(phases) - 1):
        earlier = get_allowed_edges_for_phase(phases[i])
        later = get_allowed_edges_for_phase(phases[i + 1])

        # Every module in the earlier phase must exist in the later phase
        for mod in earlier:
            assert mod in later, \
                f"Module {mod} present in {phases[i]} but missing in {phases[i+1]}"

        # Every allowed target in the earlier phase must also be allowed later
        for mod, targets in earlier.items():
            for tgt in targets:
                assert tgt in later.get(mod, set()), \
                    f"Edge {mod}→{tgt} allowed in {phases[i]} but forbidden in {phases[i+1]}"

    print(f"✓ All {len(phases)} phases are strictly additive (no edge removals)")


def test_get_allowed_edges_for_phase_p0_strict():
    """P0 edges are the strict minimum — no extra modules beyond core + physics/boid.

    If a module not in P0 PHASE_EDGES appears, the phase-gating is broken.
    """
    p0_edges = get_allowed_edges_for_phase("P0")
    p0_phase_def = PHASE_EDGES.get("P0", {})
    assert isinstance(p0_phase_def, dict), f"P0 must be a dict, got {type(p0_phase_def)}"

    # get_allowed_edges_for_phase("P0") must return exactly the P0 phase definition
    assert set(p0_edges.keys()) == set(p0_phase_def.keys()), \
        f"P0 edges have unexpected modules: {set(p0_edges.keys()) ^ set(p0_phase_def.keys())}"

    # Each module's allowed targets must match exactly
    for mod in p0_phase_def:
        assert p0_edges[mod] == p0_phase_def[mod], \
            f"P0 edge mismatch for {mod}: expected {p0_phase_def[mod]}, got {p0_edges[mod]}"

    print("✓ P0 edges match PHASE_EDGES[\"P0\"] exactly — strict minimum")


def test_all_imports_within_allowed_edges():
    """Every pymurmur .py file's imports must be in ALLOWED_EDGES."""
    edges = _collect_import_edges()

    violations: list[str] = []
    known: list[str] = []

    for source, target, lineno, in_tc in edges:
        if _module_is_allowed(source, target, in_tc):
            continue
        tc_note = " (TYPE_CHECKING)" if in_tc else ""
        detail = f"  {source}:{lineno} → {target}{tc_note}"
        if _is_known_violation(source, target):
            known.append(detail)
        else:
            violations.append(detail)

    if known:
        print(f"\n⚠️  {len(known)} known violation(s) (scheduled for future phases):")
        for k in known:
            print(k)

    if violations:
        msg = (
            f"\n❌ {len(violations)} architecture DAG violation(s):\n"
            + "\n".join(violations)
            + "\n\nThese imports are not in ALLOWED_EDGES. "
            + "If this is a deliberate new edge, add it to ALLOWED_EDGES "
            + "and update the phase acceptance boundary.\n"
        )
        raise AssertionError(msg)

    print(f"✓ {len(edges)} import edges checked — all within ALLOWED_EDGES")
    if known:
        print(f"  ({len(known)} known violations deferred)")


def test_forbidden_edges_not_present():
    """No import edge (excluding TYPE_CHECKING) matches a FORBIDDEN_EDGES pair."""
    edges = _collect_import_edges()

    failures: list[str] = []
    for source, target, lineno, in_tc in edges:
        if in_tc:
            continue
        for f_src, f_tgt in FORBIDDEN_EDGES:
            if (source == f_src or source.startswith(f_src + ".")) and \
               (target == f_tgt or target.startswith(f_tgt + ".")):
                if not _is_known_violation(source, target):
                    failures.append(
                        f"  {source}:{lineno} → {target}  "
                        f"(violates FORBIDDEN: {f_src} !→ {f_tgt})"
                    )

    if failures:
        msg = (
            f"\n❌ {len(failures)} forbidden import edge(s) detected at runtime:\n"
            + "\n".join(failures)
            + "\n\nThese edges are permanently forbidden per arch.md §5.\n"
        )
        raise AssertionError(msg)

    print("✓ No forbidden import edges found at runtime")


def test_no_module_level_numpy_random():
    """No .py file under pymurmur/ has top-level np.random.* calls."""
    pattern = re.compile(r"^\s{0,3}np\.random\.")
    failures: list[str] = []

    for py_file in sorted(Path("pymurmur").rglob("*.py")):
        lines = py_file.read_text().split("\n")
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("import "):
                continue
            if stripped.startswith("from "):
                continue
            if pattern.search(line):
                failures.append(f"  {py_file}:{lineno}: {line.strip()}")

    if failures:
        msg = (
            f"\n❌ {len(failures)} top-level np.random.* call(s) found:\n"
            + "\n".join(failures)
            + "\n\nAll randomness must go through flock.rng. "
            + "Top-level np.random.* breaks determinism (P0.4).\n"
        )
        raise AssertionError(msg)

    print("✓ No top-level np.random.* calls found")


def test_no_cKDTree_in_forces():
    """No cKDTree construction in physics/forces/."""
    known = {
        "pymurmur/physics/forces/spatial.py",
        "pymurmur/physics/forces/vicsek.py",
        "pymurmur/physics/forces/angle.py",
    }
    failures: list[str] = []

    for py_file in Path("pymurmur/physics/forces").rglob("*.py"):
        rel = str(py_file)
        text = py_file.read_text()
        if "cKDTree(" in text or "KDTree(" in text:
            if rel in known:
                print(f"  ⚠️  Known: {rel} builds private cKDTree (P2.3)")
                continue
            failures.append(f"  {py_file}: contains cKDTree/KDTree construction")

    if failures:
        msg = (
            "\n❌ cKDTree construction found in forces/:\n"
            + "\n".join(failures)
            + "\n\nSpatial index construction belongs in physics/flock.py "
            + "(P0.2). Forces modules must use flock.index.\n"
        )
        raise AssertionError(msg)

    print("✓ No cKDTree constructions in physics/forces/")


def test_viz_no_simulation_import():
    """viz/ modules must not import from simulation/ at runtime."""
    edges = _collect_import_edges()

    failures: list[str] = []
    for source, target, lineno, in_tc in edges:
        if not source.startswith("pymurmur.viz."):
            continue
        if not target.startswith("pymurmur.simulation"):
            continue
        if in_tc:
            continue
        if _is_known_violation(source, target):
            continue
        failures.append(f"  {source}:{lineno} → {target}")

    if failures:
        msg = (
            f"\n❌ {len(failures)} viz → simulation runtime import(s) found:\n"
            + "\n".join(failures)
            + "\n\nviz/ modules must not import simulation/ modules at runtime.\n"
        )
        raise AssertionError(msg)

    print("✓ No runtime viz → simulation imports found")


def _scan_print_violations(root: Path, exempt: set[str]) -> list[str]:
    """G3: Scan *root* for `print(` calls, skipping comments/blanks and
    any file whose path (relative to cwd) is in *exempt*.  Extracted
    from the test body so the scan logic itself is unit-testable
    against a synthetic tree, not just the real `pymurmur/` package."""
    pattern = re.compile(r"print\(")
    failures: list[str] = []

    for py_file in sorted(root.rglob("*.py")):
        rel = str(py_file)
        if rel in exempt:
            continue
        text = py_file.read_text()
        for lineno, line in enumerate(text.split("\n"), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if pattern.search(stripped):
                failures.append(f"  {rel}:{lineno}: {stripped}")

    return failures


def test_no_print_in_package_sources():
    """G3: No `print(` calls in pymurmur/ package sources (except
    __main__.py, logging, and cli-helper modules which legitimately
    use print for CLI output)."""
    # Files that are legitimately allowed to use print()
    exempt = {
        "pymurmur/__main__.py",
        "pymurmur/core/logging.py",
    }
    failures = _scan_print_violations(Path("pymurmur"), exempt)

    if failures:
        msg = (
            f"\n❌ {len(failures)} print() call(s) in package sources:\n"
            + "\n".join(failures)
            + "\n\nprint() should be replaced with proper logging "
            + "(core/logging.py) or be exempted in the test.\n"
        )
        raise AssertionError(msg)

    print("✓ No unexpected print() calls in package sources")


def test_print_scan_catches_real_violation(tmp_path):
    """G3: The scan mechanism itself catches a real `print(` call in a
    synthetic file — proving the check isn't vacuous (i.e. it wouldn't
    just as happily pass if `pymurmur/` genuinely had one)."""
    (tmp_path / "clean.py").write_text("x = 1\n# print('commented out, ignored')\n")
    (tmp_path / "dirty.py").write_text("def f():\n    print('oops')\n")

    failures = _scan_print_violations(tmp_path, exempt=set())

    assert len(failures) == 1, f"Expected exactly 1 violation, got: {failures}"
    assert "dirty.py" in failures[0] and "print('oops')" in failures[0]


def test_print_scan_respects_exempt_set(tmp_path):
    """G3: A file explicitly listed in the exempt set is not flagged,
    matching how __main__.py/core/logging.py are exempted for real."""
    dirty = tmp_path / "dirty.py"
    dirty.write_text("print('cli output')\n")

    failures = _scan_print_violations(tmp_path, exempt={str(dirty)})
    assert failures == []
