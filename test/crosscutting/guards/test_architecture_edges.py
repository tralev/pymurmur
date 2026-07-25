"""P14.1 — Architecture DAG matrix enforcement.

AST-walks every .py file under pymurmur/, resolves relative imports
to absolute module paths, and asserts every import edge is within
ALLOWED_EDGES and not in FORBIDDEN_EDGES.

Split out of test_architecture.py (file-size split) — the enforcement
logic + phase-gating tests; static data (ALLOWED_EDGES/FORBIDDEN_EDGES/
KNOWN_VIOLATIONS/PHASE_EDGES) lives in test_architecture_edges_data.py.
test_architecture_scans.py imports _collect_import_edges/
_is_known_violation from this file for its own independent scans.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from test.crosscutting.guards.test_architecture_edges_data import (
    ALLOWED_EDGES,
    FORBIDDEN_EDGES,
    KNOWN_VIOLATIONS,
    PHASE_EDGES,
    PHASE_VIOLATION_REMOVALS,
    STDLIB,
    THIRD_PARTY,
)

pytestmark = pytest.mark.guard


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


