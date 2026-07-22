"""EvoFlock gene-usage drift test — every EVOLVABLE_PARAMS gene must be
consumed by physics or the evaluation harness itself.

Mirrors test_config_drift.py's pattern (AST attribute-access scan +
string-occurrence fallback), scoped to pymurmur/physics/ and
analysis/evoflock.py's own consumers rather than all of pymurmur/ —
EvoFlock genes are read via setattr(cfg, name, value) + config.<name>
attribute access in physics/forces modules, or directly inside
_ObjectiveCollector/_evaluate_single in evoflock.py itself.

Rationale (S6.5): a gene that GA search explores but that no physics
reader consumes wastes a search dimension silently — the exact
`predictive_avoid_weight`/`static_avoid_weight` dead-gene bug this
guard exists to catch permanently.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pymurmur.analysis.evoflock import EVOLVABLE_PARAMS

# ── Source scanning ────────────────────────────────────────────────

def _read_source_files() -> dict[str, str]:
    """Read all pymurmur/physics/*.py files plus analysis/evoflock.py
    itself (the genome->config->physics chain lives across both)."""
    sources: dict[str, str] = {}
    for py_file in sorted(Path("pymurmur/physics").rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        sources[str(py_file)] = py_file.read_text()
    evoflock_path = Path("pymurmur/analysis/evoflock.py")
    sources[str(evoflock_path)] = evoflock_path.read_text()
    return sources


def _find_attr_accesses(source_text: str) -> set[str]:
    """Find attribute names accessed via `config.X`/`cfg.X`/`self.X`."""
    found: set[str] = set()
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in (
                "config", "cfg", "self",
            ):
                found.add(node.attr)
    return found


# ── Known exceptions ───────────────────────────────────────────────

# Genes intentionally consumed only inside evoflock.py's own evaluation
# harness (not read by a physics/ module directly) — none as of S6.5.
KNOWN_HARNESS_ONLY: set[str] = set()


def test_every_evolvable_gene_used():
    """Every EVOLVABLE_PARAMS key is referenced in physics/ or
    analysis/evoflock.py — an unreferenced gene is a dead GA dimension.
    """
    sources = _read_source_files()

    all_attrs: set[str] = set()
    for _path, text in sources.items():
        all_attrs |= _find_attr_accesses(text)

    orphans: list[str] = []
    for gene_name in sorted(EVOLVABLE_PARAMS):
        if gene_name in KNOWN_HARNESS_ONLY:
            continue
        if gene_name in all_attrs:
            continue
        found_as_string = any(gene_name in text for text in sources.values())
        if not found_as_string:
            orphans.append(gene_name)

    if orphans:
        msg = (
            f"\n❌ {len(orphans)} orphan EvoFlock gene(s) — not referenced "
            f"in pymurmur/physics/ or analysis/evoflock.py:\n"
            + "\n".join(f"  • {g}" for g in orphans)
            + "\n\nThese genes are declared in EVOLVABLE_PARAMS but never "
            + "read — the GA wastes a search dimension on them. Either wire "
            + "a physics reader or delete the gene.\n"
        )
        raise AssertionError(msg)
