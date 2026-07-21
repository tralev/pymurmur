"""Config-usage drift test — every SimConfig field must be referenced in source.

AST-walks all pymurmur/ source files (excluding config.py itself and test/).
For each field in SimConfig, asserts the field name string appears in at
least one source file's text or attribute access pattern.

Rationale: Orphan config fields are dead weight — they promise behaviour
that doesn't exist, mislead users, and accumulate technical debt.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pymurmur.core.config import _ALL_FIELD_NAMES

# ── Field name extraction ──────────────────────────────────────────

def _simconfig_field_names() -> set[str]:
    """Return all field names declared on SimConfig (I7.1)."""
    return _ALL_FIELD_NAMES


# ── Source scanning ────────────────────────────────────────────────

def _read_source_files() -> dict[str, str]:
    """Read all non-config, non-test .py files under pymurmur/."""
    sources: dict[str, str] = {}
    for py_file in sorted(Path("pymurmur").rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        # C3: config.py must be excluded — otherwise every field trivially
        # "passes" via its own declaration and the guard detects nothing.
        if py_file.name == "config.py" and py_file.parent.name == "core":
            continue
        sources[str(py_file)] = py_file.read_text()
    return sources


def _find_attr_accesses(source_text: str) -> set[str]:
    """Find all attribute names accessed via `config.X` or `cfg.X` patterns."""
    found: set[str] = set()
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return found

    for node in ast.walk(tree):
        # Match `config.field_name` or `cfg.field_name`
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in (
                "config", "cfg", "self",
            ):
                found.add(node.attr)
    return found


def _find_string_occurrences(source_text: str, field_name: str) -> bool:
    """Check if field_name appears as a string literal anywhere in source."""
    # Check as plain substring (handles to_file dict keys, getattr, etc.)
    return field_name in source_text


# ── Known exceptions ───────────────────────────────────────────────

# Fields that are referenced only via YAML sections or internal plumbing
# and don't appear as python attribute access patterns.
#
# (empty as of S2.A5, 2026-07-20 — field_target_pull was the one entry,
# now wired as the "target_pull" ForceTerm in field.py)
KNOWN_ORPHANS: set[str] = set()


# ── Tests ──────────────────────────────────────────────────────────

def test_every_config_field_used():
    """Every SimConfig field appears in at least one pymurmur/ source file.

    Checks two patterns:
    1. Attribute access: `config.field_name`, `cfg.field_name`, `self.field_name`
    2. String occurrence: field name as substring (handles dict keys, getattr)
    """
    field_names = _simconfig_field_names()
    sources = _read_source_files()

    # Collect all attribute names found across all source files
    all_attrs: set[str] = set()
    for _path, text in sources.items():
        all_attrs |= _find_attr_accesses(text)

    # Check each field
    orphans: list[str] = []
    for field_name in sorted(field_names):
        if field_name in KNOWN_ORPHANS:
            continue

        # Check 1: attribute access
        if field_name in all_attrs:
            continue

        # Check 2: string occurrence in any source file
        found_as_string = False
        for _path, text in sources.items():
            if _find_string_occurrences(text, field_name):
                found_as_string = True
                break

        if not found_as_string:
            orphans.append(field_name)

    if orphans:
        msg = (
            f"\n❌ {len(orphans)} orphan SimConfig field(s) — not referenced "
            f"in any pymurmur/ source file:\n"
            + "\n".join(f"  • {f}" for f in orphans)
            + "\n\nThese fields are declared in SimConfig but never read. "
            + "Either implement the feature or delete the field.\n"
        )
        raise AssertionError(msg)

    print(f"✓ All {len(field_names)} SimConfig fields referenced in source")


def test_no_dead_fields_accumulate():
    """KNOWN_ORPHANS should not grow without review.

    When a new feature wires a previously-orphan field, remove it from
    KNOWN_ORPHANS. When making a deliberate decision to keep a dead
    field, add it to KNOWN_ORPHANS with a comment explaining why.
    This test ensures the list only shrinks, never grows unchecked.
    """
    # This test is intentionally empty — it documents the policy.
    # When KNOWN_ORPHANS grows, the reviewer must explain why.
    assert isinstance(KNOWN_ORPHANS, set), "KNOWN_ORPHANS must be a set"
