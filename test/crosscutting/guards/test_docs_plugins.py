"""G2 — Plugin-taxonomy doc-drift guards: arch.md's "Other Computational
Plugins" and "Behavioural Extensions" tables stay in sync with the live
registries.

Split out of test_docs.py (file-size split) — mirrors the Force-Mode-
table-vs-MODE_REGISTRY check that stays in the original: parse the
doc, compare against live ground truth.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

FORCE_MODE_TABLE_FILE = "arch.md"

# ── G2: "Other Computational Plugins" table stays in sync ────────
#
# arch.md §6's "### Other Computational Plugins" subsection catalogues
# the 7 computational-strategy registries other than ForceMode (which
# the table above already covers). Mirrors the Force-Mode-table check:
# parse the doc, compare against live registries.

_COMPUTATIONAL_PLUGIN_FAMILIES = {
    "ForceMode",
    "BoundaryMode",
    "NeighborSelector",
    "ObstacleAvoidanceStrategy",
    "SpeedModel",
    "SpatialIndexStrategy",
    "NoiseStrategy",
}  # "Kernel registries" row is a 3-way family, checked separately below


def _extract_other_computational_plugins_table(md_file: str = FORCE_MODE_TABLE_FILE) -> dict[str, str]:
    """Parse '### Other Computational Plugins' and return
    {bolded family name: full row text} in table order."""
    lines = Path(md_file).read_text().splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if line.startswith("### Other Computational Plugins")),
        None,
    )
    assert start is not None, (
        f"{md_file} has no '### Other Computational Plugins' subsection"
    )
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## 7.")),
        len(lines),
    )
    rows: dict[str, str] = {}
    for line in lines[start:end]:
        m = re.match(r"\|\s*\*\*([\w ]+)\*\*\s*\|", line)
        if m:
            rows[m.group(1)] = line
    return rows


def test_g2_other_computational_plugins_table_matches_registries():
    """G2: arch.md's "Other Computational Plugins" table lists exactly
    the expected plugin families, and each row's Entries column mentions
    every key in that family's live registry — catches a strategy being
    added/removed/renamed without updating the table."""
    from pymurmur.physics.plugins.boundary import BOUNDARY_REGISTRY
    from pymurmur.physics.plugins.kernel_registry import (
        ALIGNMENT_KERNEL_REGISTRY,
        COHESION_KERNEL_REGISTRY,
        SEPARATION_KERNEL_REGISTRY,
    )
    from pymurmur.physics.plugins.neighbor_selection import NEIGHBOR_SELECTOR_REGISTRY
    from pymurmur.physics.plugins.noise_strategy import NOISE_STRATEGY_REGISTRY
    from pymurmur.physics.plugins.obstacle_avoidance import OBSTACLE_AVOIDANCE_REGISTRY
    from pymurmur.physics.plugins.spatial_index_strategy import (
        SPATIAL_INDEX_STRATEGY_REGISTRY,
    )
    from pymurmur.physics.plugins.speed_model import SPEED_MODEL_REGISTRY

    rows = _extract_other_computational_plugins_table()
    table_families = set(rows) - {"Plugin family"}  # exclude header row

    assert table_families == _COMPUTATIONAL_PLUGIN_FAMILIES | {"Kernel registries"}, (
        f"{FORCE_MODE_TABLE_FILE} 'Other Computational Plugins' table lists "
        f"{sorted(table_families)} but expected "
        f"{sorted(_COMPUTATIONAL_PLUGIN_FAMILIES | {'Kernel registries'})} — "
        f"the table has drifted."
    )

    registries_by_family = {
        "BoundaryMode": BOUNDARY_REGISTRY,
        "NeighborSelector": NEIGHBOR_SELECTOR_REGISTRY,
        "ObstacleAvoidanceStrategy": OBSTACLE_AVOIDANCE_REGISTRY,
        "SpeedModel": SPEED_MODEL_REGISTRY,
        "SpatialIndexStrategy": SPATIAL_INDEX_STRATEGY_REGISTRY,
        "NoiseStrategy": NOISE_STRATEGY_REGISTRY,
    }
    for family, registry in registries_by_family.items():
        row_text = rows[family]
        missing = [key for key in registry if key not in row_text]
        assert not missing, (
            f"{FORCE_MODE_TABLE_FILE} 'Other Computational Plugins' row for "
            f"{family} is missing entries {missing} (registry has "
            f"{sorted(registry)}) — row text: {row_text!r}"
        )

    # "Kernel registries" row: verify the 3 documented counts match live.
    kernel_row = rows["Kernel registries"]
    for label, registry in (
        ("SEPARATION_KERNEL_REGISTRY", SEPARATION_KERNEL_REGISTRY),
        ("ALIGNMENT_KERNEL_REGISTRY", ALIGNMENT_KERNEL_REGISTRY),
        ("COHESION_KERNEL_REGISTRY", COHESION_KERNEL_REGISTRY),
    ):
        expected = f"`{label}` ({len(registry)})"
        assert expected in kernel_row, (
            f"{FORCE_MODE_TABLE_FILE} 'Kernel registries' row missing "
            f"'{expected}' (registry has {len(registry)} entries) — "
            f"row text: {kernel_row!r}"
        )


# ── G2: Behavioural Extensions table stays in sync ────────────────

_EXTENSION_DISPLAY_NAMES = {
    "Predator": "Threat",  # arch.md §7 names the Predator extension "Threat"
}


def test_g2_behavioural_extensions_table_matches_registry():
    """G2: arch.md §7's Behavioural Extensions table lists exactly the
    extensions registered in EXTENSION_REGISTRY — no more, no less.
    This table drifted silently before (4 of 8 extensions undocumented)
    because nothing checked it; this guard prevents a repeat."""
    from pymurmur.physics.extensions.extension_registry import EXTENSION_REGISTRY

    lines = Path(FORCE_MODE_TABLE_FILE).read_text().splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if line.startswith("## 7. Behavioural Extensions")),
        None,
    )
    assert start is not None, (
        f"{FORCE_MODE_TABLE_FILE} has no '## 7. Behavioural Extensions' section"
    )
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## 8.")),
        len(lines),
    )
    table_names = set()
    for line in lines[start:end]:
        m = re.match(r"\|\s*\*\*(\w+)\*\*\s*\|", line)
        if m:
            table_names.add(m.group(1))

    registry_names = {
        _EXTENSION_DISPLAY_NAMES.get(cls.__name__, cls.__name__)
        for cls, _config_attr, _cleanup_attr in EXTENSION_REGISTRY
    }
    assert table_names == registry_names, (
        f"{FORCE_MODE_TABLE_FILE} §7 Behavioural Extensions table lists "
        f"{sorted(table_names)} but EXTENSION_REGISTRY has "
        f"{sorted(registry_names)} — the table has drifted from the "
        f"registry. Add/remove a row when registering/removing an extension."
    )

