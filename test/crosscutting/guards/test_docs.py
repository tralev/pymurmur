"""P14.4 — Doc-drift guard: every intra-repo markdown link resolves.

Verifies that:
1. Every markdown link in arch.md points to an existing file or directory.
2. arch.md does not contain stale references to the retired D0-D9/S1-S7/T0-T6 scheme.
3. G4: arch.md references the guard-rail job topology in test.md §CI.
4. G2: ALL .md files in the repo (root, sci/, TODO/) have resolving intra-repo links.

`roadmap_deepseek.md` (the P0-P14 implementation roadmap) was completed
and removed from the working tree 2026-07-21 — its history lives in git,
not as a live file this guard needs to cross-check against.  `docker.md`
was merged into `test.md` §Continuous Integration & Docker the same day.
"""

import re
from pathlib import Path

import pytest


def _extract_intra_links(filepath: str) -> list[tuple[str, str]]:
    """Parse a markdown file and return [(link_text, link_url), ...]
    for all non-HTTP links."""
    content = Path(filepath).read_text()
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    links = []
    for text, url in pattern.findall(content):
        if not url.startswith("http"):
            links.append((text, url))
    return links


def _resolve_link(md_file: str, url: str) -> Path:
    """Resolve a relative markdown link against the directory of md_file."""
    base = Path(md_file).parent
    # Remove anchor fragments (#section)
    url = url.split("#")[0]
    return (base / url).resolve()


# ── arch.md links ──────────────────────────────────────────────────





pytestmark = pytest.mark.guard

def test_arch_md_links_resolve():
    """Every intra-repo link in arch.md points to an existing target."""
    links = _extract_intra_links("arch.md")
    assert len(links) > 0, "arch.md has no intra-repo links"
    for text, url in links:
        target = _resolve_link("arch.md", url)
        assert target.exists(), (
            f"arch.md link [{text}]({url}) → {target} does not exist"
        )


def test_arch_md_no_stale_scheme_references():
    """arch.md does not contain stale references to the retired
    D0-D9/S1-S7/T0-T6 scheme (superseded by the P0-P14 scheme, whose
    tracking roadmap is now completed and removed from the tree)."""
    content = Path("arch.md").read_text()
    assert "D0–D9" not in content, "arch.md still references retired D0-D9"
    assert "S1–S7" not in content, "arch.md still references retired S1-S7"
    assert "T0–T6" not in content, "arch.md still references retired T0-T6"


# ── G2: All markdown file links resolve ──────────────────────────


def test_g2_all_markdown_links_resolve():
    """G2: Every intra-repo link in all repo .md files resolves to an
    existing target.  Recursively scans .md files in the repo root and
    subdirectories (sci/, TODO/)."""
    md_files = sorted(Path(".").rglob("*.md"))
    assert len(md_files) >= 4, (
        f"Expected at least 4 .md files, found {len(md_files)}"
    )

    failures: list[str] = []
    total_links = 0
    for md in md_files:
        links = _extract_intra_links(str(md))
        total_links += len(links)
        for text, url in links:
            target = _resolve_link(str(md), url)
            if not target.exists():
                failures.append(
                    f"  {md}: [{text}]({url}) → {target} does not exist"
                )

    if failures:
        msg = (
            f"\n❌ {len(failures)} broken markdown link(s) found across "
            f"{len(md_files)} files:\n" + "\n".join(failures) +
            "\n\nG2 requires every intra-repo markdown link to resolve.\n"
        )
        raise AssertionError(msg)

    print(f"✓ {total_links} links across {len(md_files)} .md files — all resolve")


def test_g2_broken_link_is_actually_detected(tmp_path):
    """G2: The link-checking mechanism itself catches a real broken
    link — not just "the current repo happens to have none."  Uses a
    synthetic scratch .md file so this doesn't depend on the real repo
    ever having (or not having) a dangling link."""
    md = tmp_path / "scratch.md"
    md.write_text("See [missing file](does_not_exist_xyz.md) for details.")

    links = _extract_intra_links(str(md))
    assert links == [("missing file", "does_not_exist_xyz.md")]

    target = _resolve_link(str(md), links[0][1])
    assert not target.exists(), "Scratch link target must not exist for this test to be meaningful"


def test_g2_resolve_link_strips_anchor_fragment(tmp_path):
    """G2: A link with a `#section` anchor resolves against the file
    itself, ignoring the fragment (anchors aren't real filesystem
    paths)."""
    md = tmp_path / "scratch.md"
    target_file = tmp_path / "other.md"
    target_file.write_text("# Other\n")
    md.write_text("See [other](other.md#some-section) for details.")

    links = _extract_intra_links(str(md))
    assert links == [("other", "other.md#some-section")]

    resolved = _resolve_link(str(md), links[0][1])
    assert resolved == target_file.resolve()
    assert resolved.exists()


def test_g2_extract_intra_links_skips_http_links(tmp_path):
    """G2: External http(s) links are excluded from the intra-repo
    link check — they're not subject to filesystem resolution."""
    md = tmp_path / "scratch.md"
    md.write_text(
        "[external](https://example.com/page) and "
        "[internal](arch.md) and "
        "[secure external](http://example.com)."
    )
    links = _extract_intra_links(str(md))
    assert links == [("internal", "arch.md")], (
        f"http(s) links must be excluded, got: {links}"
    )


# ── G2: Force-mode table stays pointed at the canonical file ─────
#
# arch.md §6 is the canonical force-mode table (roadmap_deepseek.md only
# mentions the 7 modes in prose).  Single-sourcing this check to arch.md
# avoids two docs independently drifting out of sync with MODE_REGISTRY.

FORCE_MODE_TABLE_FILE = "arch.md"


def _extract_force_mode_table_names(md_file: str = FORCE_MODE_TABLE_FILE) -> list[str]:
    """Parse the '## 6. Force Modes' table and return its bolded mode
    names (`| **name** | ... |` rows) in table order."""
    lines = Path(md_file).read_text().splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("## 6. Force Modes")),
        None,
    )
    assert start is not None, f"{md_file} has no '## 6. Force Modes' section"
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## 7.")),
        len(lines),
    )
    names = []
    for line in lines[start:end]:
        m = re.match(r"\|\s*\*\*(\w+)\*\*\s*\|", line)
        if m:
            names.append(m.group(1))
    return names


def test_g2_force_mode_table_matches_registry():
    """G2: arch.md §6's Force Modes table lists exactly the modes
    registered in MODE_REGISTRY — no more, no less.  Adding or removing
    a `@register`-ed mode without updating the table fails this guard."""
    from pymurmur.physics.forces import MODE_REGISTRY

    table_modes = _extract_force_mode_table_names()
    assert table_modes, f"No force-mode table rows found in {FORCE_MODE_TABLE_FILE} §6"

    registry_modes = sorted(MODE_REGISTRY)
    assert sorted(table_modes) == registry_modes, (
        f"{FORCE_MODE_TABLE_FILE} §6 Force Modes table lists "
        f"{sorted(table_modes)} but MODE_REGISTRY has {registry_modes} — "
        f"the table has drifted from the registry.  Add/remove a row when "
        f"registering/removing a mode."
    )


def test_g2_force_mode_registry_prose_matches():
    """G2: arch.md §6's 'sorted(MODE_REGISTRY) (= ...)' disambiguating
    prose line matches the actual registry — catches drift the table-row
    check wouldn't (e.g. a mode added to the table but not this line, or
    vice versa)."""
    from pymurmur.physics.forces import MODE_REGISTRY

    content = Path(FORCE_MODE_TABLE_FILE).read_text()
    m = re.search(r"sorted\(MODE_REGISTRY\)`\s*\(=\s*([^)]+)\)", content)
    assert m, (
        f"{FORCE_MODE_TABLE_FILE} §6 missing the "
        f"'sorted(MODE_REGISTRY) (= ...)' prose line"
    )
    listed = [name.strip() for name in m.group(1).replace("\n", " ").split(",")]
    assert listed == sorted(MODE_REGISTRY), (
        f"{FORCE_MODE_TABLE_FILE} §6 prose lists {listed} but "
        f"sorted(MODE_REGISTRY) is {sorted(MODE_REGISTRY)}"
    )


# ── G4: CI guard topology documentation ──────────────────────────

GUARD_RAIL_JOB_NAMES = [
    "guard-rail-dag",
    "guard-rail-golden",
    "guard-rail-config-drift",
    "guard-rail-3d",
    "guard-rail-doc-links",
    "guard-rail-collection-count",
    "guard-rail-mypy",
    "guard-rail-evolved",
    "guard-rail-composers",
    "guard-rails-summary",
]


def test_arch_md_references_guard_rail_topology():
    """G4: arch.md §5 references the guard-rail job list in test.md."""
    content = Path("arch.md").read_text()
    assert "test.md" in content, (
        "arch.md must reference test.md for guard-rail topology"
    )
    # The anchor or section reference must exist
    assert "[test.md](test.md)" in content, (
        "arch.md §5 must link to test.md (guard-rail job list)"
    )


def test_arch_md_paragraph_names_all_guard_rail_jobs():
    """G4: arch.md §5's guard-rail paragraph names every job (except the
    summary gate, referenced there by role rather than literal job id).

    Catches the paragraph's job count/list going stale when a job is
    added or removed — this bit the repo directly: adding
    `guard-rail-composers` (G1) left this paragraph saying "nine jobs"
    with composers missing from the list.
    """
    content = Path("arch.md").read_text()
    for job_name in GUARD_RAIL_JOB_NAMES:
        if job_name == "guard-rails-summary":
            continue  # referenced as "a merge-blocking ... gate", not by job id
        assert job_name in content, (
            f"arch.md §5 missing guard-rail job '{job_name}' — the "
            f"paragraph has drifted from GUARD_RAIL_JOB_NAMES."
        )


def _load_workflow_job_names(path: Path) -> set[str]:
    """G4: Parse a GitHub Actions workflow file and return its top-level
    job names.  Extracted so the comparison logic is unit-testable
    against a synthetic workflow file, not just the real one."""
    import yaml

    workflow = yaml.safe_load(path.read_text())
    return set(workflow["jobs"].keys())


def test_guard_rail_job_names_matches_workflow_file():
    """G4: GUARD_RAIL_JOB_NAMES (this file's ground truth for the docs
    checks above) matches the actual job list in
    `.github/workflows/guard-rails.yml`.

    Without this, GUARD_RAIL_JOB_NAMES is just a second hand-maintained
    list that could itself drift from CI — a job renamed/added/removed
    in the workflow file would go unnoticed by every check above, since
    they all compare docs against GUARD_RAIL_JOB_NAMES, never against
    the workflow file itself.
    """
    actual_jobs = _load_workflow_job_names(
        Path(".github/workflows/guard-rails.yml")
    )
    expected_jobs = set(GUARD_RAIL_JOB_NAMES)

    assert actual_jobs == expected_jobs, (
        f"GUARD_RAIL_JOB_NAMES has drifted from guard-rails.yml's actual "
        f"jobs.\n  In workflow but not in GUARD_RAIL_JOB_NAMES: "
        f"{sorted(actual_jobs - expected_jobs)}\n"
        f"  In GUARD_RAIL_JOB_NAMES but not in workflow: "
        f"{sorted(expected_jobs - actual_jobs)}"
    )


def test_workflow_job_mismatch_is_actually_detected(tmp_path):
    """G4: The comparison mechanism itself catches a real mismatch —
    not just "GUARD_RAIL_JOB_NAMES and the real file happen to agree
    today."  Uses a synthetic scratch workflow file with a deliberately
    different job set."""
    scratch = tmp_path / "scratch-workflow.yml"
    scratch.write_text(
        "jobs:\n"
        "  guard-rail-dag: {}\n"
        "  guard-rail-a-job-nobody-expects: {}\n"
    )

    actual_jobs = _load_workflow_job_names(scratch)
    expected_jobs = set(GUARD_RAIL_JOB_NAMES)

    assert actual_jobs != expected_jobs, (
        "Scratch workflow must differ from GUARD_RAIL_JOB_NAMES for this "
        "test to be meaningful"
    )
    assert "guard-rail-a-job-nobody-expects" in (actual_jobs - expected_jobs)


def test_test_md_contains_all_guard_jobs():
    """G4: test.md §Continuous Integration & Docker lists all guard-rail
    job names.

    A missing job name here means the guard-rail topology documentation
    has rotted — the list is the canonical enumeration of the
    merge-blocking P14 guard set.
    """
    content = Path("test.md").read_text()
    for job_name in GUARD_RAIL_JOB_NAMES:
        assert job_name in content, (
            f"test.md missing guard-rail job '{job_name}' — "
            f"topology doc has rotted.  Add it to the guard-rail job "
            f"table or remove it from GUARD_RAIL_JOB_NAMES."
        )
