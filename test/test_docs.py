"""P14.4 — Doc-drift guard: every intra-repo markdown link resolves.

Verifies that:
1. Every markdown link in arch.md points to an existing file or directory.
2. Every markdown link in roadmap_deepseek.md points to an existing file or directory.
3. arch.md references roadmap_deepseek.md with the P0-P14 phase scheme.
4. roadmap_deepseek.md references arch.md back (bidirectional sync).
5. arch.md does not contain stale references to the retired D0-D9/S1-S7/T0-T6 scheme.
"""

import re
import pytest
from pathlib import Path


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


def test_arch_md_references_roadmap():
    """arch.md links to roadmap_deepseek.md (the two docs stay in sync)."""
    content = Path("arch.md").read_text()
    # Should contain a markdown link to roadmap_deepseek.md
    assert "[`roadmap_deepseek.md`](roadmap_deepseek.md)" in content or \
           "[roadmap_deepseek.md](roadmap_deepseek.md)" in content, (
        "arch.md does not reference roadmap_deepseek.md"
    )
    # Should reference the P0-P14 phase scheme
    assert "P0" in content and "P14" in content, (
        "arch.md does not reference the P0-P14 phase numbering"
    )
    # Should NOT reference the retired D0-D9 / S1-S7 / T0-T6 scheme
    assert "D0–D9" not in content, "arch.md still references retired D0-D9"
    assert "S1–S7" not in content, "arch.md still references retired S1-S7"
    assert "T0–T6" not in content, "arch.md still references retired T0-T6"


# ── roadmap_deepseek.md links ──────────────────────────────────────


def test_roadmap_md_links_resolve():
    """Every intra-repo link in roadmap_deepseek.md points to an existing target."""
    links = _extract_intra_links("roadmap_deepseek.md")
    assert len(links) > 0, "roadmap_deepseek.md has no intra-repo links"
    for text, url in links:
        target = _resolve_link("roadmap_deepseek.md", url)
        assert target.exists(), (
            f"roadmap_deepseek.md link [{text}]({url}) → {target} does not exist"
        )


def test_roadmap_references_arch():
    """roadmap_deepseek.md links back to arch.md (bidirectional sync)."""
    content = Path("roadmap_deepseek.md").read_text()
    assert "[`arch.md`](arch.md)" in content or \
           "[arch.md](arch.md)" in content, (
        "roadmap_deepseek.md does not reference arch.md"
    )
