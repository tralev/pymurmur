"""G1 — Composer-enforcement guard: every L0 atom has at least one call site.

AST-scans the public L0 function surface (the atoms listed in arch.md §2.2)
and verifies each is called from at least one non-definition site in the
project.  Failures produce a dead-atom list so stale helpers cannot
accumulate silently.

Private underscore helpers (_*) are excluded — they're implementation
details of their parent module and are not part of the public L0 contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.guard

# ── L0 atom registry ──────────────────────────────────────────────
#
# (module_relative_path, function_name)
# These are the public L0 functions listed in arch.md §2.2 (Bottom-Up).
# Each must be called from at least one non-definition site in the project.

L0_ATOMS: list[tuple[str, str]] = [
    # core/types.py
    ("pymurmur/core/types.py", "safe_normalize"),
    ("pymurmur/core/types.py", "limit3"),
    ("pymurmur/core/types.py", "lerp"),
    ("pymurmur/core/types.py", "rotate_about"),
    ("pymurmur/core/types.py", "smoothstep"),
    ("pymurmur/core/types.py", "hash01"),
    ("pymurmur/core/types.py", "min_image"),
    ("pymurmur/core/types.py", "min_image_distance"),
    ("pymurmur/core/types.py", "fibonacci_sphere"),
    ("pymurmur/core/types.py", "seed_noise3"),
    # physics/forces/_base.py
    ("pymurmur/physics/forces/_base.py", "composeForces"),
    ("pymurmur/physics/forces/_base.py", "separation_force"),
    ("pymurmur/physics/forces/_base.py", "alignment_force"),
    ("pymurmur/physics/forces/_base.py", "cohesion_force"),
    ("pymurmur/physics/forces/_base.py", "curl_flow"),
    ("pymurmur/physics/forces/_base.py", "noise_force"),
    # physics/occlusion.py
    ("pymurmur/physics/occlusion.py", "spherical_cap_occlusion"),
    ("pymurmur/physics/occlusion.py", "spherical_cap_occlusion_batched"),
    ("pymurmur/physics/occlusion.py", "spherical_cap_occlusion_soa"),
    # physics/steric.py
    ("pymurmur/physics/steric.py", "steric_force"),
    # physics/boid.py
    ("pymurmur/physics/boid.py", "integrate"),
    ("pymurmur/physics/boid.py", "init_positions"),
    ("pymurmur/physics/boid.py", "init_velocities"),
    # core/config.py — public construction
    ("pymurmur/core/config.py", "SimConfig"),
]


def _find_call_sites(module_path: str, func_name: str) -> bool:
    """Walk pymurmur/ and test/ ASTs for calls to *func_name*.

    Skips the defining file, __init__.py re-exports, and hidden modules.
    Returns True if at least one genuine cross-file call site exists.
    """
    root = Path(".")
    defining_file = Path(module_path).resolve()

    for py_file in sorted(root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue  # skip all __init__ re-exports and implicit imports
        if py_file.name.startswith("_") and py_file.parent.name.startswith("_"):
            continue  # skip deeply private modules
        if py_file.resolve() == defining_file:
            continue  # skip the defining file — the guard is about cross-module composers

        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                caller = None
                if isinstance(node.func, ast.Name):
                    caller = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    caller = node.func.attr

                if caller == func_name:
                    return True

    return False


# ── Tests ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path, func_name", L0_ATOMS)
def test_l0_atom_has_call_site(module_path: str, func_name: str):
    """Every registered L0 atom has at least one composer call site."""
    assert _find_call_sites(module_path, func_name), (
        f"Dead L0 atom: {func_name} ({module_path}) has no call sites "
        f"outside its own definition.  Either add a composer or remove it "
        f"from L0_ATOMS."
    )


def test_L0_ATOMS_not_empty():
    """Sanity: the atom registry is populated."""
    assert len(L0_ATOMS) > 0, "L0_ATOMS must not be empty"


def test_adding_unused_helper_makes_guard_fail():
    """G1: Temporarily adding an unused atom makes the guard fail.

    Verify the guard mechanism works by testing that a non-existent
    function name in a real file produces a dead-atom result.
    """
    # A real file with a fake function name — must return False
    assert not _find_call_sites(
        "pymurmur/core/types.py", "nonexistent_helper_xyzzy"
    ), "Synthetic dead atom should have no call sites — guard would flag it"


def test_real_dead_function_is_detected():
    """G1: A genuinely-defined, never-called function is detected as
    dead — not just a made-up name (the check above), but a real
    function definition the AST scanner walks past and correctly finds
    no caller for.

    `_find_call_sites` walks `Path(".").rglob("*.py")`, so the probe
    files must live under the repo root (not pytest's `tmp_path`, which
    is outside it) to be discoverable — created and removed here.
    """
    probe_dir = Path("test/crosscutting/guards")
    defining = probe_dir / "_g1_probe_defining_tmp.py"
    try:
        defining.write_text(
            "def probe_dead_fn_never_called_xyz():\n    return 1\n"
        )
        assert not _find_call_sites(
            str(defining), "probe_dead_fn_never_called_xyz"
        ), "A real, genuinely-unused function must be detected as dead"
    finally:
        defining.unlink(missing_ok=True)


def test_real_used_function_is_not_flagged_dead():
    """G1: Complement of the above — a real function WITH a genuine
    cross-file call site is correctly recognized as alive, proving the
    detector isn't just biased toward always returning False."""
    probe_dir = Path("test/crosscutting/guards")
    defining = probe_dir / "_g1_probe_defining_tmp2.py"
    caller = probe_dir / "_g1_probe_caller_tmp2.py"
    try:
        defining.write_text(
            "def probe_alive_fn_xyz():\n    return 1\n"
        )
        caller.write_text(
            "from ._g1_probe_defining_tmp2 import probe_alive_fn_xyz\n"
            "probe_alive_fn_xyz()\n"
        )
        assert _find_call_sites(
            str(defining), "probe_alive_fn_xyz"
        ), "A function with a real cross-file call site must not be flagged dead"
    finally:
        defining.unlink(missing_ok=True)
        caller.unlink(missing_ok=True)
