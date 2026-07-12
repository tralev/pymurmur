"""Verification of critical dependency rules — no upward imports."""

import ast
import importlib
from pathlib import Path


def _get_imports(module_path: Path) -> set[str]:
    """Extract imported module names from a Python file via AST."""
    tree = ast.parse(module_path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_physics_boid_no_forces_import():
    """physics/boid.py does not import physics.flock or physics.forces."""
    path = Path("pymurmur/physics/boid.py")
    imports = _get_imports(path)
    # Must not import from physics.flock or physics.forces
    forbidden = {
        "pymurmur.physics.flock", "pymurmur.physics.forces",
        "pymurmur.physics.forces._base", "pymurmur.physics.forces.spatial",
    }
    for f in forbidden:
        assert f not in imports, f"Found forbidden import '{f}' in boid.py"


def test_occlusion_no_project_imports():
    """occlusion.py imports only numpy + core.types."""
    path = Path("pymurmur/physics/occlusion.py")
    text = path.read_text()
    assert "from ..flock" not in text
    assert "from .forces" not in text
    assert "from ..forces" not in text
    assert "import PhysicsFlock" not in text


def test_steric_no_project_imports():
    """steric.py imports only numpy + core.types."""
    path = Path("pymurmur/physics/steric.py")
    text = path.read_text()
    assert "from ..flock" not in text
    assert "from .forces" not in text
    assert "from ..forces" not in text


def test_all_core_modules_importable():
    """All pymurmur modules import cleanly."""
    modules = [
        "pymurmur.core.types",
        "pymurmur.core.config",
        "pymurmur.physics.boid",
        "pymurmur.physics.flock",
        "pymurmur.physics.occlusion",
        "pymurmur.physics.steric",
        "pymurmur.physics.forces",
        "pymurmur.physics.forces._base",
        "pymurmur.physics.forces.spatial",
        "pymurmur.physics.forces.projection",
        "pymurmur.physics.forces.field",
        "pymurmur.physics.forces.vicsek",
        "pymurmur.physics.forces.influencer",
        "pymurmur.physics.extensions",
        "pymurmur.physics.extensions._base",
        "pymurmur.physics.extensions.predator",
        "pymurmur.physics.extensions.ecology",
        "pymurmur.physics.extensions.wander",
        "pymurmur.physics.extensions.ripple",
        "pymurmur.simulation.engine",
        "pymurmur.analysis.metrics",
        "pymurmur.analysis.presets",
        "pymurmur.analysis.evoflock",
    ]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Module {mod_name} is None"
        except Exception as e:
            raise AssertionError(f"Failed to import {mod_name}: {e}")


def test_viz_modules_importable():
    """viz/camera imports cleanly (no ModernGL needed)."""
    import pymurmur.viz.camera
    import pymurmur.viz.shaders
    assert pymurmur.viz.camera is not None
    assert pymurmur.viz.shaders is not None


def test_config_no_game_imports():
    """config.py does not import pygame/modernGL."""
    path = Path("pymurmur/core/config.py")
    text = path.read_text()
    assert "import pygame" not in text
    assert "import moderngl" not in text
