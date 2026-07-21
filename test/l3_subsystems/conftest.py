"""Scoped GC fixture — see test/conftest.py for why this isn't global.

Tests here construct real moderngl GL contexts (Renderer3D/Visualizer),
directly or transitively. moderngl's Context/Program/Buffer objects hold
internal back-references to each other — a reference cycle, which
CPython's plain refcounting can't collect immediately at end-of-scope;
only the periodic cyclic GC pass does, on its own allocation-count-driven
schedule.

Without forcing collection, dozens of GL contexts can be alive at once
before the cyclic GC catches up, exceeding the driver's concurrent-context
limit and making context creation fail with `_moderngl.Error: cannot
create vertex array/buffer`. Confirmed empirically to be order-dependent
and not confined to files that directly construct a renderer — module
scope (collect once per file) wasn't frequent enough once more than one
preceding module was involved; function scope (collect after every test)
is.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def _release_gl_contexts_promptly():
    yield
    gc.collect()
