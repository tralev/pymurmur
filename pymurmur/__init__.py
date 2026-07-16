"""pymurmur — 3D murmuration simulation and visualisation.

Simulate starling flocks at any scale (150 → 300 000 birds)
using interchangeable physics models with optional real-time 3D rendering.

Public API (I7.2):
    from pymurmur import SimConfig, SimulationEngine, Recorder
"""

from .core.config import SimConfig  # noqa: F401
from .simulation.engine import SimulationEngine  # noqa: F401
from .capture.recorder import Recorder  # noqa: F401
