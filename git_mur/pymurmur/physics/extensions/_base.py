"""Extension protocol — pluggable behavioural modules.

Each extension is a class implementing apply(flock). The ExtensionManager
assembles whichever are enabled in config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..flock import PhysicsFlock


class Extension(ABC):
    """Base protocol for behavioural extensions."""

    @abstractmethod
    def apply(self, flock: PhysicsFlock) -> None:
        """Apply the extension's effect to the flock."""
        ...
