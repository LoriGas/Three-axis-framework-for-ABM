"""Abstract base class for all agents."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from config import ModelParams


class AgentBase(ABC):
    """Common interface for all simulation agents."""

    __slots__ = ("x", "y", "energy", "alive", "age", "last_dir")

    def __init__(self, x: int, y: int, energy: float) -> None:
        self.x = x
        self.y = y
        self.energy = energy
        self.alive: bool = True
        self.age: int = 0
        self.last_dir: Tuple[int, int] = (0, 0)

    @abstractmethod
    def decide_action(
        self, percept: List[float], params: ModelParams
    ) -> Tuple[int, int, bool]:
        """Return (dx, dy, reproduce) for the chosen action."""
        ...
