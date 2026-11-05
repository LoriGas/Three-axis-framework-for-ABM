"""Random agent — V0 baseline."""

import random
from typing import List, Tuple

from config import ACTIONS, ModelParams
from .base import AgentBase


class RandomAgent(AgentBase):
    def __init__(self, x: int, y: int, energy: float) -> None:
        super().__init__(x, y, energy)

    def decide_action(self, percept: List[float], params: ModelParams) -> Tuple[int, int, bool]:
        dx, dy = random.choice(ACTIONS)
        return dx, dy, random.random() < 0.5
