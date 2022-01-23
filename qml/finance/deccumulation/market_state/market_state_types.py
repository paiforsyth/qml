from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qml.tools.jax_util.types import Array

M = TypeVar("M")


class MarketStateUpdater(ABC, Generic[M]):
    @abstractmethod
    def simulate(
        self, market_state: M, allocation: Array, wealth: Array, time_to_simulate: Array
    ) -> tuple[M, Array]:
        """simulate  market state forward a given amount, updating wealth
        according to allocation amoung assets."""
