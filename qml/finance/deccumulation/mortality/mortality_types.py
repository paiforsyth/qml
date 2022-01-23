from abc import ABC, abstractmethod

from qml.tools.jax_util.types import Array


class MortalityUpdater(ABC):
    @abstractmethod
    def simulate(
        self, alive: Array, cur_time: Array, time_to_simulate: Array, key: Array
    ) -> Array:
        """
        alive: shape (num_paths,)
        cur_time: shape (num_paths,)
        time_to_simulate shape (num_paths,)
        key: jax random key
        """
