from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attr
import jax.numpy as jnp
import jax.random

from qml.finance.deccumulation.market_state.market_state_types import MarketStateUpdater
from qml.finance.deccumulation.mortality.mortality_types import MortalityUpdater
from qml.tools.control.simulation import StateUpdater
from qml.tools.jax_util.types import Array

M = TypeVar("M")


@attr.s(frozen=True, auto_attribs=True)
class BasicDeccumulationState(Generic[M]):
    wealth: Array  # num_paths
    market_state: M
    time: Array  # num_paths
    alive: Array  # num_paths
    mortality_key: Array  # jax random number generator key

    def num_paths(self) -> int:
        return len(self.wealth)

    def __attrs_post_init__(self):
        assert (
            (self.num_paths(),)
            == self.time.shape
            == self.wealth.shape
            == self.alive.shape
        ), "number of paths must be consistent"


@attr.s(frozen=True, auto_attribs=True)
class BasicDeccumulationControl:
    allocation: Array = attr.ib()  # num_paths by num_assets
    consumption: Array  # num_paths

    def num_paths(self) -> int:
        return len(self.allocation)

    @allocation.validator
    def validate_allocation(self, attribute, value):
        assert jnp.allclose(
            jnp.sum(self.allocation, axis=1), jnp.ones(self.num_paths())
        )


class BasicConstraintChecker(ABC):
    """Should raise an error if constraints are violated."""

    @abstractmethod
    def check_constraints(
        self, state: BasicDeccumulationState[M], control: BasicDeccumulationControl
    ):
        pass


@attr.s(frozen=True, auto_attribs=True)
class BasicDeccumulationStateUpdater(
    StateUpdater[BasicDeccumulationState[M], BasicDeccumulationControl],
):
    mortality_updater: MortalityUpdater
    market_updater: MarketStateUpdater[M]
    checker: BasicConstraintChecker
    timestep_size: float

    def update_state(
        self, old_state: BasicDeccumulationState[M], control: BasicDeccumulationControl
    ) -> BasicDeccumulationState[M]:
        self.checker.check_constraints(state=old_state, control=control)
        wealth_after_consumption = old_state.wealth - control.consumption
        num_paths = old_state.num_paths()
        timestep_vector = jnp.full(num_paths, self.timestep_size)
        new_market_state, wealth_after_simulation = self.market_updater.simulate(
            market_state=old_state.market_state,
            allocation=control.allocation,
            wealth=wealth_after_consumption,
            time_to_simulate=timestep_vector,
        )
        new_mortality_key, subkey = jax.random.split(old_state.mortality_key)
        new_alive = self.mortality_updater.simulate(
            old_state.alive,
            cur_time=old_state.time,
            time_to_simulate=timestep_vector,
            key=subkey,
        )
        new_state = attr.evolve(
            old_state,
            wealth=wealth_after_simulation,
            market_state=new_market_state,
            time=old_state.time + timestep_vector,
            alive=new_alive,
            mortality_key=new_mortality_key,
        )
        return new_state
