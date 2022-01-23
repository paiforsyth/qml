import attr
import jax.numpy as jnp

from qml.finance.deccumulation.control_types import (
    BasicDeccumulationControl,
    BasicDeccumulationState,
)
from qml.tools.control.simulation import Controller
from qml.tools.jax_util.types import Array


@attr.s(auto_attribs=True, frozen=True)
class ConstantController(
    Controller[BasicDeccumulationState, BasicDeccumulationControl]
):
    """Control that uses a constant proportions asset allocation strategy and
    consumes a fixed amount, as long as sufficient wealth is availible."""

    allocations: Array
    consumption: Array

    def control(self, state: BasicDeccumulationState) -> BasicDeccumulationControl:
        num_paths = state.num_paths()
        return BasicDeccumulationControl(
            allocation=self.allocations * jnp.ones((num_paths, 1)),
            consumption=jnp.minimum(self.consumption, state.wealth),
        )
