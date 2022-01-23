from typing import TypeVar

import jax.numpy as jnp

from qml.finance.deccumulation.control_types import (
    BasicConstraintChecker,
    BasicDeccumulationControl,
    BasicDeccumulationState,
)

M = TypeVar("M")


class NoShortNoNegWealthChecker(BasicConstraintChecker):
    def check_constraints(
        self, state: BasicDeccumulationState[M], control: BasicDeccumulationControl
    ):
        assert jnp.all(control.allocation >= 0), "No shorting is allowed"
        assert jnp.all(control.consumption <= state.wealth), (
            "Cannot consume more than available wealth:w" ""
        )
