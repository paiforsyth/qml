import math

import jax
import jax.numpy as jnp
from hypothesis import example, given, settings
from hypothesis import strategies as st

from qml.finance.deccumulation.market_state.gbm_market_state import (
    GBMMarketState,
    GBMMarketStateUpdater,
)


@given(
    growth_rate=st.floats(min_value=0.1, max_value=10),
    initial_wealth=st.floats(min_value=0.0, max_value=100.0),
    initial_asset_price=st.floats(min_value=1.0, max_value=10.0),
    time_to_simulate=st.floats(min_value=1.0, max_value=2.0),
)
@example(
    growth_rate=0.1,
    initial_wealth=0.0018941275193418154,
    initial_asset_price=1.0,
    time_to_simulate=1.0,
)
@settings(max_examples=10, deadline=None)
def test_deterministic_asset(
    growth_rate: float,
    initial_wealth: float,
    initial_asset_price: float,
    time_to_simulate: float,
):
    """A deterministic asset with a known growth rate should grow exponentially
    at that rate."""
    initial_state = GBMMarketState(
        asset_prices=jnp.array([[initial_asset_price], [initial_asset_price]]),
        key=jax.random.PRNGKey(42),
    )
    updater = GBMMarketStateUpdater(
        mu=jnp.array([growth_rate]), sigma=jnp.array([0.0]), C=jnp.eye(1)
    )
    initial_wealth_vector = jnp.array([initial_wealth, initial_wealth])
    time_to_simulate_vector = jnp.array([time_to_simulate, time_to_simulate])
    new_state, new_wealth = updater.simulate(
        market_state=initial_state,
        allocation=jnp.array([1.0, 1.0]),
        wealth=initial_wealth_vector,
        time_to_simulate=time_to_simulate_vector,
    )
    assert jnp.allclose(
        new_wealth,
        initial_wealth_vector
        * math.exp(growth_rate * time_to_simulate)
        * jnp.array([1.0, 1.0]),
        rtol=1e-3,
        atol=1e-4,
    )
