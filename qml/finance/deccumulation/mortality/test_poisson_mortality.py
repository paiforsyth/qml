import jax
import jax.numpy as jnp

from qml.finance.deccumulation.mortality.poisson_mortality import (
    PoissonMortalityUpdater,
)


def test_high_death_rate():
    """We should get a death with very high probability if we choose a very
    high death rate."""
    key = jax.random.PRNGKey(4)
    updater = PoissonMortalityUpdater(
        mortality_table=jnp.array([999999999999.0]), bin_starts=jnp.array([0.0])
    )
    result = updater.simulate(
        alive=jnp.array([1.0]),
        cur_time=jnp.array([1.0]),
        time_to_simulate=jnp.array([1.0]),
        key=key,
    )
    assert result == 0.0


def test_low_death_rate():
    """We should not get a death if we choose a low death rate."""
    key = jax.random.PRNGKey(4)
    updater = PoissonMortalityUpdater(
        mortality_table=jnp.array([0.0]), bin_starts=jnp.array([0.0])
    )
    result = updater.simulate(
        alive=jnp.array([1.0]),
        cur_time=jnp.array([1.0]),
        time_to_simulate=jnp.array([1.0]),
        key=key,
    )
    assert result == 1.0
