import attr
import jax
import jax.numpy as jnp

from qml.finance.deccumulation.mortality.mortality_types import MortalityUpdater
from qml.tools.jax_util.types import Array


@attr.s(frozen=True, auto_attribs=True)
class PoissonMortalityUpdater(MortalityUpdater):
    """
    mortality_table: size (num_bins,) giving mortality rate per unit time for each bin
    bin_starts: size (num_bin,) giving left boundary of each bin.
    """

    mortality_table: Array
    bin_starts: Array

    def simulate(
        self,
        alive: Array,
        cur_time: Array,
        time_to_simulate: Array,
        key: Array,
    ) -> Array:
        """
        alive: shape (num_paths,)
        cur_time: shape (num_paths,)
        time_to_simulate shape (num_paths,)
        key: jax random key
        """
        num_path = alive.size
        num_bins = self.mortality_table.size
        end_time = cur_time.reshape(num_path, 1) + time_to_simulate.reshape(num_path, 1)
        start_time = cur_time.reshape(num_path, 1)
        bin_startpoints = jnp.broadcast_to(
            self.bin_starts.reshape(1, num_bins), (num_path, num_bins)
        )
        bin_endpoints = jnp.concatenate(
            [bin_startpoints[:, 1:], end_time], axis=1
        ).reshape(num_path, num_bins)
        time_spent_in_bins = (
            (
                jnp.minimum(end_time, bin_endpoints)
                - jnp.maximum(start_time, bin_startpoints)
            )
            * (start_time <= bin_endpoints)
            * (end_time >= bin_startpoints)
        )  # shape num_paths, num bins`
        effective_rates = (
            self.mortality_table.reshape(1, num_bins) * time_spent_in_bins
        ).sum(axis=1)
        died = jax.random.poisson(key=key, lam=effective_rates, shape=[num_path]) > 0.0
        return alive * (1 - died)
