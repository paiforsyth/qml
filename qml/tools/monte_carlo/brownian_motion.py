import math

import jax
import jax.numpy as np
from jax._src.ad_util import Array


def arithmetic_brownian_motion(
    num_paths: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    final_time: float,
    steps: int,
    key: Array,
) -> Array:
    drift_array = (
        mu
        * np.linspace(
            0,
            final_time,
            num=steps,
        ).reshape(1, steps)
    )
    step_size = final_time / steps
    shocks = sigma * math.sqrt(step_size) * jax.random.normal(key, (num_paths, steps))
    cum_shocks = np.cumsum(
        np.concatenate([np.zeros((num_paths, 1)), shocks[:, :-1]], axis=1), axis=1
    )
    return drift_array + cum_shocks


# def arithmetic_brownian_motion(
#     num_paths: int, mu: float, sigma: float, final_time: float, steps: int, key: Array
# ) -> Array:
#     drift_array = (
#         mu
#         * np.linspace(
#             0,
#             final_time,
#             num=steps,
#         ).reshape(1, steps)
#     )
#     step_size = final_time / steps
#     shocks = sigma * math.sqrt(step_size) * jax.random.normal(key, (num_paths, steps))
#     cum_shocks = np.cumsum(
#         np.concatenate([np.zeros((num_paths, 1)), shocks[:, :-1]], axis=1), axis=1
#     )
#     return drift_array + cum_shocks


def geometric_brownian_motion(
    num_paths: int,
    start_value: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    final_time: float,
    steps: int,
    key: Array,
) -> Array:
    arithmetic_drift = (mu - sigma ** 2) / 2
    abm = arithmetic_brownian_motion(
        num_paths=num_paths,
        mu=arithmetic_drift,
        sigma=sigma,
        final_time=final_time,
        steps=steps,
        key=key,
    )
    return start_value * np.exp(abm)
