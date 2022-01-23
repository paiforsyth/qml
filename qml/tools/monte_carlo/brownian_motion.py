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


def vector_arithmetic_brownian_motion(
    num_paths: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    C: np.ndarray,
    steps: int,
    final_time: float,
    key: Array,
) -> Array:
    """
    Simulate a given number of paths off vector arithmetic brownian motion
    num_paths: number of paths to generate
    mu:  array of dimension (n,) indicating geometric drif
    sigma: array of dimension (n,) indicating volatility
    C: array of dimension (n,n) such that CC^T is the correlation between assets
    final_time: float
    returns:
    array of dimension (num_paths, num_steps,n) giving simulated values of Brownian Motion
    """
    assert steps >= 2, "must use at least 2 steps"
    n = mu.size
    C = C.reshape(n, n)
    drift_array = (
        mu.reshape(1, 1, n)
        * np.linspace(
            0,
            final_time,
            steps,
        ).reshape(1, steps, 1)
    )
    step_size = final_time / (steps - 1)
    correlated_normals = (
        C * (jax.random.normal(key, (num_paths, steps - 1, 1, n)))
    ).sum(axis=-1)
    shocks = sigma.reshape(1, 1, n) * math.sqrt(step_size) * correlated_normals
    cum_shocks = np.cumsum(
        np.concatenate([np.zeros((num_paths, 1, n)), shocks], axis=1), axis=1
    )
    return drift_array + cum_shocks


def vector_geometric_brownian_motion(
    num_paths: int,
    start_value: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    C: np.ndarray,
    final_time: float,
    steps: int,
    key: Array,
) -> Array:
    arithmetic_drift = (mu - sigma ** 2) / 2
    abm = vector_arithmetic_brownian_motion(
        num_paths=num_paths,
        mu=arithmetic_drift,
        sigma=sigma,
        final_time=final_time,
        steps=steps,
        key=key,
        C=C,
    )
    return start_value * np.exp(abm)
