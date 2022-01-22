import jax
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from qml.tools.monte_carlo import brownian_motion


@given(
    mean=st.floats(min_value=-10, max_value=-10),
    num_steps=st.integers(min_value=5, max_value=10),
    final_time=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=10, deadline=None)
def test_mean_invariant_to_steps(mean: float, num_steps: int, final_time: float):
    """Example test.

    Verify that mean of brownian motion is independent of number of
    steps computed
    """
    expected_mean = mean * final_time
    seed = 4
    key = jax.random.PRNGKey(seed)
    brownian_output = brownian_motion.arithmetic_brownian_motion(
        num_paths=10000,
        mu=jnp.array(mean),
        sigma=jnp.array(1.0),
        final_time=final_time,
        steps=num_steps,
        key=key,
    )
    actual_mean = brownian_output[:, -1].mean()
    assert abs(actual_mean - expected_mean) < 0.1
