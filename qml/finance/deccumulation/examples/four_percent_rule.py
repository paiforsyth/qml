import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from qml.finance.deccumulation.checker.standard_checker import NoShortNoNegWealthChecker
from qml.finance.deccumulation.control_types import (
    BasicDeccumulationState,
    BasicDeccumulationStateUpdater,
)
from qml.finance.deccumulation.controls.constant_deccumulation_controller import (
    ConstantController,
)
from qml.finance.deccumulation.market_state.gbm_market_state import (
    GBMMarketState,
    GBMMarketStateUpdater,
)
from qml.finance.deccumulation.mortality.poisson_mortality import (
    PoissonMortalityUpdater,
)
from qml.tools.control.simulation import simulate


def run(num_paths: int):
    key = jax.random.PRNGKey(42)
    key, mortality_key = jax.random.split(key)
    key, market_key = jax.random.split(key)
    initial_wealth = 700000
    num_assets = 2
    initial_state = BasicDeccumulationState(
        wealth=initial_wealth * jnp.ones(num_paths),
        market_state=GBMMarketState(
            asset_prices=jnp.ones((num_paths, num_assets)), key=market_key
        ),
        time=65.0 * jnp.ones(num_paths),
        alive=jnp.ones(num_paths),
        mortality_key=mortality_key,
    )
    mortality_updater = PoissonMortalityUpdater(
        mortality_table=jnp.array(
            [
                17.5,
                27.8,
                44,
                73.3,
                119.4,
                219.7,  # from statscan
            ]
        )
        / 1000,
        bin_starts=jnp.array([65.0, 70.0, 75.0, 80.0, 85.0, 90.0]),
    )
    market_updater = GBMMarketStateUpdater(
        mu=jnp.array(
            [0.0074, 0.0749],
        ),  # first asset is treasury bills.  Second is stocks
        sigma=jnp.array([0.0, 0.22]),
        C=jnp.eye(2),
    )
    state_updater = BasicDeccumulationStateUpdater(
        mortality_updater=mortality_updater,
        market_updater=market_updater,
        checker=NoShortNoNegWealthChecker(),
        timestep_size=1.0,
    )
    controller = ConstantController(
        allocations=jnp.array([0.5, 0.5]),
        consumption=0.04 * initial_wealth,
    )
    states, controls = simulate(
        initial_state=initial_state,
        state_updater=state_updater,
        controller=controller,
        num_steps=40,
    )
    return states, controls


WEALTH = "wealth"
ALIVE = "alive"
PATH_DIMENSION = "path"
AGE_DIMENSION = "age"


def run_xarray(num_paths: int) -> xr.Dataset:
    states, controls = run(num_paths=num_paths)
    ds = xr.Dataset(
        {
            WEALTH: xr.DataArray(
                np.array(
                    jnp.concatenate(
                        [state.wealth.reshape(num_paths, 1) for state in states], axis=1
                    ),
                ),
                dims=(PATH_DIMENSION, AGE_DIMENSION),
            ),
            ALIVE: xr.DataArray(
                np.array(
                    jnp.concatenate(
                        [state.alive.reshape(num_paths, 1) for state in states], axis=1
                    ),
                ),
                dims=(PATH_DIMENSION, AGE_DIMENSION),
            ),
        }
    )
    ds = ds.assign_coords(
        coords={
            AGE_DIMENSION: [
                state.time[0] for state in states
            ]  # paths dont have different times in this case
        }
    )
    return ds


def get_wealth_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    wealth: xr.DataArray = ds[WEALTH].where(ds[ALIVE])
    df = wealth.to_dataframe().reset_index()
    df = df.dropna()
    return df


def plot_wealth_dataframe(df: pd.DataFrame):
    fig = px.line(data_frame=df, x=AGE_DIMENSION, y=WEALTH, color=PATH_DIMENSION)
    return fig


def run_and_plot():
    ds = run_xarray(num_paths=10)
    df = get_wealth_dataframe(ds)
    fig = plot_wealth_dataframe(df)
    # fig.show()
    fig.write_html("../four_percent_plot.html")


if __name__ == "__main__":
    run_and_plot()
