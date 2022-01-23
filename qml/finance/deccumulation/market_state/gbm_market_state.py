import attr
import jax.numpy as jnp
import jax.random

from qml.finance.deccumulation.market_state.market_state_types import MarketStateUpdater
from qml.tools.jax_util.types import Array
from qml.tools.monte_carlo.brownian_motion import vector_geometric_brownian_motion


@attr.s(frozen=True, auto_attribs=True)
class GBMMarketState:
    asset_prices: Array = attr.ib()  # dimension num_paths by num assets
    key: Array  # jax rng key

    @asset_prices.validator
    def valid_asset_prices(self, attribute, value):
        assert jnp.all(self.asset_prices > 0), "Asset prices must be positive"


@attr.s(frozen=True, auto_attribs=True)
class GBMMarketStateUpdater(MarketStateUpdater[GBMMarketState]):
    mu: Array  # (num_assets,,)
    sigma: Array  # (num_assets,)
    C: Array  # array of dimension (num_asset,n_num_assets) such that CC^T is the correlation between assets

    def simulate(
        self,
        market_state: GBMMarketState,
        allocation: Array,
        wealth: Array,
        time_to_simulate: Array,
    ) -> tuple[GBMMarketState, Array]:
        """
        Allocation: dimension num_paths by num_assets
        wealth: dimension num_paths
        time_to_simulate: dimension num_paths
        """
        num_path, num_assets = market_state.asset_prices.shape
        new_key, sub_key = jax.random.split(market_state.key)
        new_prices = vector_geometric_brownian_motion(
            num_paths=num_path,
            start_value=market_state.asset_prices,
            mu=self.mu,
            sigma=self.sigma,
            C=self.C,
            final_time=time_to_simulate,
            steps=1,
            key=sub_key,
        ).reshape(num_path, num_assets)
        growth = new_prices / market_state.asset_prices
        portfolio_growth = (growth * allocation.reshape(num_path, num_assets)).sum(
            axis=1
        )
        new_wealth = wealth * portfolio_growth
        new_market_state = attr.evolve(
            market_state, asset_prices=new_prices, key=new_key
        )
        return (new_market_state, new_wealth)
