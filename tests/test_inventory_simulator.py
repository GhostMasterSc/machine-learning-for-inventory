import unittest

import numpy as np

from utils.inventory_simulator import (
    base_stock_policy,
    forecast_driven_policy,
    simulate_inventory,
)


class TestInventorySimulator(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.demand = np.clip(rng.normal(100, 15, 120), 0, None)

    def test_conservation_of_demand(self):
        """Sales plus unmet demand must equal total demand every period."""
        policy = base_stock_policy(100, 15, lead_time=3)
        result = simulate_inventory(self.demand, policy, lead_time=3)
        np.testing.assert_allclose(result.sales + result.unmet, result.demand)

    def test_service_level_in_range(self):
        policy = base_stock_policy(100, 15, lead_time=3)
        result = simulate_inventory(self.demand, policy, lead_time=3)
        self.assertGreaterEqual(result.service_level, 0.0)
        self.assertLessEqual(result.service_level, 1.0)

    def test_costs_non_negative(self):
        policy = base_stock_policy(100, 15, lead_time=3)
        result = simulate_inventory(self.demand, policy, lead_time=3)
        self.assertGreaterEqual(result.holding_cost, 0.0)
        self.assertGreaterEqual(result.stockout_cost, 0.0)
        self.assertGreaterEqual(result.ordering_cost, 0.0)
        self.assertAlmostEqual(
            result.total_cost,
            result.holding_cost + result.stockout_cost + result.ordering_cost,
        )

    def test_higher_service_level_reduces_stockouts(self):
        low = simulate_inventory(
            self.demand, base_stock_policy(100, 15, 3, service_level=0.80), lead_time=3
        )
        high = simulate_inventory(
            self.demand, base_stock_policy(100, 15, 3, service_level=0.99), lead_time=3
        )
        self.assertLessEqual(high.unmet.sum(), low.unmet.sum())

    def test_forecast_policy_runs(self):
        forecast = np.full_like(self.demand, self.demand.mean())
        policy = forecast_driven_policy(forecast, self.demand.std(), lead_time=3)
        result = simulate_inventory(self.demand, policy, lead_time=3)
        self.assertEqual(len(result.inventory), len(self.demand))

    def test_empty_demand_raises(self):
        with self.assertRaises(ValueError):
            simulate_inventory([], base_stock_policy(1, 1, 1))


if __name__ == "__main__":
    unittest.main()
