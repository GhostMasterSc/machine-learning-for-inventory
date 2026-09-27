"""Periodic-review inventory simulation.

This module turns a demand series and an ordering policy into a full inventory
trajectory: on-hand levels, orders placed, unmet demand and the resulting
holding / stockout / ordering costs. It is deliberately dependency-light (numpy
only) so it can be unit-tested and reused outside the Streamlit dashboard.

An ordering *policy* is a callable that, given the current inventory position
(on-hand + already-ordered-but-not-yet-received) and the period index, returns
the desired order-up-to level ``S``. The simulator then orders the difference
between ``S`` and the inventory position, clipped at zero. This "order-up-to"
formulation covers the classic base-stock policy (constant ``S``) as well as
forecast-driven policies (``S`` that adapts to expected demand).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """Outcome of a single policy run over a demand series."""

    demand: np.ndarray          # requested demand per period
    sales: np.ndarray           # demand actually fulfilled per period
    inventory: np.ndarray       # on-hand inventory at the end of each period
    orders: np.ndarray          # order quantity placed each period
    target_levels: np.ndarray   # order-up-to level (S) used each period
    unmet: np.ndarray           # unfulfilled demand units per period

    holding_cost: float
    stockout_cost: float
    ordering_cost: float

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.stockout_cost + self.ordering_cost

    @property
    def service_level(self) -> float:
        """Fill rate: fraction of total demand that was fulfilled."""
        total_demand = float(self.demand.sum())
        if total_demand <= 0:
            return 1.0
        return float(self.sales.sum() / total_demand)

    @property
    def average_inventory(self) -> float:
        return float(self.inventory.mean())

    @property
    def stockout_periods(self) -> int:
        return int((self.unmet > 1e-9).sum())

    def cost_breakdown(self) -> dict[str, float]:
        return {
            "Holding": self.holding_cost,
            "Stockout": self.stockout_cost,
            "Ordering": self.ordering_cost,
            "Total": self.total_cost,
        }


def simulate_inventory(
    demand,
    order_up_to_fn,
    *,
    lead_time: int = 3,
    holding_cost: float = 0.1,
    stockout_cost: float = 5.0,
    ordering_cost: float = 2.0,
    initial_inventory: float | None = None,
) -> SimulationResult:
    """Simulate a periodic-review inventory system.

    Parameters
    ----------
    demand:
        Sequence of per-period demand (non-negative).
    order_up_to_fn:
        Callable ``(inventory_position, period_index) -> target_level``.
    lead_time:
        Number of periods between placing and receiving an order.
    holding_cost / stockout_cost / ordering_cost:
        Per-unit holding cost, per-unit stockout penalty and per-order fixed
        cost respectively.
    initial_inventory:
        Starting on-hand inventory. Defaults to the mean demand times the
        lead time, a sensible warm start that avoids a spurious opening
        stockout.
    """
    demand = np.asarray(demand, dtype=float).flatten()
    demand = np.clip(demand, 0.0, None)
    n = len(demand)
    if n == 0:
        raise ValueError("demand series is empty")
    if lead_time < 0:
        raise ValueError("lead_time must be non-negative")

    if initial_inventory is None:
        initial_inventory = float(demand.mean()) * max(lead_time, 1)

    on_hand = float(initial_inventory)
    # Pipeline of orders keyed by the period in which they arrive.
    pipeline = np.zeros(n + lead_time + 1, dtype=float)

    inventory = np.zeros(n)
    sales = np.zeros(n)
    orders = np.zeros(n)
    targets = np.zeros(n)
    unmet = np.zeros(n)

    total_holding = 0.0
    total_stockout = 0.0
    total_ordering = 0.0

    for t in range(n):
        # 1. Receive any orders scheduled to arrive this period.
        on_hand += pipeline[t]

        # 2. Meet demand as far as stock allows.
        fulfilled = min(on_hand, demand[t])
        shortage = demand[t] - fulfilled
        on_hand -= fulfilled

        # 3. Decide and place a replenishment order based on inventory position.
        on_order = float(pipeline[t + 1:].sum())
        inventory_position = on_hand + on_order
        target = float(order_up_to_fn(inventory_position, t))
        order_qty = max(0.0, target - inventory_position)
        if order_qty > 1e-9:
            pipeline[t + lead_time] += order_qty
            total_ordering += ordering_cost

        # 4. Accrue costs and record the period.
        total_holding += holding_cost * on_hand
        total_stockout += stockout_cost * shortage

        inventory[t] = on_hand
        sales[t] = fulfilled
        orders[t] = order_qty
        targets[t] = target
        unmet[t] = shortage

    return SimulationResult(
        demand=demand,
        sales=sales,
        inventory=inventory,
        orders=orders,
        target_levels=targets,
        unmet=unmet,
        holding_cost=total_holding,
        stockout_cost=total_stockout,
        ordering_cost=total_ordering,
    )


def base_stock_policy(demand_mean: float, demand_std: float, lead_time: int,
                      service_level: float = 0.95):
    """Classic base-stock policy: a constant order-up-to level.

    ``S = mean demand over the lead time + safety stock``, where safety stock
    covers demand variability during the lead time at the requested service
    level.
    """
    z = _z_score(service_level)
    safety_stock = z * demand_std * np.sqrt(max(lead_time, 1))
    target = demand_mean * max(lead_time, 1) + safety_stock

    def policy(_inventory_position, _t):
        return target

    policy.target_level = target  # exposed for display
    return policy


def forecast_driven_policy(forecast, demand_std: float, lead_time: int,
                           service_level: float = 0.95):
    """Adaptive policy that sets the order-up-to level from a demand forecast.

    For each period it targets expected demand over the lead-time window
    (drawn from ``forecast``) plus safety stock. Because it reacts to the
    forecast, it holds less stock when demand is expected to fall and more
    when it is expected to rise.
    """
    forecast = np.asarray(forecast, dtype=float).flatten()
    z = _z_score(service_level)
    safety_stock = z * demand_std * np.sqrt(max(lead_time, 1))
    window = max(lead_time, 1)
    n = len(forecast)

    def policy(_inventory_position, t):
        end = min(t + window, n)
        if end > t:
            expected = float(forecast[t:end].sum())
        else:
            expected = float(forecast[-1]) * window
        return expected + safety_stock

    return policy


def _z_score(service_level: float) -> float:
    """Inverse standard-normal CDF for common service levels (no SciPy dep)."""
    table = {
        0.80: 0.8416, 0.85: 1.0364, 0.90: 1.2816,
        0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263,
    }
    # Snap to the nearest tabulated service level.
    nearest = min(table, key=lambda k: abs(k - service_level))
    return table[nearest]
