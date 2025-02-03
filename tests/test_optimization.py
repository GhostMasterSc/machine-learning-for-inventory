import unittest
import numpy as np
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, skipping DRL tests")

from models.optimization import BaseStockPolicy
# Only import DRLAgent if torch is available
if TORCH_AVAILABLE:
    from models.optimization import DRLAgent
from config import DRL_PARAMS

class TestDRLAgent(unittest.TestCase):
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def setUp(self):
        self.config = DRL_PARAMS
        self.agent = DRLAgent(self.config)
        
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_action_selection(self):
        """Test if agent can select actions"""
        state = np.random.random(self.config['state_dim'])
        action = self.agent.select_action(state, epsilon=0)
        self.assertTrue(isinstance(action, float))
        
    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_memory_buffer(self):
        """Test experience replay buffer"""
        state = np.random.random(self.config['state_dim'])
        next_state = np.random.random(self.config['state_dim'])
        self.agent.memory.append((state, 1.0, 0.5, next_state))
        self.assertEqual(len(self.agent.memory), 1)

class TestBaseStockPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = BaseStockPolicy(lead_time=3, holding_cost=0.1, stockout_cost=5.0)
        
    def test_base_stock_calculation(self):
        """Test if base stock level is calculated correctly"""
        demand_mean = 100
        demand_std = 20
        self.policy.calculate_base_stock_level(demand_mean, demand_std)
        
        # Base stock should be greater than mean demand during lead time
        self.assertGreater(self.policy.base_stock_level, demand_mean * self.policy.lead_time)
        
    def test_order_quantity(self):
        """Test if order quantity is calculated correctly"""
        self.policy.base_stock_level = 100
        current_inventory = 80
        order_qty = self.policy.get_order_quantity(current_inventory)
        self.assertEqual(order_qty, 20) 