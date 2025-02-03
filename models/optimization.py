import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DRLAgent(nn.Module):
    def __init__(self, config):
        super(DRLAgent, self).__init__()
        self.config = config
        
        # Neural network layers
        self.fc1 = nn.Linear(config['state_dim'], config['hidden_dim'])
        self.fc2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.fc3 = nn.Linear(config['hidden_dim'], config['action_dim'])
        
        # Experience replay buffer
        self.memory = deque(maxlen=config['buffer_size'])
        self.optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return np.random.uniform(0, 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            return q_value.item()
    
    def train_step(self):
        if len(self.memory) < self.config['batch_size']:
            return
        
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        current_q = self.forward(states)
        next_q = self.forward(next_states)
        
        target = rewards + self.config['gamma'] * next_q.max(1)[0]
        loss = nn.MSELoss()(current_q, target.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class BaseStockPolicy:
    def __init__(self, lead_time, holding_cost, stockout_cost):
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.base_stock_level = None
        
    def calculate_base_stock_level(self, demand_mean, demand_std):
        # Calculate safety stock using normal distribution
        z_score = 1.96  # 95% service level
        safety_stock = z_score * demand_std * np.sqrt(self.lead_time)
        
        # Base stock level = mean demand during lead time + safety stock
        self.base_stock_level = (demand_mean * self.lead_time) + safety_stock
        
    def get_order_quantity(self, current_inventory):
        if self.base_stock_level is None:
            raise ValueError("Base stock level not calculated")
        
        return max(0, self.base_stock_level - current_inventory) 