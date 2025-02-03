import unittest
import numpy as np
import pandas as pd
from utils.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.sequence_length = 5
        self.processor = DataProcessor(self.sequence_length)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'sales': np.random.randint(0, 1000, 100),
            'stock': np.random.randint(0, 2000, 100),
            'price': np.random.uniform(10, 100, 100),
            'promotion': np.random.choice([0, 1], 100),
            'holiday': np.random.choice([0, 1], 100),
            'weekday': np.random.randint(0, 7, 100),
            'month': np.random.randint(1, 13, 100),
            'year': np.full(100, 2023),
            'lead_time': np.random.randint(1, 10, 100),
            'demand_uncertainty': np.random.uniform(0.1, 0.5, 100),
            'supplier_reliability': np.random.uniform(0.7, 1.0, 100)
        }, index=dates)
        
    def test_data_preparation(self):
        """Test if data sequences are created correctly"""
        X, y = self.processor.prepare_data(self.sample_data.copy())
        
        # Check shapes
        self.assertEqual(X.shape[1], self.sequence_length)
        self.assertEqual(X.shape[2], len(self.sample_data.columns))
        self.assertEqual(len(y), len(X))
        
    def test_data_scaling(self):
        """Test if data scaling works correctly"""
        X, y = self.processor.prepare_data(self.sample_data.copy())
        
        # Check if values are scaled between 0 and 1
        self.assertTrue(np.all(X >= -1e-10))  # Allow for small numerical errors
        self.assertTrue(np.all(X <= 1 + 1e-10))
        self.assertTrue(np.all(y >= -1e-10))
        self.assertTrue(np.all(y <= 1 + 1e-10))
        
        # Check scaling for features that should vary
        varying_features = ['sales', 'stock', 'price', 'lead_time', 'demand_uncertainty', 'supplier_reliability']
        feature_indices = [list(self.sample_data.columns).index(f) for f in varying_features]
        
        for feature_idx in feature_indices:
            feature_values = X[:, :, feature_idx].flatten()
            self.assertTrue(np.min(feature_values) < 0.01,  # Close to 0
                           f"Feature {self.sample_data.columns[feature_idx]} doesn't have values close to 0")
            self.assertTrue(np.max(feature_values) > 0.99,  # Close to 1
                           f"Feature {self.sample_data.columns[feature_idx]} doesn't have values close to 1")
        
    def test_data_splitting(self):
        """Test if data splitting works correctly"""
        X, y = self.processor.prepare_data(self.sample_data.copy())
        X_train, y_train, X_val, y_val, X_test, y_test = self.processor.split_data(X, y)
        
        # Check if splits sum up to total length
        total_length = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total_length, len(X)) 