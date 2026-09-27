import unittest
import numpy as np
from models.forecasting import SalesForecaster
from config import LSTM_PARAMS

class TestSalesForecaster(unittest.TestCase):
    def setUp(self):
        self.config = LSTM_PARAMS
        self.forecaster = SalesForecaster(self.config)
        
    def test_model_structure(self):
        """Test the model has the expected layer composition.

        With ``num_lstm_layers=2`` we expect 2 LSTM, 2 Dropout and 1 Dense
        layer. We count by type rather than a fixed total, since whether the
        functional ``Input`` layer appears in ``model.layers`` varies by Keras
        version.
        """
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        layers = self.forecaster.model.layers
        self.assertEqual(sum(isinstance(l, LSTM) for l in layers), 2)
        self.assertEqual(sum(isinstance(l, Dropout) for l in layers), 2)
        self.assertEqual(sum(isinstance(l, Dense) for l in layers), 1)
        
    def test_prediction_shape(self):
        """Test if model predictions have correct shape"""
        batch_size = 10
        X_test = np.random.random((batch_size, self.config['sequence_length'], self.config['n_features']))
        predictions = self.forecaster.predict(X_test)
        self.assertEqual(predictions.shape, (batch_size, 1))
        
    def test_evaluation_metrics(self):
        """Test if evaluation metrics are correctly calculated"""
        y_test = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        X_test = np.random.random((3, self.config['sequence_length'], self.config['n_features']))
        
        # Mock predict method
        self.forecaster.predict = lambda x: y_pred
        
        metrics = self.forecaster.evaluate(X_test, y_test)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics) 