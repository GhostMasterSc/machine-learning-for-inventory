from models.forecasting import SalesForecaster
from models.optimization import DRLAgent, BaseStockPolicy
from utils.data_processor import DataProcessor
from utils.data_generator import generate_sample_data
from utils.mlflow_manager import MLflowManager
from app.dashboard import InventoryDashboard
import config

def main():
    # Generate sample data
    data = generate_sample_data()
    
    # Initialize MLflow
    mlflow_manager = MLflowManager()
    
    # Initialize components
    data_processor = DataProcessor(config.LSTM_PARAMS['sequence_length'])
    forecaster = SalesForecaster(config.LSTM_PARAMS, mlflow_manager)
    drl_agent = DRLAgent(config.DRL_PARAMS)
    base_stock_policy = BaseStockPolicy(
        config.LEAD_TIME,
        config.HOLDING_COST,
        config.STOCKOUT_COST
    )
    
    # Initialize dashboard
    dashboard = InventoryDashboard(forecaster, drl_agent, data_processor, data, mlflow_manager)
    
    # Run the application
    dashboard.run()

if __name__ == "__main__":
    main() 