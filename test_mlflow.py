from utils.mlflow_manager import MLflowManager
import numpy as np

def test_mlflow():
    # Initialize MLflow manager
    mlflow_manager = MLflowManager("test_experiment")
    
    print("MLflow setup completed")
    print(f"Experiment ID: {mlflow_manager.experiment_id}")
    
    # Try to create a run
    with mlflow_manager.start_run():
        print("Run started successfully")
        
        # Log some test metrics
        test_metrics = {
            "test_metric_1": np.random.random(),
            "test_metric_2": np.random.random()
        }
        mlflow_manager.log_metrics(test_metrics)
        print(f"Logged metrics: {test_metrics}")
        
        # Log some test parameters
        test_params = {
            "param1": "value1",
            "param2": 42
        }
        mlflow_manager.log_params(test_params)
        print(f"Logged parameters: {test_params}")
    
    # Try to get runs
    runs_df = mlflow_manager.compare_runs()
    print("\nRuns found:")
    print(runs_df)

if __name__ == "__main__":
    test_mlflow() 