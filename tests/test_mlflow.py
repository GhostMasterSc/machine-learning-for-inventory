from utils.mlflow_manager import MLflowManager
import numpy as np
import unittest
import os
import shutil

class TestMLflow(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Clean up any existing test experiment
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        test_exp_dir = os.path.join(mlruns_dir, "test_experiment")
        if os.path.exists(test_exp_dir):
            try:
                shutil.rmtree(test_exp_dir)
            except:
                pass  # Ignore cleanup errors
            
        self.mlflow_manager = MLflowManager("test_experiment")

    def test_experiment_creation(self):
        self.assertIsNotNone(self.mlflow_manager.experiment_id)
        print(f"Experiment ID: {self.mlflow_manager.experiment_id}")

    def test_run_creation_and_logging(self):
        with self.mlflow_manager.start_run():
            # Log metrics
            test_metrics = {
                "test_metric_1": np.random.random(),
                "test_metric_2": np.random.random(),
                "rmse": 0.5,
                "mae": 0.3,
                "mape": 0.2
            }
            self.mlflow_manager.log_metrics(test_metrics)
            
            # Log parameters
            test_params = {
                "param1": "value1",
                "param2": 42
            }
            self.mlflow_manager.log_params(test_params)

        # Verify run was created
        runs_df = self.mlflow_manager.compare_runs()
        self.assertFalse(runs_df.empty)
        print("\nRuns found:")
        print(runs_df)

    def test_run_deletion(self):
        # Create a run with required metrics
        with self.mlflow_manager.start_run():
            self.mlflow_manager.log_metrics({
                "rmse": 0.5,
                "mae": 0.3,
                "mape": 0.2
            })
        
        # Get runs before deletion
        runs_before = self.mlflow_manager.compare_runs()
        
        # Delete all runs
        self.mlflow_manager.delete_all_runs()
        
        # Get runs after deletion
        runs_after = self.mlflow_manager.compare_runs()
        
        self.assertTrue(len(runs_after) < len(runs_before))

    def tearDown(self):
        """Clean up after tests"""
        try:
            self.mlflow_manager.delete_all_runs()
        except:
            pass  # Ignore cleanup errors 