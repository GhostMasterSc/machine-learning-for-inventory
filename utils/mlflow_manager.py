import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import streamlit as st
from mlflow.models.signature import infer_signature
import pandas as pd
import os
import shutil

class MLflowManager:
    def __init__(self, experiment_name):
        """Initialize MLflow experiment"""
        mlruns_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        
        # Set tracking URI without cleaning directory
        mlflow.set_tracking_uri("file:" + mlruns_dir)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            except Exception as e:
                print(f"Error creating experiment: {e}")
                raise
        else:
            self.experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
    def start_run(self):
        """Start a new MLflow run"""
        return mlflow.start_run()
    
    def log_metrics(self, metrics):
        """Log metrics to current run"""
        mlflow.log_metrics(metrics)
    
    def log_params(self, params):
        """Log parameters to current run"""
        mlflow.log_params(params)
    
    def compare_runs(self):
        """Get all runs for the experiment as a DataFrame"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def delete_all_runs(self):
        """Delete all runs in the experiment"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        for run_id in runs['run_id']:
            mlflow.delete_run(run_id)
        
    def log_model(self, model, name="model", X_sample=None, y_sample=None):
        """Log model with signature and sample input"""
        if X_sample is not None and y_sample is not None:
            signature = infer_signature(X_sample, y_sample)
            mlflow.tensorflow.log_model(model, name, signature=signature)
        else:
            mlflow.tensorflow.log_model(model, name)
            
    def log_artifact(self, local_path):
        """Log additional files/artifacts"""
        mlflow.log_artifact(local_path)
        
    def get_best_run(self, metric="rmse", mode="min"):
        """Get the best run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if mode=='min' else 'DESC'}"]
        )
        return runs[0] if runs else None
    
    def delete_run(self, run_id):
        """Delete a specific run"""
        mlflow.delete_run(run_id)
    
    def compare_runs_df(self, metric_list=None):
        """Get comparison dataframe of all runs"""
        if metric_list is None:
            metric_list = ["rmse", "mae", "mape"]
            
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        runs_data = []
        for run in runs:
            # Filter out invalid runs
            if (run.data.metrics.get('rmse', 0) <= 0 or 
                run.data.metrics.get('mae', 0) <= 0 or
                run.data.metrics.get('mape', 0) <= 0):
                continue
            
            # Get start time from tag or run info
            start_time = run.data.tags.get("start_time")
            if start_time:
                start_time = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Fallback to run start time
                start_time = pd.to_datetime(run.info.start_time/1000, unit='s').strftime("%Y-%m-%d %H:%M:%S")
            
            run_data = {
                "run_id": run.info.run_id,
                "start_time": start_time,
                **{m: run.data.metrics.get(m, None) for m in metric_list},
                **run.data.params
            }
            runs_data.append(run_data)
            
        return pd.DataFrame(runs_data) 