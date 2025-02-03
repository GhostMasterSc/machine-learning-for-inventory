import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import json

class InventoryDashboard:
    def __init__(self, forecaster, optimizer, data_processor, data, mlflow_manager):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.data_processor = data_processor
        self.data = data
        self.mlflow_manager = mlflow_manager
        self.client = MlflowClient()
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for forecasting and optimization"""
        X, y = self.data_processor.prepare_data(self.data)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            self.data_processor.split_data(X, y)
        
    def run(self):
        st.title("Inventory Management System")
        
        # Sidebar for parameters
        st.sidebar.header("Parameters")
        holding_cost = st.sidebar.slider("Holding Cost", 0.0, 1.0, 0.1)
        stockout_cost = st.sidebar.slider("Stockout Cost", 1.0, 10.0, 5.0)
        lead_time = st.sidebar.slider("Lead Time (days)", 1, 10, 3)
        
        # Main dashboard tabs
        tabs = st.tabs(["Sales Forecast", "Inventory Optimization", "Cost Analysis"])
        
        with tabs[0]:
            self._show_sales_forecast()
            
        with tabs[1]:
            self._show_inventory_optimization()
            
        with tabs[2]:
            self._show_cost_analysis()
    
    def _show_sales_forecast(self):
        st.header("Sales Forecast")
        
        # MLflow Experiment Tracking
        st.subheader("MLflow Experiment History")
        
        # Management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Delete All Experiments", type="secondary"):
                if st.warning("Are you sure you want to delete all experiments?"):
                    self.mlflow_manager.delete_all_runs()
                    st.success("All experiments deleted!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Experiments"):
                st.rerun()
        
        # Get runs comparison
        if st.checkbox("Show Detailed Run Comparison"):
            comparison_df = self.mlflow_manager.compare_runs()
            
            # Format the dataframe
            if not comparison_df.empty:
                # Round numeric columns
                numeric_cols = ["rmse", "mae", "mape"]
                comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)
                
                # Add delete button for each run
                def delete_run_button(run_id):
                    if st.button("🗑️", key=f"delete_{run_id}"):
                        self.mlflow_manager.delete_run(run_id)
                        st.success(f"Run {run_id} deleted!")
                        st.rerun()
                
                comparison_df['Actions'] = comparison_df['run_id'].apply(delete_run_button)
            
            st.dataframe(comparison_df)
            
            # Download comparison as CSV
            if st.button("Download Comparison CSV"):
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "run_comparison.csv",
                    "text/csv"
                )
        
        # Get all runs for the current experiment
        runs = self.client.search_runs(
            experiment_ids=[self.mlflow_manager.experiment_id],
            order_by=["metrics.rmse ASC"]  # Sort by RMSE ascending
        )
        
        # Filter out invalid runs
        runs = [run for run in runs if (
            run.data.metrics.get('rmse', 0) > 0 and
            run.data.metrics.get('mae', 0) > 0 and
            run.data.metrics.get('mape', 0) > 0
        )]
        
        # Add run tags
        if runs:
            selected_run = st.selectbox(
                "Select Run to Tag",
                options=[run.info.run_id for run in runs]
            )
            
            tag_col1, tag_col2 = st.columns(2)
            with tag_col1:
                tag_key = st.text_input("Tag Key")
            with tag_col2:
                tag_value = st.text_input("Tag Value")
                
            if st.button("Add Tag") and tag_key and tag_value:
                self.client.set_tag(selected_run, tag_key, tag_value)
                st.success(f"Added tag {tag_key}={tag_value} to run {selected_run}")
        
        if not runs:
            st.info("No experiments yet. Train a model to see the experiment history.")
        
        if runs:
            # Create a table of runs
            runs_data = []
            for run in runs:
                # Get start time from tag or run info
                start_time = run.data.tags.get("start_time")
                if start_time:
                    start_time = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Fallback to run start time
                    start_time = pd.to_datetime(run.info.start_time/1000, unit='s').strftime("%Y-%m-%d %H:%M:%S")
                
                runs_data.append({
                    "Run ID": run.info.run_id,
                    "Start Time": start_time,
                    "RMSE": f"{run.data.metrics.get('rmse', 0):.4f}",
                    "MAE": f"{run.data.metrics.get('mae', 0):.4f}",
                    "MAPE": f"{run.data.metrics.get('mape', 0):.2f}%",
                    "Parameters": json.dumps(run.data.params, indent=2)
                })
            
            runs_df = pd.DataFrame(runs_data)
            st.dataframe(runs_df)
            
            # Plot metrics history
            st.subheader("Metrics History")
            metric_options = ["rmse", "mae", "mape"]
            selected_metric = st.selectbox("Select Metric", metric_options)
            
            fig = go.Figure()
            metric_values = [float(run.data.metrics.get(selected_metric, 0)) for run in runs]
            fig.add_trace(go.Scatter(
                y=metric_values,
                mode='lines+markers',
                name=selected_metric.upper()
            ))
            fig.update_layout(title=f"{selected_metric.upper()} History")
            st.plotly_chart(fig)
        
        # Model configuration
        st.subheader("Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Number of Epochs", min_value=10, max_value=1000, value=100)
            batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
        with col2:
            hidden_units = st.number_input("Hidden Units", min_value=16, max_value=256, value=64)
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        with col3:
            num_lstm_layers = st.number_input("Number of LSTM Layers", min_value=1, max_value=5, value=2)
            hidden_units_decay = st.number_input("Hidden Units Decay", min_value=0.1, max_value=1.0, value=0.5, format="%.2f")
        
        # Load best model
        if runs and st.button("Load Best Model"):
            best_run = min(runs, key=lambda run: run.data.metrics.get('rmse', float('inf')))
            st.write(f"Loading model from run {best_run.info.run_id}")
            model_uri = f"runs:/{best_run.info.run_id}/model"
            self.forecaster.model = mlflow.tensorflow.load_model(model_uri)
            st.success("Model loaded successfully!")
        
        # Update model config
        if st.button("Update Configuration"):
            self.forecaster.config.update({
                "epochs": epochs,
                "batch_size": batch_size,
                "hidden_units": hidden_units,
                "learning_rate": learning_rate,
                "num_lstm_layers": num_lstm_layers,
                "hidden_units_decay": hidden_units_decay
            })
            self.forecaster.model = self.forecaster._build_model()
            st.success("Configuration updated!")
        
        if st.button("Train Model"):
            with st.spinner("Training the forecasting model..."):
                history = self.forecaster.train(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val
                )
                
                # Make predictions
                y_pred = self.forecaster.predict(self.X_test)
                metrics = self.forecaster.evaluate(self.X_test, self.y_test)
                
                # Log metrics to MLflow
                if self.mlflow_manager:
                    with self.mlflow_manager.start_run():
                        self.mlflow_manager.log_metrics(metrics)
                st.success("Training completed!")
        
        # Actual vs Predicted plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name="Actual",
            y=self.data_processor.inverse_transform(self.y_test, 'sales').flatten(),
            mode="lines"
        ))
        if 'y_pred' in locals():
            fig.add_trace(go.Scatter(
                name="Predicted",
                y=self.data_processor.inverse_transform(y_pred, 'sales').flatten(),
                mode="lines"
            ))
        st.plotly_chart(fig)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{metrics['rmse']:.2f}" if 'metrics' in locals() else "0.0")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.2f}" if 'metrics' in locals() else "0.0")
        with col3:
            st.metric("MAPE", f"{metrics['mape']:.2f}%" if 'metrics' in locals() else "0.0%")
    
    def _show_inventory_optimization(self):
        st.header("Inventory Optimization")
        
        # Inventory Level Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(name="Inventory Level", mode="lines"))
        fig.add_trace(go.Scatter(name="Reorder Point", mode="lines"))
        st.plotly_chart(fig)
        
        # Performance Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Service Level", "0.0%")
        with col2:
            st.metric("Average Inventory", "0.0")
    
    def _show_cost_analysis(self):
        st.header("Cost Analysis")
        
        # Cost Comparison Bar Chart
        fig = go.Figure(data=[
            go.Bar(name="DRL Policy", x=["Holding", "Stockout", "Ordering", "Total"]),
            go.Bar(name="Base Stock Policy", x=["Holding", "Stockout", "Ordering", "Total"])
        ])
        st.plotly_chart(fig) 