import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import json
from tensorflow.keras import backend as K

class InventoryDashboard:
    def __init__(self, forecaster, optimizer, data_processor, data, mlflow_manager):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.data_processor = data_processor
        self.data = data
        self.mlflow_manager = mlflow_manager
        self.client = MlflowClient()
        # Initialize config from forecaster's config
        self.config = self.forecaster.config.copy()
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
        
        # Model Configuration Section
        with st.expander("Model Configuration"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self.config["hidden_units"] = st.number_input(
                    "Hidden Units",
                    min_value=16,
                    max_value=512,
                    value=self.config["hidden_units"],
                    step=16
                )
                self.config["batch_size"] = st.number_input(
                    "Batch Size",
                    min_value=8,
                    max_value=128,
                    value=self.config["batch_size"],
                    step=8
                )
                self.config["patience"] = st.number_input(
                    "Early Stopping Patience",
                    min_value=5,
                    max_value=50,
                    value=self.config.get("patience", 20),
                    step=5
                )
                
            with col2:
                self.config["num_lstm_layers"] = st.number_input(
                    "Number of LSTM Layers",
                    min_value=1,
                    max_value=5,
                    value=self.config["num_lstm_layers"]
                )
                self.config["dropout_rate"] = st.slider(
                    "Dropout Rate",
                    min_value=0.0,
                    max_value=0.5,
                    value=self.config["dropout_rate"],
                    step=0.1
                )
                
            with col3:
                self.config["learning_rate"] = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.01,
                    value=self.config["learning_rate"],
                    format="%.4f"
                )
                self.config["hidden_units_decay"] = st.slider(
                    "Hidden Units Decay",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config["hidden_units_decay"],
                    step=0.05
                )
            
            if st.button("Update Model Architecture"):
                self.forecaster.model = self.forecaster._build_model()
        
        # Training Section
        with st.expander("Training", expanded=True):
            if st.button("Train Model"):
                # Update forecaster config with current UI values
                self.forecaster.config.update({
                    "hidden_units": self.config["hidden_units"],
                    "batch_size": self.config["batch_size"],
                    "num_lstm_layers": self.config["num_lstm_layers"],
                    "dropout_rate": self.config["dropout_rate"],
                    "learning_rate": self.config["learning_rate"],
                    "hidden_units_decay": self.config["hidden_units_decay"]
                })
                
                # Rebuild model with new config
                with K.name_scope('model'):  # Add name scope context
                    self.forecaster.model = self.forecaster._build_model()
                
                with st.spinner("Training the forecasting model..."):
                    history = self.forecaster.train(
                        self.X_train, self.y_train,
                        self.X_val, self.y_val
                    )
                    
                    # Make predictions
                    y_pred = self.forecaster.predict(self.X_test)
                    metrics = self.forecaster.evaluate(self.X_test, self.y_test)
                    st.success("Training completed!")
        
        # Results Section
        with st.expander("Results", expanded=True):
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
            if 'metrics' in locals():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col3:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
    
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