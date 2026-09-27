import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mlflow.tracking import MlflowClient

from app import theme
from utils.inventory_simulator import (
    base_stock_policy,
    forecast_driven_policy,
    simulate_inventory,
)


class InventoryDashboard:
    def __init__(self, forecaster, optimizer, data_processor, data, mlflow_manager,
                 base_stock=None):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.data_processor = data_processor
        self.data = data
        self.mlflow_manager = mlflow_manager
        self.base_stock = base_stock
        self.client = MlflowClient()
        # Working copy of the model config that the UI can edit.
        self.config = dict(self.forecaster.config)
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for forecasting and optimization."""
        X, y = self.data_processor.prepare_data(self.data)
        (self.X_train, self.y_train, self.X_val, self.y_val,
         self.X_test, self.y_test) = self.data_processor.split_data(X, y)

    # ------------------------------------------------------------------ #
    # App shell
    # ------------------------------------------------------------------ #
    def run(self):
        st.set_page_config(
            page_title="Inventory Intelligence",
            page_icon="📦",
            layout="wide",
        )
        theme.register_plotly_template()
        st.markdown(theme.CSS, unsafe_allow_html=True)

        st.markdown(
            """
            <div class="app-hero">
                <h1>📦 Inventory Intelligence</h1>
                <p>Demand forecasting and inventory optimization — from raw sales
                to a costed replenishment policy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        self._sidebar()

        tab_forecast, tab_optimize, tab_cost = st.tabs(
            ["📈  Sales Forecast", "🏭  Inventory Optimization", "💰  Cost Analysis"]
        )
        with tab_forecast:
            self._show_sales_forecast()
        with tab_optimize:
            self._show_inventory_optimization()
        with tab_cost:
            self._show_cost_analysis()

    def _sidebar(self):
        st.sidebar.header("⚙️  Policy parameters")
        self.holding_cost = st.sidebar.slider(
            "Holding cost / unit / period", 0.0, 1.0, 0.1, 0.05,
            help="Cost of carrying one unit of inventory for one period.")
        self.stockout_cost = st.sidebar.slider(
            "Stockout penalty / unit", 1.0, 20.0, 5.0, 0.5,
            help="Penalty incurred per unit of unmet demand.")
        self.ordering_cost = st.sidebar.slider(
            "Fixed cost / order", 0.0, 20.0, 2.0, 0.5,
            help="Fixed cost charged each time an order is placed.")
        self.lead_time = st.sidebar.slider(
            "Lead time (periods)", 1, 10, 3,
            help="Delay between placing and receiving an order.")
        self.service_level = st.sidebar.select_slider(
            "Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99],
            value=0.95, format_func=lambda v: f"{v:.0%}",
            help="Desired fraction of demand fulfilled from stock.")

        st.sidebar.divider()
        st.sidebar.caption(
            f"Dataset: **{len(self.data)}** periods · "
            f"mean demand **{self.data['sales'].mean():.0f}**")

    # ------------------------------------------------------------------ #
    # Tab 1 — Sales forecast
    # ------------------------------------------------------------------ #
    def _show_sales_forecast(self):
        st.subheader("Demand history")
        self._plot_demand_history()

        st.subheader("Experiment tracking")
        self._experiment_controls()
        self._experiment_history()

        st.subheader("Model configuration")
        self._model_config()

        st.subheader("Train & evaluate")
        self._train_section()

    def _plot_demand_history(self):
        sales = self.data["sales"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data.index, y=sales, mode="lines", name="Daily sales",
            line=dict(color=theme.SERIES[0], width=2)))
        fig.add_trace(go.Scatter(
            x=self.data.index, y=sales.rolling(14, min_periods=1).mean(),
            mode="lines", name="14-day average",
            line=dict(color=theme.SERIES[1], width=2, dash="dot")))
        theme.style_figure(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)

    def _experiment_controls(self):
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄  Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🗑️  Delete all runs", use_container_width=True):
                st.session_state["confirm_delete"] = True

        if st.session_state.get("confirm_delete"):
            st.warning("Delete **all** tracked experiments? This cannot be undone.")
            c1, c2, _ = st.columns([1, 1, 3])
            if c1.button("Yes, delete", type="primary"):
                self.mlflow_manager.delete_all_runs()
                st.session_state["confirm_delete"] = False
                st.success("All experiments deleted.")
                st.rerun()
            if c2.button("Cancel"):
                st.session_state["confirm_delete"] = False
                st.rerun()

    def _get_runs(self):
        runs = self.client.search_runs(
            experiment_ids=[self.mlflow_manager.experiment_id],
            order_by=["metrics.rmse ASC"],
        )
        return [
            run for run in runs
            if run.data.metrics.get("rmse", 0) > 0
            and run.data.metrics.get("mae", 0) > 0
            and run.data.metrics.get("mape", 0) > 0
        ]

    def _experiment_history(self):
        runs = self._get_runs()
        if not runs:
            st.info("No experiments yet — train a model to populate the history.")
            return

        rows = []
        for run in runs:
            start = run.data.tags.get("start_time")
            if start:
                start = pd.to_datetime(start)
            else:
                start = pd.to_datetime(run.info.start_time / 1000, unit="s")
            rows.append({
                "Run ID": run.info.run_id[:8],
                "Started": start.strftime("%Y-%m-%d %H:%M"),
                "RMSE": run.data.metrics.get("rmse", 0),
                "MAE": run.data.metrics.get("mae", 0),
                "MAPE (%)": run.data.metrics.get("mape", 0),
            })
        runs_df = pd.DataFrame(rows)
        st.dataframe(
            runs_df, use_container_width=True, hide_index=True,
            column_config={
                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "MAPE (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        if len(runs) > 1:
            metric = st.selectbox(
                "Metric to trend", ["rmse", "mae", "mape"],
                format_func=str.upper)
            values = [float(r.data.metrics.get(metric, 0)) for r in runs]
            fig = go.Figure(go.Scatter(
                y=values, mode="lines+markers", name=metric.upper(),
                line=dict(color=theme.SERIES[0], width=2),
                marker=dict(size=9)))
            fig.update_layout(xaxis_title="Run (best → worst)",
                              yaxis_title=metric.upper())
            theme.style_figure(fig, title=f"{metric.upper()} across runs", height=300)
            st.plotly_chart(fig, use_container_width=True)

    def _model_config(self):
        with st.expander("LSTM architecture & training", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                self.config["hidden_units"] = st.number_input(
                    "Hidden units", 16, 512, int(self.config["hidden_units"]), 16)
                self.config["batch_size"] = st.number_input(
                    "Batch size", 8, 128, int(self.config["batch_size"]), 8)
                self.config["patience"] = st.number_input(
                    "Early-stopping patience", 5, 50,
                    int(self.config.get("patience", 20)), 5)
            with c2:
                self.config["num_lstm_layers"] = st.number_input(
                    "LSTM layers", 1, 5, int(self.config["num_lstm_layers"]))
                self.config["dropout_rate"] = st.slider(
                    "Dropout", 0.0, 0.5, float(self.config["dropout_rate"]), 0.05)
                self.config["epochs"] = st.number_input(
                    "Max epochs", 10, 500, int(self.config.get("epochs", 100)), 10)
            with c3:
                self.config["learning_rate"] = st.number_input(
                    "Learning rate", 0.0001, 0.01,
                    float(self.config["learning_rate"]), format="%.4f")
                self.config["hidden_units_decay"] = st.slider(
                    "Hidden-unit decay", 0.1, 1.0,
                    float(self.config["hidden_units_decay"]), 0.05)

    def _train_section(self):
        if st.button("🚀  Train model", type="primary"):
            self.forecaster.config.update(self.config)
            self.forecaster.model = self.forecaster._build_model()
            with st.spinner("Training the forecasting model…"):
                history = self.forecaster.train(
                    self.X_train, self.y_train, self.X_val, self.y_val)
                y_pred = self.forecaster.predict(self.X_test)
                metrics = self.forecaster.evaluate(self.X_test, self.y_test)
            st.session_state["forecast_result"] = {
                "y_pred": y_pred,
                "metrics": metrics,
                "loss": list(history.history.get("loss", [])),
                "val_loss": list(history.history.get("val_loss", [])),
            }
            st.success("Training complete.")

        result = st.session_state.get("forecast_result")
        if not result:
            st.info("Configure the model above and train to see forecasts here.")
            return

        m = result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{m['rmse']:.2f}")
        c2.metric("MAE", f"{m['mae']:.2f}")
        c3.metric("MAPE", f"{m['mape']:.2f}%")

        actual = self.data_processor.inverse_transform(self.y_test, "sales").flatten()
        predicted = self.data_processor.inverse_transform(
            result["y_pred"], "sales").flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual",
                                 line=dict(color=theme.SERIES[0], width=2)))
        fig.add_trace(go.Scatter(y=predicted, mode="lines", name="Predicted",
                                 line=dict(color=theme.SERIES[1], width=2)))
        theme.style_figure(fig, title="Actual vs. predicted sales (test set)")
        st.plotly_chart(fig, use_container_width=True)

        if result["loss"]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=result["loss"], mode="lines",
                                     name="Train loss",
                                     line=dict(color=theme.SERIES[0], width=2)))
            fig.add_trace(go.Scatter(y=result["val_loss"], mode="lines",
                                     name="Validation loss",
                                     line=dict(color=theme.SERIES[2], width=2)))
            theme.style_figure(fig, title="Training history", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Simulation helpers (shared by tabs 2 & 3)
    # ------------------------------------------------------------------ #
    def _run_policies(self):
        """Run both policies over historical demand with current parameters."""
        demand = self.data["sales"].to_numpy()
        mean, std = float(demand.mean()), float(demand.std())
        kwargs = dict(
            lead_time=self.lead_time,
            holding_cost=self.holding_cost,
            stockout_cost=self.stockout_cost,
            ordering_cost=self.ordering_cost,
        )
        base = simulate_inventory(
            demand,
            base_stock_policy(mean, std, self.lead_time, self.service_level),
            **kwargs)
        # A rolling forecast stands in for the demand signal (works without a
        # trained LSTM); a trained model could supply this series instead.
        forecast = pd.Series(demand).rolling(
            self.lead_time, min_periods=1).mean().to_numpy()
        adaptive = simulate_inventory(
            demand,
            forecast_driven_policy(forecast, std, self.lead_time, self.service_level),
            **kwargs)
        return base, adaptive

    # ------------------------------------------------------------------ #
    # Tab 2 — Inventory optimization
    # ------------------------------------------------------------------ #
    def _show_inventory_optimization(self):
        base, _ = self._run_policies()
        st.subheader("Base-stock policy simulation")

        c1, c2, c3, c4 = st.columns(4)
        sl = base.service_level
        c1.metric("Service level", f"{sl:.1%}",
                  delta=f"{(sl - self.service_level):.1%} vs target")
        c2.metric("Avg. inventory", f"{base.average_inventory:.0f} units")
        c3.metric("Stockout periods", f"{base.stockout_periods}")
        c4.metric("Total cost", f"${base.total_cost:,.0f}")

        idx = self.data.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=idx, y=base.demand, mode="lines", name="Demand",
            line=dict(color=theme.INK_MUTED, width=1.2)))
        fig.add_trace(go.Scatter(
            x=idx, y=base.inventory, mode="lines", name="On-hand inventory",
            line=dict(color=theme.SERIES[0], width=2.2),
            fill="tozeroy", fillcolor="rgba(42,120,214,0.10)"))
        fig.add_trace(go.Scatter(
            x=idx, y=base.target_levels, mode="lines", name="Order-up-to level",
            line=dict(color=theme.SERIES[1], width=2, dash="dash")))
        theme.style_figure(fig, title="Inventory position over time", height=400)
        fig.update_layout(yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("###### Orders placed")
        fig2 = go.Figure(go.Bar(
            x=idx, y=base.orders, name="Order quantity",
            marker_color=theme.SERIES[2]))
        theme.style_figure(fig2, height=260)
        fig2.update_layout(yaxis_title="Units ordered")
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------ #
    # Tab 3 — Cost analysis
    # ------------------------------------------------------------------ #
    def _show_cost_analysis(self):
        base, adaptive = self._run_policies()
        st.subheader("Policy cost comparison")
        st.markdown(
            '<p class="caption-muted">Base-stock holds a constant target level; '
            'the forecast-driven policy adapts the target to expected demand over '
            'the lead time.</p>', unsafe_allow_html=True)

        savings = base.total_cost - adaptive.total_cost
        pct = (savings / base.total_cost * 100) if base.total_cost else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Base-stock total", f"${base.total_cost:,.0f}")
        c2.metric("Forecast-driven total", f"${adaptive.total_cost:,.0f}")
        c3.metric("Savings", f"${savings:,.0f}", delta=f"{pct:.1f}%")

        categories = ["Holding", "Stockout", "Ordering", "Total"]
        base_vals = [base.cost_breakdown()[c] for c in categories]
        adapt_vals = [adaptive.cost_breakdown()[c] for c in categories]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Base-stock", x=categories, y=base_vals,
                             marker_color=theme.SERIES[0]))
        fig.add_trace(go.Bar(name="Forecast-driven", x=categories, y=adapt_vals,
                             marker_color=theme.SERIES[1]))
        fig.update_layout(barmode="group", bargap=0.25, yaxis_title="Cost ($)")
        theme.style_figure(fig, title="Cost breakdown by component", height=400)
        st.plotly_chart(fig, use_container_width=True)

        summary = pd.DataFrame({
            "Metric": ["Service level", "Avg. inventory", "Stockout periods",
                       "Holding cost", "Stockout cost", "Ordering cost",
                       "Total cost"],
            "Base-stock": [
                f"{base.service_level:.1%}", f"{base.average_inventory:.0f}",
                base.stockout_periods, f"${base.holding_cost:,.0f}",
                f"${base.stockout_cost:,.0f}", f"${base.ordering_cost:,.0f}",
                f"${base.total_cost:,.0f}"],
            "Forecast-driven": [
                f"{adaptive.service_level:.1%}", f"{adaptive.average_inventory:.0f}",
                adaptive.stockout_periods, f"${adaptive.holding_cost:,.0f}",
                f"${adaptive.stockout_cost:,.0f}", f"${adaptive.ordering_cost:,.0f}",
                f"${adaptive.total_cost:,.0f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
