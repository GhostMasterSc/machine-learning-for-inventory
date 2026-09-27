# 📦 Machine Learning for Inventory Management

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.3+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An inventory management system that combines **LSTM demand forecasting** with a
**lead-time-aware inventory simulator** to compare replenishment policies on
cost and service level — all through an interactive Streamlit dashboard.

<p align="center">
  📈 Forecast demand&nbsp;&nbsp;→&nbsp;&nbsp; 🏭 Simulate a policy&nbsp;&nbsp;→&nbsp;&nbsp; 💰 Compare the cost
</p>

---

## Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Dashboard](#-dashboard)
- [Testing](#-testing)
- [Development](#️-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Key Features

### Sales Forecasting
- **LSTM neural network** — configurable multi-layer architecture for time-series forecasting.
- **Interactive training** — tune depth, width, dropout and learning rate from the dashboard.
- **Metrics** — RMSE, MAE and MAPE on a held-out test set.

### Experiment Tracking
- **MLflow integration** — automatic logging of parameters and metrics.
- **Run comparison** — sortable history table, metric trends and run management.

### Inventory Optimization
- **Periodic-review simulator** — a lead-time-aware engine that turns a demand series into inventory levels, orders and costs.
- **Base-stock policy** — constant order-up-to level with service-level-driven safety stock.
- **Forecast-driven policy** — order-up-to level that adapts to expected demand over the lead time.
- **Cost analysis** — holding / stockout / ordering breakdown with side-by-side policy comparison.
- **Optional DRL agent** — a PyTorch reinforcement-learning agent is included for experimentation.

---

## 🚀 Quick Start

### Run with Docker (recommended)

```bash
git clone https://github.com/GhostMasterSc/machine-learning-for-inventory.git
cd machine-learning-for-inventory
docker-compose up --build
```

Then open the dashboard at **http://localhost:8501**.

### Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

> **Note:** The **Inventory Optimization** and **Cost Analysis** tabs work
> immediately with no training required. The **Sales Forecast** tab's training
> button needs TensorFlow installed.

---

## 🔬 How It Works

The dashboard is organized as a three-stage pipeline:

| Stage | Component | What it does |
|-------|-----------|--------------|
| 1. Forecast | `SalesForecaster` (LSTM) | Predicts future demand from historical sales |
| 2. Simulate | `simulate_inventory` | Runs a policy over the demand series, tracking on-hand stock, orders and unmet demand across the order lead time |
| 3. Compare | `base_stock` vs. `forecast_driven` | Scores each policy on service level and total cost |

The simulator uses an **order-up-to** formulation: each period it raises the
inventory position (on-hand + on-order) toward a target level `S`. A base-stock
policy keeps `S` constant; the forecast-driven policy adapts `S` to expected
demand — holding less stock when demand is falling and more when it is rising.

---

## 📁 Project Structure

```text
machine-learning-for-inventory/
├── app/
│   ├── dashboard.py           # Streamlit interface (3 tabs)
│   └── theme.py               # Shared palette, Plotly template & CSS
├── models/
│   ├── forecasting.py         # LSTM sales forecaster
│   └── optimization.py        # DRL agent + base-stock policy
├── utils/
│   ├── data_generator.py      # Synthetic demand generator
│   ├── data_processor.py      # Scaling & sequence preparation
│   ├── inventory_simulator.py # Periodic-review engine + policies
│   └── mlflow_manager.py       # MLflow utilities
├── tests/                     # Unit tests
├── .streamlit/config.toml     # Theme configuration
├── config.py                  # Model & inventory parameters
├── main.py                    # Application entry point
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Model and inventory parameters live in [`config.py`](config.py).

**LSTM parameters**

| Parameter | Default | Parameter | Default |
|-----------|---------|-----------|---------|
| `sequence_length` | 30 | `epochs` | 100 |
| `n_features` | 4 | `num_lstm_layers` | 2 |
| `hidden_units` | 128 | `hidden_units_decay` | 0.75 |
| `dropout_rate` | 0.2 | `patience` | 20 |
| `learning_rate` | 0.001 | `batch_size` | 32 |

**DRL parameters**

| Parameter | Default | Parameter | Default |
|-----------|---------|-----------|---------|
| `state_dim` | 4 | `gamma` | 0.99 |
| `action_dim` | 1 | `buffer_size` | 100,000 |
| `hidden_dim` | 128 | `batch_size` | 64 |
| `learning_rate` | 0.001 | | |

---

## 📊 Dashboard

**📈 Sales Forecast**
- Demand history with rolling average
- MLflow experiment tracking (history table, metric trends, run management)
- Configurable LSTM architecture and one-click training
- Actual vs. predicted plot and training-loss curve

**🏭 Inventory Optimization**
- Base-stock simulation over historical demand
- KPIs: service level, average inventory, stockout periods, total cost
- Inventory-position chart (on-hand, demand, order-up-to level) and orders placed

**💰 Cost Analysis**
- Side-by-side base-stock vs. forecast-driven policy comparison
- Holding / stockout / ordering cost breakdown and total savings
- Summary metrics table

All policy parameters — holding cost, stockout penalty, ordering cost, lead time
and target service level — are adjustable from the sidebar and flow through
every tab.

---

## 🧪 Testing

```bash
python run_tests.py     # unittest runner
pytest tests/           # or via pytest
docker-compose run tests  # inside Docker
```

The pure-Python tests (data processing, inventory simulation, base-stock policy)
run **without** TensorFlow or PyTorch installed; the LSTM and DRL tests skip
gracefully when those libraries are unavailable.

---

## 🛠️ Development

```bash
black .      # format
flake8       # lint
mypy .       # type-check
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit your changes: `git commit -m "Add amazing feature"`.
4. Push the branch: `git push origin feature/amazing-feature`.
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License.

## 🙏 Acknowledgments

Built with [TensorFlow](https://www.tensorflow.org/),
[PyTorch](https://pytorch.org/), [MLflow](https://mlflow.org/),
[Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/).
