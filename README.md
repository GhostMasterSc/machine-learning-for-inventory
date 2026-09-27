# Machine Learning for Inventory Management System

![Python](https://img.shields.io/badge/Python-3.9-blue.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg) ![MLflow](https://img.shields.io/badge/MLflow-2.3.0-green.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.10.0-red.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

An advanced inventory management system that leverages deep learning for sales forecasting and reinforcement learning for inventory optimization. The system provides real-time predictions and optimization through an interactive dashboard.

## Table of Contents
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Dashboard Features](#-dashboard-features)
- [Testing](#-testing)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🌟 Key Features

### Sales Forecasting
- **LSTM Neural Network**: Multi-layer LSTM architecture for time series forecasting
- **Adaptive Architecture**: Configurable network depth and width
- **Real-time Training**: Interactive model training through the dashboard
- **Performance Metrics**: RMSE, MAE, and MAPE evaluation

### Experiment Tracking
- **MLflow Integration**: Comprehensive experiment logging
- **Parameter Tracking**: Automatic logging of hyperparameters
- **Model Versioning**: Version control for trained models
- **Performance Visualization**: Interactive metric plots

### Inventory Optimization
- **Periodic-review simulation**: Lead-time-aware inventory engine that turns a demand series into inventory levels, orders and costs
- **Base Stock Policy**: Constant order-up-to level with service-level-driven safety stock
- **Forecast-driven Policy**: Order-up-to level that adapts to expected demand over the lead time
- **Cost Analysis**: Detailed breakdown of holding, stockout, and ordering costs, with side-by-side policy comparison
- **Deep Reinforcement Learning**: An optional DRL agent (PyTorch) is included for experimentation

## 🚀 Quick Start

### Prerequisites
- Docker
- Docker Compose
- Git

### Installation
Clone the repository:
git clone https://github.com/GhostMasterSc/machine-learning-for-inventory.git
cd machine-learning-for-inventory

Build and run with Docker:
docker-compose up --build

Access the dashboard at http://localhost:8501

## 💻 System Architecture

The system consists of interconnected components:
- Data Pipeline: Processes and prepares time series data
- Sales Forecast: LSTM-based prediction model
- MLflow Track: Experiment and model versioning
- Streamlit Dashboard: Interactive user interface
- DRL Optimizer: Reinforcement learning for inventory
- Cost Analysis: Performance evaluation

## 📁 Project Structure

inventory-management-ml/
├── app/                       # Dashboard and UI
│   ├── dashboard.py          # Streamlit interface (3 tabs)
│   └── theme.py              # Shared palette, Plotly template & CSS
├── models/                    # ML model implementations
│   ├── forecasting.py        # LSTM sales forecaster
│   └── optimization.py       # DRL agent + base-stock policy
├── utils/                     # Helper functions
│   ├── data_generator.py     # Synthetic demand generator
│   ├── data_processor.py     # Scaling & sequence preparation
│   ├── inventory_simulator.py# Periodic-review inventory engine + policies
│   └── mlflow_manager.py     # MLflow utilities
├── tests/                     # Unit tests
├── .streamlit/config.toml     # Theme configuration
├── config.py                  # System configuration
├── docker-compose.yml         # Docker configuration
├── requirements.txt           # pip dependencies
└── README.md                  # Documentation

## ⚙️ Configuration

LSTM Parameters:
- sequence_length: 30
- n_features: 4
- hidden_units: 128
- dropout_rate: 0.2
- learning_rate: 0.001
- batch_size: 32
- epochs: 100
- num_lstm_layers: 2
- hidden_units_decay: 0.75
- patience: 20

DRL Parameters:
- state_dim: 4
- action_dim: 1
- hidden_dim: 128
- learning_rate: 0.001
- gamma: 0.99
- buffer_size: 100000
- batch_size: 64

## 📊 Dashboard Features

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

All parameters (holding cost, stockout penalty, ordering cost, lead time, target
service level) are adjustable from the sidebar and flow through every tab.

## 🧪 Testing

Run all tests locally:

    python run_tests.py

Or with pytest:

    pytest tests/

In Docker:

    docker-compose run tests

The pure-Python tests (data processing, inventory simulation, base-stock policy)
run without TensorFlow or PyTorch installed; the LSTM and DRL tests skip
gracefully when those libraries are unavailable.

## 📈 Performance

- Sales Forecast Accuracy: MAPE < 10%
- Inventory Cost Reduction: Up to 25%
- Training Time: ~5 minutes on CPU

## 🛠️ Development

Code formatting:
black .

Linting:
flake8

Type checking:
mypy .

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See LICENSE for more information.

## 🙏 Acknowledgments

- TensorFlow - Deep Learning Framework
- PyTorch - Reinforcement Learning
- MLflow - Experiment Tracking
- Streamlit - Interactive Dashboard
- Plotly - Charting
- Docker - Containerization
