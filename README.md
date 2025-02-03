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
- [Contact](#-contact)
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
- **Deep Reinforcement Learning**: Advanced DRL agent for inventory decisions
- **Base Stock Policy**: Traditional inventory management baseline
- **Cost Analysis**: Detailed breakdown of holding, stockout, and ordering costs
- **Performance Comparison**: DRL vs Base Stock policy evaluation

## 🚀 Quick Start

### Prerequisites
- Docker
- Docker Compose
- Git

### Installation
Clone the repository:
git clone https://github.com/yourusername/inventory-management-ml.git
cd inventory-management-ml

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
├── app/                    # Dashboard and UI components
│   ├── dashboard.py       # Main Streamlit interface
│   └── components/        # Reusable UI components
├── models/                # ML model implementations
│   ├── forecasting.py    # LSTM sales forecaster
│   └── optimization.py   # DRL inventory optimizer
├── utils/                # Helper functions
│   ├── data_processor.py # Data preprocessing
│   └── mlflow_manager.py # MLflow utilities
├── tests/               # Unit tests
├── config.py           # System configuration
├── docker-compose.yml  # Docker configuration
└── README.md          # Documentation

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

1. Model Configuration
   - Adjust LSTM architecture
   - Configure training parameters
   - Set optimization constraints

2. Training Interface
   - Real-time training progress
   - Performance metrics visualization
   - Model comparison tools

3. Results Analysis
   - Sales forecasting accuracy
   - Inventory level optimization
   - Cost breakdown analysis

## 🧪 Testing

Run all tests:
docker-compose run tests

Run specific test:
docker-compose run tests pytest tests/test_forecasting.py

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

## 📧 Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/inventory-management-ml

## 🙏 Acknowledgments

- TensorFlow - Deep Learning Framework
- MLflow - Experiment Tracking
- Streamlit - Interactive Dashboard
- Docker - Containerization

For detailed documentation of each component, please refer to the docs/ directory in the repository. 