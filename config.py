from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model parameters
LSTM_PARAMS = {
    "sequence_length": 30,
    "n_features": 4,
    "hidden_units": 128,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "input_shape": (30, 4),  # (sequence_length, n_features)
    "num_lstm_layers": 2,  # Number of LSTM layers
    "hidden_units_decay": 0.75,  # Factor to reduce hidden units in deeper layers
    "patience": 20  # Early stopping patience
}

DRL_PARAMS = {
    "state_dim": 4,
    "action_dim": 1,
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "buffer_size": 100000,
    "batch_size": 64
}

# Training parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Inventory parameters
HOLDING_COST = 0.1
STOCKOUT_COST = 5.0
ORDERING_COST = 2.0
LEAD_TIME = 3 