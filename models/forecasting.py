import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import mlflow
import plotly.graph_objects as go

class SalesForecaster:
    def __init__(self, config, mlflow_manager=None):
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.model = self._build_model()
        
    def _build_model(self):
        layers = []
        
        # First LSTM layer
        layers.append(LSTM(
            self.config["hidden_units"],
            input_shape=self.config["input_shape"],
            return_sequences=self.config["num_lstm_layers"] > 1
        ))
        layers.append(Dropout(self.config["dropout_rate"]))
        
        # Middle LSTM layers
        for i in range(1, self.config["num_lstm_layers"] - 1):
            hidden_units = int(self.config["hidden_units"] * (self.config["hidden_units_decay"] ** i))
            layers.append(LSTM(hidden_units, return_sequences=True))
            layers.append(Dropout(self.config["dropout_rate"]))
        
        # Last LSTM layer (if more than one layer)
        if self.config["num_lstm_layers"] > 1:
            hidden_units = int(self.config["hidden_units"] * 
                             (self.config["hidden_units_decay"] ** (self.config["num_lstm_layers"] - 1)))
            layers.append(LSTM(hidden_units))
            layers.append(Dropout(self.config["dropout_rate"]))
        
        # Output layer
        layers.append(Dense(1))
        
        model = Sequential(layers)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        if self.mlflow_manager:
            with self.mlflow_manager.start_run():
                self.mlflow_manager.log_params(self.config)
                history = self._train_with_mlflow(X_train, y_train, X_val, y_val)
        else:
            history = self._train_without_mlflow(X_train, y_train, X_val, y_val)
        return history
    
    def _train_with_mlflow(self, X_train, y_train, X_val, y_val):
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metrics({
                        "train_loss": logs.get("loss", 0),
                        "val_loss": logs.get("val_loss", 0),
                        "train_mae": logs.get("mae", 0),
                        "val_mae": logs.get("val_mae", 0)
                    }, step=epoch)
                    
            def on_train_begin(self, logs=None):
                # Log model summary as text artifact
                model_summary = []
                self.model.summary(print_fn=lambda x: model_summary.append(x))
                with open("model_summary.txt", "w") as f:
                    f.write("\n".join(model_summary))
                mlflow.log_artifact("model_summary.txt")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                MLflowCallback()
            ]
        )
        
        # Log final model and metrics
        y_pred = self.predict(X_val)
        val_metrics = self.evaluate(X_val, y_val)
        mlflow.log_metrics(val_metrics)
        self.mlflow_manager.log_model(self.model, X_sample=X_train[:1], y_sample=y_train[:1])
        
        # Log training history plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))
        fig.write_html("training_history.html")
        mlflow.log_artifact("training_history.html")
        
        return history
    
    def _train_without_mlflow(self, X_train, y_train, X_val, y_val):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        return metrics 