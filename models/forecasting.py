import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import mlflow
import plotly.graph_objects as go
import tempfile
import os

class SalesForecaster:
    def __init__(self, config, mlflow_manager=None):
        self.config = config
        self.mlflow_manager = mlflow_manager
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layer
        inputs = Input(shape=self.config["input_shape"])
        x = inputs
        
        # First LSTM layer
        x = LSTM(
            self.config["hidden_units"],
            return_sequences=self.config["num_lstm_layers"] > 1
        )(x)
        x = Dropout(self.config["dropout_rate"])(x)
        
        # Middle LSTM layers
        for i in range(1, self.config["num_lstm_layers"] - 1):
            hidden_units = max(16, int(self.config["hidden_units"] * (self.config["hidden_units_decay"] ** i)))
            x = LSTM(hidden_units, return_sequences=True)(x)
            x = Dropout(self.config["dropout_rate"])(x)
        
        # Last LSTM layer (if more than one layer)
        if self.config["num_lstm_layers"] > 1:
            hidden_units = max(16, int(self.config["hidden_units"] * 
                             (self.config["hidden_units_decay"] ** (self.config["num_lstm_layers"] - 1))))
            x = LSTM(hidden_units)(x)
            x = Dropout(self.config["dropout_rate"])(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
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
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    f.write("\n".join(model_summary))
                    summary_path = f.name
                mlflow.log_artifact(summary_path)
                os.unlink(summary_path)  # Clean up the temporary file

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.get("patience", 20),
                    restore_best_weights=True,
                    monitor='val_loss',
                    mode='min'
                ),
                MLflowCallback()
            ]
        )
        
        # Log final model and metrics
        y_pred = self.predict(X_val)
        val_metrics = self.evaluate(X_val, y_val)
        mlflow.log_metrics(val_metrics)
        self.mlflow_manager.log_model(self.model, X_sample=X_train[:1], y_sample=y_train[:1])
        
        # Log training history plot using tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
            fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))
            fig.write_html(f.name)
            mlflow.log_artifact(f.name)
            os.unlink(f.name)  # Clean up the temporary file
        
        return history
    
    def _train_without_mlflow(self, X_train, y_train, X_val, y_val):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.get("patience", 20),
                    restore_best_weights=True,
                    monitor='val_loss',
                    mode='min'
                )
            ]
        )
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-7))) * 100  # Add small constant to avoid division by zero
        }
        return metrics 