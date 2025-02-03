import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.scalers = {}
        
    def prepare_data(self, df):
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the correct number of features
        expected_features = ['sales', 'stock', 'price', 'lead_time']
        for feature in expected_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Scale features
        for column in df.columns:
            self.scalers[column] = MinMaxScaler()
            df[column] = self.scalers[column].fit_transform(df[[column]]).flatten()
            
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            X.append(df.iloc[i:(i + self.sequence_length)].values)
            y.append(df.iloc[i + self.sequence_length]['sales'])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data, column):
        return self.scalers[column].inverse_transform(data.reshape(-1, 1))
    
    def split_data(self, X, y, train_ratio=0.8):
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Further split training data into train and validation
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        return X_train, y_train, X_val, y_val, X_test, y_test 