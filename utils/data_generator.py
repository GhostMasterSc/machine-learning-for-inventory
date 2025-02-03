import pandas as pd
import numpy as np

def generate_sample_data(start_date='2023-01-01', periods=365):
    """Generate sample inventory data"""
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Base demand with seasonality and trend
    t = np.arange(periods)
    seasonal = 20 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
    trend = 0.1 * t  # Upward trend
    base_demand = 100 + trend + seasonal
    
    # Add random noise
    np.random.seed(42)
    noise = np.random.normal(0, 10, periods)
    demand = base_demand + noise
    
    # Generate other features
    data = pd.DataFrame({
        'sales': np.maximum(demand, 0),  # Ensure non-negative sales
        'stock': np.random.uniform(80, 120, periods),
        'price': np.random.uniform(90, 110, periods),
        'lead_time': np.random.randint(1, 5, periods).astype(float)  # Convert to float for consistent scaling
    }, index=dates)
    
    # Verify data shape
    assert len(data.columns) == 4, f"Expected 4 features, got {len(data.columns)}"
    return data 