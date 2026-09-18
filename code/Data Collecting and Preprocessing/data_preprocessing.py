import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_stock_data(csv_file):
    """Loads stock data from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return pd.DataFrame()

def preprocess_data(df, target_col='Close', feature_cols=None, sequence_length=60):
    """
    Preprocess stock data for LSTM model
    
    Args:
        df: DataFrame containing stock data
        target_col: Column to predict
        feature_cols: List of columns to use as features
        sequence_length: Number of time steps to use for sequence
        
    Returns:
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler
    """
    # Default feature columns if none provided
    if feature_cols is None:
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Ensure all required columns exist
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
    
    # Select relevant columns and drop rows with NaN values
    data = df[feature_cols].copy()  # Only use feature columns for input data
    target = df[target_col].copy()  # Target column for prediction
    
    # Handle any remaining NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
    target = target.fillna(method='ffill').fillna(method='bfill')
    
    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(data)
    
    # Scale the target separately
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_target[i + sequence_length, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Ensure we have a Date column
    if 'Date' not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column for plotting with dates")
    
    # Create a list of dates corresponding to each sequence
    dates = []
    for i in range(len(scaled_features) - sequence_length):
        # The date for a sequence is the date of the target (last element + 1)
        dates.append(df['Date'].iloc[i + sequence_length])
    
    dates = np.array(dates)
    
    # Split into train and test sets (80% train, 20% test)
    split_result = train_test_split(X, y, dates, test_size=0.2, shuffle=False)
    X_train, X_test, y_train, y_test, train_dates, test_dates = split_result
    
    return X_train, X_test, y_train, y_test, feature_scaler, target_scaler

def inverse_transform_predictions(predictions, target_scaler, feature_cols=None, target_col=None):
    """
    Inverse transform scaled predictions back to original scale
    
    Args:
        predictions: Scaled predictions
        target_scaler: Fitted scaler used for scaling the target variable
        feature_cols: Not used in this version, kept for backward compatibility
        target_col: Not used in this version, kept for backward compatibility
        
    Returns:
        Predictions in original scale
    """
    # Reshape predictions to 2D array if needed
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Inverse transform directly using the target scaler
    return target_scaler.inverse_transform(predictions).flatten()
