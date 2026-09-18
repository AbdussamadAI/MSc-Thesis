import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing MAE, MSE, and RMSE
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def print_metrics(metrics, symbol):
    """
    Print evaluation metrics in a formatted way
    
    Args:
        metrics: Dictionary containing metrics
        symbol: Stock symbol
    """
    print(f"\nEvaluation metrics for {symbol}:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print("-" * 40)
