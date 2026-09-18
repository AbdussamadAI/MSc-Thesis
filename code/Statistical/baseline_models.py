import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """Comprehensive baseline models for time series forecasting"""
    
    def __init__(self):
        self.models = {}
        self.fitted_models = {}
        
    def naive_forecast(self, train_data, test_length):
        """Last value carry-forward"""
        last_value = train_data.iloc[-1]
        predictions = np.full(test_length, last_value)
        return predictions
    
    def moving_average_forecast(self, train_data, test_length, window=30):
        """Simple moving average forecast"""
        ma_value = train_data.rolling(window=window).mean().iloc[-1]
        predictions = np.full(test_length, ma_value)
        return predictions
    
    def exponential_smoothing_forecast(self, train_data, test_length, alpha=0.3):
        """Exponential smoothing forecast"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
        fitted_model = model.fit(smoothing_level=alpha)
        predictions = fitted_model.forecast(steps=test_length)
        return predictions.values
    
    def arima_forecast(self, train_data, test_length, order=(1,1,1)):
        """ARIMA model forecast with adaptive differencing"""
        try:
            # If order has d as None, determine it
            if len(order) == 3 and order[1] is None:
                d = determine_differencing_order(train_data)
                order = (order[0], d, order[2])

            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=test_length)
            return predictions.values
        except Exception as e:
            print(f"ARIMA failed: {e}, using linear trend")
            return self.linear_trend_forecast(train_data, test_length)
    
    def auto_arima_forecast(self, train_data, test_length, seasonal=False):
        """Auto-ARIMA with automatic parameter selection"""
        try:
            model = auto_arima(
                train_data, 
                seasonal=seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5, max_q=5, max_d=2
            )
            predictions = model.predict(n_periods=test_length)
            return predictions
        except Exception as e:
            print(f"Auto-ARIMA failed: {e}, using ARIMA(1,1,1)")
            return self.arima_forecast(train_data, test_length, order=(1,1,1))
    
    def linear_trend_forecast(self, train_data, test_length):
        """Linear trend extrapolation"""
        from sklearn.linear_model import LinearRegression

        X = np.arange(len(train_data)).reshape(-1, 1)
        y = train_data.values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(train_data), len(train_data) + test_length).reshape(-1, 1)
        predictions = model.predict(future_X)
        return predictions

    def garch_forecast(self, train_data, test_length):
        """GARCH-based forecast using historical volatility"""
        try:
            returns = train_data.pct_change().dropna()
            if len(returns) == 0:
                return self.linear_trend_forecast(train_data, test_length)

            mean_return = returns.mean()
            last_price = train_data.iloc[-1]

            # Deterministic forecast using mean return
            predictions = [last_price * (1 + mean_return) ** (i + 1) for i in range(test_length)]
            return np.array(predictions)
        except Exception as e:
            print(f"GARCH failed: {e}, using linear trend")
            return self.linear_trend_forecast(train_data, test_length)

    def sarima_forecast(self, train_data, test_length, order=(1,1,1), seasonal_order=(1,0,1,5)):
        """SARIMA model forecast with seasonal components"""
        try:
            # Determine d adaptively
            if order[1] is None:
                d = determine_differencing_order(train_data)
                order = (order[0], d, order[2])

            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            predictions = fitted_model.forecast(steps=test_length)
            return predictions.values
        except Exception as e:
            print(f"SARIMA failed: {e}, using ARIMA")
            return self.arima_forecast(train_data, test_length, order)
    
    def seasonal_naive_forecast(self, train_data, test_length, season_length=252):
        """Seasonal naive forecast (using previous year's pattern)"""
        if len(train_data) < season_length:
            return self.naive_forecast(train_data, test_length)
        
        seasonal_pattern = train_data.iloc[-season_length:].values
        predictions = []
        
        for i in range(test_length):
            seasonal_index = i % season_length
            predictions.append(seasonal_pattern[seasonal_index])
        
        return np.array(predictions)
    
    def evaluate_all_baselines(self, train_data, test_data):
        """Evaluate all baseline models"""
        test_length = len(test_data)
        results = {}
        
        # Define baseline methods
        baseline_methods = {
            'exponential_smoothing': lambda: self.exponential_smoothing_forecast(train_data, test_length),
            'arima_111': lambda: self.arima_forecast(train_data, test_length, (1, None, 1)),
            'sarima': lambda: self.sarima_forecast(train_data, test_length, (1, None, 1), (1, 0, 1, 5)),
            'auto_arima': lambda: self.auto_arima_forecast(train_data, test_length),
            'linear_trend': lambda: self.linear_trend_forecast(train_data, test_length),
            'garch': lambda: self.garch_forecast(train_data, test_length)
        }
        
        # Evaluate each method
        for method_name, method_func in baseline_methods.items():
            try:
                predictions = method_func()
                
                # Calculate metrics
                mae = mean_absolute_error(test_data, predictions)
                mse = mean_squared_error(test_data, predictions)
                rmse = np.sqrt(mse)
                
                # Directional accuracy
                actual_direction = np.sign(np.diff(test_data.values))
                pred_direction = np.sign(np.diff(predictions))
                directional_accuracy = np.mean(actual_direction == pred_direction)
                
                results[method_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'directional_accuracy': directional_accuracy,
                    'predictions': predictions
                }
                
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                results[method_name] = {
                    'mae': np.inf,
                    'mse': np.inf,
                    'rmse': np.inf,
                    'directional_accuracy': 0.0,
                    'error': str(e)
                }
        
        return results

def check_stationarity(data, significance_level=0.05):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(data.dropna())

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }

def determine_differencing_order(data, max_d=2, significance_level=0.05):
    """Determine the differencing order (d) for ARIMA based on stationarity"""
    d = 0
    for i in range(max_d + 1):
        if i == 0:
            series = data.dropna()
        else:
            series = data.diff(i).dropna()

        if len(series) < 10:  # Not enough data
            break

        result = adfuller(series)
        if result[1] < significance_level:
            d = i
            break

    return d

def prepare_data_for_baselines(df, target_col='Close', train_ratio=0.8):
    """Prepare data for baseline model evaluation"""
    # Ensure data is sorted by date
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    # Extract target series
    target_series = df[target_col].dropna()
    
    # Split into train/test
    split_idx = int(len(target_series) * train_ratio)
    train_data = target_series.iloc[:split_idx]
    test_data = target_series.iloc[split_idx:]
    
    return train_data, test_data

def run_baseline_comparison(csv_file, target_col='Close'):
    """Run comprehensive baseline comparison for a single stock"""
    from data_preprocessing import load_stock_data
    
    # Load data
    df = load_stock_data(csv_file)
    train_data, test_data = prepare_data_for_baselines(df, target_col)
    
    # Check stationarity
    stationarity_result = check_stationarity(train_data)
    
    # Run baseline evaluation
    baseline_models = BaselineModels()
    results = baseline_models.evaluate_all_baselines(train_data, test_data)
    
    # Add metadata
    results['metadata'] = {
        'symbol': csv_file.split('/')[-1].replace('.csv', ''),
        'train_size': len(train_data),
        'test_size': len(test_data),
        'stationarity': stationarity_result
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    csv_file = "stock_data/US/AAPL.csv"
    results = run_baseline_comparison(csv_file)
    
    # Print results
    print(f"Baseline Results for {results['metadata']['symbol']}:")
    print(f"Data: {results['metadata']['train_size']} train, {results['metadata']['test_size']} test")
    print(f"Stationary: {results['metadata']['stationarity']['is_stationary']}")
    print("\nModel Performance:")
    
    for model_name, metrics in results.items():
        if model_name != 'metadata' and 'error' not in metrics:
            print(f"{model_name:20s} MAE: {metrics['mae']:.4f} RMSE: {metrics['rmse']:.4f} Dir.Acc: {metrics['directional_accuracy']:.3f}")
