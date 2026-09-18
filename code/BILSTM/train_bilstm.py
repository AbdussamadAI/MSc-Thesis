import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time

from data_preprocessing import load_stock_data, preprocess_data, inverse_transform_predictions
from model import create_bilstm_model, train_model
from evaluation import calculate_metrics, print_metrics

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model parameters
SEQUENCE_LENGTH = 60
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'
EPOCHS = 50
BATCH_SIZE = 32
LSTM_UNITS = 50
DROPOUT_RATE = 0.2

def find_all_csv_files():
    """Find all CSV files in the stock data directory"""
    csv_files = []
    
    # Walk through all directories in stock_data
    for root, dirs, files in os.walk(STOCK_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def train_and_evaluate(csv_file):
    """Train and evaluate BiLSTM model on a single stock CSV file"""
    # Extract stock symbol from filename
    symbol = os.path.basename(csv_file).split('.')[0]
    region = os.path.basename(os.path.dirname(csv_file))
    
    print(f"\n{'='*50}")
    print(f"Processing {region}/{symbol}")
    print(f"{'='*50}")
    
    # Load and preprocess data
    df = load_stock_data(csv_file)
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler, test_dates = preprocess_data(
        df, 
        target_col=TARGET_COL, 
        feature_cols=FEATURE_COLS, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Create and train model
    model = create_bilstm_model(
        input_size=X_train.shape[2],  # Number of features
        hidden_size=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE
    )
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    start_time = time.time()
    trained_model, history = train_model(
        model, 
        X_train, y_train, 
        X_test, y_test, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
    training_time = time.time() - start_time
    
    # Make predictions with PyTorch model
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    trained_model.eval()
    with torch.no_grad():
        y_pred = trained_model(X_test_tensor).cpu().numpy()
    
    # Inverse transform predictions and actual values
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), target_scaler)
    y_pred_inv = inverse_transform_predictions(y_pred, target_scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    metrics['training_time'] = training_time
    print_metrics(metrics, symbol)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{region}_{symbol}_model.pt")
    torch.save(trained_model.state_dict(), model_path)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_inv, label='Actual')
    plt.plot(test_dates, y_pred_inv, label='Predicted')
    plt.title(f'{region}/{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Format the date on x-axis
    plt.gcf().autofmt_xdate()
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{region}_{symbol}_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'symbol': symbol,
        'region': region,
        'metrics': metrics,
        'model_path': model_path,
        'plot_path': plot_path
    }

def main():
    """Main function to train models on all stock CSV files"""
    csv_files = find_all_csv_files()
    print(f"Found {len(csv_files)} CSV files to process")
    
    results = []
    
    for csv_file in tqdm(csv_files, desc="Training models"):
        try:
            result = train_and_evaluate(csv_file)
            results.append(result)
        except Exception as e:
            symbol = os.path.basename(csv_file).split('.')[0]
            region = os.path.basename(os.path.dirname(csv_file))
            print(f"Error processing {region}/{symbol}: {str(e)}")
            results.append({
                'symbol': symbol,
                'region': region,
                'error': str(e)
            })
    
    # Save all results to JSON
    results_path = os.path.join(RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create summary table
    summary = []
    for result in results:
        if 'error' in result:
            continue
        
        summary.append({
            'Symbol': f"{result['region']}/{result['symbol']}",
            'MAE': round(result['metrics']['mae'], 4),
            'MSE': round(result['metrics']['mse'], 4),
            'RMSE': round(result['metrics']['rmse'], 4),
            'Training Time (s)': round(result['metrics']['training_time'], 2)
        })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nTraining completed. Results saved to {results_path}")
    print(f"Summary saved to {summary_path}")
    
    # Print average metrics
    print("\nAverage Metrics:")
    print(f"MAE: {summary_df['MAE'].mean():.4f}")
    print(f"MSE: {summary_df['MSE'].mean():.4f}")
    print(f"RMSE: {summary_df['RMSE'].mean():.4f}")
    print(f"Avg. Training Time: {summary_df['Training Time (s)'].mean():.2f} seconds")

if __name__ == "__main__":
    main()
