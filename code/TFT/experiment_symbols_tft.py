import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
import random
from tqdm import tqdm
from data_preprocessing import load_stock_data, preprocess_data, inverse_transform_predictions
from evaluation import calculate_metrics, print_metrics
from simple_tft import create_simple_tft_model
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

# Configuration
STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_models')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_plots')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# OPTIMIZED Model parameters
SEQUENCE_LENGTH = 60
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'
EPOCHS = 50
BATCH_SIZE = 64  # Increased from 32 for better gradient estimates
HIDDEN_SIZE = 256  # Increased from 128 for more capacity
NUM_HEADS = 8  # Increased from 4 for better attention
NUM_LAYERS = 4  # Increased from 2 for deeper learning
DROPOUT_RATE = 0.15  # Reduced from 0.2 to prevent underfitting
INITIAL_LR = 0.002  # Increased from 0.001 for faster initial learning
WEIGHT_DECAY = 1e-5  # Added L2 regularization

def get_best_symbols(summary_path, n=32, metrics=['MAE', 'RMSE']):
    """
    Get the best performing symbols from the summary file
    
    Args:
        summary_path: Path to the summary CSV file
        n: Number of symbols to return
        metrics: List of metrics to use for ranking (lower is better)
        
    Returns:
        List of CSV file paths for the best performing symbols
    """
    # First try the specified summary path
    if not os.path.exists(summary_path):
        # If not found, try the regular summary.csv file
        alt_summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
        if os.path.exists(alt_summary_path):
            print(f"Using existing summary file: {alt_summary_path}")
            summary_path = alt_summary_path
        else:
            print(f"No summary files found. Will use random symbols.")
            return None
    
    try:
        # Load summary data
        summary_df = pd.read_csv(summary_path)
        
        # Calculate a combined score based on the specified metrics
        # Normalize each metric to 0-1 range and sum them
        summary_df['combined_score'] = 0
        for metric in metrics:
            if metric in summary_df.columns:
                # Normalize the metric (lower is better)
                min_val = summary_df[metric].min()
                max_val = summary_df[metric].max()
                if max_val > min_val:
                    summary_df[f'{metric}_norm'] = (summary_df[metric] - min_val) / (max_val - min_val)
                else:
                    summary_df[f'{metric}_norm'] = 0
                
                # Add to combined score
                summary_df['combined_score'] += summary_df[f'{metric}_norm']
        
        # Sort by combined score (lower is better)
        summary_df = summary_df.sort_values('combined_score')
        
        # Get the top n symbols
        best_symbols = summary_df['Symbol'].head(n).tolist()
        
        print(f"Selected {len(best_symbols)} best performing symbols based on {', '.join(metrics)}")
        
        # Map symbol names to file paths
        csv_files = []
        for symbol in best_symbols:
            region, sym = symbol.split('/')
            file_path = os.path.join(STOCK_DATA_DIR, region, f"{sym}.csv")
            if os.path.exists(file_path):
                csv_files.append(file_path)
        
        return csv_files
    
    except Exception as e:
        print(f"Error reading summary file: {str(e)}. Will use random symbols.")
        return None

def find_all_csv_files():
    """Find all CSV files in the stock data directory"""
    csv_files = []
    
    # Walk through all directories in stock_data
    for root, dirs, files in os.walk(STOCK_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cpu'):
    """Train the TFT model with advanced optimization"""
    start_time = time.time()

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Enhanced loss function
    criterion = torch.nn.SmoothL1Loss()

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze(-1)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())

        # Update learning rate
        scheduler.step()

        # Record losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.6f}')

    training_time = time.time() - start_time
    history['training_time'] = training_time

    return model, history

def train_and_evaluate(csv_file, device=None, multi_gpu=False):
    """Train and evaluate TFT model on a single stock CSV file"""
    # Extract stock symbol from filename
    symbol = os.path.basename(csv_file).split('.')[0]
    region = os.path.basename(os.path.dirname(csv_file))
    
    print(f"\n{'='*50}")
    print(f"Processing {region}/{symbol}")
    print(f"{'='*50}")
    
    try:
        # Load and preprocess data
        df = load_stock_data(csv_file)
        X_train, X_test, y_train, y_test, feature_scaler, target_scaler = preprocess_data(
            df, 
            target_col=TARGET_COL, 
            feature_cols=FEATURE_COLS, 
            sequence_length=SEQUENCE_LENGTH
        )
            
    except Exception as e:
        print(f"Error in preprocessing {region}/{symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'region': region,
            'error': f"Preprocessing failed: {str(e)}"
        }
    
    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Insufficient data for {region}/{symbol} after preprocessing")
        return {
            'symbol': symbol,
            'region': region,
            'error': "Insufficient data after preprocessing"
        }
    
    # Create SimpleTFT model (shared architecture)
    model = create_simple_tft_model(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    )
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Multi-GPU support
    if multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Train model
    start_time = time.time()
    trained_model, history = train_model(
        model, 
        X_train, y_train, 
        X_test, y_test,  # Using test set as validation for simplicity
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        device=device
    )
    training_time = time.time() - start_time
    
    # Make predictions
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    trained_model.eval()
    with torch.no_grad():
        y_pred = trained_model(X_test_tensor).squeeze(-1).cpu().numpy()
    
    # Inverse transform predictions and actual values
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), target_scaler)
    y_pred_inv = inverse_transform_predictions(y_pred.reshape(-1, 1), target_scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    metrics['training_time'] = training_time
    print_metrics(metrics, symbol)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{region}_{symbol}_tft_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'metrics': metrics
    }, model_path)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title(f'{region}/{symbol} Stock Price Prediction (TFT)')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{region}_{symbol}_tft_prediction.png")
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'symbol': symbol,
        'region': region,
        'metrics': metrics,
        'model_path': model_path,
        'plot_path': plot_path
    }

def run_experiment(device=None, multi_gpu=False):
    """Run experiment with different numbers of symbols, with device selection"""
    # Define the number of symbols to test
    symbol_counts = [1, 2, 4, 8, 16, 32]
    
    # Get all CSV files
    all_csv_files = find_all_csv_files()
    print(f"Found {len(all_csv_files)} CSV files in total")
    
    # Try to get best symbols from previous experiments or regular summary
    summary_path = os.path.join(RESULTS_DIR, 'summary.csv')
    best_csv_files = get_best_symbols(summary_path, n=max(symbol_counts))
    
    # If no best symbols found, use random selection
    if not best_csv_files:
        print("Using random symbol selection")
        random.shuffle(all_csv_files)
        best_csv_files = all_csv_files[:max(symbol_counts)]
    
    # Experiment results
    experiment_results = []
    iteration_details = []
    all_symbol_results = []
    
    # Run experiment for each symbol count
    cumulative_time = 0
    
    for count in symbol_counts:
        print(f"\n{'='*50}")
        print(f"Training with {count} symbols")
        print(f"{'='*50}")
        
        # Select subset of symbols
        csv_files = best_csv_files[:count]
        
        # Train models
        batch_results = []
        batch_start_time = time.time()
        
        for csv_file in tqdm(csv_files, desc=f"Training {count} symbols"):
            try:
                result = train_and_evaluate(csv_file, device=device, multi_gpu=multi_gpu)
                batch_results.append(result)
                all_symbol_results.append(result)
            except Exception as e:
                symbol = os.path.basename(csv_file).split('.')[0]
                region = os.path.basename(os.path.dirname(csv_file))
                print(f"Error processing {region}/{symbol}: {str(e)}")
                batch_results.append({
                    'symbol': symbol,
                    'region': region,
                    'error': str(e)
                })
        
        batch_time = time.time() - batch_start_time
        cumulative_time += batch_time
        
        # Record experiment results
        experiment_results.append({
            'symbol_count': count,
            'batch_time': batch_time,
            'cumulative_time': cumulative_time,
            'avg_time_per_symbol': batch_time / count
        })
        
        # Record detailed results for this iteration
        iteration_details.append({
            'symbol_count': count,
            'symbols': [f"{r['region']}/{r['symbol']}" for r in batch_results if 'error' not in r],
            'mae_values': [round(r['metrics']['mae'], 4) for r in batch_results if 'error' not in r],
            'rmse_values': [round(r['metrics']['rmse'], 4) for r in batch_results if 'error' not in r],
            'training_times': [round(r['metrics']['training_time'], 2) for r in batch_results if 'error' not in r],
            'avg_mae': round(np.mean([r['metrics']['mae'] for r in batch_results if 'error' not in r]), 4),
            'avg_rmse': round(np.mean([r['metrics']['rmse'] for r in batch_results if 'error' not in r]), 4),
            'total_time': round(batch_time, 2)
        })
        
        print(f"\nCompleted training {count} symbols")
        print(f"Batch time: {batch_time:.2f} seconds")
        print(f"Cumulative time: {cumulative_time:.2f} seconds")
        print(f"Average time per symbol: {batch_time/count:.2f} seconds")
    
    # Save experiment results
    experiment_df = pd.DataFrame(experiment_results)
    experiment_path = os.path.join(RESULTS_DIR, 'symbol_experiment_results.csv')
    experiment_df.to_csv(experiment_path, index=False)
    
    # Save detailed iteration results
    iteration_details_path = os.path.join(RESULTS_DIR, 'symbol_experiment_details.json')
    with open(iteration_details_path, 'w') as f:
        json.dump(iteration_details, f, indent=4)
    
    # Create summary of all trained models
    summary = []
    for result in all_symbol_results:
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
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create a table with iteration results
    iteration_table = []
    for detail in iteration_details:
        iteration_table.append({
            'Symbol Count': detail['symbol_count'],
            'Avg MAE': detail['avg_mae'],
            'Avg RMSE': detail['avg_rmse'],
            'Total Time (s)': detail['total_time'],
            'Avg Time per Symbol (s)': round(detail['total_time'] / detail['symbol_count'], 2)
        })
    
    # Save iteration table to CSV
    iteration_table_df = pd.DataFrame(iteration_table)
    iteration_table_path = os.path.join(RESULTS_DIR, 'symbol_experiment_table.csv')
    iteration_table_df.to_csv(iteration_table_path, index=False)
    
    # Create a plot of the experiment results
    plt.figure(figsize=(12, 8))
    
    # Plot training time vs number of symbols
    plt.subplot(2, 1, 1)
    plt.plot(experiment_df['symbol_count'], experiment_df['batch_time'], 'o-', linewidth=2, markersize=8)
    plt.title('Training Time vs Number of Symbols (TFT)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True)
    
    # Plot cumulative time
    plt.subplot(2, 1, 2)
    plt.plot(experiment_df['symbol_count'], experiment_df['cumulative_time'], 'o-', linewidth=2, markersize=8)
    plt.title('Cumulative Training Time (TFT)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Cumulative Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png'))
    
    print(f"\nExperiment completed. Results saved to {experiment_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Iteration details saved to {iteration_details_path}")
    print(f"Iteration table saved to {iteration_table_path}")
    print(f"Plot saved to {os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png')}")
    
    # Print experiment results
    print("\nExperiment Results:")
    print(iteration_table_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFT symbol experiments with device selection.")
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'multi-gpu'],
                        help='Device to use: cpu, cuda, or multi-gpu (DistributedDataParallel)')
    args = parser.parse_args()

    # Device selection logic
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'multi-gpu':
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda')
            multi_gpu = True
        else:
            print('Multi-GPU requested but only one GPU available. Using single GPU.')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            multi_gpu = False
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multi_gpu = False

    # Pass device to run_experiment
    run_experiment(device=device, multi_gpu=(args.device == 'multi-gpu' and torch.cuda.device_count() > 1))
