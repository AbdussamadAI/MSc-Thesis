#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

# NVTX import and dummy fallback for profiling
try:
    import torch.cuda.nvtx as nvtx
except ImportError:
    class DummyNvtx:
        @staticmethod
        def range_push(msg):
            pass
        @staticmethod
        def range_pop():
            pass
    nvtx = DummyNvtx()
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import random
from datetime import datetime
import argparse

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Constants
SEQUENCE_LENGTH = 60
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 50
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DATA_DIR = os.path.join(BASE_DIR, 'stock_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'bilstm_results')
MODELS_DIR = os.path.join(BASE_DIR, 'bilstm_models')
PLOTS_DIR = os.path.join(BASE_DIR, 'bilstm_plots')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class StockDataset(Dataset):
    def __init__(self, X, y):
        """Initialize dataset with CPU tensors"""
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return CPU tensors - DataLoader will handle device transfer
        return self.X[idx], self.y[idx]

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Initial hidden states - match device of input tensor
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layer
        out = self.fc(out)
        
        return out.squeeze(1)

def load_stock_data(csv_file):
    """Load and preprocess a single stock CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure Date column is datetime and sort
        if 'Date' in df.columns:
            # Convert to UTC datetime to handle timezone-aware datetimes
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.sort_values('Date')
        
        # Extract symbol from filename or Ticker column
        if 'Ticker' in df.columns:
            symbol = df['Ticker'].iloc[0]
        else:
            symbol = os.path.basename(csv_file).split('.')[0]
        
        # Extract region from directory structure
        region = os.path.basename(os.path.dirname(csv_file))
        
        # Add symbol column if not present
        if 'Ticker' not in df.columns:
            df['Ticker'] = symbol
            
        return df, symbol, region
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None, None, None

def preprocess_data(df):
    """Preprocess the combined stock data for model training"""
    # Convert Date to datetime if not already and handle timezone-aware datetimes
    if 'Date' in df.columns:
        # Convert to UTC datetime to handle timezone-aware datetimes
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Drop non-numeric columns except Date and Ticker
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_to_keep = ['Date', 'Ticker']
    cols_to_drop = [col for col in non_numeric_cols if col not in cols_to_keep]
    df = df.drop(columns=cols_to_drop)
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Fill missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Scale numeric features by symbol
    scaler = RobustScaler()
    grouped = df_processed.groupby('Ticker')
    
    scaled_dfs = []
    for name, group in grouped:
        # Scale numeric columns
        numeric_data = group[numeric_cols].values
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Create a new dataframe with scaled values
        scaled_group = group.copy()
        scaled_group[numeric_cols] = scaled_data
        scaled_dfs.append(scaled_group)
    
    # Combine all scaled dataframes
    df_scaled = pd.concat(scaled_dfs, ignore_index=True)
    
    # Sort by Date and Ticker
    df_scaled = df_scaled.sort_values(['Ticker', 'Date'])
    
    return df_scaled

def create_sequences(df, seq_length=SEQUENCE_LENGTH, target_col='Close'):
    """Create sequences for training"""
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Group by symbol
    grouped = df.groupby('Ticker')
    
    all_X = []
    all_y = []
    
    for symbol, group in grouped:
        # Sort by date
        group = group.sort_values('Date')
        
        # Get feature columns (exclude Date and Ticker)
        feature_cols = [col for col in group.columns if col not in ['Date', 'Ticker']]
        
        # Create sequences
        X, y = [], []
        data = group[feature_cols].values
        
        for i in range(len(data) - seq_length):
            # Use all features as input
            features = data[i:(i+seq_length)]
            target = data[i+seq_length, group.columns.get_loc(target_col) - 2]  # -2 to adjust for Date and Ticker
            
            X.append(features)
            y.append(target)
        
        all_X.extend(X)
        all_y.extend(y)
    
    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)

def train_model(model, train_loader, val_loader, device, epochs=EPOCHS):
    """Train the BiLSTM model"""
    if torch.cuda.is_available():
        nvtx.range_push("Training Loop Start")
        print("[NVTX] Training Loop Start")
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        if torch.cuda.is_available():
            nvtx.range_push(f"Epoch {epoch+1} Start")
            print(f"[NVTX] Epoch {epoch+1} Start")
        # Training
        model.train()
        train_losses = []
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_X, batch_y in train_iterator:
            if torch.cuda.is_available():
                nvtx.range_push("Batch Training Start")
            # Move tensors to the correct device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            if torch.cuda.is_available():
                nvtx.range_push("Forward Pass")
            optimizer.zero_grad()
            outputs = model(batch_X)
            if torch.cuda.is_available():
                nvtx.range_pop()  # End Forward Pass
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            if torch.cuda.is_available():
                nvtx.range_push("Backward Pass")
            loss.backward()
            if torch.cuda.is_available():
                nvtx.range_pop()  # End Backward Pass
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.cuda.is_available():
                nvtx.range_push("Optimizer Step")
            optimizer.step()
            if torch.cuda.is_available():
                nvtx.range_pop()  # End Optimizer Step
            
            train_losses.append(loss.item())
            if torch.cuda.is_available():
                nvtx.range_pop()  # End Batch Training
        
        # Validation
        if torch.cuda.is_available():
            nvtx.range_push("Validation Phase")
            print(f"[NVTX] Validation Phase Start (Epoch {epoch+1})")
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for batch_X, batch_y in val_iterator:
                if torch.cuda.is_available():
                    nvtx.range_push("Validation Batch")
                # Move tensors to the correct device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())
                if torch.cuda.is_available():
                    nvtx.range_pop()  # End Validation Batch
        if torch.cuda.is_available():
            nvtx.range_pop()  # End Validation Phase
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model state dict
            best_model_state = model.state_dict()
        
        # Record losses
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    if torch.cuda.is_available():
        nvtx.range_pop()  # End Training Loop
        print("[NVTX] Training Loop End")
    return model, history

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    if torch.cuda.is_available():
        nvtx.range_push("Evaluation Phase")
        print("[NVTX] Evaluation Phase Start")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            if torch.cuda.is_available():
                nvtx.range_push("Evaluation Batch")
            # Move tensors to the correct device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # Move to CPU for metric calculation
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
            if torch.cuda.is_available():
                nvtx.range_pop()  # End Evaluation Batch
    if torch.cuda.is_available():
        nvtx.range_pop()  # End Evaluation Phase
        print("[NVTX] Evaluation Phase End")
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, predictions, actuals

def find_all_csv_files():
    """Find all CSV files in the stock data directory"""
    csv_files = []
    
    # Walk through all directories in stock_data
    for root, dirs, files in os.walk(STOCK_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

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
    if not os.path.exists(summary_path):
        print(f"Summary file {summary_path} not found. Will use random symbols.")
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

def train_and_evaluate_symbols(csv_files, target_col='Close', device=None):
    """Train and evaluate BiLSTM model on multiple stock CSV files"""
    if torch.cuda.is_available():
        nvtx.range_push("Data Loading and Preprocessing")
        print("[NVTX] Data Loading and Preprocessing Start")
    start_time = time.time()
    
    # Load and concatenate all stock data
    dfs = []
    symbols = []
    regions = []
    
    for csv_file in csv_files:
        df, symbol, region = load_stock_data(csv_file)
        if df is not None:
            dfs.append(df)
            symbols.append(symbol)
            regions.append(region)
    
    if not dfs:
        raise ValueError("No valid stock data found")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    # Preprocess data
    df_processed = preprocess_data(combined_df)
    
    # Create sequences
    X, y = create_sequences(df_processed, seq_length=SEQUENCE_LENGTH, target_col=target_col)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    # Create dataloaders with pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # No multiprocessing to avoid issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # No multiprocessing to avoid issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # No multiprocessing to avoid issues
        pin_memory=True if device.type == 'cuda' else False
    )
    if torch.cuda.is_available():
        nvtx.range_pop()  # End Data Loading and Preprocessing
        print("[NVTX] Data Loading and Preprocessing End")
    
    # Create model
    input_size = X_train.shape[2]  # Number of features
    model = BiLSTMModel(input_size=input_size)
    
    # Train model
    print(f"Starting model training for {len(symbols)} symbols...")
    
    training_start_time = time.time()
    model, history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS)
    training_time = time.time() - training_start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model on test data...")
    metrics, predictions, actuals = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    # Add training time to metrics
    metrics['training_time'] = training_time
    
    # Total time
    total_time = time.time() - start_time
    metrics['total_time'] = total_time
    
    # Create a directory for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(MODELS_DIR, f"symbols_{len(symbols)}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(experiment_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'history.npy')
    np.save(history_path, history)
    
    # Save metrics - convert NumPy values to Python native types for JSON serialization
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
            metrics_json[k] = float(v)  # Convert NumPy types to Python float
        else:
            metrics_json[k] = v
    
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save symbols used
    symbols_path = os.path.join(experiment_dir, 'symbols.json')
    with open(symbols_path, 'w') as f:
        json.dump({
            'symbols': [f"{r}/{s}" for r, s in zip(regions, symbols)],
            'count': len(symbols)
        }, f, indent=4)
    
    return {
        'metrics': metrics,
        'symbols': symbols,
        'regions': regions,
        'model_path': model_path,
        'history_path': history_path,
        'metrics_path': metrics_path
    }

def ddp_run_experiment(rank, world_size):
    # Device selection based on argument
    if torch.cuda.is_available():
        nvtx.range_push(f"DDP Setup (Rank {rank})")
        print(f"[NVTX] DDP Setup (Rank {rank})")
    device = torch.device('cuda', rank)
    print(f"Using GPU: {torch.cuda.get_device_name(rank)}")
    
    # Symbol counts to experiment with
    symbol_counts = [2, 4, 8, 16, 32]
    
    # Find all CSV files
    all_csv_files = find_all_csv_files()
    print(f"Found {len(all_csv_files)} CSV files")
    
    # Check if we have a previous summary to use for selecting best symbols
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    if os.path.exists(summary_path):
        # Get the best symbols based on previous results
        best_csv_files = get_best_symbols(summary_path, n=max(symbol_counts))
        if best_csv_files and len(best_csv_files) >= max(symbol_counts):
            csv_files = best_csv_files
            print(f"Using {len(csv_files)} best performing symbols from previous experiments")
        else:
            # Randomly select files if we don't have enough best symbols
            csv_files = random.sample(all_csv_files, min(max(symbol_counts), len(all_csv_files)))
            print(f"Randomly selected {len(csv_files)} symbols for experiments")
    else:
        # Randomly select files if no previous summary exists
        csv_files = random.sample(all_csv_files, min(max(symbol_counts), len(all_csv_files)))
        print(f"Randomly selected {len(csv_files)} symbols for experiments")
    
    # Results to track
    experiment_results = []
    iteration_details = []
    
    # Run experiments for each symbol count
    cumulative_time = 0
    
    for count in symbol_counts:
        print(f"\n{'='*50}")
        print(f"Running experiment with {count} symbols")
        print(f"{'='*50}")
        
        # Select subset of CSV files for this experiment
        batch_files = csv_files[:count]
        
        # Train and evaluate
        batch_start_time = time.time()
        result = train_and_evaluate_symbols(batch_files, device=device)
        batch_time = time.time() - batch_start_time
        
        # Update cumulative time
        cumulative_time += batch_time
        
        # Record results - convert NumPy values to Python native types
        experiment_results.append({
            'symbol_count': count,
            'batch_time': float(batch_time),
            'cumulative_time': float(cumulative_time),
            'mae': float(result['metrics']['mae']),
            'rmse': float(result['metrics']['rmse']),
            'r2': float(result['metrics']['r2'])
        })
        
        # Record detailed results for this iteration
        symbols_with_regions = [f"{r}/{s}" for r, s in zip(result['regions'], result['symbols'])]
        iteration_details.append({
            'symbol_count': count,
            'symbols': symbols_with_regions,
            'mae': round(float(result['metrics']['mae']), 4),
            'rmse': round(float(result['metrics']['rmse']), 4),
            'training_time': round(float(result['metrics']['training_time']), 2),
            'total_time': round(float(batch_time), 2)
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
    
    # Create a table with iteration results
    iteration_table = []
    for detail in iteration_details:
        iteration_table.append({
            'Symbol Count': detail['symbol_count'],
            'MAE': detail['mae'],
            'RMSE': detail['rmse'],
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
    plt.title('Training Time vs Number of Symbols (BiLSTM)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True)
    
    # Plot cumulative time
    plt.subplot(2, 1, 2)
    plt.plot(experiment_df['symbol_count'], experiment_df['cumulative_time'], 'o-', linewidth=2, markersize=8)
    plt.title('Cumulative Training Time (BiLSTM)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Cumulative Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png'))
    
    print(f"\nExperiment completed. Results saved to {experiment_path}")
    print(f"Iteration details saved to {iteration_details_path}")
    print(f"Iteration table saved to {iteration_table_path}")
    print(f"Plot saved to {os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png')}")
    
    # Print experiment results
    print("\nExperiment Results:")
    print(iteration_table_df.to_string(index=False))

def run_experiment(device_type="auto"):
    """Run experiment with different numbers of symbols"""
   
    # Device selection based on argument
    if device_type == "gpu":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU requested but CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
    elif device_type == "cpu":
        device = torch.device('cpu')
        print("Using CPU as requested.")
    elif device_type == "multi-gpu":
        if torch.cuda.device_count() > 1:
            world_size = torch.cuda.device_count()
            print(f"Using multi-GPU with {world_size} GPUs via DDP")
            def ddp_main(rank, world_size):
                ddp_run_experiment(rank, world_size)
            mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)
            return
        else:
            print("Multi-GPU requested but only one GPU is available. Falling back to single GPU.")
            device = torch.device('cuda')
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Running on CPU only.")
    # Symbol counts to experiment with
    symbol_counts = [2, 4, 8, 16, 32]
    
    # Find all CSV files
    all_csv_files = find_all_csv_files()
    print(f"Found {len(all_csv_files)} CSV files")
    
    # Check if we have a previous summary to use for selecting best symbols
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    if os.path.exists(summary_path):
        # Get the best symbols based on previous results
        best_csv_files = get_best_symbols(summary_path, n=max(symbol_counts))
        if best_csv_files and len(best_csv_files) >= max(symbol_counts):
            csv_files = best_csv_files
            print(f"Using {len(csv_files)} best performing symbols from previous experiments")
        else:
            # Randomly select files if we don't have enough best symbols
            csv_files = random.sample(all_csv_files, min(max(symbol_counts), len(all_csv_files)))
            print(f"Randomly selected {len(csv_files)} symbols for experiments")
    else:
        # Randomly select files if no previous summary exists
        csv_files = random.sample(all_csv_files, min(max(symbol_counts), len(all_csv_files)))
        print(f"Randomly selected {len(csv_files)} symbols for experiments")
    
    # Results to track
    experiment_results = []
    iteration_details = []
    
    # Run experiments for each symbol count
    cumulative_time = 0
    
    for count in symbol_counts:
        print(f"\n{'='*50}")
        print(f"Running experiment with {count} symbols")
        print(f"{'='*50}")
        
        # Select subset of CSV files for this experiment
        batch_files = csv_files[:count]
        
        # Train and evaluate
        batch_start_time = time.time()
        result = train_and_evaluate_symbols(batch_files, device=device)
        batch_time = time.time() - batch_start_time
        
        # Update cumulative time
        cumulative_time += batch_time
        
        # Record results - convert NumPy values to Python native types
        experiment_results.append({
            'symbol_count': count,
            'batch_time': float(batch_time),
            'cumulative_time': float(cumulative_time),
            'mae': float(result['metrics']['mae']),
            'rmse': float(result['metrics']['rmse']),
            'r2': float(result['metrics']['r2'])
        })
        
        # Record detailed results for this iteration
        symbols_with_regions = [f"{r}/{s}" for r, s in zip(result['regions'], result['symbols'])]
        iteration_details.append({
            'symbol_count': count,
            'symbols': symbols_with_regions,
            'mae': round(float(result['metrics']['mae']), 4),
            'rmse': round(float(result['metrics']['rmse']), 4),
            'training_time': round(float(result['metrics']['training_time']), 2),
            'total_time': round(float(batch_time), 2)
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
    
    # Create a table with iteration results
    iteration_table = []
    for detail in iteration_details:
        iteration_table.append({
            'Symbol Count': detail['symbol_count'],
            'MAE': detail['mae'],
            'RMSE': detail['rmse'],
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
    plt.title('Training Time vs Number of Symbols (BiLSTM)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Training Time (seconds)')
    plt.grid(True)
    
    # Plot cumulative time
    plt.subplot(2, 1, 2)
    plt.plot(experiment_df['symbol_count'], experiment_df['cumulative_time'], 'o-', linewidth=2, markersize=8)
    plt.title('Cumulative Training Time (BiLSTM)')
    plt.xlabel('Number of Symbols')
    plt.ylabel('Cumulative Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png'))
    
    print(f"\nExperiment completed. Results saved to {experiment_path}")
    print(f"Iteration details saved to {iteration_details_path}")
    print(f"Iteration table saved to {iteration_table_path}")
    print(f"Plot saved to {os.path.join(PLOTS_DIR, 'symbol_experiment_plot.png')}")
    
    # Print experiment results
    print("\nExperiment Results:")
    print(iteration_table_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BiLSTM symbol experiment with device selection.")
    parser.add_argument('--device', type=str, default="auto", choices=["auto", "cpu", "gpu", "multi-gpu"],
                        help="Device to use: 'cpu', 'gpu', 'multi-gpu', or 'auto' (default: auto)")
    args = parser.parse_args()
    run_experiment(device_type=args.device)
