#!/usr/bin/env python3
"""
Hybrid SimpleTFT implementation: 2 stocks per stream
Each stream processes 2 symbols sequentially before moving to next batch
Always distributes symbols evenly across all GPUs
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import time
import random
from tqdm import tqdm
from simple_tft import create_simple_tft_model

try:
    from numba import cuda
    import ctypes
    NUMBA_AVAILABLE = True
except ImportError:
    print("⚠️  Numba not available - CUDA streams will be disabled")
    NUMBA_AVAILABLE = False
    # Create dummy classes/functions for compatibility
    class DummyCuda:
        @staticmethod
        def stream():
            return DummyStream()
    class DummyStream:
        def handle(self):
            return type('handle', (), {'value': 0})()
        def synchronize(self):
            pass
        def close(self):
            pass
    cuda = DummyCuda()

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

from data_preprocessing import load_stock_data, preprocess_data, inverse_transform_predictions
from evaluation import calculate_metrics, print_metrics
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

# Enable deterministic operations for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Set environment variables for deterministic behavior
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Configuration
STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hybrid_tft_results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hybrid_tft_models')

# Create directories if they don't exist
for directory in [RESULTS_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Model parameters
SEQUENCE_LENGTH = 60
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'
EPOCHS = 50
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_HEADS = 8
DROPOUT_RATE = 0.15
INITIAL_LR = 0.002
WEIGHT_DECAY = 1e-5

def find_all_csv_files():
    """Find all CSV files in the stock data directory"""
    csv_files = []
    for root, dirs, files in os.walk(STOCK_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def train_model_on_stream(model, X_train, y_train, X_val, y_val, stream, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train the SimpleTFT model on a specific CUDA stream"""
    start_time = time.time()

    # Set PyTorch to use the specified stream
    torch.cuda.set_stream(stream)

    # Convert numpy arrays to PyTorch tensors (keep on CPU for DataLoader)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create data loaders without pin_memory to avoid CUDA tensor pinning error
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

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
            # Move tensors to GPU here
            batch_X = batch_X.cuda(non_blocking=True)
            batch_y = batch_y.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(-1), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
            # Clear GPU cache periodically
            if len(train_losses) % 10 == 0:
                torch.cuda.empty_cache()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move tensors to GPU here
                batch_X = batch_X.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)
                
                outputs = model(batch_X)
                val_loss = criterion(outputs.squeeze(-1), batch_y)
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

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'      Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f} - Val: {avg_val_loss:.4f} - LR: {current_lr:.6f}')

    training_time = time.time() - start_time
    history['training_time'] = training_time

    return model, history

def train_and_evaluate_single_stock(csv_file, model, stream, device_id=0):
    """Train and evaluate SimpleTFT model on a single stock CSV file using existing model on stream"""
    # Extract stock symbol from filename
    symbol = os.path.basename(csv_file).split('.')[0]
    region = os.path.basename(os.path.dirname(csv_file))

    print(f"    Processing {region}/{symbol}")

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
        print(f"    Error in preprocessing {region}/{symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'region': region,
            'error': f"Preprocessing failed: {str(e)}"
        }

    # Check if we have enough data
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"    Insufficient data for {region}/{symbol} after preprocessing")
        return {
            'symbol': symbol,
            'region': region,
            'error': "Insufficient data after preprocessing"
        }

    # Set device
    device = torch.device(f'cuda:{device_id}')

    # Reinitialize model weights for new stock
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    print(f"      Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model on the stream
    trained_model, history = train_model_on_stream(
        model,
        X_train, y_train,
        X_test, y_test,
        stream,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Make predictions
    X_test_tensor = torch.FloatTensor(X_test).to(device, non_blocking=True)
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(X_test_tensor)
        y_pred = predictions.squeeze(-1).cpu().numpy()

    # Inverse transform predictions and actual values
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), target_scaler)
    y_pred_inv = inverse_transform_predictions(y_pred.reshape(-1, 1), target_scaler)

    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    metrics['training_time'] = history['training_time']
    print_metrics(metrics, f"{region}/{symbol}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{region}_{symbol}_hybrid_tft_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'metrics': metrics
    }, model_path)

    return {
        'symbol': symbol,
        'region': region,
        'metrics': metrics,
        'model_path': model_path
    }

def train_two_stocks_on_stream(csv_file_pair, stream, device_id=0, stream_id=1):
    """Train 2 stocks sequentially on the same stream (Hybrid approach)"""
    print(f"  Stream {stream_id}: Processing 2 symbols sequentially")
    
    # Set device
    device = torch.device(f'cuda:{device_id}')
    
    # Create a single model instance for this stream
    # Use first file to determine input size
    first_csv = csv_file_pair[0]
    df_temp = load_stock_data(first_csv)
    X_temp, _, _, _, _, _ = preprocess_data(
        df_temp,
        target_col=TARGET_COL,
        feature_cols=FEATURE_COLS,
        sequence_length=SEQUENCE_LENGTH
    )
    
    model = create_simple_tft_model(
        input_size=X_temp.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=3,
        dropout=DROPOUT_RATE
    )
    model = model.to(device)
    
    results = []
    stream_start_time = time.time()
    
    # Train both stocks sequentially on this stream
    for csv_file in csv_file_pair:
        result = train_and_evaluate_single_stock(csv_file, model, stream, device_id)
        results.append(result)
    
    stream_total_time = time.time() - stream_start_time
    print(f"  Stream {stream_id}: Completed 2 symbols in {stream_total_time:.2f}s")
    
    return results, stream_total_time

def process_gpu(gpu_id, csv_file_pairs):
    """Process training for a specific GPU using streams - Hybrid approach (2 stocks per stream)"""
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)

    # Re-initialize seeds for this process to ensure determinism
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    # Re-enable deterministic operations for this process
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    results = []
    print(f"\n{'='*70}")
    print(f"GPU {gpu_id}: Processing {len(csv_file_pairs)} stream pairs (2 symbols each)")
    print(f"Total symbols for GPU {gpu_id}: {len(csv_file_pairs) * 2}")
    print(f"{'='*70}")

    # Create streams for this GPU - Always use 8 streams
    num_streams = min(len(csv_file_pairs), 8)  # Keep 8 streams, each handles 2 stocks
    if NUMBA_AVAILABLE:
        streams = [cuda.stream() for _ in range(num_streams)]
        pytorch_streams = [torch.cuda.ExternalStream(stream.handle.value) for stream in streams]
    else:
        pytorch_streams = [torch.cuda.current_stream() for _ in range(num_streams)]
        streams = pytorch_streams

    # Track parallel training timing
    training_start_time = None
    training_end_time = None
    num_batches = 0
    total_stream_times = []

    # Process file pairs in batches
    for i in range(0, len(csv_file_pairs), num_streams):
        batch_pairs = csv_file_pairs[i:i+num_streams]
        batch_streams = pytorch_streams[:len(batch_pairs)]
        num_batches += 1

        print(f"\n  Batch {num_batches}: Starting {len(batch_pairs)} streams ({len(batch_pairs)*2} symbols)")

        # Record when first batch starts training
        if training_start_time is None:
            training_start_time = time.time()

        # Train models concurrently on streams using threads
        with ThreadPoolExecutor(max_workers=len(batch_pairs)) as thread_executor:
            futures = [
                thread_executor.submit(train_two_stocks_on_stream, pair, stream, 0, stream_id=j+1)
                for j, (pair, stream) in enumerate(zip(batch_pairs, batch_streams))
            ]
            for future in as_completed(futures):
                batch_results, stream_time = future.result()
                results.extend(batch_results)
                total_stream_times.append(stream_time)

        # Record when last batch finishes training
        training_end_time = time.time()

    # Synchronize streams
    if NUMBA_AVAILABLE:
        for stream in streams:
            stream.synchronize()
    else:
        torch.cuda.synchronize()

    # Calculate parallel training metrics
    total_training_time = training_end_time - training_start_time if training_start_time and training_end_time else 0
    avg_batch_time = total_training_time / num_batches if num_batches > 0 else 0
    avg_stream_time = np.mean(total_stream_times) if total_stream_times else 0

    print(f"\n{'='*70}")
    print(f"GPU {gpu_id} Hybrid Training Metrics:")
    print(f"  Total parallel training time: {total_training_time:.2f}s")
    print(f"  Number of batches: {num_batches}")
    print(f"  Average batch time: {avg_batch_time:.2f}s")
    print(f"  Average stream time (2 stocks): {avg_stream_time:.2f}s")
    print(f"  Streams per batch: {num_streams}")
    print(f"  Stocks per stream: 2")
    print(f"{'='*70}")

    return results, {
        'gpu_id': gpu_id,
        'total_training_time': total_training_time,
        'num_batches': num_batches,
        'avg_batch_time': avg_batch_time,
        'avg_stream_time': avg_stream_time,
        'num_streams': num_streams,
        'stocks_per_stream': 2
    }

def run_hybrid_experiment(num_symbols=4, multi_gpu=False):
    """Run hybrid parallel training experiment (2 stocks per stream)
    Always distributes symbols evenly across all available GPUs
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run parallel training.")
        return

    # Ensure even number of symbols for pairing
    if num_symbols % 2 != 0:
        num_symbols += 1
        print(f"⚠️  Adjusted to {num_symbols} symbols (must be even for 2 stocks per stream)")

    # Always get all available GPUs if multi_gpu is True
    total_gpus = torch.cuda.device_count()
    
    if multi_gpu and total_gpus > 1:
        num_gpus = total_gpus
        print(f"\n{'='*70}")
        print(f"HYBRID MODE: Distributing across ALL {num_gpus} GPUs (2 stocks per stream)")
        print(f"{'='*70}")
    else:
        num_gpus = 1
        print(f"\n{'='*70}")
        print(f"HYBRID MODE: Using single GPU (GPU 1) - 2 stocks per stream")
        print(f"{'='*70}")

    # Get all CSV files
    all_csv_files = find_all_csv_files()
    print(f"Found {len(all_csv_files)} CSV files in total")

    # Select symbols (sorted for consistent ordering)
    selected_csv_files = sorted(all_csv_files)[:num_symbols]
    print(f"Selected {num_symbols} symbols for Hybrid TFT training")

    # Create pairs of stocks
    stock_pairs = []
    for i in range(0, len(selected_csv_files), 2):
        if i + 1 < len(selected_csv_files):
            stock_pairs.append([selected_csv_files[i], selected_csv_files[i+1]])

    print(f"Created {len(stock_pairs)} pairs (2 stocks per stream)")

    # ALWAYS distribute pairs evenly across all GPUs
    total_pairs = len(stock_pairs)
    pairs_per_gpu = total_pairs // num_gpus
    remaining_pairs = total_pairs % num_gpus
    
    print(f"\nDistribution Strategy:")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Base pairs per GPU: {pairs_per_gpu}")
    print(f"  Remaining pairs: {remaining_pairs}")

    # Distribute pairs across GPUs
    gpu_assignments = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        # First 'remaining_pairs' GPUs get one extra pair
        count = pairs_per_gpu + (1 if gpu_id < remaining_pairs else 0)
        # Use GPU 1 for single GPU mode, otherwise use the actual GPU ID
        actual_gpu_id = 1 if not multi_gpu else gpu_id
        assigned_pairs = stock_pairs[start_idx:start_idx + count]
        gpu_assignments.append((actual_gpu_id, assigned_pairs))
        print(f"  GPU {actual_gpu_id}: {count} pairs ({count*2} symbols)")
        start_idx += count

    all_results = []
    gpu_timing_data = []
    start_time = time.time()

    # Use ProcessPoolExecutor for parallel GPU processing
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_gpu, gpu_id, pairs) for gpu_id, pairs in gpu_assignments]
        for future in as_completed(futures):
            results, timing_data = future.result()
            all_results.extend(results)
            gpu_timing_data.append(timing_data)

    total_time = time.time() - start_time

    # Calculate parallel training metrics
    total_training_time = sum(gpu['total_training_time'] for gpu in gpu_timing_data)
    total_batches = sum(gpu['num_batches'] for gpu in gpu_timing_data)
    avg_batch_time = total_training_time / total_batches if total_batches > 0 else 0
    avg_stream_time = np.mean([gpu['avg_stream_time'] for gpu in gpu_timing_data])

    print(f"\n{'='*70}")
    print(f"HYBRID TFT PARALLEL TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total experiment time: {total_time:.2f} seconds")

    # Calculate theoretical sequential time for comparison
    if all_results:
        successful_results = [r for r in all_results if 'metrics' in r and 'training_time' in r['metrics']]
        if successful_results:
            avg_individual_time = sum(r['metrics']['training_time'] for r in successful_results) / len(successful_results)
            theoretical_sequential = avg_individual_time * num_symbols
            speedup = theoretical_sequential / total_training_time if total_training_time > 0 else 0
            efficiency = total_training_time / total_time if total_time > 0 else 0

            # Print performance summary in tabular format
            print(f"\n{'='*70}")
            print(f"{'HYBRID PARALLEL PERFORMANCE SUMMARY':^70}")
            print(f"{'='*70}")
            print(f"{'Metric':<35} | {'Value':<20} | {'Unit'}")
            print(f"{'-'*70}")
            print(f"{'Training Strategy':<35} | {'2 stocks per stream':<20} | {'-'}")
            print(f"{'Total GPUs Used':<35} | {num_gpus:<20} | {'GPUs'}")
            print(f"{'Total Symbols Processed':<35} | {num_symbols:<20} | {'symbols'}")
            print(f"{'Symbols per Stream':<35} | {2:<20} | {'symbols'}")
            print(f"{'Total Experiment Time':<35} | {total_time:<20.2f} | {'seconds'}")
            print(f"{'Average Stream Time':<35} | {avg_stream_time:<20.2f} | {'seconds'}")
            print(f"{'Parallel Speedup':<35} | {speedup:<20.2f} | {'factor'}")
            print(f"{'Parallel Efficiency':<35} | {efficiency:<19.1%} | {'percentage'}")
            print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"{'HYBRID PARALLEL PERFORMANCE SUMMARY':^70}")
        print(f"{'='*70}")
        print(f"{'Metric':<35} | {'Value':<20} | {'Unit'}")
        print(f"{'-'*70}")
        print(f"{'Training Strategy':<35} | {'2 stocks per stream':<20} | {'-'}")
        print(f"{'Total GPUs Used':<35} | {num_gpus:<20} | {'GPUs'}")
        print(f"{'Total Symbols Processed':<35} | {num_symbols:<20} | {'symbols'}")
        print(f"{'Total Experiment Time':<35} | {total_time:<20.2f} | {'seconds'}")
        print(f"{'Parallel Speedup':<35} | {'N/A':<20} | {'factor'}")
        print(f"{'Parallel Efficiency':<35} | {'N/A':<20} | {'percentage'}")
        print(f"{'='*70}")

    # Save results
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        results_df = pd.DataFrame([
            {
                'Symbol': f"{r['region']}/{r['symbol']}",
                'MAE': r['metrics']['mae'],
                'MSE': r['metrics']['mse'],
                'RMSE': r['metrics']['rmse'],
                'Training Time': r['metrics']['training_time']
            } for r in successful_results
        ])
        
        results_path = os.path.join(RESULTS_DIR, f'hybrid_tft_results_{num_symbols}_symbols.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save detailed results
        detailed_path = os.path.join(RESULTS_DIR, f'hybrid_tft_detailed_{num_symbols}_symbols.json')
        with open(detailed_path, 'w') as f:
            json.dump(all_results, f, indent=4, default=str)
        
        print(f"\nResults saved to {results_path}")
        print(f"Detailed results saved to {detailed_path}")
        
        print(f"\nHybrid TFT Average Performance:")
        print(f"MAE: {results_df['MAE'].mean():.4f}")
        print(f"RMSE: {results_df['RMSE'].mean():.4f}")
        print(f"Avg Individual Training Time: {results_df['Training Time'].mean():.2f}s")
        print(f"Total Symbols Processed: {len(results_df)}")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid TFT parallel training (2 stocks per stream)")
    parser.add_argument('--num_symbols', type=int, default=4, help='Number of symbols to train (will be adjusted to even)')
    parser.add_argument('--multi_gpu', action='store_true', help='Distribute across ALL available GPUs')
    args = parser.parse_args()

    run_hybrid_experiment(num_symbols=args.num_symbols, multi_gpu=args.multi_gpu)
