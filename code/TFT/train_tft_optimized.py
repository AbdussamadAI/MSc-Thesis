
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from simple_tft import create_simple_tft_model

from data_preprocessing import load_stock_data, preprocess_data, inverse_transform_predictions
from evaluation import calculate_metrics, print_metrics

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Configuration
STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stock_data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_results_optimized')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_models_optimized')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tft_plots_optimized')

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

def create_optimized_tft_model(input_size, hidden_size=256, num_heads=8, num_layers=4, dropout=0.15):
    """Create a SimpleTFT model (shared architecture across TFT scripts)."""
    return create_simple_tft_model(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )

def train_model_optimized(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, device='cpu'):
    """Train the SimpleTFT model with advanced techniques"""

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

    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Use PyTorch Forecasting's built-in loss/metrics
    criterion = torch.nn.SmoothL1Loss()  # More robust than MSE

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            # PyTorch Forecasting TFT expects specific input format
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(-1), batch_y)  # Squeeze output dimension

            # Gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
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

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.6f}')

    return model, history

def find_all_csv_files():
    """Find all CSV files in the stock data directory"""
    csv_files = []
    
    # Walk through all directories in stock_data
    for root, dirs, files in os.walk(STOCK_DATA_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def train_and_evaluate_optimized(csv_file):
    """Train and evaluate optimized TFT model on a single stock CSV file"""
    # Extract stock symbol from filename
    symbol = os.path.basename(csv_file).split('.')[0]
    region = os.path.basename(os.path.dirname(csv_file))
    
    print(f"\n{'='*50}")
    print(f"Processing {region}/{symbol} (OPTIMIZED TFT)")
    print(f"{'='*50}")
    
    # Load and preprocess data
    df = load_stock_data(csv_file)
    X_train, X_test, y_train, y_test, feature_scaler, target_scaler, test_dates = preprocess_data(
        df, 
        target_col=TARGET_COL, 
        feature_cols=FEATURE_COLS, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Create and train optimized model
    model = create_optimized_tft_model(
        input_size=X_train.shape[2],  # Number of features
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    )
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    trained_model, history = train_model_optimized(
        model, 
        X_train, y_train, 
        X_test, y_test, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        device=device
    )
    training_time = time.time() - start_time
    
    # Make predictions with PyTorch Forecasting TFT
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(X_test_tensor)
        # PyTorch Forecasting TFT returns predictions with shape [batch_size, output_size]
        # For single-step forecasting, squeeze the output dimension
        y_pred = predictions.squeeze(-1).cpu().numpy()
    
    # Inverse transform predictions and actual values
    y_test_inv = inverse_transform_predictions(y_test.reshape(-1, 1), target_scaler)
    y_pred_inv = inverse_transform_predictions(y_pred.reshape(-1, 1), target_scaler)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_inv, y_pred_inv)
    metrics['training_time'] = training_time
    print_metrics(metrics, symbol)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{region}_{symbol}_tft_optimized_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'metrics': metrics
    }, model_path)
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, y_test_inv, label='Actual', linewidth=2)
    plt.plot(test_dates, y_pred_inv, label='Predicted', linewidth=2)
    plt.title(f'{region}/{symbol} Stock Price Prediction (Optimized TFT)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training history
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(PLOTS_DIR, f"{region}_{symbol}_tft_optimized_prediction.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'symbol': symbol,
        'region': region,
        'metrics': metrics,
        'model_path': model_path,
        'plot_path': plot_path
    }

def main():
    """Main function to train optimized models on all stock CSV files"""
    csv_files = find_all_csv_files()
    print(f"Found {len(csv_files)} CSV files to process with OPTIMIZED TFT")
    
    results = []
    
    for csv_file in tqdm(csv_files, desc="Training optimized TFT models"):
        try:
            result = train_and_evaluate_optimized(csv_file)
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
    results_path = os.path.join(RESULTS_DIR, 'training_results_optimized.json')
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
    summary_path = os.path.join(RESULTS_DIR, 'summary_optimized.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nOptimized TFT training completed. Results saved to {results_path}")
    print(f"Summary saved to {summary_path}")
    
    # Print average metrics
    print("\nOptimized TFT Average Metrics:")
    print(f"MAE: {summary_df['MAE'].mean():.4f}")
    print(f"MSE: {summary_df['MSE'].mean():.4f}")
    print(f"RMSE: {summary_df['RMSE'].mean():.4f}")
    print(f"Avg. Training Time: {summary_df['Training Time (s)'].mean():.2f} seconds")

if __name__ == "__main__":
    main()
