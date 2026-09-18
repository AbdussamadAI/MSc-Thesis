import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM model for stock price prediction
    """
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout_rate=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer (bidirectional = 2 * hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Get the output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out

def create_bilstm_model(input_size, hidden_size=50, num_layers=2, dropout_rate=0.2):
    """
    Create a Bidirectional LSTM model for stock price prediction
    
    Args:
        input_size: Number of features in the input
        hidden_size: Number of LSTM units
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        PyTorch BiLSTM model
    """
    model = BiLSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32, patience=10, learning_rate=0.001):
    """
    Train the model for the full number of epochs (no early stopping)
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data as numpy arrays
        X_test, y_test: Validation data as numpy arrays
        epochs: Number of epochs to train for
        batch_size: Batch size
        patience: Not used (kept for backward compatibility)
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model and training history
    """
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track best model (but no early stopping)
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss = val_loss / len(test_loader)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history
