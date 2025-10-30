"""
Spatial Decoder Network for Place Cell Emergence Project

This network learns to decode spatial coordinates from Q-values.
The hidden layer representations are expected to develop place cell-like properties.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Optional
import pickle


class SpatialDecoder(nn.Module):
    """
    Feedforward neural network that decodes spatial position from Q-values.
    
    Architecture:
        Input (4) -> Hidden (n_hidden) -> Output (2)
        
    The hidden layer is expected to develop place cell-like representations
    when trained to predict spatial coordinates.
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        output_size: int = 2,
        dropout_rate: float = 0.0
    ):
        """
        Initialize the spatial decoder network.
        
        Args:
            input_size: Number of input features (4 Q-values)
            hidden_size: Number of hidden units (100-500 recommended)
            output_size: Number of outputs (2 for x, y coordinates)
            dropout_rate: Dropout probability (0 = no dropout)
        """
        super(SpatialDecoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4) containing Q-values
        
        Returns:
            Predicted coordinates of shape (batch_size, 2)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get activations of the hidden layer (for place cell analysis).
        
        Args:
            x: Input tensor of shape (batch_size, 4) containing Q-values
        
        Returns:
            Hidden layer activations of shape (batch_size, hidden_size)
        """
        with torch.no_grad():
            x = self.fc1(x)
            x = self.relu(x)
        return x
    
    def predict(self, q_values: np.ndarray) -> np.ndarray:
        """
        Predict spatial coordinates from Q-values (numpy interface).
        
        Args:
            q_values: Array of shape (n_samples, 4) or (4,)
        
        Returns:
            Predicted coordinates of shape (n_samples, 2) or (2,)
        """
        self.eval()
        
        # Handle single sample
        single_sample = False
        if q_values.ndim == 1:
            q_values = q_values[np.newaxis, :]
            single_sample = True
        
        # Convert to tensor
        x = torch.FloatTensor(q_values)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(x).numpy()
        
        if single_sample:
            return predictions[0]
        return predictions
    
    def get_hidden_activations_numpy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Get hidden activations from Q-values (numpy interface).
        
        Args:
            q_values: Array of shape (n_samples, 4)
        
        Returns:
            Hidden activations of shape (n_samples, hidden_size)
        """
        self.eval()
        x = torch.FloatTensor(q_values)
        activations = self.get_hidden_activations(x)
        return activations.numpy()


class DecoderDataset(torch.utils.data.Dataset):
    """Dataset for training the spatial decoder."""
    
    def __init__(self, q_values: np.ndarray, positions: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            q_values: Array of shape (n_samples, 4)
            positions: Array of shape (n_samples, 2)
        """
        self.q_values = torch.FloatTensor(q_values)
        self.positions = torch.FloatTensor(positions)
        
        assert len(self.q_values) == len(self.positions)
    
    def __len__(self) -> int:
        return len(self.q_values)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_values[idx], self.positions[idx]


def prepare_decoder_data(training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from Q-learning agent for decoder training.
    
    Args:
        training_data: List of dictionaries from agent.get_training_data()
                      Each dict contains 'q_values' and 'position'
    
    Returns:
        Tuple of (q_values_array, positions_array)
    """
    q_values = []
    positions = []
    
    for data_point in training_data:
        q_values.append(data_point['q_values'])
        positions.append(data_point['position'])
    
    q_values = np.array(q_values, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    
    return q_values, positions


def train_decoder(
    model: SpatialDecoder,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    device: str = 'cpu',
    verbose: bool = True,
    early_stopping_patience: int = 10
) -> Dict:
    """
    Train the spatial decoder network.
    
    Args:
        model: SpatialDecoder instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: L2 regularization strength
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
        early_stopping_patience: Stop if validation loss doesn't improve
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_q, batch_pos in train_loader:
            batch_q = batch_q.to(device)
            batch_pos = batch_pos.to(device)
            
            # Forward pass
            predictions = model(batch_q)
            loss = criterion(predictions, batch_pos)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_q, batch_pos in val_loader:
                    batch_q = batch_q.to(device)
                    batch_pos = batch_pos.to(device)
                    
                    predictions = model(batch_q)
                    loss = criterion(predictions, batch_pos)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}")
                print(f"  Val Loss: {avg_val_loss:.6f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}")
                print(f"  Train Loss: {avg_train_loss:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_decoder(
    model: SpatialDecoder,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict:
    """
    Evaluate decoder performance.
    
    Args:
        model: Trained SpatialDecoder
        test_loader: DataLoader for test data
        device: Device for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_q, batch_pos in test_loader:
            batch_q = batch_q.to(device)
            predictions = model(batch_q).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(batch_pos.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Per-coordinate metrics
    mse_x = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    mse_y = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mse_x': mse_x,
        'mse_y': mse_y,
        'predictions': predictions,
        'targets': targets
    }


def save_decoder(model: SpatialDecoder, filepath: str, metadata: Optional[Dict] = None):
    """Save decoder model and metadata."""
    state = {
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'input_size': model.input_size,
        'output_size': model.output_size,
        'metadata': metadata or {}
    }
    torch.save(state, filepath)
    print(f"Decoder saved to {filepath}")


def load_decoder(filepath: str) -> Tuple[SpatialDecoder, Dict]:
    """Load decoder model and metadata."""
    # Use weights_only=False but suppress warning since we trust our own files
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        state = torch.load(filepath, map_location='cpu', weights_only=False)
    
    model = SpatialDecoder(
        input_size=state['input_size'],
        hidden_size=state['hidden_size'],
        output_size=state['output_size']
    )
    model.load_state_dict(state['model_state_dict'])
    
    print(f"Decoder loaded from {filepath}")
    return model, state['metadata']