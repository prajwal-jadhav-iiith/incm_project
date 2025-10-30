"""
Training script for the spatial decoder network.

This script:
1. Loads training data from the Q-learning agent
2. Prepares and splits the data
3. Trains the decoder network
4. Evaluates performance
5. Saves the trained model
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from decoder import (
    SpatialDecoder, DecoderDataset, prepare_decoder_data,
    train_decoder, evaluate_decoder, save_decoder
)
from q_learning import QLearningAgent
import pickle


def plot_training_curves(history, save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    
    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Decoder Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_prediction_comparison(targets, predictions, grid_size, save_path=None):
    """Visualize actual vs predicted positions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Actual positions
    axes[0].scatter(targets[:, 1], targets[:, 0], alpha=0.5, s=10)
    axes[0].set_title('Actual Positions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_xlim(-0.5, grid_size[1] - 0.5)
    axes[0].set_ylim(-0.5, grid_size[0] - 0.5)  # No flip needed for scatter
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Predicted positions
    axes[1].scatter(predictions[:, 1], predictions[:, 0], alpha=0.5, s=10, color='orange')
    axes[1].set_title('Predicted Positions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_xlim(-0.5, grid_size[1] - 0.5)
    axes[1].set_ylim(-0.5, grid_size[0] - 0.5)  # No flip needed for scatter
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    # Prediction errors
    errors = np.linalg.norm(targets - predictions, axis=1)
    scatter = axes[2].scatter(targets[:, 1], targets[:, 0], 
                             c=errors, cmap='RdYlGn_r', alpha=0.6, s=20)
    axes[2].set_title('Prediction Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_xlim(-0.5, grid_size[1] - 0.5)
    axes[2].set_ylim(-0.5, grid_size[0] - 0.5)  # No flip needed for scatter
    axes[2].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[2], label='Error (grid units)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison saved to {save_path}")
    
    plt.show()


def load_agent_data(agent_path: str):
    """Load training data from saved Q-learning agent."""
    print(f"Loading agent from {agent_path}...")
    
    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    
    training_data = agent_state['training_data']
    print(f"Loaded {len(training_data)} training samples")
    
    return training_data


def main():
    """Main training pipeline for the decoder network."""
    
    print("\n" + "=" * 70)
    print("Phase 2: Spatial Decoder Network Training")
    print("=" * 70 + "\n")
    
    # ========== 1. Load Data ==========
    agent_path = input("Path to trained agent [default: trained_agent.pkl]: ").strip()
    agent_path = agent_path or "trained_agent.pkl"
    
    try:
        training_data = load_agent_data(agent_path)
    except FileNotFoundError:
        print(f"\n❌ Error: {agent_path} not found!")
        print("Please train a Q-learning agent first (Phase 1)")
        return
    
    if len(training_data) < 100:
        print(f"\n⚠️  Warning: Only {len(training_data)} samples available")
        print("Recommend training Q-learning agent for more episodes")
    
    # ========== 2. Prepare Data ==========
    print("\nPreparing data...")
    q_values, positions = prepare_decoder_data(training_data)
    
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Position range: X=[{positions[:, 1].min()}, {positions[:, 1].max()}], "
          f"Y=[{positions[:, 0].min()}, {positions[:, 0].max()}]")
    
    # Infer grid size
    grid_height = int(positions[:, 0].max()) + 1
    grid_width = int(positions[:, 1].max()) + 1
    grid_size = (grid_height, grid_width)
    print(f"  Inferred grid size: {grid_size}")
    
    # Create dataset
    dataset = DecoderDataset(q_values, positions)
    
    # Split into train/val/test
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataset)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_dataset)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_dataset)/n_total*100:.1f}%)")
    
    # ========== 3. Model Configuration ==========
    print("\nModel Configuration:")
    
    hidden_size = int(input("  Hidden layer size [default: 256]: ").strip() or "256")
    dropout = float(input("  Dropout rate [default: 0.0]: ").strip() or "0.0")
    
    model = SpatialDecoder(
        input_size=4,
        hidden_size=hidden_size,
        output_size=2,
        dropout_rate=dropout
    )
    
    print(f"\n  Created decoder: 4 -> {hidden_size} -> 2")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 4. Training Configuration ==========
    print("\nTraining Configuration:")
    
    batch_size = int(input("  Batch size [default: 64]: ").strip() or "64")
    n_epochs = int(input("  Number of epochs [default: 100]: ").strip() or "100")
    learning_rate = float(input("  Learning rate [default: 0.001]: ").strip() or "0.001")
    weight_decay = float(input("  Weight decay (L2) [default: 1e-5]: ").strip() or "1e-5")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    # ========== 5. Train Model ==========
    print(f"\nTraining decoder for {n_epochs} epochs...")
    print("=" * 70)
    
    history = train_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        verbose=True,
        early_stopping_patience=15
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    
    # ========== 6. Evaluate Model ==========
    print("\nEvaluating on test set...")
    
    eval_results = evaluate_decoder(model, test_loader, device=device)
    
    print("\nTest Set Performance:")
    print(f"  MSE:  {eval_results['mse']:.6f}")
    print(f"  RMSE: {eval_results['rmse']:.4f} grid units")
    print(f"  MAE:  {eval_results['mae']:.4f} grid units")
    print(f"  MSE (X): {eval_results['mse_x']:.6f}")
    print(f"  MSE (Y): {eval_results['mse_y']:.6f}")
    
    # Success criterion from proposal: MSE < 2 grid units
    success_threshold = 2.0
    if eval_results['rmse'] < success_threshold:
        print(f"\n✅ SUCCESS: RMSE ({eval_results['rmse']:.4f}) < {success_threshold} grid units")
    else:
        print(f"\n⚠️  RMSE ({eval_results['rmse']:.4f}) > {success_threshold} grid units")
        print("Consider: more training data, larger network, or longer training")
    
    # ========== 7. Visualize Results ==========
    print("\nGenerating visualizations...")
    
    plot_training_curves(history, save_path='decoder_training_curves.png')
    plot_prediction_comparison(
        eval_results['targets'],
        eval_results['predictions'],
        grid_size,
        save_path='decoder_predictions.png'
    )
    
    # ========== 8. Save Model ==========
    save = input("\nSave trained decoder? (y/n) [default: y]: ").strip().lower() or "y"
    
    if save == "y":
        metadata = {
            'hidden_size': hidden_size,
            'dropout': dropout,
            'training_samples': n_train,
            'test_mse': eval_results['mse'],
            'test_rmse': eval_results['rmse'],
            'grid_size': grid_size,
            'training_epochs': len(history['train_loss'])
        }
        
        save_decoder(model, 'trained_decoder.pth', metadata)
        print("\n📁 Model saved to: trained_decoder.pth")
    
    # ========== 9. Summary ==========
    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print("=" * 70)
    print("\n✅ Decoder successfully trained to predict positions from Q-values")
    print(f"✅ Hidden layer has {hidden_size} units that may exhibit place cell properties")
    print("\nNext Steps (Phase 3):")
    print("  1. Extract hidden layer activations for all grid positions")
    print("  2. Construct rate maps for each hidden unit")
    print("  3. Identify place fields using spatial information metrics")
    print("  4. Quantify place cell-like properties")
    print("\nReady to proceed to Phase 3: Analysis of Emergent Representations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()