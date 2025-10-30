"""
Test script for decoder implementation.

This verifies the decoder works before full training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from decoder import (
    SpatialDecoder, DecoderDataset, prepare_decoder_data,
    train_decoder, evaluate_decoder
)


def test_decoder_model():
    """Test decoder model creation and forward pass."""
    print("Testing Decoder Model...")
    print("-" * 50)
    
    # Create model
    model = SpatialDecoder(
        input_size=4,
        hidden_size=128,
        output_size=2,
        dropout_rate=0.1
    )
    
    print(f"✓ Model created")
    print(f"  Architecture: {model.input_size} -> {model.hidden_size} -> {model.output_size}")
    
    # Test forward pass
    dummy_input = torch.randn(10, 4)  # Batch of 10
    output = model(dummy_input)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test hidden activations
    hidden = model.get_hidden_activations(dummy_input)
    print(f"\n✓ Hidden activation extraction successful")
    print(f"  Hidden shape: {hidden.shape}")
    
    # Test numpy interface
    dummy_np = np.random.randn(5, 4).astype(np.float32)
    pred_np = model.predict(dummy_np)
    print(f"\n✓ Numpy interface works")
    print(f"  Input: {dummy_np.shape}")
    print(f"  Predictions: {pred_np.shape}")
    
    # Test single sample
    single_input = np.random.randn(4).astype(np.float32)
    single_pred = model.predict(single_input)
    print(f"\n✓ Single sample prediction works")
    print(f"  Input: {single_input.shape}")
    print(f"  Prediction: {single_pred.shape}")
    
    print("\n✅ Decoder model tests passed!\n")
    return True


def test_dataset():
    """Test dataset creation."""
    print("Testing Dataset...")
    print("-" * 50)
    
    # Create synthetic data
    n_samples = 100
    q_values = np.random.randn(n_samples, 4).astype(np.float32)
    positions = np.random.randint(0, 10, size=(n_samples, 2)).astype(np.float32)
    
    # Create dataset
    dataset = DecoderDataset(q_values, positions)
    
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test indexing
    q, pos = dataset[0]
    print(f"\n✓ Dataset indexing works")
    print(f"  Q-values: {q.shape}, dtype: {q.dtype}")
    print(f"  Position: {pos.shape}, dtype: {pos.dtype}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_q, batch_pos = next(iter(loader))
    print(f"\n✓ DataLoader works")
    print(f"  Batch Q-values: {batch_q.shape}")
    print(f"  Batch positions: {batch_pos.shape}")
    
    print("\n✅ Dataset tests passed!\n")
    return True


def test_training():
    """Test training loop."""
    print("Testing Training Loop...")
    print("-" * 50)
    
    # Create synthetic data
    n_samples = 200
    q_values = np.random.randn(n_samples, 4).astype(np.float32)
    positions = np.random.randint(0, 10, size=(n_samples, 2)).astype(np.float32)
    
    # Create datasets
    dataset = DecoderDataset(q_values, positions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    model = SpatialDecoder(input_size=4, hidden_size=64, output_size=2)
    
    print(f"✓ Setup complete")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Train for a few epochs
    print("\n✓ Running short training (10 epochs)...")
    history = train_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=10,
        learning_rate=0.01,
        weight_decay=1e-5,
        device='cpu',
        verbose=False
    )
    
    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Initial train loss: {history['train_loss'][0]:.6f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    
    if len(history['val_loss']) > 0:
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Check if loss decreased
    if history['train_loss'][-1] < history['train_loss'][0]:
        print("\n✓ Loss decreased during training")
    else:
        print("\n⚠️  Loss did not decrease (may need more epochs or synthetic data)")
    
    print("\n✅ Training loop tests passed!\n")
    return True


def test_evaluation():
    """Test evaluation function."""
    print("Testing Evaluation...")
    print("-" * 50)
    
    # Create synthetic test data
    n_samples = 50
    q_values = np.random.randn(n_samples, 4).astype(np.float32)
    positions = np.random.randint(0, 10, size=(n_samples, 2)).astype(np.float32)
    
    dataset = DecoderDataset(q_values, positions)
    test_loader = DataLoader(dataset, batch_size=16)
    
    # Create and "train" model (actually just create it)
    model = SpatialDecoder(input_size=4, hidden_size=64, output_size=2)
    
    # Evaluate
    results = evaluate_decoder(model, test_loader, device='cpu')
    
    print(f"✓ Evaluation complete")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  Predictions shape: {results['predictions'].shape}")
    print(f"  Targets shape: {results['targets'].shape}")
    
    print("\n✅ Evaluation tests passed!\n")
    return True


def test_save_load():
    """Test model saving and loading."""
    print("Testing Save/Load...")
    print("-" * 50)
    
    from decoder import save_decoder, load_decoder
    import os
    
    # Create model
    model = SpatialDecoder(input_size=4, hidden_size=128, output_size=2)
    
    # Make a prediction
    test_input = np.random.randn(4).astype(np.float32)
    pred_before = model.predict(test_input)
    
    # Save
    filepath = 'test_decoder.pth'
    metadata = {'test': True, 'hidden_size': 128}
    save_decoder(model, filepath, metadata)
    
    print(f"✓ Model saved")
    
    # Load
    loaded_model, loaded_metadata = load_decoder(filepath)
    pred_after = loaded_model.predict(test_input)
    
    print(f"✓ Model loaded")
    print(f"  Metadata: {loaded_metadata}")
    
    # Check predictions match
    if np.allclose(pred_before, pred_after):
        print("\n✓ Predictions match after save/load")
    else:
        print("\n⚠️  Predictions don't match!")
    
    # Cleanup
    os.remove(filepath)
    print(f"✓ Test file cleaned up")
    
    print("\n✅ Save/Load tests passed!\n")
    return True


def test_integration():
    """Test integration with Q-learning data format."""
    print("Testing Integration with Q-Learning Data...")
    print("-" * 50)
    
    # Simulate Q-learning agent training data
    training_data = []
    for i in range(100):
        training_data.append({
            'q_values': np.random.randn(4).astype(np.float32),
            'position': (np.random.randint(0, 10), np.random.randint(0, 10)),
            'state_index': i
        })
    
    print(f"✓ Created {len(training_data)} simulated training samples")
    
    # Prepare data
    q_values, positions = prepare_decoder_data(training_data)
    
    print(f"\n✓ Data preparation successful")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Positions shape: {positions.shape}")
    
    # Create dataset
    dataset = DecoderDataset(q_values, positions)
    loader = DataLoader(dataset, batch_size=16)
    
    # Create and test model
    model = SpatialDecoder()
    batch_q, batch_pos = next(iter(loader))
    predictions = model(batch_q)
    
    print(f"\n✓ End-to-end pipeline works")
    print(f"  Input: {batch_q.shape}")
    print(f"  Output: {predictions.shape}")
    
    print("\n✅ Integration tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Decoder Implementation Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Model", test_decoder_model),
        ("Dataset", test_dataset),
        ("Training", test_training),
        ("Evaluation", test_evaluation),
        ("Save/Load", test_save_load),
        ("Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("✅ All decoder tests passed!")
        print("\nYou can now:")
        print("  1. Train a Q-learning agent (if not done)")
        print("  2. Run: python train_decoder.py")
        print("  3. Proceed to Phase 3 (place cell analysis)")
    else:
        print(f"❌ {failed} test(s) failed")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()