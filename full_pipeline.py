"""
Complete Pipeline: Q-Learning → Decoder Training

This script runs the complete Phase 1 + Phase 2 pipeline:
1. Train Q-learning agent (or load existing)
2. Train spatial decoder network
3. Generate all visualizations
4. Preview hidden representations
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

# Phase 1 imports
from environment import create_open_field, create_four_rooms, create_t_maze
from q_learning import QLearningAgent, train_agent, evaluate_agent

# Phase 2 imports
from decoder import (
    SpatialDecoder, DecoderDataset, prepare_decoder_data,
    train_decoder, evaluate_decoder, save_decoder
)


def run_phase1(env_type='open_field', grid_size=10, n_episodes=5000, 
               force_retrain=False):
    """
    Run Phase 1: Q-learning agent training.
    
    Args:
        env_type: Type of environment
        grid_size: Size of grid
        n_episodes: Number of training episodes
        force_retrain: If True, train even if saved agent exists
    
    Returns:
        Trained agent and environment
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Q-LEARNING AGENT")
    print("=" * 70 + "\n")
    
    # Check if agent exists
    agent_path = 'trained_agent.pkl'
    
    if not force_retrain:
        try:
            import pickle
            with open(agent_path, 'rb') as f:
                agent_state = pickle.load(f)
            print(f"✓ Found existing agent with {len(agent_state['training_data'])} samples")
            
            use_existing = input("Use existing agent? (y/n) [y]: ").strip().lower()
            if use_existing != 'n':
                # Load agent
                env = create_open_field(size=grid_size)
                agent = QLearningAgent(n_states=env.get_state_space_size(), n_actions=4)
                agent.load(agent_path)
                
                print(f"\n✓ Loaded existing agent")
                print(f"  Training samples: {len(agent.get_training_data())}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                
                # Quick evaluation
                eval_stats = evaluate_agent(env, agent, n_episodes=50)
                print(f"  Success rate: {eval_stats['success_rate']:.1%}")
                
                return agent, env
        except FileNotFoundError:
            pass
    
    # Create environment
    print(f"Creating {env_type} environment (size={grid_size})...")
    
    if env_type == 'open_field':
        env = create_open_field(size=grid_size)
    elif env_type == 'four_rooms':
        env = create_four_rooms(room_size=grid_size // 2)
    elif env_type == 't_maze':
        env = create_t_maze(length=grid_size)
    else:
        env = create_open_field(size=grid_size)
    
    print(f"✓ Environment created: {env.height}x{env.width}")
    
    # Create agent
    print(f"\nInitializing Q-learning agent...")
    agent = QLearningAgent(
        n_states=env.get_state_space_size(),
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    print(f"✓ Agent initialized")
    
    # Train agent
    print(f"\nTraining agent for {n_episodes} episodes...")
    print("-" * 70)
    
    agent, stats = train_agent(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        verbose=True,
        verbose_interval=max(n_episodes // 10, 1),
        collect_data_interval=5
    )
    
    print("\n" + "-" * 70)
    print("Phase 1 Complete!")
    
    # Evaluate
    eval_stats = evaluate_agent(env, agent, n_episodes=100)
    print(f"\n✅ Final Performance:")
    print(f"   Success rate: {eval_stats['success_rate']:.1%}")
    print(f"   Mean reward: {eval_stats['mean_reward']:.2f}")
    print(f"   Mean steps: {eval_stats['mean_length']:.1f}")
    print(f"   Training samples: {len(agent.get_training_data())}")
    
    # Save agent
    agent.save(agent_path)
    
    return agent, env


def run_phase2(agent, env, hidden_size=256, n_epochs=100, batch_size=64):
    """
    Run Phase 2: Decoder network training.
    
    Args:
        agent: Trained Q-learning agent
        env: Environment
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        batch_size: Batch size
    
    Returns:
        Trained decoder model and evaluation results
    """
    print("\n" + "=" * 70)
    print("PHASE 2: SPATIAL DECODER NETWORK")
    print("=" * 70 + "\n")
    
    # Prepare data
    print("Preparing training data...")
    training_data = agent.get_training_data()
    
    if len(training_data) < 500:
        print(f"⚠️  Warning: Only {len(training_data)} samples available")
        print("Recommend: Re-run Phase 1 with more episodes or lower collect_data_interval")
    
    q_values, positions = prepare_decoder_data(training_data)
    print(f"✓ Data prepared: {len(q_values)} samples")
    
    # Create dataset and split
    dataset = DecoderDataset(q_values, positions)
    
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"✓ Data split: {n_train} train, {n_val} val, {n_test} test")
    
    # Create model
    print(f"\nCreating decoder: 4 → {hidden_size} → 2")
    model = SpatialDecoder(
        input_size=4,
        hidden_size=hidden_size,
        output_size=2,
        dropout_rate=0.0
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params:,} parameters")
    
    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    history = train_decoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=0.001,
        weight_decay=1e-5,
        device=device,
        verbose=True,
        early_stopping_patience=15
    )
    
    print("\n" + "-" * 70)
    print("Phase 2 Complete!")
    
    # Evaluate
    print("\n✅ Evaluating on test set...")
    eval_results = evaluate_decoder(model, test_loader, device=device)
    
    print(f"   MSE:  {eval_results['mse']:.6f}")
    print(f"   RMSE: {eval_results['rmse']:.4f} grid units")
    print(f"   MAE:  {eval_results['mae']:.4f} grid units")
    
    # Check success criterion
    if eval_results['rmse'] < 2.0:
        print(f"\n✅ SUCCESS: RMSE < 2.0 grid units (proposal criterion)")
    else:
        print(f"\n⚠️  RMSE > 2.0 (consider more training or data)")
    
    # Save model
    metadata = {
        'hidden_size': hidden_size,
        'grid_size': (env.height, env.width),
        'training_samples': n_train,
        'test_rmse': eval_results['rmse'],
        'test_mse': eval_results['mse']
    }
    save_decoder(model, 'trained_decoder.pth', metadata)
    
    return model, eval_results, history


def run_preview(model, agent, env):
    """
    Run preview analysis of hidden representations.
    
    Args:
        model: Trained decoder
        agent: Trained Q-learning agent
        env: Environment
    """
    print("\n" + "=" * 70)
    print("PREVIEW: HIDDEN LAYER ANALYSIS")
    print("=" * 70 + "\n")
    
    # Extract activations for all positions
    print("Extracting hidden activations...")
    activations = {}
    
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) not in env.walls:
                state_idx = env.state_to_index((y, x))
                q_vals = agent.get_q_values(state_idx)
                hidden = model.get_hidden_activations_numpy(q_vals[np.newaxis, :])
                activations[(y, x)] = hidden[0]
    
    print(f"✓ Extracted activations for {len(activations)} positions")
    
    # Construct rate maps
    n_units = model.hidden_size
    rate_maps = np.zeros((n_units, env.height, env.width))
    
    for (y, x), activation in activations.items():
        rate_maps[:, y, x] = activation
    
    # Find spatially tuned units
    localized_count = 0
    for unit_idx in range(n_units):
        rate_map = rate_maps[unit_idx]
        valid_rates = rate_map[rate_map > 0]
        
        if len(valid_rates) > 0:
            threshold = np.percentile(valid_rates, 85)
            n_peak = np.sum(rate_map >= threshold)
            
            if n_peak < 0.3 * len(valid_rates) and n_peak > 0:
                localized_count += 1
    
    print(f"\n✅ Analysis Results:")
    print(f"   Spatially tuned units: {localized_count}/{n_units} ({localized_count/n_units*100:.1f}%)")
    print(f"   Target (proposal): ≥30%")
    
    if localized_count / n_units >= 0.30:
        print(f"   ✅ SUCCESS CRITERION MET!")
    else:
        print(f"   ⚠️  Below target")
    
    # Visualize sample units
    print("\n📊 Generating sample visualizations...")
    
    # Select high-variance units
    variances = np.var(rate_maps.reshape(n_units, -1), axis=1)
    top_units = np.argsort(variances)[-9:]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, unit_idx in enumerate(top_units):
        ax = axes[idx]
        im = ax.imshow(rate_maps[unit_idx], cmap='hot', interpolation='nearest')
        ax.set_title(f'Hidden Unit {unit_idx}')
        ax.plot(env.goal_position[1], env.goal_position[0], 'g*', 
                markersize=12, markeredgecolor='white')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Sample Hidden Unit Spatial Tuning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hidden_unit_preview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: hidden_unit_preview.png")
    plt.show()


def main():
    """Run complete pipeline."""
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE: Q-LEARNING → DECODER → PLACE CELLS")
    print("=" * 70)
    
    # Configuration
    print("\n📋 Configuration")
    print("-" * 70)
    
    grid_size = int(input("Grid size [default: 10]: ").strip() or "10")
    n_episodes = int(input("Q-learning episodes [default: 5000]: ").strip() or "5000")
    hidden_size = int(input("Decoder hidden units [default: 256]: ").strip() or "256")
    n_epochs = int(input("Decoder training epochs [default: 100]: ").strip() or "100")
    
    print(f"\n✓ Configuration:")
    print(f"   Grid: {grid_size}x{grid_size}")
    print(f"   Q-learning: {n_episodes} episodes")
    print(f"   Decoder: {hidden_size} hidden units, {n_epochs} epochs")
    
    # Run pipeline
    try:
        # Phase 1
        agent, env = run_phase1(
            env_type='open_field',
            grid_size=grid_size,
            n_episodes=n_episodes,
            force_retrain=False
        )
        
        # Phase 2
        model, eval_results, history = run_phase2(
            agent=agent,
            env=env,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            batch_size=64
        )
        
        # Preview
        run_preview(model, agent, env)
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 COMPLETE PIPELINE FINISHED!")
        print("=" * 70)
        
        print("\n📁 Generated Files:")
        print("   • trained_agent.pkl - Q-learning agent")
        print("   • trained_decoder.pth - Spatial decoder")
        print("   • hidden_unit_preview.png - Sample rate maps")
        
        print("\n📊 Performance Summary:")
        print(f"   • Q-learning success: {eval_agent(env, agent, n_episodes=50)['success_rate']:.1%}")
        print(f"   • Decoder RMSE: {eval_results['rmse']:.4f} grid units")
        print(f"   • Hidden units: {model.hidden_size}")
        
        print("\n🎯 Next Steps:")
        print("   1. Run comprehensive analysis (Phase 3)")
        print("   2. Compute spatial information metrics")
        print("   3. Generate publication figures")
        print("   4. Compare with biological data")
        
        print("\n✅ Ready for Phase 3: Comprehensive Place Cell Analysis")
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()