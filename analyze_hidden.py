"""
Preview script for analyzing hidden layer representations.

This demonstrates how to extract and visualize hidden activations,
showing the bridge from Phase 2 to Phase 3.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from decoder import load_decoder
from q_learning import QLearningAgent
from environment import create_open_field
import pickle


def extract_all_activations(model, agent, env):
    """
    Extract hidden layer activations for all positions in the grid.
    
    Args:
        model: Trained SpatialDecoder
        agent: Trained QLearningAgent
        env: GridWorld environment
    
    Returns:
        Dictionary mapping (y, x) positions to hidden activations
    """
    model.eval()
    activations = {}
    
    print("Extracting hidden activations for all grid positions...")
    
    for y in range(env.height):
        for x in range(env.width):
            position = (y, x)
            
            # Skip walls
            if position in env.walls:
                continue
            
            # Get Q-values for this position
            state_idx = env.state_to_index(position)
            q_values = agent.get_q_values(state_idx)
            
            # Get hidden activations
            hidden = model.get_hidden_activations_numpy(q_values[np.newaxis, :])
            activations[position] = hidden[0]
    
    print(f"✓ Extracted activations for {len(activations)} positions")
    return activations


def construct_rate_maps(activations, env):
    """
    Construct spatial rate maps for each hidden unit.
    
    Args:
        activations: Dict mapping positions to hidden activations
        env: GridWorld environment
    
    Returns:
        Array of shape (n_hidden_units, height, width)
    """
    # Get dimensions
    sample_activation = next(iter(activations.values()))
    n_units = len(sample_activation)
    
    # Initialize rate maps
    rate_maps = np.zeros((n_units, env.height, env.width))
    
    # Fill in activations
    for position, activation in activations.items():
        y, x = position
        rate_maps[:, y, x] = activation
    
    # Mark walls with NaN for visualization
    for wall in env.walls:
        y, x = wall
        rate_maps[:, y, x] = np.nan
    
    return rate_maps


def visualize_sample_units(rate_maps, env, n_units=9, save_path=None):
    """
    Visualize rate maps for a sample of hidden units.
    
    Args:
        rate_maps: Array of shape (n_hidden_units, height, width)
        env: GridWorld environment
        n_units: Number of units to visualize
        save_path: Path to save figure
    """
    n_hidden = rate_maps.shape[0]
    
    # Select units to visualize (pick highest variance units)
    variances = np.nanvar(rate_maps.reshape(n_hidden, -1), axis=1)
    top_indices = np.argsort(variances)[-n_units:]
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_units + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()
    
    for idx, unit_idx in enumerate(top_indices):
        ax = axes[idx]
        
        rate_map = rate_maps[unit_idx]
        
        # Plot rate map
        im = ax.imshow(rate_map, cmap='hot', interpolation='nearest')
        ax.set_title(f'Unit {unit_idx}', fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Mark goal and start
        ax.plot(env.goal_position[1], env.goal_position[0], 'g*', 
                markersize=15, markeredgecolor='white', markeredgewidth=1)
        ax.plot(env.initial_start[1], env.initial_start[0], 'bo', 
                markersize=10, markeredgecolor='white', markeredgewidth=1)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for idx in range(n_units, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Hidden Unit Rate Maps', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Rate maps saved to {save_path}")
    
    plt.show()


def identify_localized_units(rate_maps, threshold_percentile=90):
    """
    Identify units with localized spatial tuning.
    
    Args:
        rate_maps: Array of shape (n_hidden_units, height, width)
        threshold_percentile: Percentile for defining "localized"
    
    Returns:
        Indices of units with localized activity
    """
    n_units = rate_maps.shape[0]
    localized_units = []
    
    for unit_idx in range(n_units):
        rate_map = rate_maps[unit_idx]
        
        # Remove NaN values
        valid_rates = rate_map[~np.isnan(rate_map)]
        
        if len(valid_rates) == 0:
            continue
        
        # Check if unit has a localized peak
        threshold = np.percentile(valid_rates, threshold_percentile)
        peak_locations = rate_map >= threshold
        n_peak_locations = np.sum(peak_locations)
        
        # Consider "localized" if peak is in < 30% of space
        total_locations = np.sum(~np.isnan(rate_map))
        if n_peak_locations < 0.3 * total_locations and n_peak_locations > 0:
            localized_units.append(unit_idx)
    
    return localized_units


def compute_spatial_information(rate_map):
    """
    Compute spatial information content (in bits).
    
    Formula from proposal:
    SI = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)
    
    Args:
        rate_map: 2D array of firing rates
    
    Returns:
        Spatial information in bits
    """
    valid_rates = rate_map[~np.isnan(rate_map)]
    
    if len(valid_rates) == 0:
        return 0.0
    
    # Compute mean rate
    mean_rate = np.mean(valid_rates)
    
    if mean_rate == 0:
        return 0.0
    
    # Uniform occupancy probability
    p_x = 1.0 / len(valid_rates)
    
    # Compute spatial information
    si = 0.0
    for rate in valid_rates:
        if rate > 0:
            si += p_x * (rate / mean_rate) * np.log2(rate / mean_rate)
    
    return si


def preview_analysis(model_path='trained_decoder.pth', 
                     agent_path='trained_agent.pkl'):
    """
    Preview analysis of hidden representations.
    
    This demonstrates what Phase 3 will analyze in detail.
    """
    print("\n" + "=" * 70)
    print("Preview: Hidden Layer Analysis")
    print("=" * 70 + "\n")
    
    # Load models
    print("Loading models...")
    try:
        model, metadata = load_decoder(model_path)
        print(f"  Decoder: {metadata['hidden_size']} hidden units")
    except FileNotFoundError:
        print(f"❌ Decoder not found at {model_path}")
        return
    
    try:
        with open(agent_path, 'rb') as f:
            agent_state = pickle.load(f)
        
        # Reconstruct agent
        grid_size = metadata.get('grid_size', (10, 10))
        n_states = grid_size[0] * grid_size[1]
        agent = QLearningAgent(n_states=n_states, n_actions=4)
        agent.q_table = agent_state['q_table']
        print(f"  Q-learning agent: {n_states} states")
    except FileNotFoundError:
        print(f"❌ Agent not found at {agent_path}")
        return
    
    # Create environment
    grid_size = metadata.get('grid_size', (10, 10))
    env = create_open_field(size=grid_size[0])
    print(f"  Environment: {grid_size}\n")
    
    # Extract activations
    activations = extract_all_activations(model, agent, env)
    
    # Construct rate maps
    print("\nConstructing rate maps...")
    rate_maps = construct_rate_maps(activations, env)
    print(f"✓ Rate maps shape: {rate_maps.shape}")
    
    # Identify localized units
    print("\nIdentifying spatially tuned units...")
    localized_units = identify_localized_units(rate_maps, threshold_percentile=85)
    n_localized = len(localized_units)
    n_total = rate_maps.shape[0]
    
    print(f"✓ Found {n_localized}/{n_total} units with localized tuning "
          f"({n_localized/n_total*100:.1f}%)")
    
    # Compute spatial information for sample units
    print("\nSpatial Information (sample):")
    sample_indices = np.random.choice(n_total, min(5, n_total), replace=False)
    for idx in sample_indices:
        si = compute_spatial_information(rate_maps[idx])
        print(f"  Unit {idx}: {si:.4f} bits")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_sample_units(rate_maps, env, n_units=9, 
                          save_path='sample_rate_maps.png')
    
    # Summary
    print("\n" + "=" * 70)
    print("Preview Complete!")
    print("=" * 70)
    print(f"\n📊 Key Findings:")
    print(f"  • {n_localized}/{n_total} units show localized spatial tuning")
    print(f"  • Rate maps extracted for all {n_total} hidden units")
    print(f"  • Sample visualizations saved")
    
    print(f"\n🎯 Success Criterion (from proposal):")
    print(f"  • Target: ≥30% of units with place field properties")
    print(f"  • Observed: {n_localized/n_total*100:.1f}%")
    
    if n_localized / n_total >= 0.30:
        print("  ✅ SUCCESS CRITERION MET!")
    else:
        print("  ⚠️  Below target - consider:")
        print("     - Training decoder longer")
        print("     - Collecting more Q-learning data")
        print("     - Using larger hidden layer")
    
    print(f"\n📈 Next: Phase 3 will provide comprehensive analysis:")
    print("  • Quantitative place field metrics")
    print("  • Spatial information and sparsity measures")
    print("  • Comparison with biological place cells")
    print("  • Statistical analysis across all units")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        preview_analysis(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        preview_analysis(sys.argv[1])
    else:
        preview_analysis()