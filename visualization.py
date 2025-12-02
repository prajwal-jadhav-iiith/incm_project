"""
Visualization Module for Place Cell Analysis

Publication-quality figures for analyzing emergent place cell representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Optional
from analysis import PlaceCellMetrics


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14
    sns.set_palette("husl")


def plot_rate_map_grid(
    rate_maps: np.ndarray,
    metrics: List[PlaceCellMetrics],
    env,
    n_examples: int = 25,
    selection: str = 'place_cells',
    save_path: Optional[str] = None
):
    """
    Plot grid of example rate maps.
    
    Args:
        rate_maps: Array of shape (n_units, height, width)
        metrics: List of PlaceCellMetrics
        env: GridWorld environment
        n_examples: Number of units to show
        selection: 'place_cells', 'high_si', or 'random'
        save_path: Path to save figure
    """
    set_publication_style()
    
    # Select units to display
    if selection == 'place_cells':
        candidates = [m for m in metrics if m.has_place_field]
        title_suffix = "Place Cells"
    elif selection == 'high_si':
        candidates = sorted(metrics, key=lambda m: m.spatial_information, reverse=True)
        title_suffix = "Highest Spatial Information"
    else:
        candidates = metrics
        title_suffix = "Random Sample"
    
    n_to_show = min(n_examples, len(candidates))
    selected_metrics = candidates[:n_to_show]
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = int(np.ceil(n_to_show / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(15, 3 * n_rows))
    
    for idx, metric in enumerate(selected_metrics):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        rate_map = rate_maps[metric.unit_id]
        
        # Plot rate map with origin='lower' to match gridworld coordinates
        im = ax.imshow(rate_map, cmap='hot', interpolation='nearest', 
                      aspect='auto', origin='lower')
        
        # Mark goal and start (y-coordinates already correct with origin='lower')
        ax.plot(env.goal_position[1], env.goal_position[0], 'g*', 
                markersize=10, markeredgecolor='white', markeredgewidth=0.5)
        ax.plot(env.initial_start[1], env.initial_start[0], 'co', 
                markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        
        # Title with metrics
        title = f"Unit {metric.unit_id}\nSI={metric.spatial_information:.2f}b"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Example Rate Maps: {title_suffix}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_metric_distributions(
    metrics: List[PlaceCellMetrics],
    save_path: Optional[str] = None
):
    """
    Plot distributions of key metrics.
    
    Args:
        metrics: List of PlaceCellMetrics
        save_path: Path to save figure
    """
    set_publication_style()
    
    place_cells = [m for m in metrics if m.has_place_field]
    non_place_cells = [m for m in metrics if not m.has_place_field]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Spatial Information
    ax = axes[0, 0]
    si_all = [m.spatial_information for m in metrics]
    si_pc = [m.spatial_information for m in place_cells]
    
    ax.hist(si_all, bins=30, alpha=0.6, label='All Units', color='gray')
    ax.hist(si_pc, bins=30, alpha=0.8, label='Place Cells', color='red')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.set_xlabel('Spatial Information (bits)')
    ax.set_ylabel('Count')
    ax.set_title('Spatial Information Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sparsity
    ax = axes[0, 1]
    sparsity_all = [m.sparsity for m in metrics]
    sparsity_pc = [m.sparsity for m in place_cells]
    
    ax.hist(sparsity_all, bins=30, alpha=0.6, label='All Units', color='gray')
    ax.hist(sparsity_pc, bins=30, alpha=0.8, label='Place Cells', color='red')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Count')
    ax.set_title('Sparsity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Number of Fields
    ax = axes[0, 2]
    n_fields = [m.n_fields for m in metrics]
    
    unique_counts = np.unique(n_fields, return_counts=True)
    ax.bar(unique_counts[0], unique_counts[1], alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Place Fields')
    ax.set_ylabel('Count')
    ax.set_title('Place Field Count Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Peak vs Mean Rate
    ax = axes[1, 0]
    peak_rates = [m.peak_rate for m in metrics]
    mean_rates = [m.mean_rate for m in metrics]
    
    colors = ['red' if m.has_place_field else 'gray' for m in metrics]
    ax.scatter(mean_rates, peak_rates, c=colors, alpha=0.5, s=20)
    ax.set_xlabel('Mean Rate')
    ax.set_ylabel('Peak Rate')
    ax.set_title('Peak vs Mean Firing Rate')
    ax.grid(True, alpha=0.3)
    
    # SI vs Sparsity (Scatter)
    ax = axes[1, 1]
    ax.scatter([m.spatial_information for m in non_place_cells],
              [m.sparsity for m in non_place_cells],
              c='gray', alpha=0.4, s=20, label='Non-Place')
    ax.scatter([m.spatial_information for m in place_cells],
              [m.sparsity for m in place_cells],
              c='red', alpha=0.7, s=30, label='Place Cells')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Spatial Information (bits)')
    ax.set_ylabel('Sparsity')
    ax.set_title('Spatial Information vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Field Size Distribution (for place cells)
    ax = axes[1, 2]
    field_sizes = [m.field_size for m in place_cells if m.field_size > 0]
    
    if len(field_sizes) > 0:
        ax.hist(field_sizes, bins=20, alpha=0.7, color='coral')
        ax.set_xlabel('Field Size (fraction of space)')
        ax.set_ylabel('Count')
        ax.set_title('Place Field Size Distribution')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No place fields detected', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Place Field Size Distribution')
    
    plt.suptitle('Place Cell Metrics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_summary_statistics(
    summary: dict,
    save_path: Optional[str] = None
):
    """
    Plot summary statistics and comparison to proposal criteria.
    
    Args:
        summary: Dictionary from get_summary_statistics()
        save_path: Path to save figure
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig)
    
    # Place Cell Percentage
    ax1 = fig.add_subplot(gs[0, 0])
    
    pc_pct = summary['place_cell_percentage']
    target_pct = 30.0
    
    colors = ['green' if pc_pct >= target_pct else 'orange']
    bars = ax1.bar(['Observed', 'Target'], [pc_pct, target_pct], color=colors, alpha=0.7)
    ax1.axhline(target_pct, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Place Cell Percentage\nvs Proposal Criterion', fontweight='bold')
    ax1.set_ylim(0, max(pc_pct, target_pct) * 1.2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Spatial Information Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    means = [summary['mean_si_all'], summary['mean_si_place_cells'], 0.5]
    stds = [summary['std_si_all'], summary['std_si_place_cells'], 0]
    labels = ['All Units', 'Place Cells', 'Target (≥0.5)']
    colors_si = ['gray', 'red', 'green']
    
    x_pos = np.arange(len(labels))
    ax2.bar(x_pos, means, yerr=stds, color=colors_si, alpha=0.7, capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('Spatial Information (bits)')
    ax2.set_title('Spatial Information', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Sparsity Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    
    means_sp = [summary['mean_sparsity_all'], summary['mean_sparsity_place_cells'], 0.3]
    stds_sp = [summary['std_sparsity_all'], summary['std_sparsity_place_cells'], 0]
    labels_sp = ['All Units', 'Place Cells', 'Typical (0.1-0.3)']
    
    ax3.bar(x_pos, means_sp, yerr=stds_sp, color=colors_si, alpha=0.7, capsize=5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels_sp, rotation=15, ha='right')
    ax3.set_ylabel('Sparsity')
    ax3.set_title('Sparsity', fontweight='bold')
    ax3.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Typical range')
    ax3.axhline(0.1, color='green', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Summary Text Box
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    # Create summary text
    success = pc_pct >= 30.0
    status = "✅ SUCCESS" if success else "⚠️ PARTIAL SUCCESS"
    
    summary_text = f"""
    {status}: Place Cell Emergence Analysis
    
    📊 Key Findings:
    • Total Hidden Units: {summary['n_total_units']}
    • Place Cells Identified: {summary['n_place_cells']} ({pc_pct:.1f}%)
    • Units with Fields: {summary['units_with_fields']}
    • Mean Fields per Unit: {summary['mean_n_fields']:.2f}
    
    📈 Spatial Information:
    • All Units: {summary['mean_si_all']:.3f} ± {summary['std_si_all']:.3f} bits
    • Place Cells: {summary['mean_si_place_cells']:.3f} ± {summary['std_si_place_cells']:.3f} bits
    • Target: ≥0.5 bits (from proposal)
    
    🎯 Sparsity:
    • All Units: {summary['mean_sparsity_all']:.3f} ± {summary['std_sparsity_all']:.3f}
    • Place Cells: {summary['mean_sparsity_place_cells']:.3f} ± {summary['std_sparsity_place_cells']:.3f}
    • Biological Range: 0.1-0.3 (typical)
    
    ✅ Success Criteria (from proposal):
    • Target: ≥30% units with place fields → {"MET" if pc_pct >= 30 else "NOT MET"}
    • Target: Mean SI >0.5 bits → {"MET" if summary['mean_si_place_cells'] >= 0.5 else "NOT MET"}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Summary: Emergent Place Cell Properties', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_place_field_coverage(
    rate_maps: np.ndarray,
    metrics: List[PlaceCellMetrics],
    env,
    save_path: Optional[str] = None
):
    """
    Plot spatial coverage of all place fields combined.
    
    Args:
        rate_maps: Array of rate maps
        metrics: List of metrics
        env: Environment
        save_path: Save path
    """
    set_publication_style()
    
    place_cells = [m for m in metrics if m.has_place_field]
    
    # Create coverage map
    coverage = np.zeros((env.height, env.width))
    
    for metric in place_cells:
        rate_map = rate_maps[metric.unit_id]
        # Consider active if above 75th percentile
        threshold = np.nanpercentile(rate_map, 75)
        coverage[rate_map >= threshold] += 1
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage heatmap
    im1 = ax1.imshow(coverage, cmap='YlOrRd', interpolation='nearest', origin='lower')
    ax1.set_title('Place Field Coverage\n(Number of overlapping fields)', fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Mark goal and start
    ax1.plot(env.goal_position[1], env.goal_position[0], 'g*', 
            markersize=15, markeredgecolor='white', markeredgewidth=2)
    ax1.plot(env.initial_start[1], env.initial_start[0], 'co', 
            markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    plt.colorbar(im1, ax=ax1, label='# of Place Fields')
    
    # Coverage distribution
    coverage_flat = coverage[coverage > 0].flatten()
    
    ax2.hist(coverage_flat, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Number of Overlapping Fields')
    ax2.set_ylabel('Number of Positions')
    ax2.set_title('Distribution of Field Overlap', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    if len(coverage_flat) > 0:
        mean_overlap = np.mean(coverage_flat)
        max_overlap = np.max(coverage_flat)
        ax2.axvline(mean_overlap, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={mean_overlap:.1f}')
        ax2.legend()
    
    plt.suptitle(f'Spatial Coverage by {len(place_cells)} Place Cells',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def visualize_q_values(env, agent, save_path=None):
    """Visualize Q-values as arrows on the grid."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid visualization
    grid = np.zeros((env.height, env.width))
    
    # Mark walls
    for wall in env.walls:
        grid[wall] = -1
    
    # Mark goal
    grid[env.goal_position] = 1
    
    # Plot grid
    ax.imshow(grid, cmap='RdYlGn', alpha=0.3, vmin=-1, vmax=1)
    
    # Draw Q-value arrows
    arrow_props = dict(arrowstyle='->', lw=2)
    action_offsets = {
        0: (0, -0.3),   # UP
        1: (0, 0.3),    # DOWN
        2: (-0.3, 0),   # LEFT
        3: (0.3, 0)     # RIGHT
    }
    
    for y in range(env.height):
        for x in range(env.width):
            if (y, x) in env.walls:
                continue
            
            state_idx = env.state_to_index((y, x))
            q_values = agent.get_q_values(state_idx)
            
            # Normalize Q-values for arrow length
            if np.max(np.abs(q_values)) > 0:
                q_normalized = q_values / (np.max(np.abs(q_values)) + 1e-8)
            else:
                q_normalized = q_values
            
            # Draw arrows for each action
            for action, (dx, dy) in action_offsets.items():
                q_val = q_normalized[action]
                if q_val > 0:
                    color = 'blue'
                    alpha = min(abs(q_val), 1.0)
                    ax.annotate('', xy=(x + dx * q_val, y + dy * q_val),
                              xytext=(x, y),
                              arrowprops={**arrow_props, 'color': color, 'alpha': alpha})
    
    # Mark special positions
    ax.plot(env.goal_position[1], env.goal_position[0], 'g*', 
            markersize=20, label='Goal')
    ax.plot(env.initial_start[1], env.initial_start[0], 'ro', 
            markersize=15, label='Start')

    # ax.plot(env.goal_position[0], env.goal_position[1], 'g*', 
    #         markersize=20, label='Goal')
    # ax.plot(env.initial_start[0], env.initial_start[1], 'ro', 
    #         markersize=15, label='Start')
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Learned Policy (Q-values as arrows)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def plot_decoder_training_curves(history, save_path=None):
    """Plot training and validation loss curves for the decoder."""
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
        print(f"✓ Saved: {save_path}")
    
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
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def create_comprehensive_report(
    analyzer,
    save_dir: str = '.'
):
    """
    Create all analysis figures.
    
    Args:
        analyzer: PlaceCellAnalyzer with completed analysis
        save_dir: Directory to save figures
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Generating Comprehensive Analysis Figures")
    print("=" * 70 + "\n")
    
    # 1. Example rate maps - place cells
    print("1. Plotting place cell examples...")
    plot_rate_map_grid(
        analyzer.rate_maps,
        analyzer.metrics,
        analyzer.env,
        n_examples=25,
        selection='place_cells',
        save_path=f'{save_dir}/place_cell_examples.png'
    )
    
    # 2. Example rate maps - high SI
    print("2. Plotting high spatial information examples...")
    plot_rate_map_grid(
        analyzer.rate_maps,
        analyzer.metrics,
        analyzer.env,
        n_examples=25,
        selection='high_si',
        save_path=f'{save_dir}/high_si_examples.png'
    )
    
    # 3. Metric distributions
    print("3. Plotting metric distributions...")
    plot_metric_distributions(
        analyzer.metrics,
        save_path=f'{save_dir}/metric_distributions.png'
    )
    
    # 4. Summary statistics
    print("4. Plotting summary statistics...")
    summary = analyzer.get_summary_statistics()
    plot_summary_statistics(
        summary,
        save_path=f'{save_dir}/summary_statistics.png'
    )
    
    # 5. Coverage map
    print("5. Plotting place field coverage...")
    plot_place_field_coverage(
        analyzer.rate_maps,
        analyzer.metrics,
        analyzer.env,
        save_path=f'{save_dir}/place_field_coverage.png'
    )
    
    print("\n" + "=" * 70)
    print("✅ All figures generated successfully!")
    print(f"📁 Saved to: {save_dir}/")
    print("=" * 70 + "\n")