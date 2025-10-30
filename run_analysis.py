"""
Main Analysis Script for Phase 3: Place Cell Analysis

This script performs comprehensive analysis of emergent place cell representations
in the decoder network's hidden layer.
"""

import numpy as np
import pickle
from analysis import PlaceCellAnalyzer
from visualization import create_comprehensive_report
from decoder import load_decoder
from q_learning import QLearningAgent
from environment import GridWorld
import os


def load_models(decoder_path='trained_decoder.pth', agent_path='trained_agent.pkl'):
    """
    Load trained models.
    
    Returns:
        Tuple of (decoder_model, agent, metadata)
    """
    print("Loading models...")
    
    # Load decoder
    try:
        decoder_model, metadata = load_decoder(decoder_path)
        print(f"✓ Decoder loaded: {metadata['hidden_size']} hidden units")
    except FileNotFoundError:
        print(f"❌ Decoder not found at {decoder_path}")
        print("Please train decoder first (Phase 2)")
        return None, None, None
    
    # Load agent
    try:
        with open(agent_path, 'rb') as f:
            agent_state = pickle.load(f)
        
        # Get Q-table dimensions
        n_states = agent_state['q_table'].shape[0]
        
        # Reconstruct agent
        agent = QLearningAgent(n_states=n_states, n_actions=4)
        agent.q_table = agent_state['q_table']
        
        print(f"✓ Agent loaded: {n_states} states")
    except FileNotFoundError:
        print(f"❌ Agent not found at {agent_path}")
        print("Please train agent first (Phase 1)")
        return None, None, None
    
    return decoder_model, agent, metadata


def create_environment(metadata, agent):
    """
    Recreate environment from metadata to match training setup.
    
    Args:
        metadata: Metadata from decoder
        agent: Loaded agent (for validation)
    
    Returns:
        GridWorld environment
    """
    # Get grid size from metadata
    grid_size = metadata.get('grid_size', (10, 10))
    height, width = grid_size
    
    print(f"Creating environment from metadata...")
    print(f"  Grid size: {height}×{width}")
    print(f"  Expected states: {height * width}")
    print(f"  Agent has states: {agent.n_states}")
    
    # Validate dimensions match
    if height * width != agent.n_states:
        print(f"⚠️  WARNING: Dimension mismatch!")
        print(f"  Metadata suggests {height}×{width} = {height*width} states")
        print(f"  But agent has {agent.n_states} states")
        print(f"  Attempting to infer correct dimensions...")
        
        # Try to infer dimensions from agent
        n = agent.n_states
        # Try different factorizations
        for h in range(1, int(np.sqrt(n)) + 1):
            if n % h == 0:
                w = n // h
                if abs(h - w) <= 5:  # Prefer close to square
                    height, width = h, w
                    print(f"  Using inferred dimensions: {height}×{width}")
                    break
    
    # Create environment with exact dimensions
    env = GridWorld(
        grid_size=(height, width),
        goal_position=(height - 1, width - 1),
        start_position=(0, 0),
        walls=[],
        goal_reward=10.0,
        step_penalty=-0.01
    )
    
    # Final validation
    if env.get_state_space_size() != agent.n_states:
        print(f"❌ ERROR: Environment mismatch!")
        print(f"  Environment: {env.get_state_space_size()} states")
        print(f"  Agent: {agent.n_states} states")
        return None
    
    print(f"✓ Environment created: {grid_size}")
    print(f"  Validation: {env.get_state_space_size()} states = {agent.n_states} states ✓")
    
    return env


def print_analysis_summary(analyzer):
    """
    Print detailed text summary of analysis.
    
    Args:
        analyzer: PlaceCellAnalyzer with completed analysis
    """
    summary = analyzer.get_summary_statistics()
    place_cells = analyzer.get_place_cells()
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n📊 OVERALL STATISTICS")
    print("-" * 70)
    print(f"Total Hidden Units:        {summary['n_total_units']}")
    print(f"Place Cells Detected:      {summary['n_place_cells']} ({summary['place_cell_percentage']:.1f}%)")
    print(f"Units with Place Fields:   {summary['units_with_fields']}")
    print(f"Mean Fields per Unit:      {summary['mean_n_fields']:.2f}")
    
    print("\n📈 SPATIAL INFORMATION (bits)")
    print("-" * 70)
    print(f"All Units:       {summary['mean_si_all']:.4f} ± {summary['std_si_all']:.4f}")
    print(f"Place Cells:     {summary['mean_si_place_cells']:.4f} ± {summary['std_si_place_cells']:.4f}")
    if place_cells:
        print(f"Minimum (place): {min([m.spatial_information for m in place_cells]):.4f}")
        print(f"Maximum (place): {max([m.spatial_information for m in place_cells]):.4f}")
    
    print("\n🎯 SPARSITY")
    print("-" * 70)
    print(f"All Units:       {summary['mean_sparsity_all']:.4f} ± {summary['std_sparsity_all']:.4f}")
    print(f"Place Cells:     {summary['mean_sparsity_place_cells']:.4f} ± {summary['std_sparsity_place_cells']:.4f}")
    print(f"Biological Range: 0.1 - 0.3 (typical)")
    
    print("\n🔬 PLACE FIELD CHARACTERISTICS")
    print("-" * 70)
    if place_cells:
        field_sizes = [m.field_size for m in place_cells if m.field_size > 0]
        n_fields_list = [m.n_fields for m in place_cells]
        
        print(f"Mean Field Size:       {np.mean(field_sizes):.3f} (fraction of environment)")
        print(f"Mean # Fields:         {np.mean(n_fields_list):.2f}")
        print(f"Single Field Units:    {sum(1 for m in place_cells if m.n_fields == 1)}")
        print(f"Multi-Field Units:     {sum(1 for m in place_cells if m.n_fields > 1)}")
    else:
        print("No place cells detected")
    
    print("\n✅ SUCCESS CRITERIA (from proposal)")
    print("-" * 70)
    
    # Criterion 1: ≥30% place cells
    pc_success = summary['place_cell_percentage'] >= 30.0
    print(f"{'✅' if pc_success else '❌'} Place Cell %:  {summary['place_cell_percentage']:.1f}% "
          f"(target: ≥30%)")
    
    # Criterion 2: Mean SI >0.5 bits
    si_success = summary['mean_si_place_cells'] >= 0.5
    print(f"{'✅' if si_success else '❌'} Spatial Info:  {summary['mean_si_place_cells']:.3f} bits "
          f"(target: ≥0.5 bits)")
    
    # Overall success
    overall_success = pc_success and si_success
    
    print("\n" + "=" * 70)
    if overall_success:
        print("🎉 OVERALL: SUCCESS - All criteria met!")
    elif pc_success or si_success:
        print("⚠️  OVERALL: PARTIAL SUCCESS - Some criteria met")
    else:
        print("❌ OVERALL: Criteria not met")
    print("=" * 70)
    
    print("\n📝 INTERPRETATION")
    print("-" * 70)
    
    if overall_success:
        print("""
The decoder network has successfully developed place cell-like representations!

Key Findings:
• A significant proportion of hidden units exhibit localized spatial tuning
• These units show high spatial information content
• The representations are sparse, similar to biological place cells
• Place fields tile the environment, providing comprehensive coverage

This demonstrates that place cell-like representations can emerge spontaneously
from learning to decode spatial information from task-optimized Q-values,
without requiring specialized, domain-specific neural circuits.

This supports the hypothesis that place cells are a natural computational
solution to spatial representation problems, rather than being hardwired.
        """)
    elif pc_success:
        print("""
A good proportion of units show place cell properties, though spatial
information could be higher. Consider:
• Training the decoder longer
• Collecting more Q-learning data
• Using a larger hidden layer
        """)
    else:
        print("""
Place cell properties are weak. Consider:
• Ensuring Q-learning agent is well-trained (>95% success rate)
• Collecting more training data (>10,000 samples)
• Increasing decoder hidden layer size (try 512 units)
• Training decoder for more epochs
• Verifying decoder achieves RMSE < 1.0
        """)
    
    print("\n" + "=" * 70 + "\n")


def save_detailed_report(analyzer, filepath='analysis_report.txt'):
    """
    Save detailed text report.
    
    Args:
        analyzer: PlaceCellAnalyzer with completed analysis
        filepath: Path to save report
    """
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_analysis_summary(analyzer)
    
    # Get the captured output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Save to file with UTF-8 encoding to handle emojis
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"✓ Detailed report saved to: {filepath}")


def main():
    """Main analysis pipeline."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: COMPREHENSIVE PLACE CELL ANALYSIS")
    print("=" * 70 + "\n")
    
    # Load models
    decoder_model, agent, metadata = load_models()
    
    if decoder_model is None:
        return
    
    # Create environment with validation
    env = create_environment(metadata, agent)
    
    if env is None:
        print("\n❌ Failed to create matching environment!")
        print("Please ensure the agent and decoder were trained together.")
        return
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = PlaceCellAnalyzer(env, agent, decoder_model)
    print(f"✓ Analyzer ready for {analyzer.n_units} hidden units")
    
    # Configure analysis parameters
    print("\n" + "=" * 70)
    print("Analysis Configuration")
    print("=" * 70)
    
    use_defaults = input("\nUse default parameters? (y/n) [default: y]: ").strip().lower() or 'y'
    
    if use_defaults == 'y':
        si_threshold = 0.5
        sparsity_threshold = 0.5
        print("✓ Using default thresholds:")
    else:
        si_threshold = float(input("  Spatial information threshold (bits) [0.5]: ") or 0.5)
        sparsity_threshold = float(input("  Sparsity threshold [0.5]: ") or 0.5)
        print("✓ Using custom thresholds:")
    
    print(f"   Spatial Information: ≥{si_threshold} bits")
    print(f"   Sparsity: ≤{sparsity_threshold}")
    
    # Run analysis
    print("\n" + "=" * 70)
    print("Running Analysis")
    print("=" * 70)
    
    # Extract activations and construct rate maps
    analyzer.extract_all_activations()
    analyzer.construct_rate_maps()
    
    # Analyze all units
    metrics = analyzer.analyze_all_units(
        si_threshold=si_threshold,
        sparsity_threshold=sparsity_threshold
    )
    
    # Print summary
    print_analysis_summary(analyzer)
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70 + "\n")
    
    output_dir = input("Output directory for figures [default: analysis_results]: ").strip()
    output_dir = output_dir or "analysis_results"
    
    create_comprehensive_report(analyzer, save_dir=output_dir)
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70 + "\n")
    
    save_results = input("Save analysis results? (y/n) [default: y]: ").strip().lower() or 'y'
    
    if save_results == 'y':
        results_path = f"{output_dir}/analysis_results.pkl"
        analyzer.save_results(results_path)
        
        report_path = f"{output_dir}/analysis_report.txt"
        save_detailed_report(analyzer, report_path)
        
        print(f"\n✓ Results saved to: {output_dir}/")
        print(f"  • analysis_results.pkl - Full analysis data")
        print(f"  • analysis_report.txt - Text summary")
        print(f"  • *.png - All figures")
    
    # Final message
    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70)
    
    summary = analyzer.get_summary_statistics()
    
    print(f"\n✨ Key Result: {summary['n_place_cells']}/{summary['n_total_units']} units "
          f"({summary['place_cell_percentage']:.1f}%) exhibit place cell properties")
    
    if summary['place_cell_percentage'] >= 30.0:
        print("\n✅ SUCCESS: Project objectives achieved!")
        print("   Place cell-like representations emerged spontaneously from")
        print("   learning to decode spatial coordinates from Q-values.")
    
    print("\n📊 All results available in:", output_dir)
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()