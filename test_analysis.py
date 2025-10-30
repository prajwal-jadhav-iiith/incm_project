"""
Test script for Phase 3 analysis functions.

Verifies all analysis components work correctly with synthetic data.
"""

import numpy as np
import torch
from analysis import PlaceCellAnalyzer, PlaceCellMetrics
from decoder import SpatialDecoder
from q_learning import QLearningAgent
from environment import create_open_field


def create_synthetic_setup(grid_size=10, hidden_size=50):
    """Create synthetic models for testing."""
    print(f"Creating synthetic setup ({grid_size}x{grid_size}, {hidden_size} units)...")
    
    # Environment
    env = create_open_field(size=grid_size)
    
    # Agent with random Q-values
    n_states = grid_size * grid_size
    agent = QLearningAgent(n_states=n_states, n_actions=4)
    agent.q_table = np.random.randn(n_states, 4)
    
    # Decoder
    model = SpatialDecoder(input_size=4, hidden_size=hidden_size, output_size=2)
    
    print("✓ Synthetic setup created")
    return env, agent, model


def test_analyzer_initialization():
    """Test analyzer initialization."""
    print("\nTesting Analyzer Initialization...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup(grid_size=5, hidden_size=20)
    
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    print(f"✓ Analyzer initialized")
    print(f"  Grid size: {analyzer.grid_height}x{analyzer.grid_width}")
    print(f"  Hidden units: {analyzer.n_units}")
    
    assert analyzer.n_units == 20
    assert analyzer.grid_height == 5
    assert analyzer.grid_width == 5
    
    print("✅ Initialization test passed!\n")
    return True


def test_activation_extraction():
    """Test activation extraction."""
    print("Testing Activation Extraction...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup(grid_size=5, hidden_size=20)
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    activations = analyzer.extract_all_activations()
    
    print(f"✓ Extracted activations for {len(activations)} positions")
    
    # Check dimensions
    sample_activation = next(iter(activations.values()))
    assert len(sample_activation) == 20
    
    # Check all positions covered (except walls)
    expected_positions = 5 * 5 - len(env.walls)
    assert len(activations) == expected_positions
    
    print("✅ Activation extraction test passed!\n")
    return True


def test_rate_map_construction():
    """Test rate map construction."""
    print("Testing Rate Map Construction...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup(grid_size=5, hidden_size=20)
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    rate_maps = analyzer.construct_rate_maps()
    
    print(f"✓ Constructed rate maps: {rate_maps.shape}")
    
    assert rate_maps.shape == (20, 5, 5)
    
    # Check walls are NaN
    for wall in env.walls:
        assert np.isnan(rate_maps[0, wall[0], wall[1]])
    
    print("✅ Rate map construction test passed!\n")
    return True


def test_spatial_information():
    """Test spatial information calculation."""
    print("Testing Spatial Information Calculation...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup()
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    # Create test rate map with localized activity
    rate_map = np.zeros((10, 10))
    rate_map[5, 5] = 10.0  # Strong peak
    rate_map[4:7, 4:7] = 2.0  # Surround
    
    si = analyzer.compute_spatial_information(rate_map)
    
    print(f"✓ Computed SI: {si:.4f} bits")
    assert si > 0, "SI should be positive for localized activity"
    
    # Uniform activity should have low SI
    uniform_map = np.ones((10, 10))
    si_uniform = analyzer.compute_spatial_information(uniform_map)
    
    print(f"✓ Uniform map SI: {si_uniform:.4f} bits")
    assert si_uniform < 0.1, "SI should be near 0 for uniform activity"
    
    print("✅ Spatial information test passed!\n")
    return True


def test_sparsity():
    """Test sparsity calculation."""
    print("Testing Sparsity Calculation...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup()
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    # Sparse (localized) map
    sparse_map = np.zeros((10, 10))
    sparse_map[5, 5] = 10.0
    
    sparsity_sparse = analyzer.compute_sparsity(sparse_map)
    print(f"✓ Sparse map sparsity: {sparsity_sparse:.4f}")
    
    # Dense (uniform) map
    dense_map = np.ones((10, 10))
    sparsity_dense = analyzer.compute_sparsity(dense_map)
    print(f"✓ Dense map sparsity: {sparsity_dense:.4f}")
    
    assert sparsity_sparse < sparsity_dense, "Sparse map should have lower sparsity"
    
    print("✅ Sparsity test passed!\n")
    return True


def test_place_field_identification():
    """Test place field identification."""
    print("Testing Place Field Identification...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup()
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    # Create map with clear place field
    rate_map = np.random.rand(10, 10) * 0.5
    rate_map[3:6, 3:6] = 5.0  # Place field
    
    fields = analyzer.identify_place_fields(rate_map, threshold_value=80)
    
    print(f"✓ Identified {len(fields)} place field(s)")
    
    assert len(fields) >= 1, "Should identify at least one field"
    
    # Multiple fields
    rate_map2 = np.random.rand(10, 10) * 0.5
    rate_map2[2:4, 2:4] = 5.0
    rate_map2[7:9, 7:9] = 5.0
    
    fields2 = analyzer.identify_place_fields(rate_map2, threshold_value=80)
    print(f"✓ Identified {len(fields2)} fields in multi-field map")
    
    assert len(fields2) >= 2, "Should identify multiple fields"
    
    print("✅ Place field identification test passed!\n")
    return True


def test_unit_analysis():
    """Test individual unit analysis."""
    print("Testing Unit Analysis...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup(grid_size=10, hidden_size=20)
    analyzer = PlaceCellAnalyzer(env, agent, model)
    analyzer.construct_rate_maps()
    
    # Analyze first unit
    metric = analyzer.analyze_unit(0)
    
    print(f"✓ Analyzed unit 0:")
    print(f"  SI: {metric.spatial_information:.4f} bits")
    print(f"  Sparsity: {metric.sparsity:.4f}")
    print(f"  # Fields: {metric.n_fields}")
    print(f"  Has place field: {metric.has_place_field}")
    
    assert isinstance(metric, PlaceCellMetrics)
    assert metric.unit_id == 0
    
    print("✅ Unit analysis test passed!\n")
    return True


def test_full_analysis():
    """Test full analysis pipeline."""
    print("Testing Full Analysis Pipeline...")
    print("-" * 50)
    
    env, agent, model = create_synthetic_setup(grid_size=8, hidden_size=30)
    analyzer = PlaceCellAnalyzer(env, agent, model)
    
    # Run full analysis
    metrics = analyzer.analyze_all_units()
    
    print(f"✓ Analyzed {len(metrics)} units")
    
    assert len(metrics) == 30
    
    # Get summary
    summary = analyzer.get_summary_statistics()
    
    print(f"\n✓ Summary statistics:")
    print(f"  Total units: {summary['n_total_units']}")
    print(f"  Place cells: {summary['n_place_cells']}")
    print(f"  Percentage: {summary['place_cell_percentage']:.1f}%")
    print(f"  Mean SI: {summary['mean_si_all']:.4f} bits")
    
    assert summary['n_total_units'] == 30
    
    print("\n✅ Full analysis test passed!\n")
    return True


def test_save_load():
    """Test save/load functionality."""
    print("Testing Save/Load...")
    print("-" * 50)
    
    import os
    
    env, agent, model = create_synthetic_setup(grid_size=5, hidden_size=10)
    analyzer = PlaceCellAnalyzer(env, agent, model)
    analyzer.analyze_all_units()
    
    # Save
    filepath = 'test_analysis_results.pkl'
    analyzer.save_results(filepath)
    print(f"✓ Saved results")
    
    # Create new analyzer and load
    analyzer2 = PlaceCellAnalyzer(env, agent, model)
    analyzer2.load_results(filepath)
    print(f"✓ Loaded results")
    
    # Verify
    assert analyzer2.rate_maps is not None
    assert analyzer2.metrics is not None
    assert len(analyzer2.metrics) == 10
    
    # Cleanup
    os.remove(filepath)
    print(f"✓ Cleaned up test file")
    
    print("\n✅ Save/load test passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Phase 3 Analysis Tests")
    print("=" * 60 + "\n")
    
    tests = [
        ("Initialization", test_analyzer_initialization),
        ("Activation Extraction", test_activation_extraction),
        ("Rate Map Construction", test_rate_map_construction),
        ("Spatial Information", test_spatial_information),
        ("Sparsity", test_sparsity),
        ("Place Field Detection", test_place_field_identification),
        ("Unit Analysis", test_unit_analysis),
        ("Full Analysis", test_full_analysis),
        ("Save/Load", test_save_load)
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
        print("✅ All analysis tests passed!")
        print("\nYou can now:")
        print("  1. Run: python run_analysis.py")
        print("  2. Analyze your trained models")
        print("  3. Generate comprehensive figures")
    else:
        print(f"❌ {failed} test(s) failed")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()