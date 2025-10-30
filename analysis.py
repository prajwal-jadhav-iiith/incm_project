"""
Place Cell Analysis Module

Implements comprehensive analysis of hidden layer representations:
- Rate map construction
- Spatial information content
- Sparsity measures
- Place field identification
- Statistical analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.stats import zscore


@dataclass
class PlaceCellMetrics:
    """Metrics for a single hidden unit."""
    unit_id: int
    spatial_information: float  # bits
    sparsity: float  # 0-1
    peak_rate: float
    mean_rate: float
    field_size: float  # fraction of environment
    has_place_field: bool
    n_fields: int
    peak_location: Tuple[int, int]
    
    def __repr__(self):
        return (f"Unit {self.unit_id}: SI={self.spatial_information:.3f} bits, "
                f"Sparsity={self.sparsity:.3f}, "
                f"Place Field={'Yes' if self.has_place_field else 'No'}")


class PlaceCellAnalyzer:
    """
    Comprehensive analyzer for place cell properties in decoder hidden layer.
    """
    
    def __init__(self, env, agent, decoder_model):
        """
        Initialize analyzer.
        
        Args:
            env: GridWorld environment
            agent: Trained Q-learning agent
            decoder_model: Trained spatial decoder
        """
        self.env = env
        self.agent = agent
        self.model = decoder_model
        self.model.eval()
        
        self.n_units = decoder_model.hidden_size
        self.grid_height = env.height
        self.grid_width = env.width
        
        # Storage for results
        self.rate_maps = None
        self.activations = None
        self.metrics = None
    
    def extract_all_activations(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Extract hidden layer activations for all valid grid positions.
        
        Returns:
            Dictionary mapping (y, x) positions to activation vectors
        """
        print("Extracting hidden layer activations...")
        activations = {}
        
        # Add debug info
        print(f"  Grid dimensions: {self.grid_height}×{self.grid_width}")
        print(f"  Q-table size: {self.agent.n_states}")
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                position = (y, x)
                
                # Skip walls
                if position in self.env.walls:
                    continue
                
                # Get Q-values for this position
                state_idx = self.env.state_to_index(position)
                
                # Safety check
                if state_idx >= self.agent.n_states:
                    print(f"  ⚠️  Skipping position {position}: index {state_idx} >= {self.agent.n_states}")
                    continue
                
                q_values = self.agent.get_q_values(state_idx)
                
                # Get hidden activations
                hidden = self.model.get_hidden_activations_numpy(
                    q_values[np.newaxis, :]
                )
                activations[position] = hidden[0]
        
        self.activations = activations
        print(f"✓ Extracted activations for {len(activations)} positions")
        return activations
    
    def construct_rate_maps(self) -> np.ndarray:
        """
        Construct spatial rate maps for all hidden units.
        
        Returns:
            Array of shape (n_units, height, width)
        """
        if self.activations is None:
            self.extract_all_activations()
        
        print("Constructing rate maps...")
        
        # Initialize rate maps
        rate_maps = np.zeros((self.n_units, self.grid_height, self.grid_width))
        
        # Fill in activations
        for position, activation in self.activations.items():
            y, x = position
            rate_maps[:, y, x] = activation
        
        # Mark walls as NaN
        for wall in self.env.walls:
            y, x = wall
            rate_maps[:, y, x] = np.nan
        
        self.rate_maps = rate_maps
        print(f"✓ Constructed rate maps: {rate_maps.shape}")
        return rate_maps
    
    def compute_spatial_information(self, rate_map: np.ndarray) -> float:
        """
        Compute spatial information content in bits.
        
        From proposal (Equation 2):
        SI = Σ p(x) * (r(x) / r_mean) * log2(r(x) / r_mean)
        
        Args:
            rate_map: 2D array of firing rates
        
        Returns:
            Spatial information in bits
        """
        # Get valid (non-NaN) rates
        valid_rates = rate_map[~np.isnan(rate_map)].flatten()
        
        if len(valid_rates) == 0:
            return 0.0
        
        # Mean firing rate
        mean_rate = np.mean(valid_rates)
        
        if mean_rate == 0:
            return 0.0
        
        # Uniform occupancy probability
        p_x = 1.0 / len(valid_rates)
        
        # Compute spatial information
        si = 0.0
        for rate in valid_rates:
            if rate > 0:
                ratio = rate / mean_rate
                si += p_x * ratio * np.log2(ratio)
        
        return si
    
    def compute_sparsity(self, rate_map: np.ndarray) -> float:
        """
        Compute sparsity measure.
        
        From proposal (Equation 3):
        Sparsity = (Σ r(x)/N)² / (Σ r(x)²/N)
        
        Place cells typically have low sparsity (0.1-0.3).
        
        Args:
            rate_map: 2D array of firing rates
        
        Returns:
            Sparsity value (0-1)
        """
        valid_rates = rate_map[~np.isnan(rate_map)].flatten()
        
        if len(valid_rates) == 0 or np.sum(valid_rates) == 0:
            return 1.0  # Maximum sparsity
        
        N = len(valid_rates)
        numerator = (np.sum(valid_rates) / N) ** 2
        denominator = np.sum(valid_rates ** 2) / N
        
        if denominator == 0:
            return 1.0
        
        sparsity = numerator / denominator
        return sparsity
    
    def identify_place_fields(
        self,
        rate_map: np.ndarray,
        threshold_method: str = 'percentile',
        threshold_value: float = 75.0,
        min_size: int = 1
    ) -> List[np.ndarray]:
        """
        Identify contiguous place fields in a rate map.
        
        Args:
            rate_map: 2D array of firing rates
            threshold_method: 'percentile' or 'std'
            threshold_value: Percentile (0-100) or number of std devs
            min_size: Minimum field size in pixels
        
        Returns:
            List of binary masks for each identified field
        """
        valid_rates = rate_map[~np.isnan(rate_map)]
        
        if len(valid_rates) == 0:
            return []
        
        # Determine threshold
        if threshold_method == 'percentile':
            threshold = np.percentile(valid_rates, threshold_value)
        elif threshold_method == 'std':
            threshold = np.mean(valid_rates) + threshold_value * np.std(valid_rates)
        else:
            threshold = np.mean(valid_rates)
        
        # Create binary mask
        binary_map = np.zeros_like(rate_map, dtype=bool)
        binary_map[rate_map >= threshold] = True
        binary_map[np.isnan(rate_map)] = False
        
        # Label connected components
        labeled_array, num_features = ndimage.label(binary_map)
        
        # Extract individual fields
        fields = []
        for i in range(1, num_features + 1):
            field_mask = labeled_array == i
            if np.sum(field_mask) >= min_size:
                fields.append(field_mask)
        
        return fields
    
    def analyze_unit(
        self,
        unit_id: int,
        si_threshold: float = 0.5,
        sparsity_threshold: float = 0.5,
        field_threshold_percentile: float = 75.0
    ) -> PlaceCellMetrics:
        """
        Comprehensive analysis of a single hidden unit.
        
        Args:
            unit_id: Index of the hidden unit
            si_threshold: Minimum spatial information for place cell
            sparsity_threshold: Maximum sparsity for place cell
            field_threshold_percentile: Percentile for field detection
        
        Returns:
            PlaceCellMetrics object
        """
        if self.rate_maps is None:
            self.construct_rate_maps()
        
        rate_map = self.rate_maps[unit_id]
        
        # Compute metrics
        si = self.compute_spatial_information(rate_map)
        sparsity = self.compute_sparsity(rate_map)
        
        valid_rates = rate_map[~np.isnan(rate_map)]
        peak_rate = np.max(valid_rates) if len(valid_rates) > 0 else 0.0
        mean_rate = np.mean(valid_rates) if len(valid_rates) > 0 else 0.0
        
        # Identify place fields
        fields = self.identify_place_fields(
            rate_map,
            threshold_method='percentile',
            threshold_value=field_threshold_percentile
        )
        
        n_fields = len(fields)
        
        # Calculate field size (fraction of environment)
        if n_fields > 0:
            total_field_size = sum(np.sum(field) for field in fields)
            total_valid_positions = np.sum(~np.isnan(rate_map))
            field_size = total_field_size / total_valid_positions if total_valid_positions > 0 else 0
        else:
            field_size = 0.0
        
        # Find peak location
        if len(valid_rates) > 0:
            peak_idx = np.unravel_index(np.nanargmax(rate_map), rate_map.shape)
            peak_location = tuple(peak_idx)
        else:
            peak_location = (0, 0)
        
        # Determine if this is a place cell
        # Criteria: high spatial info, low sparsity, has at least one field
        has_place_field = (
            si >= si_threshold and
            sparsity <= sparsity_threshold and
            n_fields >= 1 and
            field_size < 0.5  # Field shouldn't cover more than half the space
        )
        
        return PlaceCellMetrics(
            unit_id=unit_id,
            spatial_information=si,
            sparsity=sparsity,
            peak_rate=peak_rate,
            mean_rate=mean_rate,
            field_size=field_size,
            has_place_field=has_place_field,
            n_fields=n_fields,
            peak_location=peak_location
        )
    
    def analyze_all_units(
        self,
        si_threshold: float = 0.5,
        sparsity_threshold: float = 0.5
    ) -> List[PlaceCellMetrics]:
        """
        Analyze all hidden units.
        
        Args:
            si_threshold: Minimum spatial information for place cell
            sparsity_threshold: Maximum sparsity for place cell
        
        Returns:
            List of PlaceCellMetrics for all units
        """
        print(f"\nAnalyzing all {self.n_units} hidden units...")
        
        if self.rate_maps is None:
            self.construct_rate_maps()
        
        metrics = []
        for unit_id in range(self.n_units):
            metric = self.analyze_unit(
                unit_id,
                si_threshold=si_threshold,
                sparsity_threshold=sparsity_threshold
            )
            metrics.append(metric)
            
            if (unit_id + 1) % 50 == 0:
                print(f"  Processed {unit_id + 1}/{self.n_units} units...")
        
        self.metrics = metrics
        print(f"✓ Analysis complete!")
        return metrics
    
    def get_place_cells(self) -> List[PlaceCellMetrics]:
        """Get metrics for units classified as place cells."""
        if self.metrics is None:
            self.analyze_all_units()
        
        return [m for m in self.metrics if m.has_place_field]
    
    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics across all units.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if self.metrics is None:
            self.analyze_all_units()
        
        place_cells = self.get_place_cells()
        
        # Collect metrics
        all_si = [m.spatial_information for m in self.metrics]
        all_sparsity = [m.sparsity for m in self.metrics]
        all_n_fields = [m.n_fields for m in self.metrics]
        
        pc_si = [m.spatial_information for m in place_cells]
        pc_sparsity = [m.sparsity for m in place_cells]
        
        return {
            'n_total_units': len(self.metrics),
            'n_place_cells': len(place_cells),
            'place_cell_percentage': len(place_cells) / len(self.metrics) * 100,
            
            # Spatial information
            'mean_si_all': np.mean(all_si),
            'std_si_all': np.std(all_si),
            'mean_si_place_cells': np.mean(pc_si) if pc_si else 0,
            'std_si_place_cells': np.std(pc_si) if pc_si else 0,
            
            # Sparsity
            'mean_sparsity_all': np.mean(all_sparsity),
            'std_sparsity_all': np.std(all_sparsity),
            'mean_sparsity_place_cells': np.mean(pc_sparsity) if pc_sparsity else 0,
            'std_sparsity_place_cells': np.std(pc_sparsity) if pc_sparsity else 0,
            
            # Fields
            'mean_n_fields': np.mean(all_n_fields),
            'units_with_fields': sum(1 for m in self.metrics if m.n_fields > 0),
        }
    
    def save_results(self, filepath: str):
        """Save analysis results to file."""
        import pickle
        
        results = {
            'rate_maps': self.rate_maps,
            'metrics': self.metrics,
            'summary': self.get_summary_statistics(),
            'grid_size': (self.grid_height, self.grid_width),
            'n_units': self.n_units
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load analysis results from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.rate_maps = results['rate_maps']
        self.metrics = results['metrics']
        
        print(f"✓ Results loaded from {filepath}")