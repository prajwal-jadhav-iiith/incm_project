"""
Debug script to diagnose dimension mismatch issues.
"""

import pickle
from decoder import load_decoder
from q_learning import QLearningAgent
from environment import GridWorld

print("=" * 70)
print("DEBUGGING DIMENSION MISMATCH")
print("=" * 70)

# Load decoder metadata
print("\n1. Loading decoder metadata...")
decoder, metadata = load_decoder('trained_decoder.pth')
print(f"   Metadata grid_size: {metadata.get('grid_size', 'NOT FOUND')}")
grid_height, grid_width = metadata.get('grid_size', (10, 10))
print(f"   Height: {grid_height}, Width: {grid_width}")
print(f"   Expected states: {grid_height * grid_width}")

# Load agent
print("\n2. Loading agent...")
with open('trained_agent.pkl', 'rb') as f:
    agent_state = pickle.load(f)

q_table_shape = agent_state['q_table'].shape
print(f"   Q-table shape: {q_table_shape}")
print(f"   Number of states: {q_table_shape[0]}")
print(f"   Number of actions: {q_table_shape[1]}")

# Create environment
print("\n3. Creating environment...")
env = GridWorld(
    grid_size=(grid_height, grid_width),
    goal_position=(grid_height - 1, grid_width - 1),
    start_position=(0, 0),
    walls=[]
)
print(f"   env.height: {env.height}")
print(f"   env.width: {env.width}")
print(f"   env.get_state_space_size(): {env.get_state_space_size()}")

# Check state index conversion
print("\n4. Testing state index conversion...")
print(f"   Position (0, 0) -> index: {env.state_to_index((0, 0))}")
print(f"   Position ({grid_height-1}, {grid_width-1}) -> index: {env.state_to_index((grid_height-1, grid_width-1))}")

# Test the problematic case
print("\n5. Testing all positions...")
max_idx = -1
problematic_positions = []

for y in range(env.height):
    for x in range(env.width):
        idx = env.state_to_index((y, x))
        if idx > max_idx:
            max_idx = idx
        if idx >= q_table_shape[0]:
            problematic_positions.append(((y, x), idx))

print(f"   Maximum state index found: {max_idx}")
print(f"   Q-table has indices: 0 to {q_table_shape[0] - 1}")

if problematic_positions:
    print(f"\n❌ PROBLEM FOUND!")
    print(f"   {len(problematic_positions)} positions would be out of bounds:")
    for pos, idx in problematic_positions[:5]:  # Show first 5
        print(f"      Position {pos} -> Index {idx} (out of bounds!)")
else:
    print(f"\n✅ All positions are within bounds!")

# Check if dimensions match
print("\n6. Dimension consistency check...")
checks = {
    "Metadata states == Q-table states": 
        grid_height * grid_width == q_table_shape[0],
    "Environment states == Q-table states": 
        env.get_state_space_size() == q_table_shape[0],
    "Max index < Q-table size": 
        max_idx < q_table_shape[0],
}

for check_name, passed in checks.items():
    print(f"   {'✅' if passed else '❌'} {check_name}: {passed}")

# Recommendation
print("\n" + "=" * 70)
if all(checks.values()):
    print("✅ All checks passed! Dimensions are consistent.")
    print("The error must be elsewhere. Let's check the analysis.py code.")
else:
    print("❌ Dimension mismatch detected!")
    print("\nRECOMMENDATION:")
    print("The agent was trained on a different environment than what's")
    print("being recreated. You need to either:")
    print("  1. Retrain from scratch with consistent dimensions")
    print("  2. Fix the environment recreation to match training")
    
    print(f"\nTo retrain with correct dimensions:")
    print(f"  python example_usage.py")
    print(f"  python train_decoder.py")
    print(f"  python run_analysis.py")

print("=" * 70)