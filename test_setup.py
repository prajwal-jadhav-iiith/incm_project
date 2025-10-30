"""
Quick test script to verify environment and agent setup.

Run this to ensure everything is working before full training.
"""

import numpy as np
from environment import GridWorld, create_open_field, Action
from q_learning import QLearningAgent, train_agent, evaluate_agent


def test_environment():
    """Test basic environment functionality."""
    print("Testing Environment...")
    print("-" * 40)
    
    # Create simple environment
    env = create_open_field(size=5)
    
    # Test reset
    pos = env.reset()
    print(f"✓ Environment reset to position: {pos}")
    
    # Test rendering
    print("\n✓ Environment visualization:")
    env.render()
    
    # Test actions
    print("\n✓ Testing actions:")
    for action_name, action in [("UP", Action.UP), ("RIGHT", Action.RIGHT), 
                                 ("DOWN", Action.DOWN), ("LEFT", Action.LEFT)]:
        env.reset()
        next_pos, reward, done, info = env.step(action)
        print(f"  {action_name}: {pos} -> {next_pos}, reward={reward:.3f}")
    
    # Test episode
    print("\n✓ Testing full episode:")
    env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action = np.random.randint(4)  # Random action
        _, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"  Episode completed in {steps} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Reached goal: {done}")
    
    print("\n✅ Environment tests passed!\n")
    return True


def test_agent():
    """Test Q-learning agent."""
    print("Testing Q-Learning Agent...")
    print("-" * 40)
    
    # Create environment
    env = create_open_field(size=5)
    
    # Create agent
    agent = QLearningAgent(
        n_states=env.get_state_space_size(),
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95
    )
    
    print(f"✓ Agent created with {agent.n_states} states and {agent.n_actions} actions")
    print(f"  Q-table shape: {agent.q_table.shape}")
    
    # Test action selection
    state_idx = env.state_to_index((0, 0))
    action = agent.select_action(state_idx)
    print(f"\n✓ Action selection works: state=0 -> action={action}")
    
    # Test Q-value update
    q_before = agent.get_q_values(state_idx).copy()
    agent.update(state_idx, action, reward=1.0, next_state=state_idx+1, done=False)
    q_after = agent.get_q_values(state_idx)
    print(f"\n✓ Q-value update works:")
    print(f"  Before: {q_before}")
    print(f"  After:  {q_after}")
    
    # Test short training
    print(f"\n✓ Testing short training (100 episodes)...")
    agent, stats = train_agent(env, agent, n_episodes=100, verbose=False)
    
    print(f"  Episodes completed: {len(stats['episode_rewards'])}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Final success rate: {stats['success_rate'][-1]:.2%}")
    print(f"  Data points collected: {len(agent.get_training_data())}")
    
    # Test evaluation
    print(f"\n✓ Testing evaluation...")
    eval_stats = evaluate_agent(env, agent, n_episodes=10)
    print(f"  Mean reward: {eval_stats['mean_reward']:.2f}")
    print(f"  Success rate: {eval_stats['success_rate']:.1%}")
    
    print("\n✅ Agent tests passed!\n")
    return True


def test_data_collection():
    """Test data collection for decoder training."""
    print("Testing Data Collection for Decoder...")
    print("-" * 40)
    
    env = create_open_field(size=5)
    agent = QLearningAgent(n_states=env.get_state_space_size(), n_actions=4)
    
    # Train briefly
    agent, _ = train_agent(env, agent, n_episodes=50, verbose=False, collect_data_interval=5)
    
    # Check collected data
    data = agent.get_training_data()
    print(f"✓ Collected {len(data)} data points")
    
    if len(data) > 0:
        sample = data[0]
        print(f"\n✓ Sample data point:")
        print(f"  Position: {sample['position']}")
        print(f"  Q-values: {sample['q_values']}")
        print(f"  State index: {sample['state_index']}")
        
        # Verify data structure
        assert 'position' in sample, "Missing position"
        assert 'q_values' in sample, "Missing q_values"
        assert len(sample['q_values']) == 4, "Wrong number of Q-values"
        print("\n✓ Data structure is correct")
    
    print("\n✅ Data collection tests passed!\n")
    return True


def quick_training_demo():
    """Run a quick training demo."""
    print("Quick Training Demo...")
    print("-" * 40)
    print("Training agent for 500 episodes on 5x5 grid...")
    print()
    
    env = create_open_field(size=5)
    agent = QLearningAgent(
        n_states=env.get_state_space_size(),
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95
    )
    
    agent, stats = train_agent(
        env, agent, 
        n_episodes=500, 
        verbose=True, 
        verbose_interval=100
    )
    
    # Show final performance
    print("\nFinal Performance:")
    eval_stats = evaluate_agent(env, agent, n_episodes=20)
    print(f"  Success rate: {eval_stats['success_rate']:.1%}")
    print(f"  Mean steps: {eval_stats['mean_length']:.1f}")
    
    # Show learned policy
    print("\nLearned Policy (arrows show best action):")
    print()
    env.reset()
    
    for y in range(env.height):
        row = ""
        for x in range(env.width):
            pos = (y, x)
            if pos == env.goal_position:
                row += " G "
            elif pos in env.walls:
                row += " # "
            else:
                state_idx = env.state_to_index(pos)
                q_values = agent.get_q_values(state_idx)
                best_action = np.argmax(q_values)
                arrows = ["↑", "↓", "←", "→"]
                row += f" {arrows[best_action]} "
        print(row)
    
    print("\n✅ Demo complete!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Setup Tests for Place Cell Project")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        test_environment()
        test_agent()
        test_data_collection()
        
        # Run demo
        run_demo = input("Run quick training demo? (y/n) [default: y]: ").strip().lower() or "y"
        if run_demo == "y":
            print()
            quick_training_demo()
        
        print("=" * 60)
        print("✅ All tests passed! Setup is ready.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the full training: python example_usage.py")
        print("2. Implement the decoder network (Phase 2)")
        print("3. Analyze emergent place cells (Phase 3)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()