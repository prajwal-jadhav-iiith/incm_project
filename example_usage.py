"""
Example script demonstrating Q-Learning agent training in gridworld.

This script shows how to:
1. Create different gridworld environments
2. Initialize and train a Q-learning agent
3. Evaluate agent performance
4. Visualize learning curves
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import (
    GridWorld, create_open_field, create_t_maze, 
    create_four_rooms, create_random_barriers
)
from q_learning import QLearningAgent, train_agent, evaluate_agent


def plot_training_curves(stats, save_path=None):
    """Plot training statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'], alpha=0.3, label='Raw')
    # Moving average
    window = 100
    if len(stats['episode_rewards']) >= window:
        moving_avg = np.convolve(
            stats['episode_rewards'], 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0, 0].plot(range(window-1, len(stats['episode_rewards'])), 
                       moving_avg, label=f'{window}-episode MA', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(stats['episode_lengths'], alpha=0.3, label='Raw')
    if len(stats['episode_lengths']) >= window:
        moving_avg = np.convolve(
            stats['episode_lengths'], 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0, 1].plot(range(window-1, len(stats['episode_lengths'])), 
                       moving_avg, label=f'{window}-episode MA', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Success rate
    axes[1, 0].plot(stats['success_rate'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Success Rate (Last 100 Episodes)')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon decay
    axes[1, 1].plot(stats['epsilon_history'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
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
    ax.imshow(grid, cmap='RdYlGn', alpha=0.3, vmin=-1, vmax=1, origin="lower")
    
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
    
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Learned Policy (Q-values as arrows)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy plot saved to {save_path}")
    
    plt.show()


def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 60)
    print("Q-Learning Agent Training for Place Cell Project")
    print("=" * 60)
    print()
    
    # Choose environment type
    print("Available environments:")
    print("1. Open Field (10x10)")
    print("2. T-Maze")
    print("3. Four Rooms")
    print("4. Random Barriers")
    
    env_choice = input("\nSelect environment (1-4) [default: 1]: ").strip() or "1"
    
    if env_choice == "1":
        env = create_open_field(size=10)
        print("\nCreated Open Field environment (10x10)")
    elif env_choice == "2":
        env = create_t_maze()
        print("\nCreated T-Maze environment")
    elif env_choice == "3":
        env = create_four_rooms(room_size=5)
        print("\nCreated Four Rooms environment")
    elif env_choice == "4":
        env = create_random_barriers(size=10, n_barriers=15)
        print("\nCreated Random Barriers environment")
    else:
        env = create_open_field(size=10)
        print("\nCreated Open Field environment (10x10)")
    
    print(f"State space size: {env.get_state_space_size()}")
    print(f"Goal position: {env.goal_position}")
    print()
    
    # Visualize initial environment
    print("Initial environment:")
    env.reset()
    env.render()
    print()
    
    # Initialize agent
    agent = QLearningAgent(
        n_states=env.get_state_space_size(),
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    print("Agent initialized with parameters:")
    print(f"  Learning rate (alpha): {agent.alpha}")
    print(f"  Discount factor (gamma): {agent.gamma}")
    print(f"  Epsilon: {agent.epsilon} -> {agent.epsilon_end}")
    print()
    
    # Train agent
    n_episodes = int(input("Number of training episodes [default: 5000]: ").strip() or "5000")
    
    print(f"\nTraining agent for {n_episodes} episodes...")
    print()
    
    agent, stats = train_agent(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        verbose=True,
        verbose_interval=500,
        collect_data_interval=10
    )
    
    print("\nTraining completed!")
    print()
    
    # Evaluate agent
    print("Evaluating trained agent...")
    eval_stats = evaluate_agent(env, agent, n_episodes=100)
    
    print("\nEvaluation Results (100 episodes, greedy policy):")
    print(f"  Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean steps: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")
    print(f"  Success rate: {eval_stats['success_rate']:.2%}")
    print()
    
    # Visualize results
    print("Generating plots...")
    plot_training_curves(stats, save_path='training_curves.png')
    visualize_q_values(env, agent, save_path='learned_policy.png')
    
    # Save agent
    save = input("\nSave trained agent? (y/n) [default: y]: ").strip().lower() or "y"
    if save == "y":
        agent.save('trained_agent.pkl')
    
    # Print data collection stats
    print(f"\nCollected {len(agent.get_training_data())} data points for decoder training")
    print("This data contains Q-values and positions for training the spatial decoder.")
    print()
    
    print("=" * 60)
    print("Training complete! Next step: Train the spatial decoder network.")
    print("=" * 60)


if __name__ == "__main__":
    main()