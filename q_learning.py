"""
Q-Learning Agent for Gridworld Navigation
"""
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.
    
    Implements the Q-learning algorithm as specified in the project proposal.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions (4 for gridworld)
            learning_rate: Learning rate (alpha) for Q-updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon per episode
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table: Q[state, action]
        self.q_table = np.zeros((n_states, n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
        # Data for decoder training
        self.training_data = []
    
    def get_q_values(self, state_index: int) -> np.ndarray:
        """
        Get Q-values for a given state.
        
        Args:
            state_index: Flat index of the state
        
        Returns:
            Array of Q-values for each action
        """
        return self.q_table[state_index].copy()
    
    def select_action(self, state_index: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_index: Current state index
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state_index])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """
        Update Q-values using the Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether episode is done
        """
        current_q = self.q_table[state, action]
        
        if done:
            # No future rewards if episode is done
            target_q = reward
        else:
            # Bootstrap from best next action
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_training_data(self, state_position: Tuple[int, int], state_index: int):
        """
        Store data for decoder network training.
        
        Args:
            state_position: (y, x) position in grid
            state_index: Flat state index
        """
        q_values = self.get_q_values(state_index)
        self.training_data.append({
            'position': state_position,
            'q_values': q_values,
            'state_index': state_index
        })
    
    def get_training_data(self) -> List[Dict]:
        """Return collected training data."""
        return self.training_data
    
    def save(self, filepath: str):
        """Save agent state to file."""
        state = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_data': self.training_data
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.q_table = state['q_table']
        self.epsilon = state['epsilon']
        self.episode_rewards = state['episode_rewards']
        self.episode_lengths = state['episode_lengths']
        self.training_data = state['training_data']
        print(f"Agent loaded from {filepath}")


def train_agent(
    env,
    agent: QLearningAgent,
    n_episodes: int = 5000,
    verbose: bool = True,
    verbose_interval: int = 500,
    collect_data_interval: int = 10
) -> Tuple[QLearningAgent, Dict]:
    """
    Train Q-Learning agent in the environment.
    
    Args:
        env: GridWorld environment
        agent: QLearningAgent instance
        n_episodes: Number of training episodes
        verbose: Whether to print training progress
        verbose_interval: Print every N episodes
        collect_data_interval: Collect decoder data every N episodes
    
    Returns:
        Trained agent and training statistics
    """
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rate': [],
        'epsilon_history': []
    }
    
    recent_successes = []
    
    for episode in range(n_episodes):
        state_pos = env.reset()
        state_idx = env.state_to_index(state_pos)
        
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select and take action
            action = agent.select_action(state_idx, training=True)
            next_state_pos, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state_pos)
            
            # Update Q-values
            agent.update(state_idx, action, reward, next_state_idx, done)
            
            # Store training data periodically
            if episode % collect_data_interval == 0:
                agent.store_training_data(state_pos, state_idx)
            
            episode_reward += reward
            episode_length += 1
            
            state_pos = next_state_pos
            state_idx = next_state_idx
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record statistics
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['epsilon_history'].append(agent.epsilon)
        
        # Track success rate (last 100 episodes)
        recent_successes.append(1 if info['reached_goal'] else 0)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        success_rate = np.mean(recent_successes)
        stats['success_rate'].append(success_rate)
        
        # Verbose output
        if verbose and (episode + 1) % verbose_interval == 0:
            avg_reward = np.mean(stats['episode_rewards'][-verbose_interval:])
            avg_length = np.mean(stats['episode_lengths'][-verbose_interval:])
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print()
    
    return agent, stats


def evaluate_agent(env, agent: QLearningAgent, n_episodes: int = 100) -> Dict:
    """
    Evaluate trained agent performance.
    
    Args:
        env: GridWorld environment
        agent: Trained QLearningAgent
        n_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    lengths = []
    successes = []
    
    for _ in range(n_episodes):
        state_pos = env.reset()
        state_idx = env.state_to_index(state_pos)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state_idx, training=False)  # Greedy
            next_state_pos, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state_pos)
            
            episode_reward += reward
            episode_length += 1
            state_idx = next_state_idx
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(1 if info['reached_goal'] else 0)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean(successes)
    }