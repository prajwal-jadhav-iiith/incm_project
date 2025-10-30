"""
Gridworld Environment for Place Cell Emergence Project
"""
import numpy as np
from typing import Tuple, List, Optional
from enum import IntEnum


class Action(IntEnum):
    """Available actions in the gridworld."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    """
    2D Gridworld environment for navigation tasks.
    
    The agent navigates a grid to reach a goal location, receiving rewards
    and penalties as specified in the project proposal.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        goal_reward: float = 10.0,
        step_penalty: float = -0.01,
        wall_penalty: float = -0.1,
        max_steps: int = 200,
        random_start: bool = False
    ):
        """
        Initialize the gridworld environment.
        
        Args:
            grid_size: (height, width) of the grid
            goal_position: (y, x) coordinates of goal, random if None
            start_position: (y, x) starting position, random if None
            walls: List of (y, x) wall positions
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step (encourages efficiency)
            wall_penalty: Penalty for hitting walls
            max_steps: Maximum steps per episode
            random_start: Whether to randomize start position each episode
        """
        self.height, self.width = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.max_steps = max_steps
        self.random_start = random_start
        
        # Set goal position
        if goal_position is None:
            self.goal_position = (self.height - 1, self.width - 1)
        else:
            self.goal_position = goal_position
        
        # Set initial start position
        if start_position is None:
            self.initial_start = (0, 0)
        else:
            self.initial_start = start_position
        
        # Set walls
        self.walls = set(walls) if walls else set()
        
        # Current state
        self.agent_position = self.initial_start
        self.step_count = 0
        
        # Action mapping
        self.action_effects = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        
        # For trajectory tracking
        self.trajectory = []
    
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial agent position (y, x)
        """
        if self.random_start:
            # Random start position (not on wall or goal)
            valid_positions = [
                (y, x) for y in range(self.height) for x in range(self.width)
                if (y, x) not in self.walls and (y, x) != self.goal_position
            ]
            self.agent_position = valid_positions[np.random.randint(len(valid_positions))]
        else:
            self.agent_position = self.initial_start
        
        self.step_count = 0
        self.trajectory = [self.agent_position]
        
        return self.agent_position
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        
        Returns:
            Tuple of (next_position, reward, done, info)
        """
        self.step_count += 1
        
        # Calculate new position
        dy, dx = self.action_effects[Action(action)]
        new_y = self.agent_position[0] + dy
        new_x = self.agent_position[1] + dx
        new_position = (new_y, new_x)
        
        # Check if move is valid
        if self._is_valid_position(new_position):
            self.agent_position = new_position
            reward = self.step_penalty
        else:
            # Hit wall or boundary - stay in place
            reward = self.wall_penalty
        
        self.trajectory.append(self.agent_position)
        
        # Check if goal reached
        done = False
        if self.agent_position == self.goal_position:
            reward = self.goal_reward
            done = True
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        info = {
            'step_count': self.step_count,
            'reached_goal': self.agent_position == self.goal_position
        }
        
        return self.agent_position, reward, done, info
    
    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        y, x = position
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return False
        if position in self.walls:
            return False
        return True
    
    def state_to_index(self, position: Tuple[int, int]) -> int:
        """Convert (y, x) position to flat state index."""
        y, x = position
        return y * self.width + x
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat state index to (y, x) position."""
        y = index // self.width
        x = index % self.width
        return (y, x)
    
    def get_state_space_size(self) -> int:
        """Return total number of states."""
        return self.height * self.width
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return the trajectory of the current episode."""
        return self.trajectory.copy()
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """
        Render the current state of the environment.
        
        Args:
            mode: 'human' for print, 'array' for numpy array
        
        Returns:
            Grid array if mode='array', None otherwise
        """
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[:] = '.'
        
        # Mark walls
        for wall in self.walls:
            grid[wall] = '#'
        
        # Mark goal
        grid[self.goal_position] = 'G'
        
        # Mark agent
        grid[self.agent_position] = 'A'
        
        if mode == 'human':
            print('\n' + '=' * (self.width * 2 + 1))
            for row in grid:
                print('|' + ' '.join(row) + '|')
            print('=' * (self.width * 2 + 1))
            print(f"Step: {self.step_count}")
        elif mode == 'array':
            return grid
        
        return None


# Environment factory functions
def create_open_field(size: int = 10) -> GridWorld:
    """Create an open field environment with no obstacles."""
    return GridWorld(
        grid_size=(size, size),
        goal_position=(size-1, size-1),
        start_position=(0, 0)
    )


def create_t_maze(length: int = 10, width: int = 3) -> GridWorld:
    """Create a T-maze environment."""
    height = length
    walls = []
    
    # Create T-maze walls
    # Vertical corridor
    for y in range(height - width):
        for x in range(width):
            if x != width // 2:  # Keep middle open
                walls.append((y, x))
    
    # Horizontal top bar (only middle columns)
    for x in range(width, width * 2):
        for y in range(height - width, height):
            if y != height - width:  # Keep connection open
                continue
    
    return GridWorld(
        grid_size=(height, width * 2),
        goal_position=(0, width * 2 - 1),
        start_position=(height - 1, width // 2),
        walls=walls
    )


def create_four_rooms(room_size: int = 5) -> GridWorld:
    """Create a four-room environment with doorways."""
    size = room_size * 2 + 1  # +1 for wall in middle
    walls = []
    
    # Vertical wall
    for y in range(size):
        if y != room_size // 2 and y != size - room_size // 2 - 1:
            walls.append((y, room_size))
    
    # Horizontal wall
    for x in range(size):
        if x != room_size // 2 and x != size - room_size // 2 - 1:
            walls.append((room_size, x))
    
    return GridWorld(
        grid_size=(size, size),
        goal_position=(size - 1, size - 1),
        start_position=(0, 0),
        walls=walls
    )


def create_random_barriers(size: int = 10, n_barriers: int = 15) -> GridWorld:
    """Create environment with random barrier obstacles."""
    walls = set()
    goal_pos = (size - 1, size - 1)
    start_pos = (0, 0)
    
    # Add random walls
    while len(walls) < n_barriers:
        wall = (np.random.randint(size), np.random.randint(size))
        if wall != goal_pos and wall != start_pos:
            walls.add(wall)
    
    return GridWorld(
        grid_size=(size, size),
        goal_position=goal_pos,
        start_position=start_pos,
        walls=list(walls)
    )