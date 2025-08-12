import numpy as np
from typing import List, Tuple, Optional, Any
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size: int, min_size: int = 0):
        """Initialize replay buffer with specified maximum capacity."""
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.min_size = int(min_size)

    def add(self, obs: Any, action: Any, reward: float, next_obs: Any, done: bool) -> None:
        """Add a transition tuple to the replay buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Randomly sample a batch of transitions from the buffer."""
        size = len(self.buffer)
        if size < self.min_size:
            raise ValueError(f"Buffer size {size} < min_size {self.min_size}")
        if batch_size > size:
            raise ValueError(f"Requested batch_size {batch_size} > buffer size {size}")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer has reached maximum capacity."""
        return len(self.buffer) == self.max_size

    def get_buffer_size(self) -> int:
        """Get current number of stored transitions."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all stored transitions from buffer."""
        self.buffer.clear()

    def get_recent_transitions(self, n: int) -> List[Tuple]:
        """Get the n most recent transitions added to buffer."""
        return list(self.buffer)[-n:]

    def get_all_transitions(self) -> List[Tuple]:
        """Get all stored transitions (useful for saving/loading)."""
        return list(self.buffer)
    
    def ready(self) -> bool:
        return len(self.buffer) >= self.min_size

class EpisodeBuffer:
    def __init__(self):
        """Initialize an empty episode buffer."""
        self.buffer = deque()
        self.current_episode = []

    def add_step(self, obs: Any, action: Any, reward: float, done: bool) -> None:
        """Add a single step to the current episode."""
        self.current_episode.append((obs, action, reward, done))
        if done == True:
            self.buffer.append(self.current_episode)
            self.current_episode.clear()

    def get_episode(self) -> Optional[List[Tuple]]:
        """Retrieve the most recently completed episode."""
        if self.buffer:
            return self.buffer[-1] 
        else:
            raise IndexError("Episode buffer empty")

    def clear(self) -> None:
        """Remove all stored episodes and clear current episode."""
        self.buffer.clear()
        self.current_episode.clear()

    def get_all_episodes(self) -> List[List[Tuple]]:
        """Get all completed episodes stored in buffer."""
        return list(self.buffer)

    def get_episode_count(self) -> int:
        """Get total number of completed episodes."""
        return len(self.buffer)

    def get_latest_episode_length(self) -> int:
        """Get length of the most recent completed episode."""
        if self.buffer:
            return len(self.buffer[-1])
        else:
            return 0
        
    def get_episode_rewards(self) -> List[float]:
        """Get list of total rewards for all completed episodes."""
        if self.buffer:
            return [sum(step[2] for step in episode) for episode in self.buffer]
        else:
            return []
        
    def get_episode_lengths(self) -> List[int]:
        """Get list of lengths for all completed episodes."""
        if self.buffer:
            return [len(episode) for episode in self.buffer]
        else:
            return []

    def reset_current_episode(self) -> None:
        """Clear current episode without storing it (useful for failed episodes)."""
        self.current_episode.clear()
