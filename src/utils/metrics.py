import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

class MetricsTracker:
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker with sliding window."""
        self.rewards = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        
        self.total_episodes = 0
        self.total_steps = 0
        self.total_reward = 0
    
    def update(self, episode_reward: float, episode_length: int, loss: Optional[float] = None):
        """Update metrics with new episode data."""
        self.rewards.append(episode_reward)
        self.lengths.append(episode_length)
        if loss: 
            self.losses.append(loss)
        self.total_episodes += 1
        self.total_steps += episode_length
        self.total_reward += episode_reward
    def get_episode_stats(self) -> Dict[str, Optional[Dict[str, float]]]:
        """Get current episode statistics."""
        return {
            "reward_stats": self._calculate_stats(self.rewards),
            "length_stats": self._calculate_stats(self.lengths),
            "losses_stats": self._calculate_stats(self.losses) if self.losses else None
        }
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get overall training statistics."""
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "mean_reward": self.total_reward/self.total_episodes if self.total_episodes > 0 else 0.0
        }

    def reset(self):
        """Reset all metrics."""
        self.rewards.clear()
        self.lengths.clear()
        self.losses.clear()
        self.total_episodes = 0
        self.total_steps = 0
        self.total_reward = 0
    
    def _calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not data:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }