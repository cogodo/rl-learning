import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

class MetricsTracker:
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker with sliding window."""
        pass
    
    def update(self, episode_reward: float, episode_length: int, loss: Optional[float] = None):
        """Update metrics with new episode data."""
        pass
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get current episode statistics."""
        pass
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get overall training statistics."""
        pass
    
    def reset(self):
        """Reset all metrics."""
        pass
    
    def _calculate_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        pass