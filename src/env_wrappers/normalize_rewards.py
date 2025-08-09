from base import BaseWrapper
import numpy as np
from math import sqrt

class NormalizeRewards(BaseWrapper):
    def __init__(self, env, config):
        """Initialize reward normalization wrapper."""
        super().__init__(env, config)

        self.running_mean = 0.0
        self.running_std = 1.0
        self.epsilon = 1e-8
        self.training = True

    def step(self, action, **kwargs):
        """Take a step and normalize rewards"""
        obs, reward, done, truncated, info = self.env.step(action, **kwargs)
        if self.training:
            self._update_stats(reward)
        normalized_reward = self._normalize(reward)
        return obs, normalized_reward, done, truncated, info
    
    def _update_stats(self, reward):
        """Update running statistics for reward normalization."""
        old_mean = self.running_mean
        old_var = self.running_std ** 2

        self.running_mean = old_mean * 0.99 + reward * 0.01
        new_var = old_var * 0.99 + (reward - old_mean) ** 2 * 0.01
        self.running_std = sqrt(new_var)

    def _normalize_reward(self, reward):
        """Normalize reward to have mean 0 and std 1."""
        return (reward - self.running_mean) / (self.running_std + self.epsilon)
    
    def save_stats(self, path):
        """Save reward normalization statistics to file."""
        np.save(f"{path}/reward_running_mean.npy", self.running_mean)
        np.save(f"{path}/reward_running_std.npy", self.running_std)

    def load_stats(self, path):
        """Load reward normalization statistics from file."""
        self.running_mean = np.load(f"{path}/reward_running_mean.npy")
        self.running_std = np.load(f"{path}/reward_running_std.npy")