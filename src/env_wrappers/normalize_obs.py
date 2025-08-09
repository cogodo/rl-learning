from base import BaseWrapper
import numpy as np
from math import sqrt
from gymnasium.spaces import Box

class NormalizeObservations(BaseWrapper):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.running_mean = np.zeros(self.env.observation_space.shape)
        self.running_std = np.ones(self.running_mean.shape)
        self.epsilon = 1e-8
        self.training = True

    @property
    def observation_space(self):
        original_space = self.env.observation_space

        return Box(
            low = -np.inf,
            high = np.inf,
            shape = original_space.shape,
            dtype=original_space.dtype
        )

    def step(self, action, **kwargs):
        obs, reward, done, truncated, info = self.env.step(action, **kwargs)
        if self.training:
            self._update_stats(obs)
        normalized_obs = self._normalize(obs)
        return normalized_obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        normalized_obs = self._normalize(obs)
        return normalized_obs, info
    
    def _update_stats(self, obs):
        """Updates the running mean and std"""
        old_mean = self.running_mean
        old_var = self.running_std ** 2
        new_mean = old_mean * 0.99 + obs * 0.01
        new_var = old_var * 0.99 + (obs - old_mean) ** 2 * 0.01
        new_std = sqrt(new_var)
        self.running_mean = new_mean
        self.running_std = new_std


    def _normalize(self, obs):
        """Normalize observations to have mean 0 and std 1"""
        return (obs - self.running_mean) / (self.running_std + self.epsilon)
    
    def save_stats(self, path):
        np.save(f"{path}/running_mean.npy", self.running_mean)
        np.save(f"{path}/running_std.npy", self.running_std)


    def load_stats(self, path):
        self.running_mean = np.load(f"{path}/running_mean.npy")
        self.running_std = np.load(f"{path}/running_std.npy")

