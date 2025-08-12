from .base import BaseWrapper
import numpy as np
from gymnasium.spaces import Box

class NormalizeObservations(BaseWrapper):
    def __init__(self, env, config):
        super().__init__(env, config)
        shape = self.env.observation_space.shape
        self.epsilon = float(config.get("epsilon", 1e-8))

        self.momentum = float(config.get("momentum", 0.99))
        self.reset_stats_on_episode = bool(config.get("reset_stats_on_episode", False))
        self.training = True
        
        self.running_mean = np.zeros(shape, dtype=np.float32)
        self.running_std = np.ones(shape, dtype=np.float32)

    @property
    def observation_space(self):
        original_space = self.env.observation_space

        return Box(
            low = -np.inf,
            high = np.inf,
            shape = original_space.shape,
            dtype=np.float32
        )

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        if self.training:
            self._update_stats(obs)
        normalized_obs = self._normalize(obs)
        return normalized_obs.astype(np.float32), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.training and self.reset_stats_on_episode:
            self._reset_stats()
        normalized_obs = self._normalize(obs)
        return normalized_obs.astype(np.float32), info
    
    def _update_stats(self, obs):
        """Updates the running mean and std"""
        m = self.momentum
        old_mean = self.running_mean
        old_var = self.running_std ** 2
        new_mean = m * old_mean + obs * (1 - m)
        new_var = old_var * m + (obs - old_mean) ** 2 * (1 - m)
        new_std = np.sqrt(new_var)
        self.running_mean = new_mean.astype(np.float32)
        self.running_std = new_std.astype(np.float32)


    def _normalize(self, obs):
        """Normalize observations to have mean 0 and std 1"""
        return (obs - self.running_mean) / (self.running_std + self.epsilon)
    
    def save_stats(self, path):
        np.save(f"{path}/running_mean.npy", self.running_mean)
        np.save(f"{path}/running_std.npy", self.running_std)


    def load_stats(self, path):
        self.running_mean = np.load(f"{path}/running_mean.npy").astype(np.float32)
        self.running_std = np.load(f"{path}/running_std.npy").astype(np.float32)

    def _reset_stats(self):
        shape = self.env.observation_space.shape
        self.running_mean = np.zeros(shape, dtype=np.float32)
        self.running_std = np.ones(shape, dtype=np.float32)

    def set_training(self, training: bool) -> None:
        self.training = bool(training)