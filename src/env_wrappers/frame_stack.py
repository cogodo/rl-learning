from base import BaseWrapper
import numpy as np
from collections import deque
from gymnasium.spaces import Box

class FrameStack(BaseWrapper):
    """Stacks multiple consecutive observations into a single observation."""
    
    def __init__(self, env, config):
        """Initialize with frame stacking configuration."""
        super().__init__(env, config)
        self.num_frames = config.get('frame_stack', 1)
        self.frames = deque(maxlen=self.num_frames)
        
    @property
    def observation_space(self):
        """Get modified observation space for stacked frames."""
        if self.num_frames == 1:
            return self.env.observation_space
        
        original_space = self.env.observation_space
        original_shape = original_space.shape
        new_shape = (self.num_frames,) + original_shape

        original_low = self.env.observation_space.low
        original_high = self.env.observation_space.high
        new_low = np.tile(original_low, (self.num_frames, 1))
        new_high = np.tile(original_high, (self.num_frames, 1))

        new_dtype = self.env.observation_space.dtype

        return Box(
            low=new_low,
            high=new_high,
            dtype=new_dtype,
            shape=new_shape
        )


    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        self._initialize_frames(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action, **kwargs):
        """Take step and update frame stack."""
        obs, reward, done, truncated, info = self.env.step(action, **kwargs)

        self._update_frames(obs)
        
        return self._get_stacked_obs(), reward, done, truncated, info
    
    def _get_stacked_obs(self):
        """Get current stacked observation from frame buffer."""
        if len(self.frames) < self.num_frames:
            while len(self.frames) < self.num_frames:
                self.frames.append(self.frames[-1].copy())
        
        return np.array(list(self.frames))
    
    def _initialize_frames(self, obs):
        """Initialize frame buffer with first observation."""
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs.copy())
    
    def _update_frames(self, obs):
        """Update frame buffer with new observation."""
        self.frames.append(obs.copy())
    
    def get_frame_history(self):
        """Get current frame history for debugging."""
        return list(self.frames)
    
    def clear_frames(self):
        """Clear frame buffer between episodes."""
        self.frames.clear()