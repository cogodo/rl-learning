import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = np.array()
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        return NotImplementedError
    
    def sample(self, batch_size):
        return NotImplementedError
    
    def __len__(self):
        return  len(self.buffer)
    

class EpisodeBuffer:
    def add_step(self, obs, action, reward, done):
        return NotImplementedError
    
    def get_episode(self):
        return NotImplementedError
    
    def clear(self):
        return NotImplementedError
    