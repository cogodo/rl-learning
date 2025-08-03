class ReplayBuffer:
    def add(self, obs, action, reward, next_obs, done):
        return NotImplementedError
    
    def sample(self, batch_size):
        return NotImplementedError
    
    def __len__(self):
        return NotImplementedError
    

class EpisodeBuffer:
    def add_step(self, obs, action, reward, done):
        return NotImplementedError
    
    def get_episode(self):
        return NotImplementedError
    
    def clear(self):
        return NotImplementedError
    