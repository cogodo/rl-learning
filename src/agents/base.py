class BaseAgent:
    def __init__(self, env, config):
        return NotImplementedError
    
    def select_action(self, obs, training=True):
        return NotImplementedError
    
    def update(self, batch):
        return NotImplementedError
    
    def save(self, path):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError
    
    def reset(self):
        return NotImplementedError