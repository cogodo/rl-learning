class BaseWrapper:
    def __init__(self, env, config):
        return NotImplementedError
    
    def reset(self):
        return NotImplementedError
    
    def step(self, action):
        return NotImplementedError
    
    def close(self):
        return NotImplementedError
    
    