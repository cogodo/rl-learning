class Trainer:
    def __init__(self, agent, env, config):
        return NotImplementedError
    
    def train_episode(self):
        return NotImplementedError
    
    def evaluate(self, num_episodes):
        return NotImplementedError
    
    def train(self, num_episodes):
        return NotImplementedError
    
    def save_checkpoint(self, path):
        return NotImplementedError