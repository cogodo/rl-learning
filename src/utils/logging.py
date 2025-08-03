class Logger:
    def log_episode(self, episdoe, reward, length, loss):
        return NotImplementedError
    
    def log_step(self, step, obs, action, reward, done):
        return NotImplementedError
    
    def save_logs(self, path):
        return NotImplementedError