from .base import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)

    def _select_action_training(self, obs):
        return self.network.get_action(obs)
    
    def _select_action_evaluation(self, obs):
        return self.network.get_action(obs)