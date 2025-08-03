from torch import nn

class PolicyNetwork(nn.Module):
    def forward(self, obs):
        return NotImplementedError
    
    def get_action(self, obs, deterministic=False):
        return NotImplementedError
    
class ValueNetwork(nn.Module):
    def forward(self, obs):
        return NotImplementedError
    
class QNetwork(nn.Module):
    def forward(self, obs):
        return NotImplementedError
    
    def get_q_values(self, obs):
        return NotImplementedError
    
    