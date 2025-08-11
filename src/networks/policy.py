from torch import nn
from typing import Tuple, Optional, Any
import torch

class PolicyNetwork(nn.Module):
    """Neural network that outputs action probabilities or action distributions."""
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)
        self.activation = nn.ReLU()


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        out = self.layer1(obs)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        return out

    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample an action from the policy distribution."""
        logits = self.forward(obs)
        
        distr = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = distr.mode
        else:
            action = distr.sample()
        
        log_prob = distr.log_prob(action)

        return action, log_prob
    
    def get_action_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of taking a specific action."""
        logits = self.forward(obs)

        distr = torch.distributions.Categorical(logits=logits)

        return distr.log_prob(action)
    
    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Get entropy of the policy distribution."""
        logits = self.forward(obs)
        distr = torch.distributions.Categorical(logits=logits)

        return distr.entropy


class ValueNetwork(nn.Module):
    """Neural network that estimates state values."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        out = self.layer1(obs)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        return out
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get estimated value for given observation."""
        return self.forward(obs)
         


class QNetwork(nn.Module):
    """Neural network that estimates Q-values for state-action pairs."""
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Q-network."""
        out = self.layer1(obs)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        return out
    
    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Get Q-values for all actions."""
        return self.forward(obs)
    
    def get_max_q_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get maximum Q-value over all possible actions."""
        q_vals = self.get_q_values(obs)
        return torch.max(q_vals, dim=-1)[0]
    
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get best action (greedy selection)."""
        q_values = self.get_q_values(obs)
        return torch.argmax(q_values, dim=-1)