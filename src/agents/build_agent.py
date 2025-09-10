from typing import Any, Mapping

from .random import RandomAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .sarsa import SARSAAgent
from .grpo import GRPOAgent

def build_agent(env: Any, config: Mapping[str, Any]) -> Any:
    """Construct and return an agent instance suitable for `env`.

    This factory should inspect `config['algorithm']` to select the appropriate
    agent type (e.g., DQN, SARSA, PPO, GRPO), instantiate it with `env` and
    any relevant hyperparameters, and return the initialized agent.
    """
    
    agent_type = config['algorithm']['type']
    return RandomAgent(env, config)
    # if agent_type == "random":
    #     return RandomAgent(env, config)
    # elif agent_type == "dqn":
    #     return DQNAgent(env, config)
    # elif agent_type == "ppo":
    #     return PPOAgent(env, config)
    # elif agent_type == "grpo":
    #     return GRPOAgent(env, config)
    # else:
    #     raise ValueError(f"Invalid agent type: {agent_type}")