import torch
from ..networks.policy import ValueNetwork, PolicyNetwork, QNetwork
import numpy as np
from src.utils.metrics import MetricsTracker
class BaseAgent:
    """Base class for all reinforcement learning agents."""
    
    def __init__(self, env, config):
        """Initialize agent with environment and configuration."""
        self.env = env
        self.config = config
        self.agent_algo = config["algorithm"]["type"]
        metrics_cfg = self.config.get("training", {}).get("metrics", {})
        self.metrics = MetricsTracker(window_size=metrics_cfg.get("window_size", 100))
     
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        self._parse_configs()

        self._create_networks()

        self._setup_optimizer()

    def _parse_configs(self):
        network_config = self.config.get("network", {})
        self.network_kwargs = {
            "hidden_size": network_config.get("hidden_size"),
            "learning_rate": network_config.get("learning_rate"),
            "weight_decay": network_config.get("weight_decay")
        }

        self.network_config = network_config
        self.training_config = self.config.get("training", {})
        
    def select_action(self, obs, training=True):
        """Select an action given an observation."""
        obs = self._preprocess_obs(obs)

        if training:
            action, info = self._select_action_training(obs)
        else:
            action, info = self._select_action_evaluation(obs)

        return action, info

    def _preprocess_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs
    
    def _postprocess_action(self, action, info):
        """Common action postprocessing."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action, info

    # Subclasses implement these:
    def _select_action_training(self, obs):
        """Subclass implements training action selection."""
        raise NotImplementedError

    def _select_action_evaluation(self, obs):
        """Subclass implements evaluation action selection."""
        raise NotImplementedError
    
    def update(self, batch):
        """Update agent parameters using a batch of experience."""
        raise NotImplementedError("Subclass implements updates")
    
    def save(self, path):
        """Save agent model to specified path."""
        save_dict = {
            'network_state': self._get_network_state(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'training_state': self._get_training_state()
        }
        torch.save(save_dict, path)

    def load(self, path):
        """Load agent model from specified path."""
        checkpoint = torch.load(path, map_location=self.device)
        self._load_network_state(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self._load_training_state(checkpoint['training_state'])
    
    def reset(self):
        """Reset agent to initial state."""
        self.metrics.reset()

    def _create_networks(self):
        """Factory method to create networks."""
        if 'type' in self.network_config:
            # Single network (DQN, SARSA)
            self.network = self._create_single_network()
        else:
            # Dual networks (PPO) - may need more validation + exceptions
            self.policy_network, self.value_network = self._create_dual_networks()
    
    def _create_single_network(self):
        """Create single network (e.g., QNetwork for DQN/SARSA)."""
        hidden_size = self.network_config.hidden_size
        if hasattr(self.obs_space, 'shape'):
            input_size = self.obs_space.shape[0]
        elif hasattr(self.obs_space, 'n'):
            input_size = self.obs_space.n
        else:
            raise ValueError(f"Unsupported obs space: {self.obs_space}")
        
        if hasattr(self.action_space, 'n'):
            # For Discrete action spaces
            output_size = self.action_space.n
        elif hasattr(self.action_space, 'shape'):
            output_size = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
        
        if self.network_config.type == "QNetwork":
            return QNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=output_size)
        elif self.network_config.type == "PolicyNetwork":
            return PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=output_size)
        elif self.network_config.type == "ValueNetwork":
            return ValueNetwork(input_size=input_size, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unsupported Network type: {self.network_config.type}")
    
    def _create_dual_networks(self):
        """Create dual networks (e.g., PolicyNetwork and ValueNetwork for PPO)."""
        hidden_size = self.network_config.hidden_size
        if hasattr(self.obs_space, 'shape'):
            input_size = self.obs_space.shape[0]
        elif hasattr(self.obs_space, 'n'):
            input_size = self.obs_space.n
        else:
            raise ValueError(f"Unsupported obs space: {self.obs_space}")
        
        if hasattr(self.action_space, 'n'):
            # For Discrete action spaces
            output_size = self.action_space.n
        elif hasattr(self.action_space, 'shape'):
            output_size = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
        
        if self.network_config.get("policy_type") == "PolicyNetwork":
            polnet = PolicyNetwork(input_size=input_size, hidden_size=hidden_size, num_actions=output_size)
        else:
            raise ValueError(f"Unsupported policy type: {self.network_config.get('policy_type')}")
        if self.network_config.get("value_type") == "ValueNetwork":
            valnet = ValueNetwork(input_size=input_size, hidden_size=hidden_size)
        else:
            raise ValueError(f"Unsupported value type: {self.network_config.get('value_type')}")
        return polnet, valnet

    def _setup_optimizer(self):
        """Setup the optimizer with learning rate and weight decay."""
        opt_name = self.network_config.get("optimizer", "Adam")
        lr = self.network_kwargs.get("learning_rate", 0.001)
        wd = self.network_kwargs.get("weight_decay", 0.0)
    # make sure networks exist and are on device
        if "type" in self.network_config:  # single net
            self.network.to(self.device)
            params = self.network.parameters()
        else:  # dual nets
            self.policy_network.to(self.device)
            self.value_network.to(self.device)
            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())

        self.optimizer = getattr(torch.optim, opt_name)(params, lr=lr, weight_decay=wd)
        
    def _get_network_state(self):
        """Get network state for saving."""
        if 'type' in self.network_config:
            #single network
            return self.network.state_dict()
        else:
            #dual network
            return { 
                'policy': self.policy_network.state_dict(),
                'value': self.value_network.state_dict()
            }
    
    def _get_training_state(self):
        """Get training state for saving. Subclass may override"""
        m = self.metrics
        return {
            "metrics": {
                "window_size": m.rewards.maxlen,
                "total_episodes": m.total_episodes,
                "total_steps": m.total_steps,
                "total_reward": m.total_reward,
                "reward_window": list(m.rewards),
                "length_window": list(m.lengths),
                "loss_window": list(m.losses)
            }
        }
    
    def _load_network_state(self, network_state):
        """Load network state from checkpoint. Subclass may override

        TODO(base-default): Provide default that restores single or dual networks.
        Subclasses may override for additional components.
        """
        if "type" in self.network_config:
            self.network.load_state_dict(network_state)
            self.network.to(self.device)
        else:
            self.policy_network.load_state_dict(network_state["policy"])
            self.value_network.load_state_dict(network_state["value"])
            self.policy_network.to(self.device)
            self.value_network.to(self.device)
    
    def _load_training_state(self, training_state):
        """Load training state from checkpoint. Subclass may override"""
        m_state = training_state.get("metrics", {})
        self.metrics = MetricsTracker(window_size=m_state.get("window_size", 100))
        for r in m_state.get("reward_window", []):
            self.metrics.rewards.append(r)
        for l in m_state.get("length_window", []):
            self.metrics.lengths.append(l)
        for x in m_state.get("loss_window", []):
            self.metrics.losses.append(x)
        self.metrics.total_episodes = m_state.get("total_episodes", 0)
        self.metrics.total_steps = m_state.get("total_steps", 0)
        self.metrics.total_reward = m_state.get("total_reward", 0)
    
