from .base import BaseWrapper
import gymnasium as gym
from .frame_stack import FrameStack
from .normalize_obs import NormalizeObservations  
from .normalize_rewards import NormalizeRewards

WRAPPER_REGISTRY = {
    "base": BaseWrapper,
    "frame_stack": FrameStack,
    "normalize_observations": NormalizeObservations,
    "normalize_rewards": NormalizeRewards,
}

# Canonical default order: observation transforms → temporal transforms → reward transforms
DEFAULT_WRAPPER_ORDER = [
    "normalize_observations",
    "frame_stack",
    "normalize_rewards",
]

class EnvironmentFactory:
    def create_env(self, env_name, config, **kwargs):
        """Main method to create and configure environment."""
        self._validate_config(config, env_name)
        env_kwargs = {}
        env_cfg = (config or {}).get("environment", {})

        if env_cfg.get("render_mode") is not None:
            env_kwargs["render_mode"] = env_cfg["render_mode"]

        env = self._create_base_env(env_name, **{**env_kwargs, **kwargs})
        env = self.apply_wrappers(env, self._get_wrapper_config(config))
        return env

    def apply_wrappers(self, env, wrapper_config, **kwargs):
        """Apply wrapper chain to environment in a deterministic order."""
        current_env = env

        # Resolve explicit order from config or fall back to canonical default
        resolved_order = self._resolve_wrapper_order(wrapper_config)

        # First, apply wrappers in the resolved order
        for wrapper_name in resolved_order:
            params = wrapper_config.get(wrapper_name, {}) if isinstance(wrapper_config, dict) else {}
            if not params or not params.get("enabled", False):
                continue
            if wrapper_name not in WRAPPER_REGISTRY:
                continue
            wrapper_class = WRAPPER_REGISTRY[wrapper_name]
            current_env = wrapper_class(current_env, params)

        # Then, apply any remaining enabled wrappers not mentioned in the order, sorted for determinism
        if isinstance(wrapper_config, dict):
            extras = [
                name
                for name in wrapper_config.keys()
                if name not in resolved_order and name != "order"
            ]
            for name in sorted(extras):
                params = wrapper_config.get(name, {})
                if not params or not params.get("enabled", False):
                    continue
                if name in WRAPPER_REGISTRY:
                    current_env = WRAPPER_REGISTRY[name](current_env, params)

        return current_env

    def _create_base_env(self, env_name, **kwargs):
        """Create the base environment without wrappers."""
        try:
            return gym.make(env_name, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create environment '{env_name}': {e}")
        
    def _get_wrapper_config(self, config):
        """Extract wrapper configuration from main config."""
        return config.get("environment", {}).get("wrappers", {})

    def _resolve_wrapper_order(self, wrapper_config):
        """Determine wrapper application order from config or use default."""
        if isinstance(wrapper_config, dict):
            order = wrapper_config.get("order")
            if isinstance(order, list):
                # Only keep names that are known wrappers to avoid typos breaking flow
                return [name for name in order if name in WRAPPER_REGISTRY]
        return list(DEFAULT_WRAPPER_ORDER)

    def _validate_config(self, config, env_name=None):
        """Validate that config has required fields."""
        errors = []

        # Validate environment name
        if env_name:
            try:
                gym.make(env_name)
            except Exception as e:
                errors.append(f"Environment '{env_name}' not found: {e}")
        
        # Validate config structure
        if "environment" not in config:
            errors.append("Missing 'environment' section in config")
        else:
            env_config = config["environment"]
            
            # Check for env_name in config if not provided separately
            if not env_name and "env_name" not in env_config:
                errors.append("Missing 'env_name' in environment config")

        # Validate wrappers
        wrapper_config = self._get_wrapper_config(config)
        if isinstance(wrapper_config, dict):
            # Validate explicit order list if present
            if "order" in wrapper_config and not isinstance(wrapper_config["order"], list):
                errors.append("'environment.wrappers.order' must be a list if provided")
            if isinstance(wrapper_config.get("order"), list):
                for name in wrapper_config["order"]:
                    if name not in WRAPPER_REGISTRY:
                        errors.append(f"Unknown wrapper in order: {name}")

            # Validate wrapper names (excluding the special 'order' key)
            for wrapper_name in wrapper_config:
                if wrapper_name == "order":
                    continue
                if wrapper_name not in WRAPPER_REGISTRY:
                    errors.append(f"Unknown wrapper: {wrapper_name}")
            
            for name, params in wrapper_config.items():
                if name == "order" or not isinstance(params, dict) or not params.get("enabled", False):
                    continue
                if name == "frame_stack":
                    v = params.get("frame_stack", None)
                    if v is None or not isinstance(v, int) or v < 1:
                        errors.append("Invalid 'environment.wrappers.frame_stack.frame_stack' (int >= 1 required)")
                if name == "normalize_observations":
                    if "epsilon" in params and not isinstance(params["epsilon"], (int, float)):
                        errors.append("Invalid 'environment.wrappers.normalize_observations.epsilon' (number required)")
                    if "momentum" in params and not isinstance(params["momentum"], (int, float)):
                        errors.append("Invalid 'environment.wrappers.normalize_observations.momentum' (number required)")
                    if "reset_stats_on_episode" in params and not isinstance(params["reset_stats_on_episode"], bool):
                        errors.append("Invalid 'environment.wrappers.normalize_observations.reset_stats_on_episode' (bool required)")
                if name == "normalize_rewards":
                    if "epsilon" in params and not isinstance(params["epsilon"], (int, float)):
                        errors.append("Invalid 'environment.wrappers.normalize_rewards.epsilon' (number required)")

        
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    def get_available_environments(self):
        """Return list of supported environment names."""
        try:
            from gymnasium.envs.registration import registry

            return sorted(list(registry.keys()))
        except Exception:
            try:
                return sorted([spec.id for spec in gym.envs.registry.values()])
            except Exception:
                return []

    def get_available_wrappers(self):
        """Return list of available wrapper types."""
        return list(WRAPPER_REGISTRY.keys())