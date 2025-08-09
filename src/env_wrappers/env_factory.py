from .base import BaseWrapper
import gymnasium as gym
from .frame_stack import FrameStack
from .normalize_obs import NormalizeObservations  
from .normalize_rewards import NormalizeRewards

WRAPPER_REGISTRY = {
    "base": BaseWrapper,
    "frame_stack": FrameStack,
    "normalize_observations": NormalizeObservations,
    "normalize_rewards": NormalizeRewards
}

class EnvironmentFactory:
    def create_env(self, env_name, config, **kwargs):
        """Main method to create and configure environment."""
        self._validate_config(config, env_name)
        env = self._create_base_env(env_name, **kwargs)
        env = self.apply_wrappers(env, self._get_wrapper_config(config))
        return env

    def apply_wrappers(self, env, wrapper_config, **kwargs):
        """Apply wrapper chain to environment."""
        current_env = env

        for wrapper_name, wrapper_params in wrapper_config.items():
            if wrapper_name in WRAPPER_REGISTRY:
                wrapper_class = WRAPPER_REGISTRY[wrapper_name]
                current_env = wrapper_class(current_env, wrapper_params)

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
        if wrapper_config:
            for wrapper_name in wrapper_config:
                if wrapper_name not in WRAPPER_REGISTRY:
                    errors.append(f"Unknown wrapper: {wrapper_name}")
        
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    def get_available_environments(self):
        """Return list of supported environment names."""
        return ["CartPole-v1", "MountainCar-v0"] #update w more later

    def get_available_wrappers(self):
        """Return list of available wrapper types."""
        return list(WRAPPER_REGISTRY.keys())