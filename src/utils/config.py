import os
import yaml
class ConfigManager:
    def __init__(self, base_config_path="configs", load_defaults=True):
        self.base_config_path = base_config_path

        if not os.path.exists(base_config_path):
            raise FileNotFoundError(f"Base config directory not found: {base_config_path}")
        if not os.path.isdir(base_config_path):
            raise NotADirectoryError(f"Path is not a directory: {base_config_path}")
        
        self.algo_path = os.path.join(base_config_path, "algos")
        self.envs_path = os.path.join(base_config_path, "envs")

        if not os.path.exists(self.algo_path):
            raise FileNotFoundError(f"Algo config directory not found: {self.algo_path}")
        if not os.path.exists(self.envs_path):
            raise FileNotFoundError(f"Envs config directory not found: {self.envs_path}")
        
        if not os.path.isdir(self.algo_path):
            raise NotADirectoryError(f"Path is not a directory: {self.algo_path}")
        if not os.path.isdir(self.envs_path):
            raise NotADirectoryError(f"Path is not a directory: {self.envs_path}")
        
        self._config_cache = {}

        if load_defaults == True:
            self.defaults = self._load_yaml_file("defaults.yaml")
        else:
            self.defualts = {}

    def load_config(self, config_path):
        return NotImplementedError

    def merge_configs(self, base_config, override_config):
        return NotImplementedError

    def validate_config(self, config):
        return NotImplementedError

    def get_algorithm_config(self, algo_name):
        return NotImplementedError

    def get_environment_config(self, env_name):
        return NotImplementedError

    def get_full_config(self, algo_name, env_name):
        """Get complete config by merging defaults, algo, and env configs"""
        return NotImplementedError

    def save_config(self, config, filepath):
        """Save a config to a YAML file"""
        return NotImplementedError

    def list_available_algorithms(self):
        """Return list of available algorithm config files"""
        return NotImplementedError

    def list_available_environments(self):
        """Return list of available environment config files"""
        return NotImplementedError

    def reload_configs(self):
        """Clear cache and reload all configs"""
        return NotImplementedError

    def get_config_from_cache(self, key):
        """Get config from cache if available"""
        return NotImplementedError

    def add_config_to_cache(self, key, config):
        """Add config to cache"""
        return NotImplementedError

    def clear_cache(self):
        """Clear the config cache"""
        return NotImplementedError

    def _load_yaml_file(self, filename):
        """Load a YAML file from the config directory"""
        return NotImplementedError

    def _validate_yaml_syntax(self, filepath):
        """Validate that a YAML file has correct syntax"""
        return NotImplementedError

    def _deep_merge_dicts(self, base_dict, override_dict):
        """Recursively merge two dictionaries"""
        merged = base_dict.copy()
        for key in override_dict:
            if key in merged and isinstance(merged[key], dict) and isinstance(override_dict[key], dict):
                # Both are dictionaries, merge recursively
                merged[key] = self._deep_merge_dicts(merged[key], override_dict[key])
            else:
                # Override the value
                merged[key] = override_dict[key]
        
        return merged


    def _get_file_path(self, directory, filename):
        """Construct full file path from directory and filename"""
        return NotImplementedError

    def _is_config_file(self, filename):
        """Check if filename is a valid config file"""
        return NotImplementedError 