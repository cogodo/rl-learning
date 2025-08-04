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

    def get_full_config(self, algo_name, env_name):
        """Get complete config by merging defaults, algo, and env configs"""
        algo_path = self._get_file_path(self.algo_path, algo_name)
        env_path = self._get_file_path(self.envs_path, env_name)

        algo_config = self._load_yaml_file(algo_path)
        env_config = self._load_yaml_file(env_path)

        merged = self.defaults.copy()

        if "algorithm" not in merged:
            merged['algorithm'] = {}
        merged['algorithm'].update(algo_config)

        if "environment" not in merged:
            merged["environment"] = {}
        merged["environment"].update(env_config)

        return merged

    def save_config(self, config, filepath):
        """Save a config to a YAML file"""
        with open(filepath, 'w') as f:
            yaml.safe_dump(config, f)

    def list_available_algorithms(self):
        """Return list of available algorithm config files"""
        files = os.listdir(self.algo_path)

        yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]
        return yaml_files

    def list_available_environments(self):
        """Return list of available environment config files"""
        files = os.listdir(self.envs_path)

        yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]
        return yaml_files

    def reload_configs(self):
        """Clear cache and reload all configs"""
        self.clear_cache()

        if hasattr(self, 'defaults'):
            self.defaults = self._load_yaml_file("defaults.yaml")

    def get_config_from_cache(self, key):
        """Get config from cache if available"""
        if key in self._config_cache:
            return self._config_cache[key]
        else:
            raise KeyError(f"Key not found in cache: {key}")
            
        

    def add_config_to_cache(self, key, config):
        """Add config to cache"""
        self._config_cache[key] = config

    def clear_cache(self):
        """Clear the config cache"""
        self._config_cache.clear()

    def _load_yaml_file(self, filename):
        """Load a YAML file from the config directory"""
        
        path = self._get_file_path(self.base_config_path, filename)
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
            # Validate directly here (not calling separate method)
        if data is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")
        
        if not isinstance(data, dict):
            raise ValueError(f"YAML file must contain a dictionary: {path}")
        
        # Check for required keys, etc.
        if 'required_key' not in data:
            raise ValueError(f"Missing required key 'required_key' in {path}")

        self.add_config_to_cache(filename, data)
        return data


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
        return os.path.join(directory, filename)

    def _is_config_file(self, filename):
        """Check if filename is a valid config file"""
        return filename.endswith(('.yaml', '.yml')) 