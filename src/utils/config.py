

class ConfigManager:
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
