class EnvironmentFactory:
    def create_env(self, env_name, config):
        return NotImplementedError
    
    def apply_wrappers(self, env, wrapper_config):
        return NotImplementedError