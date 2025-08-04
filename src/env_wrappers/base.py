class BaseWrapper:
    def __init__(self, env, config, **kwargs):
        self.env = env
        self.config = config or {}
        self.kwargs = kwargs
        
    def reset(self, **kwargs):
        """Reset the environment and return initial observation."""
        return self.env.reset(**kwargs)
    
    def step(self, action, **kwargs):
        """Take a step in the environment with the given action."""
        return self.env.step(action, **kwargs)
    
    def render(self, mode="human", **kwargs):
        """Render the environment state."""
        return self.env.render(mode, **kwargs)
    
    def close(self):
        """Close the environment and free resources."""
        self.env.close()
    
    def seed(self, seed=None):
        """Set the random seed for reproducible behavior."""
        return self.env.seed(seed)
    
    @property
    def observation_space(self):
        """Get the observation space of the environment."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment."""
        return self.env.action_space
    
    @property
    def spec(self):
        """Get the environment specification."""
        return self.env.spec
    
    @property
    def unwrapped(self):
        """Get the unwrapped environment."""
        return self.env
    
    @property
    def metadata(self):
        """Get the environment metadata."""
        return self.env.metadata