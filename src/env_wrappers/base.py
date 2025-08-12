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
    
    def render(self, **kwargs):
        """Render the environment state."""
        return self.env.render(**kwargs)
    
    def close(self):
        """Close the environment and free resources."""
        self.env.close()
    
    def seed(self, seed=None):
        """Set the random seed for reproducible behavior."""
        if seed is None:
            return
        
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        if hasattr(self.env, "observation_space") and hasattr(self.env.observation_space, "seed"):
            try:
                self.env.observation_space.seed(seed)
            except Exception:
                pass
        try:
            self.env.reset(seed=seed)
        except Exception:
            pass
    
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