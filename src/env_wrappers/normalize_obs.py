from base import BaseWrapper

class NormalizeObservations(BaseWrapper):
    def _normalize_obs(self, obs):
        return NotImplementedError
    
    
