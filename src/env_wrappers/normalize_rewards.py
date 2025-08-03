from base import BaseWrapper

class NormalizedRewards(BaseWrapper):
    def _normalize_rewards(self, reward):
        return NotImplementedError