from base import BaseWrapper

class FrameStack(BaseWrapper):
    def _get_stacked_obs(self):
        return NotImplementedError