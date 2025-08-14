from typing import Any, Mapping, Dict, Optional, Tuple
from .trainer import Trainer

class OnPolicyTrainer(Trainer):
    def __init__(self, agent: Any, env: Any, config: Mapping[str, Any]) -> None:
        """Initialize on-policy trainer state and services."""
        super().__init__(agent, env, config)

    def train_episode(self, episode_index: int) -> Tuple[float, int, Optional[float]]:
        """Collect rollout(s), compute advantages, and update policy/value."""
        raise NotImplementedError

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Run policy in eval mode and aggregate episode metrics."""
        raise NotImplementedError

    def train(self, num_episodes: int, eval_every: Optional[int] = None, eval_episodes: int = 5, checkpoint_every: Optional[int] = None) -> None:
        """Run on-policy training loop with optional eval/checkpoints."""
        raise NotImplementedError

    def save_checkpoint(self, path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save agent weights and trainer state for on-policy training."""
        raise NotImplementedError

    def load_checkpoint(self, path) -> None:
        """Load agent weights and trainer state for on-policy training."""
        raise NotImplementedError

    def close(self) -> None:
        """Close vectorized envs/loggers used by on-policy training."""
        raise NotImplementedError