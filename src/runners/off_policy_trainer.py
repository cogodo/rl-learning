from typing import Any, Mapping, Dict, Optional, Tuple
from .trainer import Trainer

class OffPolicyTrainer(Trainer):
    def __init__(self, agent: Any, env: Any, config: Mapping[str, Any]) -> None:
        """Initialize off-policy trainer state (e.g., replay buffer)."""
        super().__init__(agent, env, config)

    def train_episode(self, episode_index: int) -> Tuple[float, int, Optional[float]]:
        """Interact, push to replay, sample batches, and update Q/policy."""
        raise NotImplementedError

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Greedy evaluation without exploration noise and aggregate metrics."""
        return super().evaluate(num_episodes)

    def train(self, num_episodes: int, eval_every: Optional[int] = None, eval_episodes: int = 5, checkpoint_every: Optional[int] = None) -> None:
        """Run off-policy loop with replay, target syncs, and eval/checkpoints."""
        return super().train(num_episodes, eval_every=eval_every, eval_episodes=eval_episodes, checkpoint_every=checkpoint_every)

    def save_checkpoint(self, path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save agent weights, optimizer, replay/targets as needed."""
        raise NotImplementedError

    def load_checkpoint(self, path) -> None:
        """Restore agent weights, optimizer, and replay/targets as needed."""
        raise NotImplementedError

    def close(self) -> None:
        """Close envs/loggers and flush any pending artifacts."""
        if hasattr(self.env, "close"):
            self.env.close()
        if hasattr(self.agent, "close"):
            self.agent.close()
        