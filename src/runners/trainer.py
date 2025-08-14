from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from src.utils.eval import set_env_training_mode
import torch

class Trainer(ABC):
    def __init__(self, agent: Any, env: Any, config: Mapping[str, Any]) -> None:
        """Set up trainer with agent, environment, and configuration."""
        self.agent = agent
        self.env = env
        self.config = config

    @abstractmethod
    def train_episode(self, episode_index: int) -> Tuple[float, int, Optional[float]]:
        """Run one training episode and return reward, length, and optional loss."""
        raise NotImplementedError("Subclass must implement train_episode")

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate policy for given episodes and return aggregate metrics."""
        with torch.inference_mode():
            if hasattr(self.agent, 'network'):
                torch.inference_mode()

    def train(
        self,
        num_episodes: int,
        eval_every: Optional[int] = None,
        eval_episodes: int = 5,
        checkpoint_every: Optional[int] = None,
    ) -> None:
        """Run the training loop with optional periodic evaluation and checkpoints."""
        self._before_training()
        try:
            for episode_idx in range(num_episodes):
                self.before_episode(episode_idx)
                reward, length, loss = self.train_episode(episode_idx)
                self.after_episode(episode_idx, reward, length, loss)

                if eval_every and (episode_idx + 1) % eval_every == 0:
                    metrics = self.evaluate(eval_episodes)
                    self.after_evaluation(episode_idx, metrics)

                if checkpoint_every and (checkpoint_every + 1) % checkpoint_every == 0:
                    path = self._checkpoint_path(episode_idx)
                    self.save_checkpoint(path, metadata={"episode": episode_idx})
        finally:
            self.after_training()

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist agent and trainer state to the specified path."""
        raise NotImplementedError


    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore agent and trainer state from the specified path."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (e.g., environments, loggers, file handles)."""
        raise NotImplementedError
    
    # hooks
    def before_training(self) -> None: """Hook before any training begins."""; return None
    def after_training(self) -> None: """Hook after all training ends."""; return None
    def before_episode(self, episode_index: int) -> None: """Hook before each episode."""; return None
    def after_episode(self, episode_index: int, reward: float, length: int, loss: Optional[float]) -> None: """Hook after each episode."""; return None
    def after_evaluation(self, episode_index: int, metrics: Dict[str, float]) -> None: """Hook after periodic evaluation."""; return None
    def _checkpoint_path(self, episode_index: int) -> Path: """Compute a checkpoint path for an episode index."""; return Path(f"\"checkpoint_{episode_index+1}.pt\"")