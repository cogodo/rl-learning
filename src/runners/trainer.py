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
        # Switch wrappers and policy to evaluation behavior
        set_env_training_mode(self.env, False)
        had_network = hasattr(self.agent, "network")
        prev_training = self.agent.network.training if had_network else None
        if had_network:
            self.agent.network.eval()

        rewards: list[float] = []
        lengths: list[int] = []
        with torch.inference_mode():
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0
                while not done:
                    action, _ = self.agent.select_action(obs, training=False)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = bool(terminated or truncated)
                    episode_reward += float(reward)
                    episode_length += 1
                rewards.append(episode_reward)
                lengths.append(episode_length)

        # Restore training state
        if had_network and prev_training:
            self.agent.network.train()
        set_env_training_mode(self.env, True)

        n = max(len(rewards), 1)
        return {
            "reward_mean": sum(rewards) / n,
            "length_mean": sum(lengths) / n,
        }

    def train(
        self,
        num_episodes: int,
        eval_every: Optional[int] = None,
        eval_episodes: int = 5,
        checkpoint_every: Optional[int] = None,
    ) -> None:
        """Run the training loop with optional periodic evaluation and checkpoints."""
        self.before_training()
        set_env_training_mode(self.env, True)
        try:
            for episode_idx in range(num_episodes):
                self.before_episode(episode_idx)
                reward, length, loss = self.train_episode(episode_idx)
                self.after_episode(episode_idx, reward, length, loss)

                if eval_every and (episode_idx + 1) % eval_every == 0:
                    metrics = self.evaluate(eval_episodes)
                    self.after_evaluation(episode_idx, metrics)

                if checkpoint_every and (episode_idx + 1) % checkpoint_every == 0:
                    path = self._checkpoint_path(episode_idx)
                    self.save_checkpoint(path, metadata={"episode": episode_idx})
        finally:
            self.after_training()

    def save_checkpoint(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist agent state using the agent's checkpoint format.

        Note: `metadata` (if provided) is saved to a sidecar file `<path>.meta.pt`.
        """
        # Delegate main weights/state to the agent. This defines the canonical format.
        self.agent.save(path)
        # Optionally persist trainer metadata to a sidecar, without constraining the agent format.
        if metadata is not None:
            sidecar = f"{str(path)}.meta.pt"
            torch.save(metadata, sidecar)


    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Restore agent state from the specified path using the agent's format."""
        # Delegate to the agent to load its own checkpoint format
        self.agent.load(path)

    def close(self) -> None:
        """Release resources (e.g., environments, loggers, file handles)."""
        if hasattr(self.env, "close"):
            self.env.close()
        if hasattr(self.agent, "close"):
            self.agent.close()
    
    # hooks
    def before_training(self) -> None: 
        """Hook before any training begins."""
        return None
    def after_training(self) -> None:
        """Hook after all training ends."""
        return None
    def before_episode(self, episode_index: int) -> None: 
        """Hook before each episode."""
        return None
    def after_episode(self, episode_index: int, reward: float, length: int, loss: Optional[float]) -> None: 
        """Hook after each episode."""
        return None
    def after_evaluation(self, episode_index: int, metrics: Dict[str, float]) -> None:
        """Hook after periodic evaluation."""
        return None
    def _checkpoint_path(self, episode_index: int) -> Path: 
        """Compute a checkpoint path for an episode index."""
        return Path(f"checkpoint_{episode_index+1}.pt")