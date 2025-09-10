from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Dict
import torch

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random from the action space.

    This agent performs no learning. It overrides BaseAgent hooks to avoid
    network/optimizer setup and simply samples actions from the environment's
    action space for both training and evaluation modes.
    """

    def __init__(self, env: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        self.env = env
        self.config = dict(config or {})
        super().__init__(env, self.config)

    # ---- BaseAgent lifecycle overrides (disable networks/optimizers) ----
    def _parse_configs(self) -> None:  # type: ignore[override]
        """Parse configs for RandomAgent; no networks or optimizer needed."""
        self.network_config = {}
        self.training_config = self.config.get("training", {})

    def _create_networks(self) -> None:  # type: ignore[override]
        """RandomAgent uses no neural networks."""
        return None

    def _setup_optimizer(self) -> None:  # type: ignore[override]
        """RandomAgent does not require an optimizer."""
        self.optimizer = None  # type: ignore[assignment]

    # ---- Action selection ----
    def _select_action_training(self, obs: Any) -> Tuple[Any, Dict[str, Any]]:  # type: ignore[override]
        action = self.env.action_space.sample()
        return action, {"policy": "random", "mode": "training"}

    def _select_action_evaluation(self, obs: Any) -> Tuple[Any, Dict[str, Any]]:  # type: ignore[override]
        action = self.env.action_space.sample()
        return action, {"policy": "random", "mode": "evaluation"}

    # ---- Persistence (optional minimal stubs) ----
    def save(self, path: Any) -> None:  # type: ignore[override]
        """Save minimal agent metadata for consistency with Trainer APIs."""
        payload = {
            "agent_type": "RandomAgent",
            "config": self.config,
        }
        torch.save(payload, path)

    def load(self, path: Any) -> None:  # type: ignore[override]
        """Load minimal agent metadata saved by `save`. No weights to restore."""
        try:
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and "config" in payload:
                self.config = payload["config"]
        except Exception:
            # Silently ignore incompatible checkpoints; RandomAgent has no state
            pass

    def close(self) -> None:
        """No resources to release for RandomAgent."""
        return None