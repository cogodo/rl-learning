"""Experiment orchestration.

Coordinates environment creation, agent building, trainer selection,
and running training or evaluation flows.
"""

from typing import Any, Dict, Mapping, Optional
from src.agents.build_agent import build_agent
from src.env_wrappers.env_factory import EnvironmentFactory
from src.runners.off_policy_trainer import OffPolicyTrainer
from src.runners.on_policy_trainer import OnPolicyTrainer

class ExperimentRunner:
    """Coordinates environment creation, agent building, and training/evaluation.

    Responsibilities:
    - Validate and store the experiment `config`.
    - Create the base environment via an environment factory and apply wrappers.
    - Build the agent via a builder based on algorithm selection in `config`.
    - Select an appropriate trainer (on-policy vs off-policy) based on algorithm.
    - Execute either training or evaluation-only flow.
    - Persist checkpoints and collect metrics.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Initialize the runner with a configuration mapping.

        Expected keys under `config` (non-exhaustive):
        - environment: environment settings and wrappers
        - algorithm: algorithm selection (e.g., DQN, PPO)
        - training: training hyperparameters (episodes, eval cadence, etc.)
        - evaluation: evaluation options (episodes, eval-only flag)
        """
        self.config: Mapping[str, Any] = config
        self.train_cfg = config.get("training", {})
        self.eval_cfg = config.get("evaluation", {})
        self.env_cfg = config.get("environment", {})
        self.algo_cfg = config.get("algorithm", {})
        self.env: Optional[Any] = None
        self.agent: Optional[Any] = None
        self.trainer: Optional[Any] = None

    def create_environment(self) -> Any:
        """Create and return the environment based on the config.

        Should use the environment factory and apply configured wrappers.
        """
        env_name = self.env_cfg.get("env_name")
        factory = EnvironmentFactory()
        return factory.create_env(env_name, config=self.config)

    def create_agent(self, env: Any) -> Any:
        """Construct and return the agent instance for the specified environment.

        Should select the appropriate agent implementation based on `config`.
        """
        # Delegate to the central agent factory with full config
        return build_agent(env, self.config)

    def select_trainer(self, agent: Any, env: Any) -> Any:
        """Return a trainer instance (on/off-policy) appropriate for the agent.

        Should dispatch to on-policy or off-policy trainer based on algorithm.
        """
        algo_type = (self.algo_cfg or {}).get("type")
        if algo_type in {"DQN", "SARSA", "RANDOM"}:
            return OffPolicyTrainer(agent, env, self.config)
        elif algo_type in {"PPO", "GRPO"}:
            return OnPolicyTrainer(agent, env, self.config)
        raise ValueError(f"Unknown or unsupported algorithm type: {algo_type}")

    def run(self) -> Dict[str, Any]:
        """Run the experiment according to the mode (train or eval-only).

        Returns a dictionary of summary results/metrics.
        """
        self.env = self.create_environment()
        self.agent = self.create_agent(self.env)
        self.trainer = self.select_trainer(self.agent, self.env)
        self.load_checkpoint_if_configured()
        try:
            if bool(self.eval_cfg.get("eval_only", False)):
                return self.run_evaluation_only(self.trainer)
            return self.run_training(self.trainer)
        finally:
            if hasattr(self.trainer, "close"):
                self.trainer.close()

    def run_training(self, trainer: Any) -> Dict[str, Any]:
        """Execute the training loop with configured evaluation/checkpoints.

        Should invoke the trainer to train for `num_episodes` and optionally
        perform periodic evaluations and checkpointing.
        """
        num_episodes = int(self.train_cfg.get("episodes", 100))
        eval_every = self.train_cfg.get("eval_every")
        checkpoint_every = self.train_cfg.get("checkpoint_every")
        eval_episodes = int(self.eval_cfg.get("episodes", 5))
        trainer.train(
            num_episodes=num_episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            checkpoint_every=checkpoint_every,
        )
        return {}

    def run_evaluation_only(self, trainer: Any) -> Dict[str, Any]:
        """Execute evaluation-only rollouts without training.

        Should call the trainer's `evaluate` for the configured number
        of episodes and return the resulting metrics.
        """
        episodes = int(self.eval_cfg.get("episodes", 5))
        return trainer.evaluate(episodes)

    def load_checkpoint_if_configured(self, path: Optional[str] = None) -> None:
        """Optionally load a checkpoint when configured.

        If the configuration specifies a checkpoint path, restore trainer/agent
        state before running training or evaluation.
        """
        # Prefer explicit argument, otherwise look in config
        checkpoint_path = path or self.eval_cfg.get("checkpoint") or self.train_cfg.get("resume_from")
        if checkpoint_path and self.trainer is not None:
            try:
                self.trainer.load_checkpoint(checkpoint_path)
            except NotImplementedError:
                # Subclass did not implement custom checkpointing; skip silently
                pass

    def save_results(self, path: Optional[str] = None) -> None:
        """Persist experiment results and artifacts to the given path.

        The location and format may be configured under the main config.
        """
        raise NotImplementedError


def run_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Top-level API to run an experiment using the provided configuration.

    Creates an `ExperimentRunner`, sets up environment/agent/trainer, and
    executes either training or evaluation-only. Returns summary metrics.
    """
    runner = ExperimentRunner(config)
    return runner.run()