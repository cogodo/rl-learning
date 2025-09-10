"""Visualization utilities for experiments.

Defines a `Plotter` class with method stubs for common experiment plots.
Implementations are intentionally omitted here; this serves as the public API.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class Plotter:
    """High-level plotting interface for experiment results.

    Responsibilities:
    - Produce learning curves (reward, loss, episode length) over time.
    - Compare algorithms across runs with aggregations and uncertainty bands.
    - Visualize simple hyperparameter sensitivity analyses.
    - Save figures to disk using consistent naming and formats.
    """

    def __init__(self, style: Optional[str] = None) -> None:
        """Initialize the plotter.

        - style: Optional plotting style/preset name to apply globally.
        """
        raise NotImplementedError

    def plot_learning_curves(
        self,
        rewards: Sequence[float],
        losses: Optional[Sequence[float]] = None,
        lengths: Optional[Sequence[int]] = None,
        smoothing_window: int = 1,
        show_std: bool = False,
        label: Optional[str] = None,
    ) -> None:
        """Plot learning curves for a single run.

        Inputs are episode-wise sequences. If multiple runs are desired,
        aggregate externally and call this method per aggregated series.
        """
        raise NotImplementedError

    def plot_algorithm_comparison(
        self,
        results: Mapping[str, Mapping[str, Sequence[float]]],
        metric: str = "final_performance",
        confidence_level: float = 0.95,
    ) -> None:
        """Compare algorithms across runs.

        - results: maps algorithm name → { metric_name → sequence per run or aggregate }
        - metric: which metric to compare (e.g., 'final_performance')
        - confidence_level: level for uncertainty bands/intervals
        """
        raise NotImplementedError

    def plot_hyperparameter_sensitivity(
        self,
        results: Mapping[str, Mapping[str, Sequence[float]]],
        parameters: Sequence[str],
        metric: str = "reward_mean",
    ) -> None:
        """Plot sensitivity of performance metrics to hyperparameters.

        - results: maps param_name → { param_value_str → metric series or aggregate }
        - parameters: which hyperparameters to include
        - metric: which metric to visualize
        """
        raise NotImplementedError

    def save_plots(self, path: str, format: str = "png", dpi: int = 150) -> None:
        """Persist all currently prepared figures to disk.

        - path: directory to write figures into
        - format: file format extension (e.g., 'png', 'pdf')
        - dpi: dots-per-inch for raster formats
        """
        raise NotImplementedError
