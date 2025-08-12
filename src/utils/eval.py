from typing import Any


def set_env_training_mode(env: Any, training: bool) -> None:
    """
    Walk a chain of Gymnasium-style wrappers (attributes `.env`) and, for any
    wrapper that exposes `set_training(bool)`, set its training mode.

    This is useful to freeze running statistics (e.g., normalization wrappers)
    during evaluation and re-enable them during training.
    """
    current = env
    visited_ids = set()

    while current is not None and id(current) not in visited_ids:
        visited_ids.add(id(current))

        if hasattr(current, "set_training"):
            try:
                current.set_training(bool(training))
            except Exception:
                # Ignore wrappers that define set_training but raise
                # due to unexpected internal state; continue descending
                pass

        # Descend into the next wrapper/base env if present
        current = getattr(current, "env", None)


