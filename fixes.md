- **Gymnasium API semantics**: In trainer/agent, handle time-limit bootstrapping via `info["terminal_observation"]` when `truncated=True`.

- **Seeding and reproducibility**: Consistently seed `env.reset(seed=...)`, `numpy`, and `torch`. Optionally set PyTorch determinism flags. Record seeds in logs.

- **Buffer semantics**: Expose `ready()` and prefer `len(buffer)`; deprecate `get_buffer_size` to keep one idiom.

- **Episode flushing**: Ensure the last in-progress episode is flushed at shutdown if needed (for logging/analysis), even if `done` wasn’t seen.

- **Network input sizing**: Derive network sizes from the wrapped env spaces. Flatten or handle channel-first consistently after stacking.

- **Evaluation policy**: Use deterministic action selection for eval; disable exploration, normalization updates, and learning.

- **Logging scale**: Prevent in-memory log growth; stream or rotate logs. Include timestamps, seeds, git commit, and config snapshot.

- **Testing**: Add unit tests for wrapper shapes, kwargs pass-through, buffer sampling, and network output sizes. Include truncated/terminated edge cases.

- **YAML clarity**: Keep wrapper config minimal and self-contained. Document each wrapper’s fields and defaults near the registry or in `README.md`.