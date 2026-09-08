"""Helpers for keeping environment and policy rollout horizons aligned."""

from typing import Any


def resolve_action_steps_per_chunk(env_cfg: Any, model_cfg: Any) -> int:
    """Resolve how many actions are physically executed per policy query.

    ``num_action_chunks`` can describe a model's full query-token horizon while
    ``num_executed_action_chunks`` describes the prefix returned for RL.  An
    environment-specific ``action_steps_per_chunk`` remains the highest-priority
    override.

    Args:
        env_cfg: Environment configuration with an optional
            ``action_steps_per_chunk`` value.
        model_cfg: Model configuration with ``num_action_chunks`` and an optional
            ``num_executed_action_chunks`` value.

    Returns:
        The positive number of environment actions executed per policy query.

    Raises:
        ValueError: If the resolved horizon is not positive.
    """

    action_steps = int(
        env_cfg.get(
            "action_steps_per_chunk",
            model_cfg.get(
                "num_executed_action_chunks",
                model_cfg.num_action_chunks,
            ),
        )
    )
    if action_steps <= 0:
        raise ValueError("action_steps_per_chunk must be positive")
    return action_steps


def resolve_num_chunk_steps(env_cfg: Any, model_cfg: Any) -> int:
    """Resolve the number of policy queries in one rollout epoch.

    Args:
        env_cfg: Environment configuration containing
            ``max_steps_per_rollout_epoch``.
        model_cfg: Model configuration used to resolve the execution horizon.

    Returns:
        The number of policy queries required for one rollout epoch.

    Raises:
        ValueError: If the environment horizon is not divisible by the execution
            horizon.
    """

    action_steps = resolve_action_steps_per_chunk(env_cfg, model_cfg)
    max_steps = int(env_cfg.max_steps_per_rollout_epoch)
    if max_steps % action_steps != 0:
        raise ValueError(
            "max_steps_per_rollout_epoch must be divisible by the executed "
            f"action horizon ({action_steps}), got {max_steps}"
        )
    return max_steps // action_steps
