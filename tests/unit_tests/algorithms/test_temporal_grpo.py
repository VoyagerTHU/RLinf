import torch

from rlinf.algorithms.advantages import compute_temporal_grpo_advantages
from rlinf.algorithms.registry import calculate_adv_and_returns


def test_temporal_grpo_stops_credit_after_early_milestone():
    rewards = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    dones = torch.zeros(4, 2, dtype=torch.bool)
    dones[-1] = True

    advantages, returns = compute_temporal_grpo_advantages(
        rewards,
        loss_mask,
        dones,
        group_size=2,
        gamma=1.0,
    )

    assert returns is None
    torch.testing.assert_close(
        advantages[0], torch.tensor([2**-0.5, -(2**-0.5)]), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(advantages[1:], torch.zeros_like(advantages[1:]))


def test_temporal_grpo_propagates_late_success_backward_with_discount():
    rewards = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    dones = torch.zeros(4, 2, dtype=torch.bool)
    dones[-1] = True

    advantages, _ = compute_temporal_grpo_advantages(
        rewards,
        loss_mask,
        dones,
        group_size=2,
        gamma=0.5,
    )

    # Group normalization is scale invariant, so each nonzero discounted
    # return produces the same signed two-sample GRPO advantage.
    expected = torch.tensor([2**-0.5, -(2**-0.5)]).expand(3, 2)
    torch.testing.assert_close(advantages, expected, atol=2e-5, rtol=0)


def test_temporal_grpo_excludes_completed_trajectories_from_later_baselines():
    rewards = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    loss_mask = torch.tensor(
        [
            [True, True, True, True],
            [False, True, True, True],
        ]
    )
    dones = torch.zeros(3, 4, dtype=torch.bool)
    dones[1, 0] = True
    dones[-1] = True

    advantages, _ = compute_temporal_grpo_advantages(
        rewards,
        loss_mask,
        dones,
        group_size=2,
        gamma=1.0,
    )

    assert advantages[1, 0] == 0
    assert advantages[1, 1] == 0
    torch.testing.assert_close(
        advantages[1, 2:],
        torch.tensor([2**-0.5, -(2**-0.5)]),
        atol=2e-6,
        rtol=0,
    )


def test_embodied_registry_preserves_temporal_rewards_for_temporal_grpo():
    rewards = torch.zeros(3, 2, 12)
    rewards[0, 0, 4] = 1.0
    dones = torch.zeros(4, 2, 12, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones(3, 2, 1, dtype=torch.bool)

    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="temporal_grpo",
        rewards=rewards,
        dones=dones,
        values=None,
        gamma=1.0,
        gae_lambda=1.0,
        group_size=2,
        reward_type="chunk_level",
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask.clone(),
    )

    assert result["advantages"].shape == (3, 2, 1)
    torch.testing.assert_close(
        result["advantages"][0, :, 0],
        torch.tensor([2**-0.5, -(2**-0.5)]),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        result["advantages"][1:], torch.zeros_like(result["advantages"][1:])
    )
