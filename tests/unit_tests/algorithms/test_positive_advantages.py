import torch

from rlinf.algorithms.utils import (
    positive_advantage_sample_mask,
    prioritize_positive_advantage_samples,
    retain_positive_advantages,
)


def test_retain_positive_advantages_drops_negative_zero_and_masked_values():
    advantages = torch.tensor(
        [
            [[-2.0], [0.0], [1.5]],
            [[3.0], [-4.0], [5.0]],
        ],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [
            [[True], [True], [True]],
            [[True], [True], [False]],
        ]
    )

    filtered = retain_positive_advantages(advantages, loss_mask)

    torch.testing.assert_close(
        filtered,
        torch.tensor(
            [
                [[0.0], [0.0], [1.5]],
                [[3.0], [0.0], [0.0]],
            ]
        ),
    )
    # Filtering must not mutate the rollout advantages retained for audits.
    assert advantages[0, 0, 0] == -2.0
    assert advantages[1, 2, 0] == 5.0


def test_retain_positive_advantages_without_mask_preserves_dtype():
    advantages = torch.tensor([-1.0, 2.0], dtype=torch.float64)

    filtered = retain_positive_advantages(advantages)

    assert filtered.dtype == torch.float64
    torch.testing.assert_close(filtered, torch.tensor([0.0, 2.0], dtype=torch.float64))


def test_retain_positive_advantages_rejects_misaligned_mask():
    try:
        retain_positive_advantages(torch.ones(2, 3), torch.ones(2, 1))
    except ValueError as exc:
        assert "must match advantages exactly" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for a misaligned loss mask")


def test_positive_advantage_sample_mask_reduces_trailing_dimensions():
    advantages = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [-1.0, 2.0]],
            [[0.0, 3.0], [-4.0, 0.0], [0.0, 0.0]],
        ]
    )
    loss_mask = torch.tensor(
        [
            [[True, True], [False, True], [True, True]],
            [[True, True], [True, True], [True, True]],
        ]
    )

    torch.testing.assert_close(
        positive_advantage_sample_mask(advantages, loss_mask),
        torch.tensor([False, False, True, True, False, False]),
    )


def test_prioritize_positive_advantage_samples_preserves_random_partition_order():
    advantages = torch.tensor(
        [
            [[0.0], [2.0], [0.0]],
            [[3.0], [0.0], [4.0]],
        ]
    )
    randomized_indices = torch.tensor([4, 1, 5, 0, 3, 2])

    prioritized = prioritize_positive_advantage_samples(
        advantages, randomized_indices
    )

    torch.testing.assert_close(prioritized, torch.tensor([1, 5, 3, 4, 0, 2]))
