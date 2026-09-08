import pytest
import torch

from rlinf.algorithms.utils import compute_embodied_reference_kl


def test_chunk_reference_kl_sums_scalar_action_dimensions_and_masks_batch():
    live = torch.tensor(
        [
            [[-1.0, -2.0], [-3.0, -4.0]],
            [[-2.0, -3.0], [-4.0, -5.0]],
        ]
    )
    reference = live - 0.25

    loss = compute_embodied_reference_kl(
        live,
        reference,
        kl_penalty_type="kl",
        logprob_type="chunk_level",
        single_action_dim=2,
        loss_mask=torch.tensor([[True], [False]]),
    )

    assert loss.item() == pytest.approx(1.0)


def test_low_variance_reference_kl_is_non_negative_and_zero_at_reference():
    reference = torch.randn(3, 4, 2)
    equal_loss = compute_embodied_reference_kl(
        reference,
        reference,
        kl_penalty_type="low_var_kl",
        logprob_type="chunk_level",
        single_action_dim=2,
    )
    shifted_loss = compute_embodied_reference_kl(
        reference + 0.2,
        reference,
        kl_penalty_type="low_var_kl",
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert equal_loss.item() == pytest.approx(0.0)
    assert shifted_loss.item() > 0.0


def test_reference_kl_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_embodied_reference_kl(
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, 5),
            kl_penalty_type="kl",
            logprob_type="chunk_level",
            single_action_dim=4,
        )
