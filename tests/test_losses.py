import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_closed_form() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )
    assert np.isclose(loss, 0.663597, atol=1e-5)


def test_dpo_loss_decreases_as_margin_grows() -> None:
    ref = (np.array([-0.6]), np.array([-1.0]))
    weak = dpo_loss(np.array([-0.9]), np.array([-1.0]), *ref, beta=0.1)
    strong = dpo_loss(np.array([-0.1]), np.array([-3.0]), *ref, beta=0.1)
    assert strong < weak


def test_dpo_loss_is_stable_for_extreme_inputs() -> None:
    loss = dpo_loss(
        np.array([-1e4]),
        np.array([-1.0]),
        np.array([-1.0]),
        np.array([-1.0]),
        beta=1.0,
    )
    assert np.isfinite(loss)


def test_orpo_loss_matches_closed_form() -> None:
    loss = orpo_loss(
        np.array([1.0]),
        np.array([-0.5]),
        np.array([-1.5]),
        lambda_orpo=0.1,
    )
    assert np.isclose(loss, 1.017086, atol=1e-5)


def test_orpo_loss_is_stable_at_logp_zero() -> None:
    loss = orpo_loss(
        np.array([1.0]),
        np.array([0.0]),
        np.array([-1.0]),
        lambda_orpo=0.1,
    )
    assert np.isfinite(loss)
