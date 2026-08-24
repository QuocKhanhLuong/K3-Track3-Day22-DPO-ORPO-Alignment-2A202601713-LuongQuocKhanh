from __future__ import annotations

import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute log(sigmoid(x)) without overflow."""
    return -np.logaddexp(0.0, -x)


def _validate_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("all log-probability arrays must have the same shape")


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities."""
    _validate_same_shape(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    policy_diff = policy_chosen_logps - policy_rejected_logps
    reference_diff = ref_chosen_logps - ref_rejected_logps
    margin = beta * (policy_diff - reference_diff)
    return float(-np.mean(_log_sigmoid(margin)))


def _log_odds(logp: np.ndarray) -> np.ndarray:
    """Compute log(p / (1 - p)) from log(p) with safe clipping."""
    clipped = np.clip(logp, -30.0, -1e-7)
    return clipped - np.log1p(-np.exp(clipped))


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute SFT loss plus an ORPO odds-ratio preference penalty."""
    _validate_same_shape(sft_nll, chosen_logps, rejected_logps)
    if lambda_orpo < 0.0:
        raise ValueError("lambda_orpo must be non-negative")

    log_odds_diff = _log_odds(chosen_logps) - _log_odds(rejected_logps)
    preference_term = _log_sigmoid(log_odds_diff)
    return float(np.mean(sft_nll) - lambda_orpo * np.mean(preference_term))
