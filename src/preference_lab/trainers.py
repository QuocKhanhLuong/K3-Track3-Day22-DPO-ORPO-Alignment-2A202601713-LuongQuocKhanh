from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Train the policy.

        TODO(student): implement either a mock trainer for CPU or a TRL-backed trainer.
        Keep side effects explicit: checkpoints and metrics should go to configured output_dir.
        """
        raise NotImplementedError("TODO(student): implement trainer")
