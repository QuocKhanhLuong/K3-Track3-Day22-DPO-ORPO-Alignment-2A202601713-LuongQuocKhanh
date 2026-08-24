from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return pairwise accuracy, counting ties as half a win."""
    if len(chosen_scores) != len(examples) or len(rejected_scores) != len(examples):
        raise ValueError("score lengths must match the number of examples")
    if not examples:
        return 0.0

    wins = 0.0
    for chosen_score, rejected_score in zip(chosen_scores, rejected_scores, strict=True):
        if chosen_score > rejected_score:
            wins += 1.0
        elif chosen_score == rejected_score:
            wins += 0.5

    return wins / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
