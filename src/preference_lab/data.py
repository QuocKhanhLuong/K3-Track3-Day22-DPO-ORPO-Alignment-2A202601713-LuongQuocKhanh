from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load validated preference examples from JSONL.

    Invalid JSON/schema rows include the source line number in the error message.
    Duplicate prompts are rejected after case/whitespace normalization.
    """
    source = Path(path)
    examples: list[PreferenceExample] = []
    seen_prompts: dict[str, int] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON - {exc}") from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{source}:{line_no}: invalid schema - {exc}") from exc

            prompt_key = _normalize_prompt(example.prompt)
            previous_line = seen_prompts.get(prompt_key)
            if previous_line is not None:
                raise ValueError(
                    f"{source}:{line_no}: duplicate prompt; first seen on line {previous_line}"
                )

            seen_prompts[prompt_key] = line_no
            examples.append(example)

    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt with deterministic shuffling to avoid leakage."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if not examples:
        return [], []

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(_normalize_prompt(example.prompt), []).append(example)

    prompt_keys = list(groups)
    random.Random(seed).shuffle(prompt_keys)

    if len(prompt_keys) == 1:
        return list(groups[prompt_keys[0]]), []

    validation_count = round(len(prompt_keys) * validation_ratio)
    validation_count = min(max(validation_count, 1), len(prompt_keys) - 1)
    validation_keys = set(prompt_keys[:validation_count])

    train: list[PreferenceExample] = []
    validation: list[PreferenceExample] = []
    for key in prompt_keys:
        target = validation if key in validation_keys else train
        target.extend(groups[key])

    if len(train) + len(validation) != len(examples):
        raise RuntimeError("split invariant violated")

    return train, validation
