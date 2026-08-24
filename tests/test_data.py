from pathlib import Path

import pytest
from pydantic import ValidationError

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":2:"):
        load_jsonl(bad)


def test_duplicate_prompt_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"prompt":" Same prompt ","chosen":"good","rejected":"bad"}\n'
        '{"prompt":"same   PROMPT","chosen":"better","rejected":"worse"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate prompt"):
        load_jsonl(duplicate)


def test_split_has_no_prompt_leakage() -> None:
    examples = [
        PreferenceExample(prompt="shared", chosen="good one", rejected="bad one"),
        PreferenceExample(prompt="shared", chosen="good two", rejected="bad two"),
        PreferenceExample(prompt="other", chosen="good three", rejected="bad three"),
        PreferenceExample(prompt="third", chosen="good four", rejected="bad four"),
    ]
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=42)

    assert len(train) + len(val) == len(examples)
    assert not ({example.prompt for example in train} & {example.prompt for example in val})


def test_split_is_deterministic() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    first = split_by_prompt(examples, validation_ratio=0.25, seed=7)
    second = split_by_prompt(examples, validation_ratio=0.25, seed=7)
    assert first == second


def test_chosen_rejected_validation_ignores_case_and_whitespace() -> None:
    with pytest.raises(ValidationError, match="chosen and rejected must differ"):
        PreferenceExample(prompt="p", chosen="Same   Answer", rejected=" same answer ")
