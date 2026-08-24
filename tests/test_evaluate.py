import pytest

from preference_lab.evaluate import pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_counts_ties_as_half() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
    ]
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == 0.75


def test_pairwise_accuracy_rejects_length_mismatch() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match="score lengths"):
        pairwise_accuracy(examples, [], [1.0])
