from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics

app = typer.Typer(help="Preference alignment lab CLI")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "this",
    "that",
    "with",
    "by",
    "from",
    "as",
    "it",
    "its",
    "do",
    "does",
    "during",
    "between",
    "into",
    "than",
    "while",
    "used",
    "use",
    "using",
    "explain",
    "concept",
    "purpose",
    "difference",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def _heuristic_score(prompt: str, answer: str) -> float:
    """Score answer relevance/informativeness with an explicit concision prior.

    This deterministic CPU-only scorer is deliberately imperfect: it rewards prompt
    keyword coverage and unique content, while penalizing very short/long or repetitive
    answers. Its failure cases are useful evidence for the lab report.
    """
    prompt_terms = {
        token for token in _tokens(prompt) if token not in _STOPWORDS and len(token) > 2
    }
    answer_tokens = _tokens(answer)
    answer_set = set(answer_tokens)

    coverage = len(prompt_terms & answer_set) / max(1, len(prompt_terms))
    unique_content = len(
        {token for token in answer_tokens if token not in _STOPWORDS and len(token) > 2}
    )
    substance = min(unique_content / 18.0, 1.0)

    length = len(answer_tokens)
    if length < 8:
        length_penalty = (8 - length) / 8 * 0.25
    elif length > 26:
        length_penalty = min((length - 26) / 26 * 0.25, 0.25)
    else:
        length_penalty = 0.0

    counts = Counter(answer_tokens)
    repeated = sum(max(0, count - 2) for count in counts.values()) / max(1, length)
    return 0.65 * coverage + 0.35 * substance - length_penalty - 0.2 * repeated


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Annotated[Path, typer.Option("--config")],
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    chosen_scores = [_heuristic_score(example.prompt, example.chosen) for example in examples]
    rejected_scores = [_heuristic_score(example.prompt, example.rejected) for example in examples]
    metrics = {"pairwise_accuracy": pairwise_accuracy(examples, chosen_scores, rejected_scores)}
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
