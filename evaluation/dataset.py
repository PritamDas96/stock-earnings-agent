"""Evaluation dataset definitions.

An evaluation sample pairs a question with a ground-truth reference answer and
the substring(s) that a correct retrieval must surface. Keep the reference text
aligned with documents you ingest before evaluating.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvalSample:
    """A single labelled evaluation example.

    Attributes:
        question: The user query.
        reference: Ground-truth answer used for generation metrics.
        relevant_snippets: Substrings that must appear in a retrieved passage
            for retrieval to count as a hit.
        ticker: Optional ticker filter for retrieval.
    """

    question: str
    reference: str
    relevant_snippets: list[str] = field(default_factory=list)
    ticker: str | None = None


def default_dataset() -> list[EvalSample]:
    """A small, self-contained sample dataset.

    These align with the synthetic document seeded by
    ``scripts/seed_eval_corpus.py`` so the harness is runnable out of the box.
    """
    return [
        EvalSample(
            question="What drove revenue growth this quarter?",
            reference="Revenue growth was driven by strong iPhone and Services performance.",
            relevant_snippets=["Services", "iPhone"],
            ticker="DEMO",
        ),
        EvalSample(
            question="What was the gross margin?",
            reference="Gross margin expanded to 46 percent.",
            relevant_snippets=["46", "margin"],
            ticker="DEMO",
        ),
        EvalSample(
            question="What is management's guidance for next quarter?",
            reference="Management guided to double-digit Services growth next quarter.",
            relevant_snippets=["double-digit", "guid"],
            ticker="DEMO",
        ),
    ]
