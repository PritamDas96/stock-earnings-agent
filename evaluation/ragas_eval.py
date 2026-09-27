"""Retrieval and generation quality evaluation.

The deterministic :class:`RetrievalEvaluator` measures whether the right
passages are retrieved. :func:`run_ragas_evaluation` layers RAGAS generation
metrics on top when an LLM budget is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from core import get_logger
from evaluation.dataset import EvalSample, default_dataset
from rag_pipeline import GoogleEmbeddingClient, RAGPipeline

log = get_logger("evaluation")


class GeminiEmbeddingsAdapter(Embeddings):
    """Adapt :class:`GoogleEmbeddingClient` to the LangChain ``Embeddings`` API.

    Lets the Gemini REST client be used anywhere a LangChain embeddings object is
    expected (e.g. RAGAS metrics).
    """

    def __init__(self, client: GoogleEmbeddingClient | None = None) -> None:
        self._client = client or GoogleEmbeddingClient()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


@dataclass(slots=True)
class RetrievalReport:
    """Aggregate retrieval metrics over a dataset."""

    samples: int
    hit_rate: float
    mrr: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
        }


class RetrievalEvaluator:
    """Deterministic retrieval quality metrics (no LLM tokens required)."""

    def __init__(self, pipeline: RAGPipeline | None = None, top_k: int = 5) -> None:
        self._pipeline = pipeline or RAGPipeline()
        self._top_k = top_k

    def evaluate(self, dataset: Sequence[EvalSample] | None = None) -> RetrievalReport:
        """Compute hit-rate@k and Mean Reciprocal Rank over ``dataset``."""
        data = list(dataset or default_dataset())
        if not data:
            return RetrievalReport(0, 0.0, 0.0)

        hits = 0
        reciprocal_ranks = 0.0
        for sample in data:
            chunks = self._pipeline.retrieve(
                sample.question, top_k=self._top_k, ticker=sample.ticker
            )
            rank = self._first_relevant_rank(chunks, sample.relevant_snippets)
            if rank is not None:
                hits += 1
                reciprocal_ranks += 1.0 / rank

        n = len(data)
        report = RetrievalReport(samples=n, hit_rate=hits / n, mrr=reciprocal_ranks / n)
        log.info("Retrieval eval: {}", report.as_dict())
        return report

    @staticmethod
    def _first_relevant_rank(chunks, snippets: list[str]) -> int | None:
        if not snippets:
            return None
        for position, chunk in enumerate(chunks, start=1):
            text = chunk.text.lower()
            if all(snippet.lower() in text for snippet in snippets):
                return position
        return None


def run_ragas_evaluation(
    dataset: Sequence[EvalSample] | None = None,
    top_k: int = 5,
) -> dict:
    """Run RAGAS generation metrics over the dataset.

    For each sample: retrieve contexts, generate an answer with the Groq LLM,
    then score faithfulness, answer relevancy and context precision/recall.

    Returns:
        A dict of metric name → score. On free-tier token limits or RAGAS
        version incompatibilities, returns ``{"error": ...}`` with a clear
        message rather than raising.
    """
    data = list(dataset or default_dataset())
    pipeline = RAGPipeline()

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        from agents.llm import get_llm
    except ImportError as exc:  # pragma: no cover - environment dependent
        return {"error": f"RAGAS dependencies unavailable: {exc}"}

    records = []
    for sample in data:
        chunks = pipeline.retrieve(sample.question, top_k=top_k, ticker=sample.ticker)
        contexts = [c.text for c in chunks] or ["No context retrieved."]
        prompt = (
            "Answer the question using only the context.\n\n"
            f"Context:\n{chr(10).join(contexts)}\n\nQuestion: {sample.question}"
        )
        try:
            response = get_llm().invoke(prompt).content
        except Exception as exc:  # noqa: BLE001
            return {"error": f"LLM generation failed (likely token/rate limit): {exc}"}

        records.append(
            {
                "user_input": sample.question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": sample.reference,
            }
        )

    try:
        eval_dataset = EvaluationDataset.from_list(records)
        result = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(get_llm()),
            embeddings=LangchainEmbeddingsWrapper(GeminiEmbeddingsAdapter()),
        )
        return dict(result)
    except Exception as exc:  # noqa: BLE001 - RAGAS surface area is large
        return {"error": f"RAGAS evaluation failed: {exc}"}


if __name__ == "__main__":
    report = RetrievalEvaluator().evaluate()
    print("Retrieval metrics:", report.as_dict())
