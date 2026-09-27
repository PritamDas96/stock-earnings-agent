"""RAG quality evaluation harness.

Two complementary evaluators:
  - :class:`RetrievalEvaluator` — deterministic retrieval metrics (hit-rate@k,
    MRR) that need only embeddings + the vector store (no LLM tokens).
  - :func:`run_ragas_evaluation` — RAGAS generation metrics (faithfulness,
    answer relevancy, context precision/recall) using Groq + Gemini.
"""

from evaluation.dataset import EvalSample, default_dataset
from evaluation.ragas_eval import RetrievalEvaluator, run_ragas_evaluation

__all__ = ["EvalSample", "default_dataset", "RetrievalEvaluator", "run_ragas_evaluation"]
