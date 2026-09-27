"""Seed a small synthetic corpus so the evaluation harness is runnable.

Ingests a single demo earnings snippet under ticker ``DEMO`` matching the
references in :mod:`evaluation.dataset`.

Usage:
    python -m scripts.seed_eval_corpus
"""

from __future__ import annotations

from rag_pipeline import RAGPipeline
from rag_pipeline.ingestion import Document

DEMO_TEXT = (
    "In the third quarter, the company reported record revenue of $94 billion, "
    "driven by strong iPhone sales and continued momentum in Services. "
    "Gross margin expanded to 46 percent on a favourable product mix. "
    "Management guided to double-digit Services growth next quarter and "
    "reaffirmed its commitment to returning capital to shareholders."
)


def main() -> None:
    pipeline = RAGPipeline()
    document = Document(
        content=DEMO_TEXT,
        metadata={
            "source": "demo_earnings_call.txt",
            "ticker": "DEMO",
            "doc_type": "earnings_call",
        },
    )
    count = pipeline.ingest_documents([document])
    print(f"Seeded {count} chunk(s) under ticker DEMO. Collection holds {pipeline.store.count()}.")


if __name__ == "__main__":
    main()
