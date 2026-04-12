from __future__ import annotations

from typing import Any

from rag.knowledge_base import get_or_create_collection
from src.utils.logger import get_logger


logger = get_logger("network_sentinel_retriever")


def retrieve_info(query: str, n_results: int = 2) -> str:
    """
    Retrieve relevant knowledge base text from Chroma.
    Returns combined top documents as a single string.
    """
    try:
        logger.info("Running retrieval query: %s", query)

        collection = get_or_create_collection()

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = results.get("documents", [])

        if not documents or not documents[0]:
            logger.warning("No relevant knowledge found for query.")
            return "No relevant network security guidance was found in the knowledge base."

        combined_context = "\n\n".join(documents[0])

        logger.info(
            "Retrieval completed successfully. Retrieved %d documents.",
            len(documents[0]),
        )

        return combined_context

    except Exception as exc:
        logger.exception("Retriever failed.")
        raise RuntimeError(f"Knowledge retrieval failed: {exc}") from exc


def retrieve_detailed(query: str, n_results: int = 3) -> dict[str, Any]:
    """
    Advanced retrieval for debugging / UI trace.

    Returns:
        {
            "retrieved_context": str,
            "documents": list[str],
            "ids": list[str],
            "distances": list[float] | None
        }
    """
    try:
        logger.info("Running detailed retrieval query: %s", query)

        collection = get_or_create_collection()

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else None

        if not documents:
            logger.warning("No relevant knowledge found for detailed retrieval.")
            return {
                "retrieved_context": "No relevant network security guidance found.",
                "documents": [],
                "ids": [],
                "distances": distances,
            }

        combined_context = "\n\n".join(documents)

        logger.info(
            "Detailed retrieval completed. Documents: %d",
            len(documents),
        )

        return {
            "retrieved_context": combined_context,
            "documents": documents,
            "ids": ids,
            "distances": distances,
        }

    except Exception as exc:
        logger.exception("Detailed retriever failed.")
        raise RuntimeError(f"Detailed knowledge retrieval failed: {exc}") from exc


if __name__ == "__main__":
    test_query = "Explain the DoS Hulk network attack, its indicators, and mitigation steps."
    print(retrieve_info(test_query))