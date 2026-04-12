from __future__ import annotations

from typing import Any

from rag.retriever import retrieve_info
from src.agent.llm_agent import generate_llm_response
from src.utils.logger import get_logger


logger = get_logger("network_sentinel_incident_agent")


def build_retrieval_query(
    predicted_label: str,
    confidence: float,
    user_query: str | None = None,
) -> str:
    """
    Build retrieval query based on predicted attack label and analyst question.
    """
    normalized_label = predicted_label.strip().lower()

    if normalized_label == "benign":
        base_query = (
            "Explain normal network traffic behavior, why it is considered benign, "
            "common indicators of benign traffic, and whether any action is required."
        )
    else:
        base_query = (
            f"Explain the {predicted_label} network attack, "
            "its indicators, likely impact, and mitigation steps."
        )

    if confidence < 0.70:
        base_query += " Also mention that the model confidence is moderate and manual validation is recommended."

    if user_query and user_query.strip():
        base_query += f" Also answer this analyst question: {user_query.strip()}"

    return base_query


def assess_severity(predicted_label: str, confidence: float) -> str:
    """
    Assign severity using predicted label and confidence.
    """
    normalized_label = predicted_label.strip().lower()

    if normalized_label == "benign":
        return "Low"

    if "heartbleed" in normalized_label:
        return "Critical"

    if any(
        attack in normalized_label
        for attack in ["dos", "goldeneye", "hulk", "slowloris", "slowhttptest", "ddos"]
    ):
        return "High" if confidence >= 0.80 else "Medium"

    if confidence >= 0.80:
        return "High"

    if confidence >= 0.60:
        return "Medium"

    return "Low"


def extract_guidance_from_response(final_response: str) -> str:
    """
    Derive guidance text from the generated response.
    """
    if not final_response or not final_response.strip():
        return "No guidance available."

    return final_response.strip()


def compose_incident_report(
    predicted_label: str,
    confidence: float,
    severity: str,
    retrieval_query: str,
    retrieved_context: str,
    final_response: str,
    fallback_used: bool = False,
) -> dict[str, Any]:
    return {
        "predicted_attack": predicted_label,
        "confidence": round(confidence, 4),
        "severity": severity,
        "retrieval_query": retrieval_query,
        "retrieved_context": retrieved_context,
        "summary": final_response,
        "retrieved_guidance": extract_guidance_from_response(final_response),
        "fallback_used": fallback_used,
    }


def run_incident_agent(
    predicted_attack: str,
    confidence: float,
    user_query: str | None = None,
) -> dict[str, Any]:
    """
    Run retrieval + LLM generation using already-predicted attack output.
    Matches app.py contract.
    """
    try:
        logger.info(
            "Running incident intelligence agent for predicted attack: %s",
            predicted_attack,
        )

        retrieval_query = build_retrieval_query(
            predicted_label=predicted_attack,
            confidence=confidence,
            user_query=user_query,
        )

        logger.info("Retrieving context from Chroma.")
        retrieved_context = retrieve_info(retrieval_query)

        severity = assess_severity(predicted_attack, confidence)

        logger.info("Generating final LLM response.")
        final_response = generate_llm_response(
            predicted_label=predicted_attack,
            confidence=confidence,
            severity=severity,
            retrieved_context=retrieved_context,
            user_query=user_query,
        )

        fallback_used = "fallback agent" in final_response.lower() or "openai summarization unavailable" in final_response.lower()

        logger.info("Incident report generated successfully.")
        return compose_incident_report(
            predicted_label=predicted_attack,
            confidence=confidence,
            severity=severity,
            retrieval_query=retrieval_query,
            retrieved_context=retrieved_context,
            final_response=final_response,
            fallback_used=fallback_used,
        )

    except Exception as exc:
        logger.exception("Incident agent failed.")
        raise RuntimeError(f"Incident intelligence agent failed: {exc}") from exc