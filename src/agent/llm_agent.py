from __future__ import annotations

import os
from openai import OpenAI

from src.utils.logger import get_logger


logger = get_logger("network_sentinel_llm_agent")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Lazily initialize OpenAI client.
    """
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        _client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized.")

    return _client


def build_fallback_response(
    predicted_label: str,
    confidence: float,
    severity: str,
    retrieved_context: str,
    user_query: str | None = None,
) -> str:
    """
    Fallback response when OpenAI fails (quota / error).
    """
    if predicted_label.upper() == "BENIGN":
        key_indicators_text = (
            "The model identified this traffic as normal, expected behavior with no indicators of malicious activity."
        )
        action_text = (
            "No immediate action required. Continue monitoring network activity."
        )
    else:
        key_indicators_text = (
            "The model detected abnormal traffic behavior consistent with a known attack pattern."
        )
        action_text = (
            "Investigate affected systems, review logs, and apply mitigation strategies."
        )

    return f"""
## Attack Summary
The system predicted **{predicted_label}** with confidence **{confidence:.4f}**.

## Severity
**{severity}**

## Key Indicators
{key_indicators_text}

## Retrieved Guidance
{retrieved_context}

## Recommended Action
{action_text}

## Note
OpenAI summarization unavailable. This response was generated using the fallback agent.
""".strip()


def generate_llm_response(
    predicted_label: str,
    confidence: float,
    severity: str,
    retrieved_context: str,
    user_query: str | None = None,
) -> str:
    """
    Try OpenAI first, then fall back if it fails.
    """
    try:
        client = get_openai_client()

        analyst_question = (
            user_query.strip()
            if user_query and user_query.strip()
            else "Explain the attack and mitigation steps."
        )

        prompt = f"""
You are a cybersecurity analyst assistant.

Predicted attack: {predicted_label}
Confidence: {confidence:.4f}
Severity: {severity}

Context:
{retrieved_context}

Question:
{analyst_question}

Provide a structured response with the following sections:

1. Attack Summary
2. Key Indicators
3. Impact
4. Mitigation
5. Final Recommendation

Keep it professional, precise, and practical for a security analyst.
"""

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=500,
        )

        output_text = response.output_text.strip()

        if not output_text:
            raise ValueError("Empty response from OpenAI.")

        return output_text

    except Exception as exc:
        logger.warning("OpenAI failed, using fallback. Error: %s", str(exc))
        return build_fallback_response(
            predicted_label=predicted_label,
            confidence=confidence,
            severity=severity,
            retrieved_context=retrieved_context,
            user_query=user_query,
        )