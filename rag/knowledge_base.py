from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import chromadb

from src.utils.logger import get_logger


logger = get_logger("network_sentinel_knowledge_base")

CHROMA_DB_PATH = Path("rag/chroma_db")
COLLECTION_NAME = "network_security"


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a persistent Chroma client.
    """
    try:
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        logger.info("Initializing Chroma client at path: %s", CHROMA_DB_PATH)
        return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    except Exception as exc:
        logger.exception("Failed to initialize Chroma client.")
        raise RuntimeError(f"Failed to initialize Chroma client: {exc}") from exc


def get_or_create_collection():
    """
    Return the network security collection, creating it if needed.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logger.info("Loaded or created Chroma collection: %s", COLLECTION_NAME)
        return collection

    except Exception as exc:
        logger.exception("Failed to get or create Chroma collection.")
        raise RuntimeError(f"Failed to access Chroma collection: {exc}") from exc


def _build_documents() -> List[Dict[str, str]]:
    """
    Centralized knowledge base definitions (easy to extend later).
    """
    return [
        {
            "id": "benign",
            "text": (
                "Benign network traffic represents normal, expected communication behavior. "
                "Indicators include routine session patterns, expected packet timing, valid service usage, "
                "and no clear signs of scanning, flooding, exploitation, or lateral movement. "
                "Recommended action is continued monitoring only. No immediate containment or mitigation is required "
                "unless additional evidence suggests suspicious behavior."
            ),
        },
        {
            "id": "dos_hulk",
            "text": (
                "DoS Hulk is a denial-of-service attack characterized by high request volume and burst traffic. "
                "Indicators include spikes in requests, abnormal inter-arrival times, and service degradation. "
                "Mitigation includes rate limiting, firewall rules, traffic filtering, and upstream protection."
            ),
        },
        {
            "id": "dos_slowloris",
            "text": (
                "Slowloris attacks keep connections open for long durations to exhaust server resources. "
                "Indicators include long-lived connections with minimal data transfer. "
                "Mitigation includes connection timeouts, reverse proxies, and web server tuning."
            ),
        },
        {
            "id": "dos_slowhttptest",
            "text": (
                "Slow HTTP test attacks simulate slow client behavior to exhaust resources. "
                "Indicators include incomplete HTTP requests and abnormal connection patterns. "
                "Mitigation includes request rate limiting and connection management."
            ),
        },
        {
            "id": "ddos",
            "text": (
                "DDoS attacks overwhelm systems using distributed sources. "
                "Indicators include traffic spikes, multiple IP sources, and resource exhaustion. "
                "Mitigation includes traffic scrubbing, rate limiting, and provider-level protection."
            ),
        },
        {
            "id": "portscan",
            "text": (
                "Port scanning is reconnaissance activity probing multiple ports. "
                "Indicators include rapid connection attempts across ports. "
                "Mitigation includes IDS alerts, firewall rules, and blocking suspicious IPs."
            ),
        },
        {
            "id": "bot",
            "text": (
                "Bot activity indicates automated or compromised hosts. "
                "Indicators include repetitive patterns and command-and-control behavior. "
                "Mitigation includes host isolation and malware analysis."
            ),
        },
        {
            "id": "infiltration",
            "text": (
                "Infiltration suggests unauthorized access or lateral movement. "
                "Indicators include abnormal sessions and communication patterns. "
                "Mitigation includes segmentation, credential audits, and incident response."
            ),
        },
        {
            "id": "heartbleed",
            "text": (
                "Heartbleed exploits TLS vulnerabilities to leak memory. "
                "Indicators include abnormal heartbeat requests. "
                "Mitigation includes patching systems, rotating keys, and certificate revocation."
            ),
        },
        {
            "id": "web_attack",
            "text": (
                "Web attacks include SQL injection, XSS, and brute-force attempts. "
                "Indicators include suspicious payloads and repeated request patterns. "
                "Mitigation includes WAF rules, validation, and monitoring."
            ),
        },
    ]


def seed_knowledge_base() -> None:
    """
    Seed Chroma with baseline cybersecurity knowledge.
    Safe to run multiple times (idempotent).
    """
    try:
        collection = get_or_create_collection()

        documents = _build_documents()

        existing = collection.get()
        existing_ids = set(existing.get("ids", []))

        new_ids = [doc["id"] for doc in documents if doc["id"] not in existing_ids]
        new_docs = [doc["text"] for doc in documents if doc["id"] not in existing_ids]

        if new_ids:
            collection.add(ids=new_ids, documents=new_docs)
            logger.info("Added %s new knowledge base documents.", len(new_ids))
        else:
            logger.info("Knowledge base already seeded.")

    except Exception as exc:
        logger.exception("Failed to seed knowledge base.")
        raise RuntimeError(f"Failed to seed knowledge base: {exc}") from exc


def reset_knowledge_base() -> None:
    """
    Optional utility for development: clears and reseeds DB.
    """
    try:
        client = get_chroma_client()
        client.delete_collection(name=COLLECTION_NAME)
        logger.warning("Knowledge base collection deleted.")

        seed_knowledge_base()
        logger.info("Knowledge base reset completed.")

    except Exception as exc:
        logger.exception("Failed to reset knowledge base.")
        raise RuntimeError(f"Failed to reset knowledge base: {exc}") from exc


if __name__ == "__main__":
    seed_knowledge_base()