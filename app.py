from __future__ import annotations

from datetime import datetime
from io import StringIO
from typing import Any, Dict

import pandas as pd
import streamlit as st

from rag.knowledge_base import seed_knowledge_base
from src.agent.incident_agent import run_incident_agent
from src.models.predict import predict_attack_batch
from src.utils.logger import get_logger

logger = get_logger("network_sentinel_app")

st.set_page_config(
    page_title="Network Sentinel",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #94a3b8;
            margin-bottom: 1.2rem;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            color: white;
        }
        .low { background-color: #16a34a; }
        .medium { background-color: #ca8a04; }
        .high { background-color: #dc2626; }
        .critical { background-color: #7f1d1d; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_severity(label: str) -> str:
    label = str(label).strip().lower()
    if label == "benign":
        return "low"
    if "heartbleed" in label:
        return "critical"
    if any(x in label for x in ["dos", "goldeneye", "hulk", "slowloris", "slowhttptest"]):
        return "high"
    return "medium"


def severity_badge(severity: str) -> str:
    return f'<span class="badge {severity}">{severity.upper()}</span>'


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("latin1")
        return pd.read_csv(StringIO(content))


@st.cache_resource
def initialize_resources() -> bool:
    seed_knowledge_base()
    return True


def render_header() -> None:
    st.markdown('<div class="main-title">Network Sentinel</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">ML + RAG intrusion detection dashboard</div>',
        unsafe_allow_html=True,
    )


def render_overview(df: pd.DataFrame, file_name: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("File", file_name)
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Columns", f"{df.shape[1]:,}")

    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)


def render_result(result: Dict[str, Any], row_count: int) -> None:
    label = result.get("predicted_attack", "Unknown")
    confidence = min(float(result.get("confidence", 0.0)), 0.999)
    severity = result.get("severity", get_severity(label))
    summary = result.get("summary", "No summary available.")
    retrieved_context = result.get("retrieved_context", "No retrieved context available.")
    model_name = result.get("model_name", "XGBoost")
    fallback_used = result.get("fallback_used", False)
    retrieval_query = result.get("retrieval_query", "N/A")

    st.subheader("Detection Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Attack", label)
    c2.metric("Confidence", f"{confidence:.2%}")
    c3.markdown(f"**Severity**  \n{severity_badge(severity)}", unsafe_allow_html=True)
    c4.metric("Rows Processed", f"{row_count:,}")

    st.progress(max(0.0, min(confidence, 1.0)))

    st.subheader("Incident Summary")
    st.write(summary)

    with st.expander("Technical Details", expanded=False):
        st.write(f"**Model Used:** {model_name}")
        st.write(f"**Fallback Used:** {'Yes' if fallback_used else 'No'}")
        st.write(f"**Retrieval Query:** `{retrieval_query}`")
        st.write(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("**Retrieved Context**")
        st.code(retrieved_context, language="markdown")


def main() -> None:
    render_header()
    initialize_resources()

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Upload network traffic CSV",
            type=["csv"],
            help="Upload CICIDS-style traffic data for inference.",
        )
        show_preview = st.checkbox("Show data preview", value=True)
        show_debug = st.checkbox("Show technical diagnostics", value=False)
        run_analysis = st.button("Run Analysis", use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = safe_read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
        st.session_state["uploaded_filename"] = uploaded_file.name
    except Exception as exc:
        logger.exception("Failed to load uploaded file.")
        st.error(f"Failed to read uploaded CSV: {exc}")
        return

    if show_preview:
        render_overview(df, uploaded_file.name)

    if not run_analysis:
        return

    try:
        with st.spinner("Running analysis..."):
            prediction_output = predict_attack_batch(df)

            incident_result = run_incident_agent(
                predicted_attack=prediction_output["predicted_attack"],
                confidence=prediction_output.get("confidence", 0.0),
            )

            final_result = {
                "predicted_attack": prediction_output["predicted_attack"],
                "confidence": prediction_output.get("confidence", 0.0),
                "model_name": prediction_output.get("model_name", "XGBoost"),
                "severity": get_severity(prediction_output["predicted_attack"]),
                **incident_result,
            }

            st.session_state["final_result"] = final_result

        st.success("Analysis completed successfully.")
        render_result(st.session_state["final_result"], len(df))

        if show_debug:
            with st.expander("Debug Payload", expanded=False):
                st.json(st.session_state["final_result"])

    except Exception as exc:
        logger.exception("Analysis failed.")
        st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()