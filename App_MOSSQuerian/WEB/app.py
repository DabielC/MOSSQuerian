from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from plotting import build_chart, generate_chart_with_llm, parse_instruction
from services import (
    APIRequestError,
    LLMConfig,
    LLMRequestError,
    MongoConnectionError,
    MongoCredentials,
    call_execute_api,
    call_healthcheck,
    fetch_collection_names,
    fetch_collection_sample,
    infer_collection_schema,
    test_mongo_connection,
)

WEB_MONGO_URI = "mongodb://localhost:27017/"
API_MONGO_URI = "mongodb://mongodb:27017/"

STAGE_LABELS = {
    "workflow_generate": "Generate Endpoint",
    "schema_fetch": "Fetch Schema",
    "generate_sql": "Generate Query",
    "validate_sql": "Validate Query",
    "rewrite_sql": "Rewrite Query",
    "debug_sql": "Debug Query",
    "workflow_execute": "Execute Endpoint",
}
STATUS_LABELS = {
    "start": "Started",
    "complete": "Completed",
    "error": "Error",
    "skipped": "Skipped",
}
STATUS_ICONS = {
    "start": "🟡",
    "complete": "✅",
    "error": "❌",
    "skipped": "⏭️",
}


def _init_session_state() -> None:
    defaults = {
        "mongo_uri": WEB_MONGO_URI,
        "db_name": "SME_DB",
        "api_base_url": "http://localhost:8000",
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "api_key": "",
        "llm_base_url": "",
        "last_query_payload": None,
        "last_query_response": None,
        "results_df": None,
        "plot_instruction": "",
        "llm_plot_spec": None,
        "llm_plot_plan": None,
        "llm_plot_raw_spec": None,
        "llm_plot_meta": None,
        "last_query_events": [],
        "last_stream_snapshot": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _require_connection() -> Optional[MongoCredentials]:
    uri = (st.session_state.get("mongo_uri") or "").strip()
    db_name = (st.session_state.get("db_name") or "").strip()
    if not uri or not db_name:
        st.warning("Please provide MongoDB connection details on the Connection page first.")
        return None
    return MongoCredentials(uri=uri, db_name=db_name)


def _api_mongo_uri(uri: str) -> str:
    """Return the appropriate Mongo URI for backend API requests."""
    normalized = (uri or "").strip()
    if not normalized:
        return API_MONGO_URI
    if normalized.startswith("mongodb://localhost"):
        return API_MONGO_URI
    return normalized


def _results_to_dataframe(results: Any, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Convert diverse API result payloads into a flat DataFrame."""
    if results is None:
        return pd.DataFrame()

    def _is_primitive(value: Any) -> bool:
        return not isinstance(value, (dict, list))

    def _stringify(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return str(value)
        return value

    def _expand_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        while True:
            column_modified = False
            for col in list(frame.columns):
                series = frame[col]
                if series.apply(lambda x: isinstance(x, list)).any():
                    frame[col] = series.apply(lambda x: x if isinstance(x, list) else ([] if x is None else [x]))
                    frame = frame.explode(col).reset_index(drop=True)
                    column_modified = True
                    series = frame[col]
                if series.apply(lambda x: isinstance(x, dict)).any():
                    dict_mask = series.apply(lambda x: isinstance(x, dict))
                    if dict_mask.any():
                        expanded = pd.json_normalize(series[dict_mask], sep=".")
                        expanded.columns = [f"{col}.{sub}" for sub in expanded.columns]
                        expanded = expanded.reindex(series.index)
                        frame = pd.concat([frame.drop(columns=[col]), expanded], axis=1)
                        column_modified = True
                        break
            if not column_modified:
                break
        for col in frame.columns:
            frame[col] = frame[col].apply(_stringify)
        return frame.reset_index(drop=True)

    if isinstance(results, list):
        if not results:
            return pd.DataFrame(columns=columns or [])
        if all(_is_primitive(item) for item in results):
            column = columns[0] if columns else "value"
            return pd.DataFrame({column: results})
        if all(isinstance(item, dict) for item in results):
            df = pd.json_normalize(results, sep=".")
        else:
            df = pd.DataFrame(results)
    elif isinstance(results, dict):
        df = pd.json_normalize(results, sep=".")
    else:
        column = columns[0] if columns else "value"
        df = pd.DataFrame({column: [results]})

    df = _expand_dataframe(df)
    if columns and len(columns) == len(df.columns):
        df.columns = columns
    return df


def page_connection() -> None:
    st.header("Connection & LLM Settings")

    with st.form("connection_settings"):
        st.subheader("MongoDB")
        mongo_uri = st.text_input("Mongo URI", value=st.session_state.get("mongo_uri", ""))
        db_name = st.text_input("Database name", value=st.session_state.get("db_name", ""))

        st.subheader("API")
        api_base_url = st.text_input("API base URL", value=st.session_state.get("api_base_url", ""))

        st.subheader("LLM")
        provider = st.selectbox(
            "Provider",
            options=["gemini", "openai", "openrouter"],
            index=["gemini", "openai", "openrouter"].index(st.session_state.get("provider", "gemini"))
            if st.session_state.get("provider") in {"gemini", "openai", "openrouter"}
            else 0,
        )
        model_name = st.text_input("Model name", value=st.session_state.get("model_name", ""))
        api_key = st.text_input("Model API key", value=st.session_state.get("api_key", ""), type="password")
        llm_base_url = st.text_input("Model base URL (optional)", value=st.session_state.get("llm_base_url", ""))

        submitted = st.form_submit_button("Save settings")

    if submitted:
        st.session_state.mongo_uri = mongo_uri.strip()
        st.session_state.db_name = db_name.strip()
        st.session_state.api_base_url = api_base_url.strip()
        st.session_state.provider = provider.strip() if provider else None
        st.session_state.model_name = model_name.strip()
        st.session_state.api_key = api_key.strip()
        st.session_state.llm_base_url = llm_base_url.strip()
        st.success("Settings updated.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Mongo connection"):
            credentials = _require_connection()
            if not credentials:
                return
            ok, message, collections = test_mongo_connection(credentials)
            if ok:
                st.success(message)
                st.write("Collections:", collections)
            else:
                st.error(message)
    with col2:
        if st.button("Ping API"):
            base_url = st.session_state.get("api_base_url", "").strip()
            if not base_url:
                st.warning("Set the API base URL first.")
            else:
                try:
                    health = call_healthcheck(base_url)
                except APIRequestError as exc:
                    st.error(f"API healthcheck failed: {exc}")
                else:
                    st.success("API reachable")
                    st.json(health)


def page_schema() -> None:
    st.header("Schema Explorer")
    credentials = _require_connection()
    if not credentials:
        return

    try:
        collections = fetch_collection_names(credentials)
    except MongoConnectionError as exc:
        st.error(f"Failed to fetch collections: {exc}")
        return

    if not collections:
        st.info("No collections found in the database.")
        return

    selected = st.selectbox("Select collection", options=collections)
    limit = st.slider("Sample size", min_value=5, max_value=200, value=25, step=5)

    if selected:
        with st.spinner("Fetching sample documents..."):
            try:
                frame, total = fetch_collection_sample(credentials, selected, limit=limit)
            except MongoConnectionError as exc:
                st.error(f"Failed to fetch sample: {exc}")
                return
        st.caption(f"Total documents: {total}")
        if not frame.empty:
            st.dataframe(frame, use_container_width=True)
            st.download_button(
                "Download sample as CSV",
                data=frame.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected}_sample.csv",
                mime="text/csv",
            )
        else:
            st.info("No sample data available for this collection.")

        with st.spinner("Inferring schema..."):
            try:
                schema_df = infer_collection_schema(credentials, selected)
            except MongoConnectionError as exc:
                st.error(f"Failed to infer schema: {exc}")
            else:
                if not schema_df.empty:
                    st.markdown("### Field overview")
                    st.dataframe(schema_df, use_container_width=True)
                else:
                    st.info("Schema information unavailable for this collection.")


def page_query() -> None:
    st.header("Natural Language Query")
    credentials = _require_connection()
    if not credentials:
        return

    base_url = st.session_state.get("api_base_url", "").strip()
    if not base_url:
        st.error("Set the API Base URL on the Connection page.")
        return

    provider = st.session_state.get("provider") or None
    model_name = st.session_state.get("model_name") or None
    api_key = st.session_state.get("api_key") or None
    llm_base_url = st.session_state.get("llm_base_url") or None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Query")
        query = st.text_area(
            "Ask a question about your data",
            value="",
            height=120,
            placeholder="e.g. Show all customers",
        )
        instruction = st.text_area(
            "Optional extra instruction",
            value="",
            height=80,
            placeholder="Add guardrails or clarifications for the model",
        )

        st.subheader("Configuration")
        max_debug_retries = st.slider("Max debug retries", min_value=0, max_value=10, value=2)
        timeout = st.number_input("Timeout (seconds)", min_value=10, max_value=300, value=120)
        if provider == "gemini":
            thinking_tokens = st.number_input("Thinking tokens", min_value=0, max_value=8000, value=1024)
        else:
            thinking_choice = st.radio(
                "Thinking mode",
                options=["Auto", "Off (0)", "On (1)"],
                index=0,
                horizontal=True,
            )
        streaming_enabled = st.checkbox("Stream model output", value=True)

        submitted = st.button("Run query")
    
    with col2:
        if submitted:
            if not query.strip():
                st.warning("Enter a question before running the query.")
                return
            thinking: Optional[int]
            if provider == "gemini":
                thinking = thinking_tokens
            else:
                if thinking_choice == "Off (0)":
                    thinking = 0
                elif thinking_choice == "On (1)":
                    thinking = 1
                else:
                    thinking = None

            payload: Dict[str, Any] = {
                "query": query.strip(),
                "mongo_uri": _api_mongo_uri(credentials.uri),
                "db_name": credentials.db_name,
                "provider": provider,
                "model_name": model_name,
                "api_key": api_key,
                "base_url": llm_base_url or None,
                "thinking": thinking,
                "instruction": instruction.strip() or None,
                "max_debug_retries": max_debug_retries,
                "streaming": bool(streaming_enabled),
                "timeout": timeout,
            }
            st.session_state.last_query_payload = payload

            with st.expander("Assistant Activity", expanded=True):
                status_container = st.container()
                stream_placeholder = status_container.empty()

                def render_snapshot(snapshot: Optional[Dict[str, Any]]) -> None:
                    if not snapshot:
                        stream_placeholder.empty()
                        return
                    stage_order = list(snapshot.get("stage_order") or [])
                    stage_status_map = snapshot.get("stage_status") or {}
                    if isinstance(stage_status_map, OrderedDict):
                        stage_status_map = dict(stage_status_map)
                    token_buffers_map = snapshot.get("token_buffers") or {}
                    info_lines = list(snapshot.get("info_lines") or [])
                    streaming_flag = snapshot.get("streaming", True)

                    with stream_placeholder.container():
                        st.markdown("### Assistant activity")
                        if not stage_order:
                            if streaming_flag:
                                st.markdown("🧠 **Thinking longer for a better answer...**")
                                st.caption("Skip ›")
                            else:
                                st.info("Streaming disabled for this run.")
                            if info_lines:
                                for msg in info_lines[-3:]:
                                    st.caption(msg)
                            return

                        completed = [stage for stage in stage_order if stage_status_map.get(stage) in {"complete", "skipped"}]
                        all_complete = bool(stage_order) and len(completed) == len(stage_order)
                        active_stage = next((stage for stage in reversed(stage_order) if stage_status_map.get(stage) not in {"complete", "skipped"}), stage_order[-1])
                        active_label = STAGE_LABELS.get(active_stage, active_stage.replace('_', ' ').title())

                        if not all_complete or streaming_flag:
                            st.markdown("🧠 **Thinking longer for a better answer...**")
                            st.caption(f"Currently working on: {active_label}")
                            st.caption("Skip ›")
                        else:
                            st.markdown("✅ **Assistant ready with an answer**")

                        with st.expander("Progress details", expanded=False):
                            for stage in stage_order:
                                status = stage_status_map.get(stage)
                                label = STAGE_LABELS.get(stage, stage.replace('_', ' ').title())
                                icon = STATUS_ICONS.get(status, "⏳")
                                status_text = STATUS_LABELS.get(status, status.capitalize() if status else "In progress")
                                st.write(f"{icon} **{label}** — {status_text}")
                                tokens = token_buffers_map.get(stage)
                                if tokens:
                                    st.caption(tokens[-200:])
                            if info_lines:
                                st.markdown("**Recent messages**")
                                for msg in info_lines[-3:]:
                                    st.caption(msg)

                stage_order: list[str] = []
                stage_status: OrderedDict[str, str] = OrderedDict()
                token_buffers = defaultdict(str)
                info_lines: list[str] = []

                def snapshot_dict(streaming_flag: bool = True) -> Dict[str, Any]:
                    return {
                        "stage_order": stage_order,
                        "stage_status": stage_status,
                        "token_buffers": token_buffers,
                        "info_lines": info_lines,
                        "streaming": streaming_flag,
                    }

                handler = None
                if streaming_enabled:
                    render_snapshot(snapshot_dict())

                    def _handle_event(event: Dict[str, Any]) -> None:
                        event_type = event.get("type")
                        if event_type == "stage":
                            stage = (event.get("stage") or "unknown").lower()
                            status_val = (event.get("status") or "").lower()
                            if stage not in stage_status:
                                stage_order.append(stage)
                            stage_status[stage] = status_val
                            message = event.get("message")
                            if message:
                                info_lines.append(f"{STAGE_LABELS.get(stage, stage)}: {message}")
                        elif event_type == "token":
                            stage = (event.get("stage") or "generate_sql").lower()
                            token_buffers[stage] += event.get("token", "")
                            if stage not in stage_order:
                                stage_order.append(stage)
                        elif event_type == "info":
                            message = event.get("message")
                            if message:
                                info_lines.append(str(message))
                        elif event_type == "error":
                            stage = (event.get("stage") or "unknown")
                            info_lines.append(
                                f"Error ({STAGE_LABELS.get(stage, stage)}): {event.get('message', '')}"
                            )
                        elif event_type == "final":
                            info_lines.append("Workflow finished.")
                        render_snapshot(snapshot_dict())

                    handler = _handle_event
                else:
                    info_lines.append("Streaming disabled; waiting for final response.")
                    render_snapshot(snapshot_dict(streaming_flag=False))

                try:
                    response, events = call_execute_api(
                        base_url,
                        payload,
                        timeout=timeout,
                        stream=bool(streaming_enabled),
                        event_handler=handler,
                    )
                except APIRequestError as exc:
                    error_message = f"Query failed: {exc}"
                    info_lines.append(error_message)
                    render_snapshot(snapshot_dict(streaming_enabled))
                    st.session_state.last_stream_snapshot = {
                        "stage_order": list(stage_order),
                        "stage_status": dict(stage_status),
                        "token_buffers": {k: v[-2000:] for k, v in token_buffers.items()},
                        "info_lines": info_lines[-20:],
                        "streaming": bool(streaming_enabled),
                    }
                    st.session_state.last_query_events = []
                    st.error(error_message)
                    return

                if not streaming_enabled:
                    info_lines.append("Workflow completed (no streaming).")
                    render_snapshot(snapshot_dict(streaming_flag=False))

                final_snapshot = {
                    "stage_order": list(stage_order),
                    "stage_status": dict(stage_status),
                    "token_buffers": {k: v[-2000:] for k, v in token_buffers.items()},
                    "info_lines": info_lines[-20:],
                    "streaming": bool(streaming_enabled),
                }
                render_snapshot(final_snapshot)
                st.session_state.last_stream_snapshot = final_snapshot
                st.session_state.last_query_events = events
                st.session_state.last_query_response = response
                df = _results_to_dataframe(response.get("results"), response.get("columns"))
                st.session_state.results_df = df
                st.session_state.llm_plot_spec = None
                st.session_state.llm_plot_plan = None
                st.session_state.llm_plot_raw_spec = None
                st.session_state.llm_plot_meta = None

        response = st.session_state.get("last_query_response")
        df: Optional[pd.DataFrame] = st.session_state.get("results_df")

        if not response:
            st.info("Run a query to see results.")
            return

        status = "Success" if response.get("success") else "Error"
        st.subheader(f"Last query status: {status}")
        if response.get("error"):
            st.error(response["error"])

        with st.expander("Query details", expanded=False):
            st.write("**Generated query**")
            st.json(response.get("generated_query"))
            if response.get("model_response"):
                st.write("**Model response**")
                st.code(response["model_response"], language="yaml")
            if response.get("collection_summary"):
                st.write("**Collection summary**")
                st.text(response["collection_summary"])

        if df is not None and not df.empty:
            st.subheader("Query results")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_results.csv",
                mime="text/csv",
            )

            st.markdown("### Natural language plotting")
            plot_instruction = st.text_input(
                "Describe the plot you want",
                value=st.session_state.get("plot_instruction", ""),
                placeholder="e.g. Bar chart of revenue by province",
            )
            st.session_state.plot_instruction = plot_instruction

            if st.button("Generate plot (rule-based)"):
                try:
                    plan = parse_instruction(plot_instruction, df.columns.tolist())
                    chart = build_chart(df, plan)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.subheader("Rule-based Plot")
                    st.altair_chart(chart, use_container_width=True)

            if st.button("Ask LLM to design plot"):
                if not plot_instruction.strip():
                    st.warning("Provide a plot request before asking the LLM.")
                elif not provider or not model_name:
                    st.error("Set the LLM provider and model on the Connection page.")
                elif not api_key:
                    st.error("Set the model API key on the Connection page.")
                else:
                    config = LLMConfig(
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                        base_url=llm_base_url or None,
                    )
                    with st.spinner("LLM is planning the visualization..."):
                        try:
                            outcome = generate_chart_with_llm(df, plot_instruction, config)
                        except Exception as exc:
                            st.error(f"LLM plotting failed: {exc}")
                        else:
                            spec_payload = {
                                "kind": outcome.kind,
                                "chart": outcome.chart.to_dict() if outcome.chart is not None else None,
                                "code": outcome.code,
                            }
                            st.session_state.llm_plot_spec = spec_payload
                            st.session_state.llm_plot_figure = outcome.figure
                            st.session_state.llm_plot_plan = dict(vars(outcome.plan)) if outcome.plan else {}
                            st.session_state.llm_plot_raw_spec = outcome.raw_spec
                            st.session_state.llm_plot_meta = {
                                "thinking": outcome.thinking,
                                "title": outcome.title,
                                "description": outcome.description,
                                "code": outcome.code,
                            }
                            st.success("LLM-designed chart ready")

            llm_payload = st.session_state.get("llm_plot_spec") or {}
            meta = st.session_state.get("llm_plot_meta") or {}
            title = meta.get("title") or "LLM-designed chart"
            if llm_payload:
                st.subheader("LLM-designed Plot")
                kind = llm_payload.get("kind")
                figure = st.session_state.get("llm_plot_figure")
                chart_dict = llm_payload.get("chart")
                if kind == "matplotlib" and figure is not None:
                    st.subheader(title)
                    st.pyplot(figure)
                elif chart_dict:
                    try:
                        llm_chart = alt.Chart.from_dict(chart_dict)
                    except Exception:
                        llm_chart = None
                    if llm_chart is not None:
                        st.subheader(title)
                        st.altair_chart(llm_chart, use_container_width=True)
                plan_info = st.session_state.get("llm_plot_plan") or {}
                thinking = meta.get("thinking") or plan_info.get("thinking")
                if thinking:
                    with st.expander("LLM thinking", expanded=False):
                        st.write(thinking)
                description = meta.get("description") or plan_info.get("description")
                if description:
                    st.caption(description)
                code_snippet = llm_payload.get("code") or meta.get("code")
                if code_snippet:
                    with st.expander("LLM generated code", expanded=False):
                        st.code(code_snippet, language="python")
                raw_spec = st.session_state.get("llm_plot_raw_spec")
                if raw_spec:
                    with st.expander("LLM chart specification", expanded=False):
                        st.code(json.dumps(raw_spec, indent=2))
        else:
            st.info("No tabular results returned. Check the query or try again.")


PAGES = {
    "Connection": page_connection,
    "Schema": page_schema,
    "Natural Language Query": page_query,
}


def main() -> None:
    st.set_page_config(page_title="text-2-mongo Console", layout="wide")
    _init_session_state()
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()


