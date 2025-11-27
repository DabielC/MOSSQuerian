import json
import os
import queue
import threading
import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional, Tuple

from nodes import GetSchema, SchemaLinkAgent, SubproblemAgent, QueryPlanAgent, GenerateSQL, ValidateSQL
from main import run_text_to_mongo

app = FastAPI(title="text-2-mongo API")


class StreamManager:
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._closed = False

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        self._queue.put(payload)

    def stage(self, stage: str, status: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {"type": "stage", "stage": stage, "status": status}
        if extra:
            payload.update(extra)
        self._enqueue(payload)

    def token(self, stage: str, token: str) -> None:
        if not token:
            return
        self._enqueue({"type": "token", "stage": stage, "token": token})

    def info(self, message: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {"type": "info", "message": message}
        if extra:
            payload.update(extra)
        self._enqueue(payload)

    def error(self, message: str, stage: Optional[str] = None, **extra: Any) -> None:
        payload: Dict[str, Any] = {"type": "error", "message": message}
        if stage:
            payload["stage"] = stage
        if extra:
            payload.update(extra)
        self._enqueue(payload)

    def final(self, data: Dict[str, Any]) -> None:
        self._enqueue({"type": "final", "data": data})

    def iter_sse(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)


class GenerateRequest(BaseModel):
    query: str
    mongo_uri: Optional[str] = None
    db_name: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    thinking: Optional[Any] = None  # 0/1 for Gemini, True/False for OpenAI
    instruction: Optional[str] = None  # Additional instruction for the LLM
    streaming: Optional[bool] = False


class ExecuteRequest(GenerateRequest):
    max_debug_retries: int = 2
    timeout: int = 120


def _resolve_mongo(
    mongo_uri: Optional[str],
    db_name: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, Optional[str]]:
    resolved_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
    resolved_db = db_name or os.getenv("DB_NAME", "SME_DB")
    resolved_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    return resolved_uri, resolved_db, resolved_key


def _resolve_llm(provider: Optional[str], model_name: Optional[str], base_url: Optional[str]):
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    resolved_model = model_name or os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL")
    resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return resolved_provider, resolved_model, resolved_base_url


def _build_generate_response(shared: Dict[str, Any], provider: str, model_name: Optional[str], validator_action: Any) -> Dict[str, Any]:
    collection, query_type, query = shared.get("generated_query", (None, None, None))
    yaml_payload = {"collection": collection, "query_type": query_type, "query": query}
    return {
        "collection": collection,
        "query_type": query_type,
        "query": query,
        "query_yaml": yaml.safe_dump(yaml_payload, sort_keys=False),
        "schema": shared.get("schema"),
        "collection_summary": shared.get("collection_summary_text"),
        "schema_linking": shared.get("schema_linking"),
        "schema_link_summary": shared.get("schema_link_summary"),
        "schema_link_response": shared.get("schema_link_response"),
        "schema_link_token_usage": shared.get("schema_link_token_usage"),
        "subproblem_summary": shared.get("subproblem_summary"),
        "subproblem_response": shared.get("subproblem_response"),
        "subproblem_token_usage": shared.get("subproblem_token_usage"),
        "query_plan_summary": shared.get("query_plan_summary"),
        "query_plan_response": shared.get("query_plan_response"),
        "query_plan_token_usage": shared.get("query_plan_token_usage"),
        "provider": provider,
        "model_name": model_name,
        "token_usage": shared.get("token_usage", {}),
        "validator_decision": "pass" if shared.get("validation_passed") else "fail",
        "validator_feedback": shared.get("validation_feedback"),
        "validator_action": validator_action,
        "validation_attempts": shared.get("validation_attempts"),
        "validation_responses": shared.get("validation_responses", []),
        "validation_token_usage": shared.get("validation_token_usage", []),
    }

def _build_execute_response(shared: Dict[str, Any], provider: str, model_name: Optional[str]) -> Dict[str, Any]:
    success = shared.get("final_result") is not None
    return {
        "success": success,
        "generated_query": shared.get("generated_query"),
        "results": shared.get("final_result"),
        "columns": shared.get("result_columns"),
        "error": shared.get("final_error") or shared.get("execution_error"),
        "provider": provider,
        "model_name": model_name,
        "debug_attempts": shared.get("debug_attempts"),
        "max_debug_attempts": shared.get("max_debug_attempts"),
        "total_duration_s": shared.get("total_duration_s"),
        "model_response": shared.get("model_response"),
        "response_length": shared.get("response_length"),
        "schema_link_summary": shared.get("schema_link_summary"),
        "subproblem_summary": shared.get("subproblem_summary"),
        "query_plan_summary": shared.get("query_plan_summary"),
        "correction_plan_summary": shared.get("correction_plan_summary"),
        "correction_plan_responses": shared.get("correction_plan_responses", []),
        "correction_plan_token_usage": shared.get("correction_plan_token_usage", []),
        "correction_sql_responses": shared.get("correction_sql_responses", []),
        "correction_sql_token_usage": shared.get("correction_sql_token_usage", []),
        "correction_history": shared.get("correction_history", []),
        "correction_attempts": shared.get("correction_attempts"),
        "debug_responses": shared.get("debug_responses", []),
        "debug_response_lengths": shared.get("debug_response_lengths", []),
        "rewrite_attempts": shared.get("rewrite_attempts"),
        "rewrite_responses": shared.get("rewrite_responses", []),
        "rewrite_response_lengths": shared.get("rewrite_response_lengths", []),
        "token_usage": shared.get("token_usage", {}),
        "debug_token_usage": shared.get("debug_token_usage", []),
        "rewrite_token_usage": shared.get("rewrite_token_usage", []),
        "correction_plan_token_usage": shared.get("correction_plan_token_usage", []),
        "correction_sql_token_usage": shared.get("correction_sql_token_usage", []),
        "retry_cause": shared.get("retry_cause"),
        "retry_feedback": shared.get("retry_feedback"),
        "retry_history": shared.get("retry_history", []),
        "collection_summary": shared.get("collection_summary_text"),
        "validator_decision": "pass" if shared.get("validation_passed") else "fail",
        "validator_feedback": shared.get("validation_feedback"),
        "validation_attempts": shared.get("validation_attempts"),
        "validation_responses": shared.get("validation_responses", []),
        "validation_token_usage": shared.get("validation_token_usage", []),
        "total_token_usage": shared.get("total_token_usage", {}),
    }

def _run_generate(req: GenerateRequest, stream_manager: Optional[StreamManager] = None) -> Dict[str, Any]:
    mongo_uri, db_name, api_key = _resolve_mongo(req.mongo_uri, req.db_name, req.api_key)
    provider, model_name, base_url = _resolve_llm(req.provider, req.model_name, req.base_url)
    streaming_enabled = bool(req.streaming and stream_manager)
    shared: Dict[str, Any] = {
        "mongo_uri": mongo_uri,
        "db_name": db_name,
        "natural_query": req.query,
        "api_key": api_key,
        "llm_provider": provider,
        "llm_model_name": model_name,
        "llm_base_url": base_url,
        "thinking": req.thinking,
        "instruction": req.instruction,
        "streaming_enabled": streaming_enabled,
        "stream_manager": stream_manager if streaming_enabled else None,
        "llm_retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
        "llm_retry_delay": float(os.getenv("LLM_RETRY_DELAY", "2.0")),
    }

    if streaming_enabled:
        stream_manager.stage("workflow_generate", "start", provider=provider, model=model_name)

    get_schema = GetSchema()
    if streaming_enabled:
        stream_manager.stage("schema_fetch", "start")
    try:
        prep1 = get_schema.prep(shared)
        schema = get_schema.exec(prep1)
        get_schema.post(shared, prep1, schema)
    except Exception as exc:
        if streaming_enabled:
            stream_manager.stage("schema_fetch", "error", message=str(exc))
            stream_manager.stage("workflow_generate", "error", message=str(exc))
        raise
    else:
        if streaming_enabled:
            stream_manager.stage(
                "schema_fetch",
                "complete",
                collections=len(shared.get("collection_names") or []),
            )

    try:
        schema_link_agent = SchemaLinkAgent()
        prep_link = schema_link_agent.prep(shared)
        schema_link = schema_link_agent.exec(prep_link)
        schema_link_agent.post(shared, prep_link, schema_link)

        subproblem_agent = SubproblemAgent()
        prep_sub = subproblem_agent.prep(shared)
        subproblem_data = subproblem_agent.exec(prep_sub)
        subproblem_agent.post(shared, prep_sub, subproblem_data)

        plan_agent = QueryPlanAgent()
        prep_plan = plan_agent.prep(shared)
        plan_data = plan_agent.exec(prep_plan)
        plan_agent.post(shared, prep_plan, plan_data)

        gen = GenerateSQL()
        prep2 = gen.prep(shared)
        generated = gen.exec(prep2)
        gen.post(shared, prep2, generated)

        validator = ValidateSQL()
        prep3 = validator.prep(shared)
        validation_result = validator.exec(prep3)
        validator_action = validator.post(shared, prep3, validation_result)
    except Exception as exc:
        if streaming_enabled:
            stream_manager.stage("workflow_generate", "error", message=str(exc))
        raise


    response = _build_generate_response(shared, provider, model_name, validator_action)

    if streaming_enabled:
        stream_manager.stage(
            "workflow_generate",
            "complete",
            success=shared.get("validation_passed"),
            validator_decision=response["validator_decision"],
        )

    return response


def _run_execute(req: ExecuteRequest, stream_manager: Optional[StreamManager] = None) -> Tuple[Dict[str, Any], str, Optional[str]]:
    mongo_uri, db_name, api_key = _resolve_mongo(req.mongo_uri, req.db_name, req.api_key)
    provider, model_name, base_url = _resolve_llm(req.provider, req.model_name, req.base_url)
    shared = run_text_to_mongo(
        natural_query=req.query,
        mongo_uri=mongo_uri,
        db_name=db_name,
        api_key=api_key,
        max_debug_retries=req.max_debug_retries,
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        thinking=req.thinking,
        instruction=req.instruction,
        stream_manager=stream_manager,
        streaming=bool(req.streaming and stream_manager),
        timeout=req.timeout,
    )
    return shared, provider, model_name


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/generate")
def generate(req: GenerateRequest):
    if req.streaming:
        manager = StreamManager()
        manager.info("streaming_started", route="generate")

        def _worker():
            try:
                result = _run_generate(req, manager)
                manager.final(result)
            except Exception as exc:
                manager.error(str(exc), stage="workflow_generate")
            finally:
                manager.close()

        threading.Thread(target=_worker, daemon=True).start()
        return StreamingResponse(
            manager.iter_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return _run_generate(req, None)


@app.post("/v1/execute")
def execute(req: ExecuteRequest):
    if req.streaming:
        manager = StreamManager()
        manager.info("streaming_started", route="execute")

        def _worker():
            try:
                shared, provider, model_name = _run_execute(req, manager)
                result = _build_execute_response(shared, provider, model_name)
                manager.final(result)
            except Exception as exc:
                manager.error(str(exc), stage="workflow_execute")
            finally:
                manager.close()

        threading.Thread(target=_worker, daemon=True).start()
        return StreamingResponse(
            manager.iter_sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    shared, provider, model_name = _run_execute(req, None)
    return _build_execute_response(shared, provider, model_name)


