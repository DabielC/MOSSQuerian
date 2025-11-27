import warnings
import time
import os
import sys
warnings.filterwarnings("ignore")




class _SharedLogger:
    def __init__(self, shared, original_stdout):
        self.shared = shared
        self._original = original_stdout
        self._buffer = ""
        self._lines = []

    def write(self, data):
        self._original.write(data)
        if not data:
            return
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            if line or self._lines:
                self._lines.append(line)

    def flush(self):
        self._original.flush()

    def close(self):
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line or self._lines:
                self._lines.append(line)
            self._buffer = ""
        self.shared["workflow_log"] = self._lines








import argparse
import pymongo
from flow import create_text_to_mongo_flow




def run_text_to_mongo(
    natural_query: str,
    mongo_uri: str = None,
    db_name: str = None,
    api_key: str = None,
    max_debug_retries: int = 2,
    provider: str = None,
    model_name: str = None,
    base_url: str = None,
    thinking = None,
    instruction: str = None,
    stream_manager=None,
    streaming: bool = False,
    timeout: int = 120,
):
    streaming_enabled = bool(stream_manager and streaming)
    shared = {
        "mongo_uri": mongo_uri,
        "db_name": db_name,
        "natural_query": natural_query,
        "api_key": api_key,
        "max_debug_attempts": max_debug_retries,
        "llm_provider": provider,
        "llm_model_name": model_name,
        "llm_base_url": base_url,
        "thinking": thinking,
        "instruction": instruction,
        "streaming_enabled": streaming_enabled,
        "stream_manager": stream_manager if streaming_enabled else None,
        "llm_retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
        "llm_retry_delay": float(os.getenv("LLM_RETRY_DELAY", "2.0")),
        "timeout": timeout,
    }

    original_stdout = sys.stdout
    logger = _SharedLogger(shared, original_stdout)
    sys.stdout = logger
    try:
        if streaming_enabled:
            stream_manager.stage("workflow_execute", "start", provider=provider, model=model_name)

        print()
        print("=== Starting Text-to-MongoDB Workflow ===")
        print(f"Query: '{natural_query}'")
        print(f"Mongo URI: {mongo_uri}")
        print(f"Database: {db_name}")
        print(f"Provider: {provider}")
        print(f"Model: {model_name}")
        print(f"Max Debug Retries on Query Error: {max_debug_retries}")
        print("=" * 45)

        flow = create_text_to_mongo_flow()
        _t0 = time.time()

        try:
            flow.run(shared)
            shared["total_duration_s"] = round(time.time() - _t0, 6)

            total_usage = shared.get("token_usage", {}).copy()
            debug_usage_list = shared.get("debug_token_usage", [])
            rewrite_usage_list = shared.get("rewrite_token_usage", [])
            validation_usage_list = shared.get("validation_token_usage", [])
            correction_plan_usage_list = shared.get("correction_plan_token_usage", [])
            correction_sql_usage_list = shared.get("correction_sql_token_usage", [])

            for debug_usage in debug_usage_list:
                for key, value in debug_usage.items():
                    if key != "provider" and isinstance(value, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + value

            for rewrite_usage in rewrite_usage_list:
                for key, value in rewrite_usage.items():
                    if key != "provider" and isinstance(value, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + value

            for validation_usage in validation_usage_list:
                for key, value in validation_usage.items():
                    if key != "provider" and isinstance(value, (int, float)):
                        total_usage[key] = total_usage.get(key, 0) + value

            for plan_usage in correction_plan_usage_list:
                if isinstance(plan_usage, dict):
                    for key, value in plan_usage.items():
                        if key != "provider" and isinstance(value, (int, float)):
                            total_usage[key] = total_usage.get(key, 0) + value

            for correction_usage in correction_sql_usage_list:
                if isinstance(correction_usage, dict):
                    for key, value in correction_usage.items():
                        if key != "provider" and isinstance(value, (int, float)):
                            total_usage[key] = total_usage.get(key, 0) + value

            shared["total_token_usage"] = total_usage

            if streaming_enabled:
                stream_manager.stage(
                    "workflow_execute",
                    "complete",
                    success=shared.get("final_error") is None,
                    error=shared.get("final_error"),
                )

            if shared.get("final_error"):
                print()
                print("=== Workflow Completed with Error ===")
                print(f"Error: {shared['final_error']}")
            elif shared.get("final_result") is not None:
                print()
                print("=== Workflow Completed Successfully ===")
            else:
                print()
                print("=== Workflow Completed (Unknown State) ===")
        except Exception as e:
            shared["total_duration_s"] = round(time.time() - _t0, 6)
            shared["final_error"] = str(e)
            if streaming_enabled:
                stream_manager.stage("workflow_execute", "error", message=str(e))
            print()
            print("=== Workflow Failed with Exception ===")
            print(f"Error: {e}")
            print(f"Error Type: {type(e).__name__}")
            if "structured_result" in str(e) and "NoneType" in str(e):
                print("This appears to be a Gemini model response parsing issue.")
                print("The LLM may have returned an unexpected response format.")

        print("=" * 36)
        return shared
    finally:
        sys.stdout = original_stdout
        logger.close()

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Text-to-MongoDB CLI")
    parser.add_argument("query", nargs="?", default="total products per category")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "SME_DB"))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--max-debug-retries", type=int, default=int(os.getenv("MAX_DEBUG_RETRIES", "2")))
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model-name", default=os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    args = parser.parse_args()

    run_text_to_mongo(
        args.query, args.mongo_uri, args.db_name,
        api_key=args.api_key, max_debug_retries=args.max_debug_retries,
        provider=args.provider, model_name=args.model_name, base_url=args.base_url
    )



