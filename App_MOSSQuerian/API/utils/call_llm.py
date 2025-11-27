import os
import argparse
from typing import Any, Callable, Dict, Optional
from google import genai
from google.genai import types
from openai import OpenAI
from urllib.parse import urlparse

LIST_OF_OPENROUTER_MODELS = [
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "moonshotai/kimi-k2:free",
    "qwen/qwen3-235b-a22b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free",
    "moonshotai/kimi-dev-72b:free",
    "scb10x/llama3.1-typhoon2-70b-instruct",
    "openai/gpt-oss-20b:free"
]
LIST_OF_GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]

def _openai_model_supports_reasoning(model_name: str) -> bool:
    if not model_name:
        return False
    name = model_name.lower()
    reasoning_keywords = ("o1", "o3", "o4", "gpt-4o", "gpt-4.1")
    return any(keyword in name for keyword in reasoning_keywords)

def _flatten_chat_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    try:
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if text_value:
                    parts.append(str(text_value))
                continue
            text_attr = getattr(item, "text", None)
            if text_attr:
                parts.append(str(text_attr))
    except TypeError:
        pass
    return "".join(parts)

def _usage_from_source(provider: str, usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {"provider": provider}
    if isinstance(usage_obj, dict):
        prompt_tokens = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
        completion_tokens = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
        total_tokens = usage_obj.get("total_tokens") or usage_obj.get("total") or (prompt_tokens + completion_tokens)
        result: Dict[str, Any] = {
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if "reasoning_tokens" in usage_obj:
            result["reasoning_tokens"] = usage_obj["reasoning_tokens"]
        if "thinking_tokens" in usage_obj:
            result["thinking_tokens"] = usage_obj["thinking_tokens"]
        return result
    prompt_tokens = getattr(usage_obj, "prompt_tokens", getattr(usage_obj, "input_tokens", 0))
    completion_tokens = getattr(usage_obj, "completion_tokens", getattr(usage_obj, "output_tokens", 0))
    total_tokens = getattr(usage_obj, "total_tokens", getattr(usage_obj, "total", prompt_tokens + completion_tokens))
    result = {
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    reasoning_tokens = getattr(usage_obj, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens
    thinking_tokens = getattr(usage_obj, "thinking_tokens", None)
    if thinking_tokens is not None:
        result["thinking_tokens"] = thinking_tokens
    return result

def _stream_chat_completion(client: OpenAI, provider: str, *, on_token: Optional[Callable[[str], None]] = None, **kwargs: Any) -> Dict[str, Any]:
    kwargs["stream"] = True
    stream = client.chat.completions.create(**kwargs)
    parts: list = []
    usage_info: Optional[Dict[str, Any]] = None
    for chunk in stream:
        if not chunk or not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        chunk_text = ""
        delta = getattr(choice, "delta", None)
        if delta is not None:
            chunk_text = _flatten_chat_content(getattr(delta, "content", None))
        if not chunk_text:
            message = getattr(choice, "message", None)
            if message is not None:
                chunk_text = _flatten_chat_content(getattr(message, "content", None))
        if chunk_text:
            parts.append(chunk_text)
            if on_token:
                on_token(chunk_text)
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage:
            usage_info = _usage_from_source(provider, chunk_usage)
    content = "".join(parts)
    if usage_info is None:
        usage_info = {"provider": provider}
    return {"content": content, "usage": usage_info}

def call_llm_gemini(
    prompt: str,
    model_name: Optional[str] = None,
    thinking_budget: int = 0,
    api_key: Optional[str] = None,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    model_name = model_name or os.getenv("GEMINI_MODEL") or LIST_OF_GEMINI_MODELS[1]
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        ),
    )
    content = response.text
    if stream and on_token and content:
        on_token(content)
    usage_metadata = getattr(response, "usage_metadata", None)
    usage_info = {
        "provider": "gemini",
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0),
        "total_tokens": getattr(usage_metadata, "total_token_count", 0),
        "thinking_tokens": getattr(usage_metadata, "thoughts_token_count", 0),
    }
    return {"content": content, "usage": usage_info}

def call_llm_openai(
    prompt: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    thinking: Optional[str] = None,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    host = urlparse(base_url).hostname or ""
    is_openrouter = "openrouter.ai" in host
    is_openai = "api.openai.com" in host
    is_zhipu = host.endswith("z.ai") or host.endswith("bigmodel.cn")
    if is_openai:
        api_key_local = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_local:
            raise ValueError("Missing API key. Set OPENAI_API_KEY.")
        client = OpenAI(api_key=api_key_local, base_url=base_url)
        effective_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
        supports_reasoning = _openai_model_supports_reasoning(effective_model)
        if stream:
            messages = [{"role": "user", "content": prompt}]
            extra_body: Dict[str, Any] = {}
            if thinking in {"low", "medium", "high"} and supports_reasoning:
                extra_body["reasoning"] = {"effort": thinking}
            kwargs: Dict[str, Any] = {"model": effective_model, "messages": messages}
            if extra_body:
                kwargs["extra_body"] = extra_body
            return _stream_chat_completion(client, "openai", on_token=on_token, **kwargs)
        kwargs = {"model": effective_model, "input": prompt}
        if thinking in {"low", "medium", "high"} and supports_reasoning:
            kwargs["reasoning"] = {"effort": thinking}
        resp = client.responses.create(**kwargs)
        content = getattr(resp, "output_text", None) or resp.output[0].content[0].text
        usage_info = {
            "provider": "openai",
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        if hasattr(resp.usage, "output_tokens_details") and resp.usage.output_tokens_details:
            reasoning_tokens = getattr(resp.usage.output_tokens_details, "reasoning_tokens", 0)
            usage_info["reasoning_tokens"] = reasoning_tokens
        return {"content": content, "usage": usage_info}
    if is_zhipu:
        api_key_local = api_key or os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key_local:
            raise ValueError("Missing API key. Provide api_key or set ZHIPU_API_KEY/OPENAI_API_KEY.")
        client = OpenAI(api_key=api_key_local, base_url=base_url.rstrip("/"))
        effective_model = model_name or os.getenv("ZHIPU_MODEL", "glm-4.5-flash")
        messages = [{"role": "user", "content": prompt}]
        if stream:
            kwargs = {"model": effective_model, "messages": messages}
            return _stream_chat_completion(client, "zhipu", on_token=on_token, **kwargs)
        resp = client.chat.completions.create(model=effective_model, messages=messages)
        content_raw = getattr(resp.choices[0].message, "content", "")
        content = _flatten_chat_content(content_raw)
        usage_info = _usage_from_source("zhipu", getattr(resp, "usage", None))
        return {"content": content, "usage": usage_info}
    if is_openrouter:
        api_key_local = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key_local:
            raise ValueError("Missing API key. Set OPENROUTER_API_KEY.")
        client = OpenAI(api_key=api_key_local, base_url=base_url)
        extra_body: Dict[str, Any] = {}
        if isinstance(thinking, bool):
            extra_body = {"chat_template_kwargs": {"thinking": thinking}}
        messages = [{"role": "user", "content": prompt}]
        if stream:
            kwargs = {
                "model": model_name or os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
                "messages": messages,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            return _stream_chat_completion(client, "openrouter", on_token=on_token, **kwargs)
        resp = client.chat.completions.create(
            model=model_name or os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
            messages=messages,
            extra_body=extra_body or None,
        )
        content = resp.choices[0].message.content
        usage_info = {
            "provider": "openrouter",
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return {"content": content, "usage": usage_info}
    raise ValueError(f"Unrecognized base_url host: {host!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, world!", type=str)
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--api_key", default=None, type=str)
    parser.add_argument("--thinking_budget", default=True, type=bool)
    parser.add_argument("--base_url", default=None, type=str)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    if args.provider == "gemini":
        result = call_llm_gemini(
            args.prompt,
            model_name=args.model_name,
            thinking_budget=int(args.thinking_budget),
            api_key=args.api_key,
            stream=args.stream,
        )
        print(f"Content: {result['content']}")
        print(f"Usage: {result['usage']}")
    else:
        result = call_llm_openai(
            args.prompt,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            thinking=args.thinking_budget,
            stream=args.stream,
        )
        print(f"Content: {result['content']}")
        print(f"Usage: {result['usage']}")
