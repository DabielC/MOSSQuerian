"""Utility helpers for the Streamlit front-end to interact with MongoDB and the text-to-mongo API."""
from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError


class ServiceError(RuntimeError):
    """Base error type for the front-end services."""


class MongoConnectionError(ServiceError):
    """Raised when MongoDB cannot be reached."""


class APIRequestError(ServiceError):
    """Raised when the backend API request fails."""




@dataclass
class LLMConfig:
    provider: str
    model_name: str
    api_key: str
    base_url: Optional[str] = None


class LLMRequestError(ServiceError):
    """Raised when an LLM call fails."""

@dataclass
class MongoCredentials:
    uri: str
    db_name: str


def _serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    return value


def _serialize_documents(docs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_serialize_value(doc) for doc in docs]


def _build_mongo_client(uri: str) -> MongoClient:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except PyMongoError as exc:
        raise MongoConnectionError(str(exc)) from exc


def test_mongo_connection(credentials: MongoCredentials) -> Tuple[bool, str, List[str]]:
    client: Optional[MongoClient] = None
    try:
        client = _build_mongo_client(credentials.uri)
        db = client[credentials.db_name]
        collections = sorted(db.list_collection_names())
        return True, f"Connected to {credentials.db_name}", collections
    except MongoConnectionError as exc:
        return False, str(exc), []
    finally:
        if client:
            client.close()


def fetch_collection_names(credentials: MongoCredentials) -> List[str]:
    client: Optional[MongoClient] = None
    try:
        client = _build_mongo_client(credentials.uri)
        db = client[credentials.db_name]
        return sorted(db.list_collection_names())
    finally:
        if client:
            client.close()


def _collection(credentials: MongoCredentials, collection_name: str) -> Tuple[MongoClient, Collection]:
    client = _build_mongo_client(credentials.uri)
    db: Database = client[credentials.db_name]
    return client, db[collection_name]


def fetch_collection_sample(
    credentials: MongoCredentials,
    collection_name: str,
    *,
    limit: int = 25,
    skip: int = 0,
) -> Tuple[pd.DataFrame, int]:
    client: Optional[MongoClient] = None
    try:
        client, coll = _collection(credentials, collection_name)
        total = coll.estimated_document_count()
        cursor = coll.find({}, skip=skip, limit=limit)
        documents = list(cursor)
        serialized = _serialize_documents(documents)
        if serialized:
            frame = pd.json_normalize(serialized, sep=".")
        else:
            frame = pd.DataFrame()
        return frame, total
    finally:
        if client:
            client.close()


def infer_collection_schema(
    credentials: MongoCredentials,
    collection_name: str,
    sample_size: int = 200,
) -> pd.DataFrame:
    client: Optional[MongoClient] = None
    try:
        client, coll = _collection(credentials, collection_name)
        cursor = coll.find({}, limit=sample_size)
        docs = list(cursor)
        serialized = _serialize_documents(docs)
        if not serialized:
            return pd.DataFrame(columns=["field", "types", "examples"])
        flattened = pd.json_normalize(serialized, sep=".")
        field_info: Dict[str, Dict[str, Any]] = {}
        for _, row in flattened.iterrows():
            for field, value in row.items():
                entry = field_info.setdefault(field, {"types": set(), "examples": []})
                entry["types"].add(type(value).__name__ if value is not None else "NoneType")
                if value is not None and len(entry["examples"]) < 3:
                    entry["examples"].append(str(value))
        records = [
            {
                "field": field,
                "types": ", ".join(sorted(data["types"])),
                "examples": "; ".join(data["examples"]),
            }
            for field, data in sorted(field_info.items())
        ]
        return pd.DataFrame(records)
    finally:
        if client:
            client.close()




def _iter_sse_payloads(response: requests.Response) -> Iterable[str]:
    """Yield decoded SSE data payloads from a streaming response."""
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line.strip() == "":
            yield "__EVENT_BREAK__"
            continue
        yield line


def _parse_sse_events(response: requests.Response) -> Iterable[Dict[str, Any]]:
    buffer: list[str] = []
    for line in _iter_sse_payloads(response):
        if line == "__EVENT_BREAK__":
            if not buffer:
                continue
            data_lines: list[str] = []
            event_fields: Dict[str, Any] = {}
            for entry in buffer:
                if entry.startswith(":"):
                    continue
                if entry.startswith("data:"):
                    data_lines.append(entry[5:].lstrip())
                elif entry.startswith("event:"):
                    event_fields["event"] = entry[6:].strip()
                elif entry.startswith("id:"):
                    event_fields["id"] = entry[3:].strip()
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        event_fields.update(payload)
                    else:
                        event_fields.setdefault("data", payload)
                except json.JSONDecodeError:
                    event_fields.setdefault("data", raw)
            if event_fields:
                yield event_fields
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        data_lines: list[str] = []
        event_fields: Dict[str, Any] = {}
        for entry in buffer:
            if entry.startswith(":"):
                continue
            if entry.startswith("data:"):
                data_lines.append(entry[5:].lstrip())
            elif entry.startswith("event:"):
                event_fields["event"] = entry[6:].strip()
            elif entry.startswith("id:"):
                event_fields["id"] = entry[3:].strip()
        if data_lines:
            raw = "\n".join(data_lines)
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    event_fields.update(payload)
                else:
                    event_fields.setdefault("data", payload)
            except json.JSONDecodeError:
                event_fields.setdefault("data", raw)
        if event_fields:
            yield event_fields

def call_execute_api(
    base_url: str,
    payload: Dict[str, Any],
    *,
    timeout: int = 120,
    stream: bool = False,
    event_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    url = f"{base_url.rstrip('/')}/v1/execute"
    request_payload = dict(payload)
    try:
        if stream:
            request_payload.setdefault("streaming", True)
            response = requests.post(
                url,
                json=request_payload,
                timeout=timeout,
                stream=True,
                headers={"Accept": "text/event-stream"},
            )
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "text/event-stream" not in content_type:
                data = response.json()
                return data, []
            events: List[Dict[str, Any]] = []
            final_payload: Optional[Dict[str, Any]] = None
            for event in _parse_sse_events(response):
                events.append(event)
                if event_handler:
                    event_handler(event)
                if event.get("type") == "final":
                    data = event.get("data")
                    if isinstance(data, dict):
                        final_payload = data
                    else:
                        final_payload = {"data": data}
            response.close()
            if final_payload is None:
                raise APIRequestError("Stream ended without a final payload")
            return final_payload, events
        else:
            request_payload.setdefault("streaming", False)
            response = requests.post(url, json=request_payload, timeout=timeout)
            response.raise_for_status()
            return response.json(), []
    except requests.RequestException as exc:
        raise APIRequestError(str(exc)) from exc


def call_healthcheck(base_url: str, *, timeout: int = 10) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIRequestError(str(exc)) from exc


def call_llm_chat(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = 120,
) -> str:
    """Call the configured LLM and return its text response."""
    if not config.api_key:
        raise LLMRequestError("Missing API key for LLM call")

    provider = (config.provider or "openai").lower()

    try:
        if provider == "gemini":
            try:
                from google import genai
                from google.genai import types as genai_types
            except ImportError as exc:
                raise LLMRequestError("google-genai package is required for Gemini calls") from exc

            client = genai.Client(api_key=config.api_key)
            combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
            contents = [
                genai_types.Content(role="user", parts=[genai_types.Part(text=combined_prompt)])
            ]
            generation_config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
            try:
                response = client.models.generate_content(
                    model=config.model_name,
                    contents=contents,
                    config=generation_config,
                )
            except Exception as exc:
                raise LLMRequestError(str(exc)) from exc
            text = getattr(response, "text", "")
            if not text and getattr(response, "candidates", None):
                parts = []
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    if content and getattr(content, "parts", None):
                        for part in content.parts:
                            value = getattr(part, "text", None)
                            if value:
                                parts.append(value)
                if parts:
                    text = "".join(parts)
        else:
            if provider == "openrouter" and not config.base_url:
                base = "https://openrouter.ai/api/v1"
            elif provider.startswith("openai") and not config.base_url:
                base = "https://api.openai.com/v1"
            else:
                base = config.base_url or "https://api.openai.com/v1"
            endpoint = f"{base.rstrip('/')}/chat/completions"
            payload = {
                "model": config.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise LLMRequestError("LLM response had no choices")
            text = choices[0].get("message", {}).get("content")
    except requests.RequestException as exc:
        raise LLMRequestError(str(exc)) from exc

    if not text:
        raise LLMRequestError("LLM returned an empty response")

    return text.strip()
