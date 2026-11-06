"""OpenAI-compatible chat-completion client with JSON-schema validation."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from typing import Any
import urllib.request

import jsonschema


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=64)
def validator(serialized_schema: str):
    schema = json.loads(serialized_schema)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def validate_json(value, schema):
    validator(json.dumps(schema, sort_keys=True)).validate(value)


class ResponseError(ValueError):
    """Keep the real endpoint response when completion or schema validation fails."""
    def __init__(self, message: str, raw_response: dict):
        super().__init__(message)
        self.raw_response = raw_response


def grammar_schema(schema):
    """vLLM grammar cannot enforce uniqueItems; the full local schema still does.

    Only this explicitly documented grammar limitation is relaxed on the wire.
    The prompt and acceptance validator retain the complete original schema.
    """
    if isinstance(schema, dict):
        return {key: grammar_schema(value) for key, value in schema.items() if key != "uniqueItems"}
    if isinstance(schema, list):
        return [grammar_schema(value) for value in schema]
    return schema


def complete(messages_: list, schema: dict, *, model: str, base_url: str,
             api_key: str | None, max_tokens: int, seed: int, timeout: int,
             response_format: str, disable_thinking: bool = False,
             temperature: float = 0.0, top_p: float = 1.0, top_k: int = -1,
             reasoning_effort: str | None = None,
             presence_penalty: float = 0.0,
             allow_unescaped_control_chars: bool = False) -> tuple[dict, str, dict, dict]:
    # json_object is supported by DeepSeek-style endpoints; json_schema must be
    # explicitly selected for a server that implements it. Never silently retry
    # using a different model, prompt, schema, or evidence set.
    formatting: dict = {"type": response_format}
    if response_format == "json_schema":
        formatting["json_schema"] = {"name": "material_prediction", "schema": grammar_schema(schema), "strict": True}
    body = {"model": model, "messages": messages_, "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens, "seed": seed, "response_format": formatting}
    if top_k != -1:
        body["top_k"] = top_k
    if presence_penalty:
        body["presence_penalty"] = presence_penalty
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif reasoning_effort:
        body["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": reasoning_effort}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body, allow_nan=False).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    try:
        choice = result["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"Incomplete response: finish_reason={choice.get('finish_reason')}")
        raw = choice["message"]["content"]
        parsed = json.loads(raw, strict=not allow_unescaped_control_chars)
        validate_json(parsed, schema)
        digest(parsed)
    except (ValueError, TypeError, KeyError, IndexError, jsonschema.ValidationError) as exc:
        raise ResponseError(f"{type(exc).__name__}: {exc}", result) from exc
    return parsed, raw, result.get("usage", {}), result
