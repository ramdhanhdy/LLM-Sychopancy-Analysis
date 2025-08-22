# api/openrouter_client.py
"""OpenRouter API client for LLM interactions and model capability detection."""

import json
import time
import os
import requests
from typing import Optional, Dict, Any, NamedTuple

# Cache for models API
_MODELS_CACHE = {"ts": 0.0, "by_slug": {}}

def get_models(api_key: str):
    """Fetch models list from OpenRouter. Returns JSON dict or None on error."""
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return None

def _load_models_index(api_key: str, ttl: float = 600.0):
    """Return mapping slug->set(supported_parameters), cached for ttl seconds."""
    now = time.time()
    if _MODELS_CACHE["by_slug"] and (now - _MODELS_CACHE["ts"] < ttl):
        return _MODELS_CACHE["by_slug"]
    data = get_models(api_key)
    by_slug = {}
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            try:
                params = set(item.get("supported_parameters") or [])
                for key in (item.get("id"), item.get("canonical_slug")):
                    if key:
                        by_slug[str(key).lower()] = params
            except Exception:
                continue
    _MODELS_CACHE["by_slug"] = by_slug
    _MODELS_CACHE["ts"] = now
    return by_slug

def model_supports_reasoning(model_slug: str, api_key: str) -> bool:
    """Return True if the model advertises 'reasoning' (or legacy 'include_reasoning')."""
    idx = _load_models_index(api_key)
    params = idx.get(model_slug.lower())
    return bool(params and ("reasoning" in params or "include_reasoning" in params))

def model_supports_structured_outputs(model_slug: str, api_key: str) -> bool:
    """Return True if the model advertises 'structured_outputs'."""
    idx = _load_models_index(api_key)
    params = idx.get(model_slug.lower())
    return bool(params and ("structured_outputs" in params))

class CallMetadata(NamedTuple):
    """Metadata captured from an API call."""
    latency_ms: float
    retry_count: int
    http_status: int
    provider: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    stop_reason: Optional[str]
    request_id: Optional[str]
    response_id: Optional[str]
    reasoning_applied: bool
    structured_output_applied: bool
    provider_forced: bool

def chat_one(
    *,
    model_slug: str,
    user_text: str,
    api_key: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    system: Optional[str] = None,
    retries: int = 3,
    retry_delay: float = 1.2,
    reasoning_effort: Optional[str] = None,  # e.g., "low" | "medium" | "high"
    response_format: Optional[Dict[str, Any]] = None,  # OpenRouter structured outputs schema
    provider: Optional[Dict[str, Any]] = None,  # OpenRouter provider routing preferences
    per_request_timeout: float = 45.0,
    total_timeout: float = 90.0,
    return_metadata: bool = False,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    data: Dict[str, Any] = {
        "model": model_slug,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    # Add reasoning config if requested; rely on OpenRouter to map to provider-specific behavior
    if reasoning_effort:
        effort = reasoning_effort.lower()
        data["reasoning"] = {"effort": effort}
    # Add structured outputs schema if provided
    if response_format:
        data["response_format"] = response_format
    # Add provider routing preferences if provided
    if provider:
        data["provider"] = provider
    # Determine if provider is forced: either provider.only is set, allow_fallbacks is False,
    # or env flag LLM_JUDGE_FORCE_PROVIDER is truthy.
    provider_forced = False
    if isinstance(provider, dict):
        only_list = provider.get("only") or []
        allow_fb = provider.get("allow_fallbacks")
        env_force = str(os.environ.get("LLM_JUDGE_FORCE_PROVIDER", "")).lower() in ("1", "true", "yes")
        provider_forced = bool(only_list) or (allow_fb is False) or env_force
    debug = str(os.environ.get("LLM_JUDGE_DEBUG", "")).lower() in ("1", "true", "yes")
    start = time.monotonic()
    for attempt in range(retries + 1):
        elapsed = time.monotonic() - start
        if elapsed >= total_timeout:
            raise RuntimeError(f"OpenRouter total timeout exceeded after {elapsed:.2f}s")
        if debug:
            print(
                f"[chat_one] attempt={attempt+1}/{retries+1} model={model_slug} temp={temperature} max_tokens={max_tokens} "
                f"has_provider={bool(provider)} forced_provider={provider_forced} has_reasoning={bool(reasoning_effort)} "
                f"has_schema={bool(response_format)} elapsed={elapsed:.2f}s"
            )
        try:
            r = requests.post(
                url, headers=headers, data=json.dumps(data), timeout=float(per_request_timeout)
            )
        except requests.exceptions.Timeout as e:
            if debug:
                print(f"[chat_one] request timeout after {per_request_timeout:.1f}s on attempt {attempt+1}: {e}")
            if attempt == retries:
                raise RuntimeError(f"OpenRouter request timeout after {per_request_timeout:.1f}s: {e}")
            # backoff respecting total timeout budget
            sleep_s = retry_delay * (1.5 ** attempt)
            remaining = total_timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise RuntimeError("OpenRouter total timeout after request timeout")
            time.sleep(max(0.0, min(sleep_s, remaining)))
            continue
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"[chat_one] request exception on attempt {attempt+1}: {e}")
            if attempt == retries:
                raise RuntimeError(f"OpenRouter request exception: {e}")
            sleep_s = retry_delay * (1.5 ** attempt)
            remaining = total_timeout - (time.monotonic() - start)
            time.sleep(max(0.0, min(sleep_s, remaining)))
            continue

        if r.status_code == 200:
            try:
                # When debugging, log the provider selected by OpenRouter (from headers if available)
                if debug:
                    prov_hdr = (
                        r.headers.get("x-openrouter-provider")
                        or r.headers.get("openrouter-provider")
                        or r.headers.get("x-provider")
                        or r.headers.get("provider")
                    )
                    if prov_hdr:
                        print(f"[chat_one] selected provider: {prov_hdr}")
                response_json = r.json()
                content = response_json["choices"][0]["message"]["content"]
                
                if return_metadata:
                    # Extract metadata from response
                    usage = response_json.get("usage", {})
                    choice = response_json["choices"][0]
                    
                    # Provider info from headers
                    provider_used = (
                        r.headers.get("x-openrouter-provider")
                        or r.headers.get("openrouter-provider")
                        or r.headers.get("x-provider")
                        or r.headers.get("provider")
                    )
                    
                    metadata = CallMetadata(
                        latency_ms=round((time.monotonic() - start) * 1000, 2),
                        retry_count=attempt,
                        http_status=r.status_code,
                        provider=provider_used,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        stop_reason=choice.get("finish_reason"),
                        request_id=r.headers.get("x-request-id"),
                        response_id=response_json.get("id"),
                        reasoning_applied="reasoning" in data,
                        structured_output_applied="response_format" in data,
                        provider_forced=provider_forced
                    )
                    return content, metadata
                
                return content
            except Exception as e:
                if attempt == retries:
                    raise RuntimeError(f"OpenRouter OK but invalid JSON shape: {e}")
                if debug:
                    print(f"[chat_one] invalid success payload, retrying: {e}")
                continue
        # Fallback: if provider/model rejects 'reasoning', remove and retry
        if r.status_code in (400, 422) and "reasoning" in data:
            try:
                _ = r.json()
            except Exception:
                pass
            data.pop("reasoning", None)
            if debug:
                print("[chat_one] removed reasoning and retrying due to 400/422")
            continue
        # Fallback: if provider/model rejects 'response_format', remove and retry
        if r.status_code in (400, 422) and "response_format" in data:
            try:
                _ = r.json()
            except Exception:
                pass
            data.pop("response_format", None)
            if debug:
                print("[chat_one] removed response_format and retrying due to 400/422")
            continue
        # Fallback: if provider preferences cause an error
        # If provider is forced, DO NOT drop provider; prefer trimming features (reasoning/response_format) and then raise.
        if r.status_code in (400, 409, 422, 429, 500, 502, 503, 504) and "provider" in data:
            try:
                _ = r.json()
            except Exception:
                pass
            if provider_forced:
                if debug:
                    print(
                        f"[chat_one] provider error {r.status_code} but provider is forced; keeping provider and not falling back"
                    )
                # Try to remove schema/reasoning if present and retry, otherwise raise on next loop end
                removed_any = False
                if "response_format" in data:
                    data.pop("response_format", None)
                    removed_any = True
                    if debug:
                        print("[chat_one] (forced) removed response_format due to provider error; retrying")
                if "reasoning" in data:
                    data.pop("reasoning", None)
                    removed_any = True
                    if debug:
                        print("[chat_one] (forced) removed reasoning due to provider error; retrying")
                if removed_any:
                    continue
                # Nothing else to remove; if this was final attempt, raise below; otherwise continue to backoff
            else:
                data.pop("provider", None)
                if debug:
                    print(f"[chat_one] removed provider due to status {r.status_code}; retrying without provider prefs")
                continue
        if attempt == retries:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
        # Respect total timeout budget on backoff
        sleep_s = retry_delay * (1.5 ** attempt)
        remaining = total_timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise RuntimeError(f"OpenRouter total timeout exceeded while backing off after status {r.status_code}")
        time.sleep(max(0.0, min(sleep_s, remaining)))
