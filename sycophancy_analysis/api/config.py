# api/config.py
"""Configuration settings for OpenRouter API and model management."""

import os

# Scoring configuration (matching toggles)
SCORING_CONFIG = {
    # Level 1: use regex patterns (word boundaries, flexible whitespace, a few enriched patterns)
    "USE_REGEX": False,
    # Level 2: stem words with SnowballStemmer and match stemmed phrases (falls back if NLTK unavailable)
    "USE_STEMMING": False,
    # Level 3: use an LLM judge to score responses (overrides other levels when enabled)
    "USE_LLM_JUDGE": True,
    # LLM Judge settings (uses OpenRouter)
    # OPENROUTER_API_KEY must be set in environment or passed via CLI.
    "LLM_JUDGE_MODEL": "qwen/qwen3-235b-a22b-2507",
    "LLM_JUDGE_TEMPERATURE": 0.0,
    "LLM_JUDGE_MAX_TOKENS": 800,
    # Provider routing preferences for LLM judge (OpenRouter).
    # Force Cerebras by default; you can override via env or editing this config.
    # See: https://openrouter.ai/docs/features/provider-routing
    "LLM_JUDGE_PROVIDER_PREFS": {
        "only": ["cerebras"],
        "order": ["cerebras"],
        "allow_fallbacks": False,
        # Set require_parameters if you want to error instead of silently ignoring unsupported params
        # "require_parameters": True,
    },
    # LLM judge only: optional per-model or per-tag overrides for judge behavior
    # Params: adjust judge runtime (temperature, max_tokens) based on the model being judged
    "LLM_JUDGE_PARAMS_OVERRIDES": {
        "default": {},            # e.g., {"temperature": 0.0, "max_tokens": 768}
        "by_tag": {},             # e.g., {"reasoning": {"max_tokens": 1024}}
        "by_model_slug": {},      # e.g., {"openai/gpt-4.1": {"temperature": 0.0}}
    },
    # Scoring: optional multipliers per returned score field from the judge JSON
    # Keys correspond to PromptScores fields: endorse_stance, challenge, harm_validation, devil_advocate,
    # flattery_density, intens_density, hedge_density, refusal_markers, safe_alt_markers, evasion_markers, caveat_in_open
    "LLM_JUDGE_SCORING_OVERRIDES": {
        "default": {},            # e.g., {"harm_validation": 1.1}
        "by_tag": {},             # e.g., {"reasoning": {"endorse_stance": 0.95}}
        "by_model_slug": {},      # e.g., {"qwen/qwen3-235b-a22b-2507": {"flattery_density": 1.2}}
    },
}

# Model configurations for analysis
MODEL_CONFIGS = [
    {"name": "Gpt 5", "slug": "openai/gpt-5-chat"},
    {"name": "Gpt 4.1", "slug": "openai/gpt-4.1"},
    {"name": "Gpt 4o Mini", "slug": "openai/gpt-4o-mini"},
    {"name": "Gpt 4.1 mini", "slug": "openai/gpt-4.1-mini"},
    {"name": "Gpt 5 mini", "slug": "openai/gpt-5-mini"},
    {"name": "MAI DS R1", "slug": "microsoft/mai-ds-r1"},
    {"name": "Claude 3.5 Haiku", "slug": "anthropic/claude-3.5-haiku"},
    {"name": "Gemini 2.0 flash", "slug": "google/gemini-2.0-flash-001"},
    {"name": "Llama 3.3 70b", "slug": "meta-llama/llama-3.3-70b-instruct"},
    {"name": "Gemma 3 12b", "slug": "google/gemma-3-12b-it"},
]

# Output format configuration
OUTPUT_FORMAT = {
    "responses": "csv",     # "csv" or "json"
    "sss": "csv",          # "csv" or "json" 
    "si_table": "csv",     # "csv" or "json"
}

# Helper: resolve LLM judge override maps by precedence default < by_tag < by_model_slug
def resolve_judge_overrides(model_slug: str | None, key: str, tags: list[str] | None = None) -> dict:
    conf = SCORING_CONFIG.get(key, {}) if isinstance(SCORING_CONFIG.get(key, {}), dict) else {}
    eff: dict = {}
    try:
        # default
        eff.update(conf.get("default", {}) or {})
        # tags
        for t in (tags or []):
            eff.update((conf.get("by_tag", {}) or {}).get(t, {}) or {})
        # specific model
        if model_slug:
            eff.update((conf.get("by_model_slug", {}) or {}).get(model_slug, {}) or {})
    except Exception:
        # Be robust to malformed configs
        return eff
    return eff

# ----------------------------------------------------------------------------
# MODEL_CONFIGS enrichment utilities

def _derive_family(slug: str) -> str | None:
    try:
        part = slug.split("/", 1)[1]
        return part.split("-")[0]
    except Exception:
        return None


def _derive_size_label(slug: str) -> str | None:
    # Heuristic extraction like "120b", "32b", etc.
    import re as _re
    m = _re.search(r"(\d+\s*[bBkKmM])", slug)
    return m.group(1).upper().replace(" ", "") if m else None


def enrich_model_configs(configs: list[dict]) -> list[dict]:
    """Return configs enriched with provider/family/tags and operational metadata.

    - Adds keys: provider, family, size_label, reasoning (bool)
    - Adds stubs: supports, limits, cost, defaults, analysis
    - Preserves original keys (name, slug)
    """
    enriched: list[dict] = []
    for c in configs:
        slug = str(c.get("slug", ""))
        name = c.get("name") or slug
        provider = slug.split("/", 1)[0] if "/" in slug else None
        family = _derive_family(slug)
        size_label = _derive_size_label(slug)
        # Do not infer reasoning from a static list; determined at runtime via OpenRouter
        reasoning = None  # Unknown until annotated via annotate_reasoning_support()
        base = {
            "provider": provider,
            "family": family,
            "size_label": size_label,
            "reasoning": reasoning,
            "modalities": ["text"],
            "supports": {"json_mode": True, "function_calling": True, "streaming": True, "vision": False},
            "limits": {
                "context_window": None,
                "max_output_tokens": None,
                "timeout_s": 60,
                "rpm": None,
                "tpm": None,
                "concurrency": None,
            },
            "cost": {"input_per_1k": None, "output_per_1k": None, "currency": "USD", "est_tps": None},
            "defaults": {"temperature": 0.2, "top_p": 1.0, "system_prompt": None, "max_tokens_strategy": "fixed"},
            "analysis": {"label_short": name, "color": None, "tags": []},
            "version": None,
            "notes": None,
        }
        merged = {**c, **base}
        enriched.append(merged)
    return enriched

# Enrich in-place so downstream modules get richer metadata by default
MODEL_CONFIGS = enrich_model_configs(MODEL_CONFIGS)


def annotate_reasoning_support(configs: list[dict], api_key: str | None = None) -> list[dict]:
    """Annotate configs with reasoning support using OpenRouter model capabilities.

    - Reads OPENROUTER_API_KEY from environment if api_key is None.
    - Sets config["reasoning"] = True/False and adds/removes "reasoning" tag in analysis.tags.
    - Returns a new list; does not mutate the input list.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return [dict(c) for c in configs]
    try:
        # Local import avoids import-time cycles
        from .openrouter_client import model_supports_reasoning
    except Exception:
        return [dict(c) for c in configs]

    out: list[dict] = []
    for c in configs:
        cc = dict(c)
        slug = str(cc.get("slug", ""))
        try:
            supports = model_supports_reasoning(slug, key)
        except Exception:
            supports = False
        cc["reasoning"] = bool(supports)
        # update tags
        analysis = dict(cc.get("analysis", {}))
        tags = list(analysis.get("tags", []) or [])
        if supports and "reasoning" not in tags:
            tags.append("reasoning")
        if not supports and "reasoning" in tags:
            tags = [t for t in tags if t != "reasoning"]
        analysis["tags"] = tags
        cc["analysis"] = analysis
        out.append(cc)
    return out


def annotate_reasoning_support_inplace(api_key: str | None = None) -> None:
    """Annotate the global MODEL_CONFIGS in place using OpenRouter model capabilities."""
    global MODEL_CONFIGS
    MODEL_CONFIGS = annotate_reasoning_support(MODEL_CONFIGS, api_key)
