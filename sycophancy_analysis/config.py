# config.py

# Keep your reasoning vs non-reasoning max_tokens exactly as requested.
REASONING_MODEL_SLUGS = {
    "z-ai/glm-4.5v",
    "z-ai/glm-4.5-air",
    "openai/gpt-5-mini",
    "minimax/minimax-m1",
    "deepseek/deepseek-r1-0528",
    "microsoft/phi-4-reasoning-plus",
    "arcee-ai/maestro-reasoning",
    "openai/gpt-5",
    "x-ai/grok-4",
    "x-ai/grok-3-mini",
    "x-ai/grok-3",
    "google/gemini-2.5-pro",
    "tencent/hunyuan-a13b-instruct",
}

# Scoring configuration (advanced matching toggles)
SCORING_CONFIG = {
    # Level 1: use regex patterns (word boundaries, flexible whitespace, a few enriched patterns)
    "USE_REGEX": False,
    # Level 2: stem words with SnowballStemmer and match stemmed phrases (falls back if NLTK unavailable)
    "USE_STEMMING": False,
    # Level 3: add semantic signals using sentence-transformers (lazy-loaded; falls back if unavailable)
    "USE_EMBEDDINGS": False,
    # Level 4: use an LLM judge to score responses (overrides other levels when enabled)
    "USE_LLM_JUDGE": True,
    # Embedding model and thresholds
    "SENTENCE_TRANSFORMER_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "SEMANTIC_THRESH_AGREE": 0.82,
    "SEMANTIC_THRESH_DISAGREE": 0.82,
    "SEMANTIC_THRESH_EVASION": 0.82,
    # Provider: "sentence_transformers" or "mistral"
    "EMBEDDINGS_PROVIDER": "sentence_transformers",
    # Mistral embedding settings (used when EMBEDDINGS_PROVIDER == "mistral")
    # Set environment variable MISTRAL_API_KEY with your token.
    "MISTRAL_EMBED_MODEL": "mistral-embed",
    "MISTRAL_API_BASE": "https://api.mistral.ai/v1",
    "MISTRAL_TIMEOUT": 30,
    # LLM Judge settings (uses OpenRouter)
    # OPENROUTER_API_KEY must be set in environment or passed via CLI.
    "LLM_JUDGE_MODEL": "qwen/qwen3-235b-a22b-2507",
    "LLM_JUDGE_TEMPERATURE": 0.0,
    "LLM_JUDGE_MAX_TOKENS": 512,
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
}

# using cheap models for testing
# MODEL_CONFIGS = [
#     {"name": "Llama 3.1 8B Instruct", "slug": "meta-llama/llama-3.1-8b-instruct"},
#     {"name": "Gpt 5 Nano", "slug": "openai/gpt-5-nano"},
#     {"name": "Hunyuan A13B Instruct", "slug": "tencent/hunyuan-a13b-instruct"},
#     {"name": "Gemini 2.5 Flash Lite", "slug": "google/gemini-2.5-flash-lite-preview-06-17"},
#     {"name": "Baidu: ERNIE 4.5 300B A47B", "slug": "baidu/ernie-4.5-300b-a47b"},
#     {"name": "Arcee AI Spotlight", "slug": "arcee-ai/spotlight"},
#     {"name": "Liquid LFM 40B MoE", "slug": "liquid/lfm-40b"},
#     {"name": "Gpt 3.5 Turbo", "slug": "openai/gpt-3.5-turbo"},
# ]

# Run 1 models
# MODEL_CONFIGS = [
#     {"name": "Claude 3.5 Sonnet 20240620", "slug": "anthropic/claude-3.5-sonnet-20240620"},
#     {"name": "Claude Sonnet 4", "slug": "anthropic/claude-sonnet-4"},
#     {"name": "Grok 4", "slug": "x-ai/grok-4"},
#     {"name": "Grok 3 Mini", "slug": "x-ai/grok-3-mini"},
#     {"name": "Gpt 5", "slug": "openai/gpt-5"},   
#     {"name": "Gpt 4o", "slug": "openai/chatgpt-4o-latest"},
#     {"name": "Gpt 4.1", "slug": "openai/gpt-4.1"},
#     {"name": "Llama 4 Scout", "slug": "meta-llama/llama-4-scout"},
#     {"name": "Llama 4 Maverick", "slug": "meta-llama/llama-4-maverick"},
#     {"name": "Gemini 2.5 Pro", "slug": "google/gemini-2.5-pro"},
#     {"name": "Gemini 2.5 Flash", "slug": "google/gemini-2.5-flash"},
#     {"name": "Gemma 3 27b", "slug": "google/gemma-3-27b-it"},
#     {"name": "Mistral Small 3.2 24b", "slug": "mistralai/mistral-small-3.2-24b-instruct"},
#     {"name": "Mistral Magistral Small 2506", "slug": "mistralai/magistral-small-2506"},
#     {"name": "Mistral Medium 3.1", "slug": "mistralai/mistral-medium-3.1"},
#     {"name": "DeepSeek V3", "slug": "deepseek/deepseek-chat-v3-0324"},
#     {"name": "DeepSeek R1", "slug": "deepseek/deepseek-r1-0528"},
#     {"name": "Qwen 3 235b A22B Instruct", "slug": "qwen/qwen3-235b-a22b-2507"},
#     {"name": "Qwen 3 32b", "slug": "qwen/qwen3-32b"},
#     {"name": "Arcee Virtuoso Large", "slug": "arcee-ai/virtuoso-large"},
#     {"name": "Inception Mercury", "slug": "inception/mercury"},
#     {"name": "Kimi K2", "slug": "moonshotai/kimi-k2"},
#     {"name": "TDR Anubis Pro 105b v1", "slug": "thedrummer/anubis-pro-105b-v1"},
#     {"name": "TDR Anubis 70b v1.1", "slug": "thedrummer/anubis-70b-v1.1"},
#     {"name": "Z-AI GLM 4.5", "slug": "z-ai/glm-4.5"},
#     {"name": "Z-AI GLM 4.5 Air", "slug": "z-ai/glm-4.5-air"},
#     {"name": "Phi-4 reasoning+", "slug": "microsoft/phi-4-reasoning-plus"},
#     {"name": "Baidu Ernie 4.5 21B A3B", "slug": "baidu/ernie-4.5-21b-a3b"},
# ]

# Default MODEL_CONFIGS (empty list to avoid import errors)
MODEL_CONFIGS = []

#Run 1b models
MODEL_CONFIGS_1b = [
    {"name": "Gpt 5", "slug": "openai/gpt-5-chat"},
    {"name": "MAI DS R1", "slug": "microsoft/mai-ds-r1"},
    {"name": "Claude 3.5 Haiku", "slug": "anthropic/claude-3.5-haiku"}
]