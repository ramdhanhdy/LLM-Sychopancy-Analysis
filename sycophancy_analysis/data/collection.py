# data/collection.py
"""Response collection functions for API calls."""

import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
from ..api import chat_one, model_supports_reasoning
from .persistence import save_responses


def collect_responses(
    model_configs: List[Dict],
    prompts_df: pd.DataFrame,
    api_key: str,
    samples_per_prompt: int = 1,
    base_temperature: float = 0.2,
    system_prompt: Optional[str] = None,
    save_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: model, prompt_id, response
    Preserves:
      - Reasoning models: reasoning effort="low"; omit max_tokens (use model/provider defaults)
      - Non-reasoning models: max_tokens=300
    """
    rows: List[Dict] = []
    # Global progress bar across all API calls
    total_calls = len(model_configs) * len(prompts_df) * samples_per_prompt
    with tqdm(total=total_calls, desc="Collecting responses", unit="call") as pbar:
        for cfg in model_configs:
            slug = cfg["slug"]
            name = cfg["name"]
            # Dynamic detection via OpenRouter Models API
            try:
                supports = model_supports_reasoning(slug, api_key)
            except Exception:
                supports = False
            if supports:
                # Use OpenRouter reasoning effort levels; rely on inherent provider defaults for tokens
                max_toks = None  # omit from payload
                reasoning_effort = "low"
                tqdm.write(f"[Reasoning] {name} -> effort={reasoning_effort}, tokens=<default>")
            else:
                max_toks = 300
                reasoning_effort = None
                tqdm.write(f"[Standard]  {name} -> max_tokens={max_toks}")
            for pr in prompts_df.itertuples(index=False):
                for _ in range(samples_per_prompt):
                    try:
                        result = chat_one(
                            model_slug=slug,
                            user_text=pr.text,
                            api_key=api_key,
                            max_tokens=max_toks,
                            temperature=base_temperature,
                            system=system_prompt,
                            reasoning_effort=reasoning_effort,
                            return_metadata=True,
                        )
                        if isinstance(result, tuple):
                            resp, metadata = result
                            # Add metadata fields to the row
                            row_data = {
                                "model": name,
                                "prompt_id": pr.prompt_id,
                                "response": resp,
                                "latency_ms": metadata.latency_ms,
                                "retry_count": metadata.retry_count,
                                "http_status": metadata.http_status,
                                "provider": metadata.provider,
                                "prompt_tokens": metadata.prompt_tokens,
                                "completion_tokens": metadata.completion_tokens,
                                "total_tokens": metadata.total_tokens,
                                "stop_reason": metadata.stop_reason,
                                "request_id": metadata.request_id,
                                "response_id": metadata.response_id,
                                "reasoning_applied": metadata.reasoning_applied,
                                "structured_output_applied": metadata.structured_output_applied,
                                "provider_forced": metadata.provider_forced,
                                "temperature": base_temperature,
                                "max_tokens": max_toks,
                                "model_slug": slug
                            }
                        else:
                            # Fallback for backward compatibility
                            resp = result
                            row_data = {"model": name, "prompt_id": pr.prompt_id, "response": resp}
                    except Exception as e:
                        tqdm.write(f"Error {name}/{pr.prompt_id}: {e}")
                        row_data = {
                            "model": name,
                            "prompt_id": pr.prompt_id,
                            "response": "",
                            "latency_ms": None,
                            "retry_count": None,
                            "http_status": None,
                            "provider": None,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "stop_reason": "error",
                            "request_id": None,
                            "response_id": None,
                            "reasoning_applied": reasoning_effort is not None,
                            "structured_output_applied": False,
                            "provider_forced": False,
                            "temperature": base_temperature,
                            "max_tokens": max_toks,
                            "model_slug": slug
                        }
                    rows.append(row_data)
                    pbar.update(1)
    
    new_responses_df = pd.DataFrame(rows)
    
    # Save the new responses
    if save_prefix:
        save_responses(save_prefix, new_responses_df)
        
    return new_responses_df
