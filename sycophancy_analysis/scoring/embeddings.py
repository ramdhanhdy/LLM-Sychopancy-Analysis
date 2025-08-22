# scoring/embeddings.py
"""Embedding providers and semantic similarity functions."""

import os
import time
import random
import requests
import numpy as np
from typing import List, Tuple, Optional
from functools import lru_cache
from ..api import SCORING_CONFIG

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


class MistralEmbedder:
    """Thin client for Mistral embeddings API with an interface similar to SentenceTransformer."""

    def __init__(self, api_key: str, model: str, base: str, timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base = base.rstrip("/")
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def encode(self, texts: List[str], normalize_embeddings: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        url = f"{self.base}/embeddings"
        payload = {"model": self.model, "input": texts}
        dim_fallback = 1024  # Mistral-embed dimension
        max_retries = 3
        backoff_base = 0.75
        last_exc = None
        for attempt in range(max_retries):
            try:
                r = self.session.post(
                    url, json=payload, headers=self.headers, timeout=self.timeout
                )
                # Treat 5xx/429 as retryable
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"{r.status_code}: {r.text}")
                r.raise_for_status()
                js = r.json()
                data = js.get("data", [])
                # Ensure order by index
                data = sorted(data, key=lambda x: x.get("index", 0))
                embs = [np.array(item.get("embedding", []), dtype=np.float32) for item in data]
                if not embs or embs[0].size == 0:
                    embs = [np.zeros((dim_fallback,), dtype=np.float32) for _ in texts]
                break
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    sleep_s = backoff_base * (2 ** attempt) + random.uniform(0, 0.25)
                    time.sleep(sleep_s)
                    continue
                # Out of retries
                embs = None
        if embs is None:
            # Fallback to local ST if available
            if 'SentenceTransformer' in globals() and SentenceTransformer is not None:
                try:
                    model_name = SCORING_CONFIG.get(
                        "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st = SentenceTransformer(model_name)
                    arr = st.encode(texts, normalize_embeddings=normalize_embeddings)
                    return np.array(arr, dtype=np.float32)
                except Exception:
                    pass
            # Return zeros as last resort
            embs = [np.zeros((dim_fallback,), dtype=np.float32) for _ in texts]
        arr = np.vstack(embs)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


@lru_cache(maxsize=1)
def _get_embedder():
    """Get embedder instance based on config."""
    if not SCORING_CONFIG.get("USE_EMBEDDINGS", False):
        return None
    provider = SCORING_CONFIG.get("EMBEDDINGS_PROVIDER", "sentence_transformers")
    if provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            return None
        model = SCORING_CONFIG.get("MISTRAL_EMBED_MODEL", "mistral-embed")
        base = SCORING_CONFIG.get("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
        timeout = int(SCORING_CONFIG.get("MISTRAL_TIMEOUT", 30))
        try:
            return MistralEmbedder(api_key=api_key, model=model, base=base, timeout=timeout)
        except Exception:
            return None
    # Default to sentence-transformers
    if not _HAS_ST:
        return None
    try:
        model_name = SCORING_CONFIG.get(
            "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return SentenceTransformer(model_name)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _concept_embeds():
    """Get concept embeddings for agreement/disagreement."""
    from .constants import AGREE_CONCEPTS, DISAGREE_CONCEPTS
    
    embedder = _get_embedder()
    if not embedder:
        return None, None
    agree_emb = embedder.encode(AGREE_CONCEPTS, normalize_embeddings=True)
    disagree_emb = embedder.encode(DISAGREE_CONCEPTS, normalize_embeddings=True)
    return agree_emb, disagree_emb


@lru_cache(maxsize=1)
def _evasion_embeds():
    """Get evasion concept embeddings."""
    from .constants import EVASION_CONCEPTS
    
    embedder = _get_embedder()
    if not embedder:
        return None
    try:
        return embedder.encode(EVASION_CONCEPTS, normalize_embeddings=True)
    except Exception:
        return None


def _semantic_evasion(text: str) -> float:
    """Return max cosine similarity to EVASION concepts, or 0.0 on failure."""
    embedder = _get_embedder()
    if not embedder:
        return 0.0
    ev_emb = _evasion_embeds()
    if ev_emb is None:
        return 0.0
    try:
        q = embedder.encode([text], normalize_embeddings=True)[0]
    except Exception:
        return 0.0
    # Dimensionality guard (handles provider changes/fallbacks mid-run)
    try:
        ev_dim = ev_emb.shape[1]
        q_dim = q.shape[0]
    except Exception:
        return 0.0
    if ev_dim != q_dim:
        try:
            _evasion_embeds.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        ev_emb = _evasion_embeds()
        if ev_emb is None:
            return 0.0
        if ev_emb.shape[1] != q.shape[0]:
            return 0.0
    # Cosine sims with pre-normalized vectors => dot product
    try:
        sim_ev = float(np.max(np.dot(ev_emb, q)))
    except Exception:
        return 0.0
    return sim_ev


def _semantic_signals(text: str) -> Tuple[float, float]:
    """Get semantic similarity scores for agreement/disagreement."""
    embedder = _get_embedder()
    if not embedder:
        return 0.0, 0.0
    agree_emb, disagree_emb = _concept_embeds()
    if agree_emb is None or disagree_emb is None:
        return 0.0, 0.0
    q = embedder.encode([text], normalize_embeddings=True)[0]
    # Ensure embedding dimensions match (avoid mixing Mistral 1024 with ST 384 due to fallbacks)
    try:
        agree_dim = agree_emb.shape[1]
        disagree_dim = disagree_emb.shape[1]
        q_dim = q.shape[0]
    except Exception:
        return 0.0, 0.0
    if not (agree_dim == disagree_dim == q_dim):
        # Recompute concept embeddings with the current embedder (handles provider changes/fallbacks)
        try:
            _concept_embeds.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        agree_emb, disagree_emb = _concept_embeds()
        if agree_emb is None or disagree_emb is None:
            return 0.0, 0.0
        if agree_emb.shape[1] != q.shape[0] or disagree_emb.shape[1] != q.shape[0]:
            # Still mismatched; skip semantic boost rather than erroring
            return 0.0, 0.0
    # Cosine sims with pre-normalized vectors => dot product
    sim_agree = float(np.max(np.dot(agree_emb, q)))
    sim_disagree = float(np.max(np.dot(disagree_emb, q)))
    return sim_agree, sim_disagree
