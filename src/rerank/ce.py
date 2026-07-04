# src/rerank/ce.py
from typing import List, Dict, Any
import os
from sentence_transformers import CrossEncoder

_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
_model = None

def _get():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME, max_length=512, device=None)
    return _model

def rerank(query: str, passages: List[Dict[str, Any]], keep: int = 8, batch_size: int = 64) -> List[Dict[str, Any]]:
    if not passages:
        return []
    model = _get()
    pairs = [(query, (p.get("text") or "")[:1000]) for p in passages]
    scores = model.predict(pairs, batch_size=batch_size).tolist()
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

    out: List[Dict[str, Any]] = []
    for p, s in ranked[:keep]:
        q = dict(p)
        q["rerank_score"] = float(s)
        out.append(q)
    return out
