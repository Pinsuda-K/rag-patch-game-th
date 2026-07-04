# src/answer/generate.py
from typing import List, Dict, Any, Optional
import requests, os

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct-q4_K_M")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM = (
    "You are a Thai/English assistant. Answer ONLY from the provided snippets.\n"
    "If information is insufficient, say you don't know. "
    "Cite using the provided [chunk_id] markers. "
    "Keep numbers/dates verbatim. "
    "Do not give medical/financial advice."
)

def _build_prompt(q: str, snippets: List[Dict[str, Any]]) -> str:
    ctx = []
    for i, s in enumerate(snippets, start=1):
        cid = s.get("id") or s.get("source_id") or f"ctx{i}"
        title = s.get("title", "")
        text = (s.get("text") or "")[:1200]
        url = s.get("url", "")
        header = f"[{cid}] {title}".strip()
        if url:
            header += f" ({url})"
        ctx.append(f"{header}\n{text}")
    ctx_str = "\n\n".join(ctx)
    return f"{_SYSTEM}\n\nQuestion: {q}\n\nSnippets:\n{ctx_str}\n\nAnswer in Thai if the question is Thai."

def _ollama_generate(prompt: str, timeout: int = 30) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        return f"(degraded) model error: {e}"

def _openai_generate(prompt: str, timeout: int = 30) -> Optional[str]:
    if not OPENAI_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}"},
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "system", "content": _SYSTEM},
                             {"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def answer(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    contexts: List of dicts with at least {id, text}. Optional: {title, url}.
    Returns: {"text": str, "citations": [{id, title, url}]}
    """
    prompt = _build_prompt(query, contexts)
    txt = _ollama_generate(prompt)
    if txt.startswith("(degraded)") and OPENAI_KEY:
        alt = _openai_generate(prompt)
        if alt:
            txt = alt

    citations = [{"id": c.get("id") or c.get("source_id") or "", 
                  "title": c.get("title", ""), 
                  "url": c.get("url", "")} for c in contexts]
    return {"text": txt, "citations": citations}
