# src/serve/api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from time import perf_counter
from typing import Dict, Any

from ..retrieve import hybrid as retr
from ..rerank.ce import rerank
from ..answer.generate import answer

app = FastAPI(title="RAG (TH)")

class QueryIn(BaseModel):
    q: str = Field(..., min_length=1)
    use_reranker: bool = True
    use_bm25: bool = True
    k: int = Field(10, ge=1, le=100)

@app.get("/health")
def health():
    return {"status": "ok"}

def _pipeline(inp: QueryIn) -> Dict[str, Any]:
    t0 = perf_counter()

    # toggle BM25 inside retrieval module
    retr.RETR_USE_BM25 = bool(inp.use_bm25)

    # retrieve with timings
    ret = retr.retrieve_with_timings(inp.q, k_bm25=40, k_dense=40, mmr_k=inp.k, mmr_lambda=0.5)
    passages = ret["results"]
    retrieve_ms = ret["timings"]["total_ms"]

    # rerank
    if inp.use_reranker and passages:
        t1 = perf_counter()
        passages = rerank(inp.q, passages, keep=min(8, inp.k))
        rerank_ms = (perf_counter() - t1) * 1000.0
    else:
        rerank_ms = 0.0

    # generate
    t2 = perf_counter()
    out = answer(inp.q, passages)
    llm_ms = (perf_counter() - t2) * 1000.0

    out["timings"] = {
        "retrieve_ms": round(retrieve_ms, 1),
        "rerank_ms": round(rerank_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "total_ms": round((perf_counter() - t0) * 1000.0, 1),
    }
    out["retrieve_breakdown"] = ret["timings"]
    return out

@app.post("/query")
def query(inp: QueryIn):
    return _pipeline(inp)

@app.post("/answer")
def answer_alias(inp: QueryIn):
    return _pipeline(inp)

@app.on_event("startup")
def _startup():
    try:
        retr.warmup()
    except Exception as e:
        print(f"[WARN] warmup failed: {e}")
