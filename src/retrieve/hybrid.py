# src/retrieve/hybrid.py
import json, re, os, sys, time
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from functools import lru_cache

BM25_PATH = os.getenv("BM25_INDEX_PATH", "data/bm25_idx.json")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3")
RETR_USE_BM25 = os.getenv("RETR_USE_BM25", "1") not in {"0", "false", "False", ""}

# ---------- helpers ----------
_TOKEN_RE = re.compile(r"[\wก-๙]+", re.UNICODE)
def tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall((s or "").lower())

def _sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _mmr(query_vec: List[float], cand_vecs: List[List[float]], lamb: float = 0.5, topk: int = 12) -> List[int]:
    chosen, remain = [], list(range(len(cand_vecs)))
    while remain and len(chosen) < topk:
        best, best_score = None, -1e9
        for i in remain:
            rel = _sim(query_vec, cand_vecs[i])
            div = 0.0 if not chosen else max(_sim(cand_vecs[i], cand_vecs[j]) for j in chosen)
            score = lamb * rel - (1 - lamb) * div
            if score > best_score:
                best_score, best = score, i
        chosen.append(best); remain.remove(best)
    return chosen

def _prefix_query(model_name: str, q: str) -> str:
    return ("query: " + q) if "e5" in model_name.lower() else q

def _as_list(x):
    return x if isinstance(x, list) else (list(x) if x is not None else [])

def _as_pyfloat_list(v):
    # v may be list[float] or numpy.ndarray
    try:
        return v.tolist()  # numpy
    except AttributeError:
        return list(v) if not isinstance(v, list) else v

def _norm_emb_list(arrs):
    return [_as_pyfloat_list(v) for v in _as_list(arrs)]

def _first_list(x):
    x = _as_list(x)
    return x[0] if x else []

# ---------- caches ----------
@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDER_MODEL)

@lru_cache(maxsize=1)
def _get_bm25() -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    try:
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return BM25Okapi(obj["tokens"]), obj["metas"]
    except Exception as e:
        print(f"[WARN] BM25 unavailable ({e}); dense-only.", file=sys.stderr)
        return None, []

@lru_cache(maxsize=1)
def _get_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
    # tolerant: create if missing (prevents 500; but empty if you didn't build it)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return client, col

def warmup():
    t0 = time.perf_counter()
    enc = _get_encoder()
    _ = enc.encode(["warmup", "สวัสดี", "Yorn 14→12"], normalize_embeddings=True)
    _ = _get_bm25()
    try:
        _, col = _get_chroma()
        _ = col.count()
    except Exception as e:
        print(f"[WARN] Chroma warmup: {e}", file=sys.stderr)
    print(f"[warmup] model={EMBEDDER_MODEL} chroma_dir={CHROMA_DIR} collection={CHROMA_COLLECTION}  took={round((time.perf_counter()-t0)*1000)}ms")

def _fetch_by_ids(col, ids: List[str]) -> Tuple[List[Dict[str, Any]], List[List[float]], List[str]]:
    if not ids:
        return [], [], []
    got = col.get(ids=ids, include=["metadatas", "embeddings", "documents"])
    metas = _as_list(got.get("metadatas"))
    embs  = _norm_emb_list(got.get("embeddings"))
    docs  = _as_list(got.get("documents"))
    n = min(len(metas), len(embs), len(docs))
    return metas[:n], embs[:n], docs[:n]

# ---------- API ----------
def retrieve(query: str, k_bm25: int = 30, k_dense: int = 30, mmr_k: int = 10, mmr_lambda: float = 0.5) -> List[Dict[str, Any]]:
    return retrieve_with_timings(query, k_bm25, k_dense, mmr_k, mmr_lambda)["results"]

def retrieve_with_timings(query: str, k_bm25: int = 30, k_dense: int = 30, mmr_k: int = 10, mmr_lambda: float = 0.5) -> Dict[str, Any]:
    t0 = time.perf_counter()
    enc = _get_encoder()
    bm25, metas = _get_bm25()
    _, col = _get_chroma()

    # 1) encode query
    qv = enc.encode([_prefix_query(EMBEDDER_MODEL, query)], normalize_embeddings=True)[0]
    qv = _as_pyfloat_list(qv)
    t_q = time.perf_counter()

    # 2) dense
    d = col.query(query_embeddings=[qv], n_results=k_dense, include=["metadatas", "embeddings", "documents"])
    t_dense = time.perf_counter()
    meta_list = _first_list(d.get("metadatas"))
    emb_list  = _norm_emb_list(_first_list(d.get("embeddings")))
    doc_list  = _as_list(_first_list(d.get("documents")))

    # 3) bm25 (fetch by IDs; no per-request encoding)
    t_bm25_ids = t_bm25_fetch = None
    if RETR_USE_BM25 and bm25 is not None and metas:
        bm_hits = bm25.get_top_n(tokenize(query), metas, n=k_bm25)
        t_bm25_ids = time.perf_counter()
        bm_ids = [h.get("id") for h in bm_hits if h.get("id")]
        bm_meta, bm_embs, bm_docs = _fetch_by_ids(col, bm_ids)
        t_bm25_fetch = time.perf_counter()
        meta_list = _as_list(meta_list) + _as_list(bm_meta)
        emb_list  = _norm_emb_list(emb_list) + _norm_emb_list(bm_embs)
        doc_list  = _as_list(doc_list) + _as_list(bm_docs)

    # 4) dedup by id
    by_id: Dict[str, Tuple[Dict[str, Any], List[float], str]] = {}
    for m, e, txt in zip(meta_list, emb_list, doc_list):
        mid = (m or {}).get("id")
        if mid and mid not in by_id:
            by_id[mid] = (m, _as_pyfloat_list(e), txt or (m or {}).get("text", "") or "")
    if not by_id:
        return {"results": [], "timings": {"total_ms": round((time.perf_counter()-t0)*1000, 1)}}

    metas_u, embs_u, docs_u = [], [], []
    for _, (m, e, txt) in by_id.items():
        metas_u.append(m); embs_u.append(_as_pyfloat_list(e)); docs_u.append(txt)

    # 5) mmr
    pick = _mmr(_as_pyfloat_list(qv), embs_u, lamb=mmr_lambda, topk=mmr_k)
    t_mmr = time.perf_counter()

    results: List[Dict[str, Any]] = []
    for i in pick:
        m = metas_u[i]
        results.append({
            "id": m.get("id", ""),
            "title": m.get("title", ""),
            "url": m.get("url", ""),
            "text": docs_u[i] or m.get("text", "") or "",
        })

    timings = {
        "q_embed_ms": round((t_q - t0)*1000, 1),
        "dense_query_ms": round((t_dense - t_q)*1000, 1),
        "bm25_ids_ms": None if t_bm25_ids is None else round((t_bm25_ids - t_dense)*1000, 1),
        "bm25_fetch_ms": None if t_bm25_fetch is None else round((t_bm25_fetch - t_bm25_ids)*1000, 1),
        "mmr_ms": round((t_mmr - (t_bm25_fetch or t_dense))*1000, 1),
        "total_ms": round((time.perf_counter() - t0)*1000, 1),
        "used_bm25": RETR_USE_BM25,
    }
    return {"results": results, "timings": timings}