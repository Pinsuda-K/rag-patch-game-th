# src/index/build_bm25.py
import argparse, json, re, sys
from typing import List, Dict, Any, Iterable
from rank_bm25 import BM25Okapi

DEFAULT_CORPUS = "data/corpus.jsonl"
DEFAULT_OUT = "data/bm25_idx.json"

_TOKEN_RE = re.compile(r"[\wก-๙]+", re.UNICODE)

def tokenize(s: str) -> List[str]:
    return _TOKEN_RE.findall((s or "").lower())

def _iter_jsonl(path:str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[WARN] bad line: {e}\n")

def _s(x) -> str:
    """Coerce any value (including None) to a safe string for joining/tokenizing."""
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return str(x)

def _doc_text(r: Dict[str, Any]) -> str:
    section = r.get("section")
    ent_t   = r.get("entity_type")
    ent_n   = r.get("entity_name")
    chs = r.get("changes") or []
    # coerce potential None attr/old/new to strings safely
    ch_str = " ; ".join(f"{_s(c.get('attr'))}: {_s(c.get('old'))}→{_s(c.get('new'))}" for c in chs[:6])
    parts = [
        r.get("title"),
        r.get("text"),
        section, ent_t, ent_n,
        ch_str,
    ]
    return " ".join(_s(p) for p in parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    docs, metas, ids = [], [], []
    for r in _iter_jsonl(args.inp):
        did = r.get("id")
        if not did:
            continue
        ids.append(_s(did))
        metas.append(r)
        docs.append(_doc_text(r))

    tokenized = [tokenize(d) for d in docs]
    _ = BM25Okapi(tokenized)  # not serialized; kept for symmetry

    out = {"tokens": tokenized, "ids": ids, "metas": metas}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"BM25 built on {len(docs)} docs → {args.out}")

if __name__ == "__main__":
    main()
