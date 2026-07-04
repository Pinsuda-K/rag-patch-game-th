# src/index/build_dense.py
import argparse, json, os, sys
from typing import Dict, Any, Iterable, List
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_CORPUS = "data/corpus.jsonl"
DEFAULT_DB_DIR = "data/chroma_db"
DEFAULT_MODEL = os.getenv("EMBEDDER_MODEL", "BAAI/bge-m3")  # or intfloat/multilingual-e5-base
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")

@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any]

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

def _changes_summary(rec: Dict[str, Any], max_items:int=6) -> str:
    xs = []
    for ch in (rec.get("changes") or [])[:max_items]:
        attr = ch.get("attr", "")
        old  = ch.get("old", "")
        new  = ch.get("new", "")
        direction = ch.get("direction", "")
        xs.append(f"{attr}: {old} → {new} ({direction})")
    return " ; ".join(xs)

def _structured_view(rec: Dict[str, Any]) -> str:
    section = rec.get("section", "")
    ent_t   = rec.get("entity_type", "")
    ent_n   = rec.get("entity_name", "")
    title   = rec.get("title", "")
    ch_sum  = _changes_summary(rec)
    head    = f"[{section}] {ent_t} {ent_n}".strip()
    parts = [p for p in [head, title, ch_sum] if p]
    return " — ".join(parts)

def _sanitize_meta(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten metadata to primitives only (Chroma requirement)."""
    keep = {
        "id": rec.get("id"),
        "type": rec.get("type"),
        "lang": rec.get("lang"),
        "date": rec.get("date"),
        "version": rec.get("version"),
        "section": rec.get("section"),
        "entity_type": rec.get("entity_type"),
        "entity_name": rec.get("entity_name"),
        "title": rec.get("title"),
        "url": rec.get("url"),
        "source_tier": rec.get("source_tier"),
        # String summaries only (no lists/dicts):
        "changes_summary": _changes_summary(rec),
    }
    return {k: ("" if v is None else v) for k, v in keep.items()}

def _build_passage(rec: Dict[str, Any]) -> str:
    struct = _structured_view(rec)
    body = (rec.get("text") or "")[:2000]
    return (struct + "\n" + body).strip()

def _prefix_if_e5(model_name:str, texts:List[str]) -> List[str]:
    m = model_name.lower()
    if "e5" in m:
        return [("passage: " + t) for t in texts]
    return texts

def _load_docs(corpus_path:str) -> List[Doc]:
    docs: List[Doc] = []
    for rec in _iter_jsonl(corpus_path):
        did = rec.get("id")
        if not did:
            continue
        text = _build_passage(rec)
        docs.append(Doc(id=str(did), text=text, meta=rec))
    docs.sort(key=lambda d: d.id)
    return docs

def _batched(lst: List[Any], bsz: int):
    for i in range(0, len(lst), bsz):
        yield lst[i : i + bsz]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_CORPUS)
    ap.add_argument("--db", dest="db_dir", default=DEFAULT_DB_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reset", action="store_true", help="Drop existing collection before run.")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    client = chromadb.PersistentClient(path=args.db_dir, settings=Settings(allow_reset=True))

    if args.reset:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"}
    )

    docs = _load_docs(args.inp)
    if not docs:
        print("No docs found.", file=sys.stderr)
        sys.exit(1)

    # We use UPSERT so we don't need a pre-scan for existing IDs.
    print(f"Embedding {len(docs)} docs with '{args.model}' → {args.db_dir}")

    texts = [d.text for d in docs]
    texts = _prefix_if_e5(args.model, texts)

    for batch in tqdm(list(_batched(list(zip(docs, texts)), args.batch)), desc="Encoding"):
        bs_docs, bs_texts = zip(*batch)
        embs = model.encode(
            list(bs_texts),
            batch_size=min(256, args.batch),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # UPSERT to avoid duplicate-ID headaches and to allow incremental rebuilds
        col.upsert(
            ids=[d.id for d in bs_docs],
            embeddings=[e.tolist() for e in embs],
            metadatas=[_sanitize_meta(d.meta) for d in bs_docs],
            documents=[d.text for d in bs_docs],
        )

    try:
        total = col.count()
    except Exception:
        total = "?"
    print(f"Chroma upserted {len(docs)} docs (total={total}) at {args.db_dir}")

if __name__ == "__main__":
    main()
