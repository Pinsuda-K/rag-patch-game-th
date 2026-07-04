# src/eval/bench_client.py
import argparse, time, statistics, requests, random, json, sys
from typing import List

DEFAULT_URL = "http://localhost:8000/answer"
DEFAULT_Q = [
    "ในแพตช์ 1.53 Zanis เปลี่ยนอะไรบ้าง?",
    "มีแพทช์ไหนลดคูลดาวน์ของ Yorn บ้าง?"
]

def _load_queries(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_Q
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def call(url: str, q: str, k: int, use_reranker: bool) -> float:
    t0 = time.perf_counter()
    r = requests.post(url, json={"q": q, "use_reranker": use_reranker, "k": k}, timeout=60)
    r.raise_for_status()
    _ = r.json()
    return (time.perf_counter() - t0) * 1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--queries", help="Path to .txt with one query per line")
    args = ap.parse_args()

    qs = _load_queries(args.queries)
    use_reranker = not args.no_reranker

    try:
        call(args.url, random.choice(qs), args.k, use_reranker)
    except Exception as e:
        print(f"Warmup failed: {e}", file=sys.stderr)

    xs: List[float] = []
    for _ in range(args.n):
        q = random.choice(qs)
        try:
            xs.append(call(args.url, q, args.k, use_reranker))
        except Exception as e:
            print(f"[WARN] request failed: {e}", file=sys.stderr)

    if not xs:
        print("No successful requests.")
        return

    xs.sort()
    p50 = statistics.median(xs)
    p95 = xs[int(0.95 * len(xs)) - 1]
    print(json.dumps({
        "n": len(xs),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "min_ms": round(min(xs), 1),
        "max_ms": round(max(xs), 1)
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
