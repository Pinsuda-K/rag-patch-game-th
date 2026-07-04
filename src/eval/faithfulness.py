# src/eval/faithfulness.py
import json, re, argparse, sys
from typing import List

_TOKEN_RE = re.compile(r"[\wก-๙]+", re.UNICODE)

def _tok(s: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((s or "").lower()) if len(t) > 2]

def containment(ans: str, ctx: str) -> float:
    toks = _tok(ans)
    if not toks:
        return 0.0
    bag = set(_tok(ctx))
    hits = sum(1 for t in toks if t in bag)
    return hits / max(1, len(toks))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="JSONL with {answer, context} or {answer, contexts:[]}")
    args = ap.parse_args()

    scores = []
    with open(args.qa, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[WARN] bad line: {e}\n")
                continue
            ans = row.get("answer", "")
            ctx = row.get("context")
            if ctx is None and isinstance(row.get("contexts"), list):
                ctx = "\n\n".join(row["contexts"])
            if ctx is None:
                continue
            scores.append(containment(ans, ctx))
    if not scores:
        print("faithfulness=0.000  n=0")
        return
    print(f"faithfulness={sum(scores)/len(scores):.3f}  n={len(scores)}")

if __name__ == "__main__":
    main()
