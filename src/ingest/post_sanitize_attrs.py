# src/ingest/post_sanitize_attrs.py
"""
DEPRECATED (optional safety-net).

Primary pipeline should rely on `normalize_chunk.py` with `--aggressive-infer`.
Keep this script only for one-off repairs if you discover legacy artifacts with
truncated cooldown tokens or narrative/garbage attrs that slipped through.

Usage (optional):
    python -m src.ingest.post_sanitize_attrs \
        --in data/corpus.jsonl \
        --out data/corpus_clean.jsonl
"""

import argparse, json, re, sys

TH = r"\u0E00-\u0E7F"

KEEP_CUES = [
    r"คูล\s*ดาวน์", r"คูลดาวน์", r"cool\s*down", r"cooldown", r"\bcd\b",
    r"ความเสียหาย", r"damage", r"dmg",
    r"ความเสียหาย\s*จริง", r"true\s*damage",
    r"ความเสียหายเวท|magic\s*damage|ap\s*damage",
    r"ความเสียหายกายภาพ|physical\s*damage|ad\s*damage",
    r"ความเร็วเคลื่อนที่|movement\s*speed|ms",
    r"ความเร็วโจมตี|attack\s*speed|as",
    r"พลังชีวิต|hp|max\s*hp|health",
    r"ฟื้นฟูพลังชีวิต|hp\s*regen|health\s*regen",
    r"มานา|mp|mana",
    r"ฟื้นฟูมานา|mp\s*regen|mana\s*regen",
    r"ดูดเลือด|lifesteal",
    r"เวทแวมไพร์|spell\s*vamp",
    r"โล่|shield",
    r"พลังโจมตี|ad|attack\s*damage",
    r"พลังเวท|ap|ability\s*power",
    r"เกราะเวท|magic\s*resist|mr",
    r"เกราะ|armor",
    r"อัตราคริติคอล|critical\s*rate",
    r"ความเสียหายคริติคอล|critical\s*damage",
    r"ระยะ|ระยะสกิล|range|radius",
    r"(ค่าใช้จ่าย)?มานา|mana\s*cost",
    r"ต้าน\s*สถานะ|tenacity|cc\s*resist",
]
KEEP_RX = re.compile("|".join(f"(?:{p})" for p in KEEP_CUES), re.I)

# Slot labels
SLOT_RX = re.compile(r"(สกิล\s*[123]|อัลติ(?:เมท)?|ultimate|ult)", re.I)

def _canon_slot(s: str) -> str:
    t = s.strip().lower()
    m = re.search(r"สกิล\s*([123])", t)
    if m: return f"สกิล {m.group(1)}"
    if re.search(r"(อัลติ(?:เมท)?|ultimate|ult)\b", t): return "อัลติ"
    return s.strip()

# Thai-safe “token” around คู (simulate \b using non-Thai/non-ASCII boundaries)
NOT_WORD = r"[^0-9A-Za-z" + TH + r"]"
TRUNC_COOLDOWN_RX = re.compile(
    rf"(ระยะเวลาคู|าว์น|(?:^|{NOT_WORD})คู(?:$|{NOT_WORD}))", re.I
)

# quick “รายละเอียด …” starter
LEADS_WITH_DETAIL_RX = re.compile(r"^\s*รายละเอียด", re.I)

def _looks_like_long_sentence(s: str) -> bool:
    t = s.strip()
    if len(t) <= 40:
        return False
    if re.search(rf"[{TH}]", t) and " " in t:
        if not KEEP_RX.search(t):
            return True
    return False

def _is_numeric_only(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9\s%/().:+\-]+", s or ""))

def _is_slot_only(s: str) -> bool:
    return bool(re.fullmatch(SLOT_RX, (s or "").strip()))

def _repair_and_collapse_cooldown(attr: str) -> str:
    """
    If attr contains a truncated cooldown token anywhere, collapse the entire field
    to a canonical 'คูลดาวน์' (+ slot if present).
    """
    if TRUNC_COOLDOWN_RX.search(attr) and "คูลดาวน์" not in attr:
        m = SLOT_RX.search(attr)
        slot = _canon_slot(m.group(0)) if m else None
        return "คูลดาวน์" + (f" {slot}" if slot else "")
    return attr

def _is_garbage_after_repair(attr: str) -> bool:
    if not attr or not attr.strip():
        return True
    a = attr.strip()
    if _is_numeric_only(a):
        return True
    if _is_slot_only(a):
        return True
    if LEADS_WITH_DETAIL_RX.search(a) and "คูลดาวน์" not in a:
        return True
    if _looks_like_long_sentence(a):
        return True
    return False

def process(inp: str, out: str):
    n_in = n_out = n_touched = 0
    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            # --- robustness: skip malformed JSON instead of crashing ---
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] bad JSON line skipped: {e}", file=sys.stderr)
                continue

            touched = False
            fixed_changes = []
            for ch in obj.get("changes", []):
                a = ch.get("attr")
                if a:
                    # 1) repair/collapse cooldown if truncated token appears
                    repaired = _repair_and_collapse_cooldown(a)
                    if repaired != a:
                        ch["attr"] = repaired
                        a = repaired
                        touched = True

                    # 2) after repair, drop garbage/narrative leftovers
                    if _is_garbage_after_repair(a):
                        # keep canonical 'คูลดาวน์' if already collapsed; else drop
                        if a.strip().startswith("คูลดาวน์"):
                            # keep as is
                            pass
                        else:
                            ch["attr"] = None
                            touched = True

                fixed_changes.append(ch)

            obj["changes"] = fixed_changes
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            n_touched += 1 if touched else 0
            n_in += 1
    print(f"Sanitized {n_in} rows → {n_out} rows → {out}  (touched {n_touched})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    process(args.inp, args.out)
