import argparse, json, re, sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

# ---------- Section taxonomy (Thai/EN cues) ----------
SEC_PATTERNS = [
    ("Heroes",    r"(ฮีโร่|ปรับสมดุล|บาลานซ์|hero|balance)"),
    ("Items",     r"(ไอเทม|item[s]?)"),
    ("Objectives",r"(ดาร์ค\s*สเลเยอร์|อินฟินิต|อาบิสซัล|สปิริต|ครีปใหญ่|objective|slayer|abyssal|sentinel)"),
    ("Jungle/Economy", r"(มอนสเตอร์ป่า|จังเกิล|jungle|โกลด์|ทอง|exp|เศรษฐกิจ|minion|ครีป)"),
    ("Systems/QoL", r"(ระบบ|คุณภาพชีวิต|warp|วาร์ป|เทเลพอร์ต|rank|จัดอันดับ|จับคู่|report|รายงาน|ตั้งค่า|UI|UX|ระบบเกม|matchmaking)"),
    ("Map/Mode",  r"(แผนที่|โหมด|mode|map)"),
    ("Bugfix",    r"(แก้ไขบัค|บัค|bugfix|bug)"),
]

# Seeds
HERO_HINTS = {"Zanis","Violet","Hayate","Tel'Annas","Airi","Lu Bu","Butterfly"}
ITEM_HINTS = {"Cursed Helmet","Frost Cape","Gilded Greaves","Soulreaver","Leviathan"}
OBJECTIVE_HINTS = {"Dark Slayer","Infinite Slayer","Abyssal Dragon","Spirit Sentinel","Jumpy Hare","Rock Crab","Tree Toad"}

# ---------- Regex helpers ----------
TH = r"\u0E00-\u0E7F"  # Thai block

# A → B (with optional label before numbers)
ARROW_LABELED = re.compile(
    rf"(?P<label>[\w{TH}\s/():\-]{{0,40}}?)\s*[:：]?\s*(?P<old>\d+(\.\d+)?)\s*(?:→|->|=>|➡|⟶|⟹)\s*(?P<new>\d+(\.\d+)?)"
)

# ±x%
DELTA_PCT  = re.compile(r"(?P<sign>[+\-])\s*(?P<val>\d+(\.\d+)?)\s*%")

# Thai เพิ่ม/ลด ... จาก X เป็น Y
ADD_THAI = re.compile(
    rf"(?:เพิ่ม|\+)\s*(?P<attr>[\w{TH}\s/()\-]+?)\s*(?:จาก|เป็น|:)?\s*(?P<old>\d+(\.\d+)?)[^\d%]+(?P<new>\d+(\.\d+)?)(?P<unit>[%\w{TH}]*)"
)
RED_THAI = re.compile(
    rf"(?:ลด|\-)\s*(?P<attr>[\w{TH}\s/()\-]+?)\s*(?:จาก|เป็น|:)?\s*(?P<old>\d+(\.\d+)?)[^\d%]+(?P<new>\d+(\.\d+)?)(?P<unit>[%\w{TH}]*)"
)

# Canonical attribute cues (Thai + ENG)
ATTR_CUES = [
    # cooldown / time
    (re.compile(rf"(คูล\s*ดาวน์|คูลดาวน์|cool\s*down|cooldown|cd)", re.I), "คูลดาวน์"),
    (re.compile(rf"(ระยะเวลา|duration)", re.I), "ระยะเวลา"),
    (re.compile(rf"(ระยะเวลาคูลดาวน์)", re.I), "คูลดาวน์"),
    (re.compile(r"ระยะเวลาคู"), "คูลดาวน์"),  # common truncation to cooldown

    # damage
    (re.compile(rf"(ความเสียหาย|damage|dmg)\b", re.I), "ความเสียหาย"),
    (re.compile(rf"(ความเสียหาย\s*จริง|true\s*damage)\b", re.I), "ความเสียหายจริง"),
    (re.compile(rf"(ความเสียหายเวท|magic\s*damage|AP\s*damage)\b", re.I), "ความเสียหายเวท"),
    (re.compile(rf"(ความเสียหายกายภาพ|physical\s*damage|AD\s*damage)\b", re.I), "ความเสียหายกายภาพ"),

    # speed
    (re.compile(rf"(ความเร็วเคลื่อนที่|movement\s*speed|ms)\b", re.I), "ความเร็วเคลื่อนที่"),
    (re.compile(rf"(ความเร็วโจมตี|attack\s*speed|as)\b", re.I), "ความเร็วโจมตี"),

    # resource / sustain
    (re.compile(rf"(พลังชีวิต|ชีวิตสูงสุด|hp|max\s*hp|health)\b", re.I), "พลังชีวิต"),
    (re.compile(rf"(ฟื้นฟูพลังชีวิต|hp\s*regen|health\s*regen)\b", re.I), "ฟื้นฟูพลังชีวิต"),
    (re.compile(rf"(มานา|mp|mana)\b", re.I), "มานา"),
    (re.compile(rf"(ฟื้นฟูมานา|mp\s*regen|mana\s*regen)\b", re.I), "ฟื้นฟูมานา"),
    (re.compile(rf"(ดูดเลือด|lifesteal)\b", re.I), "ดูดเลือด"),
    (re.compile(rf"(เวทแวมไพร์|spell\s*vamp)\b", re.I), "เวทแวมไพร์"),
    (re.compile(rf"(โล่|shield)\b", re.I), "โล่"),

    # offense / defense stats
    (re.compile(rf"(พลังโจมตี|ad|attack\s*damage)\b", re.I), "พลังโจมตี"),
    (re.compile(rf"(พลังเวท|ap|ability\s*power)\b", re.I), "พลังเวท"),
    (re.compile(rf"(เกราะเวท|magic\s*resist|mr)\b", re.I), "เกราะเวท"),
    (re.compile(rf"(เกราะ|armor)\b", re.I), "เกราะ"),
    (re.compile(rf"(อัตราคริติคอล|critical\s*rate|crit\s*rate)\b", re.I), "อัตราคริติคอล"),
    (re.compile(rf"(ความเสียหายคริติคอล|critical\s*damage|crit\s*dmg)\b", re.I), "ความเสียหายคริติคอล"),

    # range / radius / cost
    (re.compile(rf"(ระยะ|ระยะสกิล|range|radius)\b", re.I), "ระยะ"),
    (re.compile(rf"(ค่าใช้จ่ายมานา|ใช้มานา|mana\s*cost)\b", re.I), "ใช้มานา"),

    # misc
    (re.compile(rf"(ต้าน\s*สถานะ|tenacity|cc\s*resist)\b", re.I), "ต้านสถานะ"),
]

# Skill slot cue (e.g., "สกิล 1", "อัลติ")
SKILL_SLOT_RE = re.compile(rf"(สกิล\s*[123]|อัลติ|อัลติเมท|ultimate)", re.I)

def guess_version(text: str) -> str:
    m = re.search(r"(\b\d{1,2}\.\d{1,2}\b)", text)
    return m.group(1) if m else ""

def guess_section(block: str) -> str:
    for name, pat in SEC_PATTERNS:
        if re.search(pat, block, flags=re.I):
            return name
    return "General"

def harvest_entities(block: str) -> Tuple[Optional[str], Optional[str]]:
    low = block.lower()
    for h in HERO_HINTS:
        if h.lower() in low:
            return "Hero", h
    for it in ITEM_HINTS:
        if it.lower() in low:
            return "Item", it
    for obj in OBJECTIVE_HINTS:
        if obj.lower() in low:
            return "Objective", obj
    first = re.match(r"^([A-Z][\w'’\-]+)", block.strip())
    if first:
        name = first.group(1)
        sec = guess_section(block)
        et = {"Heroes":"Hero","Items":"Item","Objectives":"Objective"}.get(sec, "System")
        return et, name
    return None, None

# ---------- Attribute post-clean ----------
def _looks_suspicious(attr: Optional[str]) -> bool:
    if not attr: return True
    if len(attr) < 3: return True
    if "าวน์" in attr and "คูลดาวน์" not in attr: return True
    if re.search(rf"[^\w {TH}/%\-\.\(\)]", attr): return True
    return False

def _normalize_attr_text(attr: str) -> str:
    s = re.sub(r"\s+", " ", attr).strip()
    # Fix orphan “…าวน์…” → “คูลดาวน์…”
    if "าวน์" in s and "คูลดาวน์" not in s:
        s = re.sub(r"(าวน์)", "คูลดาวน์", s, count=1)
    # Handle common truncations → cooldown
    if s == "คู" or re.fullmatch(r"ระยะเวลาคู(\s+สกิล\s*[123])?", s):
        slot = SKILL_SLOT_RE.search(s)
        return "คูลดาวน์" + (f" {slot.group(0)}" if slot else "")
    # Canonicalize by cues
    for rx, canon in ATTR_CUES:
        if rx.search(s):
            slot = SKILL_SLOT_RE.search(s)
            return canon + (f" {slot.group(0)}" if slot else "")
    return s

def _has_thai(s: str) -> bool:
    return bool(re.search(rf"[{TH}]", s))

def _is_numeric_or_punct(s: str) -> bool:
    return bool(re.fullmatch(r"[0-9\s%/().:+\-]*", s)) and s.strip() != ""

def _contains_known_cue(s: str) -> bool:
    return any(rx.search(s) for rx, _ in ATTR_CUES)

def _is_slot_only(s: str) -> bool:
    return bool(SKILL_SLOT_RE.fullmatch(s.strip()))

def _is_low_quality_attr(attr: Optional[str]) -> bool:
    if attr is None: return True
    s = attr.strip()
    if not s: return True
    if len(s) <= 1 and s not in {"โล่"}: return True
    if "าวน์" in s and "คูลดาวน์" not in s: return True
    if re.fullmatch(r"[-–•:：/()\s]+", s): return True
    # Avoid narrative fragments in facets
    if s.startswith("รายละเอียด "): return True
    if re.search(rf"[^\w {TH}/%\-\.\(\)]", s): return True
    return False

def _extract_attr_from_context(ctx: str) -> Optional[str]:
    for rx, canon in ATTR_CUES:
        if rx.search(ctx):
            slot = SKILL_SLOT_RE.search(ctx)
            return canon + (f" {slot.group(0)}" if slot else "")
    m = re.search(rf"[\n\r]([ {TH}/()\-]+?)\s*[:：]\s*$", ctx)
    if m:
        candidate = _normalize_attr_text(m.group(1))
        if candidate and not _is_low_quality_attr(candidate): return candidate
    m = re.search(rf"([ {TH}/()\-]+)\s*[:：]?\s*$", ctx)
    if m:
        candidate = _normalize_attr_text(m.group(1))
        if candidate and not _is_low_quality_attr(candidate): return candidate
    return None

def _find_cue_near_span(text: str, start: int, end: int, radius: int = 140) -> Optional[str]:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    left_ctx = text[left:start]
    cand = _extract_attr_from_context(left_ctx)
    if cand: return cand
    right_ctx = text[end:right]
    for rx, canon in ATTR_CUES:
        if rx.search(right_ctx):
            slot = SKILL_SLOT_RE.search(right_ctx)
            return canon + (f" {slot.group(0)}" if slot else "")
    return None

def _merge_slot_with_nearby_cue(full_text: str, start: int, end: int, fallback_attr: Optional[str]) -> Optional[str]:
    slot = None
    if fallback_attr and SKILL_SLOT_RE.search(fallback_attr):
        slot = SKILL_SLOT_RE.search(fallback_attr).group(0)
    else:
        L = max(0, start - 60); R = min(len(full_text), end + 60)
        m = SKILL_SLOT_RE.search(full_text[L:R])
        slot = m.group(0) if m else None
    cue = _find_cue_near_span(full_text, start, end, radius=120)
    if cue and slot: return f"{cue} {slot}"
    return cue

# ---------- Change parsers ----------
def parse_changes_with_spans(text: str) -> List[Dict]:
    changes = []
    # Arrow A → B (with optional label)
    for m in ARROW_LABELED.finditer(text):
        label = (m.group("label") or "").strip()
        label = re.sub(r"^[\-\–•\s]+", "", label).strip()
        label = re.sub(r"\s{2,}", " ", label)
        old = m.group("old"); new = m.group("new")
        try:
            direction = "up" if float(new) > float(old) else "down"
        except Exception:
            direction = None
        changes.append({
            "attr": label or None,
            "old": old, "new": new, "unit": None,
            "direction": direction, "pattern": "arrow",
            "_span": m.span(), "_label_raw": label
        })

    # +x% / -x% → store numeric value + unit, sign into direction
    for m in re.finditer(r"(?P<sign>[+\-])\s*(?P<val>\d+(\.\d+)?)\s*%", text):
        sign = m.group("sign"); val = m.group("val")
        changes.append({
            "attr": None, "old": None, "new": val, "unit": "%",
            "direction": "up" if sign=="+" else "down",
            "pattern": "delta_pct",
            "_span": m.span()
        })
    # 3) Thai เพิ่ม/ลด…
    for m in ADD_THAI.finditer(text):
        changes.append({
            "attr": (m.group("attr") or "").strip(),
            "old": m.group("old"), "new": m.group("new"),
            "unit": (m.group("unit") or "").strip() or None,
            "direction": "up", "pattern": "thai_add",
            "_span": m.span()
        })
    for m in RED_THAI.finditer(text):
        changes.append({
            "attr": (m.group("attr") or "").strip(),
            "old": m.group("old"), "new": m.group("new"),
            "unit": (m.group("unit") or "").strip() or None,
            "direction": "down", "pattern": "thai_reduce",
            "_span": m.span()
        })
    return changes

# ---------- Attr cleaning/enrichment ----------
def clean_and_enrich_attrs(changes: List[Dict], full_text: str, aggressive: bool = False) -> List[Dict]:
    out = []
    for ch in changes:
        attr = ch.get("attr")
        span = ch.get("_span")

        # 1) left-context inference if suspicious/empty
        if (not attr or _looks_suspicious(attr)) and span:
            start, end = span
            left = max(0, start - 60)
            ctx_left = full_text[left:start]
            inferred = _extract_attr_from_context(ctx_left)
            if inferred: attr = inferred

        # 2) normalize whatever we have
        if attr: attr = _normalize_attr_text(attr)

        # 3) optional aggressive repair
        if aggressive and span and _is_low_quality_attr(attr):
            start, end = span
            if attr and _is_slot_only(attr):
                merged = _merge_slot_with_nearby_cue(full_text, start, end, attr)
                if merged: attr = _normalize_attr_text(merged)
            if _is_low_quality_attr(attr):
                cue = _find_cue_near_span(full_text, start, end, radius=120)
                if cue: attr = _normalize_attr_text(cue)

        # 4) final gate
        if _is_low_quality_attr(attr): attr = None

        ch["attr"] = attr
        ch.pop("_span", None)
        ch.pop("_label_raw", None)
        out.append(ch)
    return out

# ---------- Chunking ----------
def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    parts = re.split(r"(?<=[\.!?…]|。)\s+", text) if text else []
    out, cur = [], ""
    for s in parts:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur: out.append(cur)
            cur = s
    if cur: out.append(cur)
    if not out and text:
        out = [text[:max_chars]]
    return out

# ---------- Record normalize ----------
def normalize_record(rec: Dict, chunk_chars: int, aggressive: bool) -> List[Dict]:
    version = guess_version(rec.get("title","") + " " + rec.get("text",""))
    chunks = chunk_text(rec.get("text",""), max_chars=chunk_chars)
    out = []
    for i, ch in enumerate(chunks):
        section = guess_section(ch)
        etype, ename = harvest_entities(ch)
        changes_raw = parse_changes_with_spans(ch)
        changes = clean_and_enrich_attrs(changes_raw, ch, aggressive=aggressive)
        out.append({
            "id": f"{rec['id']}#c{i}",
            "type": "patch_note" if section != "Systems/QoL" else "gameplay_update",
            "lang": rec.get("lang","th"),
            "date": rec.get("date",""),
            "version": version,
            "section": section,
            "entity_type": etype,
            "entity_name": ename,
            "changes": changes,
            "text": ch,
            "url": rec.get("url",""),
            "source_tier": rec.get("source_tier","official"),
            "title": rec.get("title",""),
        })
    return out

# ---------- Facet aggregation ----------
def write_facets_from_output(out_path: str, facet_out: str, min_conf: int = 1, max_examples: int = 5) -> None:
    """
    Aggregate facets from the written normalized corpus.
    Facet key = (attr, direction, unit). Keep only entries with count >= min_conf.
    Write JSONL with: {attr, direction, unit, count, examples:[chunk_ids]}
    """
    counts: Counter = Counter()
    exs: Dict[Tuple[str, Optional[str], Optional[str]], List[str]] = defaultdict(list)

    with open(out_path, "r", encoding="utf-8") as rf:
        for ln in rf:
            obj = json.loads(ln)
            cid = obj.get("id")
            for ch in obj.get("changes", []):
                attr = ch.get("attr")
                if not attr: continue
                direction = ch.get("direction")
                unit = ch.get("unit")
                key = (attr, direction, unit)
                counts[key] += 1
                if len(exs[key]) < max_examples:
                    exs[key].append(cid)

    with open(facet_out, "w", encoding="utf-8") as wf:
        for (attr, direction, unit), c in counts.most_common():
            if c < min_conf: continue
            wf.write(json.dumps({
                "attr": attr,
                "direction": direction,
                "unit": unit,
                "count": c,
                "examples": exs[(attr, direction, unit)],
            }, ensure_ascii=False) + "\n")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-chars", type=int, default=900)
    ap.add_argument("--aggressive-infer", action="store_true",
                    help="Infer missing/suspicious attrs from local context (opt-in).")
    ap.add_argument("--qa", action="store_true",
                    help="Print a tiny QA summary after writing the file.")
    ap.add_argument("--facet-out", default=None,
                    help="If set, write aggregated facets JSONL here.")
    ap.add_argument("--facet-min-conf", type=int, default=1,
                    help="Minimum count to keep a facet in --facet-out.")
    ap.add_argument("--facet-max-examples", type=int, default=5,
                    help="Max example chunk ids to include per facet.")
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.inp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[skip] bad JSON line: {e}", file=sys.stderr)
                continue
            for r in normalize_record(rec, chunk_chars=args.chunk_chars, aggressive=args.aggressive_infer):
                w.write(json.dumps(r, ensure_ascii=False) + "\n"); n_out += 1
            n_in += 1

    print(f"Normalized {n_in} articles → {n_out} chunks → {args.out}")

    # Optional QA
    if args.qa:
        total_changes = with_attr = 0
        try:
            with open(args.out, "r", encoding="utf-8") as rf:
                for ln in rf:
                    obj = json.loads(ln)
                    for ch in obj.get("changes", []):
                        total_changes += 1
                        if ch.get("attr"):
                            with_attr += 1
            pct = (with_attr / total_changes * 100.0) if total_changes else 0.0
            print(f"QA: changes={total_changes}, with_attr={with_attr} ({pct:.1f}%)")
        except Exception as e:
            print(f"[qa] could not compute summary: {e}", file=sys.stderr)

    # Optional facets
    if args.facet_out:
        try:
            write_facets_from_output(
                args.out,
                args.facet_out,
                min_conf=args.facet_min_conf,
                max_examples=args.facet_max_examples,
            )
            print(f"Facets written → {args.facet_out} (min_conf={args.facet_min_conf})")
        except Exception as e:
            print(f"[facet] could not write facets: {e}", file=sys.stderr)
