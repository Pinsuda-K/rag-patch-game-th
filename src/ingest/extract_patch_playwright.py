# src/ingest/extract_patch_playwright.py
import argparse, json, re, asyncio, sys
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

def canon_id(url: str) -> str:
    return re.sub(r"\W+", "_", url)[-48:]

TH_MONTHS = {
    "มกราคม":"01","กุมภาพันธ์":"02","มีนาคม":"03","เมษายน":"04","พฤษภาคม":"05","มิถุนายน":"06",
    "กรกฎาคม":"07","สิงหาคม":"08","กันยายน":"09","ตุลาคม":"10","พฤศจิกายน":"11","ธันวาคม":"12"
}

CONTACT_UA = (
    "rag-game-patch-th/1.0 (independent, non-commercial research; "
    "+https://github.com/Pinsuda-K/rag-game-patch-th)"
)
def _strip_nav_noise(soup: BeautifulSoup):
    for sel in [
        "header","footer","nav","aside",".site-header",".site-footer",
        ".menu",".navbar",".breadcrumbs",".breadcrumb",".topbar",".sidebar",
        ".share","[role='banner']","[role='navigation']","[role='contentinfo']",
        "script","style","noscript"
    ]:
        for el in soup.select(sel):
            el.decompose()
    for el in soup.select("script, style, noscript"):
        el.decompose()

def _pick_article_root(soup: BeautifulSoup):
    for sel in ["article .entry-content", "article .post-content", "article",
                "main .entry-content", "main .post-content", "main",
                ".single-post", ".content", ".post-content", ".entry-content"]:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            return node
    return soup

def _parse_th_date(text: str) -> str:
    import datetime as dt
    m = re.search(r"(\d{1,2})\s+([ก-๙]+)\s+(\d{4})", text)
    if not m: 
        return ""
    d, thm, y = m.group(1), m.group(2), int(m.group(3))
    mm = TH_MONTHS.get(thm)
    if not mm:
        return ""
    if y > 2400:
        y -= 543
    try:
        return f"{y:04d}-{mm}-{int(d):02d}"
    except:
        return ""

def _dedupe_paragraph_runs(text: str) -> str:
    import hashlib
    parts = re.split(r"(?:\n|\r|\s{2,}|(?<=\.)\s+)", text)
    seen = set(); out = []
    for p in parts:
        p = p.strip()
        if len(p) < 40:
            out.append(p); continue
        h = hashlib.md5(p[:160].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h); out.append(p)
    return " ".join(out)

def extract_from_html(html: str):
    s = BeautifulSoup(html, "html.parser")
    _strip_nav_noise(s)

    title_el = s.select_one("article h1") or s.select_one("h1") or s.title
    title = title_el.get_text(" ", strip=True) if title_el else ""

    main = _pick_article_root(s)
    blocks = main.select("h1, h2, h3, h4, p, li")
    text = " ".join(b.get_text(" ", strip=True) for b in blocks)
    text = re.sub(r"\s+", " ", text).strip()
    text = _dedupe_paragraph_runs(text)

    date = ""
    meta = s.select_one('meta[property="article:published_time"]') or s.select_one('meta[name="date"]')
    if meta and meta.get("content"):
        date = meta["content"][:10]
    if not date:
        t = s.select_one("time")
        if t:
            date = _parse_th_date(t.get_text(" ", strip=True)) or date
    if not date:
        date = _parse_th_date(title) or _parse_th_date(text[:200])

    return title, date, text

async def fetch_one(page, url: str, nav_timeout_ms: int = 15000):
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
        try:
            await page.wait_for_selector("article, main, .post-content, .entry-content", timeout=8000)
        except:
            pass
        html = await page.content()
        title, date, text = extract_from_html(html)
        return {
            "id": canon_id(url),
            "type": "patch_note",
            "title": title,
            "lang": "th",
            "date": date,
            "section": "all",
            "text": text,
            "url": url,
            "source_tier": "official",
        }
    except Exception as e:
        return {"error": str(e), "url": url}

    async def run(urls, out_path, concurrency: int = 2, delay: float = 2.0,
              cache_dir: str = "data/.cache", refetch: bool = False):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        n = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        context = await browser.new_context(user_agent=CONTACT_UA)
        sem = asyncio.Semaphore(concurrency)

        async def _one(u):
            cache_file = cache / (canon_id(u) + ".json")
            if cache_file.exists() and not refetch:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            async with sem:
                page = await context.new_page()
                rec = await fetch_one(page, u)
                await page.close()
                await asyncio.sleep(delay)   # politeness gap between requests
            if "error" not in rec:
                cache_file.write_text(
                    json.dumps(rec, ensure_ascii=False), encoding="utf-8"
                )
            return rec

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in await asyncio.gather(*[_one(u) for u in urls]):
                if "error" in rec:
                    print("skip", rec["url"], rec["error"], file=sys.stderr)
                    continue
                th_chars = sum(1 for c in rec["text"] if '\u0E00' <= c <= '\u0E7F')
                if th_chars < 200 or len(rec["text"]) < 600:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

        await browser.close()
    print(f"Wrote {n} records → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True)
    ap.add_argument("--out", default="data/corpus_raw.jsonl")
    if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls-file", required=True)
    ap.add_argument("--out", default="data/corpus_raw.jsonl")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait after each request (politeness).")
    ap.add_argument("--cache-dir", default="data/.cache")
    ap.add_argument("--refetch", action="store_true",
                    help="Ignore cache and re-fetch every URL.")
    args = ap.parse_args()
    urls = [ln.strip() for ln in open(args.urls_file, encoding="utf-8") if ln.strip()]
    asyncio.run(run(
        urls,
        args.out,
        concurrency=args.concurrency,
        delay=args.delay,
        cache_dir=args.cache_dir,
        refetch=args.refetch,
    ))
    args = ap.parse_args()
    urls = [ln.strip() for ln in open(args.urls_file, encoding="utf-8") if ln.strip()]
    asyncio.run(run(urls, args.out, args.concurrency))
