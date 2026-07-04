# src/ingest/discover_patch_urls.py
import argparse, time, re, requests, sys
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ROVPatchBot/0.3)"}

SKIP_TXT_PAT = re.compile(
    r"(สกิน|สกิ้น|สกินใหม่|สกินระดับ|คูปอง|ส่วนลด|โปรโมชัน|โปรโมชั่น|ทัวร์นาเมนต์|การแข่งขัน|"
    r"รายการแข่ง|Valor\s*Pass|วาเลอร์พาส|กิจกรรม|อีเวนท์|event|promo|coupon|skin|tournament)",
    flags=re.I
)

def same_host(u, base):
    return urlparse(u).netloc == urlparse(base).netloc

def looks_like_detail(url: str, listing_root: str) -> bool:
    if not same_host(url, listing_root):
        return False
    if url.rstrip("/") == listing_root.rstrip("/"):
        return False
    return bool(re.search(r"/patch-notes/[^/]+/?$", url))

def _get(url: str, retries: int = 3, backoff: float = 0.6):
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))

def discover_listing(start_url: str, max_pages: int = 10, max_urls: int = 500):
    urls, seen_pages, seen_urls = [], set(), set()
    url = start_url

    for _ in range(max_pages):
        if url in seen_pages:
            break
        seen_pages.add(url)

        r = _get(url)
        s = BeautifulSoup(r.text, "html.parser")

        for a in s.select("a[href]"):
            txt = (a.get_text(" ", strip=True) or "")
            href = a["href"].strip()
            absu = urljoin(url, href)
            if looks_like_detail(absu, start_url) and not SKIP_TXT_PAT.search(txt):
                if absu not in seen_urls:
                    seen_urls.add(absu)
                    urls.append(absu)
                    if len(urls) >= max_urls:
                        break

        if len(urls) >= max_urls:
            break

        next_a = s.find("a", string=lambda t: t and ("ถัดไป" in t or "Next" in t or "หน้า" in t))
        if not next_a:
            next_a = s.select_one('a[rel="next"]')
        if next_a and next_a.get("href"):
            url = urljoin(url, next_a["href"])
        else:
            break

        time.sleep(0.6)

    return urls

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="https://rov.in.th/patch-notes")
    ap.add_argument("--out", default="data/urls_all.txt")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--max-urls", type=int, default=500)
    args = ap.parse_args()

    try:
        found = discover_listing(args.start, args.max_pages, args.max_urls)
    except Exception as e:
        print(f"[ERR] crawl failed: {e}", file=sys.stderr)
        sys.exit(2)

    with open(args.out, "w", encoding="utf-8") as f:
        for u in found:
            f.write(u + "\n")
    print(f"Wrote {len(found)} URLs → {args.out}")
