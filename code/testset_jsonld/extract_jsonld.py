"""Step 1: extract schema.org JSON-LD news-article metadata from the raw CC-NEWS WARC files.

For every HTML page in data/raw/ccnews_recent/warc/*.warc.gz:
  * parse all <script type="application/ld+json"> blocks (also inside @graph / lists)
  * keep pages whose JSON-LD declares a news article (NewsArticle family or Article)
  * skip sites that occur in the training / held-out CC-News pools (the test set must be new sites)
  * record headline, articleSection, BreadcrumbList names, keywords, inLanguage, dates, author,
    publisher, isAccessibleForFree, and the article text: JSON-LD articleBody when present,
    otherwise the main text extracted by trafilatura
  * language = JSON-LD inLanguage; when absent, fastText LID (lid.176) on the text

One parquet shard per WARC file (resumable). No filtering by topic happens here; that is done by
build_testset.py with the explicit section -> IPTC mapping in iptc_section_map.py.

Usage (from code/):
    python testset_jsonld/extract_jsonld.py --workers 10
"""
import argparse
import html as htmllib
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

import pandas as pd  # noqa: E402

WARC_DIR = RAW_DIR / "ccnews_recent" / "warc"
OUT_DIR = PROCESSED_DIR / "testset_jsonld" / "extracted"
LID_MODEL = MODELS_DIR / "lid" / "lid.176.bin"

ARTICLE_TYPES = {"newsarticle", "reportagenewsarticle", "analysisnewsarticle", "backgroundnewsarticle",
                 "opinionnewsarticle", "reviewnewsarticle", "article"}
LD_SCRIPT = re.compile(rb'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
HTML_LANG = re.compile(rb'<html[^>]*\slang\s*=\s*["\']?([a-zA-Z]{2,3})', re.I)
TAG = re.compile(r"<[^>]+>")

_lid = None
_excluded = None
MAX_PAGES = 0  # >0 only for smoke runs (--max-pages); the shard is then written to a separate folder


# ----------------------------------------------------------------------------- helpers
def excluded_domains() -> set[str]:
    """Every news site of the training, extra and held-out CC-News pools (with and without www.)."""
    doms = set()
    for f in ("ccnews_train", "ccnews_train_extra", "ccnews_heldout"):
        doms |= set(pd.read_parquet(PROCESSED_DIR / "teacher_labels" / f"{f}.parquet", columns=["domain"])["domain"])
    return doms | {d[4:] for d in doms if d.startswith("www.")} | {"www." + d for d in doms}


def _init(max_pages: int = 0, out_dir: str | None = None) -> None:
    # Windows starts workers with "spawn": module globals must be passed in explicitly
    global _lid, _excluded, MAX_PAGES, OUT_DIR
    MAX_PAGES = max_pages
    if out_dir:
        OUT_DIR = Path(out_dir)
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    _lid = fasttext.load_model(str(LID_MODEL))
    _excluded = excluded_domains()


def detect_lang(text: str) -> tuple[str, float]:
    # low-level API: FastText.predict() breaks on NumPy >= 2 (np.array(copy=False))
    prob, label = _lid.f.predict(text[:2000].replace("\n", " "), 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), float(prob)


def walk(node):
    """Yield every dict in a JSON-LD tree (top level, lists, @graph, nested values)."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def types_of(obj: dict) -> set[str]:
    t = obj.get("@type")
    t = t if isinstance(t, list) else [t]
    return {str(x).lower() for x in t if x}


def as_strings(value) -> list[str]:
    """Normalise a JSON-LD value (str / list / dict with name) to a list of clean strings."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("name") or value.get("@value") or value.get("alternateName")
    if isinstance(value, list):
        return [s for v in value for s in as_strings(v)]
    s = htmllib.unescape(TAG.sub(" ", str(value))).strip()
    return [s] if s else []


def breadcrumb_names(obj: dict) -> list[str]:
    items = obj.get("itemListElement") or []
    items = items if isinstance(items, list) else [items]
    ranked = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item = it.get("item")
        name = it.get("name") or (item.get("name") if isinstance(item, dict) else None)
        pos = it.get("position", len(ranked))
        try:
            pos = float(pos)
        except (TypeError, ValueError):
            pos = len(ranked)
        for s in as_strings(name):
            ranked.append((pos, s))
    return [s for _, s in sorted(ranked, key=lambda x: x[0])]


def language_code(value) -> str | None:
    for s in as_strings(value):
        m = re.match(r"([A-Za-z]{2,3})(?:[-_].*)?$", s.strip())
        if m:
            return m.group(1).lower()
    return None


def parse_blocks(html: bytes) -> list:
    out = []
    for raw in LD_SCRIPT.findall(html):
        txt = raw.decode("utf8", "ignore").strip()
        if not txt:
            continue
        try:
            out.append(json.loads(txt, strict=False))
        except json.JSONDecodeError:
            try:  # common breakage: raw control characters / trailing commas
                out.append(json.loads(re.sub(r",\s*([}\]])", r"\1", re.sub(r"[\x00-\x1f]", " ", txt)), strict=False))
            except json.JSONDecodeError:
                continue
    return out


# ----------------------------------------------------------------------------- per WARC file
def process(path: str) -> dict:
    from warcio.archiveiterator import ArchiveIterator
    import trafilatura

    name = Path(path).name.replace(".warc.gz", "")
    shard = OUT_DIR / f"{name}.parquet"
    if shard.exists():
        return {"file": name, "status": "exists"}
    t0 = time.time()
    rows, seen = [], set()
    stats = {"html": 0, "with_jsonld": 0, "article": 0, "excluded_site": 0}
    with open(path, "rb") as fh:
        for rec in ArchiveIterator(fh):
            if rec.rec_type != "response" or not rec.http_headers:
                continue
            if rec.http_headers.get_statuscode() != "200" or "html" not in (rec.http_headers.get_header("Content-Type") or ""):
                continue
            stats["html"] += 1
            if MAX_PAGES and stats["html"] > MAX_PAGES:
                break
            html = rec.content_stream().read()
            blocks = parse_blocks(html)
            if not blocks:
                continue
            stats["with_jsonld"] += 1
            objs = list(walk(blocks))
            arts = [o for o in objs if types_of(o) & ARTICLE_TYPES]
            if not arts:
                continue
            stats["article"] += 1
            url = rec.rec_headers.get_header("WARC-Target-URI")
            domain = urlparse(url).netloc.lower()
            if domain in _excluded:
                stats["excluded_site"] += 1
                continue
            art = max(arts, key=lambda o: len(o))  # the richest article object on the page
            mep = art.get("mainEntityOfPage")
            mep_id = mep.get("@id") if isinstance(mep, dict) else mep
            canonical = next(iter(as_strings(art.get("url")) + as_strings(mep_id)), url)
            if canonical in seen or url in seen:
                continue
            seen.update({canonical, url})
            crumbs = [n for o in objs if "breadcrumblist" in types_of(o) for n in breadcrumb_names(o)]

            body = " ".join(as_strings(art.get("articleBody")))
            headline = next(iter(as_strings(art.get("headline"))), "")
            text_source = "articleBody"
            if len(body.split()) < 30:  # missing or truncated teaser: fall back to the page's main text
                try:
                    body = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
                except Exception:  # noqa: BLE001 - malformed HTML
                    body = ""
                text_source = "trafilatura"
            body = body.strip()
            if not body:
                continue
            lang = language_code(art.get("inLanguage"))
            lid_lang, lid_conf = detect_lang(body)
            m = HTML_LANG.search(html[:4000])
            rows.append({
                "url": url, "canonical": canonical, "domain": domain,
                "types": ",".join(sorted(types_of(art))),
                "headline": headline,
                "article_section": json.dumps(as_strings(art.get("articleSection")), ensure_ascii=False),
                "breadcrumb": json.dumps(crumbs, ensure_ascii=False),
                "keywords": json.dumps(as_strings(art.get("keywords"))[:30], ensure_ascii=False),
                "in_language": lang, "lid_lang": lid_lang, "lid_conf": round(lid_conf, 3),
                "lang": lang or lid_lang, "lang_source": "inLanguage" if lang else "fasttext",
                "html_lang": m.group(1).decode().lower() if m else None,
                "date_published": next(iter(as_strings(art.get("datePublished"))), None),
                "date_modified": next(iter(as_strings(art.get("dateModified"))), None),
                "warc_date": rec.rec_headers.get_header("WARC-Date"),
                "author": json.dumps(as_strings(art.get("author"))[:5], ensure_ascii=False),
                "publisher": next(iter(as_strings(art.get("publisher"))), None),
                "is_free": str(art.get("isAccessibleForFree")) if art.get("isAccessibleForFree") is not None else None,
                "text": body, "text_source": text_source, "n_words": len(body.split()), "warc": name,
            })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(shard, index=False)
    return {"file": name, "kept": len(rows), **stats, "seconds": round(time.time() - t0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-files", type=int, default=0, help="process only the first N WARC files (smoke run)")
    parser.add_argument("--max-pages", type=int, default=0, help="stop after N HTML pages per file (smoke run)")
    args = parser.parse_args()
    global OUT_DIR, MAX_PAGES
    if args.max_pages:
        MAX_PAGES, OUT_DIR = args.max_pages, OUT_DIR.parent / "smoke"
    files = sorted(str(p) for p in WARC_DIR.glob("*.warc.gz"))
    if args.limit_files:
        files = files[:args.limit_files]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} WARC files -> {OUT_DIR}", flush=True)
    log = open(OUT_DIR.parent / "extract_log.jsonl", "a", encoding="utf8")
    with Pool(args.workers, initializer=_init, initargs=(MAX_PAGES, str(OUT_DIR))) as pool:
        for i, res in enumerate(pool.imap_unordered(process, files), 1):
            res["done"] = f"{i}/{len(files)}"
            print(json.dumps(res), flush=True)
            log.write(json.dumps(res) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
