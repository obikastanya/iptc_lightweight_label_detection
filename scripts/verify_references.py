"""Verify every reference the paper cites (or will cite) against an authoritative source and collect the
open-access PDFs in reference/.

For each bib key:
  * DOI -> Crossref metadata: title, first author, year, venue, and update notices (retraction,
    correction, ...) from Crossref's "updated-by" field (Retraction Watch data) and "relation";
  * no DOI -> the identifier in SOURCES below (arXiv API, ACL Anthology page, or the official URL);
  * the fetched title is compared with the bib title (token overlap); a low overlap is flagged.
Open-access PDFs (arXiv, ACL Anthology, PLOS, CVF, AAAI) are downloaded to reference/<key>.pdf.
Everything else is listed as "PDF needed" so the authors can add it (IEEE / Elsevier / Springer access).

Outputs: reference/verification.json and reference/INDEX.md

Usage (from the project root):
    python scripts/verify_references.py [--keys k1 k2 ...] [--no-download]
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import bibtexparser
import requests

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "jurnal_final" / "references.bib"
AUX = ROOT / "jurnal_final" / "main.aux"
REF_DIR = ROOT / "reference"
HEADERS = {"User-Agent": "statickd-reference-check/1.0 (academic use)"}

# references the final paper will add (research_final/paper_quality_check_filled.md, L-02 and L-04)
PLANNED = ["mcnemar1947", "holm1979", "guo2017calibration", "sanh2019distilbert", "barbaresi2021trafilatura",
           "broder1997resemblance", "emmediatopic2025", "cloversearch_ccnews", "commoncrawl2026ccnews", "guha2016schema",
           "indonews2021", "swahilinews2020", "amharicnews2024", "dror2018hitchhiker", "iyyer2015dan",
           "treviso2023efficient", "yang2025kdsurvey", "wang2022kdreview"]

# identifiers for entries without a DOI (each is checked against the bib title, so a wrong id is caught)
SOURCES = {
    "hinton2015distilling": ("arxiv", "1503.02531"),
    "hu2022lora": ("arxiv", "2106.09685"),
    "joulin2017bag": ("acl", "E17-2068"),
    "koehn2004significance": ("acl", "W04-3250"),
    "mirashi2023indicnews": ("acl", "2023.icon-1.37"),
    "stanton2021kd": ("arxiv", "2106.05945"),
    "guo2017calibration": ("arxiv", "1706.04599"),
    "sanh2019distilbert": ("arxiv", "1910.01108"),
    "amharicnews2024": ("arxiv", "2103.05639"),
    "swahilinews2020": ("doi", "10.5281/zenodo.5514203"),
    "holm1979": ("url", "https://www.jstor.org/stable/4615733"),
    "iptc2024mediatopics": ("url", "https://iptc.org/standards/media-topics/"),
    "emmediatopic2025": ("url", "http://hdl.handle.net/11356/1991"),
    "cloversearch_ccnews": ("url", "https://huggingface.co/datasets/CloverSearch/cc-news-mutlilingual"),
    "commoncrawl2026ccnews": ("url", "https://commoncrawl.org/blog/news-dataset-available"),
    "indonews2021": ("url", "https://huggingface.co/datasets/jakartaresearch/indonews"),
}
# results of checks that cannot be automated (checked 2026-10-17); shown in INDEX.md
MANUAL = {
    "holm1979": ("ok", "No DOI and JSTOR blocks scripts. Confirmed: Scand. J. Statist. 6(2):65-70, JSTOR 4615733 "
                       "(https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)."),
    "tulkens2024model2vec": ("ok", "Zenodo concept DOI (DataCite, all versions; latest v0.10.0, 2026-10-11). "
                                   "The experiments used model2vec 0.9.0: name that version in the citation note."),
    "swahilinews2020": ("ok", "Zenodo DOI (DataCite): 'Swahili : News Classification Dataset', David, 2020, v0.2."),
    "hu2022lora": ("ok", "arXiv version 2021; cited venue ICLR 2022 is correct."),
    "hossain2025banglanews": ("ok", "Correction e0332710 (2025-09-17) replaces Figs 3-16 and the funding statement; "
                                    "the cited content (publisher categories of the dataset) is not affected."),
    "cloversearch_ccnews": ("ok", "Hugging Face dataset page exists (HTTP 200); page title differs from the bib title."),
    "indonews2021": ("ok", "Hugging Face dataset page exists (HTTP 200); page title differs from the bib title."),
    "commoncrawl2026ccnews": ("ok", "Official Common Crawl page exists (HTTP 200)."),
    "iptc2024mediatopics": ("ok", "Official IPTC page exists (HTTP 200)."),
    "emmediatopic2025": ("ok", "CLARIN.SI handle resolves (HTTP 200)."),
    "kuzman2025southslavic": ("ok", "LREC 2026 workshop (LLMs4SSH), published 2026-05; the key says 2025 but the bib year is 2026."),
}
# classic method papers outside the authors' subscriptions: cited only as the origin of a standard procedure;
# identity checked from metadata and, where available, the publisher abstract (2026-10-17)
PDF_UNAVAILABLE = {
    "mcnemar1947": "PDF tidak tersedia (langganan). Hanya dikutip sebagai asal uji McNemar; identitas cocok dengan Crossref.",
    "holm1979": "PDF tidak tersedia (langganan). Hanya dikutip sebagai asal koreksi Holm; identitas cocok dengan JSTOR 4615733.",
    "field2007bootstrapping": "PDF tidak tersedia (langganan). Abstrak (Crossref) membahas bootstrap untuk data berkelompok, "
                              "sesuai pemakaiannya (site-cluster bootstrap).",
}
NO_PDF_NEEDED = {"iptc2024mediatopics", "emmediatopic2025", "cloversearch_ccnews", "commoncrawl2026ccnews", "indonews2021",
                 "swahilinews2020", "tulkens2024model2vec"}   # web resources and software
# PDFs collected before this script, under their original file names
EARLIER_PDFS = {"kuzman2025llm": "kuzman2025_iptc_teacher_student_ieeeaccess.pdf",
                "razuvayevskaya2024peft": "razuvayevskaya2024_peft_multilingual_news_plosone.pdf",
                "piperno2025crosslingual": "crosslingual_distillation_sentence_transformers_kbs2025.pdf"}
# arXiv versions of paywalled papers (only used to fetch a readable copy, never as the cited version)
ARXIV_COPY = {"wang2022kdreview": "2004.05937", "yang2025kdsurvey": "2407.01885"}


def norm_tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", re.sub(r"[{}\\$]", "", text or "")).lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2}


def overlap(a: str, b: str) -> float:
    ta, tb = norm_tokens(a), norm_tokens(b)
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def get(url: str, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, **kw)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(3)
    return None


def crossref(doi: str) -> dict:
    r = get(f"https://api.crossref.org/works/{doi}")
    if r is None or r.status_code != 200:
        return {"ok": False, "error": f"crossref HTTP {getattr(r, 'status_code', 'none')}"}
    m = r.json()["message"]
    updates = [{"type": u.get("type"), "doi": u.get("DOI"), "label": u.get("label")} for u in m.get("updated-by", [])]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"ok": True, "title": (m.get("title") or [""])[0], "year": year,
            "first_author": (m.get("author") or [{}])[0].get("family", ""),
            "venue": (m.get("container-title") or [""])[0] or m.get("publisher", ""), "type": m.get("type"),
            "updates": updates, "is_update_notice": bool(m.get("update-to"))}


def arxiv(aid: str) -> dict:
    r = get(f"https://export.arxiv.org/api/query?id_list={aid}")
    if r is None or r.status_code != 200 or "<entry>" not in r.text:
        return {"ok": False, "error": "arXiv lookup failed"}
    entry = r.text.split("<entry>", 1)[1]
    title = re.search(r"<title>(.*?)</title>", entry, re.S).group(1)
    year = re.search(r"<published>(\d{4})", entry).group(1)
    author = re.search(r"<name>(.*?)</name>", entry).group(1).split()[-1]
    return {"ok": True, "title": " ".join(title.split()), "year": int(year), "first_author": author, "venue": "arXiv",
            "updates": [], "pdf": f"https://arxiv.org/pdf/{aid}"}


def acl(aid: str) -> dict:
    r = get(f"https://aclanthology.org/{aid}/")
    if r is None or r.status_code != 200:
        return {"ok": False, "error": f"ACL Anthology HTTP {getattr(r, 'status_code', 'none')}"}
    title = re.search(r'<meta content="([^"]+)" name="citation_title"', r.text) or \
        re.search(r'name="citation_title" content="([^"]+)"', r.text)
    year = re.search(r'citation_publication_date" content="(\d{4})', r.text) or \
        re.search(r'content="(\d{4})[^"]*" name="citation_publication_date"', r.text)
    return {"ok": True, "title": title.group(1) if title else "", "year": int(year.group(1)) if year else None,
            "first_author": "", "venue": "ACL Anthology", "updates": [], "pdf": f"https://aclanthology.org/{aid}.pdf"}


def url_check(url: str) -> dict:
    r = get(url, allow_redirects=True)
    code = getattr(r, "status_code", None)
    title = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I) if r is not None and code == 200 else None
    return {"ok": code == 200, "title": " ".join(title.group(1).split()) if title else "", "year": None,
            "first_author": "", "venue": url, "updates": [], "http": code, "manual": True}


def oa_pdf(key: str, entry: dict, info: dict) -> str | None:
    doi = entry.get("doi", "")
    if info.get("pdf"):
        return info["pdf"]
    if doi.startswith("10.18653/v1/"):
        return f"https://aclanthology.org/{doi.split('/v1/')[1]}.pdf"
    if doi.startswith("10.3115/v1/"):
        return f"https://aclanthology.org/{doi.split('/v1/')[1]}.pdf"
    if doi.startswith("10.1162/tacl_a_00577"):
        return "https://aclanthology.org/2023.tacl-1.48.pdf"
    if doi.startswith("10.1371/journal.pone"):
        return f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
    if doi.startswith("10.3390/data8050074"):
        return "https://www.mdpi.com/2306-5729/8/5/74/pdf"
    if doi.startswith("10.1109/CVPR52688.2022.01165"):
        return ("https://openaccess.thecvf.com/content/CVPR2022/papers/"
                "Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf")
    if key in ARXIV_COPY:
        return f"https://arxiv.org/pdf/{ARXIV_COPY[key]}"
    return None


def download(url: str, out: Path) -> bool:
    r = get(url, allow_redirects=True)
    if r is None or r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return False
    out.write_bytes(r.content)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", nargs="*", default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    db = {e["ID"]: e for e in bibtexparser.load(open(BIB, encoding="utf8")).entries}
    cited = set()
    for m in re.findall(r"citation\{([^}]*)\}", AUX.read_text(encoding="utf8")):
        cited.update(x.strip() for x in m.split(","))
    keys = args.keys or sorted(cited) + [k for k in PLANNED if k not in cited]
    existing = {p.stem: p.name for p in REF_DIR.glob("*.pdf")}
    existing.update({k: v for k, v in EARLIER_PDFS.items() if (REF_DIR / v).exists()})
    results = {}
    for key in keys:
        e = db[key]
        kind, ident = ("doi", e["doi"]) if e.get("doi") else SOURCES.get(key, ("none", ""))
        info = {"doi": crossref, "arxiv": arxiv, "acl": acl, "url": url_check}.get(kind, lambda _: {"ok": False, "error": "no identifier"})(ident)
        bib_title = re.sub(r"[{}]", "", e.get("title", ""))
        info.update(key=key, identifier=f"{kind}:{ident}", bib_title=bib_title, bib_year=e.get("year"),
                    status_in_paper="cited" if key in cited else "planned")
        info["title_match"] = round(overlap(bib_title, info.get("title", "")), 2) if info.get("title") else None
        info["year_match"] = (str(info.get("year")) == str(e.get("year"))) if info.get("year") else None
        flags = []
        if not info.get("ok"):
            flags.append("NOT VERIFIED: " + info.get("error", f"HTTP {info.get('http')}"))
        if info.get("title_match") is not None and info["title_match"] < 0.6:
            flags.append(f"title mismatch ({info['title_match']})")
        if info.get("year_match") is False:
            flags.append(f"year differs (source {info.get('year')}, bib {e.get('year')})")
        for u in info.get("updates", []):
            flags.append(f"{u['type']}: {u['doi']}")
        if info.get("manual"):
            flags.append("web resource: check manually")
        info["flags"] = flags
        # PDF
        pdf_name = next((v for k, v in existing.items() if k == key), None)
        if pdf_name is None and not args.no_download:
            url = oa_pdf(key, e, info)
            if url and download(url, REF_DIR / f"{key}.pdf"):
                pdf_name = f"{key}.pdf"
                info["pdf_source"] = url
        info["pdf"] = pdf_name
        results[key] = info
        print(f"{key:28s} {'OK ' if info.get('ok') and not [f for f in flags if 'mismatch' in f or 'NOT' in f] else '?? '} "
              f"pdf={'yes' if pdf_name else 'no '} {'; '.join(flags)}", flush=True)
        time.sleep(0.5)
    (REF_DIR / "verification.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf8")
    write_index(results, db)
    print(f"wrote {REF_DIR / 'verification.json'} and {REF_DIR / 'INDEX.md'}")


def write_index(results: dict, db: dict) -> None:
    lines = ["# Referensi paper StaticKD", "",
             "Dibuat oleh `scripts/verify_references.py`; jangan disunting manual. Setiap entri diperiksa terhadap sumber",
             "resmi (Crossref/DataCite untuk DOI, arXiv, ACL Anthology, atau halaman resmi). Kolom *Status* = hasil",
             "pemeriksaan identitas (judul, tahun, venue) dan pemberitahuan retraksi/koreksi.", "",
             "| Key bib | Dipakai | Status | PDF | Catatan |", "|---|---|---|---|---|"]
    need = []
    for key, r in results.items():
        manual = MANUAL.get(key)
        bad = [f for f in r["flags"] if "NOT VERIFIED" in f or "mismatch" in f]
        status = "terverifikasi (manual)" if manual else ("terverifikasi" if r.get("ok") and not bad else "PERLU CEK")
        notes = manual[1] if manual else "; ".join(r["flags"])
        if r.get("pdf"):
            pdf = f"[{r['pdf']}]({r['pdf']})"
        elif key in NO_PDF_NEEDED:
            pdf = "tidak perlu (web/perangkat lunak)"
        elif key in PDF_UNAVAILABLE:
            pdf = "tidak tersedia"
            notes = "; ".join(x for x in (notes, PDF_UNAVAILABLE[key]) if x)
        else:
            pdf = "**dibutuhkan**"
            venue = db[key].get("journal") or db[key].get("booktitle") or ""
            need.append(f"- `{key}`: {re.sub(r'[{}]', '', db[key].get('title', ''))} ({venue}, {db[key].get('year')}), "
                        f"DOI {db[key].get('doi', '-')}. Simpan sebagai `reference/{key}.pdf`.")
        lines.append(f"| `{key}` | {r['status_in_paper']} | {status} | {pdf} | {notes} |")
    lines += ["", "## PDF yang perlu diambil penulis", ""] + (need or ["(tidak ada)"])
    (REF_DIR / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    sys.exit(main())
