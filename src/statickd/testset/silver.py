"""Step 2: build the silver-labelled IPTC test set from the extracted JSON-LD articles.

Pipeline (every step is counted in the report):
  1. label: articleSection / breadcrumb names -> IPTC via iptc_section_map (unambiguous names only;
     conflicting names reject the article)
  2. quality: datePublished in 2024-01-01 .. crawl date, >= 75 words (>= 200 characters for
     zh/ja/th), not behind a paywall (isAccessibleForFree != false), language consistent
     (declared inLanguage must agree with fastText; without inLanguage fastText needs p >= 0.7)
  3. dedup: same canonical URL / identical text, then near-duplicates (MinHash LSH, Jaccard >= 0.7)
  4. contamination: drop articles sharing >= 30% of their word 8-grams with ANY document the
     teacher or the students saw (CC-News train/extra/held-out, EMMediaTopic, LLM test set)
  5. sample: per (language, label) at most --per-cell articles and at most --per-domain from one
     site, so no single outlet dominates a cell

Outputs in data/processed/testset_jsonld/:
  pool.parquet          all articles that passed 1-4
  silver_test.jsonl     the sampled test set (text = title + body, first 512 words)
  audit_sample.csv      random articles per label for a human spot check of the silver labels
  report.md             funnel, language x label counts, most frequent section names per label

Usage (from the project root, with src/ on PYTHONPATH):
    python -m statickd.testset.silver --per-cell 50 --per-domain 5 --min-lang-docs 100
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from statickd.config import MIN_WORDS, PROCESSED_DIR, truncate_words  # noqa: E402
from statickd.testset.iptc_map import LABELS, label_article, normalise  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = PROCESSED_DIR / "testset_jsonld"
NO_SPACE = {"zh", "ja", "th", "my", "km", "lo"}
LEGACY_LANG = {"in": "id", "iw": "he", "jw": "jv", "ji": "yi", "mo": "ro"}
DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ----------------------------------------------------------------------------- n-gram fingerprints
def _tokens(text: str, lang: str) -> list[str]:
    text = text.lower()
    return [c for c in text if not c.isspace()] if lang in NO_SPACE else re.findall(r"\w+", text)


def gram_hashes(text: str, lang: str) -> np.ndarray:
    """Hashes of word 8-grams (character 16-grams for scripts without spaces)."""
    n = 16 if lang in NO_SPACE else 8
    toks = _tokens(text, lang)
    if len(toks) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.array([hash(t) for t in toks], dtype=np.int64).view(np.uint64)
    out = np.zeros(len(toks) - n + 1, dtype=np.uint64)
    mul = np.uint64(0x100000001B3)
    for i in range(n):  # polynomial rolling combination (wraps mod 2^64)
        out = out * mul + h[i:len(toks) - n + 1 + i]
    return out


def training_grams() -> np.ndarray:
    """Sorted unique n-gram hashes of every document used for distillation or evaluation."""
    tl = PROCESSED_DIR / "teacher_labels"
    frames = [pd.read_parquet(tl / f"{f}.parquet", columns=["text", "lang"])
              for f in ("ccnews_train", "ccnews_train_extra", "ccnews_heldout", "emmediatopic")]
    llm = PROCESSED_DIR / "llm_test.jsonl"
    if llm.exists():
        frames.append(pd.read_json(llm, lines=True)[["text", "lang"]])
    docs = pd.concat(frames, ignore_index=True)
    parts = [gram_hashes(t, l) for t, l in zip(docs["text"], docs["lang"])]
    grams = np.unique(np.concatenate(parts))
    print(f"training fingerprint: {len(docs):,} docs, {len(grams):,} unique n-grams", flush=True)
    return grams


# ----------------------------------------------------------------------------- filters
def parse_date(s) -> str | None:
    m = DATE.search(str(s)) if s else None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def language(row) -> tuple[str | None, str]:
    """Declared inLanguage must agree with fastText (this also rejects mojibake from non-UTF-8 pages,
    whose garbled text fastText does not recognise); without inLanguage, fastText needs p >= 0.7."""
    lid = row["lid_lang"]
    if row["lang_source"] == "inLanguage":
        lang = LEGACY_LANG.get(row["in_language"], row["in_language"])
        if lid != lang:
            return None, "lang_conflict"
        return lang, "ok"
    if row["lid_conf"] < 0.7:
        return None, "lang_low_conf"
    return lid, "ok"


def long_enough(text: str, lang: str) -> bool:
    return len(text) >= 200 if lang in NO_SPACE else len(text.split()) >= MIN_WORDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=50, help="max articles per (language, label)")
    parser.add_argument("--per-domain", type=int, default=5, help="max articles per site within a cell")
    parser.add_argument("--min-lang-docs", type=int, default=100, help="drop languages with fewer sampled docs")
    parser.add_argument("--audit-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.concat([pd.read_parquet(p) for p in sorted((BASE / "extracted").glob("*.parquet"))], ignore_index=True)
    funnel = [("JSON-LD news articles from new sites", len(df))]

    # 1. label from section names
    sections = df["article_section"].map(json.loads)
    crumbs = df["breadcrumb"].map(json.loads)
    res = [label_article(s, c, l) for s, c, l in zip(sections, crumbs, df["lang"])]
    df["label"] = [r[0] for r in res]
    df["label_source"] = [r[1] for r in res]
    df["section_names"] = [json.dumps(s + c[1:], ensure_ascii=False) for s, c in zip(sections, crumbs)]
    funnel.append(("  with any section/breadcrumb name", int(((sections.map(len) + crumbs.map(len)) > 0).sum())))
    funnel.append(("  rejected: conflicting section names", int((df["label_source"] == "conflict").sum())))
    funnel.append(("  rejected: astrology / horoscope sections", int((df["label_source"] == "rejected_term").sum())))
    df = df[df["label"].notna()].copy()
    funnel.append(("1. confidently mapped to one IPTC label", len(df)))

    # 2. quality filters
    df["date"] = df["date_published"].map(parse_date)
    crawl = df["warc_date"].map(parse_date)
    df = df[df["date"].notna() & (df["date"] >= "2024-01-01") & (df["date"] <= crawl)]
    funnel.append(("2a. published 2024-01-01 .. crawl date", len(df)))
    df = df[~df["is_free"].astype(str).str.lower().isin({"false", "0"})]
    funnel.append(("2b. not behind a paywall", len(df)))
    lang_res = df.apply(language, axis=1)
    df["lang"] = [r[0] for r in lang_res]
    df = df[df["lang"].notna()]
    funnel.append(("2c. language consistent", len(df)))
    df["full_text"] = [f"{h}\n\n{t}" if h and not t.startswith(h) else t for h, t in zip(df["headline"], df["text"])]
    df = df[[long_enough(t, l) for t, l in zip(df["text"], df["lang"])]]
    funnel.append(("2d. long enough", len(df)))
    df["full_text"] = [t[:2000] if l in NO_SPACE else truncate_words(t) for t, l in zip(df["full_text"], df["lang"])]

    # 3. dedup: canonical URL, identical text, then MinHash near-duplicates
    df = df.sample(frac=1.0, random_state=args.seed).drop_duplicates("canonical")
    df["text_hash"] = [hashlib.md5(re.sub(r"\W+", "", t.lower())[:2000].encode()).hexdigest() for t in df["text"]]
    df = df.drop_duplicates("text_hash")
    funnel.append(("3a. unique URL and text", len(df)))
    from datasketch import MinHash, MinHashLSH
    lsh, keep = MinHashLSH(threshold=0.7, num_perm=64), []
    for idx, t, l in zip(df.index, df["text"], df["lang"]):
        toks = _tokens(t, l)
        n = 10 if l in NO_SPACE else 5
        m = MinHash(num_perm=64, seed=1)
        m.update_batch([" ".join(toks[i:i + n]).encode("utf8") for i in range(max(1, len(toks) - n + 1))])
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep.append(idx)
    df = df.loc[keep]
    funnel.append(("3b. no near-duplicates (MinHash J >= 0.7)", len(df)))

    # 4. contamination with the training / evaluation corpora
    train = training_grams()  # sorted (np.unique), so membership is a binary search per n-gram
    overlap = []
    for t, l in zip(df["text"], df["lang"]):
        g = np.unique(gram_hashes(t, l))
        if not len(g):
            overlap.append(0.0)
            continue
        pos = np.searchsorted(train, g).clip(max=len(train) - 1)
        overlap.append(float((train[pos] == g).mean()))
    df["train_overlap"] = overlap
    df = df[df["train_overlap"] < 0.3]
    funnel.append(("4. < 30% n-gram overlap with training/eval data", len(df)))

    cols = ["url", "canonical", "domain", "lang", "lang_source", "label", "label_source", "section_names",
            "headline", "full_text", "n_words", "date", "publisher", "text_source", "train_overlap", "warc"]
    pool = df[cols].rename(columns={"full_text": "text"}).reset_index(drop=True)
    pool.to_parquet(BASE / "pool.parquet", index=False)

    # 5. sample with per-cell and per-site caps
    picked = []
    for (lang, label), g in pool.groupby(["lang", "label"]):
        g = g.sample(frac=1.0, random_state=args.seed)
        g = g.groupby("domain", group_keys=False).head(args.per_domain)
        picked.append(g.head(args.per_cell))
    test = pd.concat(picked)
    lang_sizes = test["lang"].value_counts()
    test = test[test["lang"].isin(lang_sizes[lang_sizes >= args.min_lang_docs].index)]
    test = test.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test.insert(0, "doc_id", [f"js{i:05d}" for i in range(len(test))])
    test["label_id"] = test["label"].map({l: i for i, l in enumerate(LABELS)})
    test.to_json(BASE / "silver_test.jsonl", orient="records", lines=True, force_ascii=False)
    funnel.append((f"5. sampled test set (<= {args.per_cell}/cell, <= {args.per_domain}/site, "
                   f"languages with >= {args.min_lang_docs} docs)", len(test)))

    audit = test.sample(frac=1.0, random_state=args.seed).groupby("label").head(args.audit_per_label)
    audit = audit.sort_values(["label", "lang"])
    audit[["doc_id", "lang", "label", "section_names", "headline", "url"]].assign(
        text_start=audit["text"].str[:300], human_label="", correct="").to_csv(
        BASE / "audit_sample.csv", index=False, encoding="utf-8-sig")

    write_report(funnel, pool, test, args)


def write_report(funnel, pool, test, args) -> None:
    (BASE / "silver_funnel.json").write_text(json.dumps(funnel, indent=1), encoding="utf8")
    lines = ["# Silver IPTC test set from CC-NEWS 2024-2026 JSON-LD", "",
             "Labels come from publisher section names (articleSection / breadcrumb) mapped to IPTC with the "
             "high-precision table in `code/testset_jsonld/iptc_section_map.py`. They are silver labels: "
             "verify `audit_sample.csv` before treating them as gold.", "", "## Funnel", "",
             "| Step | Articles |", "|---|---|"]
    lines += [f"| {name} | {n:,} |" for name, n in funnel]
    lines += ["", f"Label source in test set: {test['label_source'].value_counts().to_dict()}",
              f"Language source in test set: {test['lang_source'].value_counts().to_dict()}",
              f"Sites in test set: {test['domain'].nunique():,}", "", "## Test set: language x label", ""]
    ct = pd.crosstab(test["lang"], test["label"]).reindex(columns=LABELS, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    short = {l: l.split(",")[0].split(" and ")[0][:10] for l in LABELS}
    lines += ["| lang | " + " | ".join(short[l] for l in LABELS) + " | total |",
              "|---" * (len(LABELS) + 2) + "|"]
    for lang, row in ct.sort_values("total", ascending=False).iterrows():
        lines.append(f"| {lang} | " + " | ".join(str(int(row[l])) for l in LABELS) + f" | {int(row['total'])} |")
    lines += ["", "## Pool: articles per label (before sampling)", ""]
    lines += [f"- {l}: {int((pool['label'] == l).sum()):,}" for l in LABELS]
    lines += ["", "## Most frequent section names per label (pool)", ""]
    for l in LABELS:
        names = pool.loc[pool["label"] == l, "section_names"].map(json.loads).explode().dropna().map(normalise)
        top = names.value_counts().head(12)
        lines.append(f"- **{l}**: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
    (BASE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
