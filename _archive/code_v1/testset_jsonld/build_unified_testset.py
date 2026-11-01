"""Step 3: one unified multilingual IPTC test set = CC-NEWS JSON-LD silver pool + public datasets.

Every source goes through the same treatment as the CC-NEWS pool:
  * label  : the source's category name is mapped with iptc_section_map.map_section (IPTC-grounded,
             ambiguous names never mapped); MN-DS is already labelled with IPTC level-1 topics
  * quality: >= 75 words (>= 200 characters for zh/ja/th); declared language must agree with
             fastText lid.176 whenever lid.176 knows that language
  * dedup  : identical text, then MinHash near-duplicates across ALL sources
  * leakage: < 30% word 8-gram overlap with any training / evaluation document
  * sample : at most --per-cell articles per (language, label); human-annotated sources first,
             then publisher categories. There is no per-site cap: most (language, label) cells come from
             only a handful of sites, so a cap removes articles without adding site diversity. Site
             dependence is handled at evaluation time with a site-cluster bootstrap instead, and the
             report lists the site diversity of every language.

Label origin is kept per article: human_annotation (MasakhaNEWS, MN-DS), publisher_category
(L3Cube-IndicNews, Bangla PLOS, Amharic, Swahili, IndoNews) or publisher_section_jsonld (CC-NEWS).

Outputs in data/processed/testset_jsonld/: unified_pool.parquet, unified_test.jsonl, unified_report.md

Usage (from code/):
    python testset_jsonld/build_unified_testset.py --per-cell 50 --min-lang-docs 30
"""
import argparse
import glob
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, RAW_DIR, truncate_words  # noqa: E402
from testset_jsonld.build_testset import BASE, NO_SPACE, _tokens, gram_hashes, long_enough, training_grams  # noqa: E402
from testset_jsonld.iptc_section_map import LABELS, map_section, normalise  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet  # noqa: E402,F401

EXT = RAW_DIR / "testset_external"
MASAKHA_LANG = {"amh": "am", "eng": "en", "fra": "fr", "hau": "ha", "ibo": "ig", "lin": "ln", "lug": "lg", "orm": "om",
                "pcm": "pcm", "run": "rn", "sna": "sn", "som": "so", "swa": "sw", "tir": "ti", "xho": "xh", "yor": "yo"}
L3CUBE_LANG = {"Hindi": "hi", "Bengali": "bn", "Marathi": "mr", "Telugu": "te", "Tamil": "ta", "Gujarati": "gu",
               "Kannada": "kn", "Odia": "or", "Malayalam": "ml", "Punjabi": "pa"}
SWAHILI_NAMES = ["uchumi", "kitaifa", "michezo", "kimataifa", "burudani", "afya"]
MNDS_LABEL = {"religion and belief": "religion"}  # every other MN-DS level-1 name is already an IPTC label
HUMAN = {"masakhanews", "mnds"}
MAX_CANDIDATES = 300  # per (source, language, source label) before dedup: plenty for a 50-per-cell sample


def _row(source, lang, source_label, headline, body, url=None):
    return {"source": source, "lang": lang, "source_label": str(source_label), "headline": (headline or "").strip(),
            "body": (body or "").strip(), "url": url, "domain": urlparse(url).netloc.lower() if url else None}


def load_external() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(EXT / "masakhanews" / "*" / "*.parquet")):
        if Path(p).parent.name == "default":
            continue
        d = pd.read_parquet(p)
        lang = MASAKHA_LANG[Path(p).parent.name]
        rows += [_row("masakhanews", lang, r.category, r.headline, r.text, r.url) for r in d.itertuples()]
    for name, lang in L3CUBE_LANG.items():
        for split in ("test", "valid"):
            d = pd.read_csv(EXT / "l3cube_indic" / name / f"{split}.csv")
            lab = next(c for c in d.columns if c.lower() in ("labels", "label"))
            txt = next(c for c in d.columns if c.lower() == "text")
            rows += [_row("l3cube_indic", lang, l, "", t) for l, t in zip(d[lab], d[txt]) if isinstance(t, str)]
    d = pd.concat([pd.read_parquet(p) for p in glob.glob(str(EXT / "bangla_plos" / "default" / "*.parquet"))])
    rows += [_row("bangla_plos", "bn", c, "", t) for c, t in zip(d["Category"], d["Text"])]
    d = pd.read_parquet(glob.glob(str(EXT / "amharic_news" / "default" / "*.parquet"))[0])
    rows += [_row("amharic_news", "am", r.category, r.headline, r.article, r.link) for r in d.itertuples()]
    d = pd.concat([pd.read_parquet(p) for p in glob.glob(str(EXT / "swahili_news" / "*" / "*.parquet"))])
    rows += [_row("swahili_news", "sw", SWAHILI_NAMES[int(l)], "", t) for l, t in zip(d["label"], d["text"])]
    d = pd.concat([pd.read_parquet(p) for p in glob.glob(str(EXT / "indonews" / "default" / "*.parquet"))])
    rows += [_row("indonews", "id", l, "", t) for l, t in zip(d["label"], d["text"])]
    d = pd.read_csv(EXT / "mnds" / "MN-DS-news-classification.csv")
    rows += [_row("mnds", "en", c, t, b, u) for c, t, b, u in zip(d["category_level_1"], d["title"], d["content"], d["url"])]
    return pd.DataFrame(rows)


def iptc_label(source: str, source_label: str, lang: str) -> str | None:
    s = normalise(source_label)
    if source == "mnds":
        s = MNDS_LABEL.get(s, s)
        return s if s in LABELS else None
    s = re.sub(r"-news(-hindi)?$", "", s)  # L3Cube Hindi: "health-news-hindi", "crime-news-hindi"
    return map_section(s, lang)


def lid_model():
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    return fasttext.load_model(str(MODELS_DIR / "lid" / "lid.176.bin"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=50)
    parser.add_argument("--min-lang-docs", type=int, default=30, help="drop languages with fewer sampled docs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    funnel = []

    ext = load_external()
    funnel.append(("external: articles loaded", len(ext)))
    ext["label"] = [iptc_label(s, l, g) for s, l, g in zip(ext["source"], ext["source_label"], ext["lang"])]
    ext = ext[ext["label"].notna()].copy()
    funnel.append(("external: category unambiguously mapped to IPTC", len(ext)))
    ext = (ext.sample(frac=1.0, random_state=args.seed)
           .groupby(["source", "lang", "source_label"], group_keys=False).head(MAX_CANDIDATES))
    funnel.append((f"external: candidates (<= {MAX_CANDIDATES} per source/lang/category)", len(ext)))

    ext["text"] = [f"{h}\n\n{b}" if h and not b.startswith(h) else b for h, b in zip(ext["headline"], ext["body"])]
    ext = ext[[long_enough(b, l) for b, l in zip(ext["body"], ext["lang"])]]
    funnel.append(("external: long enough", len(ext)))
    lid = lid_model()
    known = {l.replace("__label__", "") for l in lid.get_labels()}
    pred = [lid.f.predict(b[:2000].replace("\n", " "), 1, 0.0, "strict")[0][1].replace("__label__", "") for b in ext["body"]]
    ext["lid_checked"] = ext["lang"].isin(known)
    ext = ext[~ext["lid_checked"] | (np.array(pred) == ext["lang"].to_numpy())]
    funnel.append(("external: language confirmed by fastText (when lid.176 knows it)", len(ext)))
    ext["label_origin"] = np.where(ext["source"].isin(HUMAN), "human_annotation", "publisher_category")

    silver = pd.read_parquet(BASE / "pool.parquet")
    silver = silver.assign(source="ccnews_jsonld", source_label=silver["section_names"], body=silver["text"],
                           label_origin="publisher_section_jsonld", lid_checked=True)
    funnel.append(("CC-NEWS 2024-2026 JSON-LD pool (already filtered)", len(silver)))
    cols = ["source", "label_origin", "lang", "label", "source_label", "headline", "text", "body", "url", "domain",
            "lid_checked"]
    df = pd.concat([ext[cols], silver[cols]], ignore_index=True)
    df["text"] = [t[:2000] if l in NO_SPACE else truncate_words(t) for t, l in zip(df["text"], df["lang"])]

    # dedup across all sources, human-labelled articles win
    df["prio"] = df["label_origin"].map({"human_annotation": 0, "publisher_category": 1, "publisher_section_jsonld": 2})
    df = df.sample(frac=1.0, random_state=args.seed).sort_values("prio", kind="stable")
    df["text_hash"] = [hashlib.md5(re.sub(r"\W+", "", t.lower())[:2000].encode()).hexdigest() for t in df["body"]]
    df = df.drop_duplicates("text_hash")
    from datasketch import MinHash, MinHashLSH
    lsh, keep = MinHashLSH(threshold=0.7, num_perm=64), []
    for idx, t, l in zip(df.index, df["body"], df["lang"]):
        toks = _tokens(t, l)
        n = 10 if l in NO_SPACE else 5
        m = MinHash(num_perm=64, seed=1)
        m.update_batch([" ".join(toks[i:i + n]).encode("utf8") for i in range(max(1, len(toks) - n + 1))])
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            keep.append(idx)
    df = df.loc[keep]
    funnel.append(("all sources: no exact / near duplicates", len(df)))

    train = training_grams()
    overlap = []
    for t, l in zip(df["body"], df["lang"]):
        g = np.unique(gram_hashes(t, l))
        pos = np.searchsorted(train, g).clip(max=len(train) - 1) if len(g) else None
        overlap.append(float((train[pos] == g).mean()) if len(g) else 0.0)
    df["train_overlap"] = overlap
    removed = df[df["train_overlap"] >= 0.3]["source"].value_counts().to_dict()
    df = df[df["train_overlap"] < 0.3]
    funnel.append((f"all sources: < 30% n-gram overlap with training/eval data (removed {removed})", len(df)))
    pool = df.drop(columns=["prio", "text_hash", "body"]).reset_index(drop=True)
    pool.to_parquet(BASE / "unified_pool.parquet", index=False)

    picked = []
    for (lang, label), g in pool.groupby(["lang", "label"]):
        g = g.sample(frac=1.0, random_state=args.seed)
        g = g.assign(prio=g["label_origin"].map({"human_annotation": 0, "publisher_category": 1,
                                                 "publisher_section_jsonld": 2})).sort_values("prio", kind="stable")
        picked.append(g.sort_values("prio", kind="stable").head(args.per_cell))
    test = pd.concat(picked).drop(columns="prio")
    sizes = test["lang"].value_counts()
    test = test[test["lang"].isin(sizes[sizes >= args.min_lang_docs].index)]
    test = test.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test.insert(0, "doc_id", [f"u{i:05d}" for i in range(len(test))])
    test["label_id"] = test["label"].map({l: i for i, l in enumerate(LABELS)})
    test.to_json(BASE / "unified_test.jsonl", orient="records", lines=True, force_ascii=False)
    funnel.append((f"unified test set (<= {args.per_cell}/cell, languages with >= {args.min_lang_docs} docs)", len(test)))
    write_report(funnel, pool, test)


def write_report(funnel, pool, test) -> None:
    short = {l: l.split(",")[0].split(" and ")[0][:9] for l in LABELS}
    lines = ["# Unified multilingual IPTC test set", "", "## Funnel", "", "| Step | Articles |", "|---|---|"]
    lines += [f"| {name} | {n:,} |" for name, n in funnel]
    lines += ["", f"Test set: {len(test):,} articles, {test['lang'].nunique()} languages, "
                  f"{test['label'].nunique()} labels", "", "## Test set by source", "",
              "| source | label origin | articles | languages |", "|---|---|---|---|"]
    for (src, org), g in test.groupby(["source", "label_origin"]):
        lines.append(f"| {src} | {org} | {len(g):,} | {g['lang'].nunique()} |")
    lines += ["", "## Test set: language x label", "",
              "| lang | " + " | ".join(short[l] for l in LABELS) + " | total | labels |", "|---" * (len(LABELS) + 3) + "|"]
    ct = pd.crosstab(test["lang"], test["label"]).reindex(columns=LABELS, fill_value=0)
    for lang, row in ct.assign(total=ct.sum(axis=1)).sort_values("total", ascending=False).iterrows():
        lines.append(f"| {lang} | " + " | ".join(str(int(row[l])) for l in LABELS) +
                     f" | {int(row['total'])} | {int((row[LABELS] > 0).sum())} |")
    # cluster = site when known, otherwise the source dataset (curated sets without URLs)
    cluster = test["domain"].fillna("").mask(lambda d: d.eq(""), "src:" + test["source"])
    lines += ["", "## Site diversity per language", "",
              "Cluster = site, or the source dataset when a source has no URLs. Effective clusters = "
              "1 / sum(share^2); top share = fraction of the language's articles from its largest cluster.", "",
              "| lang | articles | clusters | effective | top share | largest cluster |", "|---|---|---|---|---|---|"]
    for lang, c in cluster.groupby(test["lang"]):
        share = c.value_counts(normalize=True)
        lines.append(f"| {lang} | {len(c)} | {len(share)} | {1 / (share ** 2).sum():.1f} | "
                     f"{share.iloc[0]:.2f} | {share.index[0]} |")
    lines += ["", "## Test set: articles per label", ""]
    lines += [f"- {l}: {int((test['label'] == l).sum())}" for l in LABELS]
    (BASE / "unified_report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
