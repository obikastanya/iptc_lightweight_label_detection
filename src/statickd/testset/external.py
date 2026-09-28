"""Download public news-topic datasets in extra languages for the unified IPTC test set.

Raw files go to data/raw/testset_external/<source>/ unchanged. Hugging Face datasets are fetched
through their Parquet export (no dataset scripts are executed); L3Cube-IndicNews comes from its
GitHub release; MN-DS from Zenodo.

| source         | languages                          | labels                          | licence      |
|----------------|------------------------------------|---------------------------------|--------------|
| masakhanews    | 16 African languages (+ en, fr)    | human-annotated, 7 topics       | CC BY-NC 4.0 |
| l3cube_indic   | 10 Indic languages (LDC, full docs)| publisher URL category          | CC BY 4.0    |
| bangla_plos    | Bengali                            | publisher category              | see HF card  |
| amharic_news   | Amharic (press.et)                 | publisher category              | CC BY 4.0    |
| swahili_news   | Swahili                            | publisher category              | CC BY 4.0    |
| indonews       | Indonesian (Liputan6)              | publisher category              | CC BY 4.0    |
| thucnews       | Chinese (Sina, 10-class subset)    | publisher channel               | see HF card  |
| mnds           | English                            | human IPTC Media Topic level 1  | CC BY 4.0    |

Usage (from the project root, with src/ on PYTHONPATH):
    python -m statickd.testset.external
"""
from pathlib import Path

from statickd.config import RAW_DIR  # noqa: E402

import requests  # noqa: E402

OUT = RAW_DIR / "testset_external"
HF_SOURCES = {
    "masakhanews": "mteb/masakhanews",
    "bangla_plos": "kawsarahmd/bangla-news-category-plos-one",
    "amharic_news": "rasyosef/amharic-news-category-classification",
    "swahili_news": "community-datasets/swahili_news",
    "indonews": "jakartaresearch/indonews",
    "thucnews": "seamew/THUCNews",
}
L3CUBE = ["Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", "Kannada", "Odia", "Malayalam", "Punjabi"]
L3CUBE_FILE = {"Bengali": "Bengal"}
MNDS_URL = "https://zenodo.org/api/records/7394851/files/MN-DS-news-classification.csv/content"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  {dest.relative_to(OUT)} ({dest.stat().st_size / 2**20:.1f} MB)", flush=True)


def hf_parquet(name: str, repo: str) -> None:
    files = requests.get(f"https://huggingface.co/api/datasets/{repo}/parquet", timeout=60).json()
    if not isinstance(files, dict) or not all(isinstance(v, dict) for v in files.values()):
        # no Parquet export (script-based repo): take the data files committed to the repo as they are
        tree = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", timeout=60).json()
        for t in tree:
            if t["path"].endswith((".arrow", ".parquet", ".csv", ".jsonl", ".json")) and t["path"] != "dataset_info.json":
                fetch(f"https://huggingface.co/datasets/{repo}/resolve/main/{t['path']}", OUT / name / "raw" / t["path"])
        return
    # {config: {split: [urls]}}
    for config, splits in files.items():
        for split, urls in splits.items():
            for i, url in enumerate(urls):
                fetch(url, OUT / name / config / f"{split}-{i:03d}.parquet")


def main() -> None:
    for name, repo in HF_SOURCES.items():
        print(f"{name} <- {repo}", flush=True)
        hf_parquet(name, repo)
    print("l3cube_indic <- github l3cube-pune/indic-nlp (LDC test + valid)", flush=True)
    base = "https://raw.githubusercontent.com/l3cube-pune/indic-nlp/main/L3Cube-IndicNews"
    for lang in L3CUBE:
        stem = L3CUBE_FILE.get(lang, lang)
        for split in ("Test", "Valid"):
            fetch(f"{base}/{lang}/LDC/{stem}_LDC_{split}.csv", OUT / "l3cube_indic" / lang / f"{split.lower()}.csv")
    print("mnds <- zenodo 7394851", flush=True)
    fetch(MNDS_URL, OUT / "mnds" / "MN-DS-news-classification.csv")


if __name__ == "__main__":
    main()
