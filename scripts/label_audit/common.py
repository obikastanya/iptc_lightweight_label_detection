"""Shared helpers for the article-level label audit (see results/label_audit/protocol.md)."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from statickd.config import LABELS, NO_SPACE_LANGS, truncate  # noqa: E402,F401

OUT = ROOT / "results" / "label_audit"
BATCH_DIR = OUT / "batches"
TEST_PATH = ROOT / "data" / "processed" / "testset_jsonld" / "unified_test.jsonl"
IPTC_PATH = ROOT / "src" / "statickd" / "testset" / "iptc_reference" / "mediatopic_en-GB.json"
AUDIT_ORIGINS = ("publisher_section_jsonld", "publisher_category")
STRATA = {"ccnews_jsonld": 200, "l3cube_indic": 70, "bangla_plos": 15, "indonews": 15}
MAX_ITEMS, MAX_CHARS = 50, 80_000
CONF = ("high", "medium", "low")


def definitions() -> dict[str, str]:
    """Official definitions of the 17 top-level topics, keyed by the repository label string."""
    concepts = json.loads(IPTC_PATH.read_text(encoding="utf-8"))
    by_name = {c["name"]: c["definition"] for c in concepts if c["level"] == 1 and not c["retired"]}
    by_name["politics"] = by_name.pop("politics and government")
    return {lab: by_name[lab] for lab in LABELS}


def label_block() -> str:
    lines = []
    for lab, d in definitions().items():
        if lab == "politics":
            lines.append(f'{lab}: (official IPTC name "politics and government") {d}')
        else:
            lines.append(f"{lab}: {d}")
    return "\n".join(lines)


def norm_label(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x == "politics and government":
        x = "politics"
    return x if x in LABELS else "__invalid__"


def read_jsonl(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wrap(text: str, width: int = 160) -> str:
    out = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False) or [""])
    return "\n".join(out)


def render_item(r: dict, extra: dict | None = None) -> str:
    parts = [f"===== ITEM {r['item_id']} | lang: {r['lang']} =====",
             "HEADLINE: " + (r.get("headline") or ""),
             "TEXT:", wrap(r.get("text") or "")]
    for k, v in (extra or {}).items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts) + "\n\n"


def make_batches(rows: list[dict], max_items: int = MAX_ITEMS, max_chars: int = MAX_CHARS) -> list[list[dict]]:
    batches, cur, size = [], [], 0
    for r in rows:
        n = len(r.get("text") or "") + len(r.get("headline") or "")
        if cur and (len(cur) >= max_items or size + n > max_chars):
            batches.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        batches.append(cur)
    return batches


def write_batch_files(prefix: str, batches, scratch: Path, extra_fn=None) -> list[Path]:
    """Render batches to plain text, in the scratch dir (for agents) and in BATCH_DIR (record)."""
    scratch.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, b in enumerate(batches, 1):
        body = "".join(render_item(r, extra_fn(r) if extra_fn else None) for r in b)
        name = f"{prefix}_{i:02d}.txt"
        (scratch / name).write_text(body, encoding="utf-8")
        (BATCH_DIR / name).write_text(body, encoding="utf-8")
        paths.append(scratch / name)
    return paths


def validate_labels(rows: list[dict], expected_ids: list[str]) -> tuple[dict, list[str]]:
    """Validate annotator output. Returns (valid rows by id, ids that must be re-run)."""
    good = {}
    for r in rows:
        iid = r.get("item_id")
        if iid not in expected_ids or iid in good:
            continue
        p = norm_label(r.get("primary"))
        s = norm_label(r.get("secondary"))
        c = str(r.get("confidence", "")).lower()
        u = r.get("unusable")
        if p in (None, "__invalid__") or s == "__invalid__" or c not in CONF or not isinstance(u, bool):
            continue
        if s == p:
            s = None
        good[iid] = {"item_id": iid, "primary": p, "secondary": s, "confidence": c,
                     "rationale": str(r.get("rationale", "")), "unusable": u}
    return good, [i for i in expected_ids if i not in good]
