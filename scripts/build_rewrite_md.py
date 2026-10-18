"""Build research_final/paper_rewrite.md: a plain-text worksheet for rewriting jurnal_final/main.tex paragraph by paragraph.

Each block shows
  Current     - the paragraph as readable plain text (citations as [n], references as "Table IV", math as τ = 1)
  Makna       - its meaning in Indonesian (technical terms stay English)
  Locked      - numbers, citations with authors, and references the rewrite must keep
  Paraphrase  - plain text written by the author
Tables, figures and equations are one-line placeholders.

Re-running keeps every Paraphrase that already has text (matched by block ID). The Indonesian meaning is checked
against the paragraph (same numbers, none missing or extra; same citations); problems stop the build.

Usage (from the project root):
    python scripts/rewrite_lock.py && python scripts/build_rewrite_md.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rewrite_lock import labels_from_aux, ref_name  # noqa: E402

BLOCKS = ROOT / "results" / "final" / "rewrite_blocks.json"
TRANS = ROOT / "results" / "final" / "rewrite_translations_id.json"
BIB = ROOT / "jurnal_final" / "references.bib"
OUT = ROOT / "research_final" / "paper_rewrite.md"
BS = chr(92)
EMPTY = "(tulis di sini)"

PREFIX = {"Introduction": "PEN", "Related Work": "PT", "Methodology": "MET", "Results": "HAS", "Discussion": "PEM",
          "Limitations": "BTS", "Conclusion": "KES", "Code, Model, and Data Availability": "KTS", "Acknowledgment": "UCP"}
FRONT = {"title": "JDL", "author": "PNL", "abstract": "ABS", "keywords": "KK"}
FRONT_NAME = {"title": "Title", "author": "Authors", "abstract": "Abstract", "keywords": "Index Terms"}
SECTION_ID = {"Front matter": "Judul, penulis, abstrak, kata kunci", "Introduction": "I. Introduction",
              "Related Work": "II. Related Work", "Methodology": "III. Methodology", "Results": "IV. Results",
              "Discussion": "V. Discussion", "Limitations": "VI. Limitations", "Conclusion": "VII. Conclusion",
              "Code, Model, and Data Availability": "Code, Model, and Data Availability", "Acknowledgment": "Acknowledgment"}
SYMBOLS = {"tau": "τ", "theta": "θ", "rho": "ρ", "eta": "η", "Delta": "Δ", "pm": "±", "ge": "≥", "le": "≤",
           "times": "×", "rightarrow": "→", "cdot": "·", "in": "∈", "dots": "…", "sum": "Σ", "top": "ᵀ"}
ACCENTS = {("v", "s"): "š", ("'", "c"): "ć", ("'", "e"): "é", ("v", "c"): "č", ("v", "z"): "ž"}


def bib_short_names() -> dict:
    """bib key -> 'Surname YEAR', 'A & B YEAR' or 'A et al. YEAR'."""
    names = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", BIB.read_text(encoding="utf8"), re.S):
        key, body = m.group(1), m.group(2)
        a = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", body, re.S)
        y = re.search(r"year\s*=\s*[{\"]?(\d{4})", body)
        year = y.group(1) if y else "n.d."
        if not a:
            names[key] = f"{key} {year}"
            continue
        people = [p.strip() for p in re.split(r"\s+and\s+", re.sub(r"\s+", " ", a.group(1))) if p.strip()]

        def surname(p: str) -> str:
            p = plain_latex(p).replace("{", "").replace("}", "").strip()
            return p.split(",")[0].strip() if "," in p else p.split(" ")[-1]
        if len(people) == 1:
            who = surname(people[0])
        elif len(people) == 2:
            who = f"{surname(people[0])} & {surname(people[1])}"
        else:
            who = f"{surname(people[0])} et al."
        names[key] = f"{who} {year}"
    return names


def plain_latex(t: str) -> str:
    """Readable plain text from a LaTeX snippet (no \\cite or \\ref handling; see to_plain)."""
    for (acc, ch), out in ACCENTS.items():
        t = t.replace(BS + acc + "{" + ch + "}", out).replace(BS + acc + ch, out)
    return t


def to_plain(t: str, labels: dict, cites: dict) -> str:
    t = re.sub(re.escape(BS) + r"IEEEPARstart\{([^}]*)\}\{([^}]*)\}", r"\1\2", t)
    t = re.sub(re.escape(BS) + r"cite\{([^}]*)\}", lambda m: ", ".join(f"[{cites.get(k.strip(), '?')}]" for k in m.group(1).split(",")), t)
    t = re.sub(re.escape(BS) + r"eqref\{([^}]*)\}", lambda m: ref_name(m.group(1), labels.get(m.group(1), "?")), t)
    t = re.sub(re.escape(BS) + r"ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), t)
    t = re.sub(re.escape(BS) + r"todo\{([^}]*)\}", r"[TODO: \1]", t)
    t = re.sub(re.escape(BS) + r"thanks\{", "(", t)
    t = re.sub(re.escape(BS) + r"(title|author)\{", "", t)
    t = re.sub(re.escape(BS) + r"([a-zA-Z]+)(?![a-zA-Z])", lambda m: SYMBOLS.get(m.group(1), m.group(0)), t)
    for _ in range(3):  # nested formatting
        t = re.sub(re.escape(BS) + r"(emph|textbf|textit|texttt|url|mathrm|mathbf|mathcal|text)\{([^{}]*)\}", r"\2", t)
    t = re.sub(re.escape(BS) + r"mathbb\{R\}", "ℝ", t)
    t = re.sub(re.escape(BS) + r"frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(re.escape(BS) + r"bar\{([^{}]*)\}", r"\1̄", t)
    t = re.sub(re.escape(BS) + r"([a-zA-Z]+)", lambda m: SYMBOLS.get(m.group(1), m.group(0)), t)
    t = plain_latex(t)
    t = t.replace(BS + "%", "%").replace(BS + "{", "{").replace(BS + "}", "}").replace(BS + ",", " ").replace(BS + "!", "")
    t = t.replace("$-$", "−").replace("$+$", "+").replace("$<$", "< ").replace("$>$", "> ").replace("$", "")
    t = re.sub(r"\^\{([^{}]*)\}", r"^\1", t)
    t = re.sub(r"_\{([^{}]*)\}", r"_\1", t)
    t = t.replace("---", "—").replace("--", "–").replace("~", " ").replace("%\n", "\n")
    t = re.sub(r"(?<=\S)%(?=\s*$)", "", t, flags=re.M)  # line-end comment markers
    t = re.sub(r"\}+\s*$", "", t.strip()) if t.strip().endswith("}}") else t
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", " ", t)).strip()


def id_name(ref: str) -> str:
    if ref.startswith("("):
        return "Persamaan " + ref
    return (ref.replace("Table ", "Tabel ").replace("Fig. ", "Gambar ").replace("Section ", "Bagian ")
            .replace("Algorithm ", "Algoritma "))


def bare_numbers(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\b(RQ|Bagian|Tabel|Gambar|Persamaan|Algoritma)\s*[\dIVX-]+[A-Z]?\b", " ", text)
    return [n.lstrip("+-") for n in re.findall(r"(?<![\w.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?(?![\w])", text)]


def block_ids(blocks: list) -> list[str]:
    ids, count = [], {}
    for b in blocks:
        if b["section"] == "Front matter":
            ids.append(FRONT[b["kind"]])
        elif b["kind"] in ("table", "figure", "equation", "algorithm"):
            ids.append(b["number"])
        else:
            p = PREFIX[b["section"]]
            count[p] = count.get(p, 0) + 1
            ids.append(f"{p}-{count[p]:02d}")
    return ids


def existing_paraphrases() -> dict:
    """Paraphrases already written in the worksheet: text after **Paraphrase:** up to the next block."""
    if not OUT.exists():
        return {}
    found = {}
    # a paraphrase ends at the next block, section heading, separator or table/figure/equation placeholder
    end = r"(?=^### |^## |^---\s*$|^\[(?:TABEL|GAMBAR|PERSAMAAN|ALGORITMA) |\Z)"
    for m in re.finditer(r"^### (\S+)[^\n]*\n(.*?)" + end, OUT.read_text(encoding="utf8"), re.S | re.M):
        p = re.search(r"\*\*Paraphrase:\*\*\s*\n(.*)", m.group(2), re.S)
        if p:
            text = re.sub(r"^```\w*\s*|\s*```$", "", p.group(1).strip()).strip()
            if text and text != EMPTY:
                found[m.group(1)] = text
    return found


def main() -> None:
    labels, cites = labels_from_aux()
    short = bib_short_names()
    blocks = json.loads(BLOCKS.read_text(encoding="utf8"))
    trans = json.loads(TRANS.read_text(encoding="utf8"))
    kept = existing_paraphrases()
    problems, out, current = [], [HEADER], None

    def fill(text: str) -> str:
        text = re.sub(r"\{cite:([^}]+)\}", lambda m: f"[{cites.get(m.group(1), '?')}]", text)
        text = re.sub(r"\{ref:([^}]+)\}", lambda m: id_name(ref_name(m.group(1), labels.get(m.group(1), "?"))), text)
        return re.sub(r"[Pp]ersamaan Persamaan", "Persamaan", text)

    for bid, b in zip(block_ids(blocks), blocks):
        if b["section"] != current:
            current = b["section"]
            out.append(f"\n## {SECTION_ID[current]}\n")
        meaning = trans.get(bid)
        if meaning is None:
            problems.append(f"{bid}: no Indonesian text")
            continue
        meaning = fill(meaning)
        note = ""
        m = re.search(r"\s*\((Catatan[^)]*(?:\([^)]*\)[^)]*)*)\)\s*$", meaning)
        if m:
            note, meaning = m.group(1), meaning[: m.start()].rstrip()

        if b["kind"] not in ("title", "author", "equation"):
            locked = [re.sub(r"(ms|MB|GB)$", "", n.split(" ")[0].rstrip("%").lstrip("+-")) for n in b["numbers"]]
            got = [n.rstrip("%") for n in bare_numbers(meaning)]
            in_math = set(re.findall(r"\d+(?:\.\d+)?", " ".join(b["math"])))
            missing = sorted(set(locked) - set(got))
            extra = sorted(set(got) - set(locked) - in_math - {"1", "2", "3", "4", "5", "6"})
            if missing:
                problems.append(f"{bid}: number(s) missing in the Indonesian text: {missing}")
            if extra:
                problems.append(f"{bid}: number(s) not in the paragraph: {extra}")
            want = {c.split(" ")[0] for c in b["citations"]}
            have = set(re.findall(r"\[\d+\]", meaning))
            if want != have:
                problems.append(f"{bid}: citations differ: paragraph {sorted(want)} vs text {sorted(have)}")

        if b["kind"] in ("table", "figure", "equation", "algorithm"):
            what = {"table": "TABEL", "figure": "GAMBAR", "equation": "PERSAMAAN", "algorithm": "ALGORITMA"}[b["kind"]]
            out.append(f"\n[{what} {id_name(b['number']).split(' ', 1)[1]} — placeholder] {meaning}\n")
            continue
        head = FRONT_NAME.get(b["kind"], b["subsection"] or "")
        out.append(f"\n---\n\n### {bid}" + (f" · {head}" if head else "") + f" · main.tex baris {b['line']}\n\n")
        current_text = to_plain(b["text"], labels, cites)
        if b["kind"] == "title":
            current_text = current_text.rstrip("}").strip()
        out.append(f"**Current:**\n{current_text}\n\n")
        out.append(f"**Makna:**\n{meaning}\n\n")
        if note:
            out.append(f"**Catatan:** {note}\n\n")
        out.append(f"**Locked:** {locked_line(b, labels, cites, short)}\n\n")
        out.append(f"**Paraphrase:**\n{kept.get(bid, EMPTY)}\n")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s); fix results/final/rewrite_translations_id.json")
    text = "".join(out)
    leftover = sorted(set(re.findall(re.escape(BS) + r"[a-zA-Z]+", text)))
    if leftover:
        print("LaTeX left in the worksheet:", leftover)
    OUT.write_text(text, encoding="utf8")
    print(f"wrote {OUT} ({len(blocks)} blocks, {len(kept)} paraphrase(s) kept)")


def locked_line(b: dict, labels: dict, cites: dict, short: dict) -> str:
    parts = []
    if b["numbers"]:
        parts.append("angka " + ", ".join(b["numbers"]))
    keys = []
    for k in re.findall(re.escape(BS) + r"cite\{([^}]*)\}", b["text"]):
        keys += [x.strip() for x in k.split(",")]
    if keys:
        parts.append("sitasi " + "; ".join(f"[{cites.get(k, '?')}] {short.get(k, k)}" for k in dict.fromkeys(keys)))
    refs = [ref_name(l, labels.get(l, "?")) for l in re.findall(re.escape(BS) + r"(?:eq)?ref\{([^}]*)\}", b["text"])]
    if refs:
        parts.append("rujukan " + ", ".join(dict.fromkeys(refs)))
    if b["math"]:
        parts.append("simbol " + ", ".join(dict.fromkeys(to_plain(m, labels, cites) for m in b["math"])))
    if b["identifiers"]:
        parts.append("nama " + ", ".join(b["identifiers"]))
    return " · ".join(parts) if parts else "—"


HEADER = """# Paper Rewrite (StaticKD)

Lembar kerja untuk menulis ulang naskah paragraf demi paragraf. Untuk setiap blok:

1. Baca **Current** (teks paragraf sekarang) dan **Makna** (artinya dalam bahasa Indonesia).
2. Tulis versi Anda di bawah **Paraphrase:** (ganti tulisan "(tulis di sini)"). Cukup teks biasa; sitasi ditulis seperti di Current, misalnya [2], dan rujukan seperti "Section IV-D" atau "Table IV".
3. Pastikan semua yang ada di **Locked** muncul lagi di tulisan Anda: angka sama persis, nomor sitasi tetap menempel pada klaim yang sama.

Aturan singkat: mudah dipahami dan benar; gaya impersonal, tanpa subjek orang (*I, you, we, they*); istilah teknis tetap bahasa Inggris; definisikan singkatan saat pertama muncul; hindari pola di [ai_text_criteria_to_be_avoided.txt](ai_text_criteria_to_be_avoided.txt). Tabel, gambar, dan persamaan hanya placeholder.

Membangun ulang file ini (`python scripts/rewrite_lock.py && python scripts/build_rewrite_md.py`) tidak menghapus Paraphrase yang sudah diisi.
"""


if __name__ == "__main__":
    main()
