"""Split jurnal_final/main.tex into paragraph blocks and extract the facts that a rewrite must keep.

For every block (abstract, paragraph, list item, figure, table, acknowledgment, ...) it records the
section, the line in main.tex, a short anchor (first words), and the locked facts:
  * numbers exactly as printed (0.545, 2,971, 91.1%, ...),
  * citations as printed numbers with their bib keys ([18] adelani2023masakhanews),
  * cross-references resolved from main.aux (Table IV, Fig. 3, Section V-A, (2)),
  * inline math, model or dataset identifiers in \\texttt, and emphasized terms.

Usage (from the project root, after compiling jurnal_final/main.tex):
    python scripts/rewrite_lock.py            # writes results/final/rewrite_blocks.json
The JSON is the input for research_final/paper_rewrite_kunci.md (Indonesian meaning + locked facts).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "jurnal_final" / "main.tex"
AUX = ROOT / "jurnal_final" / "main.aux"
OUT = ROOT / "results" / "final" / "rewrite_blocks.json"
BS = chr(92)


def labels_from_aux() -> tuple[dict, dict]:
    aux = AUX.read_text(encoding="latin-1")
    # section numbers are stored as {\mbox {III-D}}
    labels = {m.group(1): m.group(3) or m.group(2)
              for m in re.finditer(re.escape(BS) + r"newlabel\{([^}]+)\}\{\{(?:" + re.escape(BS) + r"mbox\s*\{([^}]*)\}|([^}]*))\}", aux)}
    cites = {m.group(1): int(m.group(2)) for m in re.finditer(re.escape(BS) + r"bibcite\{([^}]+)\}\{(\d+)\}", aux)}
    return labels, cites


def ref_name(label: str, num: str) -> str:
    if label.startswith("tab:"):
        return f"Table {num}"
    if label.startswith("fig:"):
        return f"Fig. {num}"
    if label.startswith("eq:"):
        return f"({num})"
    if label.startswith("alg:"):
        return f"Algorithm {num}"
    return f"Section {num}"


def plain(t: str) -> str:
    t = re.sub(re.escape(BS) + r"(emph|textit|textbf|texttt|url)\{([^{}]*)\}", r"\2", t)
    t = re.sub(re.escape(BS) + r"(cite|ref|eqref|label)\{[^}]*\}", " ", t)
    t = t.replace("~", " ").replace("--", "-").replace(BS + "%", "%")
    return re.sub(r"\s+", " ", t).strip()


def facts(text: str, labels: dict, cites: dict) -> dict:
    no_math = re.sub(r"\$[^$]*\$", " ", text)
    p = plain(no_math)
    numbers = re.findall(r"(?<![\w.])[-+]?\d(?:[\d,]*\d)?(?:\.\d+)?(?:%| ?ms| ?MB| ?GB)?(?![\w])", p)
    cite_list = []
    for m in re.finditer(re.escape(BS) + r"cite\{([^}]*)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            cite_list.append(f"[{cites.get(k, '?')}] {k}")
    refs = []
    for m in re.finditer(re.escape(BS) + r"(eqref|ref)\{([^}]*)\}", text):
        refs.append(ref_name(m.group(2), labels.get(m.group(2), "?")))
    return {
        "numbers": list(dict.fromkeys(n.strip() for n in numbers)),
        "citations": list(dict.fromkeys(cite_list)),
        "refs": list(dict.fromkeys(refs)),
        "math": list(dict.fromkeys(m.group(0) for m in re.finditer(r"\$[^$]*\$", text))),
        "identifiers": list(dict.fromkeys(re.findall(re.escape(BS) + r"(?:texttt|url)\{([^}]*)\}", text))),
        "terms": list(dict.fromkeys(re.findall(re.escape(BS) + r"emph\{([^}]*)\}", text))),
    }


def main() -> None:
    labels, cites = labels_from_aux()
    lines = TEX.read_text(encoding="utf8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(BS + "begin{document}"))
    end = next(i for i, l in enumerate(lines) if l.startswith(BS + "bibliographystyle"))
    blocks, section, subsection = [], "Front matter", ""
    counters: dict = {}
    i = start + 1

    def add(kind: str, line: int, text: str, extra: dict | None = None) -> None:
        key = (section, subsection)
        counters[key] = counters.get(key, 0) + 1
        b = {"kind": kind, "section": section, "subsection": subsection, "n": counters[key], "line": line + 1,
             "anchor": " ".join(plain(re.sub(r"\$[^$]*\$", "X", text)).split()[:7]), "text": text}
        b.update(facts(text, labels, cites))
        if extra:
            b.update(extra)
        blocks.append(b)

    while i < end:
        l = lines[i]
        s = l.strip()
        if not s or s.startswith("%") or s.startswith(BS + "maketitle") or s.startswith(BS + "IEEEpeerreviewmaketitle") \
                or s.startswith(BS + "label{"):
            i += 1
            continue
        if s.startswith(BS + "markboth"):  # running header: two lines, layout only
            i += 2 if s.rstrip().endswith("%") else 1
            continue
        m = re.match(re.escape(BS) + r"(section|subsection)\*?\{(.*)\}", s)
        if m:
            if m.group(1) == "section":
                section, subsection = m.group(2), ""
            else:
                subsection = m.group(2)
            i += 1
            continue
        env = re.match(re.escape(BS) + r"begin\{([a-zA-Z*]+)\}", s)
        if env:
            name = env.group(1)
            j = i
            while not lines[j].strip().startswith(BS + "end{" + name + "}"):
                j += 1
            body = "\n".join(lines[i: j + 1])
            if name in ("figure", "figure*", "table", "table*", "algorithm"):
                cap = re.search(re.escape(BS) + r"caption\{(.*?)\}\s*\n" + re.escape(BS) + "label", body, re.S)
                lab = re.search(re.escape(BS) + r"label\{([^}]*)\}", body)
                label = lab.group(1) if lab else ""
                kind = "algorithm" if name == "algorithm" else "table" if name.startswith("table") else "figure"
                add(kind, i, cap.group(1) if cap else "",
                    {"label": label, "number": ref_name(label, labels.get(label, "?")) if label else ""})
            elif name in ("itemize", "enumerate"):
                items = re.split(re.escape(BS) + r"item\s", body)[1:]
                for it in items:
                    it = re.sub(re.escape(BS) + r"end\{" + re.escape(name) + r"\}\s*$", "", it.strip())
                    add("item", i, it.strip())
            elif name in ("equation", "align"):
                lab = re.search(re.escape(BS) + r"label\{([^}]*)\}", body)
                label = lab.group(1) if lab else ""
                add("equation", i, body, {"label": label, "number": ref_name(label, labels.get(label, "?"))})
            elif name == "abstract":
                add("abstract", i, "\n".join(lines[i + 1: j]))
            elif name == "IEEEkeywords":
                add("keywords", i, "\n".join(lines[i + 1: j]))
            i = j + 1
            continue
        if s.startswith(BS + "title{") or s.startswith(BS + "author{") or s.startswith(BS + "bibliography"):
            j = i
            depth = s.count("{") - s.count("}")
            while depth > 0:
                j += 1
                depth += lines[j].count("{") - lines[j].count("}")
            add("title" if s.startswith(BS + "title") else "author", i, "\n".join(lines[i: j + 1]))
            i = j + 1
            continue
        if s.startswith(BS) and re.match(re.escape(BS) + r"Tab[A-Z]", s):
            i += 1
            continue
        j = i
        stop = (BS + "begin{", BS + "section", BS + "subsection", BS + "label{")
        while j + 1 < end and lines[j + 1].strip() and not lines[j + 1].strip().startswith(stop):
            j += 1
        add("paragraph", i, "\n".join(lines[i: j + 1]))
        i = j + 1
    OUT.write_text(json.dumps(blocks, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"wrote {OUT}: {len(blocks)} blocks")
    kinds: dict = {}
    for b in blocks:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    print(kinds)


if __name__ == "__main__":
    main()
