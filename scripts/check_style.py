"""Scan the final paper for the patterns listed in research_final/ai_text_criteria_to_be_avoided.txt.

The scan is a first pass: every hit still needs a human (or agent) judgement, because one occurrence of a
pattern is not wrong by itself; a pile-up is. Checks:
  1. word list (section 1 of the criteria file, Indonesian and English);
  2. a transition word opening more than one sentence in the same paragraph;
  3. "tidak hanya ... tetapi juga", "bukan sekadar ... melainkan", "not only ... but also";
  4. em-dashes and long dashes in prose;
  5. bold labels inside paragraphs (\\textbf{...} not in a list item or table);
  6. list environments (allowed only for RQ, contributions, procedures);
  7. generic paragraph closers, rhetorical questions, "sepengetahuan kami", LLM artefacts;
  8. sentence length spread per section (uniform length is a warning sign).

Usage (from the project root):
    python scripts/check_style.py [--file jurnal_final/main.tex]
"""
import argparse
import re
import statistics
from pathlib import Path

WORDS_ID = ["memainkan peran", "peran krusial", "peran penting", "menyoroti", "menggarisbawahi", "lanskap",
            "komprehensif", "holistik", "wawasan berharga", "membuka jalan", "secara signifikan", "inovatif",
            "mutakhir", "revolusioner", "sangat efektif", "sangat efisien", "mulus", "tangguh", "era digital",
            "perlu dicatat", "penting untuk dicatat", "patut dicatat"]
WORDS_EN = ["delve", "intricate", "pivotal", "underscore", "showcase", "leverag", "seamless", "robust", "comprehensive",
            "meticulous", "commendable", "notably", "multifaceted", "paramount", "landscape", "realm", "tapestry",
            "shed light", "pave the way", "testament", "harness", "unlock", "elevate", "crucial", "game-changer",
            "cutting-edge", "groundbreaking", "fast-paced", "worth noting"]
TRANSITIONS = ["Selain itu", "Lebih lanjut", "Dengan demikian", "Oleh karena itu", "Oleh sebab itu", "Namun demikian",
               "Di sisi lain", "Akan tetapi", "Furthermore", "Moreover", "Additionally", "Therefore", "However"]
CLOSERS = ["Secara keseluruhan", "Overall", "Hal ini menunjukkan pentingnya", "Kesimpulannya", "Singkatnya"]
ARTEFACTS = ["Certainly", "As an AI", "insert citation", "citation needed", "knowledge cutoff", "Berikut adalah", "XX"]


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(table\*?|figure\*?|tabular|equation)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\\(eng|emph|textit|texttt|url)\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\val\{[^}]*\}", "N", text)
    text = re.sub(r"\\(cite|ref|eqref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\$[^$]*\$", "X", text)
    return text


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() and not p.strip().startswith("\\Tab")]


def sentences(par: str) -> list[str]:
    par = re.sub(r"\\(begin|end)\{(itemize|enumerate)\}|\\item", " ", par)
    par = re.sub(r"\b(dkk|misalnya|mis|vs|No|Gbr|Fig)\.", r"\1<dot>", par)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", par) if len(s.strip()) > 3]


class Part:
    """One section of the single-file paper, scanned like a file."""

    def __init__(self, name: str, raw: str):
        self.name, self.raw = name, raw

    def read_text(self, encoding: str = "utf8") -> str:
        return self.raw


def split_paper(tex: str) -> list[Part]:
    """Abstract and every \\section of main.tex, up to the bibliography."""
    body = tex[tex.index("\\begin{abstract}"):tex.index("\\bibliographystyle")]
    abstract, rest = body.split("\\end{abstract}", 1)
    parts = [Part("abstract", abstract.replace("\\begin{abstract}", ""))]
    chunks = re.split(r"\\section\*?\{([^}]*)\}", rest)
    parts += [Part(chunks[i], chunks[i + 1]) for i in range(1, len(chunks) - 1, 2)]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="jurnal_final/main.tex")
    args = parser.parse_args()
    files = split_paper(Path(args.file).read_text(encoding="utf8"))
    total = 0
    for f in files:
        raw = f.read_text(encoding="utf8")
        text = strip_tex(raw)
        hits = []
        low = text.lower()
        for w in WORDS_ID + WORDS_EN:
            for m in re.finditer(re.escape(w.lower()), low):
                hits.append(f"kata '{w}': ...{text[max(0, m.start() - 40):m.end() + 40].strip()}...")
        for i, par in enumerate(paragraphs(text), 1):
            starts = [t for s in sentences(par) for t in TRANSITIONS if s.startswith(t)]
            if len(starts) > 1:
                hits.append(f"paragraf {i}: {len(starts)} kalimat diawali kata transisi {starts}")
            last = sentences(par)[-1] if sentences(par) else ""
            if any(last.startswith(c) for c in CLOSERS):
                hits.append(f"paragraf {i}: penutup generik: {last[:80]}")
        for pat in [r"tidak hanya\b.*?\btetapi juga", r"bukan sekadar\b.*?\bmelainkan", r"not only\b.*?\bbut also"]:
            hits += [f"pola '{pat}'" for _ in re.finditer(pat, low, flags=re.S)]
        hits += [f"em-dash: ...{text[max(0, m.start() - 30):m.end() + 30]}..." for m in re.finditer(r"—|---", text)]
        for line in raw.splitlines():
            if "\\textbf{" in line and not line.lstrip().startswith(("\\item", "%")) and "&" not in line:
                hits.append(f"label tebal di paragraf: {line.strip()[:90]}")
        n_lists = len(re.findall(r"\\begin\{(itemize|enumerate)\}", raw))
        if n_lists:
            hits.append(f"info: {n_lists} blok daftar (boleh untuk RQ/kontribusi/prosedur)")
        hits += [f"pertanyaan retoris: {s[:80]}" for p in paragraphs(text) for s in sentences(p) if s.endswith("?")]
        hits += [f"'sepengetahuan kami'" for _ in re.finditer("sepengetahuan kami", low)]
        hits += [f"artefak LLM '{a}'" for a in ARTEFACTS if a in text]
        hits += [f"karakter non-LaTeX '{c}'" for c in "“”‘’…" if c in raw]
        lens = [len(s.split()) for p in paragraphs(text) for s in sentences(p)]
        spread = f"kalimat: n={len(lens)}, median {statistics.median(lens):.0f} kata, min {min(lens)}, maks {max(lens)}, SD {statistics.pstdev(lens):.1f}" if lens else ""
        print(f"== {f.name}: {spread}")
        for h in hits:
            print("   -", h)
        total += len([h for h in hits if not h.startswith("info")])
    print(f"temuan (di luar info): {total}")


if __name__ == "__main__":
    main()
