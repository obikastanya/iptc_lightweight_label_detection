"""Convert paper_final_id/main_id.tex into an editable Word file (main_id.docx) for paraphrasing.

Pandoc does not know the ieeeaccess class, so a cleaned copy of the LaTeX is made first:
  * \\cite, \\ref and \\eqref become the numbers of the compiled PDF (read from main_id.bbl and main_id.aux);
  * class-only commands (\\history, \\doi, \\markboth, \\PARstart, \\resizebox, biographies, ...) are removed;
  * figures are converted from PDF to PNG so Word can show them;
  * the reference list from main_id.bbl is placed where \\bibliography stood.
Compile the LaTeX first (bash paper_final_id/build_id.sh), then run from the project root:
    env/Scripts/python.exe paper_final_id/tex_to_docx.py
"""
import re
import subprocess
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
FIG_SRC = HERE.parent / "paper_final"
BUILD = HERE / "docx_build"
TEX, AUX, BBL, OUT = HERE / "main_id.tex", HERE / "main_id.aux", HERE / "main_id.bbl", HERE / "main_id.docx"


def braced(text: str, start: int) -> int:
    """Index just after the brace group that opens at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def drop_command(text: str, name: str, nargs: int, keep: int | None = None) -> str:
    """Remove \\name{..}..{..}; with keep=k the k-th argument (0-based) stays in place of the command."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    while (m := pat.search(text)):
        pos, args = m.end(), []
        for _ in range(nargs):
            while text[pos] in " \n":
                pos += 1
            end = braced(text, pos)
            args.append(text[pos + 1:end - 1])
            pos = end
        text = text[:m.start()] + (args[keep] if keep is not None else "") + text[pos:]
    return text


def main() -> None:
    tex = TEX.read_text(encoding="utf8")
    body = tex[tex.index(r"\begin{document}"):]

    labels = {k: re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", v)
              for k, v in re.findall(r"\\newlabel\{([^}]+)\}\{\{((?:[^{}]|\{[^{}]*\})*)\}", AUX.read_text(encoding="utf8"))}
    bbl = BBL.read_text(encoding="utf8")
    cite_no = {k: i + 1 for i, k in enumerate(re.findall(r"\\bibitem\{([^}]+)\}", bbl))}

    body = re.sub(r"~?\\cite\{([^}]+)\}",
                  lambda m: " " + ", ".join(f"[{n}]" for n in sorted(cite_no[k.strip()] for k in m.group(1).split(","))), body)
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels[m.group(1)]})", body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], body)

    # \en{} is defined in the preamble, which pandoc does not see; without this pandoc drops every English term
    body = re.sub(r"\\en\{", r"\\textit{", body)

    # class-only commands
    for name, n in (("history", 1), ("doi", 1), ("markboth", 2), ("corresp", 1), ("tfootnote", 1)):
        keep = 0 if name in ("corresp", "tfootnote") else None
        body = drop_command(body, name, n, keep)
    body = re.sub(r"\\address\[\d\]", r"\\par ", body)
    body = re.sub(r"\\authorrefmark\{\d\}", "", body).replace("} AND \\uppercase{", "} dan \\uppercase{")
    body = drop_command(body, "uppercase", 1, 0)
    body = re.sub(r"\\PARstart\{(\w)\}\{(\w+)\}", r"\1\2", body)
    body = re.sub(r"\\titlepgskip=-?\d+pt|\\EOD", "", body)
    body = re.sub(r"\\begin\{IEEEbiographynophoto\}.*?\\end\{IEEEbiographynophoto\}", "", body, flags=re.S)
    body = re.sub(r"\\todo\{([^{}]*)\}", r"[TODO: \1]", body).replace(r"\bibliographystyle{IEEEtran}", "")
    body = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{%?", r"\\begingroup", body)
    body = re.sub(r"\\end\{tabular\}\}", r"\\end{tabular}\\endgroup", body)
    body = body.replace("{@{}l}", "{l}")  # pandoc fails to parse a tabular whose \multicolumn spec has @{}
    body = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", body)  # pandoc prints "(lr)2-3" as text
    body = body.replace(r"\begin{keywords}", r"\paragraph{Kata kunci.}").replace(r"\end{keywords}", "")
    body = re.sub(r"\\(footnotesize|scriptsize|centering|maketitle)\b|\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)

    # caption numbers as in the PDF ("Tabel 3. ...", "Gambar 1. ...")
    def number_caption(m: re.Match) -> str:
        env, block = m.group(1), m.group(0)
        label = re.search(r"\\label\{([^}]+)\}", block)
        if not label:
            return block
        prefix = ("Tabel " if env.startswith("table") else "Gambar ") + labels[label.group(1)] + ". "
        return block.replace(r"\caption{", r"\caption{" + prefix, 1)

    body = re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", body)  # pandoc drops captions of starred floats
    body = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", number_caption, body, flags=re.S)

    # figures: PDF -> PNG
    BUILD.mkdir(exist_ok=True)
    for fig in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))):
        src = FIG_SRC / fig if (FIG_SRC / fig).exists() else HERE / fig  # same order as \graphicspath in main_id.tex
        png = BUILD / (Path(fig).stem + ".png")
        if not png.exists() or png.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(src), str(png.with_suffix(""))], check=True)
        body = body.replace("{" + fig + "}", "{" + png.as_posix() + "}")
    body = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics[width=6in]", body)

    # reference list from the compiled bibliography
    refs = bbl[bbl.index(r"\bibitem"):bbl.index(r"\end{thebibliography}")]
    refs = re.sub(r"\\bibitem\{([^}]+)\}", lambda m: f"\n\n[{cite_no[m.group(1)]}] ", refs)
    refs = re.sub(r"\\BIBentry\w+|\\newblock", " ", refs)
    body = re.sub(r"\\bibliography\{[^}]*\}", lambda _: "\\section*{Referensi}\n" + refs, body)

    preamble = "\\documentclass{article}\n\\usepackage{amsmath,amssymb,graphicx,booktabs,multirow,url}\n"
    clean = BUILD / "main_id_clean.tex"
    clean.write_text(preamble + body, encoding="utf8")
    pypandoc.convert_file(str(clean), "docx", outputfile=str(OUT), extra_args=["--resource-path", str(BUILD), "-M", "abstract-title=Abstrak"])
    print("written", OUT)


if __name__ == "__main__":
    main()
