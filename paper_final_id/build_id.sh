#!/usr/bin/env bash
# Compile the Indonesian version paper_final_id/main_id.tex (IEEE Access class); print page count and LaTeX problems.
# Figures and references are read from ../paper_final/. Usage: bash paper_final_id/build_id.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# ieeeaccess.cls and its fonts are licensed and not in git; copy them from the local template folder
TPL=paper_writing/template_ieeeaccess/ACCESS_latex_template_20260513
cp "$TPL"/{ieeeaccess.cls,spotcolor.sty,logo.png,notaglinelogo.png,bullet.png} "$TPL"/t1* paper_final_id/
cd paper_final_id
rm -f main_id.aux main_id.bbl
(pdflatex -interaction=nonstopmode main_id.tex; bibtex main_id; pdflatex -interaction=nonstopmode main_id.tex; pdflatex -interaction=nonstopmode main_id.tex) >/dev/null 2>&1
echo "== main_id"; grep -n "Output written\|^! \|Reference .* undefined\|Citation .* undefined" main_id.log | head -20
