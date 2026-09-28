"""Audit the LaTeX paper against the notebook output.

Checks: every \\val{key} used in the paper exists in generated/numbers.tex; every generated table and
figure referenced by the paper exists; the compiled log has no errors, undefined references or "??".

Usage (from the project root, after running the notebook and compiling the paper):
    python scripts/check_paper.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "jurnal_latex"
BS = chr(92)


def main() -> None:
    sources = [TEX / "main.tex", *sorted((TEX / "sections").glob("*.tex"))]
    raw = "".join(p.read_text(encoding="utf8") for p in sources)
    # drop LaTeX comments (an unescaped % to the end of the line)
    text = "\n".join(re.sub(r"(?<!" + re.escape(BS) + r")%.*$", "", line) for line in raw.splitlines())
    used = set(re.findall(re.escape(BS) + r"val\{([^}]+)\}", text))
    numbers = (TEX / "generated" / "numbers.tex").read_text(encoding="utf8")
    have = set(re.findall(r"val@(.+?)" + re.escape(BS) + "endcsname", numbers))
    missing = sorted(used - have)
    print(f"number keys used: {len(used)}, defined: {len(have)}, missing: {len(missing)}")
    for k in missing:
        print("   missing:", k)
    rows = set(re.findall(re.escape(BS) + r"rows\{([^}]+)\}", text))
    for r in sorted(rows):
        f = TEX / f"{r}.tex"
        n = len([l for l in f.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("%")]) if f.exists() else -1
        print(f"table {r}: {'MISSING' if n < 0 else f'{n} rows'}")
    for fig in sorted(set(re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", text))):
        print(f"figure {fig}: {'ok' if (TEX / fig).exists() else 'MISSING'}")
    log = TEX / "main.log"
    if log.exists():
        lines = log.read_text(encoding="latin-1").splitlines()
        errors = [l for l in lines if l.startswith("!")]
        undefined = [l for l in lines if "undefined" in l and ("Reference" in l or "Citation" in l)]
        print(f"LaTeX errors: {len(errors)}, undefined refs/cites: {len(undefined)}")
        for l in (errors + undefined)[:10]:
            print("   ", l)
        pages = [l for l in lines if "Output written" in l]
        print(pages[-1] if pages else "no PDF written")


if __name__ == "__main__":
    main()
