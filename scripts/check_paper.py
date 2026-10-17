"""Audit the single-file paper jurnal_final/main.tex against the notebook output.

Checks:
  * main.tex is self-contained: no \\input / \\include of local files, no \\val / \\rows left;
  * every figure it includes exists in jurnal_final/figures/, and that folder holds nothing else;
  * the compiled log has no errors, undefined references or citations;
  * numbers: every key recorded by scripts/build_paper.py (results/final/paper_number_map.json) still has the
    value that the notebook now writes (results/final/paper_generated/numbers.tex), and that literal still
    occurs in main.tex. After re-running experiments, this lists the numbers to update by hand.

Usage (from the project root, after building and compiling the paper):
    python scripts/check_paper.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper import load_numbers  # noqa: E402

TEX = ROOT / "jurnal_final"
MAP = ROOT / "results" / "final" / "paper_number_map.json"
BS = chr(92)


def main() -> None:
    text = (TEX / "main.tex").read_text(encoding="utf8")
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())
    leftovers = [c for c in (BS + "input{", BS + "include{", BS + "val{", BS + "rows{") if c in body]
    print("self-contained:", "yes" if not leftovers else f"NO, found {leftovers}")

    figs = sorted(set(re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)))
    missing = [f for f in figs if not (TEX / f).exists()]
    extra = sorted(p.name for p in (TEX / "figures").iterdir() if f"figures/{p.name}" not in figs)
    print(f"figures: {len(figs)} included, missing {missing or 'none'}, unused files in figures/ {extra or 'none'}")

    log = TEX / "main.log"
    if log.exists():
        lines = log.read_text(encoding="latin-1").splitlines()
        errors = [l for l in lines if l.startswith("!")]
        undefined = [l for l in lines if "undefined" in l and ("Reference" in l or "Citation" in l)]
        print(f"LaTeX errors: {len(errors)}, undefined refs/cites: {len(undefined)}")
        for l in (errors + undefined)[:10]:
            print("   ", l)
        out = [l for l in lines if l.startswith("Output written")]
        print(out[-1] if out else "no PDF written")

    if MAP.exists():
        built = json.loads(MAP.read_text(encoding="utf8"))
        now = {k: re.sub(re.escape(BS) + r"eng\{([^{}]*)\}", r"\1", v) for k, v in load_numbers().items()}
        changed = {k: (v["value"], now.get(k)) for k, v in built.items() if now.get(k) != v["value"]}
        absent = [k for k, v in built.items() if v["value"] not in text]
        print(f"numbers: {len(built)} keys built into main.tex, changed in the notebook since: {len(changed)}")
        for k, (old, new) in sorted(changed.items()):
            print(f"   {k}: {old} -> {new}")
        if absent:
            print(f"   literal no longer found in main.tex (edited by hand?): {len(absent)} keys, e.g. {absent[:5]}")


if __name__ == "__main__":
    main()
