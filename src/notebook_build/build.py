"""Assemble notebooks/StaticKD_IPTC_final.ipynb from part*.py and optionally execute it.

    python src/notebook_build/build.py [--parts part1 part2] [--execute] [--out notebooks/x.ipynb]
"""
import argparse
import importlib
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

parser = argparse.ArgumentParser()
parser.add_argument("--parts", nargs="*", default=sorted(p.stem for p in HERE.glob("part*.py")))
parser.add_argument("--execute", action="store_true")
parser.add_argument("--out", default=str(ROOT / "notebooks" / "StaticKD_IPTC_final.ipynb"))
parser.add_argument("--timeout", type=int, default=3600)
args = parser.parse_args()

cells = []
for part in args.parts:
    cells += importlib.import_module(part).CELLS
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
if args.execute:
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, out)
else:
    nbformat.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
