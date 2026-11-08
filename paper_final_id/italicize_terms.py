"""Wrap English technical terms in the Indonesian paper with \\en{...} (italic, as PUEBI asks for foreign words).

Protected and left untouched: math, equation/tabular/algorithm environments, \\cite/\\ref/\\label/\\url/\\texttt and
other command arguments that are not prose, text already inside \\emph{} or \\en{}, and the header block before
\\begin{abstract}. Proper names (StaticKD, fastText, Model2Vec, XLM-R, ...) are not in the list and stay upright.
Usage: env/Scripts/python.exe paper_final_id/italicize_terms.py paper_final_id/main_id.tex
"""
import re
import sys
from pathlib import Path

TERMS = [
    # longest first, so a phrase wins over its parts
    "site-cluster bootstrap", "cluster bootstrap", "two one-sided tests", "knowledge distillation",
    "deep averaging network", "multi-head self-attention", "self-attention", "sentence transformer",
    "sentence encoder", "multilayer perceptron", "rectified linear unit", "fully connected", "large language model",
    "central processing unit", "graphics processing unit", "random-access memory", "bag-of-embeddings",
    "bag-of-words", "bag of n-grams", "soft prediction", "soft label", "hard label", "token dropout", "weight decay",
    "dark knowledge", "ground truth", "deep learning", "grid search", "half-precision", "cross-entropy",
    "held-out set", "development set", "dev set", "test set", "test", "thread", "subset", "loss", "paywall", "grid", "fine-tuning", "real-time", "zero-shot",
    "hyperparameter", "transformer", "pipeline", "teacher", "student", "held-out", "agreement", "throughput",
    "embedding", "tokenizer", "ensemble", "bootstrap", "resample", "temperature", "baseline", "cascade",
    "benchmarking", "benchmark", "lookup", "folding", "pooling", "dropout", "encoder", "subword", "regularizer",
    "trainer", "offline", "streaming", "dataset", "batch", "epoch", "head", "seed", "noise", "split", "core",
]
TERM_RE = re.compile(r"(?<![\w\\-])(" + "|".join(re.escape(t) for t in TERMS) + r")(?![\w])", re.IGNORECASE)

ARG_CMDS = r"cite|ref|eqref|label|url|texttt|includegraphics|emph|en|bibliography|bibliographystyle|begin|end|" \
           r"markboth|history|doi|author|address|corresp|uppercase|authorrefmark|PARstart|todo|setlength|resizebox"
PROTECT = re.compile(
    r"\$[^$]*\$"                                                     # inline math
    r"|\\begin\{(equation|tabular|algorithmic|algorithm)\*?\}.*?\\end\{\1\*?\}"
    r"|\\(?:" + ARG_CMDS + r")\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})+"  # command with its arguments
    r"|\\[A-Za-z]+",                                                 # any other command name
    re.S)


def wrap(text: str) -> str:
    return TERM_RE.sub(lambda m: r"\en{" + m.group(1) + "}", text)


def main(path: str) -> None:
    p = Path(path)
    raw = p.read_bytes().decode("utf8")
    head, sep, body = raw.partition(r"\begin{abstract}")
    out, pos, n = [], 0, 0
    for m in PROTECT.finditer(body):
        out.append(wrap(body[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(wrap(body[pos:]))
    new_body = "".join(out).replace("Intel \\en{Core}", "Intel Core")  # product name, not a term
    n = new_body.count(r"\en{") - body.count(r"\en{")
    if r"\newcommand{\en}" not in head:
        head = head.replace(r"\begin{document}", "\\newcommand{\\en}[1]{\\textit{#1}}\n\n\\begin{document}", 1)
    p.write_bytes((head + sep + new_body).encode("utf8"))
    print(f"wrapped {n} term occurrences")


if __name__ == "__main__":
    main(sys.argv[1])
