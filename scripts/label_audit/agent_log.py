"""Write results/label_audit/agent_tool_log.csv: every tool call of the audit subagents (blindness
check). Reads the Claude Code subagent transcripts of the given session directory.

Usage: python agent_log.py --transcripts <...>/subagents
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    args = ap.parse_args()
    rows = []
    for f in sorted(Path(args.transcripts).glob("agent-*.jsonl")):
        calls = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line).get("message", {})
            except json.JSONDecodeError:
                continue
            if isinstance(m, dict) and isinstance(m.get("content"), list):
                for c in m["content"]:
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        inp = c.get("input", {})
                        target = inp.get("file_path") or inp.get("path") or inp.get("command") or ""
                        calls.append((c["name"], str(target), str(inp.get("pattern", "")) if c["name"] == "Grep" else ""))
        first = next((t for n, t, _ in calls if n == "Read"), "")
        if "scratchpad" not in first or "\\la\\" not in first.replace("/", "\\"):
            continue  # not an audit subagent
        batch = Path(first.replace("\\", "/")).stem
        for i, (n, t, p) in enumerate(calls, 1):
            rows.append({"agent": f.stem, "batch": batch, "call": i, "tool": n,
                         "target": Path(t.replace("\\", "/")).name if t else "", "grep_pattern": p,
                         "outside_own_files": batch not in Path(t.replace("\\", "/")).stem})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "agent_tool_log.csv", index=False, encoding="utf-8")
    print(df.groupby("batch")["tool"].agg(lambda x: ",".join(x)).to_string())
    print("calls touching other files:", int(df["outside_own_files"].sum()))


if __name__ == "__main__":
    main()
