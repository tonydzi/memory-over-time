# -*- coding: utf-8 -*-
"""sanitize.py: pseudonymise the 15 real lead slugs in the as-of corpus before publication.

Machine: ZBOOKG8-2023PAL · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Input: data/hub-raw/zep-memory-20261103/{our_side_asof.json,our_side_episodes.jsonl,graphiti_*asof*.json} (private)
Output: data/asof/ (public). Every slug token (len>=3) is replaced case-insensitively by Lead-NN; the script
then fails (rc 1) if any original token survives in the output. The mapping itself is never written.
"""
import glob, json, re, sys
from pathlib import Path
K = Path(__file__).resolve().parents[1]
SRC = K / "data/hub-raw/zep-memory-20261103"; OUT = K / "data/asof"; OUT.mkdir(exist_ok=True)
eps = [json.loads(l) for l in (SRC / "our_side_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
slugs = []
for e in eps:
    s = re.match(r"Lead (\S+)", e["body"]).group(1)
    if s not in slugs:
        slugs.append(s)
pairs = []
for i, s in enumerate(slugs, 1):
    pseudo = f"lead-{i:02d}"
    pairs.append((re.compile(re.escape(s), re.I), pseudo))
    pairs.append((re.compile(r"\b" + r"[\s_-]+".join(map(re.escape, s.split("-"))) + r"\b", re.I), pseudo))
    for tok in s.split("-"):
        if len(tok) >= 3:
            pairs.append((re.compile(r"\b" + re.escape(tok) + r"\b", re.I), pseudo))
tokens = {t.lower() for s in slugs for t in s.split("-") if len(t) >= 3}

def clean(text):
    for rx, p in pairs:
        text = rx.sub(p, text)
    return text

files = [SRC / "our_side_asof.json", SRC / "our_side_episodes.jsonl"] + [Path(p) for p in sorted(glob.glob(str(SRC / "graphiti_*asof*.json")))]
for f in files:
    (OUT / f.name).write_text(clean(f.read_text(encoding="utf-8")), encoding="utf-8")
left = []
for f in OUT.iterdir():
    low = f.read_text(encoding="utf-8").lower()
    left += [(f.name, t) for t in tokens if re.search(r"\b" + re.escape(t) + r"\b", low)]
print("slugs", len(slugs), "files", len(files), "leftover", len(left))
sys.exit(1 if left else 0)
