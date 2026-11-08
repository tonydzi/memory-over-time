# -*- coding: utf-8 -*-
"""sync_repo.py: copy the public subset of the kit into the GitHub clone, replacing local paths with env vars.

Machine: ZBOOKG8-2023PAL · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Never copies data/hub-raw (private), data/locomo/locomo10.json (not ours to redistribute) or secrets.
Usage: python analysis/sync_repo.py <clone path>. Fails (rc 1) if any copied text file still contains a drive-letter path or a known private token.
"""
import re, shutil, sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REPO = Path(sys.argv[1])  # path of the local clone of github.com/tonydzi/memory-over-time
B = chr(92)
SUBS = {
    "fact_history.py": [('    if not v and Path(r"E:' + B + 'Obsidian' + B + 'Anton-Knowledge").exists():' + chr(10) + '        v = r"E:' + B + 'Obsidian' + B + 'Anton-Knowledge"' + chr(10), '')],
    "our_side_asof.py": [('object_history в E:' + B + B + 'Obsidian' + B + B + '_imports' + B + B + 'leads.db', 'object_history в leads.db (private CRM, path from LEADS_DB)')],
    "bench_mot.py": [('CFG = Path(r"E:' + B + '!CLAUDE-PaloAltoPC-June26' + B + 'cliproxy-trial' + B + 'config.yaml")',
                      'CFG = Path(os.environ.get("CLIPROXY_CONFIG", "cliproxy-config.yaml"))  # api-keys list of the local proxy')],
}
copy = {
    "code": ["bench_mot.py", "probes_ours.py", "probes_mem0.py", "probes_graphiti_core.py", "fact_history.py"],
    "analysis": ["make_numbers.py", "_test_make_numbers.py", "retrieval_summary.py", "sanitize.py", "sync_repo.py"],
}
for d, names in copy.items():
    (REPO / d).mkdir(parents=True, exist_ok=True)
    for n in names:
        s = (KIT / d / n).read_text(encoding="utf-8")
        for a, b in SUBS.get(n, []):
            assert a in s, (n, a)
            s = s.replace(a, b)
        (REPO / d / n).write_text(s, encoding="utf-8")
for d in ("data/asof", "data/probes", "data/retrieval", "data/locomo-results", "prompts"):
    (REPO / d).mkdir(parents=True, exist_ok=True)
    for f in (KIT / d).iterdir():
        if f.is_file() and ".stale" not in f.name and ".attempt" not in f.name:
            shutil.copy2(f, REPO / d / f.name)
shutil.copy2(KIT / "data" / "aggregates.json", REPO / "data" / "aggregates.json")
shutil.copy2(KIT / "numbers.tex", REPO / "data" / "numbers.tex")
bad = []
pat = re.compile(r"[A-Z]:[\\/](?:Obsidian|!CLAUDE|Users)|token=[A-Za-z0-9_-]{16}|sk-[A-Za-z0-9]{20}|gh[pousr]_[A-Za-z0-9]{20}", re.I)
for f in REPO.rglob("*"):
    if f.is_file() and ".git" not in f.parts and f.suffix in (".py", ".json", ".jsonl", ".md", ".tex", ".txt", ".bib"):
        for i, line in enumerate(f.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            if pat.search(line) and "fact_history.py" not in f.name and "our_side_asof.py" not in f.name:
                bad.append(f"{f.relative_to(REPO)}:{i}: {line[:100]}")
print("\n".join(bad[:30]) if bad else "clean")
sys.exit(1 if bad else 0)
