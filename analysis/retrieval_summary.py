# -*- coding: utf-8 -*-
"""retrieval_summary.py: aggregate recall@k of the 03.11 frozen retrieval bench (40 private notes, 24 answerable).
Machine: ZBOOKG8-2023PAL · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Recomputes R@1/5/10 from the per-question rank lists and gold support sets; publishes only aggregates and hashes
(note ids are private vault slugs)."""
import json, sys
from pathlib import Path
K = Path(__file__).resolve().parents[1]; RB = K / "data/hub-raw/real-bench"; MC = K / "data/hub-raw/memory-compare-20261103"
gold = json.loads((RB / "gold.json").read_text(encoding="utf-8"))["questions"]
sup = {q["id"]: set(q["support"]) for q in gold if q.get("support")}
src = {"Mem0 OSS 2.2.1 (dense, infer=False)": MC / "mem0/run-a/result.json",
       "Anima connectome (Jina RTX/MATH)": RB / "ranks-jina-rtx-math.json",
       "Our SGM (dense + cross-encoder)": RB / "ranks-sgm-vector.json",
       "SQLite FTS5": RB / "ranks-sqlite-fts5.json", "BM25": RB / "ranks-bm25.json"}
out = {"n_answerable": len(sup), "gold_sha256": None, "systems": {}}
for name, p in src.items():
    d = json.loads(p.read_text(encoding="utf-8"))
    out["gold_sha256"] = out["gold_sha256"] or d.get("gold_sha256")
    pq = {r["id"]: r["retrieved"] for r in d["per_question"]}
    rec = {f"r@{k}": sum(bool(sup[q] & set(pq.get(q, [])[:k])) for q in sup) for k in (1, 5, 10)}
    out["systems"][name] = {**rec, "reported_overall": d.get("overall"), "gold_sha256": d.get("gold_sha256")}
abl = json.loads((MC / "claude-review/ablation2.json").read_text(encoding="utf-8"))
pq = {r["id"]: r["dense"] for r in abl}
out["systems"]["Our SGM, dense only (ablation)"] = {**{f"r@{k}": sum(bool(sup[q] & set(pq.get(q, [])[:k])) for q in sup) for k in (1, 5, 10)},
                                                    "reported_overall": None, "gold_sha256": None, "source": "claude-review/ablation2.json"}
(K / "data/retrieval").mkdir(exist_ok=True)
(K / "data/retrieval/retrieval_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for n, v in out["systems"].items():
    print(n, v["r@1"], v["r@5"], v["r@10"], "/", len(sup), "gold", (v["gold_sha256"] or "")[:8])
