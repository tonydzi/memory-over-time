# -*- coding: utf-8 -*-
"""make_numbers.py: every number in the paper comes from instrument JSON through this script.

Machine: ZBOOKG8-2023PAL · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Input  (relative to the kit):
  data/locomo-results/{ctx,ans,jud,ingest}_*.jsonl   LoCoMo temporal subset, bench_mot.py on the hub
  data/asof/*.json                                    30 as-of questions, pseudonymised (sanitize.py)
  data/probes/*.json                                  room-402 scenario receipts
  data/retrieval/retrieval_summary.json               03.11 retrieval bench, aggregate only
Output: data/aggregates.json and numbers.tex (\newcommand macros). Do not edit numbers.tex by hand.
Test: analysis/_test_make_numbers.py (Wilson CI values against known references, red on a broken formula).
рельса: локально, 0 LLM.
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
D = KIT / "data"
SYSTEMS = [("ours", "Time-stamped log (ours)"), ("mem0", "Mem0 OSS 2.2.1"),
           ("graphiti", "Graphiti 0.31.2"), ("full", "Full context")]


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def jl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()] if p.exists() else []


def pct(x):
    return f"{100 * x:.1f}"


def pci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{pct(p)}\\% [{pct(lo)}, {pct(hi)}]"


def locomo():
    R = D / "locomo-results"
    out = {}
    for s, _ in SYSTEMS:
        runs = {}
        for f in sorted(R.glob(f"jud_{s}_r*.jsonl")):
            rows = jl(f)
            k = sum(r["label"] == "CORRECT" for r in rows)
            runs[f.stem.split("_")[-1]] = {"k": k, "n": len(rows), "wilson": wilson(k, len(rows))}
        if not runs:
            continue
        accs = [v["k"] / v["n"] for v in runs.values() if v["n"]]
        K = sum(v["k"] for v in runs.values()); N = sum(v["n"] for v in runs.values())
        ent = {"runs": runs, "mean": st.mean(accs), "sd": st.stdev(accs) if len(accs) > 1 else 0.0,
               "min": min(accs), "max": max(accs), "pooled": {"k": K, "n": N, "wilson": wilson(K, N)}}
        ctx = jl(R / f"ctx_{s}.jsonl") if s in ("ours", "full") else [r for f in sorted(R.glob(f"ctx_{s}_r*.jsonl")) for r in jl(f)]
        ms = [r["retr_ms"] for r in ctx if "retr_ms" in r]
        if ms:
            ent["retr_ms_median"] = st.median(ms)
        if s == "ours":
            hits = [r["evidence_hit"] for r in ctx if "evidence_hit" in r]
            ent["evidence_hit"] = {"k": sum(hits), "n": len(hits)}
        unk = []
        for f in sorted(R.glob(f"ans_{s}_r*.jsonl")):
            a = jl(f); unk.append(sum(r["answer"].strip().lower().startswith("unknown") for r in a) / max(1, len(a)))
        ent["unknown_rate_mean"] = st.mean(unk) if unk else None
        ing = [r for f in sorted(R.glob(f"ingest_{s}_r*.jsonl")) for r in jl(f) if "ingest_sec" in r]
        if ing:
            runs_sec = {}
            for f in sorted(R.glob(f"ingest_{s}_r*.jsonl")):
                rr = [r for r in jl(f) if "ingest_sec" in r]
                runs_sec[f.stem.split("_")[-1]] = {"conv": len(rr), "sessions": sum(r["sessions"] for r in rr),
                                                   "sec_sum": sum(r["ingest_sec"] for r in rr),
                                                   "errors": sum(r.get("errors", 0) for r in rr),
                                                   "failed_sessions": sum(1 for r in jl(f) if "error" in r)}
            ent["ingest"] = runs_sec
            # a run that crashed and was resumed has extraction time split over two logs: average complete logs only
            full = [v["sec_sum"] for v in runs_sec.values() if v["sessions"] == 272]
            ent["ingest_complete_runs"] = len(full)
            ent["ingest_min_mean"] = st.mean(full) / 60 if full else None
        lab = labels(s)
        byq = {}
        for (run, q), ok in lab.items():
            byq.setdefault(q, []).append(ok)
        maj = {q: sum(v) * 2 > len(v) for q, v in byq.items()}
        km = sum(maj.values())
        ent["majority"] = {"k": km, "n": len(maj), "wilson": wilson(km, len(maj))}
        out[s] = ent
    pairs = {}
    for a, b in (("mem0", "ours"), ("ours", "graphiti"), ("mem0", "graphiti"), ("full", "mem0"), ("full", "ours")):
        la, lb = labels(a), labels(b)
        keys = la.keys() & lb.keys()
        if not keys:
            continue
        per_run = {}
        for run in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == run]
            per_run[run] = mcnemar([la[k] for k in ks], [lb[k] for k in ks])
        ma = majority(la); mb = majority(lb); qs = sorted(ma.keys() & mb.keys())
        pairs[f"{a}_vs_{b}"] = {"per_run": per_run, "majority": mcnemar([ma[q] for q in qs], [mb[q] for q in qs])}
    return out, pairs


def labels(s, tag=""):
    R = D / "locomo-results"
    m = {}
    for f in sorted(R.glob(f"jud{tag}_{s}_r*.jsonl")):
        for r in jl(f):
            m[(f.stem.split("_")[-1], r["qid"])] = r["label"] == "CORRECT"
    return m


def majority(lab):
    byq = {}
    for (run, q), ok in lab.items():
        byq.setdefault(q, []).append(ok)
    return {q: sum(v) * 2 > len(v) for q, v in byq.items()}


def mcnemar(xa, xb):
    b10 = sum(x and not y for x, y in zip(xa, xb)); b01 = sum(y and not x for x, y in zip(xa, xb))
    n = b10 + b01
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b10, b01) + 1)) / 2 ** n) if n else 1.0
    return {"a_only": b10, "b_only": b01, "n": len(xa), "p": p}


def judge_agreement():
    """Second judge (other vendor) on the same answers: raw agreement and Cohen's kappa."""
    out = {}
    tot = [0, 0, 0, 0]  # both correct, a only, b only, both wrong
    for s, _ in SYSTEMS:
        a, b = labels(s), labels(s, "2")
        keys = a.keys() & b.keys()
        if not keys:
            continue
        c = [sum(a[k] and b[k] for k in keys), sum(a[k] and not b[k] for k in keys),
             sum(b[k] and not a[k] for k in keys), sum(not a[k] and not b[k] for k in keys)]
        tot = [x + y for x, y in zip(tot, c)]
        out[s] = kappa(c)
    if sum(tot):
        out["all"] = kappa(tot)
    return out


def kappa(c):
    n = sum(c); po = (c[0] + c[3]) / n
    pa = (c[0] + c[1]) / n; pb = (c[0] + c[2]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agree": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0, "judge1_rate": pa, "judge2_rate": pb}


def asof():
    A = D / "asof"
    ours = json.loads((A / "our_side_asof.json").read_text(encoding="utf-8-sig"))
    res = {"ours": {"k": ours["correct"], "n": ours["questions"], "p50_ms": ours["p50_ms"]}, "graphiti": {}}
    for f in sorted(A.glob("graphiti_*_asof*.json")):
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        if "items" not in d:
            continue
        model = d["system"].split(" ")[-1]
        run = "r1"
        if f.stem.endswith(("_r2", "_r3")):
            run = f.stem[-2:]
        if "run1-search-only" in f.stem:
            run = "r0"
        res["graphiti"].setdefault(model, {})[run] = {
            "search": d.get("correct"), "graph": d.get("memory_correct"), "n": d.get("questions"),
            "edges": len(d.get("edges", [])) if isinstance(d.get("edges"), list) else None,
            "episodes_ok": d.get("episodes_ok"), "wall_sec": d.get("wall_sec")}
    return res


def probes():
    P = D / "probes"
    out = {}
    o = json.loads((P / "probes_ours.json").read_text(encoding="utf-8-sig"))
    ob = json.loads((P / "probes_ours_broken.json").read_text(encoding="utf-8-sig"))
    out["ours"] = {"passed": o["passed"], "broken_passed": ob["passed"], "ms": o["wall_ms"]}
    g = json.loads((P / "probes_graphiti_core.json").read_text(encoding="utf-8-sig"))
    gb = json.loads((P / "probes_graphiti_core_broken.json").read_text(encoding="utf-8-sig"))
    cnt = lambda d: sum(v["ok"] for k, v in d["checks"].items() if k != "S1_history_persisted")
    out["graphiti_core"] = {"passed": cnt(g), "broken_passed": cnt(gb),
                            "per_check": {k: v["ok"] for k, v in g["checks"].items() if k != "S1_history_persisted"}}
    mp = P / "probes_mem0.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8-sig"))
        out["mem0"] = {"runs": [r["passed"] for r in m["runs"]],
                       "per_check": {k: [r["checks"][k]["ok"] for r in m["runs"]] for k in m["runs"][0]["checks"]}}
    return out


def retrieval():
    p = D / "retrieval" / "retrieval_summary.json"
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {}


def tex(agg):
    m = {}
    L = agg["locomo"]
    names = {"ours": "Ours", "mem0": "Memzero", "graphiti": "Graphiti", "full": "Full"}
    nq = None
    for s, e in L.items():
        n = names[s]
        m[f"loc{n}Mean"] = pct(e["mean"]); m[f"loc{n}Sd"] = pct(e["sd"])
        m[f"loc{n}Min"] = pct(e["min"]); m[f"loc{n}Max"] = pct(e["max"])
        m[f"loc{n}Pooled"] = pci(e["pooled"]["k"], e["pooled"]["n"])
        m[f"loc{n}Runs"] = str(len(e["runs"]))
        for r, v in e["runs"].items():
            m[f"loc{n}Run{r[1:].replace('1','One').replace('2','Two').replace('3','Three')}"] = pci(v["k"], v["n"])
        nq = nq or next(iter(e["runs"].values()))["n"]
        if "retr_ms_median" in e:
            m[f"loc{n}RetrMs"] = f"{e['retr_ms_median']:.1f}"
        if e.get("unknown_rate_mean") is not None:
            m[f"loc{n}Unknown"] = pct(e["unknown_rate_mean"])
        if "ingest" in e:
            if e.get("ingest_min_mean"):
                m[f"loc{n}IngestMin"] = f"{e['ingest_min_mean']:.0f}"
            m[f"loc{n}FailedSessions"] = str(sum(v["failed_sessions"] for v in e["ingest"].values()))
        if "evidence_hit" in e:
            m["locOursEvidence"] = pci(e["evidence_hit"]["k"], e["evidence_hit"]["n"])
    m["locN"] = str(nq or 0)
    for s_, e in L.items():
        m[f"loc{names[s_]}Maj"] = pci(e["majority"]["k"], e["majority"]["n"])
    fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    for k, v in agg["locomo_pairs"].items():
        kk = "".join(w.capitalize() for w in k.replace("mem0", "memzero").split("_"))
        mj = v["majority"]
        m[f"mc{kk}A"] = str(mj["a_only"]); m[f"mc{kk}B"] = str(mj["b_only"]); m[f"mc{kk}P"] = fp(mj["p"])
        m[f"mc{kk}PerRun"] = ", ".join(fp(r["p"]) for _, r in sorted(v["per_run"].items()))
    J = agg.get("judge_agreement", {})
    if "all" in J:
        m["judgeN"] = str(J["all"]["n"]); m["judgeAgree"] = pct(J["all"]["agree"]); m["judgeKappa"] = f"{J['all']['kappa']:.2f}"
        m["judgeOneRate"] = pct(J["all"]["judge1_rate"]); m["judgeTwoRate"] = pct(J["all"]["judge2_rate"])
        for s_ in L:
            l2 = labels(s_, "2")
            runs2 = sorted({k[0] for k in l2})
            accs = [sum(v for (r, q), v in l2.items() if r == run) / sum(1 for (r, q) in l2 if r == run) for run in runs2]
            m[f"jtwo{names[s_]}Mean"] = pct(st.mean(accs))
            m[f"jtwo{names[s_]}Runs"] = ", ".join(pct(a) for a in accs)
    A = agg["asof"]
    m["asofOurs"] = f"{A['ours']['k']}/{A['ours']['n']}"; m["asofOursCI"] = pci(A["ours"]["k"], A["ours"]["n"])
    m["asofOursMs"] = f"{A['ours']['p50_ms']:.2f}"; m["asofN"] = str(A["ours"]["n"])
    P = agg["probes"]
    m["prOurs"] = str(P["ours"]["passed"]); m["prOursBroken"] = str(P["ours"]["broken_passed"])
    m["prGraphiti"] = str(P["graphiti_core"]["passed"]); m["prGraphitiBroken"] = str(P["graphiti_core"]["broken_passed"])
    if "mem0" in P:
        m["prMemzeroRuns"] = ", ".join(map(str, P["mem0"]["runs"]))
    lines = ["% generated by analysis/make_numbers.py from data/*.json; do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    return "\n".join(lines) + "\n"


def tables(agg):
    """tab_asof.tex, tab_locomo.tex, tab_retrieval.tex: tables are generated, never typed."""
    A = agg["asof"]; L = agg["locomo"]; R = agg["retrieval"]
    def fmt_runs(d, key):
        xs = [d[r][key] for r in sorted(d) if r != "r0" and d[r].get(key) is not None]
        return ", ".join(map(str, xs))
    rows = []
    order = sorted(A["graphiti"].items(), key=lambda kv: -max((v.get("graph") or 0) for v in kv[1].values()))
    for model, runs in order:
        ok_runs = [r for r in runs if r != "r0"]
        transport = all((runs[r].get("episodes_ok") or 0) == 0 for r in ok_runs)
        note = "endpoint error, 0 episodes" if transport else ""
        rows.append(f"Graphiti + \\texttt{{{model}}} & {len(ok_runs)} & {fmt_runs(runs, 'graph')} & {fmt_runs(runs, 'search')} & {fmt_runs(runs, 'edges')} & {note} \\\\")
    n = A["ours"]["n"]
    t = ["\\begin{table}[h]\centering\small",
         f"\caption{{As-of questions over 15 real CRM changes ({n} questions). \emph{{Graph}}: correct live state read directly from the stored edges at date $D$; \emph{{Search}}: correct value among live facts returned by Graphiti search. One number per run.}}",
         "\label{tab:asof}", "\\begin{tabular}{lccccl}", "\\toprule",
         "System & Runs & Graph & Search & Edges & Note \\\\", "\midrule",
         f"Our fact log (0 LLM, median {A['ours']['p50_ms']:.2f} ms) & 1 & {A['ours']['k']} & {A['ours']['k']} & n/a & deterministic \\\\",
         *rows, "\\bottomrule", "\end{tabular}", "\end{table}"]
    (KIT / "tab_asof.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    names = dict(SYSTEMS)
    rows = []
    for s in ("full", "ours", "mem0", "graphiti"):
        if s not in L:
            continue
        e = L[s]
        runs = ", ".join(f"{v['k']}" for _, v in sorted(e["runs"].items()))
        ing = f"{e['ingest_min_mean']:.0f}" if e.get("ingest_min_mean") else ""
        unk = pct(e["unknown_rate_mean"]) if e.get("unknown_rate_mean") is not None else ""
        rows.append(f"{names[s]} & {len(e['runs'])} & {runs} & {pct(e['mean'])} & {pci(e['majority']['k'], e['majority']['n'])} & {unk} & {ing or 'none'} \\\\")
    nq = next(iter(L.values()))["runs"]["r1"]["n"]
    t = ["\\begin{table}[h]\centering\small",
         f"\caption{{LoCoMo temporal-reasoning subset ({nq} questions, 10 dialogues, 272 sessions). Correct answers per run, mean accuracy over runs, question-level accuracy (a question counts as correct when the majority of runs answer it correctly) with 95\% Wilson interval over {nq} questions, share of `unknown' answers, and LLM extraction time summed over the 10 dialogues (minutes, mean over complete runs).}}",
         "\label{tab:locomo}", "\\begin{tabular}{lcccccc}", "\\toprule",
         "System & Runs & Correct per run & Mean \% & Majority \% [95\% CI] & Unknown \% & Ingest min \\\\", "\midrule",
         *rows, "\\bottomrule", "\end{tabular}", "\end{table}"]
    P = agg["locomo_pairs"]
    if P:
        t += ["", "\\begin{table}[h]\centering\small",
              "\caption{Paired exact two-sided McNemar tests. Per run: $p$ for each run the two systems share. Question level: majority-of-runs correctness per question; questions only the first system answers, only the second, and $p$.}",
              "\label{tab:mcnemar}", "\\begin{tabular}{lcccc}", "\\toprule", "Pair & $p$ per run & First only & Second only & $p$ (questions) \\\\", "\midrule"]
        for k, v in P.items():
            a, b = k.split("_vs_")
            fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
            mj = v["majority"]
            t.append(f"{names[a]} vs {names[b]} & {', '.join(fp(r['p']) for _, r in sorted(v['per_run'].items()))} & {mj['a_only']} & {mj['b_only']} & {fp(mj['p'])} \\\\")
        t += ["\\bottomrule", "\end{tabular}", "\end{table}"]
    (KIT / "tab_locomo.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    if R:
        n = R["n_answerable"]
        t = ["\\begin{table}[h]\centering\small",
             f"\caption{{Plain retrieval on a frozen set of 40 real notes ({n} answerable questions, one supporting note each, gold SHA-256 prefix {R['gold_sha256'][:8]}): questions whose supporting note is in the top $k$. No temporal reasoning involved.}}",
             "\label{tab:retrieval}", "\\begin{tabular}{lccc}", "\\toprule", "System & R@1 & R@5 & R@10 \\\\", "\midrule"]
        for name, v in R["systems"].items():
            t.append(f"{name} & {v['r@1']}/{n} & {v['r@5']}/{n} & {v['r@10']}/{n} \\\\")
        t += ["\\bottomrule", "\end{tabular}", "\end{table}"]
        (KIT / "tab_retrieval.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def fit_wide(name):
    p = KIT / name
    s = p.read_text(encoding="utf-8-sig")
    B = chr(92)
    if B + "resizebox" not in s:
        s = s.replace(B + "begin{tabular}", B + "resizebox{" + B + "textwidth}{!}{" + B + "begin{tabular}")
        s = s.replace(B + "end{tabular}", B + "end{tabular}}")
        p.write_text(s, encoding="utf-8")


def main():
    loc, pairs = locomo()
    agg = {"locomo": loc, "locomo_pairs": pairs, "judge_agreement": judge_agreement(), "asof": asof(), "probes": probes(), "retrieval": retrieval()}
    (D / "aggregates.json").write_text(json.dumps(agg, indent=1, default=str), encoding="utf-8")
    (KIT / "numbers.tex").write_text(tex(agg), encoding="utf-8")
    tables(agg)
    perconv_table()
    for n in ("tab_asof.tex", "tab_locomo.tex"):
        fit_wide(n)
    print("ok systems", list(loc), "pairs", list(pairs))
    return 0



def perconv_table():
    """tab_perconv.tex: mean accuracy per LoCoMo dialogue and system (runs pooled)."""
    R = D / "locomo-results"
    acc = {}
    for s, _ in SYSTEMS:
        for f in sorted(R.glob(f"jud_{s}_r*.jsonl")):
            for r in jl(f):
                c = r["qid"].split("#")[0]
                k, n = acc.setdefault((s, c), [0, 0]); acc[(s, c)] = [k + (r["label"] == "CORRECT"), n + 1]
    convs = sorted({c for _, c in acc})
    syss = [s for s, _ in SYSTEMS if any((s, c) in acc for c in convs)]
    names = dict(SYSTEMS)
    nq = {c: acc[(syss[0], c)][1] // max(1, len(list(R.glob(f'jud_{syss[0]}_r*.jsonl')))) for c in convs}
    t = ["\\begin{table}[h]\centering\small",
         "\caption{Accuracy (\%) per LoCoMo dialogue, runs pooled. $n$: temporal questions in the dialogue.}",
         "\label{tab:perconv}", "\\begin{tabular}{lr" + "c" * len(syss) + "}", "\\toprule",
         "Dialogue & $n$ & " + " & ".join(names[s] for s in syss) + " \\\\", "\midrule"]
    for c in convs:
        t.append(f"{c} & {nq[c]} & " + " & ".join(pct(acc[(s, c)][0] / acc[(s, c)][1]) for s in syss) + " \\\\")
    t += ["\\bottomrule", "\end{tabular}", "\end{table}"]
    (KIT / "tab_perconv.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
