# -*- coding: utf-8 -*-
"""probes_mem0.py: room-402 scenario (S1, S2, S3) on Mem0 OSS 2.2.1 with LLM extraction, N runs.

Machine: ZBOOKG8-2023PAL (author) -> A-2022BAYAREA · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Events are fed as dated natural-language statements in the same order as probes_ours.py / probes_graphiti_core.py.
After each phase the questions are answered by the shared reader (bench_mot.READER_MODEL) from Mem0 search hits
(top 10, each hit shown with the date of the statement that produced it); a check passes when every question of
the phase names the expected occupant. S3: the only correct answer is Sidorov, the occupant before the withdrawn record (as in the structured probes).
Uses bench_mot.py (same folder) for the LLM rail and the E5 embed server. рельса: Gemini extraction, Codex reader.
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_mot as B  # noqa: E402

EVENTS1 = [("1 October 2026", "Front desk log: room 402 is vacant."),
           ("5 October 2026", "Front desk log: Mr Ivanov checked into room 402."),
           ("9 October 2026", "Front desk log: Mr Ivanov checked out; room 402 is vacant again."),
           ("20 October 2026", "Front desk log: Ms Petrova checked into room 402.")]
EVENT2 = ("5 November 2026", "Late report from the night shift: Mr Sidorov checked into room 402 on 12 October 2026; "
          "this was missing from the log.")
EVENT3 = ("6 November 2026", "Correction: the entry saying Ms Petrova checked into room 402 on 20 October 2026 was an "
          "operator error. Ms Petrova never stayed in room 402; please disregard that entry.")
Q = "Who was staying in room 402 on {d}? Answer with a surname, or 'vacant'."


def ask(m, d):
    q = Q.format(d=d)
    res = m.search(q, filters={"user_id": "probe402"}, top_k=10, threshold=0.0)["results"]
    ctx = "\n".join(f"- [{(h.get('metadata') or {}).get('date', '?')}] {h.get('memory', '')}" for h in res)
    a = B.chat(B.READER_MODEL, [{"role": "system", "content": B.READER_SYS},
                                {"role": "user", "content": f"Memory:\n{ctx or '(empty)'}\n\nQuestion: {q}\nAnswer:"}], max_tokens=2000)
    return a.strip().splitlines()[-1][:120] if a.strip() else "", ctx


def has(a, name):
    return name.lower() in a.lower()


def one_run(run):
    from mem0 import Memory
    base = B.OUT / f"probe_mem0_{run}"
    shutil.rmtree(base, ignore_errors=True); base.mkdir(parents=True)
    m = Memory.from_config({
        "embedder": {"provider": "huggingface", "config": {"model": "intfloat/multilingual-e5-base", "embedding_dims": 768}},
        "llm": {"provider": "openai", "config": {"model": B.EXTRACT_MODEL, "api_key": B.proxy_key(), "openai_base_url": B.PROXY,
                                                "temperature": 0, "max_tokens": 4096}},
        "vector_store": {"provider": "qdrant", "config": {"path": str(base / "q"), "collection_name": "p", "on_disk": True,
                                                           "embedding_model_dims": 768}},
        "history_db_path": str(base / "h.sqlite")})

    def add(d, t):
        m.add(f"Date: {d}. {t}", user_id="probe402", infer=True, metadata={"date": d})
    rec = {"run": run, "checks": {}}
    for d, t in EVENTS1:
        add(d, t)
    want1 = {"3 October 2026": "vacant", "7 October 2026": "Ivanov", "15 October 2026": "vacant", "25 October 2026": "Petrova"}
    got1 = {d: ask(m, d)[0] for d in want1}
    rec["checks"]["S1_four_changes_asof"] = {"got": got1, "ok": all(has(got1[d], w) for d, w in want1.items())}
    add(*EVENT2)
    want2 = {"10 October 2026": "vacant", "13 October 2026": "Sidorov", "25 October 2026": "Petrova"}
    got2 = {d: ask(m, d)[0] for d in want2}
    rec["checks"]["S2_late_fact"] = {"got": got2, "ok": all(has(got2[d], w) for d, w in want2.items())}
    add(*EVENT3)
    a3, ctx3 = ask(m, "25 October 2026")
    rec["checks"]["S3_retract_last"] = {"got": a3, "ctx": ctx3,
                                        "ok": has(a3, "Sidorov") and not has(a3, "Petrova")}
    mems = m.get_all(filters={"user_id": "probe402"})
    rec["memories_after"] = [x.get("memory") for x in (mems.get("results", mems) if isinstance(mems, dict) else mems)]
    rec["passed"] = sum(v["ok"] for v in rec["checks"].values())
    return rec


def main():
    sys.path.insert(0, os.environ.get("MEM0_DIR_CODE", str(B.HERE)))
    os.environ.update(MEM0_TELEMETRY="False", HF_HUB_OFFLINE="1")
    from mem0.utils.factory import EmbedderFactory
    from mem0.embeddings.base import EmbeddingBase

    class HttpE5(EmbeddingBase):
        def __init__(self, config=None):
            super().__init__(config); self.config.embedding_dims = 768

        def embed(self, text, memory_action=None):
            return B.embed([text], "query" if memory_action == "search" else "passage")[0]

        def embed_batch(self, texts, memory_action="add"):
            return B.embed(list(texts), "query" if memory_action == "search" else "passage") if texts else []
    sys.modules[__name__].HttpE5 = HttpE5
    EmbedderFactory.provider_to_class["huggingface"] = f"{__name__}.HttpE5"
    runs = [one_run(f"r{i}") for i in range(1, int(os.environ.get("RUNS", "3")) + 1)]
    out = {"system": "mem0-oss-2.2.1 infer=True", "extract_model": B.EXTRACT_MODEL, "reader_model": B.READER_MODEL,
           "runs": runs, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (B.OUT / "probes_mem0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("probes mem0", [r["passed"] for r in runs], flush=True)


if __name__ == "__main__":
    main()
