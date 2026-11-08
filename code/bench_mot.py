# -*- coding: utf-8 -*-
"""bench_mot.py: LoCoMo temporal subset (category 2) + room-402 probes on four memory systems.

Machine: ZBOOKG8-2023PAL (author) -> runs on A-2022BAYAREA (hub) · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08

Systems
  mem0      Mem0 OSS 2.2.1, infer=True (LLM fact extraction), local E5 embedder, embedded Qdrant, top_k=K
  graphiti  Graphiti 0.31.2 + Kuzu, one graph per conversation, sessions as episodes with reference_time,
            LLM extraction, E5 embedder served over local HTTP, edges with valid_at / invalid_at, top K
  ours      time-stamped turn log (no extraction): every turn stored with its session date (valid time),
            dense E5 retrieval top K; same record shape as fact_history rows
  full      full-context reference: the whole conversation with session dates (upper bound, no memory)
Same reader LLM, same judge LLM, same K for all retrieval systems.

Stages (each resumable, outputs under OUT_DIR):
  embed-server            (mem0 venv)      HTTP E5 on 127.0.0.1:8779
  ingest-mem0 --run rN    (mem0 venv)      -> ctx_mem0_rN.jsonl
  ingest-graphiti --run rN (graphiti venv) -> ctx_graphiti_rN.jsonl
  ctx-ours / ctx-full     (mem0 venv)      -> ctx_ours.jsonl, ctx_full.jsonl
  answer --system S --run rN               -> ans_S_rN.jsonl   (reader)
  judge  --system S --run rN               -> jud_S_rN.jsonl   (judge)
  aggregate                                -> results_locomo.json
LLMs go through the local CLIProxy (OpenAI-compatible, our subscriptions). Key read from its config, never printed.
рельса: extraction gemini-3-flash (Gemini), reader gpt-6-sol (Codex), judge grok-4.20-0309-non-reasoning (Grok).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUT_DIR", HERE))
DATA = Path(os.environ.get("LOCOMO", HERE / "locomo10.json"))
PROXY = "http://127.0.0.1:8317/v1"
CFG = Path(os.environ.get("CLIPROXY_CONFIG", "cliproxy-config.yaml"))  # api-keys list of the local proxy
EXTRACT_MODEL = os.environ.get("EXTRACT_MODEL", "gemini-3-flash")
READER_MODEL = os.environ.get("READER_MODEL", "gpt-6-sol")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "grok-4.20-0309-non-reasoning")
K = int(os.environ.get("TOPK", "10"))
EMBED_PORT = int(os.environ.get("EMBED_PORT", "8779"))
EMBED_URL = f"http://127.0.0.1:{EMBED_PORT}/embed"
CONV_LIMIT = int(os.environ.get("CONV_LIMIT", "10"))
SES_LIMIT = int(os.environ.get("SES_LIMIT", "999"))  # smoke only
_lock = threading.Lock()


def proxy_key() -> str:
    lines = CFG.read_text(encoding="utf-8-sig").splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("api-keys:"))
    return lines[i + 1].split("-", 1)[1].strip().strip('"')


def chat(model: str, messages: list, max_tokens: int = 300, tries: int = 6) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if not model.startswith("gpt-"):
        body["temperature"] = 0
    else:
        body["max_completion_tokens"] = body.pop("max_tokens")
    data = json.dumps(body).encode()
    err = None
    for a in range(tries):
        try:
            req = urllib.request.Request(PROXY + "/chat/completions", data=data, headers={
                "Authorization": "Bearer " + proxy_key(), "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                j = json.loads(r.read())
            txt = (j["choices"][0]["message"].get("content") or "").strip()
            if txt:
                return txt
            err = "empty"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
        time.sleep(min(60, 4 * 2 ** a))
    raise RuntimeError(f"chat failed {model}: {err}")


def load():
    d = json.loads(DATA.read_text(encoding="utf-8-sig"))[:CONV_LIMIT]
    convs = []
    for c in d:
        conv = c["conversation"]
        ses = sorted((int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k)))[:SES_LIMIT]
        sessions = []
        for n in ses:
            raw = conv[f"session_{n}_date_time"]
            sessions.append({"n": n, "date_raw": raw, "date": parse_date(raw),
                             "turns": [{"speaker": t["speaker"], "id": t["dia_id"], "text": t["text"]
                                        + (f" [shares image: {t['blip_caption']}]" if t.get("blip_caption") else "")}
                                       for t in conv[f"session_{n}"]]})
        qs = [{"qid": f"{c['sample_id']}#{i}", "question": q["question"], "gold": str(q["answer"]),
               "evidence": q.get("evidence", [])}
              for i, q in enumerate(c["qa"]) if q.get("category") == 2]
        convs.append({"id": c["sample_id"], "sessions": sessions, "qs": qs})
    return convs


def parse_date(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def human(d: datetime) -> str:
    return d.strftime("%d %B %Y").lstrip("0")


def session_text(s) -> str:
    return f"Conversation date: {human(s['date'])}\n" + "\n".join(f"{t['speaker']}: {t['text']}" for t in s["turns"])


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for l in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            out.add(json.loads(l)["qid"])
        except Exception:  # noqa: BLE001
            pass
    return out


def write(path: Path, row: dict):
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ---------------- embedder ----------------
def embed_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    sys.path.insert(0, os.environ.get("MEM0_DIR_CODE", str(HERE)))
    from sentence_transformers import SentenceTransformer
    from local_providers import E5_SNAPSHOT
    import torch
    torch.set_num_threads(6)
    model = SentenceTransformer(str(E5_SNAPSHOT), device="cpu", local_files_only=True)
    ml = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            j = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            pre = "query: " if j.get("kind") == "query" else "passage: "
            with ml:
                v = model.encode([pre + t for t in j["texts"]], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False, batch_size=32).tolist()
            b = json.dumps(v).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass
    print("embed-server ready", flush=True)
    ThreadingHTTPServer(("127.0.0.1", EMBED_PORT), H).serve_forever()


def embed(texts, kind="passage"):
    req = urllib.request.Request(EMBED_URL, data=json.dumps({"texts": texts, "kind": kind}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


# ---------------- ours: time-stamped turn log ----------------
def ctx_ours():
    out = OUT / "ctx_ours.jsonl"
    have = done_ids(out)
    for c in load():
        rows = [{"valid_at": s["date"].isoformat(), "date": human(s["date"]), "session": s["n"],
                 "speaker": t["speaker"], "text": t["text"], "id": t["id"]} for s in c["sessions"] for t in s["turns"]]
        vec = embed([f"[{r['date']}] {r['speaker']}: {r['text']}" for r in rows])
        for q in c["qs"]:
            if q["qid"] in have:
                continue
            t0 = time.perf_counter()
            qv = embed([q["question"]], "query")[0]
            sc = sorted(((sum(a * b for a, b in zip(qv, v)), i) for i, v in enumerate(vec)), reverse=True)[:K]
            hits = [rows[i] for _, i in sc]
            ms = (time.perf_counter() - t0) * 1000
            ctx = "\n".join(f"- [{h['date']}] {h['speaker']}: {h['text']}" for h in hits)
            ev = set(q["evidence"])
            write(out, {"qid": q["qid"], "ctx": ctx, "retr_ms": round(ms, 2),
                        "evidence_hit": bool(ev & {h["id"] for h in hits})})
        print("ours", c["id"], flush=True)


def ctx_full():
    out = OUT / "ctx_full.jsonl"
    have = done_ids(out)
    for c in load():
        full = "\n\n".join(session_text(s) for s in c["sessions"])
        for q in c["qs"]:
            if q["qid"] not in have:
                write(out, {"qid": q["qid"], "ctx": full})


# ---------------- Mem0 ----------------
def ingest_mem0(run: str):
    sys.path.insert(0, os.environ.get("MEM0_DIR_CODE", str(HERE)))
    os.environ.update(MEM0_TELEMETRY="False", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from mem0.utils.factory import EmbedderFactory
    from mem0.embeddings.base import EmbeddingBase
    from mem0 import Memory

    class HttpE5(EmbeddingBase):
        def __init__(self, config=None):
            super().__init__(config); self.config.embedding_dims = 768

        def embed(self, text, memory_action=None):
            return embed([text], "query" if memory_action == "search" else "passage")[0]

        def embed_batch(self, texts, memory_action="add"):
            return embed(list(texts), "query" if memory_action == "search" else "passage") if texts else []
    mod = sys.modules[__name__]
    mod.HttpE5 = HttpE5
    EmbedderFactory.provider_to_class["huggingface"] = f"{__name__}.HttpE5"
    out = OUT / f"ctx_mem0_{run}.jsonl"
    log = OUT / f"ingest_mem0_{run}.jsonl"
    have = done_ids(out)
    key = proxy_key()

    def one(c):
        if all(q["qid"] in have for q in c["qs"]):
            return
        base = OUT / f"mem0_{run}" / c["id"]
        if base.exists():  # partial ingest from a crash: start this conversation clean
            import shutil; shutil.rmtree(base)
        base.mkdir(parents=True)
        m = Memory.from_config({
            "embedder": {"provider": "huggingface", "config": {"model": "intfloat/multilingual-e5-base", "embedding_dims": 768}},
            "llm": {"provider": "openai", "config": {"model": EXTRACT_MODEL, "api_key": key, "openai_base_url": PROXY,
                                                    "temperature": 0, "max_tokens": 4096}},
            "vector_store": {"provider": "qdrant", "config": {"path": str(base / "qdrant"), "collection_name": "c",
                                                               "on_disk": True, "embedding_model_dims": 768}},
            "history_db_path": str(base / "history.sqlite")})
        t0 = time.time(); nmem = 0; errs = 0
        for s in c["sessions"]:
            for attempt in range(3):
                try:
                    r = m.add(session_text(s), user_id=c["id"], infer=True,
                              metadata={"session": s["n"], "session_date": human(s["date"])})
                    nmem += len(r.get("results", [])); break
                except Exception as e:  # noqa: BLE001
                    errs += 1; time.sleep(10 * (attempt + 1))
                    if attempt == 2:
                        write(log, {"conv": c["id"], "session": s["n"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
        ingest_s = time.time() - t0
        for q in c["qs"]:
            t1 = time.perf_counter()
            res = m.search(q["question"], filters={"user_id": c["id"]}, top_k=K, threshold=0.0)["results"]
            ms = (time.perf_counter() - t1) * 1000
            ctx = "\n".join(f"- [{(h.get('metadata') or {}).get('session_date', '?')}] {h.get('memory', '')}" for h in res)
            write(out, {"qid": q["qid"], "ctx": ctx, "retr_ms": round(ms, 1)})
        write(log, {"conv": c["id"], "memories_events": nmem, "errors": errs, "ingest_sec": round(ingest_s, 1),
                    "sessions": len(c["sessions"])})
        print("mem0", run, c["id"], nmem, round(ingest_s), flush=True)

    with cf.ThreadPoolExecutor(int(os.environ.get("PAR", "5"))) as ex:
        list(ex.map(one, load()))


# ---------------- Graphiti ----------------
def ingest_graphiti(run: str):
    import asyncio
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.nodes import EpisodeType

    class E5(EmbedderClient):
        async def create(self, input_data):
            t = input_data if isinstance(input_data, str) else " ".join(map(str, input_data))
            return (await asyncio.to_thread(embed, [t]))[0]

        async def create_batch(self, input_data_list):
            return await asyncio.to_thread(embed, list(input_data_list))

    class NoRerank(CrossEncoderClient):
        async def rank(self, query, passages):
            return [(p, 1.0 - i * 1e-3) for i, p in enumerate(passages)]

    class Strict(OpenAIGenericClient):  # same schema fix as graphiti_glm_asof.py (CLIProxy strict json_schema)
        def _build_response_format(self, response_model):
            rf = super()._build_response_format(response_model)
            if rf.get("type") != "json_schema":
                return rf

            def s_(s):
                if isinstance(s, dict):
                    if s.get("type") == "object" and "properties" in s:
                        s["additionalProperties"] = False; s["required"] = list(s["properties"])
                    for k in ("properties", "$defs", "definitions"):
                        for v in (s.get(k) or {}).values():
                            s_(v)
                    for k in ("items", "anyOf", "allOf", "oneOf"):
                        v = s.get(k)
                        for x in (v if isinstance(v, list) else [v] if v else []):
                            s_(x)
                    s.pop("default", None)
                return s
            s_(rf["json_schema"]["schema"])
            return rf

    out = OUT / f"ctx_graphiti_{run}.jsonl"
    log = OUT / f"ingest_graphiti_{run}.jsonl"
    have = done_ids(out)
    key = proxy_key()
    sem = asyncio.Semaphore(int(os.environ.get("PAR", "5")))

    def utc(x):
        return x if x is None or x.tzinfo else x.replace(tzinfo=timezone.utc)

    async def one(c):
        if all(q["qid"] in have for q in c["qs"]):
            return
        async with sem:
            llm = Strict(config=LLMConfig(api_key=key, base_url=PROXY, model=EXTRACT_MODEL, small_model=EXTRACT_MODEL,
                                          temperature=0, max_tokens=8192))
            dbdir = OUT / f"graphiti_{run}"; dbdir.mkdir(exist_ok=True)
            dbp = dbdir / f"{c['id']}.kuzu"
            if dbp.exists():
                import shutil
                shutil.rmtree(dbp) if dbp.is_dir() else dbp.unlink()
            import kuzu
            from unittest.mock import patch
            _orig = kuzu.Database
            # Kuzu's default buffer pool is 80% of RAM per database; 5 parallel graphs exhausted the hub (bad allocation)
            with patch.object(kuzu, "Database", side_effect=lambda *a, **kw: _orig(
                    *a, **{**kw, "buffer_pool_size": int(os.environ.get("KUZU_POOL_MB", "512")) * 1024 * 1024})):
                driver = KuzuDriver(str(dbp))
            g = Graphiti(graph_driver=driver, llm_client=llm, embedder=E5(), cross_encoder=NoRerank())
            await g.build_indices_and_constraints()
            await driver.graph_ops.build_indices_and_constraints(driver)
            t0 = time.time(); ok = 0; errs = 0
            for s in c["sessions"]:
                for attempt in range(3):
                    try:
                        await g.add_episode(name=f"{c['id']}-s{s['n']}", episode_body=session_text(s),
                                            source=EpisodeType.message, source_description="LoCoMo dialogue session",
                                            reference_time=s["date"])
                        ok += 1; break
                    except Exception as e:  # noqa: BLE001
                        errs += 1; await asyncio.sleep(10 * (attempt + 1))
                        if attempt == 2:
                            write(log, {"conv": c["id"], "session": s["n"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
            ingest_s = time.time() - t0
            for q in c["qs"]:
                t1 = time.perf_counter()
                res = await g.search(q["question"], num_results=K)
                ms = (time.perf_counter() - t1) * 1000
                lines = []
                for r in res:
                    va, ia = utc(r.valid_at), utc(r.invalid_at)
                    lines.append(f"- {r.fact} (valid from {human(va) if va else 'unknown'}"
                                 f" to {human(ia) if ia else 'present'})")
                write(out, {"qid": q["qid"], "ctx": "\n".join(lines), "retr_ms": round(ms, 1)})
            write(log, {"conv": c["id"], "episodes_ok": ok, "errors": errs, "ingest_sec": round(ingest_s, 1),
                        "sessions": len(c["sessions"])})
            print("graphiti", run, c["id"], ok, round(ingest_s), flush=True)
            await g.close()

    async def main():
        await asyncio.gather(*(one(c) for c in load()))
    asyncio.run(main())


# ---------------- reader & judge ----------------
READER_SYS = ("You answer a question about a long conversation between two people, using only the memory provided. "
              "Memory lines may carry dates. Resolve relative time expressions (e.g. 'yesterday', 'last week', "
              "'next month') against the date attached to the line where they occur. Answer with a short phrase: "
              "a date, a month, a year, or a duration. If the memory does not contain the answer, reply 'unknown'.")
JUDGE_SYS = ("You grade an answer to a question about time. Given the question, the gold answer and a generated answer, "
             "reply CORRECT if the generated answer refers to the same date, period or duration as the gold answer "
             "(be generous with format: '7 May 2023' and 'May 7th, 2023' match; a relative phrase matches if it "
             "denotes the same time as the gold), otherwise WRONG. Reply with exactly one word: CORRECT or WRONG.")


def qindex():
    return {q["qid"]: q for c in load() for q in c["qs"]}


def answer(system: str, run: str):
    ctxf = OUT / (f"ctx_{system}.jsonl" if system in ("ours", "full") else f"ctx_{system}_{run}.jsonl")
    ctx = {json.loads(l)["qid"]: json.loads(l)["ctx"] for l in ctxf.read_text(encoding="utf-8-sig").splitlines()}
    out = OUT / f"ans_{system}_{run}.jsonl"
    have = done_ids(out); qi = qindex()

    def one(qid):
        q = qi[qid]
        a = chat(READER_MODEL, [{"role": "system", "content": READER_SYS},
                                {"role": "user", "content": f"Memory:\n{ctx[qid] or '(empty)'}\n\nQuestion: {q['question']}\nAnswer:"}],
                 max_tokens=2000)
        write(out, {"qid": qid, "answer": a.strip().splitlines()[-1][:300] if a.strip() else ""})
    todo = [k for k in ctx if k not in have]
    with cf.ThreadPoolExecutor(int(os.environ.get("PAR", "6"))) as ex:
        for f in cf.as_completed([ex.submit(one, k) for k in todo]):
            f.result()
    print("answer", system, run, len(todo), flush=True)


def judge(system: str, run: str):
    ans = {json.loads(l)["qid"]: json.loads(l)["answer"] for l in (OUT / f"ans_{system}_{run}.jsonl").read_text(encoding="utf-8-sig").splitlines()}
    out = OUT / f"jud{os.environ.get('JUDGE_TAG', '')}_{system}_{run}.jsonl"
    have = done_ids(out); qi = qindex()

    def one(qid):
        q = qi[qid]
        v = chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYS},
                               {"role": "user", "content": f"Question: {q['question']}\nGold answer: {q['gold']}\nGenerated answer: {ans[qid]}\nVerdict:"}],
                 max_tokens=10)
        write(out, {"qid": qid, "label": "CORRECT" if v.strip().upper().rstrip(".").startswith("CORRECT") else "WRONG",  # strict: "not correct" -> WRONG
                    "raw": v[:40]})
    todo = [k for k in ans if k not in have]
    with cf.ThreadPoolExecutor(int(os.environ.get("PAR", "6"))) as ex:
        for f in cf.as_completed([ex.submit(one, k) for k in todo]):
            f.result()
    print("judge", system, run, len(todo), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--run", default="r1")
    ap.add_argument("--system", default="ours")
    a = ap.parse_args()
    {"embed-server": embed_server, "ctx-ours": ctx_ours, "ctx-full": ctx_full,
     "ingest-mem0": lambda: ingest_mem0(a.run), "ingest-graphiti": lambda: ingest_graphiti(a.run),
     "answer": lambda: answer(a.system, a.run), "judge": lambda: judge(a.system, a.run)}[a.stage]()


if __name__ == "__main__":
    main()
