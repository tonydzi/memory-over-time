# -*- coding: utf-8 -*-
"""Graphiti (открытый Zep) на подписке GLM Coding (anthropic-совместимый endpoint) против нашей истории лидов.
Machine: A-2022BAYAREA (Хаб) · Agent: Claude Code H75300 · Operator: Anton · 2026-11-05

Вход: our_side_episodes.jsonl (те же 15 смен лидов сырым текстом) + our_side_asof.json (30 вопросов с эталоном).
Graphiti сам извлекает факты LLM-ом, закрывает старые датой; ответ на дату D = рёбра лида, живые на D
(valid_at <= D и invalid_at пуст или > D), эталон ищется строкой в тексте факта (судья без LLM).
Эмбеддер: детерминированный хэш слов (0 сети, 0 загрузки), честно хуже настоящего; извлечение и
закрытие фактов он не трогает. Телеметрия Graphiti выключена. Квитанция: graphiti_glm_asof.json.
Ключ GLM из secrets/glm_coding.env, в вывод не печатается.
"""
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
HERE = Path(__file__).resolve().parent
MODEL = os.environ.get("GLM_MODEL", "glm-5.3")
LIMIT = int(os.environ.get("EP_LIMIT", "30"))


def glm_token() -> str:
    for line in Path(os.environ.get("GLM_ENV_FILE", "glm_coding.env")).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("GLM_CODING_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("нет GLM_CODING_TOKEN")


def dt(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


async def main() -> int:
    from anthropic import AsyncAnthropic
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.nodes import EpisodeType

    class HashEmbedder(EmbedderClient):
        dim = 256

        def _v(self, text):
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", str(text).lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

        async def create(self, input_data):
            return self._v(input_data if isinstance(input_data, str) else " ".join(map(str, input_data)))

        async def create_batch(self, input_data_list):
            return [self._v(t) for t in input_data_list]

    class NoRerank(CrossEncoderClient):
        async def rank(self, query, passages):
            return [(p, 1.0 - i * 1e-3) for i, p in enumerate(passages)]

    tok = glm_token()
    cfg = LLMConfig(api_key=tok, model=MODEL, small_model=MODEL, temperature=0, max_tokens=8192)
    ac = AsyncAnthropic(api_key=tok, base_url="https://api.z.ai/api/anthropic", max_retries=2)
    # anthropic 2.6.0 убрал temperature из messages.create, Graphiti 0.31.2 его шлёт -> TypeError на каждом эпизоде
    _create = ac.messages.create

    async def _create_no_temp(*a, **kw):
        kw.pop("temperature", None)
        return await _create(*a, **kw)

    ac.messages.create = _create_no_temp
    llm = AnthropicClient(config=cfg, client=ac)
    if os.environ.get("LLM_RAIL") in ("mistral", "cliproxy"):  # 06.11 Антон: «у нас куча подписок»
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        if os.environ["LLM_RAIL"] == "cliproxy":  # подписки Codex/Gemini/Grok как OpenAI API через локальный CLIProxy
            mk = re.search(r'api-keys:\s*\n\s*-\s*"([^"]+)"', Path(os.environ.get("CLIPROXY_CONFIG", "cliproxy-config.yaml"))
                           .read_text(encoding="utf-8")).group(1)
            base = "http://127.0.0.1:8317/v1"
            mm = os.environ.get("CLIPROXY_MODEL", "gpt-6-sol")
        else:
            mk = [l.split("=", 1)[1].strip().strip('"') for l in Path(os.environ.get("MISTRAL_ENV_FILE", "mistral_api.env"))
                  .read_text(encoding="utf-8-sig").splitlines() if l.startswith("MISTRAL_API_KEY=")][0]
            base = "https://api.mistral.ai/v1"
            mm = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        class MistralUnwrap(OpenAIGenericClient):
            # Mistral на json_schema иногда отдаёт {"properties": {...}} вместо самого объекта -> ValidationError
            def _build_response_format(self, response_model):
                # CLIProxy -> Codex включает strict: схема обязана иметь additionalProperties:false и все поля required
                rf = super()._build_response_format(response_model)
                if rf.get("type") != "json_schema":
                    return rf

                def strictify(s):
                    if isinstance(s, dict):
                        if s.get("type") == "object" and "properties" in s:
                            s["additionalProperties"] = False
                            s["required"] = list(s["properties"])
                        for k in ("properties", "$defs", "definitions"):
                            for v in (s.get(k) or {}).values():
                                strictify(v)
                        for k in ("items", "anyOf", "allOf", "oneOf"):
                            v = s.get(k)
                            for x in (v if isinstance(v, list) else [v] if v else []):
                                strictify(x)
                        s.pop("default", None)
                    return s

                strictify(rf["json_schema"]["schema"])
                return rf

            async def _generate_response(self, *a, **kw):
                r = await super()._generate_response(*a, **kw)
                if isinstance(r, dict) and set(r) <= {"properties", "type", "required", "title"} and isinstance(r.get("properties"), dict):
                    return r["properties"]
                return r

        llm = MistralUnwrap(config=LLMConfig(api_key=mk, base_url=base, model=mm,
                                             small_model=mm, temperature=0, max_tokens=8192))
        rec_system = f"graphiti-0.31.2 + {os.environ['LLM_RAIL']} {mm}"
    else:
        rec_system = f"graphiti-0.31.2 + GLM subscription {MODEL}"
    driver = KuzuDriver(":memory:")
    g = Graphiti(graph_driver=driver, llm_client=llm, embedder=HashEmbedder(), cross_encoder=NoRerank())
    await g.build_indices_and_constraints()
    await driver.graph_ops.build_indices_and_constraints(driver)  # полнотекстовые индексы Kuzu, иначе search падает
    eps = [json.loads(l) for l in (HERE / "our_side_episodes.jsonl").read_text(encoding="utf-8").splitlines()][:LIMIT]
    rec = {"system": rec_system, "embedder": "hash-bow-256 (смок, не качество поиска)",
           "episodes": [], "items": []}
    t0 = time.time()
    for e in eps:
        t = time.time()
        try:
            await g.add_episode(name=e["name"], episode_body=e["body"], source=EpisodeType.text,
                                source_description="leads.db object_history, H75300", reference_time=dt(e["reference_time"]))
            rec["episodes"].append({"name": e["name"], "sec": round(time.time() - t, 1), "ok": True})
        except Exception as ex:  # noqa: BLE001
            rec["episodes"].append({"name": e["name"], "ok": False, "err": f"{type(ex).__name__}: {str(ex)[:200]}"})
        print(rec["episodes"][-1], flush=True)
    names = {e["name"].rsplit("-", 2)[0] for e in eps}
    qs = [q for q in json.loads((HERE / "our_side_asof.json").read_text(encoding="utf-8"))["items"]
          if q["subject"].split(":", 1)[1] in names]
    for q in qs:
        at = dt(q["at"])
        lead = q["subject"].split(":", 1)[1]
        res = await g.search(q["q"], num_results=20)
        def utc(x):  # Kuzu отдаёт даты без пояса -> считаем их UTC
            return x if x is None or x.tzinfo else x.replace(tzinfo=timezone.utc)

        live = [r for r in res if lead.split("-")[0].lower() in (r.fact or "").lower()
                and (r.valid_at is None or utc(r.valid_at) <= at) and (r.invalid_at is None or utc(r.invalid_at) > at)]
        hit = any(str(q["gold"]).lower() in (r.fact or "").lower() for r in live)
        rec["items"].append({"q": q["q"], "gold": q["gold"], "ok": hit,
                             "live_facts": [f"{r.fact} | {r.valid_at} -> {r.invalid_at}" for r in live][:4],
                             "all_facts": [f"{r.fact} | {r.valid_at} -> {r.invalid_at}" for r in res][:6]})
    # Память отдельно от поиска: все рёбра графа, таймлайн лида проверяем напрямую (без эмбеддера)
    recs, _, _ = await driver.execute_query(
        "MATCH (a:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
        "RETURN e.fact AS fact, e.valid_at AS valid_at, e.invalid_at AS invalid_at")
    edges = [{"fact": r["fact"], "valid_at": r["valid_at"], "invalid_at": r["invalid_at"]} for r in recs]
    rec["edges"] = [{**e, "valid_at": str(e["valid_at"]), "invalid_at": str(e["invalid_at"])} for e in edges]

    def utc2(x):
        return x if x is None or x.tzinfo else x.replace(tzinfo=timezone.utc)

    mem_ok = 0
    for i, q in zip(rec["items"], qs):
        at = dt(q["at"])
        lead = q["subject"].split(":", 1)[1].lower()
        live = [e for e in edges if lead in (e["fact"] or "").lower()
                and (e["valid_at"] is None or utc2(e["valid_at"]) <= at) and (e["invalid_at"] is None or utc2(e["invalid_at"]) > at)]
        i["mem_ok"] = any(str(q["gold"]).lower() in e["fact"].lower() for e in live) and not any(
            str(q["gold"]).lower() not in e["fact"].lower() and q["field"].split("_")[0] in e["fact"].lower()
            and "previous" not in e["fact"].lower() for e in live)
        mem_ok += i["mem_ok"]
    rec["memory_correct"] = mem_ok
    ok = sum(i["ok"] for i in rec["items"])
    rec.update({"questions": len(qs), "correct": ok, "episodes_ok": sum(e["ok"] for e in rec["episodes"]),
                "wall_sec": round(time.time() - t0, 1)})
    (HERE / os.environ.get("OUT", "graphiti_glm_asof.json")).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nэпизодов ок {rec['episodes_ok']}/{len(eps)}, вопросов верно {ok}/{len(qs)}, {rec['wall_sec']} с")
    await g.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
