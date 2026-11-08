# -*- coding: utf-8 -*-
"""Наша сторона сравнения «память во времени»: вопросы «как было на дату D» по живой истории лидов.
Machine: A-2022BAYAREA (Хаб) · Agent: Claude Code H75300 · Operator: Anton · 2026-11-05

Источник правды: object_history в leads.db (private CRM, path from LEADS_DB) (history_gen.py, включено 04.11).
Каждая смена поля даёт 2 вопроса: «до смены» (эталон old_json) и «после» (эталон value_json).
Ответчик = чтение истории на дату (asof), 0 LLM. Выход: our_side_asof.json рядом
(вопросы + эталон + ответ + задержка) и та же пачка вопросов как эпизоды текста для Graphiti
(our_side_episodes.jsonl), чтобы чужая память получила те же факты, но сырым текстом.
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = "file:" + os.environ.get("LEADS_DB", "leads.db") + "?mode=ro"  # private CRM; published output is pseudonymised
FIELDS = ("temperature", "reactivation_band", "status")


def asof(c, subject, field, at):
    """Значение поля на момент at: последняя смена с observed_at <= at, иначе old_json первой смены."""
    row = c.execute("SELECT value_json FROM object_history WHERE subject=? AND field=? AND operation IN ('I','U') "
                    "AND valid_from<=? ORDER BY valid_from DESC, hid DESC LIMIT 1", (subject, field, at)).fetchone()
    if row:
        return json.loads(row[0])
    first = c.execute("SELECT old_json FROM object_history WHERE subject=? AND field=? ORDER BY valid_from, hid LIMIT 1",
                      (subject, field)).fetchone()
    return json.loads(first[0]) if first and first[0] else None


def main():
    c = sqlite3.connect(DB, uri=True)
    changes = c.execute(f"SELECT subject, field, old_json, value_json, valid_from FROM object_history "
                        f"WHERE operation='U' AND field IN {FIELDS} ORDER BY valid_from").fetchall()
    qs, eps = [], []
    for subj, field, old, new, vf in changes:
        t = datetime.fromisoformat(vf.replace("Z", "+00:00"))
        before = (t - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        after = (t + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        name = subj.split(":", 1)[1]
        for at, gold in ((before, json.loads(old)), (after, json.loads(new))):
            t0 = time.perf_counter()
            ans = asof(c, subj, field, at)
            qs.append({"q": f"What was the {field} of lead {name} at {at}?", "subject": subj, "field": field,
                       "at": at, "gold": gold, "answer": ans, "ok": ans == gold,
                       "ms": round((time.perf_counter() - t0) * 1000, 2)})
        eps.append({"name": f"{name}-{field}-old", "reference_time": (t - timedelta(days=1)).isoformat(),
                    "body": f"Lead {name} has {field} {json.loads(old)}."})
        eps.append({"name": f"{name}-{field}-new", "reference_time": t.isoformat(),
                    "body": f"Lead {name} now has {field} {json.loads(new)}."})
    ok = sum(q["ok"] for q in qs)
    out = {"source": "leads.db object_history", "changes": len(changes), "questions": len(qs), "correct": ok,
           "p50_ms": sorted(q["ms"] for q in qs)[len(qs) // 2] if qs else None, "items": qs}
    (HERE / "our_side_asof.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "our_side_episodes.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in eps), encoding="utf-8")
    print(f"смен {len(changes)}, вопросов {len(qs)}, верно {ok}, p50 {out['p50_ms']} мс")
    return 0 if ok == len(qs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
