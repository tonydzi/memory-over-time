#!/usr/bin/env python
"""fact_history.py: память с историей изменений (би-темпоральная, как Zep/Graphiti).

ПАСПОРТ
  Зачем: факт о мире ("где Антон", "во что мы верим") не перезаписывается.
         Новое знание добавляет запись и ЗАКРЫВАЕТ старую датой, с причиной.
         Цепочка изменений остаётся целиком: что было, когда, почему сменилось.
  Две оси времени (Graphiti EntityEdge, getzep/graphiti 7e0342a, 03.11.2026):
    valid_at / invalid_at   = когда факт был верен В МИРЕ;
    recorded_at / expired_at = когда МЫ его записали / признали ошибкой.
  Правило закрытия = resolve_edge_contradictions из Graphiti
    (graphiti_core/utils/maintenance/edge_operations.py): новый факт того же
    subject+attribute с более поздним valid_at ставит старому invalid_at.
    Отличие от Graphiti (АК-47): противоречие = тот же ключ subject+attribute,
    без LLM; многозначный атрибут помечается --multi и не закрывается.
    Наше расширение: факт, пришедший "из прошлого" (valid_at раньше живого),
    встаёт на своё место в цепочке, живой не трогается. Равные valid_at:
    побеждает поздняя запись. Отличие: expired_at у нас = только отзыв ошибки,
    у Graphiti он ставится и при закрытии.
  Хранилище: append-only JSONL, один файл на узел (один писатель):
    <vault>/00-System/_fact-history/<HOST>.jsonl ; читатель сливает все файлы.
    Строки не правятся и не удаляются. invalid_at НЕ хранится: он выводится при
    чтении из всех файлов (следующий факт цепочки по оси мира закрывает предыдущий),
    поэтому два узла, писавшие не видя друг друга, сходятся к одной истории, а
    отзыв ошибочного факта возвращает жизнь предыдущему (ревью H33837, 03.11).
  Вход/выход:
    assert SUBJ ATTR VALUE --valid-at D --reason R [--source S] [--multi] [--until D]
    retract ID --reason R        (факт был ошибкой записи: expired_at, не invalid_at)
    now SUBJ [ATTR]              текущее значение
    history SUBJ ATTR            вся цепочка с причинами
    asof SUBJ ATTR D [--known-at K]  что было верно на дату D, как мы знали на K
    subjects                     список объектов
  Кто зовёт: сессии и рутины любого вендора (CLI, 0 LLM, 0 сети).
  Что ломается: нет --root и нет волта -> exit 2 с причиной; битая строка
    JSONL не роняет чтение, а печатается как "не могу судить N строк".
  Счётчик: ~/.claude/usage/fact_history.jsonl (ts·node·actor·event·outcome).
  Тест: _test_fact_history.py.  рельса: локально (0 LLM).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HOST = os.environ.get("COMPUTERNAME") or socket.gethostname()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse(d: str | None) -> datetime | None:
    if not d:
        return None
    s = d.strip().replace("Z", "+00:00")
    if len(s) == 10:
        s += "T00:00:00+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso(d: str | None) -> str | None:
    dt = _parse(d)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else None


def default_root() -> Path | None:
    v = os.environ.get("OBSIDIAN_VAULT")
    if not v:
        env = Path.home() / ".claude" / "machine.env"
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("OBSIDIAN_VAULT="):
                    v = line.split("=", 1)[1].strip().strip('"')
    return Path(v) / "00-System" / "_fact-history" if v else None


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.bad = 0
        self.lines = 0

    def _append(self, row: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row.setdefault("node", HOST)
        row.setdefault("recorded_at", _now())
        with open(self.root / f"{HOST}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rows(self) -> list[dict]:
        out = []
        self.bad = self.lines = 0
        if not self.root.exists():
            return out
        for p in sorted(self.root.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                self.lines += 1
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    self.bad += 1
        return out

    def facts(self, known_at: datetime | None = None) -> dict[str, dict]:
        """Свёртка при ЧТЕНИИ из всех файлов узлов. known_at режет по оси записи.

        invalid_at не хранится, а выводится: в однозначной цепочке subject+attribute
        (без отозванных и без multi) факт закрывается valid_at следующего по оси мира.
        Так две машины, писавшие не видя друг друга, сходятся к одной истории,
        а отзыв ошибочного факта возвращает жизнь предыдущему.
        """
        facts: dict[str, dict] = {}
        retracts = []
        for r in self.rows():
            rec = _parse(r.get("recorded_at"))
            if known_at and rec and rec > known_at:
                continue
            if r.get("kind") == "fact":
                facts[r["id"]] = dict(r, invalid_at=r.get("until"), expired_at=None, closed_by=None,
                                      close_reason=None, retract_reason=None)
            elif r.get("kind") == "retract":
                retracts.append(r)
        for e in retracts:
            f = facts.get(e.get("target"))
            if f and f["expired_at"] is None:
                f["expired_at"] = e.get("recorded_at")
                f["retract_reason"] = e.get("reason")
        chains: dict[tuple, list] = {}
        for f in facts.values():
            if f["expired_at"] is None and not f.get("multi"):
                chains.setdefault((f["subject"], f["attribute"]), []).append(f)
        for ch in chains.values():
            ch.sort(key=lambda f: (_parse(f["valid_at"]), f.get("recorded_at") or "", f["id"]))
            for prev, nxt in zip(ch, ch[1:]):
                own = _parse(prev.get("until"))
                if own is not None and own <= _parse(nxt["valid_at"]):
                    continue  # сам кончился раньше, чем пришёл следующий: щель в истории
                prev["invalid_at"] = nxt["valid_at"]  # равные valid_at: побеждает поздняя запись
                prev["closed_by"] = nxt["id"]
                prev["close_reason"] = nxt.get("reason")
        return facts

    def chain(self, subj: str, attr: str, known_at: datetime | None = None) -> list[dict]:
        fs = [f for f in self.facts(known_at).values()
              if f["subject"] == subj and f["attribute"] == attr]
        return sorted(fs, key=lambda f: (_parse(f["valid_at"]), f.get("recorded_at") or "", f["id"]))

    def assert_fact(self, subj, attr, value, valid_at, reason, source=None,
                    multi=False, actor="session", until=None) -> tuple[dict, list[dict]]:
        """Дописывает факт. Возвращает (факт после свёртки, кого он закрыл)."""
        va = _iso(valid_at) or _now()
        fid = uuid.uuid4().hex[:12]
        self._append({"kind": "fact", "id": fid, "subject": subj, "attribute": attr,
                      "value": value, "valid_at": va, "reason": reason, "source": source,
                      "multi": multi, "actor": actor, "until": _iso(until)})
        allf = self.facts()
        closed = [f for f in allf.values() if f["closed_by"] == fid]
        return allf[fid], closed

    def retract(self, fid: str, reason: str, actor="session") -> bool:
        if fid not in self.facts():
            return False
        self._append({"kind": "retract", "target": fid, "reason": reason, "actor": actor})
        return True


def _live_at(f: dict, at: datetime) -> bool:
    va, ia = _parse(f["valid_at"]), _parse(f["invalid_at"])
    return f["expired_at"] is None and (va is None or va <= at) and (ia is None or at < ia)


def _fmt(f: dict) -> str:
    end = f["invalid_at"][:10] if f["invalid_at"] else "сейчас"
    if f["invalid_at"] and not f.get("closed_by") and f["expired_at"] is None:
        end += " (кончился сам, замены нет)"
    s = f"[{f['id']}] {f['valid_at'][:10]} → {end}: {f['value']}"
    s += f"\n    почему: {f.get('reason') or '-'}"
    if f.get("source"):
        s += f" · источник: {f['source']}"
    s += f"\n    записал: {f.get('node')} {f.get('recorded_at', '')[:16]}"
    if f.get("close_reason") and f["invalid_at"]:
        s += f"\n    закрыт фактом {f.get('closed_by')}: {f['close_reason']}"
    if f["expired_at"]:
        s += f"\n    ⛔ ОТОЗВАН {f['expired_at'][:16]}: {f.get('retract_reason')}"
    return s


def _count(event: str, outcome: str, actor: str) -> None:
    try:
        p = Path.home() / ".claude" / "usage" / "fact_history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "node": HOST, "actor": actor,
                                "event": event, "outcome": outcome}) + "\n")
    except OSError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="папка хранилища (по умолчанию волт/00-System/_fact-history)")
    ap.add_argument("--actor", default="session")
    ap.add_argument("--json", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("assert"); a.add_argument("subject"); a.add_argument("attribute"); a.add_argument("value")
    a.add_argument("--valid-at"); a.add_argument("--reason", required=True); a.add_argument("--source")
    a.add_argument("--multi", action="store_true")
    a.add_argument("--until", help="известная дата конца без замены (бронь до 30.10); как invalid_at у Graphiti")
    r = sp.add_parser("retract"); r.add_argument("id"); r.add_argument("--reason", required=True)
    n = sp.add_parser("now"); n.add_argument("subject"); n.add_argument("attribute", nargs="?")
    h = sp.add_parser("history"); h.add_argument("subject"); h.add_argument("attribute")
    o = sp.add_parser("asof"); o.add_argument("subject"); o.add_argument("attribute"); o.add_argument("date")
    o.add_argument("--known-at")
    sp.add_parser("subjects")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    root = Path(args.root) if args.root else default_root()
    if root is None:
        print("fact_history: не нашёл волт (OBSIDIAN_VAULT / machine.env); дай --root", file=sys.stderr)
        return 2
    for d in (getattr(args, "valid_at", None), getattr(args, "date", None), getattr(args, "known_at", None),
              getattr(args, "until", None)):
        try:
            _parse(d)
        except ValueError:
            print(f"fact_history: дата {d!r} не ISO (нужно ГГГГ-ММ-ДД[THH:MM[+ZZ:ZZ]])", file=sys.stderr)
            return 2
    st = Store(root)
    out: object = None
    rc = 0
    if args.cmd == "assert":
        fact, closes = st.assert_fact(args.subject, args.attribute, args.value, args.valid_at,
                                      args.reason, args.source, args.multi, args.actor, args.until)
        out = {"fact": fact, "closed": [c["id"] for c in closes]}
        if not args.json:
            print(f"записал {fact['id']}: {args.subject}.{args.attribute} = {args.value} с {fact['valid_at'][:10]}")
            for c in closes:
                print(f"  закрыл старый {c['id']} ({c['value']}) датой {c['invalid_at'][:10]}")
            if fact["invalid_at"] and fact.get("closed_by"):
                print(f"  факт из прошлого: верен до {fact['invalid_at'][:10]} (дальше уже есть новее)")
            elif fact["invalid_at"]:
                print(f"  известный конец: верен до {fact['invalid_at'][:10]}")
            out = None
    elif args.cmd == "retract":
        ok = st.retract(args.id, args.reason, args.actor)
        rc = 0 if ok else 1
        if not args.json:
            print("отозван" if ok else f"нет факта {args.id}")
        else:
            out = {"retracted": ok}
    elif args.cmd in ("now", "asof"):
        at = _parse(args.date) if args.cmd == "asof" else datetime.now(timezone.utc)
        known = _parse(getattr(args, "known_at", None))
        fs = [f for f in st.facts(known).values() if f["subject"] == args.subject
              and (args.attribute is None or f["attribute"] == args.attribute) and _live_at(f, at)]
        out = sorted(fs, key=lambda f: (f["attribute"], f["valid_at"]))
        if not args.json:
            print("\n".join(f"{f['attribute']}: {_fmt(f)}" for f in out) or "нет живого факта на эту дату")
            out = None
    elif args.cmd == "history":
        out = st.chain(args.subject, args.attribute)
        if not args.json:
            print(f"история {args.subject}.{args.attribute}: {len(out)} версий")
            for f in out:
                print(_fmt(f))
            out = None
    elif args.cmd == "subjects":
        seen: dict[str, set] = {}
        for f in st.facts().values():
            seen.setdefault(f["subject"], set()).add(f["attribute"])
        out = {k: sorted(v) for k, v in sorted(seen.items())}
        if not args.json:
            for k, v in out.items():
                print(f"{k}: {', '.join(v)}")
            out = None
    if out is not None:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    if st.bad:
        print(f"⚠️ не могу судить {st.bad} из {st.lines} строк (битый JSON)", file=sys.stderr)
    _count(args.cmd, "ok" if rc == 0 else "fail", args.actor)
    return rc


if __name__ == "__main__":
    sys.exit(main())
