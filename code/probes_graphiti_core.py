# -*- coding: utf-8 -*-
"""Один объект, много событий во времени: Graphiti temporal core против нашего fact_history.

Machine: A-2022BAYAREA (Хаб) · Agent: Claude Code · Operator: Anton · 2026-11-05
Сессия H35889, голосовая 001 xcae52950430309 (материалы к встречам Zep / Mem0 / Anima).

Что меряем. Номер отеля room-402 проживает 4 смены жильца, ремонт, поздно пришедший факт и
отзыв ошибочной записи. Graphiti: настоящий upstream `_apply_temporal_invalidation` +
встроенная Kuzu, без LLM. Кандидатов на противоречие подаём «идеальным судьёй» (все прежние
рёбра того же отношения): в бою их выбирает LLM, и это отдельный риск (getzep/graphiti#1728).
Значит, этот прогон = потолок Graphiti, а не его средний случай.

Запуск: venv\\Scripts\\python.exe timeline_H35889.py [--broken]
--broken выключает закрытие интервалов: сценарий обязан покраснеть (red proof).
Выход: JSON-квитанция в stdout, rc=0 PASS / rc=1 FAIL.
рельса: локальный Python, 0 LLM, 0 сети.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch

os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
from graphiti_core.driver.kuzu_driver import KuzuDriver  # noqa: E402
from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.nodes import EntityNode  # noqa: E402
from graphiti_core.utils.maintenance import edge_operations  # noqa: E402
import kuzu  # noqa: E402

G = 'room-402-timeline'


def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


async def main(broken):
    orig = kuzu.Database
    with patch.object(kuzu, 'Database', side_effect=lambda *a, **kw: orig(
            *a, buffer_pool_size=64 * 1024 * 1024, **kw)):
        driver = KuzuDriver(':memory:')
    room = EntityNode(name='Room 402', group_id=G)
    nodes = {n: EntityNode(name=n, group_id=G) for n in ('Vacant', 'Ivanov', 'Petrova', 'Sidorov')}
    await room.save(driver)
    for n in nodes.values():
        await n.save(driver)
    edges = []

    def apply(new):
        cands = [e for e in edges if e.name == new.name and e.uuid != new.uuid]
        if broken:
            return []
        return edge_operations._apply_temporal_invalidation(new, cands)

    async def add(who, valid, fact, rel='HAS_GUEST'):
        e = EntityEdge(source_node_uuid=room.uuid, target_node_uuid=nodes[who].uuid, group_id=G,
                       name=rel, fact=fact, valid_at=dt(valid), created_at=datetime.now(timezone.utc),
                       episodes=[f'pms-{valid}'])
        inval = apply(e)
        for old in inval:
            await old.save(driver)
        await e.save(driver)
        edges.append(e)
        return e

    def state_at(t, rel='HAS_GUEST'):
        t = dt(t)
        live = [e for e in edges if e.name == rel and e.valid_at <= t
                and (e.invalid_at is None or e.invalid_at > t)]
        return sorted(e.fact for e in live)

    res = {'checks': {}, 'graphiti_core': None}
    from importlib.metadata import version
    res['graphiti_core'] = version('graphiti-core')

    # S1: четыре смены жильца по порядку
    await add('Vacant', '2026-10-01', 'Room 402 is vacant')
    await add('Ivanov', '2026-10-05', 'Ivanov checked into room 402')
    await add('Vacant', '2026-10-09', 'Room 402 is vacant again')
    p = await add('Petrova', '2026-10-20', 'Petrova checked into room 402')
    s1 = {d: state_at(d) for d in ('2026-10-03', '2026-10-07', '2026-10-15', '2026-10-25')}
    want1 = {'2026-10-03': ['Room 402 is vacant'], '2026-10-07': ['Ivanov checked into room 402'],
             '2026-10-15': ['Room 402 is vacant again'], '2026-10-25': ['Petrova checked into room 402']}
    res['checks']['S1_four_changes_asof'] = {'got': s1, 'ok': s1 == want1}

    # Сохранность истории в настоящей базе: все 4 ребра читаются обратно
    back = [await EntityEdge.get_by_uuid(driver, e.uuid) for e in edges]
    res['checks']['S1_history_persisted'] = {'edges_in_db': len(back), 'ok': len(back) == 4}

    # S2: поздно пришедший факт (узнали 05.11, что 12-14.10 жил Sidorov)
    await add('Sidorov', '2026-10-12', 'Sidorov stayed in room 402 (late report)')
    s2 = {d: state_at(d) for d in ('2026-10-10', '2026-10-13', '2026-10-25')}
    res['checks']['S2_late_fact'] = {
        'got': s2,
        'note': 'вставка в середину: 12.10 должен быть Sidorov, 25.10 по-прежнему Petrova',
        'ok': s2['2026-10-13'] == ['Sidorov stayed in room 402 (late report)']
        and s2['2026-10-25'] == ['Petrova checked into room 402']}

    # S3: отзыв ошибочной записи (Petrova внесена по ошибке оператора) = удалить ребро
    edges.remove(p)
    await p.delete(driver)
    s3 = state_at('2026-10-25')
    res['checks']['S3_retract_last'] = {
        'got_now': s3,
        'expected_by_meaning': 'previous state (Sidorov), not an empty room',
        'ok': s3 == ['Sidorov stayed in room 402 (late report)']}

    await driver.close()
    allok = all(v['ok'] for v in res['checks'].values())
    res['status'] = 'PASS' if allok else 'FAIL'
    res['mode'] = 'broken (no invalidation)' if broken else 'oracle candidates, upstream temporal core'
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0 if allok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--broken', action='store_true')
    sys.exit(asyncio.run(main(ap.parse_args().broken)))
