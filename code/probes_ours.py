# -*- coding: utf-8 -*-
"""probes_ours.py: room-402 scenario (S1 four changes, S2 late fact, S3 retraction) on fact_history.

Machine: ZBOOKG8-2023PAL · Agent: Claude Code Opus 5.5 (Z14128) · Operator: Anton · 2026-11-08
Same events and checks as probes_graphiti_core.py (Graphiti upstream temporal core, oracle candidates).
--broken: read-time closure disabled (every fact stays open), the run must turn red.
Output: JSON receipt on stdout, rc 0 PASS / 1 FAIL. рельса: локально, 0 LLM, 0 сети.
"""
import argparse, json, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fact_history as fh  # noqa: E402


def at(d):
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc)


def main(broken: bool) -> int:
    root = Path(tempfile.mkdtemp(prefix="probe402-"))
    st = fh.Store(root)
    if broken:
        orig = st.facts

        def no_close(known_at=None):
            fs = orig(known_at)
            for f in fs.values():
                if f.get("closed_by"):
                    f["invalid_at"], f["closed_by"] = None, None
            return fs
        st.facts = no_close
    S, A = "room-402", "guest"
    ids = {}

    def add(who, d):
        f, _ = st.assert_fact(S, A, who, d, reason="pms event")
        ids[who + d] = f["id"]

    def state(d):
        return sorted(f["value"] for f in st.facts().values()
                      if f["subject"] == S and f["attribute"] == A and fh._live_at(f, at(d)))
    t0 = time.perf_counter()
    for who, d in (("vacant", "2026-10-01"), ("Ivanov", "2026-10-05"), ("vacant", "2026-10-09"), ("Petrova", "2026-10-20")):
        add(who, d)
    s1 = {d: state(d) for d in ("2026-10-03", "2026-10-07", "2026-10-15", "2026-10-25")}
    w1 = {"2026-10-03": ["vacant"], "2026-10-07": ["Ivanov"], "2026-10-15": ["vacant"], "2026-10-25": ["Petrova"]}
    add("Sidorov", "2026-10-12")
    s2 = {d: state(d) for d in ("2026-10-10", "2026-10-13", "2026-10-25")}
    st.retract(ids["Petrova2026-10-20"], reason="operator error")
    s3 = state("2026-10-25")
    ms = (time.perf_counter() - t0) * 1000
    res = {"system": "fact_history (ours)", "mode": "broken (no read-time closure)" if broken else "normal",
           "checks": {"S1_four_changes_asof": {"got": s1, "ok": s1 == w1},
                      "S2_late_fact": {"got": s2, "ok": s2["2026-10-13"] == ["Sidorov"] and s2["2026-10-25"] == ["Petrova"]
                                       and s2["2026-10-10"] == ["vacant"]},
                      "S3_retract_last": {"got_now": s3, "ok": s3 == ["Sidorov"]}},
           "wall_ms": round(ms, 2)}
    res["passed"] = sum(v["ok"] for v in res["checks"].values())
    res["status"] = "PASS" if res["passed"] == 3 else "FAIL"
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--broken", action="store_true")
    sys.exit(main(ap.parse_args().broken))
