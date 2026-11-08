# -*- coding: utf-8 -*-
"""Wilson CI against published reference values; must go red if the formula is broken (BROKEN=1)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as M
if os.environ.get("BROKEN"):
    M.wilson = lambda k, n, z=1.96: (k / n, k / n - 0.01, k / n + 0.01)  # naive band, deliberately wrong
cases = [((81, 263), (0.2553, 0.3662)),   # Wilson 1927 / Newcombe 1998 worked example
         ((0, 10), (0.0, 0.2775)),
         ((30, 30), (0.8865, 1.0))]
bad = 0
for (k, n), (lo, hi) in cases:
    p, a, b = M.wilson(k, n)
    ok = abs(a - lo) < 5e-4 and abs(b - hi) < 5e-4
    bad += not ok
    print(("ok  " if ok else "FAIL"), k, n, round(a, 4), round(b, 4))
sys.exit(1 if bad else 0)
