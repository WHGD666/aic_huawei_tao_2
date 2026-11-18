"""跑 tests.test_e2e 的一个子集（沙箱单命令时限短时用）。用法: python tests/_run_subset.py test_a test_b ..."""
import sys, tempfile, pathlib, types, inspect, traceback
sys.path.insert(0, '.')
pt = types.ModuleType('pytest')
class _A:
    def __init__(s, v, rel=1e-6, ab=1e-9): s.v = v; s.rel = rel; s.ab = ab
    def __eq__(s, o): return abs(o - s.v) <= max(s.rel * abs(s.v), s.ab)
pt.approx = lambda v, rel=1e-6, abs=1e-9: _A(v, rel, abs); sys.modules['pytest'] = pt
import tests.test_e2e as T
p = f = 0
for n in sys.argv[1:]:
    fn = getattr(T, n)
    try:
        with tempfile.TemporaryDirectory() as d:
            fn(pathlib.Path(d))
        p += 1; print('PASS', n, flush=True)
    except Exception:
        f += 1; print('FAIL', n); traceback.print_exc()
print(f"{p} passed, {f} failed")
