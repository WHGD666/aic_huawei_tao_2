import sys; sys.path.insert(0,'.')
import tempfile, pathlib, types, inspect, traceback
pt = types.ModuleType('pytest')
class _A:
    def __init__(s,v,rel=1e-6,ab=1e-9): s.v=v; s.rel=rel; s.ab=ab
    def __eq__(s,o): return abs(o-s.v) <= max(s.rel*abs(s.v), s.ab)
    def __repr__(s): return f"approx({s.v})"
pt.approx=lambda v,rel=1e-6,abs=1e-9:_A(v,rel,abs); sys.modules['pytest']=pt
class _Raises:
    def __init__(s,exc): s.exc=exc
    def __enter__(s): return s
    def __exit__(s,t,v,tb):
        if t is None: raise AssertionError(f"did not raise {s.exc}")
        return issubclass(t, s.exc)
pt.raises=_Raises
p=f=0
for mod in sys.argv[1:]:
    T = __import__(mod, fromlist=['x'])
    for n,fn in inspect.getmembers(T, inspect.isfunction):
        if not n.startswith('test_'): continue
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d)) if 'tmp_path' in inspect.signature(fn).parameters else fn()
            p+=1; print('PASS', n)
        except Exception:
            f+=1; print('FAIL', n); traceback.print_exc()
print(f"\n{p} passed, {f} failed")
