"""
tiny_torch.py -- 一个极小的 numpy 反向自动微分，暴露 torch 风格的接口子集。

用途：在没有 PyTorch 的环境里（CI / 沙箱）让任务包替身、门禁的可微性冒烟、代理训练都能跑。
有真实 torch 时 `backend.py` 会优先用真实 torch，本文件不参与。

支持：广播四则运算、幂、matmul、sum/mean（dim）、exp/log/log1p/sigmoid/tanh/relu/softplus/abs/sqrt/
clamp、整数索引 gather、index_add（out-of-place）、where、logsumexp、cat/stack、max/min 归约、
no_grad、detach、item、backward。不支持高阶导。
"""
from __future__ import annotations

import builtins
import contextlib
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

float32 = np.float32
float64 = np.float64
int64 = np.int64
long = np.int64
bool_ = np.bool_
bool = np.bool_  # noqa: A001  (torch.bool)
_grad_enabled = True


class Generator:
    def __init__(self):
        self._rng = np.random.default_rng(0)

    def manual_seed(self, s: int):
        self._rng = np.random.default_rng(int(s))
        return self


def manual_seed(s: int):
    global _default_gen
    _default_gen = Generator().manual_seed(s)


_default_gen = Generator()


def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    if g.shape == shape:
        return g
    nd = g.ndim - len(shape)
    if nd > 0:
        g = g.sum(axis=tuple(range(nd)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g.reshape(shape)


def _as_array(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, requires_grad: bool = False, _prev: Sequence["Tensor"] = (), _op: str = ""):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        self.data = arr
        self.requires_grad = builtins.bool(requires_grad) and _grad_enabled
        self.grad: Optional["Tensor"] = None
        self._backward = lambda: None
        self._prev = tuple(p for p in _prev if isinstance(p, Tensor))
        self._op = _op

    # ---------- 基本属性 ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def device(self):
        return "cpu"

    def dim(self):
        return self.data.ndim

    def size(self, d: Optional[int] = None):
        return self.data.shape if d is None else self.data.shape[d]

    def numel(self):
        return int(self.data.size)

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"tensor({self.data!r}{', requires_grad=True' if self.requires_grad else ''})"

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def clone(self):
        return _unary(self, self.data.copy(), lambda g: g, "clone")

    def cpu(self):
        return self

    def to(self, *a, **k):
        return self

    def float(self):
        return _unary(self, self.data.astype(np.float32), lambda g: g, "float")

    def double(self):
        return self.float()

    def long(self):
        return Tensor(self.data.astype(np.int64))

    def bool(self):
        return Tensor(self.data.astype(np.bool_))

    def requires_grad_(self, flag: bool = True):
        self.requires_grad = flag
        return self

    def zero_(self):
        self.data[...] = 0
        return self

    # ---------- autograd ----------
    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        topo: List[Tensor] = []
        seen = set()

        def build(v: Tensor):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        g0 = np.ones_like(self.data, dtype=np.float32) if gradient is None else _as_array(gradient).astype(np.float32)
        self.grad = Tensor(g0)
        for v in reversed(topo):
            if v.requires_grad and v.grad is not None:
                v._backward()

    def _accum(self, g: np.ndarray):
        if not self.requires_grad:
            return
        g = _unbroadcast(np.asarray(g, dtype=np.float32), self.data.shape)
        if self.grad is None:
            self.grad = Tensor(g.copy())
        else:
            self.grad.data = self.grad.data + g

    # ---------- 算子 ----------
    def __add__(self, o):
        return _binary(self, o, lambda a, b: a + b, lambda g, a, b: (g, g), "add")

    __radd__ = __add__

    def __sub__(self, o):
        return _binary(self, o, lambda a, b: a - b, lambda g, a, b: (g, -g), "sub")

    def __rsub__(self, o):
        return _binary(o, self, lambda a, b: a - b, lambda g, a, b: (g, -g), "rsub")

    def __mul__(self, o):
        return _binary(self, o, lambda a, b: a * b, lambda g, a, b: (g * b, g * a), "mul")

    __rmul__ = __mul__

    def __truediv__(self, o):
        return _binary(self, o, lambda a, b: a / b, lambda g, a, b: (g / b, -g * a / (b * b)), "div")

    def __rtruediv__(self, o):
        return _binary(o, self, lambda a, b: a / b, lambda g, a, b: (g / b, -g * a / (b * b)), "rdiv")

    def __neg__(self):
        return _unary(self, -self.data, lambda g: -g, "neg")

    def __pow__(self, e):
        if isinstance(e, Tensor):
            return _binary(self, e, lambda a, b: np.power(a, b),
                           lambda g, a, b: (g * b * np.power(a, b - 1), g * np.power(a, b) * np.log(np.maximum(a, 1e-30))), "pow")
        e = float(e)
        return _unary(self, np.power(self.data, e), lambda g: g * e * np.power(self.data, e - 1), "pow")

    def __matmul__(self, o):
        return matmul(self, o)

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        if isinstance(idx, tuple):
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        out_data = self.data[idx]

        def bw(g):
            full = np.zeros_like(self.data, dtype=np.float32)
            np.add.at(full, idx, g)
            return full

        return _unary(self, out_data, bw, "index")

    # 比较（不可微；返回 bool Tensor）
    def __lt__(self, o): return Tensor(self.data < _as_array(o))
    def __le__(self, o): return Tensor(self.data <= _as_array(o))
    def __gt__(self, o): return Tensor(self.data > _as_array(o))
    def __ge__(self, o): return Tensor(self.data >= _as_array(o))
    def __eq__(self, o): return Tensor(self.data == _as_array(o))  # type: ignore[override]
    def __ne__(self, o): return Tensor(self.data != _as_array(o))  # type: ignore[override]
    __hash__ = object.__hash__

    def __invert__(self): return Tensor(~self.data)
    def __and__(self, o): return Tensor(self.data & _as_array(o))
    def __or__(self, o): return Tensor(self.data | _as_array(o))

    # ---------- 归约 / 逐元素（方法形式） ----------
    def sum(self, dim=None, keepdim=False): return sum(self, dim, keepdim)
    def mean(self, dim=None, keepdim=False): return mean(self, dim, keepdim)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def sqrt(self): return sqrt(self)
    def abs(self): return abs(self)
    def sigmoid(self): return sigmoid(self)
    def tanh(self): return tanh(self)
    def relu(self): return relu(self)
    def clamp(self, min=None, max=None): return clamp(self, min, max)
    def clamp_min(self, m): return clamp(self, m, None)
    def clamp_max(self, m): return clamp(self, None, m)
    def pow(self, e): return self ** e
    def square(self): return self ** 2
    def max(self, dim=None, keepdim=False): return max(self, dim, keepdim)
    def min(self, dim=None, keepdim=False): return min(self, dim, keepdim)
    def argmax(self, dim=None): return Tensor(np.argmax(self.data, axis=dim))
    def argmin(self, dim=None): return Tensor(np.argmin(self.data, axis=dim))
    def t(self): return transpose(self, 0, 1)
    def T(self): return transpose(self, 0, 1)
    def transpose(self, a, b): return transpose(self, a, b)
    def reshape(self, *shape): return reshape(self, shape)
    def view(self, *shape): return reshape(self, shape)
    def unsqueeze(self, d): return reshape(self, _ins(self.shape, d))
    def squeeze(self, d=None): return Tensor(np.squeeze(self.data, axis=d), requires_grad=False) if not self.requires_grad else _unary(self, np.squeeze(self.data, axis=d), lambda g: g.reshape(self.shape), "squeeze")
    def flatten(self): return reshape(self, (-1,))
    def expand(self, *shape): return _unary(self, np.broadcast_to(self.data, shape).copy(), lambda g: _unbroadcast(g, self.shape), "expand")
    def repeat(self, *reps): return _unary(self, np.tile(self.data, reps), lambda g: _tile_bw(g, self.shape, reps), "repeat")
    def masked_fill(self, mask, value): return where(mask, full_like(self, value), self)
    def index_add(self, dim, index, source): return index_add(self, dim, index, source)
    def scatter_add(self, dim, index, src): return index_add(self, dim, index, src)
    def norm(self, p=2, dim=None, keepdim=False): return sqrt(sum(self ** 2, dim, keepdim) + 1e-12)
    def softmax(self, dim=-1): return softmax(self, dim)
    def log_softmax(self, dim=-1): return log_softmax(self, dim)
    def any(self): return builtins.bool(self.data.any())
    def all(self): return builtins.bool(self.data.all())
    def isnan(self): return Tensor(np.isnan(self.data))
    def isinf(self): return Tensor(np.isinf(self.data))
    def isfinite(self): return Tensor(np.isfinite(self.data))
    def nonzero(self): return Tensor(np.argwhere(self.data))
    def fill_(self, v): self.data[...] = v; return self
    def copy_(self, o): self.data[...] = _as_array(o); return self
    def __iadd__(self, o): self.data = self.data + _as_array(o); return self
    def __isub__(self, o): self.data = self.data - _as_array(o); return self
    def __imul__(self, o): self.data = self.data * _as_array(o); return self
    def __itruediv__(self, o): self.data = self.data / _as_array(o); return self
    def __float__(self): return float(self.data)
    def __int__(self): return int(self.data)
    def __bool__(self): return builtins.bool(self.data)
    def __iter__(self): return iter(Tensor(x) for x in self.data)


def _ins(shape, d):
    s = list(shape)
    if d < 0:
        d = len(s) + 1 + d
    s.insert(d, 1)
    return tuple(s)


def _tile_bw(g, shape, reps):
    g = g.reshape([r for pair in zip(reps, shape) for r in pair])
    return g.sum(axis=tuple(range(0, 2 * len(shape), 2)))


def _unary(x: Tensor, out_data, bw, op) -> Tensor:
    out = Tensor(out_data, requires_grad=x.requires_grad, _prev=(x,), _op=op)

    def _b():
        x._accum(bw(out.grad.data))

    out._backward = _b
    return out


def _binary(a, b, f, bw, op) -> Tensor:
    ta, tb = (a if isinstance(a, Tensor) else Tensor(a)), (b if isinstance(b, Tensor) else Tensor(b))
    out = Tensor(f(ta.data, tb.data), requires_grad=(ta.requires_grad or tb.requires_grad), _prev=(ta, tb), _op=op)

    def _b():
        ga, gb = bw(out.grad.data, ta.data, tb.data)
        ta._accum(ga)
        tb._accum(gb)

    out._backward = _b
    return out


# ---------- 构造 ----------
def tensor(data, dtype=None, requires_grad=False, device=None):
    arr = np.asarray(data.data if isinstance(data, Tensor) else data)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr, requires_grad=requires_grad)


def as_tensor(data, dtype=None, device=None):
    return tensor(data, dtype)


def from_numpy(a):
    return Tensor(np.asarray(a))


def zeros(*shape, dtype=np.float32, requires_grad=False, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, dtype=np.float32, requires_grad=False, device=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def full(shape, v, dtype=np.float32, device=None):
    return Tensor(np.full(shape, v, dtype=dtype))


def zeros_like(x, dtype=None): return Tensor(np.zeros_like(_as_array(x), dtype=dtype or np.float32))
def ones_like(x, dtype=None): return Tensor(np.ones_like(_as_array(x), dtype=dtype or np.float32))
def full_like(x, v): return Tensor(np.full_like(_as_array(x), v, dtype=np.float32))
def arange(*a, device=None, dtype=None): return Tensor(np.arange(*a, dtype=dtype))


def randn(*shape, generator: Optional[Generator] = None, requires_grad=False, device=None, dtype=None):
    g = (generator or _default_gen)._rng
    return Tensor(g.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)


def rand(*shape, generator: Optional[Generator] = None, device=None, dtype=None):
    g = (generator or _default_gen)._rng
    return Tensor(g.random(shape).astype(np.float32))


def randperm(n, generator: Optional[Generator] = None, device=None):
    g = (generator or _default_gen)._rng
    return Tensor(g.permutation(n))


def randint(low, high, size, generator: Optional[Generator] = None, device=None):
    g = (generator or _default_gen)._rng
    return Tensor(g.integers(low, high, size))


# ---------- 函数式算子 ----------
def sum(x, dim=None, keepdim=False):  # noqa: A001
    x = x if isinstance(x, Tensor) else Tensor(x)
    out_data = x.data.sum(axis=dim, keepdims=keepdim)

    def bw(g):
        if dim is not None and not keepdim:
            g = np.expand_dims(g, dim)
        return np.broadcast_to(g, x.shape)

    return _unary(x, out_data, bw, "sum")


def mean(x, dim=None, keepdim=False):
    x = x if isinstance(x, Tensor) else Tensor(x)
    n = x.data.size if dim is None else x.data.shape[dim]
    return sum(x, dim, keepdim) / float(n)


def exp(x): x = _t(x); return _unary(x, np.exp(x.data), lambda g: g * np.exp(x.data), "exp")
def log(x): x = _t(x); return _unary(x, np.log(x.data), lambda g: g / x.data, "log")
def log1p(x): x = _t(x); return _unary(x, np.log1p(x.data), lambda g: g / (1 + x.data), "log1p")
def sqrt(x): x = _t(x); return _unary(x, np.sqrt(x.data), lambda g: g * 0.5 / np.sqrt(np.maximum(x.data, 1e-30)), "sqrt")
def abs(x): x = _t(x); return _unary(x, np.abs(x.data), lambda g: g * np.sign(x.data), "abs")  # noqa: A001
def square(x): return _t(x) ** 2
def neg(x): return -_t(x)


def sigmoid(x):
    x = _t(x)
    s = 1.0 / (1.0 + np.exp(-x.data))
    return _unary(x, s, lambda g: g * s * (1 - s), "sigmoid")


def tanh(x):
    x = _t(x)
    t = np.tanh(x.data)
    return _unary(x, t, lambda g: g * (1 - t * t), "tanh")


def relu(x):
    x = _t(x)
    return _unary(x, np.maximum(x.data, 0), lambda g: g * (x.data > 0), "relu")


def softplus(x, beta=1.0):
    x = _t(x)
    z = beta * x.data
    sp = np.where(z > 20, z, np.log1p(np.exp(np.minimum(z, 20)))) / beta
    return _unary(x, sp, lambda g: g * (1.0 / (1.0 + np.exp(-z))), "softplus")


def clamp(x, min=None, max=None):  # noqa: A002
    x = _t(x)
    lo = -np.inf if min is None else float(min)
    hi = np.inf if max is None else float(max)
    out = np.clip(x.data, lo, hi)
    return _unary(x, out, lambda g: g * ((x.data >= lo) & (x.data <= hi)), "clamp")


clip = clamp


def where(cond, a, b):
    c = _as_array(cond).astype(np.bool_)
    ta, tb = _t(a), _t(b)
    out = Tensor(np.where(c, ta.data, tb.data), requires_grad=(ta.requires_grad or tb.requires_grad), _prev=(ta, tb), _op="where")

    def _b():
        ta._accum(np.where(c, out.grad.data, 0))
        tb._accum(np.where(c, 0, out.grad.data))

    out._backward = _b
    return out


def maximum(a, b):
    ta, tb = _t(a), _t(b)
    return where(ta.data >= tb.data, ta, tb)


def minimum(a, b):
    ta, tb = _t(a), _t(b)
    return where(ta.data <= tb.data, ta, tb)


class _MaxMin:
    def __init__(self, values, indices):
        self.values, self.indices = values, indices

    def __iter__(self):
        return iter((self.values, self.indices))

    def __getitem__(self, i):
        return (self.values, self.indices)[i]


def max(x, dim=None, keepdim=False):  # noqa: A001
    x = _t(x)
    if dim is None:
        idx = np.argmax(x.data)
        out = _unary(x, x.data.reshape(-1)[idx], lambda g: _onehot_grad(x.shape, idx, g), "max")
        return out
    idx = np.argmax(x.data, axis=dim)
    vals = np.take_along_axis(x.data, np.expand_dims(idx, dim), dim)
    if not keepdim:
        vals = np.squeeze(vals, dim)

    def bw(g):
        gg = np.expand_dims(g, dim) if not keepdim else g
        full = np.zeros_like(x.data, dtype=np.float32)
        np.put_along_axis(full, np.expand_dims(idx, dim), gg, dim)
        return full

    return _MaxMin(_unary(x, vals, bw, "max"), Tensor(idx))


def min(x, dim=None, keepdim=False):  # noqa: A001
    x = _t(x)
    if dim is None:
        r = max(-x)
        return -r
    v, i = max(-x, dim, keepdim)
    return _MaxMin(-v, i)


def _onehot_grad(shape, flat_idx, g):
    full = np.zeros(int(np.prod(shape)), dtype=np.float32)
    full[flat_idx] = g
    return full.reshape(shape)


def amax(x, dim=None, keepdim=False):
    r = max(x, dim, keepdim)
    return r if dim is None else r.values


def amin(x, dim=None, keepdim=False):
    r = min(x, dim, keepdim)
    return r if dim is None else r.values


def logsumexp(x, dim=None, keepdim=False):
    x = _t(x)
    m = np.max(x.data, axis=dim, keepdims=True)
    return log(sum(exp(x - Tensor(m)), dim, keepdim=True)) + Tensor(m) if keepdim or dim is None and False else _lse(x, dim, keepdim, m)


def _lse(x, dim, keepdim, m):
    s = log(sum(exp(x - Tensor(m)), dim, keepdim=True)) + Tensor(m)
    if dim is None:
        return reshape(s, ())
    return s if keepdim else squeeze(s, dim)


def squeeze(x, dim=None):
    x = _t(x)
    return _unary(x, np.squeeze(x.data, axis=dim), lambda g: g.reshape(x.shape), "squeeze")


def softmax(x, dim=-1):
    x = _t(x)
    e = exp(x - Tensor(np.max(x.data, axis=dim, keepdims=True)))
    return e / sum(e, dim, keepdim=True)


def log_softmax(x, dim=-1):
    x = _t(x)
    return x - logsumexp(x, dim, keepdim=True)


def matmul(a, b):
    ta, tb = _t(a), _t(b)
    return _binary(ta, tb, lambda x, y: x @ y,
                   lambda g, x, y: (g @ np.swapaxes(y, -1, -2) if y.ndim > 1 else np.outer(g, y),
                                    np.swapaxes(x, -1, -2) @ g if x.ndim > 1 else np.outer(x, g)), "matmul")


mm = matmul


def transpose(x, a, b):
    x = _t(x)
    return _unary(x, np.swapaxes(x.data, a, b), lambda g: np.swapaxes(g, a, b), "transpose")


def reshape(x, shape):
    x = _t(x)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return _unary(x, x.data.reshape(shape), lambda g: g.reshape(x.shape), "reshape")


def cat(ts: Iterable[Tensor], dim=0):
    ts = [_t(t) for t in ts]
    out = Tensor(np.concatenate([t.data for t in ts], axis=dim), requires_grad=any(t.requires_grad for t in ts), _prev=tuple(ts), _op="cat")
    sizes = [t.data.shape[dim] for t in ts]

    def _b():
        off = 0
        for t, s in zip(ts, sizes):
            sl = [slice(None)] * out.data.ndim
            sl[dim] = slice(off, off + s)
            t._accum(out.grad.data[tuple(sl)])
            off += s

    out._backward = _b
    return out


def stack(ts: Iterable[Tensor], dim=0):
    return cat([unsqueeze(t, dim) for t in ts], dim)


def unsqueeze(x, d):
    x = _t(x)
    return reshape(x, _ins(x.shape, d))


def index_add(x, dim, index, source):
    """out-of-place: out = x.clone(); out[index] += source（沿 dim=0）。"""
    x, src = _t(x), _t(source)
    idx = _as_array(index).astype(np.int64)
    out_data = x.data.copy()
    np.add.at(out_data, idx, src.data)
    out = Tensor(out_data, requires_grad=(x.requires_grad or src.requires_grad), _prev=(x, src), _op="index_add")

    def _b():
        x._accum(out.grad.data)
        src._accum(out.grad.data[idx])

    out._backward = _b
    return out


def scatter_add(x, dim, index, src):
    return index_add(x, dim, index, src)


def gather(x, dim, index):
    x = _t(x)
    idx = _as_array(index)
    return x[idx] if dim == 0 and x.ndim == 1 else _unary(x, np.take_along_axis(x.data, idx, dim), lambda g: _gather_bw(x.shape, idx, dim, g), "gather")


def _gather_bw(shape, idx, dim, g):
    full = np.zeros(shape, dtype=np.float32)
    np.put_along_axis(full, idx, g, dim)
    return full


def bincount(x, weights=None, minlength=0):
    return Tensor(np.bincount(_as_array(x).astype(np.int64), weights=None if weights is None else _as_array(weights), minlength=minlength))


def unique(x, return_counts=False, return_inverse=False):
    r = np.unique(_as_array(x), return_counts=return_counts, return_inverse=return_inverse)
    return tuple(Tensor(a) for a in r) if isinstance(r, tuple) else Tensor(r)


def isnan(x): return Tensor(np.isnan(_as_array(x)))
def isinf(x): return Tensor(np.isinf(_as_array(x)))
def isfinite(x): return Tensor(np.isfinite(_as_array(x)))
def sort(x, dim=-1, descending=False):
    a = _as_array(x)
    idx = np.argsort(-a if descending else a, axis=dim, kind="stable")
    return _MaxMin(Tensor(np.take_along_axis(a, idx, dim)), Tensor(idx))
def argsort(x, dim=-1, descending=False): return sort(x, dim, descending).indices
def topk(x, k, dim=-1, largest=True):
    s = sort(x, dim, descending=largest)
    sl = [slice(None)] * _as_array(x).ndim
    sl[dim] = slice(0, k)
    return _MaxMin(Tensor(s.values.data[tuple(sl)]), Tensor(s.indices.data[tuple(sl)]))
def cumsum(x, dim=0):
    x = _t(x)
    return _unary(x, np.cumsum(x.data, axis=dim), lambda g: np.flip(np.cumsum(np.flip(g, dim), axis=dim), dim), "cumsum")
def quantile(x, q, dim=None): return Tensor(np.quantile(_as_array(x), q, axis=dim).astype(np.float32))
def median(x): return Tensor(np.median(_as_array(x)).astype(np.float32))
def std(x, dim=None, unbiased=True): return sqrt(var(x, dim, unbiased) + 1e-12)
def var(x, dim=None, unbiased=True):
    x = _t(x)
    m = mean(x, dim, keepdim=True) if dim is not None else mean(x)
    n = x.data.size if dim is None else x.data.shape[dim]
    return sum((x - m) ** 2, dim) / float(n - 1 if unbiased and n > 1 else n)
def dot(a, b): return sum(_t(a) * _t(b))
def outer(a, b): return unsqueeze(_t(a), 1) * unsqueeze(_t(b), 0)
def cdist(a, b):
    a, b = _t(a), _t(b)
    d = unsqueeze(a, 1) - unsqueeze(b, 0)
    return sqrt(sum(d * d, -1) + 1e-12)
def allclose(a, b, rtol=1e-5, atol=1e-8): return builtins.bool(np.allclose(_as_array(a), _as_array(b), rtol, atol))
def eye(n, device=None): return Tensor(np.eye(n, dtype=np.float32))
def linspace(a, b, n): return Tensor(np.linspace(a, b, n).astype(np.float32))
def diag(x): return Tensor(np.diag(_as_array(x)))
def numel(x): return int(_as_array(x).size)
def is_tensor(x): return isinstance(x, Tensor)
def is_grad_enabled(): return _grad_enabled


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old


class enable_grad:
    def __enter__(self):
        global _grad_enabled
        self.old = _grad_enabled
        _grad_enabled = True

    def __exit__(self, *a):
        global _grad_enabled
        _grad_enabled = self.old


class _NNFunctional:
    relu = staticmethod(relu)
    sigmoid = staticmethod(sigmoid)
    softplus = staticmethod(softplus)
    tanh = staticmethod(tanh)
    softmax = staticmethod(softmax)
    log_softmax = staticmethod(log_softmax)
    logsigmoid = staticmethod(lambda x: -softplus(-_t(x)))

    @staticmethod
    def normalize(x, p=2, dim=1, eps=1e-12):
        x = _t(x)
        return x / (x.norm(p, dim, keepdim=True) + eps)

    @staticmethod
    def binary_cross_entropy(p, y, reduction="mean"):
        p, y = _t(p), _t(y)
        l = -(y * log(clamp(p, 1e-7, 1 - 1e-7)) + (1 - y) * log(clamp(1 - p, 1e-7, 1 - 1e-7)))
        return mean(l) if reduction == "mean" else sum(l) if reduction == "sum" else l

    @staticmethod
    def mse_loss(a, b, reduction="mean"):
        d = (_t(a) - _t(b)) ** 2
        return mean(d) if reduction == "mean" else sum(d)

    @staticmethod
    def cosine_similarity(a, b, dim=1, eps=1e-8):
        a, b = _t(a), _t(b)
        return sum(a * b, dim) / (a.norm(2, dim) * b.norm(2, dim) + eps)


class _NN:
    functional = _NNFunctional()


nn = _NN()


class _Autograd:
    @staticmethod
    def grad(outputs, inputs, retain_graph=False, create_graph=False, allow_unused=False):
        outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        ins = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        for i in ins:
            i.grad = None
        for o in outs:
            o.backward()
        return tuple(i.grad for i in ins)


autograd = _Autograd()


def seed_all(s: int):
    manual_seed(s)
    np.random.seed(int(s) % (2 ** 32))
