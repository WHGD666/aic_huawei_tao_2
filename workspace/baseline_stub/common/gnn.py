"""
gnn.py -- 固定的图神经网络（结构、归一化、优化器、训练预算都不可演化）。

两层均值聚合 + 残差 + 输出投影。参数只是一组张量，手写 Adam，这样真实 torch 与 tiny_torch 都能跑。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .backend import torch, HAS_TORCH


def _gen(seed: int):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class TinyGNN:
    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int, n_layers: int = 2):
        g = _gen(seed)
        self.params: List = []

        def W(a, b):
            t = torch.randn(a, b, generator=g) * float(np.sqrt(2.0 / (a + b)))
            t = t.detach() if HAS_TORCH else t
            t.requires_grad_(True)
            self.params.append(t)
            return t

        def B(b):
            t = torch.zeros(b)
            t.requires_grad_(True)
            self.params.append(t)
            return t

        self.W_in, self.b_in = W(in_dim, hidden), B(hidden)
        self.layers = [(W(hidden, hidden), B(hidden)) for _ in range(n_layers)]
        self.W_out, self.b_out = W(hidden, out_dim), B(out_dim)

    def forward(self, x, edge_index, n: int):
        h = torch.relu(x @ self.W_in + self.b_in)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n).index_add(0, dst, torch.ones(src.shape[0])) + 1.0
        for Wl, bl in self.layers:
            agg = torch.zeros(n, h.shape[1]).index_add(0, dst, h[src]) / deg.unsqueeze(1)
            h = torch.relu((h + agg) @ Wl + bl) + h
        return h @ self.W_out + self.b_out

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam:
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = [np.zeros(tuple(p.shape), dtype=np.float32) for p in params]
        self.v = [np.zeros(tuple(p.shape), dtype=np.float32) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = _np(p.grad)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            upd = self.lr * mh / (np.sqrt(vh) + self.eps)
            if HAS_TORCH:
                with torch.no_grad():
                    p -= torch.as_tensor(upd, dtype=p.dtype)
            else:
                p.data = p.data - upd.astype(np.float32)


def _np(t) -> np.ndarray:
    if HAS_TORCH:
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t.data, dtype=np.float32)


def to_numpy(t) -> np.ndarray:
    return _np(t)


def normalize_features(x: np.ndarray) -> np.ndarray:
    """固定的表示归一化：按算例、按列做 z-score（单位无关）。"""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)
