"""
backend.py -- 选择张量后端：有真实 PyTorch 就用它，否则用 tiny_torch（numpy 自动微分替身）。

任务包替身、门禁冒烟、代理训练统一 `from baseline_stub.common.backend import torch, HAS_TORCH`。
候选函数源码里只写 `torch.xxx`，两边都能跑。
"""
from __future__ import annotations

try:
    import torch as _torch  # type: ignore
    torch = _torch
    HAS_TORCH = True
except Exception:  # ModuleNotFoundError or broken install
    import sys as _sys
    from . import tiny_torch as _tt
    torch = _tt
    HAS_TORCH = False
    # 候选函数源码里写的是 `import torch`；无 torch 时让它解析到替身，源码两边保持一致
    _sys.modules.setdefault("torch", _tt)

__all__ = ["torch", "HAS_TORCH"]
