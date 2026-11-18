"""
fake_llm.py -- 测试用的脚本化 LLM：不联网、零费用。按调用顺序返回预置响应；可注入故障验证 gate 与修复逻辑。
目标基线 = baseline_stub/prob（compute_loss）与 baseline_stub/embed（compute_loss + decode_partition）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from harness.budget import Usage

# 基线 prob/loss.py 中逐字存在的片段
_BAL = "    l_bal = (ratio - 0.5) ** 2\n"
_BIN = "    l_bin = (p * (1.0 - p)).mean()\n"
_RET = "    return l_cut + 10.0 * l_bal + 1.0 * l_bin\n"


def cand(hyp: str, search: str, replace: str, n: int = 1, file: str = "loss.py", kw: str = "balance, hinge") -> str:
    return (f"CANDIDATE {n}\nHYPOTHESIS: {hyp}\nEXPECTED: TNS down a few percent on all circuits; repair flips down.\nKEYWORDS: {kw}\n"
            f"<<<EDIT file={file}\n<<<SEARCH\n{search.rstrip()}\n===\n{replace.rstrip()}\n>>>\n>>>\n")


def hinge(coef: float, n: int = 1) -> str:
    return cand(f"Balance hinge with coefficient {coef} keeps the raw partition inside the dead-band so repair barely moves nodes.",
                _BAL, f"    l_bal = (ratio - 0.5) ** 2 + {coef} * torch.clamp(torch.abs(ratio - 0.5) - 0.01, min=0.0) ** 2\n", n)


def crit_weight(alpha: float, n: int = 1) -> str:
    return cand(f"Criticality-weighted cut with alpha={alpha}: slack-rank quantile per node, max over pins per net.",
                _RET,
                "    slack = feats[:, 0]\n"
                "    rank = torch.argsort(torch.argsort(slack)).float()\n"
                "    crit = (1.0 - rank / float(max(graph['n_nodes'] - 1, 1))) ** 2\n"
                "    cnt = torch.zeros(n_nets).index_add(0, net_of_pin, torch.ones(net_nodes.shape[0]))\n"
                "    net_crit = torch.zeros(n_nets).index_add(0, net_of_pin, crit[net_nodes]) / torch.clamp(cnt, 1.0)\n"
                f"    w = 1.0 + {alpha} * net_crit\n"
                "    l_wcut = (w * cut_prob).sum() / w.sum()\n"
                "    return l_wcut + 10.0 * l_bal + 1.0 * l_bin\n", n, kw="slack, crit, rank")


GOOD = hinge(20.0)
GOOD2 = crit_weight(3.0)
TWO = hinge(20.0, 1) + crit_weight(3.0, 2)
BAD_SEARCH = cand("hinge", "    l_bal = (ratio - 0.50) ** 2\n", "    l_bal = 2.0 * (ratio - 0.5) ** 2\n")
FIXED_SEARCH = hinge(5.0)
ARGMAX = cand("hard threshold", _BIN, "    hard = (p > 0.5).float()\n    l_bin = ((p - hard.detach()) ** 2).mean()\n")
SIGNATURE = ("CANDIDATE 1\nHYPOTHESIS: add alpha arg\nEXPECTED: n/a\nKEYWORDS: x\n<<<EDIT file=loss.py\n<<<SEARCH\n"
             "def compute_loss(p, graph, feats):\n===\ndef compute_loss(p, graph, feats, alpha=2.0):\n>>>\n>>>\n")
IMPORT = cand("use scipy", "    net_ptr = graph[\"net_ptr\"]\n", "    import scipy.sparse as sp\n    net_ptr = graph[\"net_ptr\"]\n")
ABS_THRESH = cand("absolute slack threshold", _RET, "    crit = (feats[:, 0] < 0.1).float()\n    return l_cut + 10.0 * l_bal + 1.0 * l_bin + 0.0 * crit.sum()\n")
SYNTAX_ERR = cand("broken", _BIN, "    l_bin = (p * (1.0 - p).mean()\n")
NO_CHANGE = cand("noop", _BIN, _BIN)
ANALYSIS = "Hypothesis partially held: TNS improved on two circuits; seed spread acceptable. Next: scale coefficient x2."


class FakeLLM:
    def __init__(self, script: List[str], usage: Usage = Usage(2500, 400, 1800)):
        self.script = list(script)
        self.usage = usage
        self.calls: List[Tuple[str, List[Dict[str, str]]]] = []
        self._n_auto = 0

    def chat(self, messages, temperature=0.7, max_tokens=2048, tag=""):
        self.calls.append((tag, messages))
        if tag == "analyze":
            return ANALYSIS, Usage(400, 60, 0)
        if not self.script:
            return self._auto(messages), self.usage
        return self.script.pop(0), self.usage

    def _auto(self, messages) -> str:
        """脚本用尽：像会读代码的模型一样，从展示的代码里找当前 hinge 系数并提出一个新值（能叠在父代上）。"""
        shown = "\n".join(m["content"] for m in messages if m["role"] == "user")
        i = shown.find("## Current code")
        region = shown[i:] if i >= 0 else shown
        m = re.search(r"^    l_bal = \(ratio - 0\.5\) \*\* 2 \+ ([0-9.]+) \* torch\.clamp", region, re.M)
        self._n_auto += 1
        if m:
            cur = m.group(1)
            new = f"{float(cur) * 1.5 + self._n_auto:.1f}"
            search = f"    l_bal = (ratio - 0.5) ** 2 + {cur} * torch.clamp(torch.abs(ratio - 0.5) - 0.01, min=0.0) ** 2\n"
            repl = search.replace(f"+ {cur} *", f"+ {new} *")
            return cand(f"Scale hinge coefficient {cur} -> {new}.", search, repl)
        return hinge(10.0 + self._n_auto)


def varied(n: int) -> List[str]:
    return [hinge(10.0 + 5 * i) for i in range(n)]
