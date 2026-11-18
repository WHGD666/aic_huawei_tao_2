"""
calibrate_pricing.py -- 用 /user/balance 的余额差反推真实单价（定价页打不开时的替代方案）。

做法：发 N 次固定大小的请求（一组全 cache-miss、一组全 cache-hit、一组大 output），
前后各查一次余额，最小二乘解三个单价。总花费约 0.05 元。余额精度是 0.01 元，所以每组要
攒到 ≥ 0.02 元才可分辨 —— 脚本会按估计单价自动决定次数。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
from harness.llm import LLMClient  # noqa: E402
from harness.loop import load_dotenv  # noqa: E402


def balance(c: LLMClient) -> float:
    r = requests.get(f"{c.base_url}/user/balance", headers={"Authorization": f"Bearer {c.api_key}"}, timeout=30)
    r.raise_for_status()
    return float(r.json()["balance_infos"][0]["total_balance"])


def main():
    load_dotenv(WS / ".env")
    c = LLMClient(thinking=False)
    print("model:", c.model)
    b0 = balance(c)
    print(f"balance before: {b0:.2f} CNY")

    # 组 1：大输入、唯一前缀（全 miss）、极小输出
    big = "x " * 6000  # ~6k tokens
    tot_in = 0
    for i in range(4):
        _, u = c.chat([{"role": "user", "content": f"{i}-{big}\nReply: ok"}], temperature=0, max_tokens=2)
        tot_in += u.prompt_tokens - u.cache_hit_tokens
    time.sleep(3)
    b1 = balance(c)
    miss_price = (b0 - b1) / tot_in * 1e6
    print(f"group1 miss tokens={tot_in} cost={b0 - b1:.2f} -> input(miss) ≈ {miss_price:.3f} CNY/M")

    # 组 2：同一前缀重复（全 hit）
    tot_hit = 0
    for i in range(6):
        _, u = c.chat([{"role": "user", "content": f"{big}\nReply: ok {i}"}], temperature=0, max_tokens=2)
        tot_hit += u.cache_hit_tokens
    time.sleep(3)
    b2 = balance(c)
    hit_price = (b1 - b2) / max(tot_hit, 1) * 1e6
    print(f"group2 hit tokens={tot_hit} cost={b1 - b2:.2f} -> input(hit) ≈ {hit_price:.3f} CNY/M (noisy; miss part ignored)")

    # 组 3：大输出
    tot_out = 0
    for i in range(3):
        _, u = c.chat([{"role": "user", "content": f"Write 1500 words about seed {i} of 3D-IC placement."}], temperature=0.7, max_tokens=2500)
        tot_out += u.completion_tokens
    time.sleep(3)
    b3 = balance(c)
    out_price = (b2 - b3) / tot_out * 1e6
    print(f"group3 out tokens={tot_out} cost={b2 - b3:.2f} -> output ≈ {out_price:.3f} CNY/M")
    print(f"\ntotal spent: {b0 - b3:.2f} CNY. Update config/pricing.yaml accordingly (balance granularity 0.01 CNY → treat as ±30%).")


if __name__ == "__main__":
    main()
