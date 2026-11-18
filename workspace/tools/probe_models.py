"""probe_models.py -- 列出账号可见模型，并用一次最小调用测连通与计费（约 0.001 元）。"""
from __future__ import annotations

import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from harness.budget import Pricing, Usage  # noqa: E402
from harness.llm import LLMClient  # noqa: E402
from harness.loop import load_dotenv  # noqa: E402


def main():
    load_dotenv(WS / ".env")
    c = LLMClient()
    print("base_url:", c.base_url)
    print("models  :", c.list_models())
    print(f"\nminimal chat with model={c.model} ...")
    text, usage = c.chat([{"role": "user", "content": "Reply with the single word: ready"}], temperature=0, max_tokens=5)
    print("reply   :", text.strip())
    print("usage   :", vars(usage))
    p = Pricing(WS / "config" / "pricing.yaml")
    print(f"cost    : {p.cost(c.model, usage):.6f} CNY (per config/pricing.yaml; verify against official price page)")


if __name__ == "__main__":
    main()
