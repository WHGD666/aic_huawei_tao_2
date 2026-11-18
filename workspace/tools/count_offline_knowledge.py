"""
count_offline_knowledge.py -- 按《离线专家知识注入的说明》的规则统计计数单位。

规则（原文）：
  * 每个汉字计 1 个单位
  * 每个英文单词计 2 个单位
  * 连续数字计 1 个单位（一段连续数字整体算 1）
  * 重复出现分别计数（不去重）
  * 标题及正文合并计算
  * 标点、空格、换行不计入

用法:
    python tools/count_offline_knowledge.py docs/offline_knowledge_binary_loss_zh.md
    python tools/count_offline_knowledge.py --all        # 统计 docs/offline_knowledge_*.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent

# 汉字：CJK 统一表意文字（含扩展 A）+ 兼容表意文字
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# 英文单词：连续的 ASCII 字母（可含内部撇号/连字符，如 e-fold、doesn't）
WORD = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
# 连续数字：整数或小数，一段连续数字（含中间的小数点）整体计 1
NUM = re.compile(r"\d+(?:\.\d+)?")
# 标点/符号（不计）
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def count(text: str) -> dict:
    cjk = len(CJK.findall(text))
    words = len(WORD.findall(PUNCT.sub(" ", text)))
    nums = len(NUM.findall(text))
    return {"cjk": cjk, "words": words, "numbers": nums,
            "units": cjk + 2 * words + nums}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="统计 docs/offline_knowledge_*.md")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args(argv)
    files = [Path(f) for f in a.files]
    if a.all or not files:
        files = sorted((WS / "docs").glob("offline_knowledge_*.md"))
    if not files:
        print("no files", file=sys.stderr)
        return 2
    bad = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        c = count(text)
        status = "OK" if c["units"] <= a.limit else f"OVER by {c['units'] - a.limit}"
        print(f"{f.name:48s} cjk={c['cjk']:4d} words={c['words']:3d} nums={c['numbers']:3d} "
              f"units={c['units']:4d}  {status}")
        if c["units"] > a.limit:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
