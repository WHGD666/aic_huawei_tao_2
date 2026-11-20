"""Populate the five organizer-facing folders without changing runtime paths.

NTFS hard links make the large physical-result tree visible in the numbered
folder without copying its bytes. The ZIP writer reads only that numbered copy.
"""
import os
import shutil
from pathlib import Path

root = Path(r"F:\huawei_aic_tao\task2\AIC-2026-18059446_赛题2_初赛")


def copy_small(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def link_tree(source, target):
    count = 0
    for item in source.rglob("*"):
        if not item.is_file() or item.suffix == ".pyc" or "__pycache__" in item.parts:
            continue
        dest = target / item.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if os.path.samefile(item, dest):
                continue
            dest.unlink()
        os.link(item, dest)
        count += 1
    return count


copy_small(root / "README.md", root / "1说明与清单/README.md")
copy_small(root / "requirements.txt", root / "1说明与清单/requirements.txt")
for item in (root / "AgenticBench").rglob("*.py"):
    copy_small(item, root / "2Harness源码/AgenticBench" / item.relative_to(root / "AgenticBench"))
copy_small(root / "AgenticBench/agent/binary_loss/program.py", root / "3算法代码/program.py")
copy_small(root / "技术报告.pdf", root / "技术方案.pdf")
new_results = link_tree(root / "results", root / "4切分与布局等物理设计结果/results")
new_logs = link_tree(root / "logs", root / "5运行日志/logs")
print("LAYOUT_SYNCED", "new_results", new_results, "new_logs", new_logs)
