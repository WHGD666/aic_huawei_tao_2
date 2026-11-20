"""Refresh the physical-file inventory without changing the submission code."""
import json
from pathlib import Path

root = Path("/mnt/f/huawei_aic_tao/task2/AIC-2026-18059446_赛题2_初赛")
readme = root / "README.md"
text = readme.read_text(encoding="utf-8")
head = text.split("## 存储说明", 1)[0].split("## 物理文件说明", 1)[0].rstrip()
head = head.replace("4切分与布局等物理设计结果/results/", "results/")
head = head.replace("5运行日志/logs/", "logs/")
head = head.replace("results/", "4切分与布局等物理设计结果/results/")
head = head.replace("logs/", "5运行日志/logs/")
head = head.replace("技术报告.pdf", "技术方案.pdf")
index = json.loads((root / "results/physical_replay_index.json").read_text(encoding="utf-8"))
complete = index["completed_cells"]
lines = [
    "## 物理文件说明",
    "",
    "最佳候选的原始完整训练评估使用官方默认存储模式：9/9 组均有切分归属、指标、时序报告、训练曲线和阶段日志；默认清理逻辑未留下上下层 DEF/ODB。",
    "为补齐布局文件，对同一个最终程序按算例和种子进行了独立原生物理复跑。补采文件位于 `4切分与布局等物理设计结果/results/train/<算例>/<种子>/physical_files/`，包含切分方案、OPT 后上下层 DEF/ODB、时序报告和逐文件哈希清单。",
    "补采使用组委会提供的运行镜像归档，加载后的镜像 ID 为 `sha256:5c10a1ed68f98d56d8296148ee6135f754d814c450db0ca45ef77be72881aeca`。原训练评估记录的镜像配置 ID 为 `sha256:f2401e978e5504c11cbc048266cc088152f3d5cf0a1a5438d91933175c9b7821`；两者配置 ID 不同，但 56 个 rootfs 层摘要逐一相同。补采目录与原训练评分目录分开，不把补采当作新的官方评分。每个补采目录的 `replay_manifest.json` 核验最终程序哈希，分别列出补采与原评分运行的 WNS/TNS；重复运行可能产生不同的切分与时序。",
    f"已完整补采 {len(complete)}/9 组：" + ("、".join(f"`{cell}`" for cell in complete) if complete else "暂无") + "。",
    "其他组若未列入补采清单，其上下层 DEF/ODB 仍缺失；对应的原训练切分结果、时序报告与有效评估结果保留在各组的原始结果目录。",
    "五个编号目录供人工查看；包根目录同时提供 README、requirements.txt 及相同源码的 `AgenticBench/agent/binary_loss/`，便于官方 harness 直接按规定路径加载。两份代码与算法文件在打包时逐字节核对。",
    "",
]
readme.write_text(head + "\n\n" + "\n".join(lines), encoding="utf-8")
print("README physical files", len(complete), "of 9")
