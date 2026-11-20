"""Copy verified standalone task-2 physical outputs into the submission tree.

The replay uses the supplied image archive. These files are labelled separately
from the official scored evaluation and are never used to invent a score.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

root = Path("/root/aic-task2/contest-work/task2-physical-standalone")
submission = Path("/mnt/f/huawei_aic_tao/task2/AIC-2026-18059446_赛题2_初赛")
best = submission / "AgenticBench/agent/binary_loss/program.py"
rows = {}
with (submission / "results/train_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        rows[(row["case"], row["seed"])] = row


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


selection = {
    "20_partition": [
        "stage_manifest.json", "metrics.json", "partition_metrics.json",
        "partition_assignment.csv", "raw_binary_assignment.csv",
        "feasible_binary_assignment.csv", "binary_probabilities.csv",
        "evolution_program.json", "binary_report.json", "tp_gnn_report.json",
        "run.log", "SUMMARY.md",
    ],
    "50_hbt": [
        "stage_manifest.json", "metrics.json", "hbt.csv", "SUMMARY.md",
    ],
    "60_opt": [
        "stage_manifest.json", "metrics.json", "SUMMARY.md", "opt.log",
        "top/design.def", "top/design.odb", "top/design.v",
        "top/design.sdc", "top/setup.rpt", "top/metrics.tsv",
        "bottom/design.def", "bottom/design.odb", "bottom/design.v",
        "bottom/design.sdc", "bottom/setup.rpt", "bottom/metrics.tsv",
        "assembly/design.v", "assembly/design.sdc",
        "assembly/hbt.lib", "assembly/hbt_timing.lef",
        "assembly/passive/design.spef", "assembly/passive/setup.rpt",
        "assembly/passive/hold.rpt", "assembly/passive/metrics.tsv",
        "assembly/passive/parasitic_annotation.rpt",
    ],
}

completed = []
for (case, seed), expected in sorted(rows.items()):
    source_dir = root / f"{case}_{seed}"
    run = source_dir / "run"
    state_file = run / "run.json"
    if not state_file.is_file():
        print("PENDING", case, seed, flush=True)
        continue
    state = json.loads(state_file.read_text())
    if state["stages"].get("opt", {}).get("status") != "success":
        print("PENDING", case, seed, "opt", state["stages"].get("opt"), flush=True)
        continue
    program = json.loads((run / "20_partition/evolution_program.json").read_text())["source"]
    assert program == best.read_text(), (case, seed, "candidate mismatch")
    metrics = json.loads((run / "60_opt/metrics.json").read_text())["final_global"]
    wns, tns = metrics["wns_setup"], metrics["tns_setup"]
    assert abs(wns - float(expected["final_wns_ns"])) < 0.01, (case, seed, "WNS mismatch", wns)
    assert abs(tns - float(expected["final_tns_ns"])) < 0.1, (case, seed, "TNS mismatch", tns)
    target = submission / "results/train" / case / seed / "physical_files"
    copied = []
    copy(state_file, target / "run.json")
    copied.append(target / "run.json")
    for stage, names in selection.items():
        for name in names:
            dest = target / stage / name
            copy(run / stage / name, dest)
            copied.append(dest)
    copy(source_dir / "container.log", target / "container.log")
    copied.append(target / "container.log")
    record = {
        "case": case,
        "seed": seed,
        "status": "native_standalone_replay_success",
        "scored_evaluation": False,
        "candidate_program_sha256": sha(best),
        "runtime_image_from_supplied_archive": "sha256:5c10a1ed68f98d56d8296148ee6135f754d814c450db0ca45ef77be72881aeca",
        "final_wns_setup_ns": wns,
        "final_tns_setup_ns": tns,
        "matched_scored_evaluation_metrics": True,
        "files": [{"path": str(p.relative_to(target)).replace("\\", "/"), "bytes": p.stat().st_size,
                   "sha256": sha(p)} for p in copied],
    }
    (target / "replay_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed.append(f"{case}/{seed}")
    print("ARCHIVED", case, seed, "files", len(copied), "wns", wns, "tns", tns, flush=True)

(submission / "results/physical_replay_index.json").write_text(
    json.dumps({"completed_cells": completed, "total_training_cells": len(rows),
                "note": "Standalone physical replay of the selected program; scores remain from the earlier official training evaluation."},
               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("TOTAL", len(completed), "of", len(rows), flush=True)
