"""Build and audit the final task-2 ZIP from an explicit allowlist."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

base = Path("/mnt/f/huawei_aic_tao/task2")
root = base / "AIC-2026-18059446_赛题2_初赛"
zip_path = base / "AIC-2026-18059446_赛题2.zip"
required_files = [
    "README.md", "requirements.txt", "技术报告.pdf",
    "AgenticBench/agent/binary_loss/agent.py",
    "AgenticBench/agent/binary_loss/common/agent_common.py",
    "AgenticBench/agent/binary_loss/program.py",
    "results/train_metrics.csv", "results/train_evaluation_result.json",
    "results/physical_replay_index.json", "logs/lineage.json",
]
for name in required_files:
    assert (root / name).is_file(), name

files = []
for name in ("README.md", "requirements.txt", "技术报告.pdf"):
    files.append(root / name)
for folder in ("AgenticBench", "results", "logs"):
    for item in (root / folder).rglob("*"):
        if not item.is_file():
            continue
        if item.is_symlink():
            raise RuntimeError(f"symlink not allowed: {item}")
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        files.append(item)
files.sort(key=lambda p: p.relative_to(root).as_posix())

secret = re.compile(rb"sk-[A-Za-z0-9_-]{16,}|(?i:api[_-]?key)\s*[:=]\s*['\"][^'\"]{12,}")
for item in files:
    rel = item.relative_to(root).as_posix()
    assert not any(x in rel.lower() for x in ("task1", "赛题1", "赛题一", ".env")), rel
    if item.suffix.lower() in {".py", ".json", ".md", ".txt", ".csv", ".log", ".yaml", ".rpt"}:
        data = item.read_bytes()
        if b"task1" in data.lower() or "赛题一".encode() in data or "赛题1".encode() in data:
            raise RuntimeError(f"task-1 trace: {rel}")
        if secret.search(data):
            raise RuntimeError(f"possible secret: {rel}")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "submission": "AIC-2026-18059446_赛题2",
    "file_count": len(files),
    "files": [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
               "sha256": digest(p)} for p in files],
}
manifest_path = root / "submission_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
files.append(manifest_path)

temp = zip_path.with_suffix(".zip.tmp")
with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5,
                     allowZip64=True) as archive:
    for item in files:
        archive.write(item, item.relative_to(root).as_posix())
with zipfile.ZipFile(temp) as archive:
    bad = archive.testzip()
    assert bad is None, bad
    names = set(archive.namelist())
    assert all(name in names for name in required_files)
    assert len(names) == len(files)
temp.replace(zip_path)
print("ZIP", zip_path)
print("FILES", len(files), "BYTES", zip_path.stat().st_size)
print("SHA256", digest(zip_path))
print("PHYSICAL_CELLS", len(json.loads((root / "results/physical_replay_index.json").read_text())["completed_cells"]))
