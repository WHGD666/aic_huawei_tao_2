"""Replay the remaining task-2 training cells for physical-file retention."""
import subprocess
import sys
import time
from pathlib import Path

script = Path(__file__).with_name("standalone_physical_task2.py")
cells = [
    ("ariane133-ng45", "1001"),
    ("bp-ng45", "1001"),
    ("bp_fe-ng45", "1001"),
    ("ariane133-ng45", "1002"),
    ("bp-ng45", "1002"),
    ("bp_fe-ng45", "1003"),
    ("ariane133-ng45", "1003"),
    ("bp-ng45", "1003"),
]
for case, seed in cells:
    start = time.time()
    print("CELL_START", case, seed, flush=True)
    result = subprocess.run([sys.executable, str(script), case, seed])
    print("CELL_END", case, seed, "rc", result.returncode,
          "seconds", round(time.time() - start), flush=True)
