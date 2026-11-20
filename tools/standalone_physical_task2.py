"""Standalone task-2 physical replay of the selected binary loss on bp_fe-ng45 seed 1002.
Uses the official supplied source and image archive. Does not produce an official score.
"""
import json, subprocess, sys, shutil
from pathlib import Path
FRAME=Path("/root/aic-task2/contest-source/AgenticBench")
sys.path.insert(0,str(FRAME))
from local_bench.workspace import isolated_clone, copy_upstream_stages, materialize_dreamplace_source_tree, prepare_dreamplace_overlay
from local_bench.model import load_task
from grt_bench.runtime import docker_command

SOURCE=FRAME.parent
CASE=sys.argv[1] if len(sys.argv)>1 else "bp_fe-ng45"
SEED=sys.argv[2] if len(sys.argv)>2 else "1002"
BEST=Path("/root/aic-task2/contest-work/task2/binary_loss/evaluations/d4bc8894a91747f992de7d056e5d329d/cells")/CASE/SEED/"result"
ROOT=Path("/root/aic-task2/contest-work/task2-physical-standalone")
OUT=ROOT/(CASE+"_"+SEED)
WORKSPACE=OUT/"workspace/Open3DBench-V2.0"
RUN=OUT/"run"
TASK=load_task(FRAME/"local_tasks/tp_gnn_binary_loss_timing")
BASE=Path("/root/aic-task2/contest-data/task2/inputs")/CASE
IMAGE="sha256:5c10a1ed68f98d56d8296148ee6135f754d814c450db0ca45ef77be72881aeca"
OUT.mkdir(parents=True,exist_ok=True)
assert BEST.is_dir() and (BEST/"candidate_files/binary_loss_program.json").is_file()
assert (BEST/"harness_inputs/frozen/tp_gnn_timing").is_dir()
print("PREPARE",OUT,flush=True)
if not WORKSPACE.is_dir(): isolated_clone(SOURCE,WORKSPACE)
materialize_dreamplace_source_tree(SOURCE,WORKSPACE,[])
if not RUN.is_dir(): copy_upstream_stages(BASE,RUN,TASK.upstream_stages)
config=BEST/"resolved_config.yaml"
command=["-m","open3dbench","--repo",str(WORKSPACE),"run",str(config),
         "--from","partition","--to","opt","--resume","--run-dir",str(RUN)]
resources={"cpu":4,"memory_gb":24,"storage_gb":16,"gpu_count":1,"build_slots":0}
mounts=[(SOURCE,"/workspace",True),(FRAME,"/trusted",True),
        (BEST,str(BEST),True),(BASE,str(BASE),True),(OUT,str(OUT),False)]
full=docker_command(IMAGE,"grt-task2-standalone-"+CASE.replace("_", "-")+"-"+SEED,resources,["0"],mounts,command)
(OUT/"command.json").write_text(json.dumps(full,ensure_ascii=False,indent=2)+"\n")
print("RUN",flush=True)
with (OUT/"container.log").open("w") as log:
    result=subprocess.run(full,stdout=log,stderr=subprocess.STDOUT)
print("EXIT",result.returncode,flush=True)
for p in sorted(RUN.rglob("*.def")):
    if "top" in p.parts or "bottom" in p.parts:
        print("DEF",p.relative_to(RUN),p.stat().st_size,flush=True)
raise SystemExit(result.returncode)
