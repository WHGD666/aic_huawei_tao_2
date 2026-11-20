import csv, hashlib, json, shutil
from pathlib import Path
W=Path("/root/aic-task2/contest-work/task2/binary_loss")
D=Path("/root/aic-task2/contest-data/task2/binary_loss")
R=Path("/mnt/f/huawei_aic_tao/task2")
O=R/"AIC-2026-18059446_赛题2_初赛"
BEST="64a1b258e2571a714f016fbef0036125b25ab320fa341c7de8c95729026413b1"
REQ="d4bc8894a91747f992de7d056e5d329d"
T=R/"official_kit/agent_templates/binary_loss"

def cp(a,b):
    assert a.is_file(), a
    b.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a,b)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(1048576),b""):h.update(x)
    return h.hexdigest()

session=json.loads((W/"session-result.json").read_text())
assert session["status"]=="succeeded" and session["candidate_id"]==BEST
best_record=json.loads((W/"candidates"/BEST/"candidate.json").read_text())
source=best_record["candidate_files"]["binary_loss_program.json"]["source"]
assert source==(W/"participant-agent/binary_loss/program.py").read_text()
agent=O/"AgenticBench/agent/binary_loss"
cp(T/"agent.py",agent/"agent.py")
cp(T/"common/agent_common.py",agent/"common/agent_common.py")
(agent/"program.py").write_text(source,encoding="utf-8")
assert "MAX_NEW_SUBMISSIONS_PER_SESSION = 10" in (agent/"agent.py").read_text()
be=W/"evaluations"/REQ
matrix=json.loads((be/"evaluation_result.json").read_text())["metrics"]
cells=matrix["matrix"]["cells"]
assert len(cells)==9 and all(c["valid"] for c in cells)
results=O/"results"
rows=[]
for c in cells:
    k,s=c["case"],str(c["seed"])
    a=be/"cells"/k/s/"result"
    b=results/"train"/k/s
    for name in ["evaluation_result.json","storage-retention.json","stage.log"]:
        cp(a/name,b/name)
    shutil.copytree(a/"retained_artifacts",b/"retained_artifacts",dirs_exist_ok=True)
    ref=D/"train/references"/k/s/"evaluation_result.json"
    cp(ref,results/"initial_reference"/k/s/"evaluation_result.json")
    m=json.loads(ref.read_text())["metrics"];n=c["metrics"]
    rows.append(dict(case=k,seed=s,initial_wns_ns=m["final_global_wns_setup_ns"],
      final_wns_ns=n["final_global_wns_setup_ns"],initial_tns_ns=m["final_global_tns_setup_ns"],
      final_tns_ns=n["final_global_tns_setup_ns"],final_score=c["score"],valid=c["valid"]))
rows.sort(key=lambda r:(r["case"],r["seed"]))
with (results/"train_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
cp(be/"evaluation_result.json",results/"train_evaluation_result.json")
cp(be/"matrix.json",results/"train_matrix.json")
logs=O/"logs"
cp(W/"session-result.json",logs/"session-result.json")
cp(W/"participant-agent/binary_loss/MEMORY.md",logs/"runtime-memory.md")
lineage=[]
for idx,e in enumerate(session["budget"]["evaluations"],1):
    cid,rid=e["candidate_id"],e["request_id"]
    cp(W/"candidates"/cid/"candidate.json",logs/"candidates"/f"{idx:02d}_{cid}.json")
    for name in ["matrix.json","evaluation_result.json"]:
        p=W/"evaluations"/rid/name
        if p.is_file():cp(p,logs/"evaluations"/rid/name)
    p=W/"agent-evaluations/binary_loss"/rid/"feedback.json"
    if p.is_file():cp(p,logs/"evaluations"/rid/"feedback.json")
    fb=json.loads(p.read_text()).get("feedback",{}) if p.is_file() else {}
    lineage.append(dict(order=idx,candidate_id=cid,request_id=rid,
      parent_selected_for_proposal="official_initial" if idx==1 else BEST,
      status=e["status"],score=fb.get("score") if e["status"]=="succeeded" else None,
      reason=e.get("reason"),accepted_at_utc=e.get("accepted_at"),completed_at_utc=e.get("completed_at")))
(logs/"lineage.json").write_text(json.dumps(lineage,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(logs/"selected_candidate.json").write_text(json.dumps(dict(
  baseline="binary_loss",candidate_id=BEST,request_id=REQ,train_score=matrix["combined_score"],
  train_cells=len(cells),program_sha256=sha(agent/"program.py"),agent_sha256=sha(agent/"agent.py")),
  ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
src_hash=sha(R/"2/赛题2/contest-source.tar.gz")
(O/"requirements.txt").write_text("PyYAML==6.0.3\n",encoding="utf-8")
readme=f"""# 赛题二作品材料

参赛编号：AIC-2026-18059446。所选基线：概率预测 GNN（binary_loss）。
任务包：contest-source.tar.gz，SHA-256 {src_hash}。
官方运行镜像：{session["image"]}。基座模型：DeepSeek-V4.1-Flash，由官方统一代理调用。
本地选定候选：{BEST}，对应训练评估 {REQ}。9/9 组通过，训练集得分 {matrix["combined_score"]:.6f}。该值不是七算例正式成绩。本地未运行公开验证。

## 材料
AgenticBench/agent/binary_loss/：最终 Agent 源码；agent.py 实现 Agent.run(context)，program.py 为选定的损失函数。
requirements.txt：额外主机依赖；其余环境由官方任务包及镜像提供。
results/train/：最佳候选的原生训练结果，按 train/算例/种子组织，含切分归属、评测指标和时序报告。
results/initial_reference/：官方任务包初始程序参考结果。results/train_metrics.csv：逐种子对比。
logs/：候选代码快照、演化父子关系、评估结果、失败原因和模型费用记录。
技术报告.pdf：技术方案与实验结果。

## 复现
1. 按官方 README 解压 contest-source.tar.gz 与 contest-data.tar.gz，加载 open3dbench-contest-runtime.tar.gz。
2. 将本包 AgenticBench/agent/binary_loss/ 的全部源码复制至官方 contest-source/AgenticBench/agent/binary_loss/。
3. 在主机安装 requirements.txt 所列依赖，按官方说明配置 Docker 和 GPU。模型调用由官方受控代理提供，本包不含个人 API 密钥。
4. 在 contest-source/AgenticBench 下执行 `python3 -m contest2 check --baseline binary_loss`，再以全新工作目录执行 `python3 -m contest2 run --baseline binary_loss`。Agent 从官方初始算法起步，在 10 次评估和 15 元模型预算内自主迭代并返回已完成训练评估的最优候选 ID。
5. 最终成绩以组委会统一复跑并评测七个算例的结果为准。

## 存储说明
本地最佳训练评估采用官方默认清理模式。原生切分归属、指标、时序报告、训练曲线及阶段日志已保存；上下层 DEF/ODB 布局数据库由清理逻辑删除，未用其他候选或参考运行的文件代替。
"""
(O/"README.md").write_text(readme,encoding="utf-8")
print("root",O,"score",matrix["combined_score"],"cells",len(rows))
print("program_sha256",sha(agent/"program.py"))
print("source_sha256",src_hash)
