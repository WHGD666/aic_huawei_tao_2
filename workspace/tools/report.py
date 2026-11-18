"""
report.py -- 把一次 harness 运行目录渲染成技术报告素材（Markdown）。

用法:  python tools/report.py logs/R30_prob [-o report.md]

输出：预算消耗、演化效率（第 k 次评估后的最优）、候选表、血统树、代理↔真实秩相关、
基线 vs 最终候选的三算例对比、失败分布、最终导出的函数片段。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent


def load(run: Path):
    cands = []
    for mp in sorted((run / "candidates").glob("c*/meta.json")):
        cands.append(json.loads(mp.read_text(encoding="utf-8")))
    cands.sort(key=lambda m: m["created_at"])
    rd = lambda n: json.loads((run / n).read_text(encoding="utf-8")) if (run / n).exists() else {}  # noqa: E731
    base = {}
    bp = run / "candidates" / "baseline" / "eval.json"
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))
    return cands, rd("budget.json"), rd("summary.json"), base, (run / "lineage_tree.txt").read_text(encoding="utf-8") if (run / "lineage_tree.txt").exists() else ""


def f(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


def render(run: Path) -> str:
    cands, budget, summary, base, tree = load(run)
    aliases = summary.get("case_aliases") or {}
    order = sorted(aliases, key=lambda c: aliases[c])
    u = budget.get("usage", {})
    out = [f"# Harness run report — `{run.name}`\n"]
    out.append(f"baseline = **{summary.get('baseline', '?')}** · model = `{summary.get('code_version', {}).get('model', '?')}` · "
               f"evaluator = `{summary.get('code_version', {}).get('evaluator_mode', '?')}`\n")

    out.append("## 1. 预算消耗\n")
    out.append("| 评估次数 | API 费用 (CNY) | LLM 调用 | prompt tok (缓存命中) | completion tok | 候选总数 | 已评估 | gate 拦截 | 代理未提交 | 重复切分 | 无效 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    out.append(f"| {budget.get('evaluations', 0)} | {budget.get('cost_cny', 0):.3f} | {budget.get('llm_calls', 0)} | "
               f"{u.get('prompt_tokens', 0)} ({u.get('cache_hit_tokens', 0)}) | {u.get('completion_tokens', 0)} | "
               f"{len(cands)} | {summary.get('n_evaluated', '—')} | {summary.get('n_gate_failed', '—')} | "
               f"{summary.get('n_screened', '—')} | {summary.get('n_duplicate', '—')} | {summary.get('n_invalid', '—')} |\n")
    n_eval = budget.get("evaluations", 1) or 1
    out.append(f"平均每次评估的 API 成本：{budget.get('cost_cny', 0) / n_eval:.4f} 元；"
               f"缓存命中率：{(u.get('cache_hit_tokens', 0) / max(u.get('prompt_tokens', 1), 1)):.0%}\n")

    out.append("## 2. 演化效率：每次评估后的最优稳健分 / 总分\n")
    out.append("| k | 候选 | 本轮总分 | 本轮稳健分 | 累计最优总分 | 累计最优稳健分 |")
    out.append("|---|---|---|---|---|---|")
    best_t = best_r = None
    k = 0
    for m in cands:
        if m["status"] not in ("evaluated", "invalid"):
            continue
        k += 1
        t, r = m.get("score_total"), m.get("score_robust")
        if m["status"] == "evaluated" and m.get("all_valid"):
            best_t = t if best_t is None else max(best_t, t)
            best_r = r if best_r is None else max(best_r, r)
        out.append(f"| {k} | {m['id']} | {f(t)} | {f(r)} | {f(best_t)} | {f(best_r)} |")
    out.append("")

    out.append("## 3. 候选表（已评估，按总分降序）\n")
    out.append("| 候选 | 父代 | 阶段 | 总分 | 稳健分 | 种子间σ | " + " | ".join(aliases.get(c, c) for c in order) + " | 假设 |")
    out.append("|---|---|---|---|---|---|" + "---|" * len(order) + "---|")
    ev = [m for m in cands if m["status"] == "evaluated"]
    for m in sorted(ev, key=lambda m: -(m["score_total"] or -9e9)):
        pc = " | ".join(f(m.get("per_case", {}).get(c)) for c in order)
        out.append(f"| {m['id']} | {m['parent'] or 'base'} | {m['phase']} | {f(m['score_total'])} | {f(m['score_robust'])} | "
                   f"{('%.3f' % m['seed_std']) if m.get('seed_std') is not None else '—'} | {pc} | {m['hypothesis'][:90].replace('|', '/')} |")
    out.append("")

    out.append("## 4. 代理指标 ↔ 真实得分的秩相关（运行期末）\n")
    corr = {k2: v for k2, v in (summary.get("proxy_correlation") or {}).items() if not k2.startswith("smoke_")}
    if corr:
        out.append("| 指标 | Spearman ρ | n | 方向解读 |")
        out.append("|---|---|---|---|")
        for k2, v in sorted(corr.items(), key=lambda kv: -abs(kv[1].get("rho") or 0)):
            rho = v.get("rho")
            if rho is None:
                continue
            hints = {"cut": "割网越少越好", "cut_ratio": "同 cut", "wcut": "关键加权割越少越好", "crit_cross": "关键相邻对跨层比例越低越好",
                     "repair_flips": "修复翻动越多分越高？（**方向待核实**）", "area_dev": "平衡偏差", "hamming_ref": "与基线差异",
                     "macro_cnt_dev": "宏数量偏差", "macro_area_dev": "宏面积偏差"}
            out.append(f"| {k2} | {rho:+.2f} | {v.get('n')} | {hints.get(k2, '')} |")
        out.append("\n注：正 ρ 表示该指标越大真实得分越高。代理指标只用于筛选与给模型提示方向，排名永远用真实评估。\n")
    else:
        out.append("（样本不足）\n")

    out.append("## 5. 基线 vs 最终候选（每算例 = 有效种子的均值；违例量 = |负 slack|，正值=改善）\n")
    best = summary.get("best") or {}
    if base and best:
        bruns = base.get("runs", {})
        out.append("| 算例 | 基线 WNS(ns) | 基线 TNS(ns) | 候选 WNS | 候选 TNS | 算例得分 |")
        out.append("|---|---|---|---|---|---|")
        for c in order:
            brs = [r for key, r in bruns.items() if r.get("case") == c and r.get("valid", True)]
            if not brs:
                continue
            bw = sum(r["wns"] for r in brs) / len(brs)
            bt = sum(r["tns"] for r in brs) / len(brs)
            sc = best.get("per_case", {}).get(c)
            # 候选的 WNS/TNS 从 eval.json 拿
            evp = run / "candidates" / best["id"] / "eval.json"
            cw = ct = None
            if evp.exists():
                d = json.loads(evp.read_text(encoding="utf-8"))
                crs = [r for key, r in d.get("runs", {}).items() if r.get("case") == c and r.get("valid", True)]
                if crs:
                    cw = sum(r["wns"] for r in crs) / len(crs)
                    ct = sum(r["tns"] for r in crs) / len(crs)
            out.append(f"| {aliases.get(c, c)} | {bw:.3f} | {bt:.1f} | {'%.3f' % cw if cw is not None else '—'} | "
                       f"{'%.1f' % ct if ct is not None else '—'} | {f(sc)} |")
        out.append("")
        if best.get("analysis"):
            out.append(f"最终候选的 Analyzer 结论：{best['analysis']}\n")

    out.append("## 6. 失败与弃权分布\n")
    by_st = {}
    for m in cands:
        by_st[m["status"]] = by_st.get(m["status"], 0) + 1
    out.append("| 状态 | 数量 | 说明 |")
    out.append("|---|---|---|")
    desc = {"evaluated": "有真实分数", "invalid": "评估/代理运行无效（不参与排序与父代）", "gate_failed": "本地门禁拦截，未花评估",
            "screened": "过了门槛但本轮未被选中，或与基线差异过小", "duplicate": "切分与已评估候选相同，继承分数", "discarded": "预算耗尽/中断"}
    for st, n in sorted(by_st.items(), key=lambda kv: -kv[1]):
        out.append(f"| {st} | {n} | {desc.get(st, '')} |")
    gf = [m for m in cands if m["status"] == "gate_failed"]
    if gf:
        out.append("\n门禁失败原因：")
        for m in gf:
            out.append(f"- {m['id']}: {m['gate_report'][:160].replace(chr(10), ' / ')}")
    out.append("")

    out.append("## 7. 血统树\n")
    out.append("```\n" + (tree or "(empty)") + "\n```\n")

    out.append("## 8. 最终导出的函数（algorithms/ 下同名文件为完整版）\n")
    alg = WS / "algorithms" / str(summary.get("baseline", ""))
    if alg.exists():
        for p in sorted(alg.glob("*.py")):
            out.append(f"### `{p.name}`\n")
            out.append("```python\n" + p.read_text(encoding="utf-8")[:6000] + "\n```\n")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    if not run.is_absolute():
        run = (Path.cwd() / run).resolve()
    text = render(run)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"written {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
