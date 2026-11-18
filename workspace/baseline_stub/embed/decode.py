"""
decode.py -- 表示学习基线的聚类解码函数（**可演化文件之一**）。

接口（签名与参数顺序固定，不得修改）：

    decode_partition(z, graph, feats, seed) -> numpy int64 array [N]，元素 ∈ {0,1}，1 = 上层

参数
    z      : numpy float32 [N, D]，训练完成后的节点表示（已 L2 归一化）
    graph  : dict，键同损失函数，但张量已转成 numpy（edge_index / net_ptr / net_nodes / net_of_pin / area / is_macro 等）
    feats  : numpy float32 [N, F]，原始 2D 时序特征
    seed   : int，所有随机数必须由它派生（结果须可复现）

约束
    返回的切分会再经官方修复程序满足平衡约束；但修复翻动越多、学到的结构越容易被破坏，
    解码函数应尽量自行给出接近平衡（上层面积 ≈ 50%，宏数量/面积 ≈ 50%）的结果。
    只能用 numpy；不得读写文件、不得使用全局随机状态。

初始实现 = 面积加权两路 k-means：
    1) k-means(k=2) 得到两个簇；2) 以面积加权的方式把簇分配到两层；
    3) 若面积偏差过大，按"到另一簇中心距离更近"的顺序翻节点，直到上层面积占比落入 [0.49, 0.51]。
"""
import numpy as np


def decode_partition(z, graph, feats, seed):
    rng = np.random.default_rng(int(seed))
    n = z.shape[0]
    area = graph["area"].astype(np.float64)

    # ---- 两路 k-means（固定 20 次迭代、3 次重启取 SSE 最小） ----
    best_lab, best_sse = None, np.inf
    for _ in range(3):
        c = z[rng.choice(n, 2, replace=False)].copy()
        for _it in range(20):
            d = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            for k in range(2):
                if (lab == k).any():
                    c[k] = z[lab == k].mean(axis=0)
        sse = float(d[np.arange(n), lab].sum())
        if sse < best_sse:
            best_sse, best_lab, best_c = sse, lab.copy(), c.copy()
    lab, c = best_lab, best_c

    # ---- 簇→层：面积大的簇做下层(0) ----
    a0 = area[lab == 0].sum()
    a1 = area[lab == 1].sum()
    part = (lab == (0 if a0 < a1 else 1)).astype(np.int64)   # 面积小的簇在上层

    # ---- 面积修正：按到对面簇中心的距离翻最"边缘"的节点 ----
    d = ((z[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    tot = area.sum()
    for _ in range(n):
        r = area[part == 1].sum() / tot
        if 0.49 <= r <= 0.51:
            break
        if r > 0.51:
            cand = np.where(part == 1)[0]
            own, other = d[cand, lab[cand]], d[cand, 1 - lab[cand]]
            i = cand[np.argmin(other - own)]
            part[i] = 0
        else:
            cand = np.where(part == 0)[0]
            own, other = d[cand, lab[cand]], d[cand, 1 - lab[cand]]
            i = cand[np.argmin(other - own)]
            part[i] = 1
    return part
