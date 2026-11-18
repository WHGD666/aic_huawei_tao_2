"""
loss.py -- 表示学习基线的训练损失（**可演化文件之一**）。

接口（签名与参数顺序固定，不得修改）：

    compute_loss(z, graph, feats, neg_idx) -> scalar tensor

参数
    z       : [N, D] tensor，GNN 输出的节点表示（已做固定的 L2 归一化），requires_grad=True
    graph   : dict，固定键同概率预测基线：
                'edge_index' [2,E] / 'net_ptr' [M+1] / 'net_nodes' / 'net_of_pin' / 'area' [N] / 'is_macro' [N] /
                'n_nodes' / 'n_nets' / 'feature_names'
    feats   : [N, F] float tensor，原始 2D 时序特征（slack, slew, arrival, required, cell_delay,
              timing_coverage, degree, fanout_max）
    neg_idx : [E, K] long tensor，训练器为每条边采好的 K 个随机负样本节点（采样固定、不可演化）

返回
    标量 tensor。

初始实现 = 无监督对比：邻居为正样本（拉近），随机节点为负样本（推远）。
"""
import torch


def compute_loss(z, graph, feats, neg_idx):
    ei = graph["edge_index"]
    src, dst = ei[0], ei[1]
    tau = 0.5

    # 正样本：相邻节点表示的内积要大
    pos = (z[src] * z[dst]).sum(dim=1) / tau
    l_pos = -torch.nn.functional.logsigmoid(pos).mean()

    # 负样本：随机节点表示的内积要小
    zs = z[src].unsqueeze(1)                      # [E,1,D]
    zn = z[neg_idx]                               # [E,K,D]
    neg = (zs * zn).sum(dim=2) / tau              # [E,K]
    l_neg = -torch.nn.functional.logsigmoid(-neg).mean()

    return l_pos + l_neg
