"""
loss.py -- 概率预测基线的损失函数（**唯一可演化的文件**）。

接口（签名与参数顺序固定，不得修改）：

    compute_loss(p, graph, feats) -> scalar tensor

参数
    p      : [N] tensor，GNN 输出经 sigmoid 后的"节点在上层的概率"，requires_grad=True
    graph  : dict，固定键：
               'edge_index' : [2, E] long tensor，无向（两个方向都在）
               'net_ptr'    : [M+1] long tensor，CSR 指针
               'net_nodes'  : [sum fanout] long tensor，每条网的引脚节点
               'net_of_pin' : [sum fanout] long tensor，与 net_nodes 等长，每个引脚所属网 id
               'area'       : [N] float tensor
               'is_macro'   : [N] bool tensor
               'n_nodes', 'n_nets' : int
    feats  : [N, F] float tensor，**原始**（未归一化）2D 时序特征，列顺序见 graph['feature_names']
             （slack, slew, arrival, required, cell_delay, timing_coverage, degree, fanout_max）

返回
    标量 tensor，可反向传播。训练器会按固定预算最小化它。

初始实现 = 割网期望 + 面积平衡 + 二值化。
"""
import torch


def compute_loss(p, graph, feats):
    net_ptr = graph["net_ptr"]
    net_nodes = graph["net_nodes"]
    net_of_pin = graph["net_of_pin"]
    area = graph["area"]
    n_nets = graph["n_nets"]

    # ---- 割网期望：P(net is cut) = 1 - prod(p) - prod(1-p) ----
    pp = p[net_nodes]
    log_p = torch.log(torch.clamp(pp, 1e-6, 1.0))
    log_q = torch.log(torch.clamp(1.0 - pp, 1e-6, 1.0))
    sum_log_p = torch.zeros(n_nets).index_add(0, net_of_pin, log_p)
    sum_log_q = torch.zeros(n_nets).index_add(0, net_of_pin, log_q)
    cut_prob = 1.0 - torch.exp(sum_log_p) - torch.exp(sum_log_q)
    l_cut = cut_prob.sum() / float(n_nets)

    # ---- 面积平衡：上层面积占比接近 0.5 ----
    ratio = (area * p).sum() / area.sum()
    l_bal = (ratio - 0.5) ** 2

    # ---- 二值化：p 远离 0.5 ----
    l_bin = (p * (1.0 - p)).mean()

    return l_cut + 10.0 * l_bal + 1.0 * l_bin
