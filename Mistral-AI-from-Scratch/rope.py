import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    预计算旋转位置编码（RoPE）所需的频率。
    RoPE 通过将位置信息编码为旋转矩阵，并将其应用于查询和键向量，从而在自注意力机制中引入相对位置信息。

    Args:
        dim (int): 模型的维度。
        end (int): 序列的最大长度。
        theta (float): RoPE 的基数。通常设置为 10000.0。

    Returns:
        torch.Tensor: 预计算的频率，以复数形式存储，形状为 (end, dim // 2)。
    """
    # 计算频率：freqs = 1 / (theta^(2i/dim))，其中 i 是维度的索引
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 创建位置索引：t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # 计算每个位置和每个频率的外积，得到 m*theta_i
    freqs = torch.outer(t, freqs).float()
    
    # 将频率转换为复数形式 (cisoid)，即 e^(i*m*theta_i) = cos(m*theta_i) + i*sin(m*theta_i)
    # torch.polar 使用模长和角度创建复数张量
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将旋转位置编码应用于输入的查询（xq）和键（xk）张量。

    Args:
        xq (torch.Tensor): 查询张量，形状为 (..., seq_len, dim)。
        xk (torch.Tensor): 键张量，形状为 (..., seq_len, dim)。
        freqs_cis (torch.Tensor): 预计算的复数频率。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 经过旋转编码的查询和键张量。
    """
    # 将 xq 和 xk 的最后一个维度（dim）重塑为复数形式，即将 (..., dim) 变为 (..., dim/2, 2)
    # 然后将其视为复数张量
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整 freqs_cis 的形状以进行广播
    freqs_cis = freqs_cis[:, None, :]
    
    # 执行复数乘法，实现旋转操作
    # (v_r + i*v_i) * (c + i*s) = (v_r*c - v_i*s) + i*(v_r*s + v_i*c)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    
    # 将输出张量转换回原始数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)