import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn

@dataclasses.dataclass
class MoeArgs(Serializable):
    """混合专家模型（MoE）的参数配置。"""
    num_experts: int  # 专家网络的总数
    num_experts_per_tok: int  # 每个令牌（token）选择的专家数量

class MoeLayer(nn.Module):
    """
    混合专家（MoE）层。
    该层取代了标准 Transformer 中的前馈网络（FFN），允许模型在保持计算成本不变的情况下，
    极大地增加参数量，从而提升性能。
    """
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        """
        初始化 MoE 层。
        Args:
            experts (List[nn.Module]): 专家网络列表，每个专家都是一个独立的 nn.Module。
            gate (nn.Module): 门控网络，通常是一个线性层，用于决定每个令牌由哪些专家处理。
            moe_args (MoeArgs): MoE 的配置参数。
        """
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)  # 将专家列表注册为模块
        self.gate = gate  # 门控网络
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        """
        MoE 层的前向传播。
        Args:
            inputs (torch.Tensor): 输入张量，形状为 (num_tokens, dim)。
        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 1. 通过门控网络获取每个令牌对于每个专家的分数（logits）
        gate_logits = self.gate(inputs)
        
        # 2. 为每个令牌选择分数最高的 top-k 个专家
        # weights 是选定专家的分数，selected_experts 是选定专家的索引
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        
        # 3. 对选出的专家的分数应用 softmax，得到它们的组合权重
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        
        # 初始化最终的输出张量
        results = torch.zeros_like(inputs)
        
        # 4. 遍历所有专家，让每个专家只处理分配给它的令牌
        for current_expert_index, current_expert in enumerate(self.experts):
            # 找到哪些令牌被分配给了当前专家
            # token_index 是令牌在批次中的索引
            # token_expert_index 是令牌在其被选中的专家列表中的索引
            token_index, token_expert_index = torch.where(selected_experts == current_expert_index)
            
            if token_index.numel() > 0:
                # 获取这些令牌的输入
                selected_inputs = inputs[token_index]
                
                # 获取这些令牌对应的权重
                selected_weights = weights[token_index, token_expert_index, None]
                
                # 让当前专家处理这些令牌，并将输出用权重加权
                expert_output = current_expert(selected_inputs)
                
                # 将加权后的专家输出累加到最终结果中
                results.index_add_(0, token_index, selected_weights * expert_output)
                
        return results