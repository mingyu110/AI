import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from simple_parsing.helpers import Serializable

from rope import precompute_freqs_cis, apply_rotary_emb
from cache import CacheView, RotatingBufferCache
from moe import MoeArgs, MoeLayer

from xformers.ops.fmha import memory_efficient_attention


@dataclass
class ModelArgs(Serializable):
    """模型的参数配置。"""
    dim: int  # 模型维度
    n_layers: int  # Transformer 层数
    head_dim: int  # 每个注意力头的维度
    hidden_dim: int  # FFN 层的隐藏层维度
    n_heads: int  # 注意力头的数量
    n_kv_heads: int  # KV 注意力头的数量（用于 GQA）
    norm_eps: float  # RMSNorm 中的 epsilon 值
    vocab_size: int  # 词汇表大小

    max_batch_size: int = 0  # 最大批处理大小

    # RoPE 的 theta 参数，如果不指定，则根据是否使用滑动窗口自动设置
    rope_theta: Optional[float] = None
    # 滑动窗口大小，如果不指定，则不使用滑动窗口注意力
    sliding_window: Optional[int] = None
    # MoE 配置，如果不指定，则使用标准的前馈网络
    moe: Optional[MoeArgs] = None


@dataclass
class SimpleInputMetadata:
    """不使用 KV 缓存时输入的元数据。"""
    positions: torch.Tensor  # 令牌的位置索引

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        """根据序列长度列表创建元数据。"""
        positions = torch.cat([torch.arange(0, seqlen) for seqlen in seqlens])
        return SimpleInputMetadata(positions.to(device=device, dtype=torch.long))


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    """
    为分组查询注意力（GQA）重复键（Key）和值（Value）。
    在 GQA 中，查询头的数量多于键/值头，因此需要将键/值头重复以匹配查询头的数量。
    """
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    """多头注意力模块。"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        # GQA 中，查询头的数量是 KV 头的整数倍
        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        # 查询、键、值和输出的线性投影层
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]) -> torch.Tensor:
        seqlen_sum, _ = x.shape  # (总令牌数, dim)

        # 1. 线性投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 重塑为多头形式
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        
        # 2. 应用旋转位置编码 (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 3. KV 缓存处理
        if cache is None:  # 不使用缓存（例如，训练时）
            key, val = xk, xv
        elif cache.prefill:  # 预填充缓存（处理初始提示）
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:  # 增量生成（一次一个令牌）
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)

        # 4. 为 GQA 重复 KV
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # 5. 使用 xformers 进行高效的内存注意力计算
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)
        
        # 6. 输出投影
        return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))


class FeedForward(nn.Module):
    """标准的前馈网络（FFN）。"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        # 使用 SwiGLU 激活函数: silu(w1(x)) * w3(x)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    """均方根层归一化（RMSNorm）。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 归一化公式：x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    """单个 Transformer 块。"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        # 根据配置选择使用标准 FFN 还是 MoE
        self.feed_forward: nn.Module
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe)
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]) -> torch.Tensor:
        # 1. 注意力模块（前置归一化 + 残差连接）
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        # 2. 前馈网络模块（前置归一化 + 残差连接）
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    """完整的 Transformer 模型。"""
    def __init__(self, args: ModelArgs, pipeline_rank: int = 0, num_pipeline_ranks: int = 1):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        
        # 流水线并行相关参数
        assert pipeline_rank < num_pipeline_ranks
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks

        # 根据流水线并行中的 rank，决定是否初始化特定层
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        
        # 第一个 rank 初始化词嵌入层
        if pipeline_rank == 0:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            
        # 最后一个 rank 初始化最终的归一化层和输出层
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 计算并分配当前 rank 拥有的 Transformer 层
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        """惰性计算 RoPE 频率，并缓存。"""
        if self._precomputed_freqs_cis is None:
            theta = self.args.rope_theta
            if theta is None:
                # 根据是否使用滑动窗口设置默认 theta
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000, theta)
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(device=self.device)
        return self._precomputed_freqs_cis

    def forward(self, input_ids: torch.Tensor, seqlens: List[int], cache: Optional[RotatingBufferCache] = None) -> torch.Tensor:
        """完整的模型前向传播。"""
        # 1. 词嵌入
        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            # 从上一个 rank 接收中间结果
            h = torch.empty(input_ids.shape[0], self.args.dim, device=self.device, dtype=self.dtype)
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        # 2. 获取 RoPE 频率
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        # 3. 依次通过当前 rank 的所有 Transformer 层
        for local_layer_id, layer in enumerate(self.layers.values()):
            cache_view = cache.get_view(local_layer_id, input_metadata) if cache is not None else None
            h = layer(h, freqs_cis, cache_view)

        # 4. 流水线并行处理
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # 将中间结果发送到下一个 rank
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            # 非最后一个 rank 不需要计算最终输出
            return torch.empty(h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype)
        else:
            # 最后一个 rank 计算最终输出
            assert self.norm is not None and self.output is not None
            outs = self.output(self.norm(h))
            if self.num_pipeline_ranks > 1:
                # 将最终结果广播给所有其他 rank
                torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
            return outs.float()

    def load_state_dict(self, state_dict, *args, **kwargs):
        """根据流水线并行的 rank 加载对应的模型权重。"""
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        super().load_state_dict(state_to_load, *args, **kwargs)

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, num_pipeline_ranks: int = 1, device="cuda", dtype=torch.float16) -> "Transformer":
        """从文件夹加载模型权重和配置。"""
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
            
        # 使用 'meta' device 可以在不实际分配内存的情况下初始化模型结构
        with torch.device("meta"):
            model = Transformer(model_args, pipeline_rank=pipeline_rank, num_pipeline_ranks=num_pipeline_ranks)
            
        # 加载模型权重
        loaded = torch.load(str(folder / "consolidated.00.pth"), mmap=True)
        model.load_state_dict(loaded, assign=True)
        
        return model.to(device=device, dtype=dtype)