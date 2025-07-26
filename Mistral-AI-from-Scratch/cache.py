import torch
from typing import List, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (AttentionBias, BlockDiagonalCausalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask, BlockDiagonalMask)


@dataclass
class RotatingCacheInputMetadata:
    """存储旋转缓存输入的元数据。"""
    positions: torch.Tensor  # 令牌的绝对位置
    to_cache_mask: torch.Tensor  # 一个布尔掩码，指示哪些令牌需要被缓存
    cached_elements: torch.Tensor  # 每个序列中已缓存的元素数量
    cache_positions: torch.Tensor  # 新令牌在缓存中的存储位置

    prefill: bool  # 是否为预填充阶段
    mask: AttentionBias  # xformers 使用的注意力掩码
    seqlens: List[int]  # 输入序列的长度列表


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    """交错合并两个列表。"""
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    """将旋转后的缓存恢复到顺序状态。"""
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    """为单个 Transformer 层提供 KV 缓存的视图和操作。"""
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """使用新的键（xk）和值（xv）更新缓存。"""
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        # 根据元数据中的位置，将新的 KV 对复制到缓存中
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """在预填充阶段，将缓存的 KV 与新的 KV 交错合并。"""
        if all([s == 0 for s in self.metadata.seqlens]):
            return xk, xv

        # ... (此处的实现较为复杂，主要用于预填充阶段的注意力计算)
        # ...
        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    # ... (其他属性，如 key, value, mask 等)


class RotatingBufferCache:
    """
    旋转缓冲区缓存，用于实现滑动窗口注意力（SWA）。
    它为模型的所有层管理一个固定大小的 KV 缓存。
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        # 初始化键（K）和值（V）的缓存张量
        self.cache_k = torch.empty((n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim))
        self.cache_v = torch.empty((n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim))
        self.kv_seqlens = None  # 跟踪每个序列的缓存长度

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        """为特定层获取一个 CacheView。"""
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        """重置缓存状态。"""
        self.kv_seqlens = None

    def to(self, device: torch.device, dtype: torch.dtype):
        """将缓存移动到指定设备和数据类型。"""
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)
        return self

    def update_seqlens(self, seqlens: List[int]):
        """更新每个序列的缓存长度。"""
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """为当前输入计算元数据，包括位置、掩码等。"""
        if self.kv_seqlens is None:
            self.kv_seqlens = torch.zeros((len(seqlens),), device=self.device, dtype=torch.long)
        
        # 计算绝对位置
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(self.kv_seqlens.tolist(), seqlens)])
        
        # 计算在循环缓冲区中的缓存位置
        cache_positions = positions % self.sliding_window
        
        # 创建注意力掩码 (使用 xformers)
        # ... (根据是预填充还是增量生成，创建不同类型的掩码)
        # ...

        return RotatingCacheInputMetadata(
            positions=positions.to(device=self.device, dtype=torch.long),
            # ... 其他元数据
        )