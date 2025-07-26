from cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from model import Transformer
from tokenizer import Tokenizer


def top_p_sampling(probabilities: torch.Tensor, threshold_p: float):
    """
    Top-p (Nucleus) 采样。
    从累积概率超过阈值 p 的最小词汇集中进行采样。
    Args:
        probabilities (torch.Tensor): 输入的概率分布。
        threshold_p (float): Top-p 的阈值。
    Returns:
        torch.Tensor: 采样到的令牌索引。
    """
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到累积概率超过阈值的点，并将之后的概率置为0
    cutoff_mask = cumulative_probs - sorted_probs > threshold_p
    sorted_probs[cutoff_mask] = 0.0

    # 重新归一化概率分布
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    # 从新的分布中进行多项式采样
    sampled_token = torch.multinomial(sorted_probs, 1)

    return torch.gather(sorted_indices, -1, sampled_token)


def token_sampling(logits_tensor: torch.Tensor, temp: float, nucleus_p: float):
    """
    根据温度和 top-p 值对 logits 进行采样。
    Args:
        logits_tensor (torch.Tensor): 模型的输出 logits。
        temp (float): 温度系数。值越小，采样越趋向于贪心；值越大，采样越随机。
        nucleus_p (float): Top-p 采样的阈值。
    Returns:
        torch.Tensor: 采样到的令牌索引。
    """
    if temp > 0.0:
        # 应用温度缩放并计算概率
        scaled_probs = torch.softmax(logits_tensor / temp, dim=-1)
        # 使用 top-p 采样
        token = top_p_sampling(scaled_probs, nucleus_p)
    else:
        # 贪心采样，直接选择概率最高的令牌
        token = torch.argmax(logits_tensor, dim=-1, keepdim=True)

    return token.view(-1)


@torch.inference_mode()
def generate_text(
    prompts: List[str], 
    model: Transformer, 
    tokenizer: Tokenizer, 
    *, 
    max_new_tokens: int, 
    temperature: float, 
    chunk_step: int = None
):
    """
    核心文本生成函数。
    Args:
        prompts (List[str]): 输入的提示文本列表。
        model (Transformer): Transformer 模型实例。
        tokenizer (Tokenizer): 分词器实例。
        max_new_tokens (int): 要生成的最大新令牌数。
        temperature (float): 采样温度。
        chunk_step (int, optional): 处理长提示时，每次送入模型的块大小。默认为 None。
    Returns:
        Tuple[List[str], List[List[float]]]: 生成的文本列表和对应的 log-probabilities 列表。
    """
    model.eval()  # 设置为评估模式

    batch_size = len(prompts)
    
    # 1. 编码输入提示
    encoded_inputs = [tokenizer.encode(text, bos=True) for text in prompts]
    input_lengths = [len(seq) for seq in encoded_inputs]

    # 2. 初始化 KV 缓存
    total_cache_size = max(input_lengths) + max_new_tokens
    if model.args.sliding_window is not None and total_cache_size > model.args.sliding_window:
        total_cache_size = model.args.sliding_window
    buffer_cache = RotatingBufferCache(model.n_local_layers, model.args.max_batch_size, total_cache_size, model.args.n_kv_heads, model.args.head_dim)
    buffer_cache.to(device=model.device, dtype=model.dtype)
    buffer_cache.reset()
    
    token_logprobs = [[] for _ in range(batch_size)]
    previous_logits = None

    # 3. 分块处理输入提示（预填充阶段）
    max_input_len = max(input_lengths)
    if chunk_step is None:
        chunk_step = max_input_len

    for start_idx in range(0, max_input_len, chunk_step):
        current_chunks = [seq[start_idx : start_idx + chunk_step] for seq in encoded_inputs]
        input_tensor = torch.tensor(sum(current_chunks, []), device=model.device, dtype=torch.long)
        
        # 模型前向传播，填充 KV 缓存
        pre_logits = model.forward(input_tensor, seqlens=[len(seq) for seq in current_chunks], cache=buffer_cache)
        # ... (logprobs 计算，为简洁省略)
        
        # 获取最后一个令牌的 logits，用于下一步生成
        token_positions = torch.tensor([len(seq) for seq in current_chunks], device=pre_logits.device).cumsum(dim=0) - 1
        previous_logits = pre_logits.index_select(0, token_positions)

    # 4. 自回归生成新令牌
    next_tokens_collected = []
    assert previous_logits is not None

    for _ in range(max_new_tokens):
        # 采样下一个令牌
        sampled_token = token_sampling(previous_logits, temp=temperature, nucleus_p=0.8)
        next_tokens_collected.append(sampled_token[:, None])
        
        # 将采样到的令牌作为输入，进行下一次前向传播
        previous_logits = model.forward(sampled_token, seqlens=[1] * batch_size, cache=buffer_cache)

    # 5. 解码生成的令牌
    final_texts = []
    if next_tokens_collected:
        all_generated_tokens = torch.cat(next_tokens_collected, dim=1)
        for seq_idx, original_tokens in enumerate(encoded_inputs):
            full_sequence = original_tokens + all_generated_tokens[seq_idx].tolist()
            final_texts.append(tokenizer.decode(full_sequence))

    return final_texts, token_logprobs


def chat_session(model_directory: str, max_generated_tokens: int = 35, temp: float = 0.7, use_instruction: bool = False):
    """启动一个交互式聊天会话。"""
    tokenizer = Tokenizer(str(Path(model_directory) / "tokenizer.model"))
    model = Transformer.from_folder(Path(model_directory), max_batch_size=3)

    while True:
        user_input = input("Enter your prompt: ")
        if use_instruction:
            user_input = f"[INST] {user_input} [/INST]"

        responses, _ = generate_text([user_input], model, tokenizer, max_new_tokens=max_generated_tokens, temperature=temp)
        print(responses[0])
        print("-----------")

def run_demo(model_directory: str, max_generated_tokens: int = 35, temp: float = 0.0, pipeline_ranks: int = 1):
    """运行一个批量生成的演示。"""
    if pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        display_output = torch.distributed.get_rank() == 0
    else:
        display_output = True

    tokenizer = Tokenizer(str(Path(model_directory) / "tokenizer.model"))
    model = Transformer.from_folder(Path(model_directory), max_batch_size=3, num_pipeline_ranks=pipeline_ranks)

    prompts = [
        "I'm working on an AI project. Can you recommend some datasets?",
        "Tell me a quick joke.",
        "Explain why Mistral AI models are efficient."
    ]

    generated_texts, log_probs = generate_text(prompts, model, tokenizer, max_new_tokens=max_generated_tokens, temperature=temp)

    if display_output:
        for result, logs in zip(generated_texts, log_probs):
            print(result)
            logging.debug("Logprobs: %s", logs)
            print("-----------")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 使用 fire 库创建命令行接口
    fire.Fire({
        "chat": chat_session,
        "demo": run_demo,
    })