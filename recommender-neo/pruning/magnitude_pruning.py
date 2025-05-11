import os
import torch
import torch.nn.utils.prune as prune
import numpy as np
import sys
import json
import time
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ncf import NCF
from models.evaluate import evaluate_model

def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model, unit='MB'):
    """获取模型大小"""
    # 获取模型参数的总字节数
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # 获取缓冲区的总字节数(如BN层的running_mean)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_bytes = param_size + buffer_size
    
    # 转换单位
    if unit == 'KB':
        size = size_bytes / 1024
    elif unit == 'MB':
        size = size_bytes / (1024 * 1024)
    else:
        size = size_bytes
    
    return size

def apply_structured_pruning(model, prune_rate=0.6, embedding_prune_rate=0.3):
    """
    应用结构化剪枝策略针对NCF模型
    
    参数:
    - model: 待剪枝的NCF模型
    - prune_rate: 线性层剪枝比例
    - embedding_prune_rate: 嵌入层剪枝比例
    
    返回:
    - pruned_model: 剪枝后的模型
    """
    print(f"应用结构化剪枝, 线性层剪枝率: {prune_rate}, 嵌入层剪枝率: {embedding_prune_rate}")
    
    # 获取嵌入层范数
    embedding_norms = model.get_embedding_norm()
    user_norms = embedding_norms['user_embedding_norm']
    item_norms = embedding_norms['item_embedding_norm']
    
    # 1. 结构化剪枝嵌入层 - 移除不活跃的用户和物品嵌入
    # 用户嵌入剪枝
    num_users = model.user_embedding.weight.size(0)
    num_users_to_prune = int(num_users * embedding_prune_rate)
    
    if num_users_to_prune > 0:
        # 找出范数最小的用户嵌入
        _, user_indices_to_prune = torch.topk(user_norms, k=num_users_to_prune, largest=False)
        
        # 为不活跃用户创建掩码
        user_mask = torch.ones(num_users, dtype=torch.bool, device=user_norms.device)
        user_mask[user_indices_to_prune] = False
        
        # 更新用户嵌入矩阵
        model.user_embedding.weight.data[user_indices_to_prune] = 0.0
        print(f"用户嵌入剪枝: 从 {num_users} 中移除 {num_users_to_prune} 个不活跃用户嵌入")
    
    # 物品嵌入剪枝
    num_items = model.item_embedding.weight.size(0)
    num_items_to_prune = int(num_items * embedding_prune_rate)
    
    if num_items_to_prune > 0:
        # 找出范数最小的物品嵌入
        _, item_indices_to_prune = torch.topk(item_norms, k=num_items_to_prune, largest=False)
        
        # 为不活跃物品创建掩码
        item_mask = torch.ones(num_items, dtype=torch.bool, device=item_norms.device)
        item_mask[item_indices_to_prune] = False
        
        # 更新物品嵌入矩阵
        model.item_embedding.weight.data[item_indices_to_prune] = 0.0
        print(f"物品嵌入剪枝: 从 {num_items} 中移除 {num_items_to_prune} 个不活跃物品嵌入")
    
    # 2. 对MLP层应用结构化行剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight
            out_features, in_features = weight.shape
            
            # 对权重按行计算L2范数
            row_norms = torch.norm(weight, p=2, dim=1)
            
            # 确定要剪枝的行数
            num_rows_to_prune = int(out_features * prune_rate)
            
            if num_rows_to_prune > 0:
                # 找到范数最小的行
                _, row_indices = torch.topk(row_norms, k=num_rows_to_prune, largest=False)
                
                # 将这些行设置为0
                weight.data[row_indices, :] = 0.0
                
                # 如果存在偏置，也将对应的偏置置零
                if module.bias is not None:
                    module.bias.data[row_indices] = 0.0
                
                print(f"层 {name} 形状 {weight.shape}: 剪枝了 {num_rows_to_prune} 行")
    
    # 计算剪枝后的非零参数比例
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = 1.0 - (nonzero_params / total_params)
    print(f"总参数数量: {total_params}, 非零参数: {nonzero_params}, 稀疏度: {sparsity:.4f}")
    
    return model

def apply_magnitude_pruning(model, prune_rate=0.7, modules_to_prune=None):
    """
    应用幅度剪枝
    
    参数:
    - model: 待剪枝的模型
    - prune_rate: 剪枝比例，表示移除的权重比例
    - modules_to_prune: 需要剪枝的模块列表，如果为None则剪枝所有线性层和嵌入层
    
    返回:
    - pruned_model: 剪枝后的模型
    """
    # 创建剪枝模块列表
    if modules_to_prune is None:
        modules_to_prune = []
        for name, module in model.named_modules():
            # 针对NCF模型的特定结构剪枝
            if isinstance(module, torch.nn.Linear):
                modules_to_prune.append((module, 'weight'))
                # 也剪枝偏置项
                if hasattr(module, 'bias') and module.bias is not None:
                    modules_to_prune.append((module, 'bias'))
            elif isinstance(module, torch.nn.Embedding):
                # 对嵌入层应用较低的剪枝率以保留更多语义信息
                modules_to_prune.append((module, 'weight'))
    
    print(f"找到{len(modules_to_prune)}个可剪枝模块")
    
    # 检查是否找到了可剪枝的模块
    if len(modules_to_prune) == 0:
        print("警告: 没有找到可剪枝的模块!")
        return model
    
    # 应用幅度剪枝
    for i, (module, param_name) in enumerate(modules_to_prune):
        # 对嵌入层使用较低的剪枝率
        current_rate = prune_rate * 0.6 if isinstance(module, torch.nn.Embedding) else prune_rate
        
        # 打印剪枝信息
        param = getattr(module, param_name)
        print(f"剪枝模块 {i+1}/{len(modules_to_prune)}: {param_name} - 形状:{param.shape} - 剪枝率:{current_rate:.2f}")
        
        # 应用L1范数非结构化剪枝
        prune.l1_unstructured(module, name=param_name, amount=current_rate)
        
        # 检查剪枝后的稀疏度
        mask = getattr(module, f"{param_name}_mask", None)
        if mask is not None:
            sparsity = 1.0 - (torch.sum(mask).item() / mask.numel())
            print(f"  剪枝后稀疏度: {sparsity:.4f}")
    
    # 使剪枝永久化
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    return model

def save_pruned_model(model, path, model_info=None):
    """保存剪枝后的模型"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存模型权重
    torch.save(model.state_dict(), path)
    
    # 如果提供了模型信息，保存为JSON
    if model_info:
        info_path = f"{os.path.splitext(path)[0]}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
    
    print(f"剪枝模型已保存到 {path}")

def prune_model(original_model_path, output_model_path, prune_rate=0.7, test_loader=None, device='cpu', use_structured_pruning=True):
    """
    剪枝模型并对比性能变化
    
    参数:
    - original_model_path: 原始模型路径
    - output_model_path: 剪枝后模型保存路径
    - prune_rate: 剪枝比例
    - test_loader: 测试数据加载器(用于性能评估)
    - device: 计算设备
    - use_structured_pruning: 是否使用结构化剪枝
    
    返回:
    - pruned_model: 剪枝后的模型
    - performance_diff: 性能变化指标
    """
    # 加载原始模型信息
    original_info_path = f"{os.path.splitext(original_model_path)[0]}_info.json"
    if os.path.exists(original_info_path):
        with open(original_info_path, 'r') as f:
            original_info = json.load(f)
        
        config = original_info.get('config', {})
        dataset_stats = original_info.get('dataset_stats', {})
    else:
        print("找不到原始模型信息文件，使用默认配置")
        config = {}
        dataset_stats = {'num_users': 1000, 'num_items': 2000}
    
    # 创建模型实例
    model = NCF(
        num_users=dataset_stats.get('num_users'),
        num_items=dataset_stats.get('num_items'),
        embedding_dim=config.get('embedding_dim', 32),
        mlp_layers=config.get('mlp_layers', [64, 32, 16]),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # 加载原始模型权重
    model.load_state_dict(torch.load(original_model_path, map_location=device))
    
    # 计算原始模型大小和参数数量
    original_size = get_model_size(model, 'MB')
    original_params = count_parameters(model)
    original_nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    
    # 测试原始模型性能
    original_metrics = None
    if test_loader:
        print("测试原始模型性能...")
        original_metrics = evaluate_model(model, test_loader, device)
        print(f"原始模型性能: RMSE={original_metrics['rmse']:.4f}, HR@10={original_metrics.get('hr@10', 0):.4f}")
    
    # 应用剪枝
    print(f"使用{'结构化剪枝' if use_structured_pruning else '幅度剪枝'}, 剪枝比例: {prune_rate*100:.0f}%")
    if use_structured_pruning:
        # 结构化剪枝更适合NCF模型
        pruned_model = apply_structured_pruning(
            model, 
            prune_rate=prune_rate, 
            embedding_prune_rate=prune_rate*0.5  # 嵌入层使用较低的剪枝率
        )
    else:
        # 幅度剪枝
        pruned_model = apply_magnitude_pruning(model, prune_rate)
    
    # 计算剪枝后模型大小和参数数量
    pruned_size = get_model_size(pruned_model, 'MB')
    pruned_params = count_parameters(pruned_model)
    pruned_nonzero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    
    # 测试剪枝后模型性能
    pruned_metrics = None
    if test_loader:
        print("测试剪枝后模型性能...")
        pruned_metrics = evaluate_model(pruned_model, test_loader, device)
        print(f"剪枝后模型性能: RMSE={pruned_metrics['rmse']:.4f}, HR@10={pruned_metrics.get('hr@10', 0):.4f}")
    
    # 计算性能变化
    performance_diff = {}
    if original_metrics and pruned_metrics:
        for metric in original_metrics:
            performance_diff[metric] = pruned_metrics[metric] - original_metrics[metric]
    
    # 打印结果
    print("\n===== 剪枝结果 =====")
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"剪枝后模型大小: {pruned_size:.2f} MB")
    print(f"模型大小减少: {original_size - pruned_size:.2f} MB ({(original_size - pruned_size) / original_size * 100:.2f}%)")
    print(f"原始模型参数数量: {original_params}")
    print(f"剪枝后模型参数数量: {pruned_params}")
    print(f"原始模型非零参数: {original_nonzero_params} ({original_nonzero_params/original_params*100:.2f}%)")
    print(f"剪枝后模型非零参数: {pruned_nonzero_params} ({pruned_nonzero_params/pruned_params*100:.2f}%)")
    print(f"参数稀疏度: {1.0 - pruned_nonzero_params/pruned_params:.4f}")
    
    if performance_diff:
        print("\n===== 性能变化 =====")
        for metric, diff in performance_diff.items():
            if metric in ['rmse', 'mae']:
                print(f"{metric.upper()}: {diff:.4f} ({'变差' if diff > 0 else '变好' if diff < 0 else '不变'})")
            elif metric.startswith('hr@') or metric.startswith('ndcg@'):
                print(f"{metric}: {diff:.4f} ({'变差' if diff < 0 else '变好' if diff > 0 else '不变'})")
    
    # 保存剪枝后的模型
    pruned_info = {
        'original_model': original_model_path,
        'prune_rate': prune_rate,
        'pruning_method': 'structured' if use_structured_pruning else 'magnitude',
        'original_size': original_size,
        'pruned_size': pruned_size,
        'size_reduction': (original_size - pruned_size) / original_size,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'original_nonzero_params': int(original_nonzero_params),
        'pruned_nonzero_params': int(pruned_nonzero_params),
        'sparsity': float(1.0 - pruned_nonzero_params/pruned_params),
        'original_metrics': original_metrics,
        'pruned_metrics': pruned_metrics,
        'performance_diff': performance_diff,
        'config': config,
        'dataset_stats': dataset_stats
    }
    
    save_pruned_model(pruned_model, output_model_path, pruned_info)
    
    return pruned_model, performance_diff

"""
# 注意：以下函数未成功实现 Neo 优化，建议使用 aws/auto_neo_ncf_compat_test.py
# 使用SageMaker Neo进行模型编译和优化（不是剪枝）
def prepare_for_sagemaker_neo(model_path, output_dir, model_config):
    # 此函数已被弃用，请使用 aws/auto_neo_ncf_compat_test.py 进行 Neo 优化
    pass
"""

if __name__ == "__main__":
    print("示例: 剪枝NCF模型")
    print("注意: 这只是一个示例，实际使用需要提供模型文件和测试数据")
    print("使用方法: python -m pruning.magnitude_pruning [模型路径] [输出路径] [剪枝率]") 