import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def count_parameters(model):
    """
    统计模型的参数数量
    
    参数:
    - model: PyTorch模型
    
    返回:
    - total_params: 总参数数量
    - trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def analyze_pruned_model(model):
    """
    分析剪枝后模型的稀疏度
    
    参数:
    - model: 剪枝后的PyTorch模型
    
    返回:
    - layer_stats: 每一层的稀疏度统计信息
    """
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if param.dim() > 1:  # 只分析权重矩阵
            # 计算非零元素的数量
            nonzero_count = torch.sum(param != 0).item()
            total_count = param.numel()
            sparsity = 1.0 - (nonzero_count / total_count)
            
            layer_stats[name] = {
                'nonzero_params': nonzero_count,
                'total_params': total_count,
                'sparsity': sparsity
            }
    
    return layer_stats

def analyze_weights_distribution(model, save_path=None):
    """
    分析模型权重分布并绘制直方图
    
    参数:
    - model: PyTorch模型
    - save_path: 保存图表的路径(如果为None则不保存)
    
    返回:
    - stats: 权重统计信息
    """
    weights = []
    weight_names = []
    
    # 收集所有权重
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy().flatten())
            weight_names.append(name)
    
    # 将所有权重合并
    all_weights = np.concatenate(weights)
    
    # 计算统计信息
    stats = {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'median': float(np.median(all_weights)),
        'sparsity': float((np.abs(all_weights) < 1e-6).mean())
    }
    
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_weights, bins=100, alpha=0.7, color='blue')
    plt.title('模型权重分布')
    plt.xlabel('权重值')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    info_text = f"""
    均值: {stats['mean']:.6f}
    标准差: {stats['std']:.6f}
    最小值: {stats['min']:.6f}
    最大值: {stats['max']:.6f}
    稀疏度: {stats['sparsity']*100:.2f}%
    """
    plt.figtext(0.15, 0.7, info_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
        print(f"权重分布图已保存到 {save_path}")
    
    plt.close()
    
    return stats

def find_pruning_threshold(model, prune_rate=0.4):
    """
    根据指定的剪枝率找到剪枝阈值
    
    参数:
    - model: PyTorch模型
    - prune_rate: 剪枝比例
    
    返回:
    - threshold: 剪枝阈值
    """
    weights = []
    
    # 收集所有权重
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.cpu().numpy().flatten())
    
    # 将所有权重合并
    all_weights = np.concatenate(weights)
    
    # 计算绝对值
    abs_weights = np.abs(all_weights)
    
    # 根据剪枝率找到阈值
    threshold = np.quantile(abs_weights, prune_rate)
    
    return threshold

def compare_pruned_models(models_info, save_path=None):
    """
    比较不同剪枝率下的模型性能
    
    参数:
    - models_info: 包含模型信息的列表，每个元素是一个字典，包含剪枝率和性能指标
    - save_path: 保存图表的路径(如果为None则不保存)
    """
    prune_rates = [info['prune_rate'] for info in models_info]
    rmse_values = [info['metrics']['rmse'] for info in models_info]
    hr10_values = [info['metrics'].get('hr@10', 0) for info in models_info]
    model_sizes = [info['model_size'] for info in models_info]
    
    # 绘制性能和模型大小的比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：性能指标
    ax1.plot(prune_rates, rmse_values, 'o-', label='RMSE', color='blue')
    ax1.set_xlabel('剪枝率')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(prune_rates, hr10_values, 'o-', label='HR@10', color='red')
    ax1_twin.set_ylabel('HR@10', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('剪枝率对模型性能的影响')
    ax1.grid(True, alpha=0.3)
    
    # 右图：模型大小
    ax2.plot(prune_rates, model_sizes, 'o-', label='模型大小', color='green')
    ax2.set_xlabel('剪枝率')
    ax2.set_ylabel('模型大小 (MB)')
    ax2.set_title('剪枝率对模型大小的影响')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"模型比较图已保存到 {save_path}")
    
    plt.close()

def summarize_pruning_experiment(experiments_dir, output_path=None):
    """
    总结剪枝实验结果并生成报告
    
    参数:
    - experiments_dir: 包含实验结果的目录路径
    - output_path: 报告保存路径(如果为None则只返回结果)
    
    返回:
    - summary: 实验结果摘要DataFrame
    """
    results = []
    
    # 遍历实验目录
    for filename in os.listdir(experiments_dir):
        if filename.endswith('_info.json'):
            file_path = os.path.join(experiments_dir, filename)
            
            with open(file_path, 'r') as f:
                info = json.load(f)
            
            # 提取关键信息
            if 'pruned' in filename.lower():
                exp_type = 'Pruned'
            elif 'finetuned' in filename.lower():
                exp_type = 'Finetuned'
            else:
                exp_type = 'Original'
            
            result = {
                'Model': os.path.splitext(os.path.splitext(filename)[0])[0],
                'Type': exp_type,
                'Size (MB)': info.get('model_size', info.get('pruned_size', 0)),
                'RMSE': info.get('metrics', info.get('pruned_metrics', info.get('fine_tuned_metrics', {})))
                .get('rmse', 0),
                'HR@10': info.get('metrics', info.get('pruned_metrics', info.get('fine_tuned_metrics', {})))
                .get('hr@10', 0),
                'Prune Rate': info.get('prune_rate', 0) if exp_type != 'Original' else 0
            }
            
            results.append(result)
    
    # 创建DataFrame
    summary = pd.DataFrame(results)
    
    # 按类型和剪枝率排序
    summary = summary.sort_values(['Type', 'Prune Rate'])
    
    # 保存为CSV
    if output_path:
        summary.to_csv(output_path, index=False)
        print(f"实验摘要已保存到 {output_path}")
    
    return summary 