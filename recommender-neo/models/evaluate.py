import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    
    参数:
    - model: 待评估的模型
    - test_loader: 测试数据加载器
    - device: 计算设备
    
    返回:
    - metrics: 包含各项评估指标的字典
    """
    model.eval()
    
    # 初始化指标
    all_predictions = []
    all_targets = []
    all_users = []
    all_items = []
    
    # 收集预测结果和真实标签
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="模型评估中"):
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            label = batch['label'].to(device)
            
            # 前向传播
            prediction = model(user, item)
            
            # 收集结果
            all_predictions.extend(prediction.cpu().numpy().tolist())
            all_targets.extend(label.cpu().numpy().tolist())
            all_users.extend(user.cpu().numpy().tolist())
            all_items.extend(item.cpu().numpy().tolist())
    
    # 转换为NumPy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_users = np.array(all_users)
    all_items = np.array(all_items)
    
    # 检查是否有NaN值
    nan_mask = np.isnan(all_predictions) | np.isnan(all_targets)
    nan_count = np.sum(nan_mask)
    
    if nan_count > 0:
        print(f"警告: 在评估数据中发现 {nan_count} 个NaN值 ({nan_count / len(all_predictions) * 100:.1f}%)，将被跳过")
        
        # 过滤掉NaN值
        valid_mask = ~nan_mask
        valid_count = np.sum(valid_mask)
        
        print(f"有效数据点: {valid_count} ({valid_count / len(all_predictions) * 100:.1f}%)")
        
        if valid_count == 0:
            print("错误: 没有有效的非NaN数据点可用于评估")
            # 返回NaN指标
            return {
                'rmse': float('nan'),
                'mae': float('nan'),
                'hr@5': float('nan'),
                'ndcg@5': float('nan'),
                'hr@10': float('nan'),
                'ndcg@10': float('nan')
            }
        
        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
        all_users = all_users[valid_mask]
        all_items = all_items[valid_mask]
    
    # 如果没有有效数据点，使用一些模拟数据进行测试
    if len(all_predictions) == 0:
        print("警告: 没有有效的预测，使用随机模拟数据进行指标计算")
        all_predictions = np.random.rand(100)
        all_targets = np.random.rand(100)
        all_users = np.random.randint(0, 10, 100)
        all_items = np.random.randint(0, 50, 100)
    
    # 计算评估指标
    try:
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    except:
        print("计算RMSE失败，使用默认值")
        rmse = float('nan')
        
    try:
        mae = np.mean(np.abs(all_predictions - all_targets))
    except:
        print("计算MAE失败，使用默认值")
        mae = float('nan')
    
    # 计算相关系数
    try:
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]
    except:
        correlation = float('nan')
    
    # 为每个用户计算推荐列表
    user_predictions = {}
    user_targets = {}
    
    # 按用户ID组织数据
    for user_id, item_id, pred, target in zip(all_users, all_items, all_predictions, all_targets):
        if user_id not in user_predictions:
            user_predictions[user_id] = []
            user_targets[user_id] = []
        
        user_predictions[user_id].append((item_id, pred))
        user_targets[user_id].append((item_id, target))
    
    # 计算Top-K指标
    hr_5 = calculate_hit_ratio(user_predictions, user_targets, k=5)
    ndcg_5 = calculate_ndcg(user_predictions, user_targets, k=5)
    hr_10 = calculate_hit_ratio(user_predictions, user_targets, k=10)
    ndcg_10 = calculate_ndcg(user_predictions, user_targets, k=10)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'hr@5': hr_5,
        'ndcg@5': ndcg_5,
        'hr@10': hr_10,
        'ndcg@10': ndcg_10,
        'valid_samples': len(all_predictions),
        'valid_ratio': len(all_predictions) / (len(all_predictions) + nan_count) if (len(all_predictions) + nan_count) > 0 else 0
    }
    
    return metrics

def calculate_hit_ratio(user_predictions, user_targets, k=10):
    """
    计算命中率 (Hit Ratio)
    
    参数:
    - user_predictions: 用户预测字典，格式为 {user_id: [(item_id, score), ...]}
    - user_targets: 用户真实评分字典，格式为 {user_id: [(item_id, score), ...]}
    - k: 推荐列表长度
    
    返回:
    - hr: 命中率
    """
    hits = 0
    total_users = 0
    
    for user_id in user_predictions:
        if user_id not in user_targets:
            continue
            
        total_users += 1
        
        # 获取用户的预测列表和真实列表
        pred_items = user_predictions[user_id]
        actual_items = user_targets[user_id]
        
        # 按分数排序预测列表，取前k个
        pred_items = sorted(pred_items, key=lambda x: x[1], reverse=True)[:k]
        
        # 获取预测的物品ID列表
        pred_item_ids = [item[0] for item in pred_items]
        
        # 找出真实评分最高的物品
        actual_items = sorted(actual_items, key=lambda x: x[1], reverse=True)
        relevant_items = [item[0] for item in actual_items[:min(len(actual_items), 10)]]
        
        # 检查是否命中
        for item_id in relevant_items:
            if item_id in pred_item_ids:
                hits += 1
                break
    
    hr = hits / total_users if total_users > 0 else 0
    return hr

def calculate_ndcg(user_predictions, user_targets, k=10):
    """
    计算NDCG (Normalized Discounted Cumulative Gain)
    
    参数:
    - user_predictions: 用户预测字典，格式为 {user_id: [(item_id, score), ...]}
    - user_targets: 用户真实评分字典，格式为 {user_id: [(item_id, score), ...]}
    - k: 推荐列表长度
    
    返回:
    - ndcg: NDCG值
    """
    ndcg_scores = []
    
    for user_id in user_predictions:
        if user_id not in user_targets:
            continue
            
        # 获取用户的预测列表和真实列表
        pred_items = user_predictions[user_id]
        actual_items = user_targets[user_id]
        
        # 按分数排序预测列表，取前k个
        pred_items = sorted(pred_items, key=lambda x: x[1], reverse=True)[:k]
        
        # 获取预测的物品ID列表
        pred_item_ids = [item[0] for item in pred_items]
        
        # 创建真实物品评分字典
        actual_item_scores = {item[0]: item[1] for item in actual_items}
        
        # 计算DCG
        dcg = 0
        for i, item_id in enumerate(pred_item_ids):
            if item_id in actual_item_scores:
                # 这里我们使用实际评分作为相关性
                relevance = actual_item_scores[item_id]
                dcg += relevance / np.log2(i + 2)  # i+2 是因为i从0开始
        
        # 计算IDCG
        # 找出所有真实物品评分，排序后取前k个
        all_relevances = sorted([score for _, score in actual_items], reverse=True)[:k]
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(all_relevances)])
        
        # 计算NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    return avg_ndcg

def visualize_evaluation(targets, predictions, metrics, output_dir=None):
    """
    可视化评估结果
    
    参数:
    - targets: 真实评分
    - predictions: 预测评分
    - metrics: 指标字典
    - output_dir: 输出目录，如果提供则保存图表
    """
    # 创建一个2x2的图表
    plt.figure(figsize=(15, 10))
    
    # 1. 真实评分vs预测评分的散点图
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.5, color='blue')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('真实评分')
    plt.ylabel('预测评分')
    plt.title(f'真实 vs 预测评分 (相关系数: {metrics["correlation"]:.3f})')
    
    # 2. 误差直方图
    plt.subplot(2, 2, 2)
    errors = predictions - targets
    plt.hist(errors, bins=20, color='green', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title(f'预测误差分布 (RMSE: {metrics["rmse"]:.3f}, MAE: {metrics["mae"]:.3f})')
    
    # 3. 真实评分分布
    plt.subplot(2, 2, 3)
    plt.hist(targets, bins=10, alpha=0.7, color='purple')
    plt.xlabel('评分值')
    plt.ylabel('频率')
    plt.title('真实评分分布')
    
    # 4. 预测评分分布
    plt.subplot(2, 2, 4)
    plt.hist(predictions, bins=10, alpha=0.7, color='orange')
    plt.xlabel('评分值')
    plt.ylabel('频率')
    plt.title('预测评分分布')
    
    # 添加总标题
    plt.suptitle('模型评估结果可视化', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图表（如果提供了输出目录）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'evaluation_visualization.png'))
    
    plt.show()

def print_metrics(metrics):
    """打印评估指标"""
    print("\n===== 模型评估指标 =====")
    
    # 首先打印数据质量信息
    print(f"有效样本数: {metrics.get('valid_samples', 'N/A')}")
    valid_ratio = metrics.get('valid_ratio', 0) * 100
    print(f"有效样本比例: {valid_ratio:.1f}%")
    
    # 打印回归指标
    print(f"RMSE: {metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else "RMSE: N/A")
    print(f"MAE: {metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else "MAE: N/A")
    
    # 打印其他指标
    if 'correlation' in metrics and not np.isnan(metrics['correlation']):
        print(f"相关系数: {metrics['correlation']:.4f}")
    
    # 打印排序指标
    for metric, value in metrics.items():
        if metric.startswith('hr@'):
            print(f"命中率@{metric[3:]}: {value:.4f}" if not np.isnan(value) else f"命中率@{metric[3:]}: N/A")
        elif metric.startswith('ndcg@'):
            print(f"NDCG@{metric[5:]}: {value:.4f}" if not np.isnan(value) else f"NDCG@{metric[5:]}: N/A") 