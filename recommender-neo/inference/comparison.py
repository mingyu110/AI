import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ncf import NCF
from models.evaluate import evaluate_model
from models.dataloader import RecSysDataset
from data.preprocess import preprocess_data

def load_model(model_path, device):
    """加载模型"""
    # 从模型信息文件中获取配置
    info_path = f"{os.path.splitext(model_path)[0]}_info.json"
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # 提取配置和数据集统计
        config = model_info.get('config', {})
        dataset_stats = model_info.get('dataset_stats', {})
        
        # 获取模型类型
        if "pruned" in model_path.lower():
            model_type = "剪枝模型"
        elif "finetuned" in model_path.lower():
            model_type = "微调模型"
        else:
            model_type = "Unknown"
    else:
        # 使用默认值
        config = {}
        dataset_stats = {'num_users': 1000, 'num_items': 2000}
        model_type = "Unknown"
    
    # 创建模型实例
    model = NCF(
        num_users=dataset_stats.get('num_users', 1000),
        num_items=dataset_stats.get('num_items', 2000),
        embedding_dim=config.get('embedding_dim', 32),
        mlp_layers=config.get('mlp_layers', [64, 32, 16]),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, model_type, dataset_stats, config

def measure_performance(model, device, num_samples=1000, batch_size=64, num_users=1000, num_items=2000):
    """测量模型的推理性能"""
    # 生成随机用户和物品索引，确保在有效范围内
    max_user_id = num_users - 1
    max_item_id = num_items - 1
    
    user_indices = torch.randint(0, max_user_id, (num_samples,), device=device)
    item_indices = torch.randint(0, max_item_id, (num_samples,), device=device)
    
    # 预热
    with torch.no_grad():
        model(user_indices[:10], item_indices[:10])
    
    # 分批测量时间
    total_time = 0
    batch_times = []
    
    with torch.no_grad():
        # 整体时间
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            batch_users = user_indices[i:i+batch_size]
            batch_items = item_indices[i:i+batch_size]
            
            # 批处理时间
            batch_start = time.time()
            model(batch_users, batch_items)
            batch_end = time.time()
            
            batch_times.append(batch_end - batch_start)
        
        end_time = time.time()
        total_time = end_time - start_time
    
    # 单次推理时间
    single_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(user_indices[0:1], item_indices[0:1])
    single_time = (time.time() - single_time) / 100
    
    # 模拟物品推荐时间 - 确保用户ID有效
    recommend_time = time.time()
    user_id = 0  # 使用第一个用户
    with torch.no_grad():
        for _ in range(10):
            # 为指定用户推荐最多max_item_id个物品
            items_count = min(1000, max_item_id)
            user_tensor = torch.tensor([user_id] * items_count, device=device)
            items_tensor = torch.arange(items_count, device=device)
            ratings = model(user_tensor, items_tensor)
            _, top_indices = torch.topk(ratings, k=min(10, items_count))
    recommend_time = (time.time() - recommend_time) / 10
    
    # 结果
    return {
        "total_pairs": num_samples,
        "total_time": total_time,
        "throughput": num_samples / total_time,
        "avg_batch_time": sum(batch_times) / len(batch_times),
        "avg_single_time": single_time,
        "avg_recommend_time": recommend_time,
        "batch_size": batch_size
    }

def get_recommendations(model, user_id, device, num_items=1000, top_k=10):
    """为用户生成推荐"""
    model.eval()
    with torch.no_grad():
        # 确保不超过嵌入表的大小
        embedding_size = model.item_embedding.weight.shape[0]
        actual_items = min(num_items, embedding_size)
        
        user_tensor = torch.tensor([user_id] * actual_items, device=device)
        items_tensor = torch.arange(actual_items, device=device)
        
        # 预测评分
        ratings = model(user_tensor, items_tensor)
        
        # 获取top-k物品，确保k不超过实际物品数
        actual_k = min(top_k, actual_items)
        scores, indices = torch.topk(ratings, k=actual_k)
        
        return {
            "items": indices.cpu().numpy().tolist(),
            "scores": scores.cpu().numpy().tolist()
        }

def compare_recommendations(models, user_ids, device):
    """比较不同模型的推荐结果"""
    results = {
        "common_recommendations": {},
        "recommendation_similarity": {},
        "all_recommendations": {}
    }
    
    similarity_matrix = np.zeros((len(models), len(models)))
    
    # 为每个用户生成推荐并比较
    for user_id in user_ids:
        recommendations = []
        
        # 从每个模型获取推荐
        for model in models:
            rec = get_recommendations(model, user_id, device)
            recommendations.append(rec)
        
        # 存储所有推荐
        results["all_recommendations"][str(user_id)] = recommendations
        
        # 找出共同的推荐
        common_items = set(recommendations[0]["items"])
        for rec in recommendations[1:]:
            common_items &= set(rec["items"])
        
        results["common_recommendations"][str(user_id)] = list(common_items)
        
        # 计算模型间推荐相似度
        for i in range(len(models)):
            for j in range(len(models)):
                common = len(set(recommendations[i]["items"]) & set(recommendations[j]["items"]))
                similarity = common / 10.0  # 按照top-10计算
                similarity_matrix[i, j] += similarity / len(user_ids)
    
    # 保存相似度矩阵
    results["recommendation_similarity"]["matrix"] = similarity_matrix.tolist()
    
    return results

def plot_performance_comparison(performance_data, output_path):
    """绘制性能对比图"""
    models = ["Original Model", "Pruned Model", "Finetuned Model"]
    
    # 提取性能指标
    throughputs = [data["throughput"] for data in performance_data]
    batch_times = [data["avg_batch_time"] * 1000 for data in performance_data]  # 转换为毫秒
    
    # 创建图表
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # 吞吐量对比
    ax[0].bar(models, throughputs, color=['blue', 'green', 'orange'])
    ax[0].set_title('Throughput (samples/sec)')
    ax[0].set_ylabel('Samples per second')
    
    # 批处理时间对比
    ax[1].bar(models, batch_times, color=['blue', 'green', 'orange'])
    ax[1].set_title('Batch Processing Time (ms)')
    ax[1].set_ylabel('Milliseconds')
    
    # 为每个条形添加标签
    for i, v in enumerate(throughputs):
        ax[0].text(i, v, f"{int(v)}", ha='center', va='bottom')
    
    for i, v in enumerate(batch_times):
        ax[1].text(i, v, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_recommendation_similarity(similarity_matrix, output_path):
    """绘制推荐相似度图"""
    models = ["Original Model", "Pruned Model", "Finetuned Model"]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(similarity_matrix, cmap='YlGn')
    
    # 添加标签
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(models)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加相似度数值
    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, f"{similarity_matrix[i, j]:.3f}",
                          ha="center", va="center", color="black")
    
    ax.set_title("Recommendation Similarity between Models")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(model_paths, output_dir, num_users=50, device='cpu'):
    """
    比较多个模型的性能和推荐结果
    
    参数:
    - model_paths: 模型文件路径列表
    - output_dir: 输出目录
    - num_users: 用于比较推荐的用户数量
    - device: 计算设备
    
    返回:
    - results: 比较结果字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    models = []
    model_info = []
    
    for path in model_paths:
        model, model_type, dataset_stats, config = load_model(path, device)
        models.append(model)
        
        model_info.append({
            "model_type": model_type,
            "model_path": path,
            "device": str(device),
            "num_users": dataset_stats.get("num_users", 1000),
            "num_items": dataset_stats.get("num_items", 2000),
            "embedding_dim": config.get("embedding_dim", 32),
            "mlp_layers": config.get("mlp_layers", [64, 32, 16]),
            "prune_rate": config.get("prune_rate", None)
        })
    
    # 测量性能
    performance_metrics = []
    for i, model in enumerate(models):
        num_users_model = model_info[i]["num_users"] 
        num_items_model = model_info[i]["num_items"]
        metrics = measure_performance(
            model, 
            device, 
            num_samples=1000, 
            batch_size=64,
            num_users=num_users_model,
            num_items=num_items_model
        )
        performance_metrics.append(metrics)
    
    # 获取所有模型通用的用户数范围
    max_common_user_id = min(info["num_users"] for info in model_info) - 1
    
    # 随机选择用户进行推荐比较，确保在所有模型的有效范围内
    np.random.seed(42)
    compare_user_count = min(num_users, max_common_user_id + 1)
    user_ids = np.random.choice(max_common_user_id + 1, size=compare_user_count, replace=False)
    
    # 比较推荐
    recommendation_results = compare_recommendations(models, user_ids, device)
    
    # 合并结果
    results = {
        "model_info": model_info,
        "performance_metrics": performance_metrics,
        **recommendation_results
    }
    
    # 绘制性能比较图
    plot_performance_comparison(
        performance_metrics, 
        os.path.join(output_dir, "performance_comparison.png")
    )
    
    # 绘制推荐相似度图
    plot_recommendation_similarity(
        np.array(recommendation_results["recommendation_similarity"]["matrix"]),
        os.path.join(output_dir, "recommendation_similarity.png")
    )
    
    # 保存结果为JSON
    with open(os.path.join(output_dir, "comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    # 测试代码
    model_paths = [
        "./output/models/original_model.pth",
        "./output/models/pruned_model.pth",
        "./output/models/finetuned_model.pth"
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare_models(model_paths, "./output/comparison_results", device=device) 