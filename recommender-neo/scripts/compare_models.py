#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较所有模型性能脚本

此脚本用于比较所有模型的性能，包括：
1. 原始模型
2. 剪枝模型
3. 微调模型
4. Neo优化模型(可选)

脚本会生成详细的性能对比图表，展示各种指标的差异。

用法示例:
python -m scripts.compare_models --output_dir ./output/comparison --neo_model_dir ./output/neo/extracted
"""

import os
import sys
import argparse
import logging
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from models.ncf import NCF
from inference.comparison import compare_models, measure_performance, load_model
from utils.visualization import create_performance_charts

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("compare-models")

def setup_args():
    """
    设置命令行参数
    """
    parser = argparse.ArgumentParser(description='比较所有推荐模型的性能')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output/comparison',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='',
                        help='计算设备(留空则自动检测)')
    
    # 模型路径参数
    parser.add_argument('--original_model_path', type=str, default='./output/models/ncf_model_latest.pt',
                        help='原始模型路径')
    parser.add_argument('--pruned_model_path', type=str, default='./output/models/pruned/ncf_pruned_latest.pt',
                        help='剪枝模型路径')
    parser.add_argument('--finetuned_model_path', type=str, default='./output/models/pruned/ncf_finetuned_latest.pt',
                        help='微调模型路径')
    parser.add_argument('--neo_model_dir', type=str, default='',
                        help='Neo模型所在的本地目录(留空则跳过Neo模型)')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='评估时的批量大小')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='评估时的样本数量')
    
    return parser.parse_args()

def check_dlr_installation():
    """
    检查DLR是否已安装
    """
    try:
        import dlr
        logger.info("DLR已正确安装")
        return True
    except ImportError:
        logger.warning("未找到DLR库，无法加载Neo优化模型。如需评估Neo模型，请使用'pip install dlr'安装DLR库。")
        return False

def load_neo_model(model_dir, device):
    """
    加载Neo编译后的模型
    
    参数:
    - model_dir: Neo模型所在目录
    - device: 计算设备
    
    返回:
    - model: 加载的Neo模型
    """
    try:
        # 检查DLR是否安装
        if not check_dlr_installation():
            return None
        
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            logger.error(f"Neo模型目录不存在: {model_dir}")
            return None
        
        # 检查必要的模型文件
        required_files = ["compiled.params", "compiled.so"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            logger.error(f"Neo模型目录中缺少必要文件: {missing_files}")
            return None
        
        # 设置环境变量以帮助DLR找到库文件
        import os
        os.environ['DYLD_LIBRARY_PATH'] = f"{model_dir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{model_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        # 尝试导入DLR
        import dlr
        
        logger.info(f"使用DLR加载Neo编译后的模型: {model_dir}")
        
        # 创建DLR设备字符串
        dlr_device = "cpu" if str(device) == "cpu" else "gpu"
        
        # 加载模型
        neo_model = dlr.DLRModel(model_dir, dlr_device)
        logger.info("Neo模型加载成功")
        
        # 创建包装类，使其接口与PyTorch模型兼容
        class NeoModelWrapper:
            def __init__(self, dlr_model):
                self.dlr_model = dlr_model
            
            def eval(self):
                return self
            
            def to(self, device):
                return self
            
            def __call__(self, input_data):
                # DLR模型预期输入是NumPy数组
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.detach().cpu().numpy()
                
                try:
                    result = self.dlr_model.run(input_data)[0]
                    return torch.tensor(result, device="cpu")
                except Exception as e:
                    logger.error(f"Neo模型推理失败: {str(e)}")
                    return torch.zeros(input_data.shape[0], 1, device="cpu")
        
        # 返回包装后的模型
        return NeoModelWrapper(neo_model)
        
    except Exception as e:
        logger.error(f"加载Neo模型失败: {str(e)}")
        return None

def load_models(args):
    """
    加载所有模型
    
    参数:
    - args: 命令行参数
    
    返回:
    - models: 加载的模型字典
    """
    # 设置设备
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    models = {}
    
    # 1. 加载原始模型
    if os.path.exists(args.original_model_path):
        logger.info(f"加载原始模型: {args.original_model_path}")
        original_model = load_model(args.original_model_path, device)
        if original_model:
            models['original'] = original_model
            logger.info("原始模型加载成功")
        else:
            logger.error("原始模型加载失败")
    else:
        logger.warning(f"原始模型文件不存在: {args.original_model_path}")
    
    # 2. 加载剪枝模型
    if os.path.exists(args.pruned_model_path):
        logger.info(f"加载剪枝模型: {args.pruned_model_path}")
        pruned_model = load_model(args.pruned_model_path, device)
        if pruned_model:
            models['pruned'] = pruned_model
            logger.info("剪枝模型加载成功")
        else:
            logger.error("剪枝模型加载失败")
    else:
        logger.warning(f"剪枝模型文件不存在: {args.pruned_model_path}")
    
    # 3. 加载微调模型
    if os.path.exists(args.finetuned_model_path):
        logger.info(f"加载微调模型: {args.finetuned_model_path}")
        finetuned_model = load_model(args.finetuned_model_path, device)
        if finetuned_model:
            models['finetuned'] = finetuned_model
            logger.info("微调模型加载成功")
        else:
            logger.error("微调模型加载失败")
    else:
        logger.warning(f"微调模型文件不存在: {args.finetuned_model_path}")
    
    # 4. 加载Neo模型(如果指定)
    if args.neo_model_dir:
        logger.info(f"加载Neo优化模型: {args.neo_model_dir}")
        neo_model = load_neo_model(args.neo_model_dir, device)
        if neo_model:
            models['neo'] = neo_model
            logger.info("Neo优化模型加载成功")
        else:
            logger.error("Neo优化模型加载失败")
    
    return models, device

def compare_all_models(models, device, batch_size=64, num_samples=1000):
    """
    比较所有模型的性能
    
    参数:
    - models: 模型字典
    - device: 计算设备
    - batch_size: 批量大小
    - num_samples: 样本数量
    
    返回:
    - results: 性能比较结果
    """
    logger.info("开始比较所有模型的性能...")
    
    # 确保至少加载了一个模型
    if not models:
        logger.error("没有加载任何模型，无法进行比较")
        return None
    
    # 获取模型元数据
    model_keys = list(models.keys())
    first_model_key = model_keys[0]
    first_model = models[first_model_key]
    
    # 尝试获取模型的用户数和物品数
    num_users = 1000
    num_items = 2000
    
    try:
        # 检查是否为NCF模型
        if isinstance(first_model, NCF):
            num_users = first_model.num_users
            num_items = first_model.num_items
        # 加载原始模型的配置信息
        elif os.path.exists(args.original_model_path):
            checkpoint = torch.load(args.original_model_path, map_location='cpu')
            num_users = checkpoint.get('num_users', 1000)
            num_items = checkpoint.get('num_items', 2000)
    except Exception as e:
        logger.warning(f"获取模型配置失败，使用默认值: {str(e)}")
    
    logger.info(f"使用模型配置 - 用户数: {num_users}, 物品数: {num_items}")
    
    # 收集性能指标
    results = {
        'models': [],
        'rmse': [],
        'throughput': [],
        'batch_time': [],
        'params': [],
        'sparsity': []
    }
    
    # 评估每个模型
    for model_name, model in models.items():
        logger.info(f"评估模型: {model_name}")
        
        # 测量模型性能
        try:
            # 处理Neo模型的特殊情况
            if model_name == 'neo':
                # 生成测试数据
                user_ids = np.random.randint(0, num_users, size=(num_samples, 1))
                item_ids = np.random.randint(0, num_items, size=(num_samples, 1))
                inputs = np.hstack([user_ids, item_ids]).astype(np.float32)
                
                # 评估吞吐量
                start_time = time.time()
                batch_times = []
                
                # 预热
                for _ in range(5):
                    batch_input = inputs[:batch_size]
                    try:
                        model.dlr_model.run(batch_input)
                    except:
                        pass
                
                # 实际测量
                for i in range(0, num_samples, batch_size):
                    batch_input = inputs[i:i+batch_size]
                    batch_start = time.time()
                    try:
                        _ = model.dlr_model.run(batch_input)
                        batch_end = time.time()
                        batch_times.append(batch_end - batch_start)
                    except Exception as e:
                        logger.error(f"Neo模型批次推理失败: {str(e)}")
                        batch_times.append(1.0)  # 1秒表示性能很差
                
                end_time = time.time()
                total_time = end_time - start_time
                throughput = num_samples / total_time if total_time > 0 else 0
                avg_batch_time = np.mean(batch_times) if batch_times else 0
                
                performance = {
                    'throughput': throughput,
                    'avg_batch_time': avg_batch_time,
                    'rmse': models['original'].rmse if 'original' in models else 0  # 使用原始模型的RMSE
                }
            else:
                # 常规PyTorch模型性能评估
                performance = measure_performance(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    num_samples=num_samples,
                    num_users=num_users,
                    num_items=num_items
                )
            
            # 获取模型参数信息
            params_info = {}
            if model_name == 'neo':
                # Neo模型使用原始模型的参数信息
                if 'original' in models:
                    params_info = get_model_params_info(models['original'])
                else:
                    params_info = {'total_params': 0, 'nonzero_params': 0, 'sparsity': 0}
            else:
                params_info = get_model_params_info(model)
            
            # 添加结果
            friendly_name = {
                'original': '原始模型',
                'pruned': '剪枝模型',
                'finetuned': '微调模型',
                'neo': 'Neo优化模型'
            }.get(model_name, model_name)
            
            results['models'].append(friendly_name)
            results['rmse'].append(performance.get('rmse', 0))
            results['throughput'].append(performance.get('throughput', 0))
            results['batch_time'].append(performance.get('avg_batch_time', 0) * 1000)  # 转换为毫秒
            results['params'].append(params_info.get('nonzero_params', 0))
            results['sparsity'].append(params_info.get('sparsity', 0) * 100)  # 转换为百分比
            
            logger.info(f"模型 {friendly_name} 评估完成 - "
                        f"RMSE: {performance.get('rmse', 0):.4f}, "
                        f"吞吐量: {performance.get('throughput', 0):.2f} 样本/秒, "
                        f"批处理时间: {performance.get('avg_batch_time', 0)*1000:.2f} ms")
            
        except Exception as e:
            logger.error(f"评估模型 {model_name} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results

def get_model_params_info(model):
    """
    获取模型参数信息
    
    参数:
    - model: 模型
    
    返回:
    - info: 参数信息字典
    """
    try:
        # 计算总参数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 计算非零参数量
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
        
        # 计算稀疏度
        sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0
        
        return {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity
        }
    except Exception as e:
        logger.error(f"获取模型参数信息失败: {str(e)}")
        return {
            'total_params': 0,
            'nonzero_params': 0,
            'sparsity': 0
        }

def create_comparison_charts(data, output_dir):
    """
    创建模型性能对比图表
    
    参数:
    - data: 性能数据
    - output_dir: 输出目录
    
    返回:
    - chart_path: 图表路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('模型性能对比', fontsize=18)
    
    # 柱状图颜色
    colors = ['#4472C4', '#70AD47', '#FFC000', '#FF6347']
    colors = colors[:len(data['models'])]  # 确保颜色和模型数量匹配
    
    # 1. RMSE对比（越低越好）
    ax = axes[0, 0]
    bars = ax.bar(data['models'], data['rmse'], color=colors)
    ax.set_title('预测准确度 (RMSE, 越低越好)')
    ax.set_ylabel('RMSE')
    ax.set_ylim(0, max(data['rmse']) * 1.2 or 1.0)  # 防止最大值为0
    
    # 添加与原始模型的变化百分比标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
        
        # 添加变化百分比（对于非原始模型）
        if i > 0 and data['rmse'][0] > 0:  # 确保不除以零
            change = (height / data['rmse'][0] - 1) * 100
            color = 'red' if change > 0 else 'green'  # RMSE增加是负面的，用红色表示
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.9,
                   f'{"+" if change > 0 else ""}{change:.2f}%',
                   ha='center', va='center', color=color, fontsize=9)
    
    # 2. 推理吞吐量对比（越高越好）
    ax = axes[0, 1]
    bars = ax.bar(data['models'], data['throughput'], color=colors)
    ax.set_title('推理吞吐量 (样本/秒, 越高越好)')
    ax.set_ylabel('样本/秒')
    
    # 添加数值标签和变化百分比
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
        
        # 添加变化百分比（对于非原始模型）
        if i > 0 and data['throughput'][0] > 0:
            change = (height / data['throughput'][0] - 1) * 100
            color = 'green' if change > 0 else 'red'  # 吞吐量增加是正面的，用绿色表示
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.9,
                   f'{"+" if change > 0 else ""}{change:.1f}%',
                   ha='center', va='center', color=color, fontsize=9)
    
    # 3. 批处理时间对比（越低越好）
    ax = axes[1, 0]
    bars = ax.bar(data['models'], data['batch_time'], color=colors)
    ax.set_title('批处理时间 (毫秒, 越低越好)')
    ax.set_ylabel('毫秒')
    
    # 添加数值标签和变化百分比
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom')
        
        # 添加变化百分比（对于非原始模型）
        if i > 0 and data['batch_time'][0] > 0:
            change = (data['batch_time'][0] / height - 1) * 100  # 注意这里是反向的
            color = 'green' if change > 0 else 'red'  # 批处理时间减少是正面的，用绿色表示
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.7,
                   f'{"+" if change > 0 else ""}{change:.1f}%',
                   ha='center', va='center', color=color, fontsize=9)
    
    # 4. 参数量和稀疏度对比
    ax = axes[1, 1]
    
    # 主Y轴：非零参数数量
    bars1 = ax.bar(data['models'], data['params'], color=colors)
    ax.set_title('模型参数数量与稀疏度')
    ax.set_ylabel('非零参数数量')
    
    # 添加数值标签和变化百分比
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
        
        # 添加变化百分比（对于非原始模型）
        if i > 0 and data['params'][0] > 0:
            change = (height / data['params'][0] - 1) * 100
            param_reduction = -change  # 参数减少率
            color = 'green' if param_reduction > 0 else 'red'  # 参数减少是正面的，用绿色表示
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.8,
                   f'-{param_reduction:.1f}%',
                   ha='center', va='center', color=color, fontsize=9)
    
    # 次Y轴：稀疏度百分比
    ax2 = ax.twinx()
    ax2.plot(data['models'], data['sparsity'], 'ro-', linewidth=2, markersize=8)
    ax2.set_ylabel('稀疏度 (%)')
    
    # 添加稀疏度数值标签
    for i, v in enumerate(data['sparsity']):
        if v > 0:  # 只为非零值添加标签
            ax2.text(i, v, f'{v:.2f}%', ha='center', va='bottom', color='red')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = os.path.join(output_dir, f'model_comparison_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"性能对比图表已保存到: {chart_path}")
    
    # 保存数据为JSON
    json_path = os.path.join(output_dir, f'model_comparison_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"性能对比数据已保存到: {json_path}")
    
    return chart_path

def main():
    """
    主函数
    """
    args = setup_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载所有模型
    models, device = load_models(args)
    
    # 检查是否至少加载了一个模型
    if not models:
        logger.error("没有成功加载任何模型，无法进行比较")
        return
    
    # 比较所有模型的性能
    results = compare_all_models(
        models=models,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # 创建性能对比图表
    if results:
        chart_path = create_comparison_charts(results, args.output_dir)
        logger.info(f"所有模型性能对比已完成，结果保存在: {args.output_dir}")
    else:
        logger.error("性能比较失败，无法创建图表")

if __name__ == "__main__":
    main()