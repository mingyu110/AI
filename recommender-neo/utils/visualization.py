"""
可视化工具模块

此模块包含生成各种性能图表的函数。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_performance_charts(data, output_dir):
    """
    创建模型性能对比图表
    
    参数:
    - data: 性能数据字典，包含models, rmse, throughput, batch_time, params, sparsity等键
    - output_dir: 输出目录
    
    返回:
    - chart_path: 图表保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('模型性能对比', fontsize=16)
    
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
                f'{height:.3f}',
                ha='center', va='bottom')
        
        # 添加变化百分比（对于非原始模型）
        if i > 0 and len(data['rmse']) > 0 and data['rmse'][0] > 0:
            change = (height / data['rmse'][0] - 1) * 100
            color = 'red' if change > 0 else 'green'  # RMSE增加是负面的，用红色表示
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.9,
                   f'{"+" if change > 0 else ""}{change:.1f}%',
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
        if i > 0 and len(data['throughput']) > 0 and data['throughput'][0] > 0:
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
        if i > 0 and len(data['batch_time']) > 0 and data['batch_time'][0] > 0:
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
        if i > 0 and len(data['params']) > 0 and data['params'][0] > 0:
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
    chart_path = os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据为JSON
    json_path = os.path.join(output_dir, f'performance_data_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return chart_path

def create_comparison_radar_chart(data, output_dir):
    """
    创建模型性能雷达图
    
    参数:
    - data: 性能数据字典
    - output_dir: 输出目录
    
    返回:
    - chart_path: 图表保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备雷达图数据
    categories = ['预测准确度', '推理吞吐量', '参数量减少', '省内存空间']
    
    # 标准化数据 (0-1之间)
    norm_data = []
    
    # 处理RMSE (越低越好，需要反转) - 转为准确度
    if 'rmse' in data and len(data['rmse']) > 0 and max(data['rmse']) > 0:
        max_rmse = max(data['rmse'])
        accuracy = [1 - (rmse / max_rmse) for rmse in data['rmse']]
    else:
        accuracy = [0] * len(data['models'])
    
    # 处理吞吐量 (越高越好)
    if 'throughput' in data and len(data['throughput']) > 0 and max(data['throughput']) > 0:
        max_throughput = max(data['throughput'])
        norm_throughput = [t / max_throughput for t in data['throughput']]
    else:
        norm_throughput = [0] * len(data['models'])
    
    # 处理参数量 (越少越好，需要反转)
    if 'params' in data and len(data['params']) > 0 and max(data['params']) > 0:
        max_params = max(data['params'])
        # 原始模型参数量最大，所以反转后得分最低
        param_reduction = [1 - (p / max_params) for p in data['params']]
    else:
        param_reduction = [0] * len(data['models'])
    
    # 使用稀疏度作为省内存指标
    if 'sparsity' in data:
        memory_saving = [s / 100 for s in data['sparsity']]  # 稀疏度已经是百分比形式
    else:
        memory_saving = [0] * len(data['models'])
    
    # 组合所有指标
    for i in range(len(data['models'])):
        norm_data.append([
            accuracy[i],
            norm_throughput[i],
            param_reduction[i],
            memory_saving[i]
        ])
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 添加各个指标的标签
    ax.set_theta_offset(np.pi / 2)  # 从上方开始
    ax.set_theta_direction(-1)  # 顺时针方向
    
    plt.xticks(angles[:-1], categories)
    
    # 设置y轴刻度
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 5)])
    
    # 绘制每个模型的雷达图
    colors = ['#4472C4', '#70AD47', '#FFC000', '#FF6347']
    markers = ['o', 's', '^', 'D']
    
    for i, (model_name, values) in enumerate(zip(data['models'], norm_data)):
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, markersize=8, 
                color=colors[i % len(colors)], marker=markers[i % len(markers)],
                label=model_name)
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('模型性能雷达图', fontsize=15)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    radar_path = os.path.join(output_dir, f'radar_chart_{timestamp}.png')
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return radar_path

def create_improvement_chart(data, output_dir):
    """
    创建相对改进百分比图表
    
    参数:
    - data: 性能数据字典
    - output_dir: 输出目录
    
    返回:
    - chart_path: 图表保存路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据 - 计算相对于原始模型的性能变化百分比
    throughput_improvement = []
    batch_time_improvement = []
    rmse_change = []
    param_reduction = []
    
    if len(data['models']) <= 1:
        return None  # 只有一个模型，无法比较
    
    # 计算各项指标的改进百分比
    for i in range(1, len(data['models'])):
        # 吞吐量 (越高越好)
        if data['throughput'][0] > 0:
            throughput_improvement.append((data['throughput'][i] / data['throughput'][0] - 1) * 100)
        else:
            throughput_improvement.append(0)
        
        # 批处理时间 (越低越好)
        if data['batch_time'][0] > 0:
            batch_time_improvement.append((1 - data['batch_time'][i] / data['batch_time'][0]) * 100)
        else:
            batch_time_improvement.append(0)
        
        # RMSE (越低越好)
        if data['rmse'][0] > 0:
            rmse_change.append((1 - data['rmse'][i] / data['rmse'][0]) * 100)
        else:
            rmse_change.append(0)
        
        # 参数量 (越少越好)
        if data['params'][0] > 0:
            param_reduction.append((1 - data['params'][i] / data['params'][0]) * 100)
        else:
            param_reduction.append(0)
    
    # 组织数据
    models = data['models'][1:]  # 排除原始模型
    metrics = ['预测准确度改进', '吞吐量提升', '批处理时间减少', '参数量减少']
    
    improvements = np.array([
        rmse_change,
        throughput_improvement,
        batch_time_improvement,
        param_reduction
    ])
    
    # 创建横向条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_height = 0.2
    positions = np.arange(len(metrics))
    
    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * bar_height
        bars = ax.barh(positions + offset, improvements[:, i], height=bar_height, label=model)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            label_x = max(width + 1, 1) if width > 0 else min(width - 7, -7)
            color = 'black' if width > 0 else 'white'
            ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}%', ha='center', va='center', color=color, fontsize=9)
    
    # 添加垂直线表示0%位置
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 设置坐标轴
    ax.set_yticks(positions)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('相对于原始模型的改进百分比 (%)')
    
    # 设置图例
    ax.legend(loc='lower right')
    
    plt.title('各模型相对于原始模型的性能改进')
    plt.tight_layout()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    imp_path = os.path.join(output_dir, f'improvements_{timestamp}.png')
    plt.savefig(imp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return imp_path