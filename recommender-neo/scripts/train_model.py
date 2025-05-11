#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练NCF模型脚本

此脚本用于训练NCF推荐模型，支持以下功能：
1. 加载和预处理数据
2. 配置和训练NCF模型
3. 评估模型性能
4. 保存训练好的模型

用法示例:
python -m scripts.train_model --data_dir ./data/ml-1m --output_dir ./output/models --batch_size 256 --epochs 20 --factors 64 --layers 256,128,64,32 --lr 0.001 --negative_samples 4
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from models.ncf import NCF
from models.dataloader import NCFDataset
from models.train import train_model
from models.evaluate import evaluate_model
from data.preprocess import preprocess_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("train-model")

def setup_args():
    """
    设置命令行参数
    """
    parser = argparse.ArgumentParser(description='训练NCF推荐模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/ml-1m',
                        help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='./output/models',
                        help='模型输出目录')
    parser.add_argument('--negative_samples', type=int, default=4,
                        help='每个正样本的负样本数量')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集比例')
    
    # 模型参数
    parser.add_argument('--factors', type=int, default=64,
                        help='嵌入向量维度')
    parser.add_argument('--layers', type=str, default='256,128,64,32',
                        help='MLP层结构，逗号分隔')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='',
                        help='训练设备(留空则自动检测)')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = setup_args()
    
    # 设置设备
    device = args.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载并预处理数据
    logger.info(f"加载并预处理数据集: {args.data_dir}")
    train_df, test_df = preprocess_data(
        data_dir=args.data_dir,
        test_size=args.test_ratio,
        min_interactions=5,
        enhance_diversity=True
    )
    
    # 准备数据集
    # 转换训练集和测试集为交互列表格式
    train_data = [(row['user_id'], row['item_id'], row['rating']) 
                  for _, row in train_df.iterrows()]
    test_data = [(row['user_id'], row['item_id'], row['rating']) 
                 for _, row in test_df.iterrows()]
    
    # 创建用户和物品的映射字典
    user_mapping = {user_id: user_id for user_id in train_df['user_id'].unique()}
    item_mapping = {item_id: item_id for item_id in train_df['item_id'].unique()}
    
    interactions = {'train': train_data, 'test': test_data}
    
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    logger.info(f"用户数量: {num_users}, 物品数量: {num_items}")
    logger.info(f"训练集样本: {len(train_data)}, 测试集样本: {len(test_data)}")
    
    # 创建数据集和数据加载器
    train_dataset = NCFDataset(
        interactions=train_data,
        num_users=num_users,
        num_items=num_items,
        negative_samples=args.negative_samples
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    test_dataset = NCFDataset(
        interactions=test_data,
        num_users=num_users,
        num_items=num_items,
        negative_samples=args.negative_samples,
        is_training=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 解析MLP层结构
    mlp_layers = [int(x) for x in args.layers.split(',')]
    
    # 创建模型
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.factors,
        mlp_layers=mlp_layers
    )
    
    # 打印模型结构
    logger.info(f"模型结构:\n{model}")
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {num_params}")
    
    # 移动模型到指定设备
    model = model.to(device)
    
    # 训练模型
    logger.info("开始训练模型...")
    start_time = time.time()
    
    train_losses, train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    train_time = time.time() - start_time
    logger.info(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    # 评估模型
    logger.info("评估模型性能...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    logger.info(f"测试集指标: {test_metrics}")
    
    # 保存模型
    model_path = os.path.join(args.output_dir, f"ncf_model_{int(time.time())}.pt")
    
    torch.save({
        'state_dict': model.state_dict(),
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': args.factors,
        'mlp_layers': mlp_layers,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'user_mapping': user_mapping,
        'item_mapping': item_mapping
    }, model_path)
    
    logger.info(f"模型保存到: {model_path}")
    
    # 还保存一个最新模型的链接
    latest_model_path = os.path.join(args.output_dir, "ncf_model_latest.pt")
    
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    
    try:
        # 创建符号链接
        os.symlink(os.path.basename(model_path), latest_model_path)
        logger.info(f"创建最新模型链接: {latest_model_path}")
    except Exception as e:
        # 如果符号链接失败，则复制文件
        import shutil
        shutil.copy2(model_path, latest_model_path)
        logger.info(f"复制最新模型: {latest_model_path}")
    
    logger.info("模型训练和评估流程完成")

if __name__ == "__main__":
    main() 