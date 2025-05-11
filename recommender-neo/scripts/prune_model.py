#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
剪枝NCF模型脚本

此脚本用于对训练好的NCF模型进行剪枝，支持以下功能：
1. 加载预训练模型
2. 应用幅度剪枝或结构化剪枝
3. 评估剪枝后模型性能
4. 保存剪枝后的模型

用法示例:
python -m scripts.prune_model --model_path ./output/models/ncf_model_latest.pt --output_dir ./output/models/pruned --prune_percent 0.5 --prune_type magnitude --fine_tune
"""

import os
import sys
import argparse
import logging
import time
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from models.ncf import NCF
from models.dataloader import NCFDataset
from models.evaluate import evaluate_model
from pruning.magnitude_pruning import apply_magnitude_pruning
from pruning.pruning_utils import count_parameters, analyze_pruned_model
from pruning.fine_tuning import fine_tune_pruned_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("prune-model")

def setup_args():
    """
    设置命令行参数
    """
    parser = argparse.ArgumentParser(description='对NCF模型进行剪枝')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./output/models/pruned',
                        help='剪枝后模型输出目录')
    
    # 剪枝参数
    parser.add_argument('--prune_type', type=str, default='magnitude', choices=['magnitude', 'structured'],
                        help='剪枝类型: magnitude (幅度剪枝) 或 structured (结构化剪枝)')
    parser.add_argument('--prune_percent', type=float, default=0.5,
                        help='剪枝比例 (0.0 - 1.0)')
    parser.add_argument('--exclude_layers', type=str, default='embedding_user.weight,embedding_item.weight',
                        help='排除剪枝的层，逗号分隔')
    
    # 微调参数
    parser.add_argument('--fine_tune', action='store_true',
                        help='是否对剪枝后的模型进行微调')
    parser.add_argument('--ft_epochs', type=int, default=5,
                        help='微调轮数')
    parser.add_argument('--ft_lr', type=float, default=0.0001,
                        help='微调学习率')
    parser.add_argument('--ft_batch_size', type=int, default=128,
                        help='微调批量大小')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='评估批量大小')
    parser.add_argument('--device', type=str, default='',
                        help='运行设备(留空则自动检测)')
    
    return parser.parse_args()

def load_model_and_metadata(model_path, device):
    """
    加载模型和元数据
    
    参数:
    - model_path: 模型文件路径
    - device: 运行设备
    
    返回:
    - model: 加载的模型
    - metadata: 模型元数据
    """
    try:
        logger.info(f"加载模型: {model_path}")
        
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 提取元数据
        metadata = {
            'num_users': checkpoint.get('num_users', 0),
            'num_items': checkpoint.get('num_items', 0),
            'embedding_dim': checkpoint.get('embedding_dim', 64),
            'mlp_layers': checkpoint.get('mlp_layers', [256, 128, 64, 32]),
            'train_metrics': checkpoint.get('train_metrics', {}),
            'test_metrics': checkpoint.get('test_metrics', {}),
            'user_mapping': checkpoint.get('user_mapping', {}),
            'item_mapping': checkpoint.get('item_mapping', {})
        }
        
        # 创建模型实例
        model = NCF(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            embedding_dim=metadata['embedding_dim'],
            mlp_layers=metadata['mlp_layers']
        )
        
        # 加载模型权重
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        # 移动模型到指定设备
        model = model.to(device)
        model.eval()
        
        logger.info(f"模型加载成功，用户数: {metadata['num_users']}, 物品数: {metadata['num_items']}")
        
        return model, metadata
    
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

def create_test_dataloader(interactions, num_users, num_items, batch_size):
    """
    创建测试数据加载器
    
    参数:
    - interactions: 交互数据
    - num_users: 用户数量
    - num_items: 物品数量
    - batch_size: 批量大小
    
    返回:
    - test_loader: 测试数据加载器
    """
    # 创建测试数据集
    test_dataset = NCFDataset(
        interactions=interactions,
        num_users=num_users,
        num_items=num_items,
        negative_samples=4,
        is_training=False
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return test_loader

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
    
    # 加载模型和元数据
    model, metadata = load_model_and_metadata(args.model_path, device)
    
    # 分析原始模型参数
    orig_total_params, orig_trainable_params = count_parameters(model)
    logger.info(f"原始模型总参数: {orig_total_params}, 可训练参数: {orig_trainable_params}")
    
    # 排除的层
    exclude_layers = args.exclude_layers.split(',') if args.exclude_layers else []
    
    # 应用剪枝
    if args.prune_type == 'magnitude':
        logger.info(f"应用幅度剪枝，剪枝比例: {args.prune_percent}")
        pruned_model = apply_magnitude_pruning(
            model=model,
            prune_rate=args.prune_percent
        )
    elif args.prune_type == 'structured':
        logger.info(f"应用结构化剪枝，剪枝比例: {args.prune_percent}")
        from pruning.magnitude_pruning import apply_structured_pruning
        pruned_model = apply_structured_pruning(
            model=model,
            prune_percent=args.prune_percent,
            exclude_layers=exclude_layers
        )
    else:
        logger.error(f"不支持的剪枝类型: {args.prune_type}")
        return
    
    # 分析剪枝后的模型
    pruned_total_params, pruned_trainable_params = count_parameters(pruned_model)
    logger.info(f"剪枝后模型总参数: {pruned_total_params}, 可训练参数: {pruned_trainable_params}")
    
    # 计算压缩率
    compression_ratio = orig_trainable_params / max(1, pruned_trainable_params)
    logger.info(f"模型压缩率: {compression_ratio:.2f}x")
    
    # 详细分析剪枝结果
    layer_stats = analyze_pruned_model(pruned_model)
    for layer_name, stats in layer_stats.items():
        logger.info(f"层 {layer_name}: 非零参数 {stats['nonzero_params']}/{stats['total_params']} "
                    f"({stats['sparsity']*100:.2f}% 稀疏度)")
    
    # 生成测试数据加载器
    test_interactions = None
    try:
        # 尝试从数据目录加载测试数据
        from data.preprocess import load_preprocessed_data
        data_dir = os.path.dirname(args.model_path)
        interactions_path = os.path.join(data_dir, "interactions.pt")
        
        if os.path.exists(interactions_path):
            logger.info(f"从 {interactions_path} 加载交互数据")
            interactions = torch.load(interactions_path)
            test_interactions = interactions.get('test', None)
    except Exception as e:
        logger.warning(f"加载测试数据失败: {str(e)}")
    
    # 如果无法加载真实测试数据，则生成模拟数据
    if test_interactions is None:
        logger.info("生成模拟测试数据")
        import random
        import numpy as np
        
        num_users = metadata['num_users']
        num_items = metadata['num_items']
        
        # 生成100个随机交互
        test_interactions = []
        for _ in range(100):
            user_id = random.randint(0, num_users - 1)
            item_id = random.randint(0, num_items - 1)
            rating = random.random() * 5.0
            test_interactions.append((user_id, item_id, rating))
    
    # 创建测试数据加载器
    test_loader = create_test_dataloader(
        interactions=test_interactions,
        num_users=metadata['num_users'],
        num_items=metadata['num_items'],
        batch_size=args.batch_size
    )
    
    # 评估剪枝后的模型
    logger.info("评估剪枝后的模型性能...")
    pruned_metrics = evaluate_model(
        model=pruned_model,
        test_loader=test_loader,
        device=device
    )
    
    logger.info(f"剪枝后模型指标: {pruned_metrics}")
    
    # 微调模型(如果需要)
    ft_metrics = None
    if args.fine_tune:
        logger.info(f"微调剪枝后的模型，轮数: {args.ft_epochs}，学习率: {args.ft_lr}")
        
        # 创建训练数据加载器
        train_interactions = None
        try:
            # 尝试从数据目录加载训练数据
            from data.preprocess import load_preprocessed_data
            data_dir = os.path.dirname(args.model_path)
            interactions_path = os.path.join(data_dir, "interactions.pt")
            
            if os.path.exists(interactions_path):
                interactions = torch.load(interactions_path)
                train_interactions = interactions.get('train', None)
        except Exception as e:
            logger.warning(f"加载训练数据失败: {str(e)}")
        
        # 如果无法加载真实训练数据，则生成模拟数据
        if train_interactions is None:
            logger.info("生成模拟训练数据")
            import random
            
            num_users = metadata['num_users']
            num_items = metadata['num_items']
            
            # 生成500个随机交互
            train_interactions = []
            for _ in range(500):
                user_id = random.randint(0, num_users - 1)
                item_id = random.randint(0, num_items - 1)
                rating = random.random() * 5.0
                train_interactions.append((user_id, item_id, rating))
        
        # 创建训练数据加载器
        train_dataset = NCFDataset(
            interactions=train_interactions,
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            negative_samples=4
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.ft_batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # 微调模型
        pruned_model_path = os.path.join(args.output_dir, f"pruned_model_temp_{int(time.time())}.pt")
        
        # 保存完整的模型信息，而不仅仅是状态字典
        pruned_info = {
            'state_dict': pruned_model.state_dict(),
            'num_users': metadata['num_users'],
            'num_items': metadata['num_items'],
            'embedding_dim': metadata['embedding_dim'],
            'mlp_layers': metadata['mlp_layers'],
            'config': {
                'embedding_dim': metadata['embedding_dim'],
                'mlp_layers': metadata['mlp_layers'],
                'dropout': 0.3  # 默认值
            },
            'dataset_stats': {
                'num_users': metadata['num_users'],
                'num_items': metadata['num_items']
            }
        }
        torch.save(pruned_info, pruned_model_path)
        
        finetuned_model_path = os.path.join(args.output_dir, f"finetuned_model_temp_{int(time.time())}.pt")
        
        finetuned_model, ft_metrics = fine_tune_pruned_model(
            pruned_model_path=pruned_model_path,
            output_model_path=finetuned_model_path,
            train_loader=train_loader,
            test_loader=test_loader,
            config={
                'lr': args.ft_lr,
                'epochs': args.ft_epochs,
                'early_stopping': 3,
                'model_dir': args.output_dir,
                'save_best': True
            },
            device=device
        )
        
        logger.info(f"微调后模型指标: {ft_metrics}")
        
        # 使用微调后的模型
        pruned_model = finetuned_model
    
    # 保存剪枝后的模型
    timestamp = int(time.time())
    if args.fine_tune:
        model_name = f"ncf_pruned_finetuned_{args.prune_type}_{int(args.prune_percent*100)}percent_{timestamp}.pt"
    else:
        model_name = f"ncf_pruned_{args.prune_type}_{int(args.prune_percent*100)}percent_{timestamp}.pt"
    
    pruned_model_path = os.path.join(args.output_dir, model_name)
    
    # 保存模型和元数据
    torch.save({
        'state_dict': pruned_model.state_dict(),
        'num_users': metadata['num_users'],
        'num_items': metadata['num_items'],
        'embedding_dim': metadata['embedding_dim'],
        'mlp_layers': metadata['mlp_layers'],
        'orig_metrics': metadata['test_metrics'],
        'pruned_metrics': pruned_metrics,
        'ft_metrics': ft_metrics,
        'prune_type': args.prune_type,
        'prune_percent': args.prune_percent,
        'compression_ratio': compression_ratio,
        'layer_stats': layer_stats,
        'user_mapping': metadata['user_mapping'],
        'item_mapping': metadata['item_mapping']
    }, pruned_model_path)
    
    logger.info(f"剪枝后模型保存到: {pruned_model_path}")
    
    # 更新最新剪枝模型链接
    latest_pruned_path = os.path.join(args.output_dir, "ncf_pruned_latest.pt")
    
    if os.path.exists(latest_pruned_path):
        os.remove(latest_pruned_path)
    
    try:
        # 创建符号链接
        os.symlink(os.path.basename(pruned_model_path), latest_pruned_path)
        logger.info(f"创建最新剪枝模型链接: {latest_pruned_path}")
    except Exception as e:
        # 如果符号链接失败，则复制文件
        import shutil
        shutil.copy2(pruned_model_path, latest_pruned_path)
        logger.info(f"复制最新剪枝模型: {latest_pruned_path}")
    
    # 如果进行了微调，也更新最新微调模型链接
    if args.fine_tune:
        latest_finetuned_path = os.path.join(args.output_dir, "ncf_finetuned_latest.pt")
        
        if os.path.exists(latest_finetuned_path):
            os.remove(latest_finetuned_path)
        
        try:
            # 创建符号链接
            os.symlink(os.path.basename(pruned_model_path), latest_finetuned_path)
            logger.info(f"创建最新微调模型链接: {latest_finetuned_path}")
        except Exception as e:
            # 如果符号链接失败，则复制文件
            import shutil
            shutil.copy2(pruned_model_path, latest_finetuned_path)
            logger.info(f"复制最新微调模型: {latest_finetuned_path}")
    
    logger.info("模型剪枝流程完成")

if __name__ == "__main__":
    main() 