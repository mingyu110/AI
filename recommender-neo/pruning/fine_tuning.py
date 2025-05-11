import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ncf import NCF
from models.evaluate import evaluate_model
from data.dataset import get_data_loaders
from pruning.magnitude_pruning import get_model_size

def finetune(model, train_loader, test_loader, config, device):
    """
    对模型进行微调
    
    参数:
    - model: 待微调的模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - config: 微调配置
    - device: 计算设备
    
    返回:
    - model: 微调后的模型
    - metrics: 微调过程中的指标记录
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 记录微调过程
    train_losses = []
    val_metrics = []
    best_rmse = float('inf')
    early_stopping_counter = 0
    
    print(f"开始微调，共 {config['epochs']} 个epoch...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # 训练
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"微调 Epoch {epoch+1}/{config['epochs']}"):
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            rating = batch['rating'].to(device)
            
            # 前向传播
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        metrics = evaluate_model(model, test_loader, device)
        val_metrics.append(metrics)
        
        print(f"Epoch {epoch+1}/{config['epochs']}, "
              f"训练损失: {train_loss:.4f}, "
              f"验证RMSE: {metrics['rmse']:.4f}, "
              f"验证MAE: {metrics['mae']:.4f}, "
              f"命中率@10: {metrics['hr@10']:.4f}")
        
        # 保存最佳模型
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            early_stopping_counter = 0
            
            # 保存模型
            if config.get('save_best', True):
                model_path = os.path.join(config['model_dir'], 'ncf_pruned_finetuned_best.pth')
                torch.save(model.state_dict(), model_path)
                print(f"保存最佳模型到 {model_path}")
        else:
            early_stopping_counter += 1
        
        # 早停
        if early_stopping_counter >= config['early_stopping']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"微调完成，耗时 {training_time:.2f} 秒")
    
    return model, {'train_losses': train_losses, 'val_metrics': val_metrics}

def fine_tune_pruned_model(pruned_model_path, output_model_path, train_loader, test_loader, config=None, device='cpu'):
    """
    微调剪枝后的模型
    
    参数:
    - pruned_model_path: 剪枝后模型路径
    - output_model_path: 微调后模型保存路径
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器
    - config: 微调配置
    - device: 计算设备
    
    返回:
    - fine_tuned_model: 微调后的模型
    - performance: 性能变化指标
    """
    # 默认配置
    if config is None:
        config = {
            'lr': 1e-5,
            'epochs': 5,
            'early_stopping': 3,
            'model_dir': './models/saved',
            'save_best': True
        }
    
    # 加载剪枝后模型权重
    checkpoint = torch.load(pruned_model_path, map_location=device)
    
    # 检查是否有state_dict键
    if 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
        dataset_stats = checkpoint.get('dataset_stats', {})
        model_config = checkpoint.get('config', {})
    else:
        model_state_dict = checkpoint
        dataset_stats = {'num_users': 1000, 'num_items': 2000}
        model_config = {}
    
    # 创建模型实例
    model = NCF(
        num_users=dataset_stats.get('num_users'),
        num_items=dataset_stats.get('num_items'),
        embedding_dim=model_config.get('embedding_dim', 64),
        mlp_layers=model_config.get('mlp_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.2)
    ).to(device)
    
    # 加载剪枝后模型权重
    model.load_state_dict(model_state_dict)
    
    # 测试剪枝后模型性能
    print("测试剪枝后模型性能...")
    pruned_metrics_measured = evaluate_model(model, test_loader, device)
    print(f"剪枝后模型性能: RMSE={pruned_metrics_measured['rmse']:.4f}, HR@10={pruned_metrics_measured.get('hr@10', 0):.4f}")
    
    # 进行微调
    print("开始微调...")
    fine_tuned_model, training_history = finetune(model, train_loader, test_loader, config, device)
    
    # 测试微调后模型性能
    print("测试微调后模型性能...")
    fine_tuned_metrics = evaluate_model(fine_tuned_model, test_loader, device)
    print(f"微调后模型性能: RMSE={fine_tuned_metrics['rmse']:.4f}, HR@10={fine_tuned_metrics.get('hr@10', 0):.4f}")
    
    # 计算性能变化
    performance_diff = {}
    if pruned_metrics_measured:
        for metric in pruned_metrics_measured:
            performance_diff[metric] = fine_tuned_metrics[metric] - pruned_metrics_measured[metric]
    
    # 打印性能变化
    if performance_diff:
        print("\n===== 微调后性能变化 =====")
        for metric, diff in performance_diff.items():
            if metric in ['rmse', 'mae']:
                print(f"{metric.upper()}: {diff:.4f} ({'变差' if diff > 0 else '变好' if diff < 0 else '不变'})")
            elif metric.startswith('hr@') or metric.startswith('ndcg@'):
                print(f"{metric}: {diff:.4f} ({'变好' if diff > 0 else '变差' if diff < 0 else '不变'})")
    
    # 保存微调后的模型
    directory = os.path.dirname(output_model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    torch.save(fine_tuned_model.state_dict(), output_model_path)
    
    # 保存模型信息
    model_size = get_model_size(fine_tuned_model, 'MB')
    fine_tuned_info = {
        'pruned_model': pruned_model_path,
        'fine_tuning_config': config,
        'model_size': model_size,
        'pruned_metrics': pruned_metrics_measured,
        'fine_tuned_metrics': fine_tuned_metrics,
        'performance_diff': performance_diff,
        'training_history': {
            'train_losses': training_history['train_losses'],
            'final_val_metrics': training_history['val_metrics'][-1]
        },
        'dataset_stats': dataset_stats,
        'model_config': model_config
    }
    
    info_path = f"{os.path.splitext(output_model_path)[0]}_info.json"
    with open(info_path, 'w') as f:
        json.dump(fine_tuned_info, f, indent=4)
    
    print(f"微调后模型已保存到 {output_model_path}")
    
    return fine_tuned_model, performance_diff

def sagemaker_finetune_job(pruned_model_path, sagemaker_config):
    """
    使用SageMaker启动微调作业
    
    参数:
    - pruned_model_path: 剪枝后模型路径
    - sagemaker_config: SageMaker配置
    
    返回:
    - job_name: 训练作业名称
    """
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch
    import logging
    
    logger = logging.getLogger("recommender-pipeline")
    
    try:
        logger.info("开始准备SageMaker微调作业...")
        logger.info(f"使用剪枝模型: {pruned_model_path}")
        
        # 检查必要参数
        if not sagemaker_config.get('role_arn'):
            logger.error("错误: 未提供SageMaker IAM角色")
            return None
            
        if not sagemaker_config.get('data_s3_path'):
            logger.warning("警告: 未提供训练数据路径，将使用默认路径")
        
        # 确定实例类型
        instance_type = sagemaker_config.get('instance_type', 'ml.c5.large')
        logger.info(f"微调将使用实例类型: {instance_type}")
        
        # 创建SageMaker会话
        sagemaker_session = sagemaker.Session()
        
        # 上传模型文件到S3
        logger.info("上传剪枝模型到S3...")
        model_data = sagemaker_session.upload_data(
            path=pruned_model_path,
            key_prefix='pruned-model'
        )
        logger.info(f"模型已上传到: {model_data}")
        
        # 配置超参数
        hyperparameters = {
            'model_path': model_data,
            'epochs': sagemaker_config.get('epochs', 5),
            'learning_rate': sagemaker_config.get('learning_rate', 1e-5),
            'batch_size': sagemaker_config.get('batch_size', 64),
            'data_fraction': sagemaker_config.get('data_fraction', 0.1)  # 使用数据的比例
        }
        
        logger.info(f"微调参数: epochs={hyperparameters['epochs']}, lr={hyperparameters['learning_rate']}")
        
        # 创建PyTorch估计器
        logger.info("创建PyTorch估计器...")
        estimator = PyTorch(
            entry_point='finetune.py',
            source_dir='sagemaker_scripts',
            role=sagemaker_config['role_arn'],
            instance_count=1,
            instance_type=instance_type,
            framework_version='1.8.0',
            py_version='py3',
            hyperparameters=hyperparameters
        )
        
        # 启动训练作业
        data_path = sagemaker_config.get('data_s3_path', 's3://bucket/data')
        logger.info(f"使用数据路径: {data_path}")
        logger.info("启动微调作业...")
        
        estimator.fit(
            inputs={'data': data_path},
            wait=False
        )
        
        job_name = estimator.latest_training_job.name
        logger.info(f"微调作业已启动: {job_name}")
        return job_name
        
    except Exception as e:
        logger.error(f"启动SageMaker微调作业失败: {str(e)}")
        return None

if __name__ == "__main__":
    print("示例: 微调剪枝后的NCF模型")
    print("注意: 此脚本需要在运行模型剪枝后执行")
    print("使用方法: python -m pruning.fine_tuning [剪枝模型路径] [输出路径]") 