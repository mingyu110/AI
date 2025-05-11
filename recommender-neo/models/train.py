import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import preprocess_data
from data.dataset import get_data_loaders, get_dataset_stats
from models.ncf import NCF
from models.evaluate import evaluate_model

logger = logging.getLogger("recommender-pipeline")

# 自定义JSON编码器，处理numpy数据类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            # 将torch.device对象转换为字符串
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def train(model, train_loader, criterion, optimizer, device, epoch_idx=None, total_epochs=None):
    """训练单个epoch"""
    model.train()
    total_loss = 0
    
    # 创建带有epoch信息的进度条
    epoch_info = f"Epoch {epoch_idx}/{total_epochs}" if epoch_idx is not None and total_epochs is not None else "模型训练中"
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    
    with tqdm(train_loader, desc=epoch_info, ncols=100, bar_format=bar_format, position=2, leave=False) as pbar:
        for batch in pbar:
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
            
            batch_loss = loss.item()
            total_loss += batch_loss
            
            # 更新进度条描述，显示当前batch的损失
            pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", refresh=True)
    
    return total_loss / len(train_loader)

def save_model(model, path, model_info=None):
    """保存模型和相关信息"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存模型权重
    torch.save(model.state_dict(), path)
    
    # 如果提供了模型信息，保存为JSON，使用自定义编码器处理numpy数据类型
    if model_info:
        # 处理模型信息中的特殊类型
        processed_info = {}
        for key, value in model_info.items():
            if key == 'metrics' or key == 'dataset_stats' or key == 'final_metrics':
                # 将所有metrics和dataset_stats中的值转换为Python原生类型
                processed_info[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                     for k, v in value.items()}
            elif key == 'config':
                # 处理config中可能的特殊类型
                processed_config = {}
                for cfg_key, cfg_val in value.items():
                    if isinstance(cfg_val, (np.floating, np.integer)):
                        processed_config[cfg_key] = float(cfg_val) if isinstance(cfg_val, np.floating) else int(cfg_val)
                    elif isinstance(cfg_val, (list, tuple)) and all(isinstance(x, (np.floating, np.integer)) for x in cfg_val):
                        processed_config[cfg_key] = [float(x) if isinstance(x, np.floating) else int(x) for x in cfg_val]
                    else:
                        processed_config[cfg_key] = cfg_val
                processed_info[key] = processed_config
            else:
                # 处理其他可能的numpy类型
                if isinstance(value, (np.floating, np.integer)):
                    processed_info[key] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    processed_info[key] = value
        
        # 保存处理后的信息
        info_path = f"{os.path.splitext(path)[0]}_info.json"
        with open(info_path, 'w') as f:
            json.dump(processed_info, f, indent=4, cls=NumpyEncoder)
    
    print(f"模型已保存到 {path}")

def train_model(model, train_loader, test_loader=None, epochs=10, learning_rate=0.001, 
               weight_decay=0.0001, device='cpu', early_stopping=3, use_scheduler=True, 
               use_amp=False):
    """
    训练NCF模型
    
    参数:
    - model: 待训练的模型
    - train_loader: 训练数据加载器
    - test_loader: 测试数据加载器(可选)
    - epochs: 训练轮数
    - learning_rate: 学习率
    - weight_decay: 权重衰减(L2正则化)系数
    - device: 训练设备
    - early_stopping: 早停轮数(0表示不使用早停)
    - use_scheduler: 是否使用学习率调度器
    - use_amp: 是否使用混合精度训练
    
    返回:
    - model: 训练后的模型
    - metrics: 训练指标
    """
    logger.info(f"Starting model training: epochs={epochs}, lr={learning_rate}, device={device}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 早停机制
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 记录训练指标
    metrics = {
        'epoch_losses': [],
        'val_losses': [],
        'training_time': 0,
        'best_epoch': 0
    }
    
    # 训练开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            label = batch['label'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播(使用混合精度)
            if use_amp:
                with autocast():
                    prediction = model(user, item)
                    loss = criterion(prediction, label)
                
                # 反向传播 with 梯度缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 常规训练
                prediction = model(user, item)
                loss = criterion(prediction, label)
                loss.backward()
                optimizer.step()
            
            # 更新累计损失
            epoch_loss += loss.item()
            batch_count += 1
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        metrics['epoch_losses'].append(avg_loss)
        
        # 评估验证集
        if test_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    user = batch['user'].to(device)
                    item = batch['item'].to(device)
                    label = batch['label'].to(device)
                    
                    prediction = model(user, item)
                    loss = criterion(prediction, label)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            metrics['val_losses'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
            
            # 更新学习率
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # 早停检查
            if early_stopping > 0:
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    metrics['best_epoch'] = epoch + 1
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        logger.info(f"Early stopping triggered! No improvement for {early_stopping} epochs")
                        break
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}")
    
    # 计算训练时间
    metrics['training_time'] = time.time() - start_time
    logger.info(f"Training completed! Total time: {metrics['training_time']:.2f} seconds")
    
    # 恢复最佳模型(如果使用早停)
    if early_stopping > 0 and best_model_state:
        logger.info(f"Restoring best model (Epoch {metrics['best_epoch']})")
        model.load_state_dict(best_model_state)
    
    # 返回模型和训练指标
    return model, metrics

if __name__ == "__main__":
    train_model() 