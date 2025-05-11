"""
SageMaker 微调入口点脚本
用于在 SageMaker 上微调剪枝后的 NCF 模型
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# SageMaker 特定的导入
try:
    import sagemaker_containers
except ImportError:
    # 当在本地环境运行时，sagemaker_containers可能不可用
    print("警告: sagemaker_containers 不可用，这在本地环境中是正常的")
    sagemaker_containers = None

import torch.nn as nn
import torch.optim as optim

# 定义 NCF 模型架构（需要与原始模型一致）
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP 部分
        self.mlp_layers = nn.ModuleList()
        input_dim = 2 * embedding_dim
        
        for i, layer_size in enumerate(mlp_layers):
            self.mlp_layers.append(nn.Linear(input_dim, layer_size))
            self.mlp_layers.append(nn.ReLU())
            if dropout > 0:
                self.mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_size
        
        # 输出层
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        
    def forward(self, user_indices, item_indices):
        user_embeddings = self.user_embedding(user_indices)
        item_embeddings = self.item_embedding(item_indices)
        
        # MLP 部分
        mlp_input = torch.cat([user_embeddings, item_embeddings], dim=-1)
        
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        
        output = self.output_layer(mlp_input)
        return output.squeeze()

# 训练函数
def train(model, train_loader, val_loader, device, args):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                outputs = model(users, items)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1} complete! Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
            print(f"Saved new best model! Val Loss: {val_loss:.6f}")
    
    print(f"Training complete! Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}")

# 加载数据函数
def load_data(data_dir, batch_size, data_fraction=1.0):
    print(f"Loading data from {data_dir}...")
    
    # 在SageMaker上，数据通常由SageMaker挂载到/opt/ml/input/data/
    training_path = os.path.join(data_dir, 'train.csv')
    
    # 加载和预处理数据
    import pandas as pd
    df = pd.read_csv(training_path)
    
    # 如果需要使用部分数据
    if data_fraction < 1.0:
        df = df.sample(frac=data_fraction, random_state=42)
    
    # 数据集大小
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    
    # 训练验证分割
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 转换为Tensor
    train_users = torch.LongTensor(train_df['user_id'].values)
    train_items = torch.LongTensor(train_df['item_id'].values)
    train_ratings = torch.FloatTensor(train_df['rating'].values)
    
    val_users = torch.LongTensor(val_df['user_id'].values)
    val_items = torch.LongTensor(val_df['item_id'].values)
    val_ratings = torch.FloatTensor(val_df['rating'].values)
    
    # 创建DataLoader
    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    val_dataset = TensorDataset(val_users, val_items, val_ratings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, num_users, num_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker参数
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_DATA', '.'))
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='pruned_model.pth')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--mlp_layers', type=str, default='128,64,32')
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_fraction', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # 将mlp_layers转换为列表
    args.mlp_layers = [int(x) for x in args.mlp_layers.split(',')]
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, val_loader, num_users, num_items = load_data(
        args.data_dir, args.batch_size, args.data_fraction)
    
    print(f"Dataset info: {num_users} users, {num_items} items")
    
    # 创建模型
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout
    ).to(device)
    
    # 尝试加载预训练模型权重
    try:
        model_path = os.path.join(args.data_dir, args.model_path)
        if os.path.exists(model_path):
            print(f"Loading pretrained model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No pretrained model found at {model_path}, starting from scratch")
    except Exception as e:
        print(f"Error loading pretrained model: {str(e)}")
    
    # 训练模型
    train(model, train_loader, val_loader, device, args)
    
    # 保存模型信息
    model_info = {
        'num_users': int(num_users),
        'num_items': int(num_items),
        'embedding_dim': args.embedding_dim,
        'mlp_layers': args.mlp_layers,
        'dropout': args.dropout
    }
    
    with open(os.path.join(args.model_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)
        
    print("Finetuning complete! Model saved to:", args.model_dir) 