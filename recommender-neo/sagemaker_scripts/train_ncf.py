import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import json

# 检查是否在SageMaker环境中运行
try:
    # SageMaker环境变量
    SM_MODEL_DIR = os.environ['SM_MODEL_DIR']
    SM_CHANNEL_TRAIN = os.environ['SM_CHANNEL_TRAIN']
    SM_CHANNEL_TEST = os.environ['SM_CHANNEL_TEST']
except KeyError:
    # 本地环境
    print("警告: 未检测到SageMaker环境变量，将使用默认值")
    SM_MODEL_DIR = './output/models'
    SM_CHANNEL_TRAIN = './data/train'
    SM_CHANNEL_TEST = './data/test'

# 解析SageMaker提供的参数
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--embedding-dim', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--early-stopping', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.2)

# SageMaker参数
parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR)
parser.add_argument('--train', type=str, default=SM_CHANNEL_TRAIN)
parser.add_argument('--test', type=str, default=SM_CHANNEL_TEST)

args = parser.parse_args()

# NCF模型定义
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        input_dim = embedding_dim * 2
        
        # MLP层
        for layer_size in mlp_layers:
            self.fc_layers.append(nn.Linear(input_dim, layer_size))
            input_dim = layer_size
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        
    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        
        # 连接embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        
        # 前向传播
        for layer in self.fc_layers:
            vector = layer(vector)
            vector = torch.relu(vector)
            vector = self.dropout(vector)
            
        # 输出层
        prediction = self.output_layer(vector)
        return torch.sigmoid(prediction).squeeze()

# 数据加载函数
def load_data():
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.test, 'test.csv'))
    
    # 获取用户和项目的独特ID
    unique_users = train_df['user_id'].unique()
    unique_items = train_df['item_id'].unique()
    
    # 创建ID映射
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    # 映射IDs
    train_df['user_id_mapped'] = train_df['user_id'].map(user_id_map)
    train_df['item_id_mapped'] = train_df['item_id'].map(item_id_map)
    
    # 对测试集做同样的处理
    test_df['user_id_mapped'] = test_df['user_id'].map(user_id_map)
    test_df['item_id_mapped'] = test_df['item_id'].map(item_id_map)
    
    # 删除映射失败的行
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # 转换为整数ID
    train_df['user_id_mapped'] = train_df['user_id_mapped'].astype(int)
    train_df['item_id_mapped'] = train_df['item_id_mapped'].astype(int)
    test_df['user_id_mapped'] = test_df['user_id_mapped'].astype(int)
    test_df['item_id_mapped'] = test_df['item_id_mapped'].astype(int)
    
    return train_df, test_df, len(unique_users), len(unique_items)

# 创建PyTorch数据集和加载器
class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'user': torch.tensor(row['user_id_mapped'], dtype=torch.long),
            'item': torch.tensor(row['item_id_mapped'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.float)
        }

def get_data_loaders(train_df, test_df, batch_size):
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            rating = batch['rating'].to(device)
            
            outputs = model(user, item)
            predictions.extend(outputs.cpu().tolist())
            targets.extend(rating.cpu().tolist())
    
    # 计算RMSE
    rmse = np.sqrt(np.mean([(p - t)**2 for p, t in zip(predictions, targets)]))
    # 计算MAE
    mae = np.mean([abs(p - t) for p, t in zip(predictions, targets)])
    
    # 简化的hit ratio@10
    hit_count = sum(1 for p, t in zip(predictions, targets) if abs(p - t) < 0.1)
    hr_10 = hit_count / len(targets)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'hr@10': hr_10
    }

# 训练函数
def train_model(train_config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    train_df, test_df, num_users, num_items = load_data()
    train_loader, test_loader = get_data_loaders(train_df, test_df, train_config['batch_size'])
    
    # 创建模型
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=train_config['embedding_dim'],
        mlp_layers=train_config['mlp_layers'],
        dropout=train_config['dropout']
    ).to(device)
    
    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    
    # 训练循环
    best_rmse = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(train_config['epochs']):
        # 训练模式
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
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
            
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        
        # 评估模型
        metrics = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{train_config['epochs']}, Loss: {avg_loss:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        # 早停
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            early_stopping_counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(train_config['model_dir'], 'ncf_best.pth'))
            
            # 保存模型信息
            model_info = {
                'epoch': epoch + 1,
                'embedding_dim': train_config['embedding_dim'],
                'num_users': num_users,
                'num_items': num_items,
                'metrics': metrics,
                'hyperparameters': vars(train_config)
            }
            
            with open(os.path.join(train_config['model_dir'], 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= train_config['early_stopping']:
            print(f"早停触发，在epoch {epoch+1}停止训练")
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(train_config['model_dir'], 'ncf_final.pth'))
    
    # 记录最终指标
    final_metrics = evaluate_model(model, test_loader, device)
    final_info = {
        'final_metrics': final_metrics,
        'num_users': num_users,
        'num_items': num_items,
        'hyperparameters': vars(train_config)
    }
    
    with open(os.path.join(train_config['model_dir'], 'final_metrics.json'), 'w') as f:
        json.dump(final_info, f, indent=2)
    
    print("训练完成")

if __name__ == "__main__":
    train_model()
