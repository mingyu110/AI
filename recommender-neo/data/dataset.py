import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class UserItemRatingDataset(Dataset):
    """用户-物品-评分数据集"""
    
    def __init__(self, df):
        self.df = df
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.ratings = df['rating'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        rating = self.ratings[idx]
        
        return {
            'user': torch.tensor(user, dtype=torch.long),
            'item': torch.tensor(item, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

def get_data_loaders(train_df, test_df, batch_size=64):
    """创建训练和测试数据加载器"""
    train_dataset = UserItemRatingDataset(train_df)
    test_dataset = UserItemRatingDataset(test_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

def get_dataset_stats(train_df, test_df):
    """获取数据集的统计信息"""
    num_users = max(train_df['user_id'].max(), test_df['user_id'].max()) + 1
    num_items = max(train_df['item_id'].max(), test_df['item_id'].max()) + 1
    
    stats = {
        'num_users': int(num_users),
        'num_items': int(num_items),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'rating_min': min(train_df['rating'].min(), test_df['rating'].min()),
        'rating_max': max(train_df['rating'].max(), test_df['rating'].max()),
        'sparsity': 1.0 - (len(train_df) + len(test_df)) / (num_users * num_items)
    }
    
    return stats 