import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RecSysDataset(Dataset):
    """
    Recommender System Dataset class
    """
    def __init__(self, dataframe):
        self.users = dataframe['user_id'].values
        self.items = dataframe['item_id'].values
        self.ratings = dataframe['rating'].values if 'rating' in dataframe.columns else None
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        
        # Convert to tensor
        user_tensor = torch.tensor(user, dtype=torch.long)
        item_tensor = torch.tensor(item, dtype=torch.long)
        
        if self.ratings is not None:
            rating = self.ratings[idx]
            rating_tensor = torch.tensor(rating, dtype=torch.float)
            return {
                'user': user_tensor,
                'item': item_tensor,
                'rating': rating_tensor
            }
        else:
            return {
                'user': user_tensor,
                'item': item_tensor
            }

class NCFDataset(Dataset):
    """
    Neural Collaborative Filtering Dataset
    
    支持生成负样本用于NCF训练
    """
    def __init__(self, interactions, num_users, num_items, negative_samples=4, is_training=True):
        """
        初始化NCF数据集
        
        参数:
        - interactions: 用户-物品交互列表，格式为 [(user_id, item_id, rating), ...]
        - num_users: 用户数量
        - num_items: 物品数量
        - negative_samples: 每个正样本生成的负样本数量
        - is_training: 是否为训练模式
        """
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.negative_samples = negative_samples
        self.is_training = is_training
        
        # 构建用户-物品交互矩阵（稀疏表示）
        self.user_item_matrix = {}
        for user_id, item_id, _ in self.interactions:
            if user_id not in self.user_item_matrix:
                self.user_item_matrix[user_id] = set()
            self.user_item_matrix[user_id].add(item_id)
        
        # 为训练模式预生成部分负样本
        if self.is_training:
            self._generate_train_samples()
            
    def _generate_train_samples(self):
        """生成训练样本（正样本+负样本）"""
        self.train_samples = []
        
        # 对每个正样本生成负样本
        for user_id, item_id, rating in self.interactions:
            # 添加正样本
            self.train_samples.append((user_id, item_id, 1.0))  # 正样本标签为1
            
            # 生成指定数量的负样本
            neg_samples_added = 0
            neg_attempts = 0
            
            while neg_samples_added < self.negative_samples and neg_attempts < 100:
                # 随机选择一个物品作为负样本
                neg_item = np.random.randint(0, self.num_items)
                
                # 确保负样本不在用户的交互列表中
                if user_id in self.user_item_matrix and neg_item in self.user_item_matrix[user_id]:
                    neg_attempts += 1
                    continue
                
                # 添加负样本
                self.train_samples.append((user_id, neg_item, 0.0))  # 负样本标签为0
                neg_samples_added += 1
    
    def __len__(self):
        if self.is_training:
            return len(self.train_samples)
        else:
            return len(self.interactions)
    
    def __getitem__(self, idx):
        if self.is_training:
            user_id, item_id, label = self.train_samples[idx]
        else:
            user_id, item_id, rating = self.interactions[idx]
            # 针对测试集，使用原始评分作为标签
            label = float(rating > 0)  # 二分类标签
        
        return {
            'user': torch.tensor(user_id, dtype=torch.long),
            'item': torch.tensor(item_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

def create_train_test_dataloader(train_df, test_df, batch_size=128, num_workers=4):
    """
    Create train and test data loaders
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        batch_size: Batch size
        num_workers: Number of workers for dataloader
        
    Returns:
        train_loader: Training data loader
        test_loader: Testing data loader
    """
    # Create datasets
    train_dataset = RecSysDataset(train_df)
    test_dataset = RecSysDataset(test_df)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader 