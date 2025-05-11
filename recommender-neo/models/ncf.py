import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    """
    改进版神经协同过滤(NCF)模型
    
    简化了模型结构，增加了正则化，优化了参数量
    """
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16], dropout=0.3):
        super(NCF, self).__init__()
        
        # 使用单一嵌入层代替双重嵌入，减少参数量
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 增加嵌入层初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 添加特征交互层
        self.feature_interaction = nn.Sequential(
            nn.Linear(embedding_dim * 2, mlp_layers[0]),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_layers[0]),
            nn.Dropout(dropout)
        )
        
        # MLP层
        self.mlp_layers = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp_layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.BatchNorm1d(mlp_layers[i+1]))
            self.mlp_layers.append(nn.Dropout(dropout))
        
        # 预测层
        self.prediction = nn.Linear(mlp_layers[-1], 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.normal_(self.prediction.weight, std=0.01)
        nn.init.zeros_(self.prediction.bias)
    
    def manual_normalize(self, x, dim=1, eps=1e-8):
        """手动实现normalize操作，避免使用F.normalize"""
        # 计算平方和
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        # 添加eps避免除零 - 使用clamp而非torch.max
        norm = torch.sqrt(square_sum.clamp(min=eps))
        # 归一化
        normalized = x / norm
        return normalized
    
    def forward(self, user_indices, item_indices):
        # 获取嵌入
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # 嵌入正则化 - 使用手动实现的函数代替F.normalize
        user_embed = self.manual_normalize(user_embed, dim=1)
        item_embed = self.manual_normalize(item_embed, dim=1)
        
        # 拼接嵌入
        concat_embed = torch.cat([user_embed, item_embed], dim=1)
        
        # 特征交互
        x = self.feature_interaction(concat_embed)
        
        # MLP层
        for layer in self.mlp_layers:
            x = layer(x)
        
        # 输出层
        rating = self.prediction(x)
        
        return rating.squeeze()
    
    def get_embedding_norm(self):
        """
        获取嵌入层范数，用于监控嵌入质量并帮助剪枝决策
        """
        user_norm = torch.norm(self.user_embedding.weight, p=2, dim=1)
        item_norm = torch.norm(self.item_embedding.weight, p=2, dim=1)
        return {
            'user_embedding_norm': user_norm.detach(),
            'item_embedding_norm': item_norm.detach()
        } 