import numpy as np
import pandas as pd
import os

def generate_mock_data(num_users=1000, num_items=2000, interactions_per_user=20, 
                       train_ratio=0.8, seed=42):
    """
    生成模拟的用户-物品交互数据
    
    参数:
    - num_users: 用户数量
    - num_items: 物品数量
    - interactions_per_user: 每位用户的平均交互次数
    - train_ratio: 训练集占比
    - seed: 随机种子
    
    返回:
    - train_data: 训练数据集
    - test_data: 测试数据集
    """
    np.random.seed(seed)
    
    # 生成用户的偏好向量
    user_factors = np.random.normal(0, 1, (num_users, 20))
    
    # 生成物品的特征向量
    item_factors = np.random.normal(0, 1, (num_items, 20))
    
    # 生成交互数据
    interactions = []
    
    for user_id in range(num_users):
        # 计算该用户与所有物品的相似度得分
        scores = np.dot(user_factors[user_id], item_factors.T)
        
        # 增加一些随机性
        scores += np.random.normal(0, 0.1, size=len(scores))
        
        # 选择得分最高的物品作为交互物品
        top_items = np.argsort(-scores)[:interactions_per_user]
        
        for item_id in top_items:
            # 生成更多样化的评分(1-5)，确保分布更合理
            # 根据相似度得分的高低分配不同概率的评分
            score_normalized = (scores[item_id] - scores[top_items].min()) / (scores[top_items].max() - scores[top_items].min())
            
            # 使用Beta分布生成更自然的评分分布
            if score_normalized > 0.8:  # 最高相似度的物品
                rating_float = np.random.beta(4, 1) * 4 + 1  # 偏向4-5分
            elif score_normalized > 0.6:
                rating_float = np.random.beta(3, 2) * 4 + 1  # 偏向3-4分
            elif score_normalized > 0.4:
                rating_float = np.random.beta(2, 2) * 4 + 1  # 中间评分
            elif score_normalized > 0.2:
                rating_float = np.random.beta(2, 3) * 4 + 1  # 偏向2-3分
            else:
                rating_float = np.random.beta(1, 4) * 4 + 1  # 偏向1-2分
            
            # 转换为1-5的整数评分
            rating = int(round(rating_float))
            
            # 确保在1-5的范围内
            rating = max(1, min(5, rating))
            
            # 生成时间戳 (最近30天内)
            timestamp = int(pd.Timestamp.now().timestamp() - np.random.randint(0, 30*24*3600))
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(interactions)
    
    # 确保没有NaN值
    if df['rating'].isna().any():
        print(f"警告: 生成的数据中存在{df['rating'].isna().sum()}个NaN评分，将使用随机评分替换")
        # 使用1-5的随机整数填充
        df.loc[df['rating'].isna(), 'rating'] = np.random.randint(1, 6, size=df['rating'].isna().sum())
    
    # 随机打乱
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 划分训练集和测试集
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # 再次确认没有NaN值
    assert not train_data.isna().any().any(), "训练数据中存在NaN值"
    assert not test_data.isna().any().any(), "测试数据中存在NaN值"
    
    # 打印评分分布
    ratings_count = df['rating'].value_counts().sort_index()
    print(f"成功生成数据: 用户数={num_users}, 物品数={num_items}")
    print(f"评分分布: {ratings_count.to_dict()}")
    print(f"评分范围: {train_data['rating'].min()}-{train_data['rating'].max()}")
    
    return train_data, test_data

def save_data(train_data, test_data, data_dir='./data'):
    """保存生成的数据"""
    os.makedirs(data_dir, exist_ok=True)
    train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    
    print(f"数据已保存到 {data_dir} 目录")
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"评分统计: 最小值={train_data['rating'].min()}, 最大值={train_data['rating'].max()}, 平均值={train_data['rating'].mean():.2f}")

if __name__ == "__main__":
    train_data, test_data = generate_mock_data()
    save_data(train_data, test_data, data_dir='./data/raw') 