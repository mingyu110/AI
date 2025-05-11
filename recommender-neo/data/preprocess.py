import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_data(data_dir='./data/raw'):
    """加载数据"""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from .mock_data import generate_mock_data, save_data
        train_data, test_data = generate_mock_data()
        save_data(train_data, test_data, data_dir)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 检查NaN值
    train_nan = train_df.isna().sum().sum()
    test_nan = test_df.isna().sum().sum()
    
    if train_nan > 0 or test_nan > 0:
        print(f"警告: 发现NaN值 - 训练集: {train_nan}, 测试集: {test_nan}")
        
        # 处理rating列中的NaN值
        if train_df['rating'].isna().any() or test_df['rating'].isna().any():
            print("发现rating列中的NaN值，使用平均值填充")
            
            # 检查是否所有评分都是NaN
            if train_df['rating'].isna().all() and test_df['rating'].isna().all():
                # 如果所有评分都是NaN，使用默认值3.0
                mean_rating = 3.0
                print(f"所有评分都是NaN，使用默认评分: {mean_rating}")
            else:
                # 计算非NaN值的平均值
                all_ratings = pd.concat([
                    train_df['rating'][~train_df['rating'].isna()], 
                    test_df['rating'][~test_df['rating'].isna()]
                ])
                mean_rating = all_ratings.mean()
                print(f"计算得到的平均评分: {mean_rating}")
            
            # 填充NaN值
            train_df['rating'] = train_df['rating'].fillna(mean_rating)
            test_df['rating'] = test_df['rating'].fillna(mean_rating)
    
    return train_df, test_df

def normalize_ratings(train_df, test_df, scale=(0, 1)):
    """将评分标准化到指定范围"""
    # 确保没有NaN值
    if train_df['rating'].isna().any() or test_df['rating'].isna().any():
        raise ValueError("评分数据中含有NaN值，请在归一化前先处理NaN值")
    
    min_rating = min(train_df['rating'].min(), test_df['rating'].min())
    max_rating = max(train_df['rating'].max(), test_df['rating'].max())
    
    scale_min, scale_max = scale
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['rating'] = scale_min + (train_df['rating'] - min_rating) * (scale_max - scale_min) / (max_rating - min_rating)
    test_df['rating'] = scale_min + (test_df['rating'] - min_rating) * (scale_max - scale_min) / (max_rating - min_rating)
    
    return train_df, test_df

def filter_sparse_interactions(train_df, test_df, min_user_interactions=5, min_item_interactions=10):
    """过滤掉交互较少的用户和物品"""
    # 确保没有NaN值
    if train_df.isna().any().any() or test_df.isna().any().any():
        raise ValueError("数据中含有NaN值，请在过滤前先处理NaN值")
    
    # 统计每个用户和物品的交互次数
    user_counts = pd.concat([train_df['user_id'], test_df['user_id']]).value_counts()
    item_counts = pd.concat([train_df['item_id'], test_df['item_id']]).value_counts()
    
    # 获取满足条件的用户和物品
    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index
    
    # 过滤数据
    train_filtered = train_df[
        train_df['user_id'].isin(valid_users) & 
        train_df['item_id'].isin(valid_items)
    ].copy()  # 使用copy()避免SettingWithCopyWarning
    
    test_filtered = test_df[
        test_df['user_id'].isin(valid_users) & 
        test_df['item_id'].isin(valid_items)
    ].copy()  # 使用copy()避免SettingWithCopyWarning
    
    # 重新映射用户ID和物品ID，使其连续
    user_map = {u: i for i, u in enumerate(valid_users)}
    item_map = {i: j for j, i in enumerate(valid_items)}
    
    train_filtered['user_id'] = train_filtered['user_id'].map(user_map)
    train_filtered['item_id'] = train_filtered['item_id'].map(item_map)
    
    test_filtered['user_id'] = test_filtered['user_id'].map(user_map)
    test_filtered['item_id'] = test_filtered['item_id'].map(item_map)
    
    return train_filtered, test_filtered

def preprocess_data(data_dir='./data/raw', test_size=0.2, min_interactions=5, enhance_diversity=True):
    """
    预处理数据集，支持真实数据集或生成合成数据集
    
    参数:
    - data_dir: 数据目录
    - test_size: 测试集比例
    - min_interactions: 用户或物品的最小交互次数
    - enhance_diversity: 是否增强数据多样性
    
    返回:
    - train_df: 训练数据集
    - test_df: 测试数据集
    """
    # 检查data_dir是否存在，如果不存在则创建
    os.makedirs(data_dir, exist_ok=True)
    
    # 检查MovieLens数据集是否存在
    movielens_file = os.path.join(data_dir, 'ratings.csv')
    if os.path.exists(movielens_file):
        print(f"使用现有MovieLens数据集: {movielens_file}")
        
        # 读取MovieLens数据
        df = pd.read_csv(movielens_file)
        
        # 重命名列
        if all(col in df.columns for col in ['userId', 'movieId', 'rating']):
            df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
            
        # 确保有必要的列
        if not all(col in df.columns for col in ['user_id', 'item_id', 'rating']):
            print("警告: 数据集缺少必要的列，将生成合成数据集")
            return generate_synthetic_data(data_dir, test_size)
        
        print(f"原始数据集统计:\n用户数: {df['user_id'].nunique()}, 物品数: {df['item_id'].nunique()}, 交互数: {len(df)}")
        
        # 数据清洗
        df = clean_dataset(df, min_interactions)
        
        # 增强数据多样性
        if enhance_diversity:
            df = enhance_dataset_diversity(df)
            
        # 划分训练集和测试集
        return split_train_test(df, test_size)
    else:
        # 检查自定义评分数据
        ratings_file = os.path.join(data_dir, 'user_item_ratings.csv')
        if os.path.exists(ratings_file):
            print(f"使用现有评分数据: {ratings_file}")
            df = pd.read_csv(ratings_file)
            
            # 数据清洗
            df = clean_dataset(df, min_interactions)
            
            # 增强数据多样性
            if enhance_diversity:
                df = enhance_dataset_diversity(df)
                
            # 划分训练集和测试集
            return split_train_test(df, test_size)
        else:
            print("未找到现有数据集，生成合成数据集")
            return generate_synthetic_data(data_dir, test_size, enhance_diversity)

def clean_dataset(df, min_interactions=5):
    """
    清洗数据集，移除交互过少的用户和物品
    
    参数:
    - df: 数据集
    - min_interactions: 最小交互次数
    
    返回:
    - df: 清洗后的数据集
    """
    print("进行数据清洗...")
    
    # 确保列类型正确
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    
    if 'rating' in df.columns:
        # 确保评分在合理范围内
        if df['rating'].max() > 10 or df['rating'].min() < 0:
            print("警告: 评分范围异常，进行规范化处理")
            df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min()) * 5
    
    # 移除重复交互
    df_size_before = len(df)
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    if df_size_before > len(df):
        print(f"移除了 {df_size_before - len(df)} 条重复交互")
    
    # 统计每个用户和物品的交互次数
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    
    # 筛选交互频次达到要求的用户和物品
    active_users = user_counts[user_counts >= min_interactions].index
    popular_items = item_counts[item_counts >= min_interactions].index
    
    # 移除交互过少的用户和物品
    df_filtered = df[df['user_id'].isin(active_users) & df['item_id'].isin(popular_items)]
    
    # 输出过滤结果
    users_removed = df['user_id'].nunique() - df_filtered['user_id'].nunique()
    items_removed = df['item_id'].nunique() - df_filtered['item_id'].nunique()
    interactions_removed = len(df) - len(df_filtered)
    
    print(f"移除了 {users_removed} 个交互少于 {min_interactions} 次的用户")
    print(f"移除了 {items_removed} 个交互少于 {min_interactions} 次的物品")
    print(f"移除了总计 {interactions_removed} 条交互记录")
    
    # 重新映射用户ID和物品ID为连续整数
    user_map = {old_id: new_id for new_id, old_id in enumerate(df_filtered['user_id'].unique())}
    item_map = {old_id: new_id for new_id, old_id in enumerate(df_filtered['item_id'].unique())}
    
    df_filtered['user_id'] = df_filtered['user_id'].map(user_map)
    df_filtered['item_id'] = df_filtered['item_id'].map(item_map)
    
    print(f"清洗后数据集:\n用户数: {df_filtered['user_id'].nunique()}, 物品数: {df_filtered['item_id'].nunique()}, 交互数: {len(df_filtered)}")
    
    return df_filtered

def enhance_dataset_diversity(df):
    """
    增强数据集多样性，添加特征
    
    参数:
    - df: 数据集
    
    返回:
    - df: 增强后的数据集
    """
    print("增强数据多样性...")
    
    # 复制原始数据集，以免修改原始数据
    enhanced_df = df.copy()
    
    # 1. 添加上下文特征
    if 'timestamp' not in enhanced_df.columns:
        # 添加伪时间戳
        enhanced_df['timestamp'] = np.random.randint(
            1546300800,  # 2019-01-01
            1609459200,  # 2021-01-01
            size=len(enhanced_df)
        )
    
    # 从时间戳提取时间特征
    if 'timestamp' in enhanced_df.columns:
        enhanced_df['datetime'] = pd.to_datetime(enhanced_df['timestamp'], unit='s')
        enhanced_df['hour'] = enhanced_df['datetime'].dt.hour
        enhanced_df['day_of_week'] = enhanced_df['datetime'].dt.dayofweek
        enhanced_df['weekend'] = enhanced_df['day_of_week'].isin([5, 6]).astype(int)
        enhanced_df['time_of_day'] = pd.cut(
            enhanced_df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # 删除中间列
        enhanced_df.drop(['datetime'], axis=1, inplace=True, errors='ignore')
    
    # 2. 添加物品类别
    num_items = enhanced_df['item_id'].nunique()
    num_categories = min(20, max(5, num_items // 100))  # 根据物品数量决定类别数量
    
    # 为每个物品分配一个类别
    np.random.seed(42)  # 确保可重复
    item_categories = {}
    for item_id in enhanced_df['item_id'].unique():
        item_categories[item_id] = np.random.randint(0, num_categories)
    
    enhanced_df['item_category'] = enhanced_df['item_id'].map(item_categories)
    
    # 3. 添加用户分组
    num_users = enhanced_df['user_id'].nunique()
    num_groups = min(10, max(3, num_users // 200))
    
    # 为每个用户分配一个兴趣组
    user_groups = {}
    for user_id in enhanced_df['user_id'].unique():
        user_groups[user_id] = np.random.randint(0, num_groups)
    
    enhanced_df['user_group'] = enhanced_df['user_id'].map(user_groups)
    
    # 4. 添加交互上下文
    # 访问设备
    devices = ['mobile', 'desktop', 'tablet', 'tv']
    enhanced_df['device'] = np.random.choice(devices, size=len(enhanced_df), p=[0.5, 0.3, 0.15, 0.05])
    
    # 交互地点
    locations = ['home', 'work', 'traveling', 'other']
    enhanced_df['location'] = np.random.choice(locations, size=len(enhanced_df), p=[0.6, 0.25, 0.1, 0.05])
    
    # 5. 对评分稍作调整，使其更符合真实分布
    if 'rating' in enhanced_df.columns:
        # 根据类别和用户组做微调
        for cat in enhanced_df['item_category'].unique():
            # 特定类别的物品评分偏好
            cat_adjustment = np.random.uniform(-0.2, 0.2)
            mask = enhanced_df['item_category'] == cat
            enhanced_df.loc[mask, 'rating'] += cat_adjustment
        
        for group in enhanced_df['user_group'].unique():
            # 特定用户组的评分偏好
            group_adjustment = np.random.uniform(-0.3, 0.3)
            mask = enhanced_df['user_group'] == group
            enhanced_df.loc[mask, 'rating'] += group_adjustment
            
        # 确保评分在合理范围内
        enhanced_df['rating'] = enhanced_df['rating'].clip(0.5, 5.0)
    
    # 输出增强后的特征
    print(f"添加了以下特征: {[col for col in enhanced_df.columns if col not in df.columns]}")
    print(f"增强后数据集大小: {len(enhanced_df)} 行, {enhanced_df.shape[1]} 列")
    
    return enhanced_df

def split_train_test(df, test_size=0.2):
    """
    划分训练集和测试集，确保每个用户在训练集中有足够的交互
    
    参数:
    - df: 数据集
    - test_size: 测试集比例
    
    返回:
    - train_df: 训练集
    - test_df: 测试集
    """
    print(f"划分训练集和测试集，测试集比例: {test_size}")
    
    # 确保每个用户至少有80%的交互记录在训练集中
    user_groups = df.groupby('user_id')
    train_indices = []
    test_indices = []
    
    for user_id, group in user_groups:
        n_items = len(group)
        # 确保每个用户至少有一条记录在测试集中
        if n_items >= 5:
            # 为活跃用户分配更多测试样本
            n_test = max(1, int(n_items * test_size))
        else:
            # 为非活跃用户仅分配一个测试样本
            n_test = 1
            
        indices = group.index.tolist()
        np.random.shuffle(indices)
        
        train_indices.extend(indices[:-n_test])
        test_indices.extend(indices[-n_test:])
    
    # 创建训练集和测试集
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # 确保测试集中的物品在训练集中出现过
    test_items = set(test_df['item_id'].unique())
    train_items = set(train_df['item_id'].unique())
    
    # 找出只在测试集中出现的物品
    test_only_items = test_items - train_items
    
    if test_only_items:
        print(f"发现{len(test_only_items)}个仅在测试集中出现的物品，将其交互移到训练集")
        test_only_mask = test_df['item_id'].isin(test_only_items)
        
        # 将这些交互移到训练集
        train_df = pd.concat([train_df, test_df[test_only_mask]])
        test_df = test_df[~test_only_mask]
    
    # 打印训练集和测试集统计信息
    print(f"训练集: {len(train_df)} 交互, {train_df['user_id'].nunique()} 用户, {train_df['item_id'].nunique()} 物品")
    print(f"测试集: {len(test_df)} 交互, {test_df['user_id'].nunique()} 用户, {test_df['item_id'].nunique()} 物品")
    
    # 验证每个测试集用户在训练集中都有交互
    test_users = set(test_df['user_id'].unique())
    train_users = set(train_df['user_id'].unique())
    if not test_users.issubset(train_users):
        missing_users = test_users - train_users
        print(f"警告: 有 {len(missing_users)} 个用户仅出现在测试集中")
    
    # 保存到CSV文件
    train_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/raw/train.csv'), index=False)
    test_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/raw/test.csv'), index=False)
    
    return train_df, test_df

def generate_synthetic_data(data_dir, test_size=0.2, enhance_diversity=True):
    """
    生成合成数据集
    
    参数:
    - data_dir: 数据目录
    - test_size: 测试集比例
    - enhance_diversity: 是否增强数据多样性
    
    返回:
    - train_df: 训练集
    - test_df: 测试集
    """
    print("生成合成推荐系统数据集...")
    
    # 参数
    n_users = 1000
    n_items = 5000
    n_interactions = 100000  # 稀疏程度约为1%
    
    # 生成用户-物品交互
    np.random.seed(42)
    
    # 生成偏好矩阵 - 使用更真实的分布
    # 用户活跃度服从长尾分布
    user_activity = np.random.exponential(scale=1.0, size=n_users)
    user_activity = user_activity / user_activity.sum()
    
    # 物品流行度服从幂律分布
    item_popularity = np.random.exponential(scale=0.5, size=n_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    # 生成用户-物品交互
    interactions = []
    for _ in range(n_interactions):
        # 基于活跃度抽样用户
        user_id = np.random.choice(n_users, p=user_activity)
        
        # 基于流行度抽样物品
        item_id = np.random.choice(n_items, p=item_popularity)
        
        # 生成评分 - 使用更真实的评分分布
        # 基础评分：偏高的评分分布(反映现实中的正偏差)
        rating_base = np.random.choice([3, 3.5, 4, 4.5, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # 用户偏好：每个用户有一个偏好偏移
        user_bias = np.random.normal(0, 0.5)
        
        # 物品品质：每个物品有一个品质偏移
        item_bias = np.random.normal(0, 0.5)
        
        # 最终评分
        rating = max(0.5, min(5.0, rating_base + user_bias * 0.2 + item_bias * 0.3))
        
        interactions.append((user_id, item_id, rating))
    
    # 创建数据框
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
    
    # 删除重复交互
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    # 增强数据多样性
    if enhance_diversity:
        df = enhance_dataset_diversity(df)
    
    # 划分训练集和测试集
    train_df, test_df = split_train_test(df, test_size)
    
    # 保存到CSV文件
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, 'user_item_ratings.csv'), index=False)
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = preprocess_data()
    print(f"处理后训练集大小: {len(train_df)}")
    print(f"处理后测试集大小: {len(test_df)}")
    print(f"训练集中的NaN值总数: {train_df.isna().sum().sum()}")
    print(f"测试集中的NaN值总数: {test_df.isna().sum().sum()}") 