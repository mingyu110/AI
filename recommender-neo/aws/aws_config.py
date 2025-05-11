"""
AWS配置文件，包含SageMaker和S3相关的配置
"""

# S3配置
S3_CONFIG = {
    'bucket': 'mingyu110',  # 您的S3存储桶名称
    'data_prefix': 'data',  # 数据集在S3存储桶中的前缀路径
    'model_prefix': 'models',  # 模型在S3存储桶中的前缀路径
    'region': 'us-east-1'  # AWS区域，需与您的凭证匹配
}

# SageMaker配置
SAGEMAKER_CONFIG = {
    'role_arn': 'arn:aws:iam::933505494323:role/SageMakerExecutionRole',  # 您的SageMaker执行角色ARN
    'instance_type': 'ml.c5.2xlarge',  # 实例类型
    'framework': 'PYTORCH',  # 模型框架
    'framework_version': '1.8',  # 框架版本
    'output_path': 's3://{}/sagemaker-output'.format(S3_CONFIG['bucket'])  # SageMaker输出路径
}

# 编译配置 - 针对GPU优化
COMPILATION_CONFIG = {
    'target_instance_family': 'ml_p3',  # 目标实例系列 (GPU实例)
    'target_platform_os': 'LINUX',  # 目标平台操作系统
    'target_platform_arch': 'X86_64',  # 目标平台架构
    'target_platform_accelerator': 'NVIDIA',  # 目标平台加速器
    'compiler_options': {
        "dtype": "float32",
        "cuda-ver": "10.2",             # CUDA版本
        "trt-ver": "7.1.3",             # TensorRT版本
        "gpu-code": "sm_70"             # GPU代码，Tesla V100适用
    }
}

def validate_config():
    """验证AWS配置"""
    import boto3
    import botocore
    
    try:
        # 验证S3存储桶
        s3 = boto3.client('s3', region_name=S3_CONFIG['region'])
        s3.head_bucket(Bucket=S3_CONFIG['bucket'])
        print(f"S3存储桶 {S3_CONFIG['bucket']} 验证成功")
        
        # 验证IAM角色
        if 'ACCOUNT_ID' in SAGEMAKER_CONFIG['role_arn']:
            print("警告: 请更新SAGEMAKER_CONFIG['role_arn']为您的SageMaker执行角色ARN")
        else:
            iam = boto3.client('iam')
            role_name = SAGEMAKER_CONFIG['role_arn'].split('/')[-1]
            iam.get_role(RoleName=role_name)
            print(f"IAM角色 {role_name} 验证成功")
            
        return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'NoSuchBucket':
            print(f"错误: S3存储桶 {S3_CONFIG['bucket']} 不存在，请创建存储桶或更新配置")
        elif error_code == 'NoSuchEntity':
            print(f"错误: IAM角色不存在，请检查角色ARN: {SAGEMAKER_CONFIG['role_arn']}")
        else:
            print(f"AWS配置验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 验证配置
    validate_config() 