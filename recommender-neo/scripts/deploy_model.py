#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
部署模型到AWS SageMaker脚本

此脚本用于将模型部署到AWS SageMaker，支持以下功能：
1. 将训练好的模型编译为Neo优化模型
2. 将Neo优化模型部署到SageMaker端点
3. 测试模型端点

用法示例:
python -m scripts.deploy_model --model_path ./output/models/ncf_model_latest.pt --s3_bucket your-bucket-name --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole
"""

import os
import sys
import argparse
import logging
import time
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块
from aws.neo_compile import optimize_model_with_neo
from aws.sagemaker_deploy import build_and_deploy_with_sagemaker
from aws.aws_utils import validate_role_arn, validate_s3_bucket, check_aws_connectivity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("deploy-model")

def setup_args():
    """
    设置命令行参数
    """
    parser = argparse.ArgumentParser(description='部署模型到AWS SageMaker')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--output_dir', type=str, default='./output/neo',
                        help='Neo输出目录')
    
    # AWS参数
    parser.add_argument('--s3_bucket', type=str, required=True,
                        help='S3存储桶名称')
    parser.add_argument('--s3_prefix', type=str, default='neo',
                        help='S3前缀路径')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS区域')
    parser.add_argument('--role_arn', type=str, required=True,
                        help='SageMaker执行角色ARN')
    
    # 部署参数
    parser.add_argument('--target_device', type=str, default='ml_m5',
                        help='目标设备(ml_c4, ml_c5, ml_m4, ml_m5, ml_p2, ml_p3等)')
    parser.add_argument('--instance_type', type=str, default='ml.m5.large',
                        help='推理实例类型')
    parser.add_argument('--endpoint_name', type=str, default='',
                        help='端点名称(留空则自动生成)')
    parser.add_argument('--skip_compilation', action='store_true',
                        help='跳过Neo编译步骤(使用已有的编译作业)')
    parser.add_argument('--compilation_job_name', type=str, default='',
                        help='已有的Neo编译作业名称(如果跳过编译)')
    
    return parser.parse_args()

def validate_aws_params(args):
    """
    验证AWS参数
    
    参数:
    - args: 命令行参数
    
    返回:
    - is_valid: 是否有效
    """
    # 验证IAM角色ARN
    is_valid_role, role_message = validate_role_arn(args.role_arn)
    if not is_valid_role:
        logger.error(f"IAM角色ARN无效: {role_message}")
        return False
    
    # 验证S3存储桶名称
    is_valid_bucket, bucket_message = validate_s3_bucket(args.s3_bucket)
    if not is_valid_bucket:
        logger.error(f"S3存储桶名称无效: {bucket_message}")
        return False
    
    # 检查AWS连接性
    is_connected, conn_message = check_aws_connectivity(args.region)
    if not is_connected:
        logger.error(f"AWS连接失败: {conn_message}")
        return False
    
    return True

def test_endpoint(endpoint_name, region, user_id=1, item_id=42):
    """
    测试SageMaker端点
    
    参数:
    - endpoint_name: 端点名称
    - region: AWS区域
    - user_id: 用户ID
    - item_id: 物品ID
    
    返回:
    - response: 响应
    """
    import boto3
    
    logger.info(f"测试端点 {endpoint_name} 中...")
    
    # 创建SageMaker运行时客户端
    runtime = boto3.client('runtime.sagemaker', region_name=region)
    
    # 准备请求数据
    request_data = {
        'user_id': user_id,
        'item_id': item_id
    }
    
    # 调用端点
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(request_data)
        )
        
        # 解析响应
        result = json.loads(response['Body'].read().decode())
        logger.info(f"推理成功: 用户ID={user_id}, 物品ID={item_id}, 预测值={result}")
        
        return result
    except Exception as e:
        logger.error(f"调用端点失败: {str(e)}")
        return None

def main():
    """
    主函数
    """
    args = setup_args()
    
    # 验证AWS参数
    if not validate_aws_params(args):
        logger.error("AWS参数验证失败，无法继续")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    compilation_job_name = args.compilation_job_name
    
    # 如果不跳过编译，则编译模型
    if not args.skip_compilation:
        logger.info(f"开始Neo编译流程，目标设备: {args.target_device}")
        
        # 检查模型文件是否存在
        if not os.path.exists(args.model_path):
            logger.error(f"模型文件不存在: {args.model_path}")
            return
        
        # 编译模型
        compilation_job_name = optimize_model_with_neo(
            model_path=args.model_path,
            output_dir=args.output_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            role_arn=args.role_arn,
            target_device=args.target_device,
            region=args.region
        )
        
        if not compilation_job_name:
            logger.error("Neo编译失败，无法继续")
            return
        
        logger.info(f"Neo编译成功，作业名称: {compilation_job_name}")
    else:
        if not compilation_job_name:
            logger.error("跳过编译但未提供编译作业名称，无法继续")
            return
        
        logger.info(f"跳过Neo编译，使用已有作业: {compilation_job_name}")
    
    # 部署模型
    logger.info("开始部署Neo优化模型")
    
    deploy_args = argparse.Namespace(
        neo_job_name=compilation_job_name,
        s3_bucket=args.s3_bucket,
        region=args.region,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name
    )
    
    endpoint_name = build_and_deploy_with_sagemaker(deploy_args)
    
    if not endpoint_name:
        logger.error("部署模型失败")
        return
    
    logger.info(f"模型部署成功，端点名称: {endpoint_name}")
    
    # 保存端点信息
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'compilation_job_name': compilation_job_name,
        'model_path': args.model_path,
        's3_bucket': args.s3_bucket,
        'instance_type': args.instance_type,
        'creation_time': int(time.time()),
        'region': args.region
    }
    
    endpoint_info_path = os.path.join(args.output_dir, "endpoint_info.json")
    with open(endpoint_info_path, "w") as f:
        json.dump(endpoint_info, f, indent=2)
    
    logger.info(f"端点信息已保存到: {endpoint_info_path}")
    
    # 测试端点
    test_result = test_endpoint(endpoint_name, args.region)
    
    if test_result:
        logger.info("可以使用以下代码调用端点:")
        logger.info(f"""
import boto3
import json

def invoke_endpoint(endpoint_name, user_id, item_id, region="{args.region}"):
    runtime = boto3.client('runtime.sagemaker', region_name=region)
    
    # 准备请求数据
    request_data = {
        'user_id': user_id,
        'item_id': item_id
    }
    
    # 调用端点
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(request_data)
    )
    
    # 解析响应
    result = json.loads(response['Body'].read().decode())
    return result

# 测试调用端点
result = invoke_endpoint('{endpoint_name}', 1, 42)
print(f"推理结果: {{result}}")
""")
    
    logger.info("模型部署流程完成")

if __name__ == "__main__":
    main() 