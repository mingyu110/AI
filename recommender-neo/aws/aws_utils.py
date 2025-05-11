"""
AWS辅助工具函数和配置验证
"""

import re
import os
import boto3
import logging
import tarfile
import tempfile
import json
from pathlib import Path
from botocore.exceptions import ClientError

from .aws_config import S3_CONFIG, SAGEMAKER_CONFIG, COMPILATION_CONFIG

logger = logging.getLogger("recommender-pipeline")

def validate_role_arn(role_arn):
    """
    验证IAM角色ARN格式是否有效
    
    参数:
    - role_arn: IAM角色ARN字符串
    
    返回:
    - is_valid: 是否有效
    - message: 错误信息（如果有）
    """
    if not role_arn:
        return False, "IAM角色ARN为空"
        
    # 检查占位符
    placeholders = ['YOUR', 'XXXX', 'ACCOUNT_ID', 'ROLE_NAME']
    for placeholder in placeholders:
        if placeholder in role_arn:
            return False, f"IAM角色ARN包含占位符: {placeholder}"
            
    # 验证ARN格式
    pattern = r'^arn:aws:iam::\d{12}:role/[\w+=,.@-]+$'
    if not re.match(pattern, role_arn):
        return False, f"IAM角色ARN格式无效: {role_arn}"
        
    return True, ""
    
def validate_s3_bucket(bucket_name):
    """
    验证S3存储桶名称是否有效
    
    参数:
    - bucket_name: S3存储桶名称
    
    返回:
    - is_valid: 是否有效
    - message: 错误信息（如果有）
    """
    if not bucket_name:
        return False, "S3存储桶名称为空"
        
    # 检查占位符
    placeholders = ['YOUR', 'XXXX', 'BUCKET_NAME']
    for placeholder in placeholders:
        if placeholder in bucket_name:
            return False, f"S3存储桶名称包含占位符: {placeholder}"
            
    # 验证S3存储桶名称格式
    pattern = r'^[a-z0-9][a-z0-9\.-]{1,61}[a-z0-9]$'
    if not re.match(pattern, bucket_name):
        return False, f"S3存储桶名称格式无效: {bucket_name}"
        
    return True, ""
    
def check_aws_connectivity(region=None):
    """
    检查AWS连接性
    
    参数:
    - region: AWS区域
    
    返回:
    - is_connected: 是否连接成功
    - message: 错误信息（如果有）
    """
    try:
        # 尝试列出S3存储桶来验证连接性
        boto3.client('s3', region_name=region).list_buckets()
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if "AccessDenied" in error_msg:
            return False, "AWS访问被拒绝，请检查IAM权限"
        elif "InvalidAccessKeyId" in error_msg:
            return False, "无效的AWS访问密钥"
        elif "SignatureDoesNotMatch" in error_msg:
            return False, "AWS签名不匹配，请检查密钥是否正确"
        elif "InvalidToken" in error_msg:
            return False, "无效的AWS令牌"
        else:
            return False, f"AWS连接失败: {error_msg}"

def check_s3_bucket_exists(bucket_name, region=None):
    """
    检查S3存储桶是否存在
    
    参数:
    - bucket_name: S3存储桶名称
    - region: AWS区域
    
    返回:
    - exists: 是否存在
    - message: 错误信息（如果有）
    """
    try:
        s3 = boto3.client('s3', region_name=region)
        
        # 尝试获取存储桶位置 (不支持us-east-1之外的区域直接列出桶)
        s3.get_bucket_location(Bucket=bucket_name)
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if "NoSuchBucket" in error_msg:
            return False, f"S3存储桶 {bucket_name} 不存在"
        elif "AccessDenied" in error_msg:
            # 如果返回AccessDenied，通常意味着存储桶存在但没有权限
            return True, "S3存储桶可能存在，但当前用户没有访问权限"
        else:
            return False, f"检查S3存储桶失败: {error_msg}"

def check_sagemaker_access(role_arn, region=None):
    """
    检查SageMaker访问权限
    
    参数:
    - role_arn: IAM角色ARN
    - region: AWS区域
    
    返回:
    - has_access: 是否有权限
    - message: 错误信息（如果有）
    """
    try:
        # 尝试列出SageMaker训练作业
        boto3.client('sagemaker', region_name=region).list_training_jobs(MaxResults=1)
        return True, ""
    except Exception as e:
        error_msg = str(e)
        if "AccessDenied" in error_msg:
            return False, "SageMaker访问被拒绝，请检查IAM权限"
        else:
            return False, f"检查SageMaker访问失败: {error_msg}"
            
def verify_aws_config(aws_config):
    """
    验证AWS配置
    
    参数:
    - aws_config: AWS配置字典，包含角色ARN、S3存储桶等信息
    
    返回:
    - is_valid: 是否有效
    - issues: 问题列表
    """
    issues = []
    
    # 验证角色ARN
    role_arn = aws_config.get('sagemaker_config', {}).get('role_arn')
    is_valid, message = validate_role_arn(role_arn)
    if not is_valid:
        issues.append(f"IAM角色ARN无效: {message}")
    
    # 验证S3存储桶
    bucket_name = aws_config.get('s3_config', {}).get('bucket')
    is_valid, message = validate_s3_bucket(bucket_name)
    if not is_valid:
        issues.append(f"S3存储桶名称无效: {message}")
    
    # 检查AWS连接性
    region = aws_config.get('aws_region')
    is_connected, message = check_aws_connectivity(region)
    if not is_connected:
        issues.append(f"AWS连接失败: {message}")
    else:
        # 检查S3存储桶是否存在
        exists, message = check_s3_bucket_exists(bucket_name, region)
        if not exists:
            issues.append(f"S3存储桶检查失败: {message}")
        
        # 检查SageMaker访问权限
        has_access, message = check_sagemaker_access(role_arn, region)
        if not has_access:
            issues.append(f"SageMaker访问检查失败: {message}")
    
    return len(issues) == 0, issues

def create_s3_bucket_if_not_exists(bucket_name, region=None):
    """
    如果S3存储桶不存在则创建
    
    参数:
    - bucket_name: S3存储桶名称
    - region: AWS区域
    
    返回:
    - success: 是否成功
    - message: 错误信息（如果有）
    """
    s3 = boto3.client('s3', region_name=region)
    
    try:
        # 检查存储桶是否存在
        s3.head_bucket(Bucket=bucket_name)
        logger.info(f"S3存储桶 {bucket_name} 已存在")
        return True, ""
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        
        # 如果存储桶不存在，则创建
        if error_code == '404' or error_code == 'NoSuchBucket':
            try:
                if region is None or region == 'us-east-1':
                    # us-east-1区域不需要指定LocationConstraint
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    # 其他区域需要指定LocationConstraint
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                logger.info(f"S3存储桶 {bucket_name} 创建成功")
                return True, ""
            except Exception as create_error:
                error_msg = str(create_error)
                logger.error(f"创建S3存储桶失败: {error_msg}")
                return False, f"创建S3存储桶失败: {error_msg}"
        else:
            logger.error(f"检查S3存储桶失败: {str(e)}")
            return False, f"检查S3存储桶失败: {str(e)}"

def upload_file_to_s3(file_path, bucket, object_key=None):
    """
    上传文件到S3
    
    参数:
    - file_path: 本地文件路径
    - bucket: S3存储桶
    - object_key: S3对象键（如果为None，则使用文件名）
    
    返回:
    - success: 是否成功
    - s3_uri: S3 URI（如果成功）
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False, None
    
    # 如果未指定对象键，则使用文件名
    if object_key is None:
        object_key = os.path.basename(file_path)
    
    try:
        s3 = boto3.client('s3', region_name=S3_CONFIG.get('region'))
        s3.upload_file(file_path, bucket, object_key)
        s3_uri = f"s3://{bucket}/{object_key}"
        logger.info(f"文件已上传到 {s3_uri}")
        return True, s3_uri
    except Exception as e:
        logger.error(f"上传文件到S3失败: {str(e)}")
        return False, None

def upload_directory_to_s3(dir_path, bucket, prefix=""):
    """
    上传目录及其内容到S3
    
    参数:
    - dir_path: 本地目录路径
    - bucket: S3存储桶
    - prefix: S3对象键前缀
    
    返回:
    - success: 是否成功
    - uploaded_files: 上传的文件数量
    """
    if not os.path.isdir(dir_path):
        logger.error(f"目录不存在: {dir_path}")
        return False, 0
    
    try:
        s3 = boto3.client('s3', region_name=S3_CONFIG.get('region'))
        uploaded_files = 0
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                local_path = os.path.join(root, file)
                # 计算相对路径以构建S3对象键
                relative_path = os.path.relpath(local_path, dir_path)
                s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
                
                try:
                    s3.upload_file(local_path, bucket, s3_key)
                    uploaded_files += 1
                except Exception as file_error:
                    logger.warning(f"上传文件 {local_path} 失败: {str(file_error)}")
        
        logger.info(f"已上传 {uploaded_files} 个文件到 s3://{bucket}/{prefix}")
        return True, uploaded_files
    except Exception as e:
        logger.error(f"上传目录到S3失败: {str(e)}")
        return False, 0

def upload_dataset_to_s3(data_dir, bucket=None, prefix=None):
    """
    上传数据集到S3
    
    参数:
    - data_dir: 本地数据目录路径
    - bucket: S3存储桶（如果为None，则使用配置中的存储桶）
    - prefix: S3对象键前缀（如果为None，则使用配置中的数据前缀）
    
    返回:
    - success: 是否成功
    - s3_data_path: S3 URI（如果成功）
    """
    bucket = bucket or S3_CONFIG.get('bucket')
    prefix = prefix or S3_CONFIG.get('data_prefix')
    
    # 验证存储桶
    success, message = create_s3_bucket_if_not_exists(bucket, S3_CONFIG.get('region'))
    if not success:
        return False, message
    
    # 上传数据
    success, uploaded_files = upload_directory_to_s3(data_dir, bucket, prefix)
    if not success or uploaded_files == 0:
        return False, "上传数据集失败或没有文件被上传"
    
    s3_data_path = f"s3://{bucket}/{prefix}"
    return True, s3_data_path

def upload_model_to_s3(model_path, bucket=None, prefix=None):
    """
    上传模型到S3
    
    参数:
    - model_path: 本地模型文件路径
    - bucket: S3存储桶（如果为None，则使用配置中的存储桶）
    - prefix: S3对象键前缀（如果为None，则使用配置中的模型前缀）
    
    返回:
    - success: 是否成功
    - s3_model_uri: S3 URI（如果成功）
    """
    bucket = bucket or S3_CONFIG.get('bucket')
    prefix = prefix or S3_CONFIG.get('model_prefix')
    
    # 验证存储桶
    success, message = create_s3_bucket_if_not_exists(bucket, S3_CONFIG.get('region'))
    if not success:
        return False, message
    
    # 如果是目录，则打包为tar.gz
    if os.path.isdir(model_path):
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            temp_path = temp_file.name
            
            with tarfile.open(temp_path, 'w:gz') as tar:
                for root, _, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        tar.add(file_path, arcname=arcname)
            
            model_name = os.path.basename(model_path)
            object_key = f"{prefix}/{model_name}.tar.gz"
            success, s3_uri = upload_file_to_s3(temp_path, bucket, object_key)
            
            # 删除临时文件
            os.unlink(temp_path)
            
            return success, s3_uri
    else:
        # 单个文件直接上传
        model_name = os.path.basename(model_path)
        object_key = f"{prefix}/{model_name}"
        return upload_file_to_s3(model_path, bucket, object_key)

def download_file_from_s3(s3_uri, local_path):
    """
    从S3下载文件
    
    参数:
    - s3_uri: S3 URI (格式: s3://bucket/key)
    - local_path: 本地保存路径
    
    返回:
    - success: 是否成功
    """
    try:
        # 解析S3 URI
        s3_uri = s3_uri.strip()
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"无效的S3 URI: {s3_uri}")
        
        bucket, key = s3_uri[5:].split('/', 1)
        
        # 确保本地目录存在
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        # 下载文件
        s3 = boto3.client('s3', region_name=S3_CONFIG.get('region'))
        s3.download_file(bucket, key, local_path)
        
        logger.info(f"文件已从 {s3_uri} 下载到 {local_path}")
        return True
    except Exception as e:
        logger.error(f"从S3下载文件失败: {str(e)}")
        return False 

def prepare_model_for_neo(model_path, output_dir, temp_dir=None):
    """
    准备模型文件用于SageMaker Neo编译
    
    参数:
    - model_path: 本地模型文件路径
    - output_dir: 输出目录路径
    - temp_dir: 临时目录路径（如果为None，则使用临时目录）
    
    返回:
    - success: 是否成功
    - tar_path: 打包后的模型tar.gz文件路径（如果成功）
    """
    try:
        model_name = os.path.basename(os.path.splitext(model_path)[0])
        
        # 创建临时目录
        temp_dir = temp_dir or tempfile.mkdtemp()
        
        # 创建模型和推理代码目录
        model_dir = os.path.join(temp_dir, "model")
        code_dir = os.path.join(temp_dir, "code")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(code_dir, exist_ok=True)
        
        # 复制模型权重到code目录
        import shutil
        shutil.copy(model_path, os.path.join(code_dir, "model.pth"))
        
        # 加载模型信息
        info_path = f"{os.path.splitext(model_path)[0]}_info.json"
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                
            config = model_info.get('config', {})
            dataset_stats = model_info.get('dataset_stats', {})
        else:
            logger.warning(f"模型信息文件不存在: {info_path}")
            config = {}
            dataset_stats = {'num_users': 1000, 'num_items': 2000}
        
        # 创建模型定义文件
        with open(os.path.join(code_dir, "model.py"), "w") as f:
            # 直接读取已准备好的模型文件内容
            with open('fix_tmp/model.py', 'r') as model_file:
                model_content = model_file.read()
                
            # 替换占位符
            model_content = model_content.replace('num_users = 1000', f'num_users = {dataset_stats.get("num_users", 1000)}')
            model_content = model_content.replace('num_items = 2000', f'num_items = {dataset_stats.get("num_items", 2000)}')
            model_content = model_content.replace('embedding_dim = 32', f'embedding_dim = {config.get("embedding_dim", 32)}')
            model_content = model_content.replace('mlp_layers = [64, 32, 16]', f'mlp_layers = {config.get("mlp_layers", [64, 32, 16])}')
            model_content = model_content.replace('dropout = 0.3', f'dropout = {config.get("dropout", 0.3)}')
            
            # 写入内容
            f.write(model_content)
        
        # 创建inference.py文件
        with open(os.path.join(code_dir, "inference.py"), "w") as f:
            f.write("""
import torch
import numpy as np
import os
import json
import sys

# 添加代码目录到模块搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

try:
    import model as model_module
except ImportError:
    sys.path.append('code')
    import model as model_module

def model_fn(model_dir):
    # 加载编译后的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 尝试加载Neo编译的模型
    try:
        # Neo编译后的模型文件路径
        neo_model_path = os.path.join(model_dir, "compiled_model")
        if os.path.exists(neo_model_path):
            # Neo编译后的模型
            model = torch.jit.load(neo_model_path, map_location=device)
            print("成功加载Neo编译后的模型")
        else:
            # 原始脚本化模型
            model_path = os.path.join(model_dir, "model.pt")
            model = torch.jit.load(model_path, map_location=device)
            print("成功加载原始脚本化模型")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        raise
    
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        import json
        data = json.loads(request_body)
        
        # 支持批量预测
        if isinstance(data, list):
            # 列表形式的批量预测
            tensor = torch.tensor([
                [float(item.get('user_id', 0)), float(item.get('item_id', 0))]
                for item in data
            ], dtype=torch.float)
        else:
            # 单条预测
            tensor = torch.tensor([[
                float(data.get('user_id', 0)), 
                float(data.get('item_id', 0))
            ]], dtype=torch.float)
        
        return tensor
    else:
        raise ValueError(f"不支持的内容类型: {request_content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(model, 'to'):
        model = model.to(device)
    if hasattr(input_data, 'to'):
        input_data = input_data.to(device)
    
    with torch.no_grad():
        prediction = model(input_data)
    
    return prediction.cpu().numpy()

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        import json
        # 如果是批量预测，返回列表；否则返回单个值
        if len(prediction.shape) > 0 and prediction.shape[0] > 1:
            return json.dumps({'predictions': prediction.tolist()})
        else:
            return json.dumps({'prediction': float(prediction.item() if hasattr(prediction, 'item') else prediction[0])})
    else:
        raise ValueError(f"不支持的内容类型: {response_content_type}")
""")
        
        # 创建trace_model.py用于生成Torch脚本
        with open(os.path.join(code_dir, "trace_model.py"), "w") as f:
            f.write("""
import torch
import os
import model as model_module

def main():
    # 获取模型
    model = model_module.get_model_for_neo()
    
    # 为模型加载权重，这部分在get_model_for_neo中已经完成
    # model.load_state_dict(torch.load("model.pth"))
    
    # 确保模型处于评估模式
    model.eval()
    
    # 示例输入 - [batch_size, 2]
    example_input = torch.tensor([[1.0, 2.0]], dtype=torch.float)
    
    # 先运行一次前向传播，确保模型可以工作
    with torch.no_grad():
        output = model(example_input)
        print(f"测试输出: {output.item()}")
    
    # 跟踪并导出模型
    with torch.no_grad():
        # 确保输入是在CPU上
        example_input = example_input.cpu()
        model = model.cpu()
        
        # 跟踪模型
        try:
            traced_model = torch.jit.trace(model, example_input)
            
            # 验证跟踪模型
            traced_output = traced_model(example_input)
            print(f"跟踪后输出: {traced_output.item()}")
            
            # 保存模型
            torch.jit.save(traced_model, "model.pt")
            print("模型已跟踪并保存为model.pt")
        except Exception as e:
            print(f"跟踪模型时出错: {e}")
            
            # 尝试使用脚本化替代方法
            print("尝试使用torch.jit.script方法...")
            scripted_model = torch.jit.script(model)
            scripted_output = scripted_model(example_input)
            print(f"脚本化后输出: {scripted_output.item()}")
            torch.jit.save(scripted_model, "model.pt")
            print("模型已脚本化并保存为model.pt")

if __name__ == "__main__":
    main()
""")
        
        # 创建requirements.txt
        with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
            f.write("torch>=1.8.0\nnumpy>=1.19.0\n")
        
        # 执行跟踪脚本
        logger.info("生成跟踪模型...")
        try:
            # 确保当前目录包含模型文件
            current_dir = os.getcwd()
            os.chdir(code_dir)
            
            import subprocess
            result = subprocess.run(["python", "trace_model.py"], 
                                 capture_output=True, text=True, check=False)
            
            # 切回原目录
            os.chdir(current_dir)
            
            if result.returncode != 0:
                logger.error(f"生成跟踪模型失败: {result.stderr}")
                return False, None
            
            logger.info("跟踪模型生成成功")
        except Exception as e:
            logger.error(f"生成跟踪模型时出错: {str(e)}")
            return False, None
        
        # 打包模型为tar.gz
        os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, f"{model_name}_neo.tar.gz")
        
        with tarfile.open(tar_path, "w:gz") as tar:
            # 只添加脚本化模型文件（PyTorch只允许一个.pt或.pth文件）
            tar.add(os.path.join(code_dir, "model.pt"), arcname="model.pt")
            
            # 添加推理相关文件
            tar.add(os.path.join(code_dir, "inference.py"), arcname="code/inference.py")
            tar.add(os.path.join(code_dir, "model.py"), arcname="code/model.py")
            tar.add(os.path.join(code_dir, "requirements.txt"), arcname="code/requirements.txt")
        
        logger.info(f"模型已打包到 {tar_path}")
        return True, tar_path
    except Exception as e:
        logger.error(f"准备模型失败: {str(e)}")
        return False, None

def optimize_model_with_neo(model_path, output_dir=None, s3_bucket=None, s3_prefix=None, role_arn=None):
    """
    使用SageMaker Neo优化模型
    
    参数:
    - model_path: 本地模型文件路径
    - output_dir: 输出目录路径
    - s3_bucket: S3存储桶（如果为None，则使用配置中的存储桶）
    - s3_prefix: S3对象键前缀（如果为None，则使用配置中的模型前缀）
    - role_arn: SageMaker执行角色ARN（如果为None，则使用配置中的角色ARN）
    
    返回:
    - success: 是否成功
    - compilation_job_name: 编译作业名称（如果成功）
    """
    import boto3
    import time
    from datetime import datetime
    
    # 设置默认值
    s3_bucket = s3_bucket or S3_CONFIG.get('bucket')
    s3_prefix = s3_prefix or S3_CONFIG.get('model_prefix')
    role_arn = role_arn or SAGEMAKER_CONFIG.get('role_arn')
    output_dir = output_dir or './output/neo'
    region = S3_CONFIG.get('region')
    
    # 准备模型
    logger.info("准备模型用于Neo优化...")
    success, model_tar_path = prepare_model_for_neo(model_path, output_dir)
    if not success:
        return False, None
    
    # 上传模型到S3
    logger.info("上传模型到S3...")
    s3_model_prefix = f"{s3_prefix}/neo"
    success, s3_model_uri = upload_file_to_s3(
        model_tar_path, 
        s3_bucket, 
        f"{s3_model_prefix}/{os.path.basename(model_tar_path)}"
    )
    if not success:
        return False, None
    
    # 创建SageMaker客户端
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    # 创建编译作业名称
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    compilation_job_name = f"ncf-neo-compile-{timestamp}"
    
    # 设置编译作业参数
    target_platform = {
        "Os": COMPILATION_CONFIG.get('target_platform_os', 'LINUX'),
        "Arch": COMPILATION_CONFIG.get('target_platform_arch', 'X86_64')
    }
    
    # 如果配置中指定了Accelerator，则添加到参数中
    if 'target_platform_accelerator' in COMPILATION_CONFIG and COMPILATION_CONFIG['target_platform_accelerator']:
        target_platform["Accelerator"] = COMPILATION_CONFIG['target_platform_accelerator']
    else:
        # 默认使用NVIDIA作为加速器
        target_platform["Accelerator"] = "NVIDIA"
    
    # 创建Neo编译作业
    logger.info(f"创建Neo编译作业: {compilation_job_name}")
    try:
        response = sagemaker_client.create_compilation_job(
            CompilationJobName=compilation_job_name,
            RoleArn=role_arn,
            InputConfig={
                'S3Uri': s3_model_uri,
                'DataInputConfig': '{"input0": [1, 2]}',
                'Framework': SAGEMAKER_CONFIG.get('framework', 'PYTORCH'),
                'FrameworkVersion': SAGEMAKER_CONFIG.get('framework_version', '1.8')
            },
            OutputConfig={
                'S3OutputLocation': f"s3://{s3_bucket}/{s3_prefix}/neo/compiled",
                'TargetPlatform': target_platform,
                'CompilerOptions': json.dumps(COMPILATION_CONFIG.get('compiler_options', {}))
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 7200
            }
        )
        
        # 等待编译作业完成
        logger.info("等待编译作业完成...")
        
        max_wait_time = 1200  # 20分钟
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                logger.warning("编译作业超时")
                break
                
            # 获取编译作业状态
            response = sagemaker_client.describe_compilation_job(
                CompilationJobName=compilation_job_name
            )
            
            status = response['CompilationJobStatus']
            logger.info(f"编译作业状态: {status}")
            
            if status == 'COMPLETED':
                logger.info("编译作业成功完成!")
                return True, compilation_job_name
            elif status in ['FAILED', 'STOPPED']:
                failure_reason = response.get('FailureReason', '未知错误')
                logger.error(f"编译作业失败: {failure_reason}")
                return False, None
                
            # 等待30秒后再检查
            time.sleep(30)
    
    except Exception as e:
        logger.error(f"创建Neo编译作业失败: {str(e)}")
        return False, None

def deploy_neo_model(compilation_job_name, endpoint_name=None, instance_type=None, role_arn=None):
    """
    部署Neo优化的模型
    
    参数:
    - compilation_job_name: 编译作业名称
    - endpoint_name: 端点名称（如果为None，则自动生成）
    - instance_type: 实例类型（如果为None，则使用配置中的实例类型）
    - role_arn: SageMaker执行角色ARN（如果为None，则使用配置中的角色ARN）
    
    返回:
    - success: 是否成功
    - endpoint_name: 端点名称（如果成功）
    """
    import boto3
    import time
    from datetime import datetime
    
    # 设置默认值
    instance_type = instance_type or SAGEMAKER_CONFIG.get('instance_type', 'ml.m5.xlarge')
    role_arn = role_arn or SAGEMAKER_CONFIG.get('role_arn')
    region = S3_CONFIG.get('region')
    
    if endpoint_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        endpoint_name = f"ncf-neo-endpoint-{timestamp}"
    
    # 创建SageMaker客户端
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        # 获取编译作业信息
        response = sagemaker_client.describe_compilation_job(
            CompilationJobName=compilation_job_name
        )
        
        if response['CompilationJobStatus'] != 'COMPLETED':
            logger.error(f"编译作业未完成，当前状态: {response['CompilationJobStatus']}")
            return False, None
        
        # 获取编译后的模型位置
        model_data_url = response['ModelArtifacts']['S3ModelArtifacts']
        
        # 创建模型
        model_name = f"neo-{compilation_job_name}"
        logger.info(f"创建模型: {model_name}")
        
        # 根据目标平台选择容器镜像
        target_platform = response['OutputConfig']['TargetPlatform']
        framework = response['InputConfig']['Framework']
        
        # 根据区域、框架和平台获取正确的容器URI
        # 这里使用简化的逻辑，实际应用中需要更复杂的映射
        # 参考: https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-container-images.html
        container_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference-neo:latest"
        
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20'
                }
            },
            ExecutionRoleArn=role_arn
        )
        
        # 创建端点配置
        config_name = f"{model_name}-config"
        logger.info(f"创建端点配置: {config_name}")
        
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type
                }
            ]
        )
        
        # 创建端点
        logger.info(f"创建端点: {endpoint_name}")
        
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # 等待端点创建完成
        logger.info("等待端点创建完成...")
        
        max_wait_time = 1800  # 30分钟
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                logger.warning("端点创建超时")
                break
                
            # 获取端点状态
            response = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            status = response['EndpointStatus']
            logger.info(f"端点状态: {status}")
            
            if status == 'InService':
                logger.info("端点创建成功!")
                return True, endpoint_name
            elif status in ['Failed', 'OutOfService']:
                failure_reason = response.get('FailureReason', '未知错误')
                logger.error(f"端点创建失败: {failure_reason}")
                return False, None
                
            # 等待30秒后再检查
            time.sleep(30)
    
    except Exception as e:
        logger.error(f"部署Neo模型失败: {str(e)}")
        return False, None

def invoke_endpoint(endpoint_name, user_id, item_id, region=None):
    """
    调用SageMaker端点进行推理
    
    参数:
    - endpoint_name: 端点名称
    - user_id: 用户ID
    - item_id: 物品ID
    - region: AWS区域（如果为None，则使用配置中的区域）
    
    返回:
    - success: 是否成功
    - prediction: 预测结果（如果成功）
    """
    import boto3
    import json
    
    region = region or S3_CONFIG.get('region')
    
    try:
        # 创建SageMaker Runtime客户端
        runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        # 准备输入数据
        payload = json.dumps({'user_id': user_id, 'item_id': item_id})
        
        # 调用端点
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # 解析响应
        result = json.loads(response['Body'].read().decode())
        
        return True, result
    except Exception as e:
        logger.error(f"调用端点失败: {str(e)}")
        return False, None

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 验证AWS配置
    print("验证AWS配置...")
    aws_config = {
        'sagemaker_config': SAGEMAKER_CONFIG,
        's3_config': S3_CONFIG,
        'aws_region': S3_CONFIG.get('region')
    }
    is_valid, issues = verify_aws_config(aws_config)
    
    if not is_valid:
        print("AWS配置验证失败:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("AWS配置验证成功!") 