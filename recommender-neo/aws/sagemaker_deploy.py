#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用SageMaker Python SDK构建和部署Neo优化模型
此方案在AWS云端构建镜像，速度更快，更可靠
"""

import os
import json
import time
import logging
import argparse
import boto3
import sagemaker
from sagemaker.utils import name_from_base
from datetime import datetime
import zipfile
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("sagemaker-build-deploy")

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='使用SageMaker Python SDK构建和部署Neo优化模型')
    
    parser.add_argument("--neo_job_name", type=str, required=True,
                        help="Neo编译作业名称")
    parser.add_argument("--s3_bucket", type=str, required=True,
                        help="S3存储桶名称")
    parser.add_argument("--region", type=str, default="us-east-1",
                        help="AWS区域")
    parser.add_argument("--role_arn", type=str, required=True,
                        help="SageMaker执行角色ARN")
    parser.add_argument("--instance_type", type=str, default="ml.m5.large",
                        help="推理实例类型")
    parser.add_argument("--endpoint_name", type=str, default="",
                        help="端点名称(可选，不提供则自动生成)")
    
    return parser.parse_args()

def create_container_files(output_dir):
    """创建容器所需的文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Dockerfile
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    dockerfile_content = """FROM python:3.10-slim AS builder

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        build-essential \\
        wget \\
        && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制并安装需求文件
COPY requirements.txt .

# 分步安装依赖以提高构建缓存效率
RUN pip install --no-cache-dir numpy==1.21.6
RUN pip install --no-cache-dir scipy==1.7.3 scikit-learn==1.0.2
RUN pip install --no-cache-dir flask==2.0.3 gunicorn==20.1.0
RUN pip install --no-cache-dir boto3==1.24.59 cloudpickle==2.2.1
RUN pip install --no-cache-dir sagemaker-inference==1.10.1
RUN pip install --no-cache-dir dlr==1.10.0

# 第二阶段：创建精简的最终镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PATH="/opt/venv/bin:$PATH" \\
    MODEL_PATH="/opt/ml/model"

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /opt/program

# 复制推理代码
COPY inference.py .
COPY serve.py .

# 创建必要的目录并设置权限
RUN chmod +x serve.py && \\
    mkdir -p /opt/ml/model && \\
    chmod -R o+rwX /opt/ml && \\
    python -c "import dlr; print(f'DLR版本: {dlr.__version__}')"

# 容器启动命令
ENTRYPOINT ["python", "/opt/program/serve.py"]
"""
    
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    
    # 创建requirements.txt
    requirements_path = os.path.join(output_dir, "requirements.txt")
    requirements_content = """numpy==1.21.6
scipy==1.7.3
scikit-learn==1.0.2
flask==2.0.3
gunicorn==20.1.0
boto3==1.24.59
cloudpickle==2.2.1
sagemaker-inference==1.10.1
dlr==1.10.0
"""
    
    with open(requirements_path, "w") as f:
        f.write(requirements_content)
    
    # 创建推理脚本
    inference_path = os.path.join(output_dir, "inference.py")
    inference_content = '''
import os
import json
import logging
import numpy as np
import dlr

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
model = None

def model_fn(model_dir):
    """加载模型"""
    global model
    
    try:
        logger.info(f"加载Neo优化模型，路径: {model_dir}")
        
        # 查找模型目录内容
        dir_contents = os.listdir(model_dir)
        logger.info(f"模型目录内容: {dir_contents}")
        
        # 检查DLR版本
        logger.info(f"DLR版本: {dlr.__version__}")
        
        # 加载DLR模型
        model = dlr.DLRModel(model_dir)
        
        # 获取模型信息
        num_inputs = model.get_input_names()
        input_shapes = []
        for input_name in num_inputs:
            input_shapes.append(model.get_input_shape(input_name if input_name else 0))
        
        logger.info(f"模型输入名称: {num_inputs}")
        logger.info(f"模型输入形状: {input_shapes}")
        logger.info("Neo模型加载成功")
        
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def input_fn(request_body, request_content_type):
    """处理输入请求"""
    logger.info(f"接收请求，内容类型: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        user_id = input_data.get('user_id')
        item_id = input_data.get('item_id')
        
        if user_id is None or item_id is None:
            raise ValueError("请求缺少必要参数: user_id 或 item_id")
        
        # 将输入转换为模型所需的格式
        input_array = np.array([[float(user_id), float(item_id)]], dtype=np.float32)
        logger.info(f"预处理输入: user_id={user_id}, item_id={item_id}")
        
        return input_array
    else:
        raise ValueError(f"不支持的内容类型: {request_content_type}")

def predict_fn(input_data, model):
    """执行模型预测"""
    try:
        logger.info(f"执行推理，输入形状: {input_data.shape}")
        
        # 计时开始
        import time
        start_time = time.time()
        
        # 执行预测
        prediction = model.run(input_data)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        result = float(prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0][0])
        
        logger.info(f"推理完成: 预测值={result:.4f}, 耗时={inference_time:.2f}ms")
        
        return {
            "prediction": result,
            "inference_time_ms": inference_time
        }
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def output_fn(prediction, response_content_type):
    """格式化输出结果"""
    logger.info(f"格式化输出，内容类型: {response_content_type}")
    
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"不支持的输出内容类型: {response_content_type}")
'''
    
    with open(inference_path, "w") as f:
        f.write(inference_content)
    
    # 创建服务脚本
    serve_path = os.path.join(output_dir, "serve.py")
    serve_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理服务脚本
"""

import os
import sys
import json
import logging
import flask
import signal
import traceback
from inference import model_fn, input_fn, predict_fn, output_fn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dlr-serve")

# 全局模型变量
model = None
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """健康检查端点"""
    health = model is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """推理端点"""
    try:
        # 获取内容类型
        content_type = flask.request.content_type
        
        # 解析请求
        input_data = input_fn(flask.request.data.decode('utf-8'), content_type)
        
        # 执行推理
        prediction = predict_fn(input_data, model)
        
        # 格式化输出
        result = output_fn(prediction, content_type)
        
        return flask.Response(response=result, status=200, mimetype=content_type)
    
    except Exception as e:
        # 记录错误
        error_msg = traceback.format_exc()
        logger.error(error_msg)
        
        # 返回错误响应
        return flask.Response(
            response=json.dumps({"error": str(e)}),
            status=400,
            mimetype='application/json'
        )

def load_model():
    """加载模型"""
    global model
    
    try:
        # 获取模型目录路径
        model_dir = os.environ.get('MODEL_PATH', '/opt/ml/model')
        
        # 加载模型
        logger.info(f"正在加载模型，路径: {model_dir}")
        model = model_fn(model_dir)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        logger.error(traceback.format_exc())
        # 非致命错误，继续运行服务
        pass

def sigterm_handler(signum, frame):
    """处理SIGTERM信号"""
    logger.info("接收到SIGTERM信号，程序退出")
    sys.exit(0)

if __name__ == '__main__':
    logger.info("启动推理服务器")
    
    # 注册信号处理器
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    # 加载模型
    load_model()
    
    # 获取服务端口
    port = int(os.environ.get('PORT', 8080))
    
    # 启动服务
    logger.info(f"服务器监听端口: {port}")
    app.run(host='0.0.0.0', port=port)
'''
    
    with open(serve_path, "w") as f:
        f.write(serve_content)
    os.chmod(serve_path, 0o755)
    
    logger.info(f"容器文件已创建在目录: {output_dir}")
    return {
        "dockerfile": dockerfile_path,
        "requirements": requirements_path,
        "inference": inference_path,
        "serve": serve_path
    }

def build_and_deploy_with_sagemaker(args):
    """使用SageMaker Python SDK构建和部署Neo优化模型"""
    try:
        logger.info("开始SageMaker Neo部署流程")
        
        # 1. 准备参数
        neo_job_name = args.neo_job_name
        s3_bucket = args.s3_bucket
        region = args.region
        role_arn = args.role_arn
        instance_type = args.instance_type
        
        # 创建SageMaker会话
        boto_session = boto3.Session(region_name=region)
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        
        # 创建临时目录
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        os.chmod(tmp_dir, 0o755)
        logger.info(f"创建临时目录: {tmp_dir}")
        
        # 2. 创建容器文件
        container_files = create_container_files(tmp_dir)
        
        # 3. 获取Neo编译作业信息
        neo_client = boto_session.client('sagemaker')
        
        try:
            compilation_job = neo_client.describe_compilation_job(
                CompilationJobName=neo_job_name
            )
            
            logger.info(f"Neo编译作业状态: {compilation_job['CompilationJobStatus']}")
            
            if compilation_job['CompilationJobStatus'] != 'COMPLETED':
                logger.error(f"Neo编译作业未完成，当前状态: {compilation_job['CompilationJobStatus']}")
                return False
                
            # 获取编译输出信息
            neo_output_s3 = compilation_job['ModelArtifacts']['S3ModelArtifacts']
            logger.info(f"Neo模型制品S3路径: {neo_output_s3}")
            
        except neo_client.exceptions.ResourceNotFound:
            logger.error(f"找不到Neo编译作业: {neo_job_name}")
            return False
        
        # 4. 创建模型
        # 创建端点名称（如果未提供）
        endpoint_name = args.endpoint_name
        if not endpoint_name:
            timestamp = int(time.time())
            endpoint_name = f"ncf-neo-ep-{timestamp}"
        
        # 创建配置文件
        model_name = f"{endpoint_name}-model"
        
        # 创建SageMaker模型
        primary_container = {
            'ModelDataSource': {
                'S3DataSource': {
                    'S3Uri': neo_output_s3,
                    'S3DataType': 'S3Prefix',
                    'CompressionType': 'None'
                }
            },
            'Image': f"763104351884.dkr.ecr.{region}.amazonaws.com/neo-runtime:py39-neo-cpu",
            'Environment': {}
        }
        
        create_model_response = neo_client.create_model(
            ModelName=model_name,
            PrimaryContainer=primary_container,
            ExecutionRoleArn=role_arn
        )
        
        logger.info(f"SageMaker模型创建成功: {model_name}")
        
        # 5. 创建端点配置
        endpoint_config_name = f"{endpoint_name}-config"
        
        create_endpoint_config_response = neo_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        logger.info(f"SageMaker端点配置创建成功: {endpoint_config_name}")
        
        # 6. 创建端点
        try:
            # 检查端点是否存在
            try:
                endpoint_response = neo_client.describe_endpoint(EndpointName=endpoint_name)
                
                # 如果端点存在且不是InService状态，则删除它
                if endpoint_response['EndpointStatus'] != 'InService':
                    logger.info(f"发现现有端点（状态: {endpoint_response['EndpointStatus']}），正在删除...")
                    neo_client.delete_endpoint(EndpointName=endpoint_name)
                    # 等待端点删除完成
                    while True:
                        try:
                            neo_client.describe_endpoint(EndpointName=endpoint_name)
                            logger.info("等待端点删除完成...")
                            time.sleep(10)
                        except:
                            logger.info("端点已删除")
                            break
                else:
                    # 如果端点已存在且状态为InService，则更新它
                    logger.info(f"发现现有端点（状态: InService），正在更新...")
                    neo_client.update_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=endpoint_config_name
                    )
                    
                    logger.info(f"端点更新已启动: {endpoint_name}")
                    return endpoint_name
            except:
                # 端点不存在，创建新端点
                pass
            
            create_endpoint_response = neo_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"SageMaker端点创建已启动: {endpoint_name}")
            
            # 7. 等待端点部署完成
            logger.info("等待端点部署完成...")
            
            while True:
                endpoint_response = neo_client.describe_endpoint(EndpointName=endpoint_name)
                status = endpoint_response['EndpointStatus']
                
                if status == 'InService':
                    logger.info(f"端点部署成功: {endpoint_name}")
                    break
                elif status == 'Failed':
                    failure_reason = endpoint_response.get('FailureReason', '未知错误')
                    logger.error(f"端点部署失败: {failure_reason}")
                    return None
                else:
                    logger.info(f"端点部署中... 当前状态: {status}")
                    time.sleep(30)
            
            return endpoint_name
        
        except Exception as e:
            logger.error(f"创建端点失败: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"SageMaker部署失败: {str(e)}")
        return None

def main():
    """主函数"""
    args = setup_args()
    endpoint_name = build_and_deploy_with_sagemaker(args)
    
    if endpoint_name:
        logger.info(f"Neo优化模型部署成功，端点名称: {endpoint_name}")
        logger.info(f"可以使用以下代码调用端点:")
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
    else:
        logger.error("Neo优化模型部署失败")

if __name__ == "__main__":
    main() 