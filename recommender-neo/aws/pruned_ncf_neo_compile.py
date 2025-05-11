import os
import torch
import torch.nn as nn
import subprocess
import sys
import boto3
import time
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("pruned-neo-compile")

# 模型和AWS配置
PRUNED_MODEL_PATH = 'output/models/pruned/ncf_pruned_latest.pt'
S3_BUCKET = 'mingyu110'
S3_PREFIX = 'recommender-demo'
ROLE_ARN = 'arn:aws:iam::933505494323:role/SageMakerExecutionRole'
REGION = 'us-east-1'
FRAMEWORK_VERSION = '1.8'

def upload_model_to_s3(model_path, bucket, prefix=None):
    """上传模型到S3"""
    try:
        logger.info(f"===== 上传模型到S3 =====")
        
        # 创建S3客户端
        s3 = boto3.client('s3')
        logger.info(f"已创建S3客户端")
        
        # 准备S3 key
        model_name = os.path.basename(model_path)
        if prefix:
            s3_key = f"{prefix}/models/{model_name}"
        else:
            s3_key = f"models/{model_name}"
        
        # 上传文件
        logger.info(f"开始上传模型到S3: s3://{bucket}/{s3_key}")
        s3.upload_file(model_path, bucket, s3_key)
        logger.info(f"模型上传成功")
        
        # 返回S3 URI
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"模型S3 URI: {s3_uri}")
        
        return s3_uri
    except Exception as e:
        logger.error(f"上传模型到S3失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def compile_with_neo(model_path, target_device="custom"):
    """使用SageMaker Neo编译模型"""
    try:
        logger.info(f"===== 开始Neo编译 =====")
        logger.info(f"模型路径: {model_path}")
        
        # 上传模型到S3
        s3_model_uri = upload_model_to_s3(model_path, S3_BUCKET, S3_PREFIX)
        if not s3_model_uri:
            logger.error("上传模型失败，无法继续Neo编译")
            return False
        
        # 创建编译作业名称
        timestamp = int(time.time())
        job_name = f"pruned-ncf-{timestamp}"
        
        # 设置Neo编译配置
        compile_config = {
            "CompilationJobName": job_name,
            "RoleArn": ROLE_ARN,
            "InputConfig": {
                "S3Uri": s3_model_uri,
                "DataInputConfig": '{"input0":[1,2]}',
                "Framework": "PYTORCH",
                "FrameworkVersion": FRAMEWORK_VERSION
            },
            "OutputConfig": {
                "S3OutputLocation": f"s3://{S3_BUCKET}/{S3_PREFIX}/compilation-output",
                "TargetPlatform": {
                    "Os": "LINUX",
                    "Arch": "X86_64",
                    "Accelerator": "NVIDIA"
                },
                "CompilerOptions": json.dumps({
                    "dtype": "float32",
                    "cuda-ver": "10.2",
                    "trt-ver": "7.1.3",
                    "gpu-code": "sm_70"
                })
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 900  # 15分钟
            }
        }
        
        # 创建Neo客户端并提交编译作业
        neo_client = boto3.client('sagemaker', region_name=REGION)
        
        try:
            response = neo_client.create_compilation_job(**compile_config)
            logger.info(f"Neo编译作业已创建: {job_name}")
            
            # 等待编译作业完成
            logger.info("等待编译作业完成...")
            
            wait_count = 0
            while True:
                response = neo_client.describe_compilation_job(CompilationJobName=job_name)
                status = response["CompilationJobStatus"]
                
                if status == "COMPLETED":
                    output_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
                    logger.info(f"编译作业成功完成: {job_name}")
                    logger.info(f"编译后模型位置: {output_s3_uri}")
                    return True
                    
                elif status == "FAILED":
                    failure_reason = response.get("FailureReason", "未知错误")
                    logger.error(f"编译作业失败: {failure_reason}")
                    return False
                    
                else:
                    wait_count += 1
                    if wait_count % 2 == 0:
                        logger.info(f"编译作业状态: {status}")
                    time.sleep(30)
                    
        except Exception as e:
            logger.error(f"创建Neo编译作业失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Neo编译过程失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_pruned():
    pruned = torch.load(PRUNED_MODEL_PATH, map_location='cpu')
    return pruned

def step1(pruned, save_path):
    # Embedding + Linear (shape与剪枝模型feature_interaction.0一致)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(pruned['num_users'], pruned['embedding_dim'])
            self.item_emb = nn.Embedding(pruned['num_items'], pruned['embedding_dim'])
            self.fc = nn.Linear(pruned['embedding_dim'] * 2, pruned['mlp_layers'][0])
        def forward(self, x):
            user = x[:, 0].long()
            item = x[:, 1].long()
            u = self.user_emb(user)
            v = self.item_emb(item)
            out = torch.cat([u, v], dim=1)
            return self.fc(out)
    model = Model()
    model.user_emb.weight.data.copy_(pruned['state_dict']['user_embedding.weight'])
    model.item_emb.weight.data.copy_(pruned['state_dict']['item_embedding.weight'])
    model.fc.weight.data.copy_(pruned['state_dict']['feature_interaction.0.weight'])
    model.fc.bias.data.copy_(pruned['state_dict']['feature_interaction.0.bias'])
    model.eval()
    traced = torch.jit.trace(model, torch.tensor([[1, 2]], dtype=torch.float32))
    traced.save(save_path)
    print(f"[Step1] Saved: {save_path}")

def step2(pruned, save_path):
    # Embedding + Linear + ReLU (shape与剪枝模型feature_interaction.0一致)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(pruned['num_users'], pruned['embedding_dim'])
            self.item_emb = nn.Embedding(pruned['num_items'], pruned['embedding_dim'])
            self.fc = nn.Linear(pruned['embedding_dim'] * 2, pruned['mlp_layers'][0])
            self.relu = nn.ReLU()
        def forward(self, x):
            user = x[:, 0].long()
            item = x[:, 1].long()
            u = self.user_emb(user)
            v = self.item_emb(item)
            out = torch.cat([u, v], dim=1)
            out = self.fc(out)
            return self.relu(out)
    model = Model()
    model.user_emb.weight.data.copy_(pruned['state_dict']['user_embedding.weight'])
    model.item_emb.weight.data.copy_(pruned['state_dict']['item_embedding.weight'])
    model.fc.weight.data.copy_(pruned['state_dict']['feature_interaction.0.weight'])
    model.fc.bias.data.copy_(pruned['state_dict']['feature_interaction.0.bias'])
    model.eval()
    traced = torch.jit.trace(model, torch.tensor([[1, 2]], dtype=torch.float32))
    traced.save(save_path)
    print(f"[Step2] Saved: {save_path}")

def step3(pruned, save_path):
    # Embedding + 多层MLP
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(pruned['num_users'], pruned['embedding_dim'])
            self.item_emb = nn.Embedding(pruned['num_items'], pruned['embedding_dim'])
            mlp_layers = pruned['mlp_layers']
            layers = []
            input_dim = pruned['embedding_dim'] * 2
            for i, dim in enumerate(mlp_layers):
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.ReLU())
                input_dim = dim
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)
        def forward(self, x):
            user = x[:, 0].long()
            item = x[:, 1].long()
            u = self.user_emb(user)
            v = self.item_emb(item)
            out = torch.cat([u, v], dim=1)
            return self.mlp(out)
    model = Model()
    model.user_emb.weight.data.copy_(pruned['state_dict']['user_embedding.weight'])
    model.item_emb.weight.data.copy_(pruned['state_dict']['item_embedding.weight'])
    # 复制MLP权重
    model.mlp[0].weight.data.copy_(pruned['state_dict']['feature_interaction.0.weight'])
    model.mlp[0].bias.data.copy_(pruned['state_dict']['feature_interaction.0.bias'])
    for i in range(len(pruned['mlp_layers']) - 1):
        mlp_idx = 2 + i * 2
        orig_idx = i * 4
        if mlp_idx < len(model.mlp):
            model.mlp[mlp_idx].weight.data.copy_(pruned['state_dict'][f'mlp_layers.{orig_idx}.weight'])
            model.mlp[mlp_idx].bias.data.copy_(pruned['state_dict'][f'mlp_layers.{orig_idx}.bias'])
    model.mlp[-1].weight.data.copy_(pruned['state_dict']['prediction.weight'])
    model.mlp[-1].bias.data.copy_(pruned['state_dict']['prediction.bias'])
    model.eval()
    traced = torch.jit.trace(model, torch.tensor([[1, 2]], dtype=torch.float32))
    traced.save(save_path)
    print(f"[Step3] Saved: {save_path}")

def step4(pruned, save_path):
    # Embedding + 多层MLP + 手动归一化
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(pruned['num_users'], pruned['embedding_dim'])
            self.item_emb = nn.Embedding(pruned['num_items'], pruned['embedding_dim'])
            mlp_layers = pruned['mlp_layers']
            layers = []
            input_dim = pruned['embedding_dim'] * 2
            for i, dim in enumerate(mlp_layers):
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.ReLU())
                input_dim = dim
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)
        def manual_normalize(self, x, dim=1, eps=1e-8):
            square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
            norm = torch.sqrt(square_sum.clamp(min=eps))
            return x / norm
        def forward(self, x):
            user = x[:, 0].long()
            item = x[:, 1].long()
            u = self.manual_normalize(self.user_emb(user))
            v = self.manual_normalize(self.item_emb(item))
            out = torch.cat([u, v], dim=1)
            return self.mlp(out)
    model = Model()
    model.user_emb.weight.data.copy_(pruned['state_dict']['user_embedding.weight'])
    model.item_emb.weight.data.copy_(pruned['state_dict']['item_embedding.weight'])
    # 复制MLP权重
    model.mlp[0].weight.data.copy_(pruned['state_dict']['feature_interaction.0.weight'])
    model.mlp[0].bias.data.copy_(pruned['state_dict']['feature_interaction.0.bias'])
    for i in range(len(pruned['mlp_layers']) - 1):
        mlp_idx = 2 + i * 2
        orig_idx = i * 4
        if mlp_idx < len(model.mlp):
            model.mlp[mlp_idx].weight.data.copy_(pruned['state_dict'][f'mlp_layers.{orig_idx}.weight'])
            model.mlp[mlp_idx].bias.data.copy_(pruned['state_dict'][f'mlp_layers.{orig_idx}.bias'])
    model.mlp[-1].weight.data.copy_(pruned['state_dict']['prediction.weight'])
    model.mlp[-1].bias.data.copy_(pruned['state_dict']['prediction.bias'])
    model.eval()
    traced = torch.jit.trace(model, torch.tensor([[1, 2]], dtype=torch.float32))
    traced.save(save_path)
    print(f"[Step4] Saved: {save_path}")

if __name__ == "__main__":
    pruned = load_pruned()
    steps = [
        ('step1_ncf.pt', step1),
        ('step2_ncf.pt', step2),
        ('step3_ncf.pt', step3),
        ('step4_ncf.pt', step4),
    ]
    for fname, func in steps:
        save_path = os.path.join('output/models/pruned', fname)
        print(f"\n==== 测试 {fname} ====")
        func(pruned, save_path)
        success = compile_with_neo(save_path)
        if not success:
            print(f"\n[Neo编译失败] 失败模型: {fname}，请重点检查本结构！")
            sys.exit(1)
        else:
            print(f"[Neo编译成功] {fname}")
    print("\n所有测试结构均通过Neo编译！") 