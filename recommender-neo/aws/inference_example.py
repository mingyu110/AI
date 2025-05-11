
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
