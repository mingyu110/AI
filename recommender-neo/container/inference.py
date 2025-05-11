
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
        output = model.run(input_data)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        prediction = float(output[0].item() if hasattr(output[0], 'item') else output[0][0])
        
        logger.info(f"推理完成: 预测值={prediction:.4f}, 耗时={inference_time:.2f}ms")
        
        return {"prediction": prediction, "inference_time_ms": inference_time}
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
