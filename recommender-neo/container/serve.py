#!/usr/bin/env python

import os
import json
import sys
import signal
import traceback
import flask
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入推理处理器
import inference

# 初始化Flask应用
app = flask.Flask(__name__)

# 全局变量
model = None
model_loaded = False

@app.route('/ping', methods=['GET'])
def ping():
    """健康检查端点"""
    global model_loaded
    
    # 检查模型目录是否存在
    model_dir = os.environ.get('MODEL_PATH', '/opt/ml/model')
    if not os.path.exists(model_dir):
        logger.error(f"模型目录不存在: {model_dir}")
        return flask.Response(response='', status=404, mimetype='application/json')
        
    # 检查模型目录内容
    try:
        dir_contents = os.listdir(model_dir)
        logger.info(f"模型目录内容: {dir_contents}")
        if len(dir_contents) == 0:
            logger.error(f"模型目录为空: {model_dir}")
            return flask.Response(response='', status=404, mimetype='application/json')
    except Exception as e:
        logger.error(f"无法读取模型目录: {str(e)}")
        return flask.Response(response='', status=404, mimetype='application/json')
    
    status = 200 if model_loaded else 404
    return flask.Response(response='', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """处理预测请求"""
    global model, model_loaded
    
    # 如果模型尚未加载，则加载模型
    if not model_loaded:
        try:
            model_dir = os.environ.get('MODEL_PATH', '/opt/ml/model')
            model = inference.model_fn(model_dir)
            model_loaded = True
            logger.info("模型已成功加载")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return flask.Response(
                response=json.dumps({"error": f"模型加载失败: {str(e)}"}),
                status=500,
                mimetype='application/json'
            )
    
    # 获取请求内容类型
    content_type = flask.request.content_type
    
    # 处理预测请求
    try:
        # 解析输入
        input_data = inference.input_fn(flask.request.data.decode('utf-8'), content_type)
        
        # 执行预测
        prediction = inference.predict_fn(input_data, model)
        
        # 格式化输出
        result = inference.output_fn(prediction, 'application/json')
        
        return flask.Response(response=result, status=200, mimetype='application/json')
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        return flask.Response(
            response=json.dumps({"error": str(e)}),
            status=400,
            mimetype='application/json'
        )

if __name__ == '__main__':
    # 设置信号处理器
    def signal_handler(sig, frame):
        logger.info('接收到信号，正在关闭服务...')
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 获取模型目录
    model_dir = os.environ.get('MODEL_PATH', '/opt/ml/model')
    
    # 加载模型
    try:
        model = inference.model_fn(model_dir)
        model_loaded = True
        logger.info("启动时模型已成功加载")
    except Exception as e:
        logger.warning(f"启动时模型加载失败: {str(e)}，将在首次请求时加载")
        logger.warning(traceback.format_exc())
    
    # 启动服务器
    app.run(host='0.0.0.0', port=8080)
