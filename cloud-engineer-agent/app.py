import streamlit as st
from cloud_engineer_agent import execute_predefined_task, execute_custom_task, get_predefined_tasks, PREDEFINED_TASKS
import time  # type: ignore
import re
import json  # type: ignore
import ast
import os
from PIL import Image

st.set_page_config(
    page_title="AWS Cloud Engineer Agent",
    page_icon="👨‍💻",
    layout="wide"
)

# 缓存代理函数
@st.cache_resource
def get_agent_functions():
    # 这只是一个占位符，用于维持缓存行为
    # 实际代理现在直接在cloud_engineer_agent.py中初始化
    return True

# 从响应中移除思考过程并处理格式化的函数
def clean_response(response):
    # 处理None或空响应
    if not response:
        return ""
    
    # 如果不是字符串则转换为字符串
    if not isinstance(response, str):
        try:
            response = str(response)
        except:
            return "Error: Could not convert response to string"
    
    # 移除<thinking>...</thinking>块
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
    
    # 检查响应是否为带有嵌套内容的JSON格式
    if cleaned.find("'role': 'assistant'") >= 0 and cleaned.find("'content'") >= 0 and cleaned.find("'text'") >= 0:
        try:
            # 尝试解析为Python字面量
            data = ast.literal_eval(cleaned)
            if isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
                for item in data['content']:
                    if isinstance(item, dict) and 'text' in item:
                        # 直接返回文本内容（保留markdown）
                        return item['text']
        except:
            # 如果解析失败，尝试使用正则作为备选方案
            match = re.search(r"'text': '(.+?)(?:'}]|})", cleaned, re.DOTALL)
            if match:
                # 反转义内容以保留markdown
                text = match.group(1)
                text = text.replace('\\n', '\n')  # 替换转义的换行符
                text = text.replace('\\t', '\t')  # 替换转义的制表符
                text = text.replace("\\'", "'")   # 替换转义的单引号
                text = text.replace('\\"', '"')   # 替换转义的双引号
                return text
    
    return cleaned.strip()

# 检查文本中的图像路径并显示它们的函数
def display_message_with_images(content):
    # 在内容中查找图像路径 - 支持嵌套的generated-diagrams目录
    image_path_pattern = r'/tmp/generated-diagrams/(?:generated-diagrams/)?[\w\-\.]+\.png'
    image_paths = re.findall(image_path_pattern, content)
    
    # 如果没有找到图像路径，只显示内容为markdown
    if not image_paths:
        st.markdown(content)
        return
    
    # 按图像路径分割内容，以便按顺序显示文本和图像
    segments = re.split(image_path_pattern, content)
    
    for i, segment in enumerate(segments):
        # 显示文本段
        if segment.strip():
            st.markdown(segment.strip())
        
        # 如果有图像则显示
        if i < len(image_paths):
            image_path = image_paths[i]
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"生成的图表", use_container_width=True)
                except Exception as e:
                    st.error(f"显示图像时出错: {e}")
            else:
                st.warning(f"未找到图像: {image_path}")

# 初始化聊天历史
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# 主应用
def main():
    init_chat_history()
    
    # 创建一个两列布局，带有侧边栏和主要内容
    # 侧边栏用于工具和预定义任务
    with st.sidebar:
        st.title("👨‍💻 AWS Cloud Engineer")
        st.markdown("---")
        
        # 预定义任务下拉菜单 - 移至顶部
        st.subheader("预定义任务")
        task_options = list(PREDEFINED_TASKS.values())
        task_keys = list(PREDEFINED_TASKS.keys())
        
        selected_task = st.selectbox(
            "选择预定义任务:",
            options=task_options,
            index=None,
            placeholder="选择任务..."
        )
        
        if selected_task:
            task_index = task_options.index(selected_task)
            task_key = task_keys[task_index]
            
            if st.button("运行选定任务", use_container_width=True):
                # 将任务作为用户消息添加到聊天中
                user_message = f"请 {selected_task.lower()}"
                st.session_state.messages.append({"role": "user", "content": user_message})
                
                # 生成响应
                get_agent_functions()  # 确保代理已缓存
                with st.spinner("正在处理..."):
                    try:
                        result = execute_predefined_task(task_key)
                        cleaned_result = clean_response(result)
                        st.session_state.messages.append({"role": "assistant", "content": cleaned_result})
                        st.rerun()
                    except Exception as e:
                        error_message = f"执行任务时出错: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.rerun()
        
        st.markdown("---")
        
        # AWS配置信息
        st.subheader("AWS配置")
        st.info("使用环境变量中的AWS凭证")
        
        # 可用工具部分
        st.subheader("可用工具")
        
        # 显示AWS CLI工具
        st.markdown("**AWS CLI工具**")
        st.markdown("- `use_aws`: 执行AWS CLI命令")
        st.markdown("**AWS文档MCP工具**")
        st.markdown("**AWS图表MCP工具**")

        # 清除聊天按钮
        st.markdown("---")
        if st.button("清除聊天历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # 带有聊天界面的主要内容区域
    st.title("👨‍💻 AWS云工程师助手")
    st.markdown("询问有关AWS资源、安全、成本优化的问题，或从侧边栏选择预定义任务。")
    
    # 显示聊天消息
    if not st.session_state.messages:
        # 如果没有消息则显示欢迎信息
        with st.chat_message("assistant"):
            st.markdown("👋 你好！我是你的AWS云工程师助手 👨‍💻。我可以帮助你管理、优化和保护你的AWS基础设施。从侧边栏选择预定义任务或询问任何关于AWS的问题！")
    else:
        # 显示现有消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # 使用能够处理图像的特殊显示函数
                display_message_with_images(message["content"])
    
    # 用户输入
    if prompt := st.chat_input("询问有关AWS的问题..."):
        # 将用户消息添加到聊天中
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成响应
        get_agent_functions()  # 确保代理已缓存
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = execute_custom_task(prompt)
                cleaned_response = clean_response(response)
                # 使用能够处理图像的特殊显示函数
                display_message_with_images(cleaned_response)
        
        # 将助手响应添加到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

if __name__ == "__main__":
    main()
