from strands import Agent
from strands.tools.mcp import MCPClient
from strands.models import BedrockModel
from mcp import StdioServerParameters, stdio_client
from strands_tools import use_aws

import os
import atexit
from typing import Dict

# 定义常见的云工程任务
PREDEFINED_TASKS = {
    "ec2_status": "列出所有EC2实例及其状态",
    "s3_buckets": "列出所有S3存储桶及其创建日期",
    "cloudwatch_alarms": "检查处于ALARM状态的CloudWatch告警",
    "iam_users": "列出所有IAM用户及其最近活动",
    "security_groups": "分析安全组潜在漏洞",
    "cost_optimization": "识别可优化成本的资源",
    "lambda_functions": "列出所有Lambda函数及其运行时",
    "rds_instances": "检查所有RDS实例的状态",
    "vpc_analysis": "分析VPC配置并提出改进建议",
    "ebs_volumes": "查找可移除的未附加EBS卷",
    "generate_diagram": "根据用户描述生成AWS架构图"
}

# 设置AWS文档MCP客户端
print("正在初始化AWS文档MCP客户端...")
aws_docs_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
))

# 设置AWS图表MCP客户端  
print("正在初始化AWS图表MCP客户端...")
aws_diagram_mcp_client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-diagram-mcp-server@latest"])
))

# 启动MCP客户端
docs_tools = []
diagram_tools = []

try:
    print("启动AWS文档MCP客户端...")
    aws_docs_mcp_client.start()
    docs_tools = aws_docs_mcp_client.list_tools_sync()
    docs_tool_names = [tool.tool_name for tool in docs_tools]
    print(f"AWS文档MCP客户端启动成功，获得 {len(docs_tools)} 个工具: {', '.join(docs_tool_names)}")
except Exception as e:
    print(f"警告：AWS文档MCP客户端启动失败: {e}")

try:
    print("启动AWS图表MCP客户端...")
    aws_diagram_mcp_client.start()
    diagram_tools = aws_diagram_mcp_client.list_tools_sync()
    diagram_tool_names = [tool.tool_name for tool in diagram_tools]
    print(f"AWS图表MCP客户端启动成功，获得 {len(diagram_tools)} 个工具: {', '.join(diagram_tool_names)}")
except Exception as e:
    print(f"警告：AWS图表MCP客户端启动失败: {e}")

# 使用系统推理配置文件创建BedrockModel
bedrock_model = BedrockModel(
    model_id="us.amazon.nova-premier-v1:0",  # 系统推理配置文件ID
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    temperature=0.1,
)

# 代理的系统提示
system_prompt = """
你是一位专家级AWS云工程师助手。你的工作是帮助管理、优化、保障AWS基础设施安全并提供最佳实践。你可以：

1. 分析AWS资源和配置
2. 提供安全改进建议
3. 识别成本优化机会
4. 排查AWS服务问题
5. 解释AWS概念和最佳实践
6. 使用AWS图表工具生成基础设施图
7. 在AWS文档中搜索特定信息

当被要求创建图表时，使用AWS图表MCP工具根据用户描述生成架构的可视化表示。
在将文本描述转换为完整架构图时要有创意和全面性。

**重要的图表生成规则：**
- 当调用generate_diagram工具时，必须设置workspace_dir参数为"/tmp/generated-diagrams"
- 生成图表后，告诉用户图表的完整文件路径
- 图表文件通常保存为PNG格式

始终提供清晰、可操作的建议，在适用时提供具体的AWS CLI命令或控制台步骤。
在建议中注重安全最佳实践和成本优化。

**关键规则：**
- 永远不要在响应中包含<thinking>标签或暴露你的内部思考过程
- 直接提供解决方案，不要显示推理步骤
- 保持专业、简洁的回答风格
"""

# 使用所有工具和Bedrock Nova Premier模型创建代理
agent = Agent(
    tools=[use_aws] + docs_tools + diagram_tools,
    model=bedrock_model,
    system_prompt=system_prompt
)

# 为MCP客户端注册清理处理程序
def cleanup():
    if 'aws_docs_mcp_client' in globals() and hasattr(aws_docs_mcp_client, '_session'):
        try:
            aws_docs_mcp_client.stop()
            print("AWS文档MCP客户端已停止")
        except Exception as e:
            print(f"停止AWS文档MCP客户端时出错: {e}")
    
    if 'aws_diagram_mcp_client' in globals() and hasattr(aws_diagram_mcp_client, '_session'):
        try:
            aws_diagram_mcp_client.stop()
            print("AWS图表MCP客户端已停止")
        except Exception as e:
            print(f"停止AWS图表MCP客户端时出错: {e}")

atexit.register(cleanup)

# 执行预定义任务的函数
def execute_predefined_task(task_key: str) -> str:
    """执行预定义的云工程任务"""
    if task_key not in PREDEFINED_TASKS:
        return f"错误: 在预定义任务中未找到'{task_key}'。"
    
    task_description = PREDEFINED_TASKS[task_key]
    return execute_custom_task(task_description)

# 执行自定义任务的函数
def execute_custom_task(task_description: str) -> str:
    """根据描述执行自定义云工程任务"""
    try:
        response = agent(task_description)
        
        # 通过提取消息处理AgentResult对象
        if hasattr(response, 'message'):
            return response.message
        
        # 处理其他类型的响应
        return str(response)
    except Exception as e:
        return f"执行任务时出错: {str(e)}"

# 获取预定义任务的函数
def get_predefined_tasks() -> Dict[str, str]:
    """返回预定义任务的字典"""
    return PREDEFINED_TASKS


if __name__ == "__main__":
    # 使用示例
    result = execute_custom_task("列出所有EC2实例及其状态")
    print(result)
