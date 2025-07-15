## 使用 Strands Agents SDK 构建智能 AWS 云工程师Agent

### 1. Strands Agents SDK

Strands Agents SDK 是一个强大的开源工具，由AWS开源，适合构建智能 AI 代理，尤其在 AWS 生态系统中。它采用模型驱动的方法，开发者只需定义一个提示和一组工具，即可创建代理，然后在本地测试并部署到云端。  

主要功能：

- 模型支持：兼容多种模型提供商，如 Amazon Bedrock、Anthropic、Ollama、Meta 等，满足不同需求。 
- 工具集成：内置预构建工具，并支持自定义工具扩展，增强代理功能,而且可以通过简单装饰器支持自定义工具扩展。
- 部署灵活性：从本地开发到生产环境，支持多种架构，适合各种规模的应用。
- 高级功能：支持多代理系统、自动化代理、流式处理等复杂用例。
- MCP 支持：原生支持 Model Context Protocol (MCP) 服务器，允许代理访问成千上万的预构建工具，增强功能。

参考文档：

1. [Strands Agents SDK的文档](https://strandsagents.com/latest/)
2. [Strands Agents Python SDK的GitHub仓库](https://github.com/strands-agents/sdk-python)

### 2. MCP协议

MCP协议，即**Model Context Protocol（模型上下文协议）**，是由Anthropic公司于2024年底推出的开放协议，旨在实现大型语言模型（LLM）与外部数据源、工具和服务之间的无缝集成。协议使用JSON-RPC 2.0作为消息格式，支持Stdio和基于HTTP的SSE作为传输方式。消息生命周期包括初始化、运行和关闭，工作流程涉及上下文请求、集成和管理。

| 消息类型              | 结构                                                         |
| --------------------- | ------------------------------------------------------------ |
| Requests（请求）      | jsonrpc="2.0", id (string/number), method (string), params (key-value object) |
| Responses（响应）     | jsonrpc="2.0", id, result (key-value object), error (code: number, message: string, data: optional) |
| Notifications（通知） | jsonrpc="2.0", method (string), params (key-value object), 无需回复 |

### 3. AWS CDK

AWS Cloud Development Kit (AWS CDK) 是一种工具，让开发者可以用熟悉的编程语言（如 TypeScript、JavaScript、Python、Java、C# 和 Go）来定义云基础设施，而不是使用传统的 YAML 或 JSON 文件。它通过将基础设施作为代码（IaC）管理，使开发者能像写软件一样管理云资源。

### 4. 基于Strands Agents SDK构建AWS云架构师智能Agent

本次实践，使用Strands Agents SDK为基础，在AWS的云环境构建一个云架构师智能Agent。

系统架构如下：

<img src="/Users/liujinxun/Documents/Dev/cloud-engineer-agent/image/1.jpg" alt="1" style="zoom:80%;" />



本架构的交互流程：

| 阶段     | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| 用户交互 | 用户通过 Streamlit UI 提交请求，发送 HTTP 请求到工作流服务   |
| 任务处理 | 工作流服务分派任务到 EC2 Fargate，执行容器化任务，进行数据处理 |
| 智能分析 | Strands Agent 进行推理与决策、工具调用、算法推理，生成结果反馈 |
| 结果返回 | 最终生成的结果返回响应给用户                                 |

本架构的核心组件

| 核心组件      | 功能                                     |
| ------------- | ---------------------------------------- |
| ECS Fargate   | 无服务器容器计算，管理任务执行和数据处理 |
| AWS Bedrock   | 托管的智能体研发平台以及模型托管市场     |
| Streamlit UI  | 用户交互界面，接收和发送请求             |
| Strands Agent | AI 代理系统，处理特征提取和算法推理      |
| 工作流服务    | 协调数据流，连接用户和后端服务           |

### 5. 核心功能

AWS云架构师智能Agent的核心功能：

- 监控和分析跨多个 AWS 服务的资源，例如 EC2、S3、EBS 等。  

- 识别安全漏洞并推荐最佳实践，例如发现未加密的 S3 桶或未更新的 IAM 策略。  

- 寻找节省成本的机会，例如识别未使用的 EBS 卷或低利用率的 EC2 实例。  

- 使用 AWS MCP 工具从文本描述生成 AWS 架构图，简化架构设计。  

- 使用 AWS MCP 工具搜索 AWS 文档中的相关信息，快速获取技术支持。  

- 直接通过接口执行 AWS CLI 命令，减少手动操作复杂度。

### 6. 源代码解释

项目的核心代码：

1. AWS文档MCP服务器

   功能：提供对AWS文档的智能搜索和访问能力，使Agent能够检索有关AWS服务、功能和最佳实践的准确信息。

   实现细节：

   - 使用`awslabs.aws-documentation-mcp-server`包通过`uvx`命令启动
   - 通过`MCPClient`类创建客户端连接
   - 使用`stdio_client`和`StdioServerParameters`配置服务器参数
   - 通过`list_tools_sync()`方法获取所有可用的文档查询工具

   作用：

   - 当用户询问关于AWS服务的问题时，Agent可以直接查询官方文档
   - 提供准确的AWS服务配置、限制和最佳实践信息
   - 减少代理依赖预训练知识，确保信息的准确性和时效性

2. AWS图表MCP服务器

   功能：使代理能够根据文本描述生成可视化的AWS架构图，帮助用户理解复杂的云架构。

   实现细节：

   - 使用`awslabs.aws-diagram-mcp-server`包通过uvx命令启动
   - 同样通过MCPClient类创建客户端连接
   - 图表生成后保存在`/tmp/generated-diagrams/`目录
   - 在`app.py`中通过`display_message_with_images`函数显示生成的图表

   作用：

   - 将用户的文本描述转换为专业的AWS架构图
   - 支持可视化现有架构或设计新架构
   - 在Streamlit界面中直接展示生成的图表

3. 资源清理

   - 代码中还包含了适当的资源清理机制，确保MCP客户端在应用退出时正确关闭：

     ```python
     # 注册清理处理程序
     
     def cleanup():
         try:
             aws_docs_mcp_client.stop()
             print("AWS Documentation MCP client stopped")
         except Exception as e:
             print(f"Error stopping AWS Documentation MCP client: {e}")
         
         try:
             aws_diagram_mcp_client.stop()
             print("AWS Diagram MCP client stopped")
         except Exception as e:
             print(f"Error stopping AWS Diagram MCP client: {e}")
     
     atexit.register(cleanup)
     ```

4. Strands Agent SDK是本项目的核心框架，它提供了构建智能代理的基础。以下是项目中使用Strands Agent SDK实现的主要功能及其关键代码：

   - 代理初始化与配置

     ```python
     from strands import Agent
     from strands.models import BedrockModel
     from strands.tools.mcp import MCPClient
     from strands_tools import use_aws
     
     # 创建 Bedrock 模型实例
     
     bedrock_model = BedrockModel(
         model_id="us.amazon.nova-premier-v1:0",  # 使用Amazon Nova Premier模型
         region_name=os.environ.get("AWS_REGION", "us-east-1"),
         temperature=0.1,  # 低温度值确保更确定性的输出
     )
     
     # 系统提示定义代理的角色和能力
     
     system_prompt = """
     你是一个专家级AWS云工程师助手。你的工作是帮助管理AWS基础设施，
     优化，安全和最佳实践。你可以：
     ...
     """
     
     # 创建代理实例，集成所有工具和模型
     
     agent = Agent(
         tools=[use_aws] + docs_tools + diagram_tools,  # 组合AWS CLI和MCP工具
         model=bedrock_model,
         system_prompt=system_prompt
     )
     ```

   - 工具集成机制

     AWS CLI工具集成

     ```python
     from strands_tools import use_aws
     
     # use_aws工具直接从strands_tools导入
     # 该工具允许代理执行AWS CLI命令
     ```

     MCP工具集成

     ```python
     from strands.tools.mcp import MCPClient
     from mcp import StdioServerParameters, stdio_client
     
     # 设置AWS文档MCP客户端
     aws_docs_mcp_client = MCPClient(lambda: stdio_client(
         StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
     ))
     aws_docs_mcp_client.start()
     
     # 获取MCP工具
     docs_tools = aws_docs_mcp_client.list_tools_sync()
     ```

   - Agent任务执行

     ```python
     # 执行自定义任务的函数
     def execute_custom_task(task_description: str) -> str:
         """执行基于描述的自定义云工程任务"""
         try:
             # 调用agent实例，传入任务描述
             response = agent(task_description)
             
             # 处理AgentResult对象
             if hasattr(response, 'message'):
                 return response.message
             
             # 处理其他类型的响应
             return str(response)
         except Exception as e:
             return f"Error executing task: {str(e)}"
     ```

   - 预定义任务系统

     ```python
     # 定义常见云工程任务
     PREDEFINED_TASKS = {
         "ec2_status": "List all EC2 instances and their status",
         "s3_buckets": "List all S3 buckets and their creation dates",
         "cloudwatch_alarms": "Check for any CloudWatch alarms in ALARM state",
         # ...更多预定义任务
     }
     
     # 执行预定义任务的函数
     def execute_predefined_task(task_key: str) -> str:
         """执行预定义的云工程任务"""
         if task_key not in PREDEFINED_TASKS:
             return f"Error: Task '{task_key}' not found in predefined tasks."
         
         task_description = PREDEFINED_TASKS[task_key]
         return execute_custom_task(task_description)
     ```

   - Agent的代理循环架构实现

     ```python
     Strands Agent SDK的核心是"代理循环"架构，这在代码中并不直接可见，但在SDK内部实现。当调用agent(task_description)时，SDK会：
     将用户请求发送给LLM
     LLM分析请求并确定最佳行动方案
     SDK执行LLM指示的工具调用
     结果返回给LLM进行进一步分析
     重复此过程直到任务完成
     这种架构使代理能够处理复杂的多步骤任务，而无需开发者显式编写决策逻辑。
     ```

   - 资源管理与清理

     这段代码展示了如何管理Strands Agent SDK创建的资源，确保在程序退出时正确清理。

     ```python
     import atexit
     
     # 注册清理处理函数
     def cleanup():
         try:
             aws_docs_mcp_client.stop()
             print("AWS Documentation MCP client stopped")
         except Exception as e:
             print(f"Error stopping AWS Documentation MCP client: {e}")
         
         try:
             aws_diagram_mcp_client.stop()
             print("AWS Diagram MCP client stopped")
         except Exception as e:
             print(f"Error stopping AWS Diagram MCP client: {e}")
     
     atexit.register(cleanup)
     ```

   - 与Streamlit集成

     这段代码展示了如何将Strands Agent SDK与Streamlit界面集成，创建交互式聊天体验。

     ```python
     # 在app.py中
     # 处理用户输入并生成响应
     if prompt := st.chat_input("Ask me about AWS..."):
         st.session_state.messages.append({"role": "user", "content": prompt})
         
         # 生成响应
         get_agent_functions()  # 确保代理已缓存
         with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                 response = execute_custom_task(prompt)
                 cleaned_response = clean_response(response)
                 display_message_with_images(cleaned_response)
         
         # 将助手响应添加到聊天历史
         st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
     ```

   - AWS CDK构建云基础设施资源

     项目结构：使用TypeScript编写的AWS CDK应用程序位于`cloud-engineer-agent-cdk`目录

     - 主要堆栈定义在`lib/cloud-engineer-agent-stack.ts`文件中

     - 入口点在`bin/cloud-engineer-agent-stack.ts`文件中

### 7. 演示效果

