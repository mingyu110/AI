# MCP-Server：大语言模型智能工具服务器

---

## 1. 简介

`mcp-server` 是一个基于 Python FastAPI 构建的轻量级、高性能的“智能工具”服务器。它的核心设计思想是充当一个桥梁，将复杂的后端服务能力（例如代码静态分析、知识图谱查询等）或任何本地 Python 函数，封装成一系列结构清晰、定义明确的“工具”，并以API的形式暴露给上层应用，特别是大语言模型（LLM）驱动的智能代理（Agent）。

本项目旨在解决大语言模型无法直接与本地代码、私有服务或复杂环境交互的痛点，为构建强大的、具备实际“行动”能力的智能代理提供了坚实的基础设施。

## 2. 核心特性

*   **高性能**：基于 FastAPI 和 Uvicorn，提供异步、高并发的API服务。
*   **易于扩展**：通过一个简单的 `@mcp.tool()` 装饰器，即可轻松将任何 Python 函数封装成一个可被远程调用的工具。
*   **动态工具发现**：客户端连接时可自动获取所有可用工具的详细定义（Schema），便于LLM进行决策。
*   **专为LLM代理设计**：完美集成 LangChain/LangGraph 等主流LLM应用框架，是构建 ReAct 类型代理的理想选择。
*   **容器化支持**：提供 `Dockerfile`，支持使用 Docker 进行标准化部署。

## 3. 架构概览

`mcp-server` 在一个典型的LLM代理系统中扮演“工具层”的角色，其标准工作流如下：

1.  **LLM代理 (Agent)**：接收用户的自然语言指令。
2.  **MCP客户端 (Client)**：代理通过客户端连接到 `mcp-server`，并获取可用的工具列表。
3.  **MCP服务器 (Server)**：接收并执行来自客户端的工具调用请求。
4.  **后端服务/本地代码**：`mcp-server` 将请求路由到具体的工具函数，该函数可能与更深层的后端服务交互，或执行本地代码。
5.  **返回结果**：`mcp-server` 将函数的执行结果返回给代理，代理再根据结果进行下一步的思考或生成最终答案。

## 4. 快速开始

### 4.1 环境准备

*   Python 3.10+
*   `uv` 或 `pip` 包管理工具
*   Git

### 4.2 安装与运行

1.  **克隆代码库**
    ```bash
    git clone https://github.com/<your-githubid>/mcp-server.git
    cd mcp-server
    ```

2.  **安装依赖**
    推荐使用 `uv` 进行安装。它会读取 `pyproject.toml` 文件并创建虚拟环境。
    
    ```bash
    uv pip install -e . 
    ```
    或者，使用 `pip`：
    ```bash
    pip install -e . 
    ```
    
3.  **运行服务器**
    ```bash
    python mcp-server.py
    ```
    服务器默认将在 `localhost:8000` 启动。

### 4.3 客户端使用示例

以下是一个简单的Python脚本，演示了如何使用 `langchain_mcp_adapters` 连接到服务器并调用工具。

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    # 配置要启动的 mcp-server 实例
    server_configs = {
        "vision": { # 服务器名称
            "command": "python",
            "args": ["./mcp-server.py"], # 启动脚本的路径
            "transport": "stdio"
        }
    }

    # 使用 async with 启动并自动管理服务器进程
    async with MultiServerMCPClient(server_configs) as client:
        # 获取所有可用的工具
        tools = client.get_tools()
        print("可用的工具:", [tool.name for tool in tools])

        # 假设存在一个名为 "applications" 的工具
        # 注意：实际的工具调用通常由 LangChain/LangGraph 代理自动完成
        # 此处为手动演示
        vision_tool = next((t for t in tools if t.name == 'vision_applications'), None)
        if vision_tool:
            print("\n调用工具 'vision_applications'...")
            result = await vision_tool.ainvoke({})
            print("调用结果:")
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## 5. 如何定义新工具

您可以非常容易地在 `mcp-server.py` 中添加自己的工具。只需编写一个标准的 Python `async` 函数，并用 `@mcp.tool()` 装饰器标记它即可。

```python
# 在 mcp-server.py 中

from mcp.server.fastmcp import mcp, Context

@mcp.tool()
async def get_server_status(ctx: Context, service_name: str) -> dict:
    """获取指定服务的状态。

    Args:
        service_name (str): 需要查询状态的服务名称。
    """
    # 你的逻辑代码
    # ...
    status = {"service": service_name, "status": "running", "uptime": "12h"}
    return status

```

*   **函数签名**：函数的参数和类型提示会被自动解析为工具的输入 Schema。
*   **文档字符串 (Docstring)**：函数的文档字符串会成为工具的描述，这对于 LLM 理解工具的功能至关重要。

## 6. 通过 Docker 部署

项目已包含 `Dockerfile` 和 `docker-compose-test.yaml`，方便进行容器化部署。

1.  **构建 Docker 镜像**
    ```bash
    docker build -t mcp-server:latest .
    ```

2.  **使用 Docker Compose 运行**
    `docker-compose-test.yaml` 文件提供了一个运行示例。
    ```bash
    docker-compose -f docker-compose-test.yaml up
    ```

## 7. 配置

本服务的一些行为可以通过环境变量进行配置。主要的配置项在 `mcp-server.py` 的开头部分定义，例如：

*   `VISION_API_BASE`: 后端 Vision 服务的 URL。
*   `VISION_PAGE_SIZE`: API 请求的分页大小。
*   `MCP_PRINT`: 是否打印调试信息。

在启动服务或运行 Docker 容器时，可以按需设置这些环境变量。

