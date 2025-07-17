# AI 开源项目与实践代码库

---

## 1. 仓库概述

本仓库 (`mingyu110/AI`) 是我个人的AI领域的项目、代码风格、二次开发经验的综合性代码库。每个子目录都代表一个独立的项目或一个特定的AI应用场景，旨在提供可复现、可学习的参考实现。

---

## 2. 项目（子目录）列表

以下是本仓库当前包含的项目及其功能简介：

### 2.1 `smart_inventory_agent/`

*   **功能**: 一个标准的AI Agent项目模板，用于演示如何构建、测试和部署生产级的AI Agent。
*   **技术栈**: Python, LangChain/LangGraph (示例), FastAPI (可选), Pytest (可选)。
*   **说明**: 此项目提供了一个模块化、可扩展的工程结构，是启动新AI Agent项目的理想模板。

### 2.2 `cloud-engineer-agent/`

*   **功能**: 一个利用Strands Agents SDK构建的智能AWS云工程师Agent。
*   **技术栈**: Python, Strands Agents SDK, AWS CDK。
*   **说明**: 该Agent旨在协助处理与AWS云环境相关的查询和操作，展示了如何将Agent技术应用于云基础设施管理。

### 2.3 `EV Analyzer AI Agent 20250414/`

*   **功能**: 基于CrewAI多智能体框架和DeepSeek大模型的新能源汽车（EV）行业智能分析助手。
*   **技术栈**: Python, CrewAI, DeepSeek LLM。
*   **说明**: 此项目通过模拟一个分析师团队（多个Agent协作），对新能源汽车行业的新闻和数据进行深度分析和报告生成。

### 2.4 `KuberAI/`

*   **功能**: 基于Spring AI Alibaba和通义千问大模型的Kubernetes资源智能优化系统。
*   **技术栈**: Java, Spring AI, Alibaba Tongyi Qwen LLM, Kubernetes。
*   **说明**: 该项目探索了如何使用AI大模型来分析Kubernetes集群的资源使用情况，并提供优化建议，以提高资源利用率和降低成本。

### 2.5 `recommender-neo/`

*   **功能**: 基于AWS SageMaker Neo的推荐系统模型优化与部署实践。
*   **技术栈**: Python, AWS SageMaker Neo, PyTorch/TensorFlow。
*   **说明**: 此项目展示了如何通过模型剪枝、微调和编译优化（使用SageMaker Neo）来提高推荐系统模型的推理性能和效率。

### 2.6 `pdf_tools/`

*   **功能**: 一个使用大模型将PDF文件批量转换为Markdown格式的实用工具。
*   **技术栈**: Python, PyMuPDF, OpenAI/Anthropic LLM。
*   **说明**: 该工具可以解析PDF的文本和布局，并利用大模型的理解能力生成结构良好、可读性强的Markdown文件。

### 2.7 `AI零代码智能数据分析决策助手.yml`

*   **功能**: 一个Dify工作流的DSL（领域特定语言）文件，定义了一个零代码的BI分析与决策助手。
*   **技术栈**: Dify, LLM。
*   **说明**: 该文件可以直接导入Dify平台，快速部署一个无需编程即可通过自然语言进行数据分析和获取决策建议的应用。

### 2.8 `Dify二次开发经验/`

*   **功能**: 记录了在将开源Dify平台（V1.3.0）集成到企业环境中，并进行二次开发与深度优化过程中的一系列技术实践与经验沉淀。
*   **技术栈**: Dify, Python, Milvus, Celery。
*   **说明**: 内容涵盖了HTTP节点增强、Celery机制解析、私有化向量数据库部署、存储持久化等企业级优化方案，是Dify二次开发的实用避坑指南。

---

### 2.9 `dify-on-dingtalk/`

*   **功能**: 一个将强大的Dify AI应用与钉钉（DingTalk）机器人连接的轻量级桥接服务。
*   **技术栈**: Python, Dify API, DingTalk Stream SDK, WebSocket。
*   **说明**: 该项目使得在Dify平台上构建的各类AI应用（如客服、HR助手、DevOps助手等）可以被无缝集成到企业钉钉中，提供带有“打字机”效果的流式响应，并支持多用户上下文管理。它也是一个优秀的参考实现，展示了如何将AI能力扩展到企业即时通讯工具（如企业微信、飞书）中。

---

## 3. 如何贡献

欢迎通过Pull Request的方式为本仓库贡献新的AI项目或改进现有项目。请确保每个新项目都以独立的子目录形式存在，并包含一个清晰的`README.md`文件以说明其用途和用法。