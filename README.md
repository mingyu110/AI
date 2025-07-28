# AI 开源项目与实践代码库

---

## 1. 仓库概述

本仓库 (`mingyu110/AI`) 是我个人的AI领域的项目、代码风格、二次开发经验的综合性代码库。每个子目录都代表一个独立的项目或一个特定的AI应用场景，旨在提供可复现、可学习的参考实现。

---

## 2. 项目列表

| 项目 (目录) | 核心功能与说明 | 主要技术栈 |
| :--- | :--- | :--- |
| [`smart_inventory_agent/`](./smart_inventory_agent/) | 标准的生产级AI Agent项目模板，用于演示如何构建、测试和部署。 | `Python`, `LangChain`, `FastAPI` |
| [`cloud-engineer-agent/`](./cloud-engineer-agent/) | 利用Strands Agents SDK构建的智能AWS云工程师Agent，协助处理云环境查询与操作。 | `Python`, `Strands Agents SDK`, `AWS CDK` |
| [`EV Analyzer AI Agent/`](./EV%20Analyzer%20AI%20Agent%2020250414/) | 基于CrewAI和DeepSeek的多智能体，用于新能源汽车行业的智能分析助手。 | `Python`, `CrewAI`, `DeepSeek LLM` |
| [`KuberAI/`](./KuberAI/) | 基于Spring AI和通义千问的Kubernetes资源智能优化系统。 | `Java`, `Spring AI`, `Kubernetes` |
| [`recommender-neo/`](./recommender-neo/) | 基于AWS SageMaker Neo的推荐系统模型优化与部署实践。 | `Python`, `SageMaker Neo`, `PyTorch` |
| [`pdf_tools/`](./pdf_tools/) | 使用大模型将PDF文件批量转换为Markdown格式的实用工具。 | `Python`, `PyMuPDF`, `OpenAI LLM` |
| [`AI零代码智能数据分析决策助手.yml`](./AI%E9%9B%B6%E4%BB%A3%E7%A0%81%E6%99%BA%E8%83%BD%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%86%B3%E7%AD%96%E5%8A%A9%E6%89%8B.yml) | Dify工作流的DSL文件，定义了一个零代码的BI分析与决策助手。 | `Dify`, `LLM` |
| [`Dify二次开发经验/`](./Dify%E4%BA%8C%E6%AC%A1%E5%BC%80%E5%8F%91%E7%BB%8F%E9%AA%8C/) | 将开源Dify平台集成到企业环境中的二次开发与深度优化技术实践。 | `Dify`, `Python`, `Milvus`, `Celery` |
| [`dify-on-dingtalk/`](./dify-on-dingtalk/) | 将Dify AI应用与钉钉机器人连接的轻量级桥接服务。 | `Python`, `Dify API`, `DingTalk SDK` |
| [`AI平台架构技术Q&A.md`](./AI%E5%B9%B3%E5%8F%B0%E6%9E%B6%E6%9E%84%E6%8A%80%E6%9C%AFQ%26A.md) | 面向AI平台/Agent开发工程师的Q&A文档，沉淀核心技术要点与架构思考。 | `MLOps`, `Kubernetes`, `PyTorch`, `RAG` |
| [`contextual-engineering-guide/`](./contextual-engineering-guide/) | 关于上下文工程的深度指南，演示如何使用LangChain/LangGraph构建和优化高级AI代理。 | `Python`, `LangChain`, `LangGraph` |
| [`Mistral-AI-from-Scratch/`](./Mistral-AI-from-Scratch/) | 从零开始、使用PyTorch实现的Mistral(7B)和Mixtral(8x7B MoE)模型。 | `Python`, `PyTorch`, `xformers` |
| [`kubernetes_stability_for_ml_platforms.md`](./kubernetes_stability_for_ml_platforms.md) | 深入探讨如何从平台工程和应用开发两个维度，保障机器学习平台在Kubernetes上的生产级稳定性。 | `Kubernetes`, `MLOps`, `SRE` |
| [`hybrid_cloud_ml_inference_platform_architecture.md`](./hybrid_cloud_ml_inference_platform_architecture.md) | 一份面向混合云的生产级机器学习推理平台架构演进方案，结合了KServe、Triton和Karmada。 | `KServe`, `Triton`, `Karmada`, `MLOps` |
| [`AI机器学习平台建设深度经验.md`](./AI机器学习平台建设深度经验.md) | 一份体系化的AI机器学习平台建设深度经验总结，全面覆盖了从宏观架构设计、MLOps自动化流程、高性能推理服务到团队构建的实践方法论。 | `MLOps`, `Kubernetes`, `Platform Engineering` |
| [`serverless_cold_start_optimization_for_ai_inference.md`](./serverless_cold_start_optimization_for_ai_inference.md) | 一份关于Serverless AI推理平台冷启动优化的深度技术方案，系统性地阐述了从基础设施到应用层的全链路优化策略。 | `Serverless`, `Cold Start`, `Knative`, `Fargate` |
| [`AI模型推理优化技术深度解析.md`](./AI模型推理优化技术深度解析.md) | 深入分析现代AI模型，特别是LLM的先进推理优化技术，从架构模式到引擎实现。 | `LLM`, `Inference`, `TensorRT`, `Triton`, `vLLM` |

---

## 3. 贡献指南

欢迎任何形式的贡献，无论是添加新的AI项目、改进现有代码，还是修复文档中的错误。请遵循以下准则：

1.  **Fork & Clone**: 首先，Fork本仓库，然后将你的Fork克隆到本地。
2.  **创建分支**: 为你的修改创建一个新的特性分支 (`git checkout -b feature/YourFeatureName`)。
3.  **提交更改**: 进行修改，并创建清晰、有意义的提交信息。
4.  **发起Pull Request**: 将你的分支推送到GitHub，并向本仓库的`main`分支发起一个Pull Request。

请确保你的代码遵循仓库中已有的风格，并为任何新项目或重要功能添加清晰的文档。

