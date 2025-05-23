## KuberAI - 基于Spring AI Alibaba的Kubernetes资源智能优化系统

KuberAI是一款智能资源调度建议工具，基于应用负载特征自动推荐最优Pod资源配置。该项目使用[Spring AI Alibaba](https://github.com/alibaba/spring-ai-alibaba) 框架，通过阿里云通义千问大模型提供更智能的资源优化建议。
### 1. 功能特性

1. 智能资源调度建议：基于Pod历史负载特征自动推荐最优资源配置
2. AI驱动分析：集成Spring AI Alibaba和通义千问大模型提供更智能化的资源优化建议
3. 便捷的REST API：提供简单易用的Web接口，可轻松集成到现有系统中
4. 丰富的指标分析：使用P95百分位指标作为基准，考虑目标CPU和内存利用率，为资源请求和限制提供合理建议

### 2. 技术架构

1. Spring Boot & Spring AI Alibaba：提供框架支持和AI能力
2. 通义千问大模型：提供智能化的资源优化建议和配置方案
3. Fabric8 Kubernetes Client：与Kubernetes集群交互，获取Pod指标
4. 资源分析引擎：基于历史指标数据分析和预测资源需求

### 3. 快速开始

#### 3.1 环境要求

1. JDK 17或更高版本
2. Maven 3.6+
3. Kubernetes集群访问权限
4. 阿里云通义千问API密钥

#### 3.2 配置

在application.yml中配置通义千问API密钥：

```yaml
spring:
  ai:
    tongyi:
      api-key: ${TONGYI_API_KEY:你的通义千问API密钥}
      model: qwen-max
```

也可以通过环境变量设置API密钥：

```yaml
export TONGYI_API_KEY=your_api_key_here
```



#### 3.3 启动服务

```shell
mvn spring-boot:run
```



### 4. API使用

#### 4.1 获取命名空间下所有Pod的资源建议

```shell
GET /api/resources/namespace?namespace={namespace}
```

#### 4.2 获取特定Pod的资源建议

```shell
GET /api/resources/pod?namespace={namespace}&podName={podName}
```

#### 4.3 AI聊天接口

```shell
GET /chat?input={your_prompt}
```

### 5. 未来功能扩展

KuberAI计划在未来版本中添加以下功能特性：

#### 5.1 核心功能扩展

1. 自动化资源调整：基于AI建议自动执行Kubernetes资源调整，无需人工干预
2. 成本优化分析：结合云服务商价格模型，提供成本节约估算和优化建议
3. 异常检测与告警：识别资源使用异常模式并发送预警通知
4. 多集群管理：支持同时分析和优化多个Kubernetes集群的资源配置
5. 历史趋势分析：提供资源使用趋势分析和长期规划建议

#### 5.2 预测性能力

1. 负载预测：基于历史数据预测未来工作负载，提前调整资源配置，引入时间序列预测模型
2. 弹性伸缩策略推荐：针对不同应用特性，自动生成最佳HPA/VPA配置
3. 资源争用检测：识别集群中的资源争用情况，提供缓解方案

#### 5.3 可视化与报告

1. 资源优化仪表板：直观展示资源使用情况和优化潜力
2. 定期优化报告：自动生成周期性资源优化分析报告
3. 配置变更历史：记录和分析历史资源配置变更及其影响

#### 5.4 集成增强

1. CI/CD集成：与常见CI/CD流水线工具集成，在部署前验证资源配置
2. GitOps支持：生成资源配置PR，支持通过GitOps工作流实现变更
3. 服务网格集成：与Istio等服务网格解决方案集成，优化网络资源
