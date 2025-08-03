[Read this document in English](README_en.md)

[Read this document in English](README_en.md)

# 分布式 MLOps 概览

本项目提供了一个基础设施设置，用于在 Kubernetes (K8s) 上使用 **Amazon EKS** 探索分布式机器学习 (ML) 工作负载。它包括用于配置基础设施的 Terraform 模块、用于构建自定义镜像的 Docker 配置，以及用于使用各种框架和调度器运行分布式训练作业的示例脚本。

## 项目概览

该项目分为三个主要目录：

1.  **[`infra/`](infra/)**: 包含用于配置 EKS 集群、GPU 支持和分布式训练工具的 Terraform 模块。
2.  **[`docker/`](docker/)**: 包含用于为 MPI、GPU 和分布式训练工作负载构建自定义镜像的 Docker 配置。
3.  **[`examples/`](examples)**: 提供用于运行分布式训练作业的示例脚本和配置，支持：
    *   **Ray**
    *   **Torchrun**
    *   **MPI**
    *   **Training Operator**
    *   **Volcano Scheduler**

## 目录结构

```
.
├── docker/                     # 用于自定义镜像的 Docker 配置
│   ├── g4ad_rocm_build/        # AMD GPU (ROCm) 构建
│   ├── g4dn_cuda_build/        # NVIDIA GPU (CUDA) 构建
│   └── standard_mpi_runner/    # 标准 MPI 运行器
├── examples/                   # 示例脚本和配置
│   ├── kuberay/                # Ray 集群示例
│   ├── local/                  # 本地分布式训练示例
│   ├── mpi/                    # 基于 MPI 的分布式训练示例
│   ├── ray/                    # 基于 Ray 的分布式训练示例
│   ├── torchrun/               # 基于 Torchrun 的分布式训练示例
│   ├── training_operator/      # Training Operator 示例
│   └── volcano_scheduler/      # Volcano 调度器示例
└── infra/                      # 用于 EKS 基础设施的 Terraform 模块
    ├── ec2/                    # EC2 实例配置
    └── eks/                    # EKS 集群配置
```

## 主要特性

-   **EKS 集群**: 完全托管的、支持 GPU 的 Kubernetes 集群。
-   **分布式训练**: 支持 Ray、Torchrun、MPI、Training Operator 和 Volcano 调度器。
-   **GPU 支持**: 为 NVIDIA (CUDA) 和 AMD (ROCm) GPU 提供配置。
-   **监控**: 集成 Prometheus 和 Grafana 进行集群监控。
-   **成本管理**: 使用 Kubecost 进行成本监控和优化。
-   **自动扩缩容**: 使用 Karpenter 进行动态节点配置。

## 先决条件

要使用此项目，您需要安装并配置以下工具：

1.  **Terraform** (>= 1.0): 用于配置基础设施。
2.  **AWS CLI**: 用于与 AWS 服务交互。
3.  **kubectl**: 用于管理 Kubernetes 集群。
4.  **Helm** (v3): 用于部署 Kubernetes 应用程序。
5.  **Python**: 用于运行分布式训练脚本。

## 快速入门

### 1. **设置 AWS CLI**

-   安装 AWS CLI：[AWS CLI 安装指南](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)。
-   配置您的 AWS 凭证：
    ```bash
    aws configure
    ```

### 2. **配置基础设施**

-   导航到 `infra/eks/` 目录。
-   初始化 Terraform：
    ```bash
    terraform init
    ```
-   应用 Terraform 配置：
    ```bash
    terraform apply
    ```

### 3. **配置 kubectl**

-   更新您的 `kubectl` 配置以连接到 EKS 集群：
    ```bash
    aws eks --region <区域> update-kubeconfig --name <集群名称>
    ```

### 4. **部署分布式训练作业**

-   导航到 `examples/` 目录，并选择您想使用的框架（例如 Ray、Torchrun、MPI、Training Operator 或 Volcano 调度器）。
-   按照每个子目录中的 README 说明来部署和运行分布式训练作业。

## 示例工作流

### 运行 Ray 作业

1.  导航到 `examples/ray/`。
2.  按照 `README.md` 中的说明部署 Ray 集群并提交作业。

### 使用 Volcano 调度器运行 MPI 作业

1.  导航到 `examples/volcano_scheduler/`。
2.  使用您的训练脚本更新 `multinode_ddp_volcano_mpi_job.yaml` 清单。
3.  提交作业：
    ```bash
    kubectl apply -f multinode_ddp_volcano_mpi_job.yaml
    ```

### 运行 Training Operator 作业

1.  导航到 `examples/training_operator/`。
2.  使用您的训练脚本更新作业清单（例如 `multinode_ddp_pytorch_job.yaml`）。
3.  提交作业：
    ```bash
    kubectl apply -f multinode_ddp_pytorch_job.yaml
    ```

## 监控与调试

-   **Grafana 仪表盘**:
    ```bash
    kubectl port-forward svc/kube-prometheus-stack-grafana 8080:80 -n kube-prometheus-stack
    ```
    在 `http://localhost:8080` 访问。

-   **Kubernetes 仪表盘**:
    ```bash
    kubectl port-forward service/kubernetes-dashboard-kong-proxy -n kubernetes-dashboard 8443:443
    ```
    在 `https://localhost:8443` 访问。

-   **作业状态**:
    使用 `kubectl get jobs` 或 `kubectl get pods` 来监控作业状态。

## 清理

要销毁基础设施并避免不必要的成本：

```bash
terraform destroy
```

## 参考资料

-   **Terraform 文档**: [https://www.terraform.io/docs/](https://www.terraform.io/docs/)
-   **AWS EKS 文档**: [https://docs.aws.amazon.com/eks/](https://docs.aws.amazon.com/eks/)
-   **Kubernetes 文档**: [https://kubernetes.io/docs/](https://kubernetes.io/docs/)
-   **Ray 文档**: [https://docs.ray.io/](https://docs.ray.io/)
-   **PyTorch 分布式训练**: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
