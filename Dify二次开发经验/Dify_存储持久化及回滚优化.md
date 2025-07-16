# Dify API & Worker 持久化存储改造及数据流分析技术文档

---

## 1. 概述

本文档旨在为 Dify v1.3.0 的 `dify-api` 和 `dify-worker` 服务提供一个从默认的 `hostPath` 持久化存储迁移到对象存储（如 S3、MinIO）的详细技术方案，并包含一套专为在服务器节点直接操作而设计的完整回滚计划。

**核心目标**:
1.  满足大规模（最高 1TB 级别）文件存储的需求。
2.  将文件存储与计算节点解耦，实现 `dify-api` 和 `dify-worker` 服务的无状态化，提升系统的可扩展性和可靠性。
3.  详细分析并对比改造前后，从用户上传文件到向量化数据存入外部 Milvus 数据库的全流程。
4.  提供详细的备份与回滚步骤，确保改造过程的安全性。

**当前环境**:

- **向量数据库**: Milvus，部署在 Kubernetes 集群外部的独立服务器上，可通过网络访问。
- **其他有状态服务**: `dify-postgres`, `dify-redis`, `dify-weaviate` 的存储方案本次**暂不改造**。

---

## 2. 背景与动机

Dify 默认的 `hostPath` 存储方案将文件直接保存在 Kubernetes 节点的本地磁盘上。此方案存在以下核心痛点：

- **存储容量限制**: 节点本地磁盘容量有限，无法满足 TB 级别的存储需求。
- **单点故障**: 节点故障将导致其上存储的所有文件不可访问，服务可靠性低。
- **扩展性差**: `dify-api` 和 `dify-worker` Pod 被绑定在特定节点，无法自由调度或水平扩展。

改用对象存储是解决以上问题的最佳实践。对象存储提供近乎无限的扩展能力、极高的数据持久性，并能将存储与计算彻底分离。

---

## 3. 改造方案

### 3.1. 目标

- 将 `dify-api` 和 `dify-worker` 的文件存储后端从本地文件系统切换为对象存储。
- 将 `dify-api` 和 `dify-worker` 的 Kubernetes 资源类型从 `StatefulSet` 变更为 `Deployment`，以实现真正的无状态化。

### 3.2. 实施步骤

1.  **配置对象存储参数**:
    在 `dify-deployment.yaml` 的 `dify-shared-config` `ConfigMap` 中，修改存储相关配置。以兼容 S3 的 MinIO 为例：

    ```yaml
    # ... ConfigMap: dify-shared-config ...
    data:
      # 1. 修改 STORAGE_TYPE
      STORAGE_TYPE: s3

      # 2. 配置 S3 相关参数
      S3_ENDPOINT: 'http://your-minio-service.namespace:9000' # MinIO 的 K8s Service 地址
      S3_BUCKET_NAME: 'dify-knowledge-base' # 事先创建好的 Bucket 名称
      S3_ACCESS_KEY: 'your-minio-access-key'
      S3_SECRET_KEY: 'your-minio-secret-key'
      S3_REGION: 'us-east-1' # 对于 MinIO，region 可随意填写

      # 3. (可选) 注释或删除旧的本地文件系统配置
      # OPENDAL_SCHEME: fs
      # OPENDAL_FS_ROOT: storage
      # ...
    ```

2.  **修改 `dify-api` 和 `dify-worker` 的部署模式**:
    由于文件状态已外置到对象存储，这两个服务不再需要稳定的本地存储，可以从 `StatefulSet` 转换为更灵活的 `Deployment`。

    - **移除 `hostPath` 卷**: 在 `dify-api` 和 `dify-worker` 的定义中，删除 `volumes` 和 `volumeMounts` 中关于 `dify-api-storage` 的部分。
    - **修改资源类型**: 将 `kind: StatefulSet` 修改为 `kind: Deployment`。同时，删除 `StatefulSet` 特有的字段，如 `serviceName`。

3.  **数据迁移与部署**:
    - **重要：执行迁移前，必须备份 `hostPath` 目录 (`/root/dify/app/api/storage`) 的所有内容，以备回滚。**
    - **迁移**: 将旧 `hostPath` 目录中的存量文件，使用 S3 客户端（如 `mc` 或 `aws-cli`）上传到新的对象存储 Bucket 中。
    - **部署**: 应用修改后的 `dify-deployment.yaml` 文件。

---

## 4. 数据全流程对比分析

| 流程阶段 | HostPath 模式 (改造前) | 对象存储模式 (改造后) |
| :--- | :--- | :--- |
| **1. 用户上传文件** | 用户通过浏览器将文件上传到 Dify 前端，前端通过 Nginx 将请求转发至 `dify-api` Pod。 | **流程不变。** 用户通过浏览器将文件上传到 Dify 前端，前端通过 Nginx 将请求转发至 `dify-api` Pod。 |
| **2. 文件接收与存储** | `dify-api` 接收到文件流，并将其直接写入到挂载的 `hostPath` 卷中，即写入到 Pod 所在节点的本地磁盘路径（如 `/root/dify/app/api/storage`）。 | `dify-api` 接收到文件流，**通过 S3 SDK 将文件流直接上传到对象存储（如 MinIO）的指定 Bucket 中**。文件不再写入本地磁盘。 |
| **3. 文件处理任务触发** | `dify-api` 在文件写入成功后，向 Redis 消息队列中推送一个文件处理任务，任务信息包含文件的本地路径。 | **流程不变。** `dify-api` 在文件上传成功后，向 Redis 消息队列中推送一个文件处理任务，但任务信息包含的是文件在**对象存储中的唯一标识（Key）**。 |
| **4. 文件读取与切片** | `dify-worker` Pod 从 Redis 中获取任务，根据任务中的文件路径，**从共享的 `hostPath` 卷中读取本地文件**。读取后，在内存中对文件内容进行切片（Chunking）。 | `dify-worker` Pod 从 Redis 中获取任务，根据任务中的对象存储 Key，**通过 S3 SDK 从对象存储中下载或流式读取文件**。读取后，在内存中对文件内容进行切片。 |
| **5. 文本向量化** | `dify-worker` 将切片后的文本块（Chunks）送入嵌入模型（Embedding Model），在内存中计算出每个文本块的向量（Vector）。 | **流程不变。** `dify-worker` 将切片后的文本块送入嵌入模型，在内存中计算出每个文本块的向量。 |
| **6. 向量数据存储** | `dify-worker` 通过网络连接到**外部的 Milvus 数据库**，调用 Milvus 的 API，将生成的向量数据批量插入到指定的 Collection 中。 | **流程不变。** `dify-worker` 通过网络连接到**外部的 Milvus 数据库**，调用 Milvus 的 API，将生成的向量数据批量插入到指定的 Collection 中。 |

---

## 5. 回滚方案 (Rollback Plan)

本方案专为在 Kubernetes 节点上直接操作的场景设计，通过备份原始部署文件和 `hostPath` 目录，确保在改造失败时能够安全、完整地恢复系统。

### 5.1. 回滚触发条件

在以下情况，应考虑启动回滚方案：
-   新配置的 `dify-api` 或 `dify-worker` Pod 无法正常启动或持续崩溃。
-   应用改造后的配置后，文件上传或知识库处理功能出现严重错误。
-   访问对象存储的延迟过高，导致系统性能无法接受。
-   数据一致性出现问题，对象存储中的文件与数据库记录不匹配。

### 5.2. 关键准备工作 (执行改造前必须完成)

在对系统进行任何修改之前，必须完成以下备份步骤。

#### 步骤 1: 查找 `dify-api` Pod 所在的节点

由于 `hostPath` 存储在特定节点上，您需要先找到这个节点。
```bash
# 执行此命令查看 dify-api Pod 运行在哪个 NODE 上
kubectl get pod dify-api-0 -n dify -o wide
```
> 记下输出中的 `NODE` 名称。后续的数据备份和恢复操作都需要登录到该节点上执行。

#### 步骤 2: 备份 `hostPath` 数据 (在目标节点上执行)

通过 SSH 或其他方式登录到上一步找到的节点，然后执行以下命令备份整个存储目录。
```bash
# 使用 tar 命令将存储目录打包并压缩，文件名包含当前日期和时间
# 请确保您有足够的权限和磁盘空间来创建备份文件
tar -czvf /root/dify_storage_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /root/dify/app/api/ storage
```
- **命令解释**:
  - `tar -czvf`: 创建 (`c`)、使用 gzip 压缩 (`z`)、显示过程 (`v`)、指定归档文件名 (`f`)。
  - `/root/dify_storage_backup_...tar.gz`: 备份文件的完整路径和名称。建议放在 `/root` 或其他安全目录下。
  - `-C /root/dify/app/api/`: **切换到** `storage` 目录的父目录，这可以避免在归档中包含完整的绝对路径，便于恢复。
  - `storage`: 需要备份的目标目录名。

> **验证**: 备份完成后，请检查备份文件是否存在且大小合理。

#### 步骤 3: 备份 Kubernetes 配置 YAML 文件

将当前有效的 `StatefulSet` 和 `ConfigMap` 配置导出为 YAML 文件，作为配置的“快照”。
```bash
# 创建一个用于存放备份文件的目录
mkdir -p /root/dify_k8s_backup

# 备份 dify-api 的 StatefulSet
kubectl get statefulset dify-api -n dify -o yaml > /root/dify_k8s_backup/dify-api-sts.yaml

# 备份 dify-worker 的 StatefulSet
kubectl get statefulset dify-worker -n dify -o yaml > /root/dify_k8s_backup/dify-worker-sts.yaml

# 备份共享配置文件
kubectl get configmap dify-shared-config -n dify -o yaml > /root/dify_k8s_backup/dify-shared-config.yaml
```
> 现在，您拥有了完整的代码和配置备份，可以安全地开始执行改造。

### 5.3. 回滚执行步骤

当确定需要回滚时，请严格按照以下流程操作。

#### 步骤 1: 暂停服务 (隔离系统)

如果您的改造是基于 `Deployment` 的，请将其副本数缩减为 0。
```bash
# 假设您已将服务修改为 Deployment
kubectl scale deployment dify-api --replicas=0 -n dify
kubectl scale deployment dify-worker --replicas=0 -n dify

# 如果服务仍在运行但已损坏，需要强制删除
kubectl delete deployment dify-api dify-worker -n dify
```

#### 步骤 2: 恢复 Kubernetes 配置

使用您在准备阶段备份的 YAML 文件来恢复原始的 `StatefulSet` 和 `ConfigMap`。
```bash
# 依次应用备份的配置文件
kubectl apply -f /root/dify_k8s_backup/dify-shared-config.yaml
kubectl apply -f /root/dify_k8s_backup/dify-api-sts.yaml
kubectl apply -f /root/dify_k8s_backup/dify-worker-sts.yaml
```
> 此操作会删除改造后的 `Deployment`（如果存在）并重新创建原始的 `StatefulSet`。

#### 步骤 3: 恢复 `hostPath` 数据 (在目标节点上执行)

1.  **清空当前目录 (可选但推荐)**:
    为避免文件冲突，可以先将当前 `hostPath` 目录的内容移走。
    ```bash
    # 登录到 dify-api-0 原本所在的节点
    mv /root/dify/app/api/storage /root/dify/app/api/storage_broken
    mkdir /root/dify/app/api/storage
    ```

2.  **从备份恢复**:
    使用 `tar` 命令解压您之前创建的备份文件。
    ```bash
    # 恢复备份到原始位置
    tar -xzvf /root/dify_storage_backup_YYYYMMDD_HHMMSS.tar.gz -C /root/dify/app/api/
    ```

3.  **同步增量数据**:
    将在对象存储中新增的文件同步回本地。
    ```bash
    # 使用 S3 客户端将对象存储的数据同步回 hostPath 目录
    # 这会补充在改造期间上传的文件
    aws s3 sync s3://dify-knowledge-base /root/dify/app/api/storage
    ```

#### 步骤 4: 验证服务

1.  **检查 Pod 状态**:
    等待 `dify-api-0` 和 `dify-worker-0` Pod 重新创建并进入 `Running` 状态。
    
    ```bash
    kubectl get pods -n dify -w
    ```
```
    
2.  **功能验证**:
    -   检查 Pod 日志，确认无存储错误。
    -   登录 Dify UI，确认所有知识库文件（包括改造前和改造期间上传的）均可访问。
    -   执行一次新的文件上传，并到节点上的 `/root/dify/app/api/storage` 目录中确认新文件已成功写入。

```