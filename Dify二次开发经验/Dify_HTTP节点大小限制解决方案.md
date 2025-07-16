### 技术文档：Dify v1.3.0 HTTP 请求节点大小限制问题分析与解决方案

#### 1. 问题概述

**问题场景：**
在 Dify v1.3.0 的私有化部署环境中，当工作流中的 "HTTP 请求节点" 用于处理文档时，如果响应体的大小超过 1MB，节点会执行失败。

**报错信息：**
`Text size is too large,max size is 1.00MB,but current size is 1.35MB`

**核心诉求：**
1.  分析此大小限制的来源。
2.  确定如何以安全、可靠的方式从根源上解决此问题。

---

#### 2. 根本原因分析

经过对源代码 (`api/core/workflow/nodes/http_request/executor.py`) 和 Kubernetes 部署文件 (`dify-deployment.yaml`) 的严肃分析，得出以下结论：

1.  **限制来源**: 大小限制的检查逻辑位于 `executor.py` 文件中，由 `HTTP_REQUEST_NODE_MAX_TEXT_SIZE` 和 `HTTP_REQUEST_NODE_MAX_BINARY_SIZE` 这两个环境变量控制。

2.  **配置方式**: 在 Kubernetes 环境中，这两个变量的值由一个名为 `dify-shared-config` 的 `ConfigMap` 统一提供给 `dify-api` 和 `dify-worker` 两个服务。

3.  **影响范围**: 由于 `dify-api`（处理同步任务）和 `dify-worker`（处理异步任务）都使用这份配置，因此必须确保两者都能获取到更新后的值。

**最终结论：**
要从根源上解决此问题，必须直接修改 `dify-shared-config` 这个 `ConfigMap`，并确保 `dify-api` 和 `dify-worker` 都已重启以加载新的配置。

---

#### 3. 关键环境变量详解

*   **`HTTP_REQUEST_NODE_MAX_TEXT_SIZE`**
    *   **用途：** 限制**文本类型**响应体的大小 (如 `application/json`, `text/html`)。
    *   **默认值 (问题所在)：** `1048576` (1MB)。

*   **`HTTP_REQUEST_NODE_MAX_BINARY_SIZE`**
    *   **用途：** 限制**二进制文件**响应体的大小 (如 `application/pdf`, `image/png`)。
    *   **默认值：** `10485760` (10MB)。

**调整目标：**
为确保能统一处理各类大文件，我们将把这两个参数的值都设置为 **10MB (`10485760`)**。

---

#### 4. 解决方案 (治本-推荐)

本方案采用 **“修改配置源头 + 重启应用服务”** 的两步法，是解决此类问题的最佳实践。它能确保配置的一致性，并通过 `kubectl` 命令保证操作的安全性和可重复性。

##### **步骤一：使用 Patch 命令更新共享配置 (ConfigMap)**

此命令会安全地更新 `dify-shared-config`，一次性将两个环境变量的值都修改为 10MB。请直接在终端中执行以下完整命令：

```bash
cat << 'EOF' | kubectl patch configmap dify-shared-config --namespace dify --type='merge' --patch-file=/dev/stdin
{
  "data": {
    "HTTP_REQUEST_NODE_MAX_TEXT_SIZE": "10485760",
    "HTTP_REQUEST_NODE_MAX_BINARY_SIZE": "10485760"
  }
}
EOF
```

*   **命令解析:**
    *   `kubectl patch configmap dify-shared-config -n dify`: 指定操作目标为 `dify` 命名空间下的 `dify-shared-config` `ConfigMap`。
    *   `--type='merge'`: 使用“合并”类型的补丁，可以方便地更新 `data` 字段下的一个或多个键值。
    *   `--patch-file=/dev/stdin`: 指示 `kubectl` 从标准输入（即 `cat` 命令的输出）读取补丁内容，避免了手动编辑文件可能引入的错误。

##### **步骤二：重启应用服务以加载新配置 (关键步骤)**

**注意：** `ConfigMap` 的变更不会自动应用到正在运行的 Pod。您必须通过以下命令触发滚动更新，强制 Pod 重启并加载最新的配置。

1.  **重启 `dify-api` 服务:**
    ```bash
    kubectl rollout restart statefulset dify-api -n dify
    ```

2.  **重启 `dify-worker` 服务:**
    ```bash
    kubectl rollout restart statefulset dify-worker -n dify
    ```

*   **命令解析:**
    *   `kubectl rollout restart statefulset <name> -n dify`: 此命令会安全地、逐一地替换 `StatefulSet` 管理的 Pod，确保服务在更新过程中不会中断。

---

#### 5. 验证

1.  执行完重启命令后，可以通过以下命令观察 Pod 是否正在重建：
    ```bash
    kubectl get pods -n dify -w
    ```
2.  等待 `dify-api` 和 `dify-worker` 的新 Pod 都进入 `Running` 状态且 `READY` 状态为 `1/1`。
3.  重新在 Dify 界面中运行之前失败的工作流，此时 HTTP 请求节点应该能够成功处理大于 1MB 的文档。