# 使用AWS CodeBuild构建剪枝NCF模型的容器镜像

本文档说明如何使用AWS CodeBuild将SageMaker Neo编译优化的剪枝NCF模型构建为容器镜像。

## 前提条件

1. 已经使用`pruned_ncf_neo_compile.py`对剪枝后的NCF模型进行Neo编译优化
2. 有AWS IAM权限来使用CodeBuild、ECR和访问S3存储桶
3. 已创建ECR仓库用于存储容器镜像

## 剪枝模型Neo编译流程

剪枝后的NCF模型需要特殊处理才能与SageMaker Neo兼容。我们使用`aws/pruned_ncf_neo_compile.py`脚本来完成这一过程：

1. 该脚本会逐步测试不同的模型结构，找到与Neo兼容的最佳模型结构
2. 它会创建多个简化版本的模型，包括：
   - 基本版本（仅Embedding + Linear层）
   - 带ReLU的版本
   - 完整MLP的版本
   - 带手动归一化的完整版本（替代F.normalize，解决Neo兼容性问题）
3. 每个模型版本都会被转换为TorchScript格式并尝试编译
4. 成功编译的模型会被保存在`output/models/pruned/`目录下

使用方法：
```bash
python -m aws.pruned_ncf_neo_compile
```

## 构建流程

1. 从S3下载Neo编译优化的模型
2. 构建Docker镜像并将模型嵌入镜像中
3. 将镜像推送到ECR仓库

## 配置CodeBuild项目

### 环境变量设置

在CodeBuild项目中，需要设置以下环境变量：

| 变量名 | 描述 | 示例值 |
|-------|------|-------|
| AWS_DEFAULT_REGION | AWS区域 | us-east-1 |
| AWS_ACCOUNT_ID | AWS账户ID | 123456789012 |
| IMAGE_REPO_NAME | ECR仓库名称 | ncf-recommender |
| IMAGE_TAG | 镜像标签 | latest |
| NEO_MODEL_S3_URI | Neo模型S3 URI (不包含s3://) | mingyu110/recommender-demo/compilation-output/job-name |

### 权限设置

确保CodeBuild服务角色有以下权限：

1. ECR权限：
   - ecr:GetAuthorizationToken
   - ecr:BatchCheckLayerAvailability
   - ecr:GetDownloadUrlForLayer
   - ecr:BatchGetImage
   - ecr:InitiateLayerUpload
   - ecr:UploadLayerPart
   - ecr:CompleteLayerUpload
   - ecr:PutImage

2. S3权限：
   - s3:GetObject
   - s3:ListBucket
   
### 通过控制台创建项目

1. 登录AWS控制台，导航至CodeBuild服务
2. 选择"创建构建项目"
3. 输入项目名称和描述
4. 来源选择您的代码仓库
5. 环境配置：
   - 选择托管镜像
   - 操作系统：Amazon Linux 2
   - 运行时：Standard
   - 镜像：aws/codebuild/amazonlinux2-x86_64-standard:3.0
   - 特权模式：启用（用于构建Docker镜像）
   - 服务角色：选择具有必要权限的角色
6. 添加前面列出的环境变量
7. 构建规范：使用存储库中的buildspec.yml
8. 构建完成时的工件：无需配置
9. 选择"创建构建项目"

## 手动触发构建

1. 导航到您创建的CodeBuild项目
2. 点击"开始构建"按钮
3. 保持默认设置并点击"开始构建"
4. 等待构建完成，检查构建日志以确认镜像是否成功构建和推送

## 使用构建好的镜像

构建完成后，您可以在Amazon ECR中找到包含Neo优化模型的Docker镜像。此镜像可用于：

1. 在SageMaker上部署为推理端点
2. 在Amazon ECS/EKS中部署
3. 在自托管环境中运行推理服务

## 故障排除

1. **找不到Neo模型**：确保NEO_MODEL_S3_URI环境变量正确指向S3中的Neo编译输出路径
2. **构建失败**：检查构建日志，确保IAM权限配置正确
3. **镜像不包含模型**：检查构建日志中的模型复制步骤，确认模型已正确下载
4. **Neo编译失败**：使用`pruned_ncf_neo_compile.py`脚本尝试不同的模型结构，找到与Neo兼容的版本 