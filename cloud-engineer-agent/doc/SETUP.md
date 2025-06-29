# 项目设置说明

## 环境变量配置

本项目需要配置以下环境变量：

### 必需的环境变量：

```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key_here"
export AWS_REGION="us-east-1"
```

### 可选的环境变量：

```bash
# 如果使用AWS配置文件
export AWS_PROFILE="your_profile_name"

# Bedrock模型配置（默认使用nova-premier）
export BEDROCK_MODEL_ID="us.amazon.nova-premier-v1:0"
```

## 快速设置

1. 在终端中设置环境变量：
```bash
# 方法1：直接设置
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_REGION="us-east-1"

# 方法2：使用AWS CLI配置
aws configure
```

2. 激活虚拟环境：
```bash
source venv/bin/activate
```

3. 运行应用：
```bash
streamlit run app.py
```

## AWS权限要求

确保您的AWS凭证具有以下权限：
- EC2: `ec2:DescribeInstances`, `ec2:DescribeSecurityGroups`
- S3: `s3:ListAllMyBuckets`
- CloudWatch: `cloudwatch:DescribeAlarms`
- IAM: `iam:ListUsers`, `iam:GetUser`
- Lambda: `lambda:ListFunctions`
- RDS: `rds:DescribeDBInstances`
- Bedrock: `bedrock:InvokeModel`

## MCP客户端配置和故障排除

### MCP服务器依赖

本项目使用AWS Model Context Protocol (MCP) 服务器：
- **AWS文档MCP服务器**: `awslabs.aws-documentation-mcp-server@latest`
- **AWS图表MCP服务器**: `awslabs.aws-diagram-mcp-server@latest`

### 常见问题和解决方案

#### 问题1：MCP客户端初始化超时
```
MCPClientInitializationError: background thread did not start in 30 seconds
```

**原因**：首次运行时，MCP服务器包需要下载和安装，可能超过30秒超时限制。

**解决方案**：
1. **预安装MCP服务器包（推荐）**：
   ```bash
   # 预安装AWS文档MCP服务器
   uvx awslabs.aws-documentation-mcp-server@latest --version
   
   # 预安装AWS图表MCP服务器  
   uvx awslabs.aws-diagram-mcp-server@latest --version
   ```

2. **检查网络连接**：确保网络连接稳定，能够访问PyPI。

3. **等待安装完成**：如果看到"Installed XX packages"消息，等待安装完成后重新启动应用。

#### 问题2：缺少uv工具
```
FileNotFoundError: [Errno 2] No such file or directory: 'uvx'
```

**解决方案**：
```bash
# 使用Homebrew安装uv
brew install uv

# 或使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 问题3：部分MCP服务器启动失败

如果某个MCP服务器启动失败，应用会显示警告信息但继续运行：
```
警告：AWS文档MCP客户端启动失败: ...
```

这种情况下，部分功能可能不可用，但基本的AWS CLI功能仍然可以使用。

### 验证MCP服务器状态

运行以下命令检查MCP服务器是否正确安装：
```bash
# 检查uv工具
uv --version

# 检查uvx命令
uvx --help

# 测试AWS文档MCP服务器
uvx awslabs.aws-documentation-mcp-server@latest --version

# 测试AWS图表MCP服务器
uvx awslabs.aws-diagram-mcp-server@latest --version
```

### 启动日志说明

正常启动时，您应该看到类似以下的日志：
```
正在初始化AWS文档MCP客户端...
启动AWS文档MCP客户端...
AWS文档MCP客户端启动成功，获得 X 个工具
正在初始化AWS图表MCP客户端...
启动AWS图表MCP客户端...
AWS图表MCP客户端启动成功，获得 X 个工具
```

### 最佳实践

1. **首次运行前预安装**：在第一次启动应用前，手动安装MCP服务器包
2. **网络环境**：确保网络连接稳定，特别是访问PyPI的能力
3. **系统资源**：MCP服务器启动需要一定的系统资源，确保有足够的内存和CPU
4. **错误处理**：应用已添加错误处理，单个MCP服务器失败不会阻止整个应用启动 