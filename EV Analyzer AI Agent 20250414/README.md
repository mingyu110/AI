# AI新能源汽车分析助手

这是一个基于CrewAI多智能体框架和DeepSeek大模型的新能源汽车行业分析系统，可以自动收集、分析新能源汽车相关新闻并生成详细的分析报告。

## 功能特点

- 使用Serper API搜索实时新能源汽车新闻
- 基于关键词收集相关新闻
- 使用CrewAI多智能体框架进行深度分析
- 由DeepSeek大模型提供强大的自然语言理解和生成能力
- 以Markdown格式输出分析报告
- 可自定义时间范围（1天到1年）和报告发送时间
- 支持邮件发送分析报告
- 全中文报告输出，自动翻译非中文内容
- 每次分析收集多达100条全球内容，确保视角多样性

## 分析内容

系统专注于新能源汽车行业，生成的分析报告包含以下内容：

1. **行业发展趋势分析**：市场规模、增长速度、未来预测
2. **主要厂商动向分析**：特斯拉、比亚迪、蔚来、小鹏、理想等的战略和动态
3. **消费者购买决策分析**：价格敏感度、续航里程需求、充电便利性等因素
4. **政策环境分析**：各国补贴政策、监管框架和发展规划
5. **技术发展分析**：电池技术、自动驾驶、充电基础设施等进展
6. **长安汽车专题分析**：深入剖析长安汽车在新能源领域的战略和表现

## DeepSeek优势

DeepSeek提供了强大的语言理解和分析能力，能够：
- 深入理解新能源汽车行业的专业术语和技术概念
- 识别关键趋势和市场变化信号
- 生成高质量、条理清晰的分析报告
- 提供有价值的战略见解和建议

## 使用方式

### 系统要求

- **Python版本**: 3.9+，推荐使用Python 3.12（注意：Python 3.13可能导致依赖兼容性问题）
- **操作系统**: Windows 10/11, macOS, Linux
- **Rust编译器**: 某些依赖（如tiktoken）需要系统安装Rust编译器

### 虚拟环境设置

强烈建议在虚拟环境中运行此程序，以避免依赖冲突。

#### 创建并激活虚拟环境 (Windows):
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate
```

#### 创建并激活虚拟环境 (macOS/Linux):
```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

### 安装Rust编译器（如需要）

某些Python包（如tiktoken）依赖于Rust编译器。如果安装过程中遇到相关错误，请安装Rust：

#### 安装Rust (所有平台):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装后重启终端或运行：
```bash
source "$HOME/.cargo/env"
```

### 网络配置说明

此程序使用Serper API进行全球内容搜索。如果您在中国大陆地区使用，可能需要：

- 确保您的网络环境能够正常访问国际互联网
- 配置合适的代理服务或VPN以确保API请求正常发送和接收
- 对于本地开发测试，可考虑使用开发代理工具转发API请求

**注意**: 程序不会自动配置网络环境，您需要确保系统能够正常访问Serper.dev和DeepSeek API。

### 依赖包内容：

程序依赖以下主要包：

```bash
streamlit
crewai>=0.11.2
crewai-tools>=0.0.1
python-dotenv==1.0.0
markdown==3.5.2
validators==0.22.0
schedule==1.2.1
langchain-openai>=0.0.3
langchain-core>=0.1.14
litellm>=1.10.0
```

各依赖包的用途：
1. **streamlit**: 构建交互式Web应用界面（未指定版本以避免依赖冲突）
2. **crewai**: 提供多智能体协作框架，用于创建专业化的AI代理
3. **crewai-tools**: CrewAI的工具库，提供搜索等功能
4. **python-dotenv**: 从.env文件加载环境变量
5. **markdown**: 处理Markdown格式内容
6. **validators**: 验证URL和电子邮件等输入
7. **schedule**: 提供定时任务调度功能
8. **langchain-openai**: LangChain框架的OpenAI集成
9. **langchain-core**: LangChain框架的核心组件
10. **litellm**: 统一多种大语言模型API的接口库，支持DeepSeek等多种模型

> **注意**：我们故意不为streamlit指定版本号，这是为了避免与crewai及其依赖的opentelemetry组件产生protobuf版本冲突。

### 安装依赖：

```bash
# 先升级pip以获取最新的依赖解析逻辑
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

如果安装过程中遇到问题，可以尝试：
```bash
# 对于tiktoken编译问题：
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install tiktoken
pip install -r requirements.txt
```

### 配置环境变量：

将`.env.example`复制为`.env`并填入您的API密钥和邮件配置：

```bash
cp .env.example .env
# 然后编辑.env文件填入您的配置
# 界面输入的授权码优先级高于.env文件配置
# 当界面未提供授权码时，系统自动使用.env中的默认配置
```

### 必需的API密钥

1. **DeepSeek API密钥**：用于自然语言处理和报告生成
   - 在[DeepSeek官网](https://platform.deepseek.com/)获取API密钥
   - 在`.env`文件中设置为`OPENAI_API_KEY`（DeepSeek兼容OpenAI接口）

2. **Serper API密钥**：用于实时新闻搜索
   - 在[Serper.dev](https://serper.dev/)注册账号并获取API密钥
   - 在`.env`文件中设置为`SERPER_API_KEY`

## 使用方法

启动Streamlit应用：

```bash
streamlit run app.py
```

## 系统演示

下面是系统运行的演示：

![EV分析系统演示](https://raw.githubusercontent.com/mingyu110/AI/main/EV%20Analyzer%20AI%20Agent%2020250414/demo.gif)

### 应用界面说明

在应用界面中，您可以：

1. 输入要分析的新能源汽车相关关键词（可选，如不填写则分析整个行业）
2. 指定接收分析报告的邮箱和邮箱密码
3. 选择新闻搜索的时间范围（从1天到1年，默认30天）
5. 使用快捷按钮选择常用时间范围（一周、一个月、三个月、一年）
5. 设置分析报告的发送时间（立即发送、今天结束时、明天早上或自定义时间）
6. 点击"开始DeepSeek新能源汽车分析"按钮启动分析流程

分析完成后，系统会预览分析报告，您可以直接下载或等待邮件发送。所有报告均以中文形式呈现，即使原始内容包含其他语言。

## 邮件发送说明

本系统提供多种配置邮件发送的方式：

1. **环境变量配置**：在`.env`文件中配置`EMAIL_HOST`、`EMAIL_PORT`、`EMAIL_USER`和`EMAIL_PASSWORD`
2. **界面输入**：在应用界面中直接输入邮箱密码（更加灵活，无需修改配置文件）
3. **一次性配置**：在界面中设置邮箱信息并选择"保存邮箱设置到配置文件"，系统会将设置保存到`.env`文件中，下次启动时自动加载

### 使用网易126邮箱

如果您使用网易126邮箱，需要进行以下设置：

1. **获取授权码**：
   - 登录126邮箱网页版 -> 设置 -> POP3/SMTP/IMAP -> 开启服务 -> 获取授权码
   - 在密码框中输入授权码而非登录密码

2. **SMTP设置**：
   - SMTP服务器：`smtp.126.com`
   - SMTP端口：`465`（SSL连接）或`25`（非SSL连接）

3. **持久化保存**：
   - 展开"邮箱服务器设置"高级选项
   - 勾选"保存邮箱设置到配置文件"
   - 点击提交按钮，设置将被保存到`.env`文件

保存后，您下次运行程序时将自动使用这些设置，无需再次输入邮箱信息。

为了安全起见，建议定期更换授权码，特别是在共享电脑上使用时。

## 系统架构

本系统基于以下技术：

- **DeepSeek**: 强大的大型语言模型，用于新能源汽车行业分析和报告生成
- **CrewAI**: 用于构建专业的多智能体协作框架
- **Serper API**: 用于实时搜索新能源汽车新闻数据
- **Streamlit**: 提供用户友好的界面
- **Python电子邮件库**: 用于发送分析报告
- **Markdown**: 用于格式化分析报告

### 技术架构图

下图展示了系统的整体技术架构和组件交互关系：

![EV Analyzer AI Agent 技术架构图](https://github.com/mingyu110/AI/raw/main/EV%20Analyzer%20AI%20Agent%2020250414/EV%20Analyzer%20AI%20Agent%20Architecture.png)

### 信息处理流程

系统的信息处理流程如下：

1. **内容搜索阶段**：
   - 使用Serper API搜索多个地区和语言的新能源汽车相关内容
   - 搜索不同类型的内容（新闻、博客、学术内容、讨论等）
   - 对搜索结果进行评分和排序，确保内容多样性
   - 获取多达100条高质量内容，覆盖全球13个不同地区和多种语言

2. **内容获取阶段**：
   - 通过爬虫访问Serper返回的URL，获取完整的内容
   - 使用多种User Agent和重试机制提高内容获取成功率
   - 清理和提取文本内容，去除广告和无关元素
   - 智能处理各种网站结构和内容格式

3. **内容分析阶段**：
   - 使用多智能体系统协作分析内容
   - 内容收集代理归纳信息来源和整体情况
   - 内容分析代理深入分析行业趋势、技术发展等
   - 自动翻译非中文内容，确保全部分析基于中文进行

4. **报告生成阶段**：
   - 报告撰写代理将分析结果整理为结构化报告
   - 使用Markdown格式增强报告可读性
   - 生成包含数据支持的深入分析和见解
   - 报告中融合全球多元化视角

5. **报告交付阶段**：
   - 通过用户界面预览分析报告
   - 通过邮件发送完整的分析报告
   - 提供报告下载功能

这种流程设计确保了系统能够从全球范围收集多元化的新能源汽车相关内容，并进行深入分析，最终生成高质量的分析报告。

### 系统智能体

系统包含以下专业智能体：

1. **新能源汽车内容收集专家**: 负责收集和整理全球新能源汽车相关的最新信息
2. **新能源汽车行业分析专家**: 负责深入分析新能源汽车行业趋势、技术发展和市场变化
3. **新能源汽车分析报告撰写专家**: 负责创建高质量、严谨专业且结构清晰的新能源汽车行业分析报告

所有智能体均由DeepSeek大模型驱动，确保了分析和报告的专业性和深度。

## 系统优化与性能

为了提供最佳的用户体验和分析质量，本系统进行了多方面的优化：

### 内容收集优化

- **内容多样性算法**：使用专门的评分机制确保收集内容的地域、语言和类型多样性
- **扩展搜索范围**：覆盖13个不同地区和10种语言，提供全球视角
- **多UA爬虫技术**：使用10种不同的User Agent和自动重试机制，提高内容获取成功率
- **关键词智能扩展**：自动扩展搜索关键词，确保搜索结果的全面性

### 分析性能优化

- **多智能体协作**：通过专业分工提高分析效率和深度
- **自动翻译集成**：无缝将多语言内容翻译为中文，确保分析的统一性
- **区域市场聚焦**：特别关注中国和全球主要市场的对比分析
- **热点趋势追踪**：自动识别和追踪行业热点技术和话题

### 系统资源优化

- **内容长度控制**：自动截断过长内容，避免资源浪费
- **异步处理设计**：使用异步模式处理内容获取，提高并行效率
- **缓存机制**：对频繁使用的内容实施缓存，提高响应速度

这些优化措施确保了系统能够高效地收集和分析最多100条全球内容，为用户提供全面、深入的新能源汽车行业分析。

## 自定义和扩展

您可以通过以下方式自定义和扩展本系统：

- 在`news_analyzer_crew.py`中修改`EV_KEYWORDS`列表，添加更多新能源汽车相关关键词
- 在`news_analyzer_crew.py`中调整智能体的定义和任务描述，改变分析重点
- 在`app.py`中添加更多预设关键词按钮
- 调整分析报告格式和内容结构
- 修改搜索参数，调整返回的内容数量
