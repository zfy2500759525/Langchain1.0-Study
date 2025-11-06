# LangChain 1.0 学习仓库

这是一个系统学习 **LangChain 1.0** 的实践仓库，涵盖从基础概念到实战项目的完整学习路径。

## 📚 关于 LangChain 1.0

LangChain 1.0 是用于构建 LLM 驱动应用程序的框架的最新主要版本（2024年9月发布）。主要特性：

- ✅ **构建在 LangGraph 运行时之上** - 提供持久化、流式处理、人在回路等能力
- ✅ **新的 `create_agent` API** - 简化 Agent 创建流程
- ✅ **中间件架构** - 提供细粒度的执行控制（before_model、after_model、wrap_model_call 等）
- ✅ **多模态支持** - 处理文本、图像、视频、文件
- ✅ **结构化输出** - 使用 Pydantic 模型定义输出格式
- ✅ **语义化版本控制** - 1.x 系列保证 API 稳定

## 🚀 快速开始

### 环境要求

- Python 3.10 或更高版本（不支持 Python 3.9）
- pip 或 uv 包管理器

### 安装步骤

1. **克隆仓库**
```bash
git clone <your-repo-url>
cd langchain_v1_study
```

2. **创建虚拟环境**
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Keys
```

需要的 API Keys：
- `OPENAI_API_KEY` - OpenAI API 密钥（https://platform.openai.com/api-keys）
- `ANTHROPIC_API_KEY` - Anthropic API 密钥（https://console.anthropic.com/）
- `LANGSMITH_API_KEY` - LangSmith API 密钥（可选，用于可观测性）

### 验证安装

运行第一个示例：
```bash
python phase1_fundamentals/01_hello_langchain/main.py
```

## 📖 学习路径

本仓库采用**四阶段渐进式学习**，共 24 个模块 + 3 个综合项目：

### 阶段一：基础知识（第1-2周）
📂 `phase1_fundamentals/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 01 | Hello LangChain | 第一次 LLM 调用，使用 `init_chat_model` |
| 02 | Prompt Templates | 创建和使用提示词模板 |
| 03 | Messages | 消息类型（System、Human、AI） |
| 04 | Custom Tools | 自定义工具（天气、计算器、搜索） |
| 05 | Simple Agent | 使用 `create_agent` 创建 Agent |
| 06 | Agent Loop | 理解 ReAct 模式执行循环 |

### 阶段二：中级特性（第3-4周）
📂 `phase2_intermediate/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 07 | Memory Basics | 使用 InMemorySaver 实现短期内存 |
| 08 | Context Management | 消息修剪和摘要 |
| 09 | Checkpointing | 使用 SQLite/Postgres 持久化状态 |
| 10 | Middleware Basics | before_model 和 after_model 钩子 |
| 11 | Middleware Monitoring | 可观测性中间件 |
| 12 | Middleware Guardrails | PII 脱敏和输入验证 |
| 13 | Structured Output | 使用 Pydantic 定义输出模式 |
| 14 | Validation Retry | 优雅地处理验证失败 |
| 15 | Multi-Tool Structured | 结合工具和结构化输出 |

### 阶段三：高级主题（第5-6周）
📂 `phase3_advanced/`

| 模块 | 主题 | 学习内容 |
|------|------|----------|
| 16 | LangGraph Basics | 创建带节点和边的状态图 |
| 17 | Multi-Agent | 协调多个专业化 Agent |
| 18 | Conditional Routing | 实现动态工作流路由 |
| 19 | Image Input | 使用视觉模型处理图像 |
| 20 | File Handling | 处理文档上传和分析 |
| 21 | Mixed Modality | 结合文本、图像和结构化数据 |
| 22 | LangSmith Integration | 设置追踪和监控 |
| 23 | Error Handling | 实现健壮的错误恢复 |
| 24 | Cost Optimization | 追踪 token 使用并优化 |

### 阶段四：实际应用（第7-8周）
📂 `phase4_projects/`

| 项目 | 描述 | 核心技术 |
|------|------|----------|
| RAG 文档问答系统 | 基于向量数据库的文档问答 | 文档加载、向量存储、检索增强生成 |
| 多 Agent 客户支持 | 智能客服系统 | 多 Agent 协作、HITL、对话内存 |
| 研究助手 | 带工具的研究助手 | 网页搜索、MCP 集成、引用格式化 |

## 📁 项目结构

```
langchain_v1_study/
├── phase1_fundamentals/     # 阶段一：基础知识
├── phase2_intermediate/     # 阶段二：中级特性
├── phase3_advanced/         # 阶段三：高级主题
├── phase4_projects/         # 阶段四：综合项目
├── shared/                  # 共享资源（工具、提示词、中间件）
├── notebooks/               # Jupyter 笔记本实验
├── docs/                    # 学习笔记和文档
└── tests/                   # 全局测试
```

详细结构请查看 [CLAUDE.md](./CLAUDE.md)

## 🎯 使用指南

### 运行单个模块

```bash
# 进入模块目录
cd phase1_fundamentals/01_hello_langchain

# 运行主程序
python main.py

# 运行测试（如果有）
python test.py
```

### 运行综合项目

```bash
# 进入项目目录
cd phase4_projects/01_rag_system

# 安装项目特定依赖
pip install -r requirements.txt

# 运行项目
python main.py
```

### 使用 Jupyter Notebook

```bash
# 安装 Jupyter
pip install jupyter

# 启动 Notebook
jupyter notebook notebooks/
```

## 📝 学习建议

1. **按顺序学习** - 从阶段一开始，每个模块都基于前面的知识
2. **动手实践** - 每个模块都有可运行的代码，修改参数观察效果
3. **记录笔记** - 在 `docs/learning_notes/` 中记录你的学习心得
4. **查看 README** - 每个模块都有独立的 README.md 说明核心概念
5. **完成测试** - 运行测试文件验证你的理解
6. **做综合项目** - 前三个阶段完成后，通过项目巩固所学

## 🔧 常用命令

```bash
# 查看已安装的包
pip list

# 更新某个包
pip install --upgrade langchain

# 激活 LangSmith 追踪（可选）
export LANGSMITH_TRACING=true  # Unix/macOS
set LANGSMITH_TRACING=true     # Windows

# 运行全局测试
pytest tests/
```

## 📚 重要资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/quickstart
- **迁移指南**: https://docs.langchain.com/oss/python/migrate/langchain-v1
- **LangGraph 文档**: https://docs.langchain.com/oss/python/langgraph
- **LangSmith 平台**: https://smith.langchain.com
- **GitHub 仓库**: https://github.com/langchain-ai/langchain

## 🆘 常见问题

### 1. 导入错误：ModuleNotFoundError

确保虚拟环境已激活并且安装了所有依赖：
```bash
pip install -r requirements.txt
```

### 2. API Key 错误

检查 `.env` 文件是否正确配置，确保 API Keys 有效。

### 3. Python 版本不兼容

LangChain 1.0 需要 Python 3.10+：
```bash
python --version  # 检查版本
```


## 🤝 贡献

这是个人学习仓库，欢迎提交问题和改进建议！

## 📄 许可证

MIT License

## 🎓 关于作者

正在学习 LangChain 1.0 的开发者，记录学习过程供参考。

---

**开始学习之旅** 👉 [01_hello_langchain](./phase1_fundamentals/01_hello_langchain/)
