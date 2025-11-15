# MCP Floating Ball - 现代化AI助手

🚀 一个技术栈现代化、架构清晰的AI助手项目，集成多个AI服务提供商，提供强大的智能交互能力。

## ✨ 项目特色

- 🎯 **多AI服务集成**: Kimi对话 + DashScope语音 + 秘塔AI搜索
- 🏗️ **现代化架构**: FastAPI + Pydantic + 异步编程
- 🧩 **模块化设计**: 36个工具函数，清晰的功能分类
- 📝 **类型安全**: 完整的类型注解和验证
- 🔧 **开发友好**: 完善的日志、测试和错误处理
- 🎛️ **配置驱动**: 灵活的多环境配置管理

## 🛠️ 技术栈

### 核心框架
- **Python 3.11+** - 现代Python特性
- **FastAPI** - 现代异步Web框架
- **Pydantic V2** - 数据验证和设置管理
- **asyncio** - 异步编程支持

### AI服务提供商
- **月之暗面Kimi** (`kimi-k2-turbo-preview`) - 主要对话AI
- **阿里云DashScope** - 语音识别和TTS服务
- **秘塔AI搜索** - 增强搜索功能

### 开发工具
- **pytest** - 测试框架
- **ruff** - 代码检查和格式化
- **mypy** - 静态类型检查
- **loguru** - 结构化日志

## 📁 项目结构

```
MCP-floating-ball/
├── src/                          # 源代码目录
│   ├── core/                     # 核心系统
│   │   ├── config.py            # 配置管理
│   │   ├── dependencies.py      # 依赖注入
│   │   ├── exceptions.py        # 自定义异常
│   │   └── logging.py           # 日志系统
│   ├── ai/                       # AI服务层
│   │   ├── providers/           # AI服务提供商
│   │   └── orchestrator.py      # AI服务编排器
│   ├── tools/                    # 工具系统
│   │   ├── base.py              # 工具基类
│   │   ├── registry.py          # 工具注册器
│   │   └── implementations/     # 36个工具实现
│   ├── voice/                    # 语音处理模块
│   ├── vision/                   # 视觉处理模块
│   ├── web/                      # 网页功能模块
│   ├── weather/                  # 天气服务模块
│   ├── system/                   # 系统控制模块
│   ├── files/                    # 文件处理模块
│   ├── agents/                   # 智能代理
│   └── interfaces/               # 接口层
├── config/                       # 配置文件
├── data/                         # 数据存储
├── tests/                        # 测试目录
├── scripts/                      # 脚本工具
└── requirements/                 # 依赖管理
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd MCP-floating-ball

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements/dev.txt
```

### 2. 配置API密钥

复制环境变量模板：
```bash
cp config/.env.example .env
```

编辑 `.env` 文件，填入您的API密钥：
```env
# 月之暗面Kimi API
MOONSHOT_API_KEY=your_moonshot_api_key_here

# 阿里云DashScope API
ALIBABA_CLOUD_ACCESS_KEY_ID=your_alibaba_api_key_here

# 秘塔AI搜索API
METASO_API_KEY=your_metaso_api_key_here
```

### 3. 运行控制台测试

```bash
# 启动控制台交互界面
python scripts/test_console.py

# 或者使用入口命令
python -m src.interfaces.console_agent
```

## 🧰 工具系统

项目包含36个强大的工具函数，分为以下类别：

### 🌤️ 天气工具 (2个)
- `get_city_weather` - 获取城市天气信息
- `open_ai_urls` - 打开AI相关网址

### 🌐 网页工具 (8个)
- `search_chat` - 智能搜索对话
- `search_in_websites` - 在指定网站搜索
- `open_urls` - 打开URL列表
- `open_popular_websites` - 打开热门网站
- `read_and_summary_webpage` - 阅读并总结网页
- `control_web` - 控制网页操作

### 🖼️ 视觉工具 (1个)
- `identify_current_screen_save_img_and_get_response` - 屏幕识别分析

### 📝 文档处理工具 (11个)
- `write_code_and_reports` - 编写代码和报告
- `explain_code` - 代码解释
- `get_text_content` - 获取文本内容
- `explain_file_content` - 文件内容解释
- `write_articles_and_reports` - 撰写文章和报告
- `markdown_to_word_server` - Markdown转Word
- `markdown_to_excel_server` - Markdown转Excel
- `change_word_file` - 修改Word文件
- `change_excel_file` - 修改Excel文件
- `read_ppt` - 读取PPT
- `read_pdf` - 读取PDF

### ⚙️ 系统控制工具 (14个)
- `control_iflow_agent` - 控制智能流程
- `open_folder` - 打开文件夹
- `open_app` - 打开应用程序
- `open_netease_music_server` - 打开网易云音乐
- `control_netease` - 控制音乐播放
- `gesture_control` - 手势控制
- `stop_gesture_control` - 停止手势控制
- `get_clipboard_content` - 获取剪贴板内容
- `windows_shortcut` - Windows快捷键
- `create_folders_in_active_directory` - 创建文件夹
- `open_other_apps` - 打开其他应用
- `activate_window` - 激活窗口

## 🏗️ 核心架构

### 配置系统
- 使用Pydantic Settings进行类型安全的配置管理
- 支持环境变量、YAML配置文件多种配置源
- 配置验证和自动类型转换

### 依赖注入
- 基于dependency-injector的现代化依赖注入
- 模块化的容器设计，便于测试和维护

### 工具系统
- 统一的工具接口和注册机制
- 支持同步和异步执行
- 完整的参数验证和错误处理

### 日志系统
- 基于Loguru的结构化日志
- 支持多种输出格式（控制台、文件、JSON）
- 详细的执行日志和性能统计

## 🧪 测试

项目提供了多层次的测试方案，确保系统稳定性和功能完整性：

### 快速验证测试（推荐首次使用）
```bash
# 快速验证基础功能
python scripts/quick_test.py

# 环境设置和依赖检查
python scripts/setup.py
```

### API服务测试
```bash
# 测试AI服务连接和基本功能
python scripts/test_api_services.py

# 启动交互式AI助手测试
python scripts/test_console.py
```

### 全面功能测试
```bash
# 完整功能测试（生成详细报告）
python scripts/test_all_features.py
```

### 开发者测试
```bash
# 运行单元测试（如果有）
pytest

# 运行特定测试
pytest tests/unit/test_tools.py

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 测试详细说明
详细的测试指南请参考：[TESTING.md](TESTING.md)

## 📝 开发指南

### 代码规范
```bash
# 代码检查
ruff check src/

# 代码格式化
ruff format src/

# 类型检查
mypy src/
```

### 添加新工具
1. 在 `src/tools/implementations/` 下创建工具文件
2. 继承 `BaseTool` 类或使用 `@tool` 装饰器
3. 实现工具逻辑
4. 在工具注册器中注册

```python
from src.tools.base import tool, ToolCategory

@tool(
    name="my_new_tool",
    description="我的新工具",
    category=ToolCategory.UTILITY,
    parameters=[
        {
            "name": "input_text",
            "type": "string",
            "description": "输入文本",
            "required": True
        }
    ]
)
def my_new_tool(input_text: str) -> str:
    """工具实现"""
    return f"处理结果: {input_text}"
```

## 🔧 配置说明

### 主要配置项
- `DEBUG`: 调试模式开关
- `LOG_LEVEL`: 日志级别 (DEBUG/INFO/WARNING/ERROR)
- `MAX_CONCURRENT_TASKS`: 最大并发任务数
- `REQUEST_TIMEOUT`: API请求超时时间

### AI服务配置
每个AI服务提供商都有独立的配置区块，支持模型选择、超时设置等参数。

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [月之暗面](https://moonshot.cn/) - 提供强大的Kimi对话服务
- [阿里云DashScope](https://dashscope.aliyun.com/) - 提供语音和视觉AI服务
- [秘塔AI](https://metaso.cn/) - 提供智能搜索服务

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](https://github.com/yourusername/mcp-floating-ball/issues)
- 发送邮件至 your.email@example.com

---

**MCP Floating Ball** - 让AI助手更智能、更可靠、更易用！ 🎉