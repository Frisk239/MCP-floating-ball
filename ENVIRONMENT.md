# MCP Floating Ball 环境配置指南

本文档详细说明了如何设置和配置 MCP Floating Ball AI 项目的运行环境。

## 🐍 推荐方案：使用 Conda 虚拟环境

Conda 是 Python 生态系统中最好的环境管理工具之一，特别适合处理复杂的依赖关系。

### 第一步：安装 Miniconda 或 Anaconda

如果您还没有安装 Conda，推荐安装 Miniconda（轻量版）：

#### Windows 用户
1. 访问 [Miniconda 官网](https://docs.conda.io/en/latest/miniconda.html)
2. 下载 Windows x64 版本的 Miniconda3 安装包
3. 运行安装程序，选择 "Add Miniconda3 to my PATH" 选项

#### Linux/macOS 用户
```bash
# 下载 Miniconda3 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 或者 macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### 第二步：创建 Conda 虚拟环境

```bash
# 进入项目目录
cd MCP-floating-ball

# 创建名为 mcp-assistant 的 conda 环境
conda create -n mcp-assistant python=3.11

# 激活虚拟环境
conda activate mcp-assistant
```

### 第三步：安装项目依赖

#### 方法1：使用 pip（推荐）
```bash
# 升级 pip 到最新版本
pip install --upgrade pip

# 安装基础依赖
pip install -r requirements/base.txt

# 安装开发依赖（如果需要开发）
pip install -r requirements/dev.txt

# 安装生产环境依赖（如果部署）
pip install -r requirements/prod.txt
```

#### 方法2：使用 conda 安装核心包
```bash
# 安装核心科学计算包
conda install numpy pandas matplotlib pillow

# 安装其他依赖
pip install -r requirements/base.txt
```

### 第四步：验证环境配置

```bash
# 验证 Python 版本
python --version
# 应该显示 Python 3.11.x

# 验证 conda 环境
conda env list
# 应该显示 mcp-assistant 环境

# 运行快速测试
python scripts/quick_test.py
```

## 🐍 备选方案：使用 Python venv

### 创建 venv 虚拟环境

#### Windows
```bash
cd MCP-floating-ball
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

#### Linux/macOS
```bash
cd MCP-floating-ball
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 安装依赖
```bash
# 升级 pip
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements/base.txt requirements/dev.txt
```

## 📦 依赖包详细说明

### requirements/base.txt - 核心依赖
```txt
# 核心框架
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
dependency-injector>=4.41.0

# AI服务
openai>=1.12.0
dashscope>=1.17.0
httpx>=0.25.0

# 异步支持
aiofiles>=23.2.0
aiohttp>=3.9.0

# 配置和环境
python-dotenv>=1.0.0
python-multipart>=0.0.6

# 日志和监控
loguru>=0.7.0
rich>=13.7.0

# 数据处理
pandas>=2.1.0
numpy>=1.24.0
python-dateutil>=2.8.0

# 图像和视觉
opencv-python>=4.8.0
pillow>=10.1.0
pytesseract>=0.3.10

# 网页和搜索
beautifulsoup4>=4.12.0
selenium>=4.16.0
playwright>=1.40.0
requests>=2.31.0

# 系统控制
pyautogui>=0.9.54
pygetwindow>=0.0.9
psutil>=5.9.0
pywin32>=306; sys_platform == 'win32'

# 文档处理
python-docx>=1.1.0
openpyxl>=3.1.0
pypdf2>=3.0.1
pypandoc-binary>=1.12

# 手势识别
mediapipe>=0.10.0

# 音频处理
pyaudio>=0.2.14
sounddevice>=0.4.6

# 其他工具
click>=8.1.0
packaging>=23.2
```

### requirements/dev.txt - 开发依赖
```txt
-r base.txt

# 测试
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
httpx-mock>=0.10.0
factory-boy>=3.3.0
faker>=20.1.0

# 代码质量
ruff>=0.1.8
black>=23.11.0
mypy>=1.8.0
pyright>=1.1.0

# 开发工具
pre-commit>=3.5.0
ipython>=8.17.0
jupyter>=1.0.0
```

## 🔧 平台特定配置

### Windows 特殊依赖

```bash
# 安装 Windows 特定的包
pip install pywin32

# 如果遇到 Visual C++ 编译器错误
# 建议安装 Visual Studio Build Tools
# 或者使用 conda-forge 的预编译包

# 使用 conda 安装（推荐）
conda install -c conda-forge pywin32
```

### macOS 特殊依赖

```bash
# 安装 macOS 系统工具
xcode-select --install

# 安装声音处理库
brew install portaudio
pip install pyaudio
```

### Linux 特殊依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev
sudo apt-get install portaudio19-dev
sudo apt-get install libasound2-dev

# CentOS/RHEL
sudo yum install python3-devel
sudo yum install alsa-lib-devel
sudo yum install portaudio-devel
```

## 🐍 环境变量配置

### Windows 环境变量设置

在系统环境变量中添加：
- `CONDA_PREFIX`: Miniconda 安装路径
- `CONDA_DEFAULT_ENV`: mcp-assistant

### 配置文件 (.condarc)

```bash
# 创建 Conda 配置文件
conda config --add channels conda-forge
conda config --add channels defaults
conda config --set channel_priority strict
```

## 🔧 常见问题解决

### 1. ImportError: No module named 'xxx'

**原因**: 依赖包未正确安装或环境未激活

**解决方法**:
```bash
# 确保环境已激活
conda activate mcp-assistant

# 重新安装依赖
pip install -r requirements/base.txt

# 检查包是否正确安装
pip list | grep package_name
```

### 2. Microsoft Visual C++ 14.0 is required

**原因**: Windows 上缺少编译工具

**解决方法**:
```bash
# 方案1：使用 conda 安装
conda install -c conda-forge 包名

# 方案2：安装 Visual Studio Build Tools
# 下载并安装 Visual Studio Build Tools 2019
# 在安装时选择 "C++ 生成工具"
```

### 3. pip install 缓慢或失败

**原因**: 网络问题或源服务器问题

**解决方法**:
```bash
# 使用国内镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements/base.txt

# 或者使用 conda-forge
conda install -c conda-forge 包名
```

### 4. GPU 相关依赖问题

**原因**: GPU 版本的包与系统不兼容

**解决方法**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或者安装特定 CUDA 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch
```

## 🔄 环境管理最佳实践

### 定期更新环境
```bash
# 更新 conda
conda update conda

# 更新包
conda update --all

# 或者重新创建环境
conda create -n mcp-assistant-new --clone mcp-assistant
conda remove --name mcp-assistant --all
```

### 导出和导入环境
```bash
# 导出环境配置
conda env export > environment.yml

# 从配置文件创建环境
conda env create -f environment.yml
```

### 激活环境脚本

创建快捷激活脚本：

#### Windows (`activate.bat`)
```batch
@echo off
call conda activate mcp-assistant
cd /d %~dp0
cmd /k
```

#### Linux/macOS (`activate.sh`)
```bash
#!/bin/bash
conda activate mcp-assistant
cd "$(dirname "$0")"
exec bash
```

## ✅ 环境验证清单

完成环境配置后，请运行以下验证：

- [ ] Conda 环境已创建并激活
- [ ] Python 版本为 3.11+
- [ ] 所有基础依赖包已安装
- [ ] 快速测试脚本运行正常
- [ ] API 服务测试（如配置密钥）
- [ ] 控制台程序可以正常启动

## 🚀 开始使用

环境配置完成后，您可以：

1. **运行基础测试**:
   ```bash
   python scripts/quick_test.py
   ```

2. **配置API密钥**:
   ```bash
   cp config/.env.example .env
   # 编辑 .env 文件，添加您的API密钥
   ```

3. **测试AI服务**:
   ```bash
   python scripts/test_api_services.py
   ```

4. **启动AI助手**:
   ```bash
   python scripts/test_console.py
   ```

---

**环境配置完成后，您就准备好开始使用 MCP Floating Ball AI助手了！** 🎉