# MCP Floating Ball 测试指南

本文档提供了完整的测试方案，用于验证 MCP Floating Ball AI 助手的所有功能。

## 🧪 测试方案概览

我们提供了多个测试脚本，从基础功能验证到完整的API服务测试：

### 1. 快速验证测试

**脚本**: `scripts/quick_test.py`

**用途**: 快速验证系统基础功能是否正常

**使用方法**:
```bash
python scripts/quick_test.py
```

**测试内容**:
- ✅ 模块导入检查
- ✅ 配置系统加载
- ✅ 日志系统功能
- ✅ 工具系统架构
- ✅ 文件结构完整性

**适用场景**:
- 首次运行项目前的基础验证
- 修改代码后的快速检查
- CI/CD流水线中的基础测试

---

### 2. 全面功能测试

**脚本**: `scripts/test_all_features.py`

**用途**: 全面测试所有已实现的功能模块

**使用方法**:
```bash
python scripts/test_all_features.py
```

**测试内容**:
- ✅ 文件系统结构
- ✅ 配置系统完整性
- ✅ 日志系统功能
- ✅ 异常处理系统
- ✅ 工具系统架构
- ✅ AI服务提供商配置
- ✅ AI服务编排器功能
- ✅ 控制台代理导入

**输出**: 生成详细的JSON格式测试报告 (`test_report.json`)

**适用场景**:
- 完整的功能验证
- 版本发布前的测试
- 详细的系统状态报告

---

### 3. API服务测试

**脚本**: `scripts/test_api_services.py`

**用途**: 专门测试AI服务提供商的连接和基本功能

**使用方法**:
```bash
python scripts/test_api_services.py
```

**测试内容**:
- ✅ Kimi对话服务
- ✅ 秘塔AI搜索服务
- ✅ DashScope语音和视觉服务
- ✅ 系统健康检查
- ✅ 基本对话功能
- ✅ 搜索功能测试

**前置条件**:
- 需要配置有效的API密钥
- 网络连接正常

**适用场景**:
- API服务集成测试
- API密钥验证
- 网络连接检查

---

### 4. 环境设置验证

**脚本**: `scripts/setup.py`

**用途**: 检查和配置项目运行环境

**使用方法**:
```bash
python scripts/setup.py
```

**检查内容**:
- ✅ Python版本验证
- ✅ 依赖包检查
- ✅ 配置文件验证
- ✅ 目录结构创建
- ✅ 平台兼容性
- ✅ 基础功能测试

**适用场景**:
- 新环境部署
- 开发环境配置
- 问题排查

---

### 5. 控制台交互测试

**脚本**: `scripts/test_console.py`

**用途**: 启动交互式AI助手进行手动测试

**使用方法**:
```bash
python scripts/test_console.py
```

**测试功能**:
- 💬 自然语言对话
- 🔍 网络搜索 (`/search 关键词`)
- 🖼️ 图像分析 (`/image 图片路径 问题`)
- 🔧 工具调用 (`/tool 工具名 参数`)
- 📋 工具列表 (`/tools`)
- ℹ️ 帮助系统 (`/help`)
- 📊 服务状态 (`/status`)

**适用场景**:
- 功能演示
- 用户体验测试
- 实际使用验证

## 🚀 测试执行流程

### 新用户首次使用

```bash
# 1. 进入项目目录
cd MCP-floating-ball

# 2. 快速基础验证
python scripts/quick_test.py

# 3. 环境设置检查
python scripts/setup.py

# 4. 配置API密钥（编辑.env文件）
# 编辑 .env 文件，添加您的API密钥

# 5. API服务测试
python scripts/test_api_services.py

# 6. 启动交互测试
python scripts/test_console.py
```

### 开发者日常测试

```bash
# 快速验证修改
python scripts/quick_test.py

# 功能完整性测试
python scripts/test_all_features.py

# API服务验证
python scripts/test_api_services.py
```

### 版本发布前测试

```bash
# 完整功能测试
python scripts/test_all_features.py

# API服务测试
python scripts/test_api_services.py

# 保存测试报告
# test_report.json 会自动生成
```

## 📊 测试结果解读

### 成功指标

- **快速测试**: 应该达到 100% 通过率
- **功能测试**: 应该达到 90% 以上通过率
- **API测试**: 取决于API密钥配置和网络状态

### 常见问题排查

#### 1. 模块导入失败
```bash
# 检查Python路径
python -c "import sys; print(sys.path)"

# 检查依赖安装
pip install -r requirements/base.txt
```

#### 2. 配置加载失败
```bash
# 检查.env文件
ls -la .env

# 验证配置格式
python -c "from src.core.config import settings; print('配置正常')"
```

#### 3. API服务测试失败
- 检查API密钥是否正确
- 验证网络连接
- 确认API配额是否充足

#### 4. 控制台启动失败
- 检查所有前置测试是否通过
- 确认API服务至少有一个可用
- 查看详细错误日志

## 🔧 自定义测试

### 添加新测试用例

在相应的测试脚本中添加新的测试方法：

```python
async def test_new_feature(self):
    """测试新功能"""
    print("🆕 测试新功能...")

    try:
        # 测试逻辑
        result = your_test_function()

        if result:
            print("   ✅ 新功能测试通过")
            return True
        else:
            print("   ❌ 新功能测试失败")
            return False

    except Exception as e:
        print(f"   ❌ 新功能测试异常: {e}")
        return False
```

### 测试配置

可以通过环境变量自定义测试行为：

```bash
# 设置测试超时时间
export TEST_TIMEOUT=30

# 设置测试模式
export TEST_MODE=development

# 启用详细日志
export LOG_LEVEL=DEBUG
```

## 📋 测试检查清单

### 环境检查清单
- [ ] Python 3.11+ 已安装
- [ ] 依赖包已安装 (`pip install -r requirements/base.txt`)
- [ ] .env 配置文件存在
- [ ] API密钥已配置
- [ ] 网络连接正常

### 功能测试清单
- [ ] 核心配置系统正常
- [ ] 日志系统正常工作
- [ ] 工具系统架构完整
- [ ] AI服务初始化成功
- [ ] 控制台界面可正常启动

### API服务检查清单
- [ ] Kimi对话服务可用
- [ ] 秘塔搜索服务可用
- [ ] DashScope服务可用
- [ ] 基本对话功能正常
- [ ] 搜索功能正常

## 🐛 故障排除

### 常见错误及解决方案

#### ImportError: No module named 'src'
**原因**: Python路径问题
**解决**: 确保在项目根目录运行脚本

#### ModuleNotFoundError: No module named 'xxx'
**原因**: 依赖包未安装
**解决**: `pip install -r requirements/base.txt`

#### API密钥验证失败
**原因**: API密钥未配置或无效
**解决**: 检查.env文件中的API密钥

#### 网络连接超时
**原因**: 网络问题或API服务不可用
**解决**: 检查网络连接和API服务状态

#### 配置文件解析错误
**原因**: .env文件格式错误
**解决**: 检查.env文件格式，确保没有语法错误

## 📞 支持与反馈

如果在测试过程中遇到问题，请：

1. 查看详细的错误日志
2. 检查本文档的故障排除部分
3. 运行 `python scripts/setup.py` 进行环境检查
4. 提交Issue描述具体问题和环境信息

---

**MCP Floating Ball** - 确保AI助手的每一项功能都经过严格验证！ 🎯