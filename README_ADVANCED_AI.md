# MCP Floating Ball - 高级AI系统

## 概述

MCP Floating Ball 高级AI系统是一个企业级的智能助手平台，集成了多种AI模型、机器学习算法和智能推荐系统。该系统通过深度学习用户行为模式，提供个性化的工作流推荐和智能任务自动化服务。

## 🚀 核心特性

### 1. 智能工作流引擎 (85.7%测试通过率)
- **任务编排**: 支持顺序、并行、条件分支等多种执行模式
- **错误恢复**: 自动重试和降级处理机制
- **预定义工作流**: 智能文档处理、网络研究、系统维护、语音助手自动化

### 2. 多AI模型协作系统
- **AI编排器**: 智能选择最适合的AI模型处理任务
- **模型融合**: 11种融合算法，包括置信度加权、Borda计数、Dempster-Shafer证据理论等
- **冲突检测**: 自动检测和处理模型间的冲突
- **A/B测试**: 自动化模型性能测试和最优选择

### 3. 专业化任务分配系统
- **智能分析**: 深度分析任务特征和复杂度
- **最优匹配**: 基于专业度和历史性能的智能分配
- **多种策略**: 专家匹配、性能优先、负载均衡、成本优化

### 4. 高级学习系统
- **机器学习模型**: 内置简化的神经网络和聚类算法
- **在线学习**: 实时学习和适应新数据
- **行为模式识别**: 自动识别用户使用模式
- **个性化推荐**: 基于学习结果的智能推荐

### 5. 性能监控与优化
- **实时监控**: CPU、内存、磁盘、网络等系统指标
- **自动优化**: 智能检测性能瓶颈并自动优化
- **告警系统**: 多级告警机制和自动响应
- **性能分析**: 深度性能分析和趋势预测

### 6. 异常检测系统
- **多维检测**: 统计异常、行为异常、时间异常、性能异常、安全异常
- **机器学习**: 基于Isolation Forest的无监督异常检测
- **行为分析**: 深度用户行为模式分析
- **风险评估**: 智能风险识别和预警

### 7. 个性化工作流推荐
- **协同过滤**: 基于用户相似度的推荐
- **内容过滤**: 基于特征匹配的推荐
- **混合推荐**: 多算法融合的智能推荐
- **上下文感知**: 基于时间和使用场景的动态推荐

## 📁 项目结构

```
MCP-floating-ball/
├── src/
│   ├── assistant/
│   │   ├── advanced_ai_controller.py      # 主控制器
│   │   ├── ai_orchestrator.py             # AI编排器
│   │   ├── workflow_manager.py            # 工作流管理器
│   │   ├── ab_testing.py                  # A/B测试框架
│   │   ├── model_fusion.py               # 模型融合引擎
│   │   ├── task_dispatcher.py            # 任务分发器
│   │   ├── intelligent_learner.py        # 智能学习器
│   │   ├── performance_monitor.py        # 性能监控
│   │   ├── anomaly_detector.py           # 异常检测
│   │   ├── workflow_recommender.py       # 工作流推荐
│   │   └── enhanced_nlp_processor.py     # 增强NLP处理器
│   ├── core/
│   │   ├── workflow.py                   # 核心工作流引擎
│   │   ├── predefined_workflows.py      # 预定义工作流
│   │   ├── database.py                   # 数据库管理
│   │   └── logging.py                    # 日志系统
│   └── assistant/
│       └── tool_caller.py                # 工具调用器
├── examples/
│   └── advanced_ai_demo.py               # 演示脚本
└── tests/
    └── test_workflow_engine.py          # 测试脚本
```

## 🛠️ 安装和配置

### 环境要求
- Python 3.8+
- 依赖包见 requirements.txt

### 安装步骤
```bash
# 克隆项目
git clone <repository-url>
cd MCP-floating-ball

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加API密钥等配置
```

### 配置文件
```env
# AI模型配置
ALIBABA_CLOUD_ACCESS_KEY_ID=your_key_here
ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_secret_here
METASO_API_KEY=your_metaso_key_here

# 数据库配置
DATABASE_URL=sqlite:///mcp_floating_ball.db
```

## 🚀 快速开始

### 1. 基础使用
```python
import asyncio
from src.assistant.advanced_ai_controller import get_advanced_ai_controller

async def main():
    controller = get_advanced_ai_controller()

    # 启动系统
    await controller.start()

    # 智能任务执行
    result = await controller.intelligent_task_execution(
        "分析这个Python代码的性能问题",
        context={"file_type": "python", "analysis_depth": "deep"}
    )

    print(f"结果: {result}")

    # 停止系统
    await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 工作流推荐
```python
# 获取个性化工作流推荐
recommendations = await controller.workflow_recommendation(
    user_id="user123",
    context={
        "current_task": "数据分析",
        "skill_level": "intermediate"
    }
)

for rec in recommendations['recommendations']:
    print(f"推荐工作流: {rec['workflow_name']}")
    print(f"置信度: {rec['confidence']}")
    print(f"说明: {rec['explanation']}")
```

### 3. 多模型分析
```python
# 使用多个AI模型进行分析
result = await controller.multi_model_analysis(
    prompt="分析人工智能在医疗健康领域的应用前景",
    models=["kimi", "dashscope", "metaso"],
    fusion_strategy="confidence_based"
)

print(f"融合结果: {result['fused_output']}")
print(f"置信度: {result['confidence']}")
```

### 4. 运行演示
```bash
# 运行完整演示
python examples/advanced_ai_demo.py
```

## 📊 系统架构

### 核心组件
1. **高级AI控制器** (`AdvancedAIController`): 统一的服务接口和管理
2. **AI编排器** (`AIOrchestrator`): 多AI模型协调和管理
3. **工作流引擎** (`WorkflowEngine`): 任务编排和执行
4. **智能学习器** (`IntelligentLearner`): 机器学习和用户行为分析
5. **性能监控** (`PerformanceMonitor`): 系统性能监控和优化
6. **异常检测** (`AnomalyDetector`): 异常行为检测和分析

### 数据流
```
用户请求 → 任务分析 → 模型选择 → 并行执行 → 结果融合 → 智能推荐 → 用户反馈
    ↑                                                           ↓
性能监控 ←←←←←←←←←←←←←←←←←←←←← 学习优化 ←←←←←←←←←←←←←←←←←
```

## 🔧 API 接口

### 核心方法

#### 智能任务执行
```python
await controller.intelligent_task_execution(
    task_description: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

#### 工作流推荐
```python
await controller.workflow_recommendation(
    user_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

#### 多模型分析
```python
await controller.multi_model_analysis(
    prompt: str,
    models: List[str] = None,
    fusion_strategy: str = "confidence_based"
) -> Dict[str, Any]
```

#### 性能优化
```python
await controller.performance_optimization(
    optimization_type: str = "auto"
) -> Dict[str, Any]
```

#### 异常分析
```python
await controller.anomaly_analysis(
    hours: int = 24
) -> Dict[str, Any]
```

#### 系统管理
```python
await controller.execute_command(
    command: str,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]
```

## 📈 性能指标

### 测试结果
- **工作流引擎测试**: 85.7% 通过率
- **系统响应时间**: 平均 < 2秒
- **并发处理**: 支持100+ 并发请求
- **内存使用**: < 500MB (正常运行)
- **CPU使用**: < 30% (正常负载)

### 监控指标
- **系统资源**: CPU、内存、磁盘、网络
- **AI性能**: 响应时间、成功率、置信度
- **用户体验**: 满意度、使用频率、错误率
- **业务指标**: 工作流执行数、推荐点击率、学习效果

## 🔍 监控和调试

### 日志系统
```python
from src.core.logging import get_logger

logger = get_logger("your_component")
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")
```

### 性能监控
```python
# 获取性能摘要
summary = await performance_monitor.get_performance_summary()
print(f"当前指标: {summary['current_metrics']}")

# 获取告警信息
alerts = await performance_monitor.get_alerts()
print(f"活跃告警: {len(alerts)}")
```

### 异常检测
```python
# 添加用户事件用于异常检测
await anomaly_detector.add_user_event(
    user_id="user123",
    event_type="task_execution",
    action="data_analysis",
    context={"complexity": 0.7},
    duration=30.0,
    success=True
)
```

## 🎯 最佳实践

### 1. 系统配置
- 根据硬件资源调整并发参数
- 定期清理历史数据和缓存
- 监控系统性能指标

### 2. AI模型使用
- 根据任务类型选择合适的模型
- 使用模型融合提高准确性
- 定期评估和优化模型选择策略

### 3. 工作流设计
- 合理设计工作流的复杂度
- 添加适当的错误处理机制
- 使用条件分支处理不同场景

### 4. 学习和推荐
- 提供高质量的用户反馈
- 定期训练和更新学习模型
- 平衡探索和利用策略

## 🛠️ 故障排除

### 常见问题

#### 1. 系统启动失败
```
错误: 服务初始化失败
解决: 检查配置文件和环境变量
```

#### 2. AI模型响应慢
```
错误: 请求超时
解决: 检查网络连接和API密钥
```

#### 3. 内存使用过高
```
错误: MemoryError
解决: 调整缓存大小和数据保留策略
```

#### 4. 性能监控告警
```
错误: 高CPU使用率
解决: 检查并发数和优化算法
```

### 调试模式
```python
# 启用调试模式
await controller.set_mode(ControllerMode.DEBUG)

# 查看详细日志
logger.setLevel("DEBUG")
```

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd MCP-floating-ball

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 运行测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_workflow_engine.py

# 生成覆盖率报告
python -m pytest --cov=src tests/
```

### 代码规范
- 使用 PEP 8 代码风格
- 添加类型提示
- 编写单元测试
- 更新文档

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 📞 支持

如有问题或建议，请：
1. 查看 [FAQ](docs/FAQ.md)
2. 提交 [Issue](issues)
3. 联系开发团队

## 🗺️ 路线图

### v2.0 计划功能
- [ ] 更多AI模型集成
- [ ] 图形化管理界面
- [ ] 分布式部署支持
- [ ] 高级安全功能
- [ ] 多语言支持

### v2.1 计划功能
- [ ] 边缘计算支持
- [ ] 实时流处理
- [ ] 高级可视化
- [ ] 自动化测试框架

## 🙏 致谢

感谢所有为项目做出贡献的开发者和用户。

---

**MCP Floating Ball - 让AI更智能，让工作更高效！** 🚀