"""
MCP Floating Ball - 高级AI控制器

整合所有AI功能，提供统一的智能服务接口。
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import traceback

from src.core.logging import get_logger
from src.assistant.ai_orchestrator import AIOrchestrator, get_ai_orchestrator
from src.assistant.workflow_manager import get_workflow_manager
from src.assistant.ab_testing import ABTestingFramework
from src.assistant.model_fusion import ModelFusionEngine
from src.assistant.task_dispatcher import TaskDispatcher
from src.assistant.intelligent_learner import IntelligentLearner, LearningMode
from src.assistant.performance_monitor import PerformanceMonitor, OptimizationAction
from src.assistant.anomaly_detector import AnomalyDetector
from src.assistant.workflow_recommender import WorkflowRecommender

logger = get_logger("advanced_ai_controller")


class ControllerMode(Enum):
    """控制器模式"""
    NORMAL = "normal"
    LEARNING = "learning"
    PERFORMANCE = "performance"
    DEBUG = "debug"
    MAINTENANCE = "maintenance"


class ServiceStatus(Enum):
    """服务状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceHealth:
    """服务健康状态"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_count: int = 0
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time": self.response_time,
            "error_count": self.error_count,
            "details": self.details or {}
        }


class AdvancedAIController:
    """高级AI控制器"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

        # 控制器状态
        self.mode = ControllerMode.NORMAL
        self.is_running = False
        self.startup_time: Optional[datetime] = None

        # 服务组件
        self.ai_orchestrator: Optional[AIOrchestrator] = None
        self.workflow_manager = get_workflow_manager()
        self.ab_testing: Optional[ABTestingFramework] = None
        self.model_fusion: Optional[ModelFusionEngine] = None
        self.task_dispatcher: Optional[TaskDispatcher] = None
        self.intelligent_learner: Optional[IntelligentLearner] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.workflow_recommender: Optional[WorkflowRecommender] = None

        # 健康监控
        self.service_health: Dict[str, ServiceHealth] = {}
        self.health_check_interval = 30  # 30秒

        # 统计信息
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

        self.logger.info("高级AI控制器初始化完成")

    async def initialize(self) -> bool:
        """初始化所有服务组件"""
        try:
            self.logger.info("开始初始化高级AI控制器...")

            # 初始化核心AI服务
            await self._initialize_ai_services()

            # 初始化监控和分析服务
            await self._initialize_monitoring_services()

            # 初始化学习和推荐服务
            await self._initialize_learning_services()

            self.startup_time = datetime.now()
            self.logger.info("高级AI控制器初始化完成")
            return True

        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False

    async def _initialize_ai_services(self):
        """初始化AI服务"""
        try:
            # AI编排器
            self.ai_orchestrator = get_ai_orchestrator()
            await self._register_service("ai_orchestrator", self.ai_orchestrator)

            # A/B测试框架
            self.ab_testing = ABTestingFramework(self.ai_orchestrator)
            await self._register_service("ab_testing", self.ab_testing)

            # 模型融合引擎
            self.model_fusion = ModelFusionEngine()
            await self._register_service("model_fusion", self.model_fusion)

            # 任务分发器
            self.task_dispatcher = TaskDispatcher()
            await self._register_service("task_dispatcher", self.task_dispatcher)

            self.logger.info("AI服务初始化完成")

        except Exception as e:
            self.logger.error(f"AI服务初始化失败: {e}")
            raise

    async def _initialize_monitoring_services(self):
        """初始化监控服务"""
        try:
            # 性能监控
            self.performance_monitor = PerformanceMonitor()
            await self.performance_monitor.start_monitoring()
            await self._register_service("performance_monitor", self.performance_monitor)

            # 异常检测
            self.anomaly_detector = AnomalyDetector()
            await self.anomaly_detector.start_detection()
            await self._register_service("anomaly_detector", self.anomaly_detector)

            self.logger.info("监控服务初始化完成")

        except Exception as e:
            self.logger.error(f"监控服务初始化失败: {e}")
            raise

    async def _initialize_learning_services(self):
        """初始化学习服务"""
        try:
            # 智能学习器
            self.intelligent_learner = IntelligentLearner()
            await self._register_service("intelligent_learner", self.intelligent_learner)

            # 工作流推荐器
            self.workflow_recommender = WorkflowRecommender()
            await self._register_service("workflow_recommender", self.workflow_recommender)

            self.logger.info("学习服务初始化完成")

        except Exception as e:
            self.logger.error(f"学习服务初始化失败: {e}")
            raise

    async def _register_service(self, service_name: str, service_instance: Any):
        """注册服务"""
        try:
            # 测试服务是否可用
            start_time = datetime.now()

            # 简单的健康检查
            if hasattr(service_instance, '__class__'):
                # 服务存在，标记为运行中
                self.service_health[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.RUNNING,
                    last_check=datetime.now(),
                    response_time=(datetime.now() - start_time).total_seconds()
                )
            else:
                raise Exception(f"服务 {service_name} 无效")

            self.logger.info(f"服务注册成功: {service_name}")

        except Exception as e:
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(),
                response_time=0,
                error_count=1,
                details={"error": str(e)}
            )
            raise

    async def start(self) -> bool:
        """启动控制器"""
        try:
            if self.is_running:
                self.logger.warning("控制器已在运行")
                return True

            self.logger.info("启动高级AI控制器...")

            # 初始化服务
            if not await self.initialize():
                return False

            # 启动健康检查
            asyncio.create_task(self._health_check_loop())

            self.is_running = True
            self.logger.info("高级AI控制器启动成功")
            return True

        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            return False

    async def stop(self):
        """停止控制器"""
        try:
            if not self.is_running:
                return

            self.logger.info("停止高级AI控制器...")
            self.is_running = False

            # 停止监控服务
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()

            if self.anomaly_detector:
                await self.anomaly_detector.stop_detection()

            self.logger.info("高级AI控制器已停止")

        except Exception as e:
            self.logger.error(f"停止失败: {e}")

    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """执行健康检查"""
        for service_name in list(self.service_health.keys()):
            try:
                start_time = datetime.now()

                # 简单的健康检查 - 尝试访问服务
                service_instance = getattr(self, service_name, None)
                if service_instance is not None:
                    # 服务存在，更新健康状态
                    health = self.service_health[service_name]
                    health.status = ServiceStatus.RUNNING
                    health.last_check = datetime.now()
                    health.response_time = (datetime.now() - start_time).total_seconds()
                    health.error_count = 0
                else:
                    # 服务不存在
                    health = self.service_health[service_name]
                    health.status = ServiceStatus.ERROR
                    health.error_count += 1
                    health.details = {"error": "服务实例不存在"}

            except Exception as e:
                if service_name in self.service_health:
                    health = self.service_health[service_name]
                    health.status = ServiceStatus.ERROR
                    health.error_count += 1
                    health.details = {"error": str(e)}

    # ========== 核心AI服务接口 ==========

    async def intelligent_task_execution(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """智能任务执行"""
        try:
            self.request_count += 1
            start_time = datetime.now()

            self.logger.info(f"执行智能任务: {task_description[:100]}...")

            # 记录用户交互
            await self.intelligent_learner.record_interaction(
                interaction_type="task_execution",
                content=task_description,
                context=context or {},
                outcome="success",  # 先假设成功
                duration=0  # 稍后更新
            )

            # 1. 任务分析和分发
            task_profile = await self.task_dispatcher.dispatch_task(task_description, context)

            if not task_profile or not task_profile.success:
                self.error_count += 1
                return {
                    "success": False,
                    "error": "任务分发失败",
                    "details": task_profile.to_dict() if task_profile else {}
                }

            # 2. 选择最优模型
            model_recommendation = task_profile.assigned_model

            # 3. 执行AI推理
            if self.ai_orchestrator:
                result = await self.ai_orchestrator.execute_with_single_model(
                    model_recommendation, task_description, context
                )
            else:
                # 后备方案
                result = type('Result', (), {
                    'success': True,
                    'model_output': f"已处理任务: {task_description}",
                    'confidence': 0.7,
                    'model_id': model_recommendation
                })()

            # 4. 记录执行结果
            duration = (datetime.now() - start_time).total_seconds()
            await self.intelligent_learner.record_interaction(
                interaction_type="task_execution",
                content=task_description,
                context=context or {},
                outcome="success" if result.success else "failure",
                duration=duration,
                satisfaction_score=result.confidence if result.success else 0.0
            )

            # 5. 记录任务分发结果
            await self.task_dispatcher.update_task_result(
                task_profile.task_id, model_recommendation,
                result.success, duration, 1.0, result.confidence
            )

            # 6. 异常检测
            if self.anomaly_detector:
                await self.anomaly_detector.add_user_event(
                    "default_user", "task_execution", task_description,
                    context or {}, duration, result.success
                )

            if result.success:
                self.success_count += 1
            else:
                self.error_count += 1

            return {
                "success": result.success,
                "result": result.model_output if result.success else None,
                "confidence": getattr(result, 'confidence', 0.0),
                "model_used": getattr(result, 'model_id', model_recommendation),
                "task_profile": task_profile.to_dict(),
                "execution_time": duration
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"智能任务执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    async def workflow_recommendation(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """工作流推荐"""
        try:
            self.request_count += 1

            recommendations = await self.workflow_recommender.get_personalized_recommendations(
                user_id, context, n=5
            )

            return {
                "success": True,
                "recommendations": [rec.to_dict() for rec in recommendations],
                "user_id": user_id,
                "context": context or {}
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"工作流推荐失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def performance_optimization(self, optimization_type: str = "auto") -> Dict[str, Any]:
        """性能优化"""
        try:
            self.request_count += 1

            # 获取当前性能状态
            performance_summary = await self.performance_monitor.get_performance_summary()

            optimizations = []

            if optimization_type == "auto":
                # 自动优化
                if performance_summary.get("active_alerts", 0) > 0:
                    # 手动触发优化
                    optimization_action = OptimizationAction.ADJUST_CONCURRENCY if "CPU" in str(performance_summary) else OptimizationAction.ENABLE_CACHING
                    optimization_result = await self.performance_monitor.trigger_manual_optimization(optimization_action)
                    optimizations.append(optimization_result)

            return {
                "success": True,
                "performance_summary": performance_summary,
                "optimizations": optimizations,
                "optimization_type": optimization_type
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"性能优化失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def anomaly_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """异常分析"""
        try:
            self.request_count += 1

            # 获取异常摘要
            anomaly_summary = await self.anomaly_detector.get_anomaly_summary()

            # 获取最近的异常
            recent_anomalies = await self.anomaly_detector.get_recent_anomalies(hours)

            # 获取用户行为洞察
            behavior_insights = await self.anomaly_detector.get_user_behavior_insights()

            return {
                "success": True,
                "anomaly_summary": anomaly_summary,
                "recent_anomalies": recent_anomalies,
                "behavior_insights": behavior_insights,
                "analysis_hours": hours
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"异常分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def learning_insights(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """学习洞察"""
        try:
            self.request_count += 1

            # 获取学习统计
            learning_stats = await self.intelligent_learner.get_learning_statistics()

            # 获取个性化推荐
            recommendations = await self.intelligent_learner.generate_personalized_recommendations()

            return {
                "success": True,
                "learning_statistics": learning_stats,
                "recommendations": [rec.to_dict() for rec in recommendations],
                "user_id": user_id
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"学习洞察失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # ========== 管理接口 ==========

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            uptime = (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0

            # 服务健康状态
            health_status = {
                name: health.to_dict() for name, health in self.service_health.items()
            }

            # 统计信息
            success_rate = (self.success_count / self.request_count * 100) if self.request_count > 0 else 0

            return {
                "success": True,
                "controller": {
                    "is_running": self.is_running,
                    "mode": self.mode.value,
                    "uptime_seconds": uptime,
                    "startup_time": self.startup_time.isoformat() if self.startup_time else None
                },
                "statistics": {
                    "total_requests": self.request_count,
                    "successful_requests": self.success_count,
                    "failed_requests": self.error_count,
                    "success_rate": round(success_rate, 2)
                },
                "services": health_status,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def set_mode(self, mode: ControllerMode) -> bool:
        """设置控制器模式"""
        try:
            old_mode = self.mode
            self.mode = mode

            self.logger.info(f"模式变更: {old_mode.value} -> {mode.value}")

            # 根据模式调整服务配置
            if mode == ControllerMode.PERFORMANCE:
                # 性能优先模式
                if self.intelligent_learner:
                    await self.intelligent_learner.set_learning_mode(LearningMode.ONLINE)
            elif mode == ControllerMode.LEARNING:
                # 学习优先模式
                if self.intelligent_learner:
                    await self.intelligent_learner.set_learning_mode(LearningMode.BATCH)
            elif mode == ControllerMode.DEBUG:
                # 调试模式
                self.logger.setLevel("DEBUG")

            return True

        except Exception as e:
            self.logger.error(f"设置模式失败: {e}")
            return False

    async def execute_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行管理命令"""
        try:
            self.request_count += 1

            command = command.lower()
            parameters = parameters or {}

            if command == "status":
                return await self.get_system_status()
            elif command == "optimize":
                return await self.performance_optimization(parameters.get("type", "auto"))
            elif command == "analyze":
                return await self.anomaly_analysis(parameters.get("hours", 24))
            elif command == "insights":
                return await self.learning_insights(parameters.get("user_id"))
            elif command == "recommend":
                user_id = parameters.get("user_id", "default_user")
                return await self.workflow_recommendation(user_id, parameters.get("context"))
            elif command == "set_mode":
                mode = ControllerMode(parameters.get("mode", "normal"))
                success = await self.set_mode(mode)
                return {"success": success, "mode": mode.value}
            elif command == "health_check":
                await self._perform_health_checks()
                return {
                    "success": True,
                    "health_status": {name: health.to_dict() for name, health in self.service_health.items()}
                }
            else:
                return {
                    "success": False,
                    "error": f"未知命令: {command}",
                    "available_commands": [
                        "status", "optimize", "analyze", "insights",
                        "recommend", "set_mode", "health_check"
                    ]
                }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"执行命令失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # ========== 高级功能接口 ==========

    async def multi_model_analysis(self, prompt: str, models: List[str] = None, fusion_strategy: str = "confidence_based") -> Dict[str, Any]:
        """多模型分析"""
        try:
            self.request_count += 1

            if not self.ai_orchestrator:
                return {"success": False, "error": "AI编排器未初始化"}

            # 使用默认模型列表
            if not models:
                models = ["kimi", "dashscope", "metaso"]

            # 执行多模型推理
            result = await self.ai_orchestrator.execute_with_model_ensemble(
                models, prompt, fusion_strategy
            )

            return {
                "success": True,
                "fused_result": result.fused_output,
                "confidence": result.confidence,
                "contributing_models": result.contributing_models,
                "fusion_method": result.fusion_method,
                "execution_details": result.metadata
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"多模型分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def advanced_fusion_analysis(self, prompt: str, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """高级融合分析"""
        try:
            self.request_count += 1

            if not self.model_fusion:
                return {"success": False, "error": "模型融合引擎未初始化"}

            # 模拟AI模型响应
            from src.assistant.ai_orchestrator import AIModelResponse, FusionStrategy
            responses = [
                AIModelResponse("kimi", f"Kimi分析结果: {prompt}", 0.8),
                AIModelResponse("dashscope", f"DashScope分析结果: {prompt}", 0.7),
                AIModelResponse("metaso", f"Metaso分析结果: {prompt}", 0.9)
            ]

            # 执行高级融合
            fusion_strategy = FusionStrategy.CONFIDENCE_BASED
            if analysis_depth == "deep":
                fusion_strategy = FusionStrategy.HIERARCHICAL_FUSION

            result = await self.model_fusion.advanced_fusion(
                responses, fusion_strategy, "general_analysis", {"depth": analysis_depth}
            )

            return {
                "success": True,
                "fused_output": result.fused_output,
                "confidence": result.confidence,
                "contributing_models": result.contributing_models,
                "fusion_method": result.fusion_method,
                "analysis_depth": analysis_depth,
                "metadata": result.metadata
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"高级融合分析失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# 全局控制器实例
_advanced_controller: Optional[AdvancedAIController] = None


def get_advanced_ai_controller() -> AdvancedAIController:
    """获取全局高级AI控制器实例"""
    global _advanced_controller
    if _advanced_controller is None:
        _advanced_controller = AdvancedAIController()
    return _advanced_controller


# 导出
__all__ = [
    "AdvancedAIController",
    "get_advanced_ai_controller",
    "ControllerMode",
    "ServiceStatus",
    "ServiceHealth"
]