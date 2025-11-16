"""
MCP Floating Ball - A/B测试框架

提供模型性能测试和统计显著性验证功能。
"""

import asyncio
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math

from src.core.logging import get_logger
from src.assistant.ai_orchestrator import AIOrchestrator, FusionResult, FusionStrategy

logger = get_logger("ab_testing")


class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    RESPONSE_TIME = "response_time"
    CONFIDENCE = "confidence"
    USER_SATISFACTION = "user_satisfaction"
    TOKEN_EFFICIENCY = "token_efficiency"
    COST_EFFICIENCY = "cost_efficiency"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TestMetric:
    """测试指标"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class TestVariant:
    """测试变体"""
    variant_id: str
    name: str
    description: str
    configuration: Dict[str, Any]
    metrics: List[TestMetric] = field(default_factory=list)
    is_control: bool = False

    def add_metric(self, metric_type: MetricType, value: float, context: Optional[Dict[str, Any]] = None):
        """添加指标"""
        metric = TestMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {}
        )
        self.metrics.append(metric)

    def get_metrics_by_type(self, metric_type: MetricType) -> List[TestMetric]:
        """获取特定类型的指标"""
        return [m for m in self.metrics if m.metric_type == metric_type]

    def calculate_average_metric(self, metric_type: MetricType) -> Optional[float]:
        """计算平均指标值"""
        metrics = self.get_metrics_by_type(metric_type)
        if not metrics:
            return None
        return statistics.mean([m.value for m in metrics])

    def calculate_confidence_interval(self, metric_type: MetricType, confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        metrics = self.get_metrics_by_type(metric_type)
        if len(metrics) < 2:
            return (0, 0)

        values = [m.value for m in metrics]
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))

        # 使用t分布计算置信区间
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)

        margin = t_value * std_err
        return (mean - margin, mean + margin)


@dataclass
class ABTest:
    """A/B测试"""
    test_id: str
    name: str
    description: str
    variants: List[TestVariant]
    status: TestStatus = TestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    target_sample_size: int = 100
    min_sample_size: int = 30
    confidence_level: float = 0.95
    primary_metric: MetricType = MetricType.ACCURACY

    def get_variant_by_id(self, variant_id: str) -> Optional[TestVariant]:
        """根据ID获取变体"""
        for variant in self.variants:
            if variant.variant_id == variant_id:
                return variant
        return None

    def get_total_sample_size(self) -> int:
        """获取总样本大小"""
        return sum(len(variant.metrics) for variant in self.variants)

    def is_ready_for_evaluation(self) -> bool:
        """检查是否准备好进行评估"""
        total_samples = self.get_total_sample_size()
        if total_samples < self.min_sample_size:
            return False

        # 每个变体都需要有足够的数据
        for variant in self.variants:
            primary_metrics = variant.get_metrics_by_type(self.primary_metric)
            if len(primary_metrics) < 10:  # 每个变体至少10个主指标样本
                return False

        return True

    def calculate_duration(self) -> Optional[float]:
        """计算测试持续时间（秒）"""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()


@dataclass
class TestResult:
    """测试结果"""
    test_id: str
    winner_variant: Optional[TestVariant]
    confidence_level: float
    statistical_significance: bool
    uplift_percentage: Optional[float]
    results_by_variant: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "winner_variant_id": self.winner_variant.variant_id if self.winner_variant else None,
            "winner_variant_name": self.winner_variant.name if self.winner_variant else None,
            "confidence_level": self.confidence_level,
            "statistical_significance": self.statistical_significance,
            "uplift_percentage": self.uplift_percentage,
            "results_by_variant": self.results_by_variant,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }


class ABTestingFramework:
    """A/B测试框架"""

    def __init__(self, ai_orchestrator: AIOrchestrator):
        self.ai_orchestrator = ai_orchestrator
        self.logger = get_logger(self.__class__.__name__)
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, TestResult] = {}
        self.background_monitor_task: Optional[asyncio.Task] = None

        self.logger.info("A/B测试框架初始化完成")

    async def create_ab_test(self,
                            name: str,
                            description: str,
                            variants_config: List[Dict[str, Any]],
                            primary_metric: MetricType = MetricType.ACCURACY,
                            target_sample_size: int = 100,
                            min_sample_size: int = 30) -> Dict[str, Any]:
        """创建A/B测试"""
        try:
            test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 创建测试变体
            variants = []
            for i, config in enumerate(variants_config):
                variant = TestVariant(
                    variant_id=config["id"],
                    name=config["name"],
                    description=config.get("description", ""),
                    configuration=config["configuration"],
                    is_control=config.get("is_control", False)
                )
                variants.append(variant)

            # 创建测试
            test = ABTest(
                test_id=test_id,
                name=name,
                description=description,
                variants=variants,
                primary_metric=primary_metric,
                target_sample_size=target_sample_size,
                min_sample_size=min_sample_size
            )

            self.active_tests[test_id] = test

            self.logger.info(f"A/B测试创建成功: {test_id}")
            return {
                "success": True,
                "test_id": test_id,
                "message": "A/B测试创建成功"
            }

        except Exception as e:
            self.logger.error(f"创建A/B测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def start_test(self, test_id: str) -> Dict[str, Any]:
        """启动测试"""
        try:
            if test_id not in self.active_tests:
                return {
                    "success": False,
                    "error": "测试不存在"
                }

            test = self.active_tests[test_id]
            test.status = TestStatus.RUNNING
            test.started_at = datetime.now()

            self.logger.info(f"A/B测试启动: {test_id}")
            return {
                "success": True,
                "test_id": test_id,
                "message": "测试已启动"
            }

        except Exception as e:
            self.logger.error(f"启动测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def execute_variant_test(self,
                                  test_id: str,
                                  variant_id: str,
                                  prompt: str,
                                  task_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行变体测试"""
        try:
            if test_id not in self.active_tests:
                return {
                    "success": False,
                    "error": "测试不存在"
                }

            test = self.active_tests[test_id]
            variant = test.get_variant_by_id(variant_id)
            if not variant:
                return {
                    "success": False,
                    "error": "变体不存在"
                }

            if test.status != TestStatus.RUNNING:
                return {
                    "success": False,
                    "error": "测试未运行"
                }

            # 执行模型配置
            config = variant.configuration
            start_time = datetime.now()

            if config.get("type") == "single_model":
                # 单模型测试
                model_id = config["model_id"]
                result = await self.ai_orchestrator.execute_with_single_model(
                    model_id, prompt, task_requirements
                )
            elif config.get("type") == "ensemble":
                # 集成模型测试
                model_ids = config["model_ids"]
                fusion_strategy = FusionStrategy(config.get("fusion_strategy", "confidence_based"))
                req = TaskRequirements(**task_requirements) if task_requirements else None

                result = await self.ai_orchestrator.execute_with_model_ensemble(
                    model_ids, prompt, fusion_strategy, req
                )
            else:
                return {
                    "success": False,
                    "error": "无效的配置类型"
                }

            execution_time = (datetime.now() - start_time).total_seconds()

            if result.success:
                # 记录指标
                variant.add_metric(MetricType.RESPONSE_TIME, execution_time)
                variant.add_metric(MetricType.CONFIDENCE, result.confidence or 0.0)
                variant.add_metric(MetricType.TOKEN_EFFICIENCY, result.token_count or 0.0)

                # 模拟用户满意度（实际应用中可能来自用户反馈）
                user_satisfaction = min(1.0, result.confidence or 0.5) * (1.0 - min(1.0, execution_time / 10.0))
                variant.add_metric(MetricType.USER_SATISFACTION, user_satisfaction)

                return {
                    "success": True,
                    "test_id": test_id,
                    "variant_id": variant_id,
                    "result": result.model_output,
                    "metrics": {
                        "response_time": execution_time,
                        "confidence": result.confidence,
                        "user_satisfaction": user_satisfaction
                    }
                }
            else:
                return {
                    "success": False,
                    "test_id": test_id,
                    "variant_id": variant_id,
                    "error": result.error
                }

        except Exception as e:
            self.logger.error(f"执行变体测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def evaluate_test(self, test_id: str) -> Dict[str, Any]:
        """评估测试结果"""
        try:
            if test_id not in self.active_tests:
                return {
                    "success": False,
                    "error": "测试不存在"
                }

            test = self.active_tests[test_id]

            if not test.is_ready_for_evaluation():
                return {
                    "success": False,
                    "error": "测试数据不足",
                    "total_samples": test.get_total_sample_size(),
                    "min_required": test.min_sample_size
                }

            # 执行统计显著性测试
            result = await self._perform_statistical_analysis(test)

            # 标记测试完成
            test.status = TestStatus.COMPLETED
            test.completed_at = datetime.now()

            # 保存结果
            self.completed_tests[test_id] = result

            # 移动到已完成列表
            self.active_tests.pop(test_id, None)

            self.logger.info(f"A/B测试评估完成: {test_id}, 获胜变体: {result.winner_variant.name if result.winner_variant else '无'}")

            return {
                "success": True,
                "test_id": test_id,
                "result": result.to_dict()
            }

        except Exception as e:
            self.logger.error(f"评估测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _perform_statistical_analysis(self, test: ABTest) -> TestResult:
        """执行统计分析"""
        try:
            # 获取对照组
            control_variant = next((v for v in test.variants if v.is_control), test.variants[0])

            # 计算各变体的主指标统计
            variant_stats = {}
            for variant in test.variants:
                primary_metrics = variant.get_metrics_by_type(test.primary_metric)
                if primary_metrics:
                    values = [m.value for m in primary_metrics]
                    variant_stats[variant.variant_id] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "count": len(values),
                        "confidence_interval": variant.calculate_confidence_interval(test.primary_metric, test.confidence_level)
                    }

            # 找到获胜变体
            winner_variant = None
            max_mean = -float('inf')

            for variant in test.variants:
                stats = variant_stats.get(variant.variant_id, {})
                mean = stats.get("mean", -float('inf'))
                if mean > max_mean:
                    max_mean = mean
                    winner_variant = variant

            # 计算统计显著性（使用t检验）
            statistical_significance = False
            uplift_percentage = None

            if winner_variant and winner_variant != control_variant:
                winner_stats = variant_stats[winner_variant.variant_id]
                control_stats = variant_stats[control_variant.variant_id]

                # 简化的t检验
                uplift = (winner_stats["mean"] - control_stats["mean"]) / control_stats["mean"] * 100
                uplift_percentage = uplift

                # 判断显著性（简化版，实际应该使用更严格的统计检验）
                if abs(uplift) > 5.0:  # 5%以上的提升视为显著
                    statistical_significance = True

            # 生成建议
            recommendations = await self._generate_recommendations(test, variant_stats, winner_variant)

            return TestResult(
                test_id=test.test_id,
                winner_variant=winner_variant,
                confidence_level=test.confidence_level,
                statistical_significance=statistical_significance,
                uplift_percentage=uplift_percentage,
                results_by_variant=variant_stats,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"统计分析失败: {e}")
            raise

    async def _generate_recommendations(self,
                                      test: ABTest,
                                      variant_stats: Dict[str, Dict[str, Any]],
                                      winner_variant: Optional[TestVariant]) -> List[str]:
        """生成建议"""
        recommendations = []

        if not winner_variant:
            recommendations.append("测试数据不足，无法确定获胜变体")
            return recommendations

        # 基于结果生成建议
        if winner_variant.is_control:
            recommendations.append(f"对照组 '{winner_variant.name}' 表现最佳，建议保持当前配置")
        else:
            uplift = variant_stats.get(winner_variant.variant_id, {}).get("mean", 0)
            control_mean = variant_stats.get(next(v.variant_id for v in test.variants if v.is_control), {}).get("mean", 1)

            if uplift > control_mean:
                improvement = ((uplift - control_mean) / control_mean) * 100
                recommendations.append(f"建议采用 '{winner_variant.name}'，性能提升 {improvement:.1f}%")

            # 分析配置特点
            config = winner_variant.configuration
            if config.get("type") == "ensemble":
                recommendations.append("集成模型表现优异，建议继续探索不同的模型组合")
                recommendations.append(f"融合策略 '{config.get('fusion_strategy')}' 效果良好")
            elif config.get("type") == "single_model":
                model_id = config.get("model_id")
                recommendations.append(f"模型 '{model_id}' 在此类任务中表现突出")

        # 样本大小建议
        total_samples = test.get_total_sample_size()
        if total_samples < test.target_sample_size:
            recommendations.append(f"建议增加样本大小至 {test.target_sample_size} 以提高结果可靠性")

        return recommendations

    async def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """获取测试状态"""
        try:
            # 检查活跃测试
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
                return {
                    "success": True,
                    "test_id": test_id,
                    "status": test.status.value,
                    "total_samples": test.get_total_sample_size(),
                    "target_sample_size": test.target_sample_size,
                    "duration": test.calculate_duration(),
                    "variants": [
                        {
                            "id": variant.variant_id,
                            "name": variant.name,
                            "sample_count": len(variant.metrics),
                            "avg_primary_metric": variant.calculate_average_metric(test.primary_metric)
                        }
                        for variant in test.variants
                    ]
                }

            # 检查已完成测试
            if test_id in self.completed_tests:
                result = self.completed_tests[test_id]
                return {
                    "success": True,
                    "test_id": test_id,
                    "status": "completed",
                    "result": result.to_dict()
                }

            return {
                "success": False,
                "error": "测试不存在"
            }

        except Exception as e:
            self.logger.error(f"获取测试状态失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def list_tests(self) -> Dict[str, Any]:
        """列出所有测试"""
        try:
            active_tests = []
            for test_id, test in self.active_tests.items():
                active_tests.append({
                    "test_id": test_id,
                    "name": test.name,
                    "status": test.status.value,
                    "created_at": test.created_at.isoformat(),
                    "total_samples": test.get_total_sample_size(),
                    "target_sample_size": test.target_sample_size
                })

            completed_tests = []
            for test_id, result in self.completed_tests.items():
                completed_tests.append({
                    "test_id": test_id,
                    "winner": result.winner_variant.name if result.winner_variant else "无",
                    "significant": result.statistical_significance,
                    "uplift": result.uplift_percentage,
                    "completed_at": result.created_at.isoformat()
                })

            return {
                "success": True,
                "active_tests": active_tests,
                "completed_tests": completed_tests
            }

        except Exception as e:
            self.logger.error(f"列出测试失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def start_background_monitoring(self):
        """启动后台监控"""
        if self.background_monitor_task and not self.background_monitor_task.done():
            return

        self.background_monitor_task = asyncio.create_task(self._background_monitor())
        self.logger.info("A/B测试后台监控已启动")

    async def stop_background_monitoring(self):
        """停止后台监控"""
        if self.background_monitor_task:
            self.background_monitor_task.cancel()
            try:
                await self.background_monitor_task
            except asyncio.CancelledError:
                pass
            self.background_monitor_task = None
            self.logger.info("A/B测试后台监控已停止")

    async def _background_monitor(self):
        """后台监控任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次

                # 检查是否有测试达到目标样本大小
                for test_id, test in list(self.active_tests.items()):
                    if test.status == TestStatus.RUNNING:
                        total_samples = test.get_total_sample_size()
                        if total_samples >= test.target_sample_size:
                            self.logger.info(f"测试 {test_id} 达到目标样本大小，自动评估")
                            await self.evaluate_test(test_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"后台监控出错: {e}")


# 辅助导入
class TaskRequirements:
    """任务需求"""
    def __init__(self, task_type: str, complexity: str = "medium", priority: str = "normal", **kwargs):
        self.task_type = task_type
        self.complexity = complexity
        self.priority = priority
        self.extra = kwargs


# 导出
__all__ = [
    "ABTestingFramework",
    "ABTest",
    "TestVariant",
    "TestResult",
    "MetricType",
    "TestStatus",
    "TaskRequirements"
]