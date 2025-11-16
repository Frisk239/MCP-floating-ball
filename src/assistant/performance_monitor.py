"""
MCP Floating Ball - 性能监控和自动优化系统

实时监控系统性能，自动识别瓶颈并进行优化调整。
"""

import asyncio
import json
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from src.core.logging import get_logger
from src.core.database import get_database

logger = get_logger("performance_monitor")


class MetricType(Enum):
    """指标类型"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    API_LATENCY = "api_latency"
    MODEL_PERFORMANCE = "model_performance"
    WORKFLOW_DURATION = "workflow_duration"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OptimizationAction(Enum):
    """优化动作"""
    SCALE_RESOURCES = "scale_resources"
    ADJUST_TIMEOUTS = "adjust_timeouts"
    ENABLE_CACHING = "enable_caching"
    OPTIMIZE_QUERIES = "optimize_queries"
    RESTART_SERVICES = "restart_services"
    ADJUST_CONCURRENCY = "adjust_concurrency"
    CLEANUP_RESOURCES = "cleanup_resources"


@dataclass
class PerformanceMetric:
    """性能指标"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    metric_type: MetricType
    current_value: float
    threshold: float
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "description": self.description,
            "metric_type": self.metric_type.value,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "actions_taken": self.actions_taken
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    action_id: str
    action_type: OptimizationAction
    description: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percentage: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class MetricThreshold:
    """指标阈值"""

    def __init__(self,
                 warning_threshold: float,
                 critical_threshold: float,
                 emergency_threshold: Optional[float] = None):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold

    def get_alert_level(self, value: float) -> Optional[AlertLevel]:
        """获取告警级别"""
        if self.emergency_threshold and value >= self.emergency_threshold:
            return AlertLevel.EMERGENCY
        elif value >= self.critical_threshold:
            return AlertLevel.CRITICAL
        elif value >= self.warning_threshold:
            return AlertLevel.WARNING
        return None


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.db = get_database()

        # 监控配置
        self.monitoring_interval = 30  # 30秒
        self.metric_retention_hours = 24  # 保留24小时数据
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # 指标存储
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2880))  # 24小时*60分钟/30秒
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.optimization_history: List[OptimizationResult] = []

        # 阈值配置
        self.thresholds = {
            MetricType.CPU_USAGE: MetricThreshold(70.0, 85.0, 95.0),
            MetricType.MEMORY_USAGE: MetricThreshold(75.0, 90.0, 95.0),
            MetricType.RESPONSE_TIME: MetricThreshold(5.0, 10.0, 20.0),  # 秒
            MetricType.ERROR_RATE: MetricThreshold(0.05, 0.10, 0.20),  # 5%, 10%, 20%
            MetricType.API_LATENCY: MetricThreshold(2.0, 5.0, 10.0),
            MetricType.WORKFLOW_DURATION: MetricThreshold(60.0, 120.0, 300.0)  # 秒
        }

        # 系统监控线程
        self.system_monitor_thread: Optional[threading.Thread] = None
        self.stop_system_monitor = threading.Event()

        # 优化动作注册
        self.optimization_actions: Dict[OptimizationAction, Callable] = {}

        # 注册默认优化动作
        self._register_default_optimization_actions()

        self.logger.info("性能监控器初始化完成")

    def _register_default_optimization_actions(self):
        """注册默认优化动作"""
        self.optimization_actions[OptimizationAction.ADJUST_TIMEOUTS] = self._adjust_timeouts
        self.optimization_actions[OptimizationAction.ENABLE_CACHING] = self._enable_caching
        self.optimization_actions[OptimizationAction.ADJUST_CONCURRENCY] = self._adjust_concurrency
        self.optimization_actions[OptimizationAction.CLEANUP_RESOURCES] = self._cleanup_resources

    async def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            self.logger.warning("监控已在运行中")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # 启动系统资源监控线程
        self.system_monitor_thread = threading.Thread(target=self._system_monitor_loop)
        self.system_monitor_thread.daemon = True
        self.system_monitor_thread.start()

        self.logger.info("性能监控已启动")

    async def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_system_monitor.set()

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)

        self.logger.info("性能监控已停止")

    async def _monitoring_loop(self):
        """主监控循环"""
        while self.is_monitoring:
            try:
                # 收集应用级指标
                await self._collect_application_metrics()

                # 检查告警
                await self._check_alerts()

                # 检查优化机会
                await self._check_optimization_opportunities()

                # 清理过期数据
                self._cleanup_expired_metrics()

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                await asyncio.sleep(5)

    def _system_monitor_loop(self):
        """系统资源监控循环"""
        while not self.stop_system_monitor.wait(1):
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self._record_metric(MetricType.CPU_USAGE, cpu_percent, "system", unit="%")

                # 内存使用率
                memory = psutil.virtual_memory()
                self._record_metric(MetricType.MEMORY_USAGE, memory.percent, "system", unit="%")

                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    read_mb = disk_io.read_bytes / (1024 * 1024)
                    write_mb = disk_io.write_bytes / (1024 * 1024)
                    self._record_metric(MetricType.DISK_IO, read_mb + write_mb, "system", unit="MB")

                # 网络IO
                network_io = psutil.net_io_counters()
                if network_io:
                    sent_mb = network_io.bytes_sent / (1024 * 1024)
                    recv_mb = network_io.bytes_recv / (1024 * 1024)
                    self._record_metric(MetricType.NETWORK_IO, sent_mb + recv_mb, "system", unit="MB")

            except Exception as e:
                self.logger.error(f"系统监控出错: {e}")

    def _record_metric(self, metric_type: MetricType, value: float, source: str, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """记录指标"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            source=source,
            tags=tags or {},
            unit=unit
        )

        # 存储到内存
        self.metrics_history[metric_type.value].append(metric)

        # 存储到数据库（异步）
        try:
            # 这里应该异步保存到数据库，暂时记录到日志
            self.logger.debug(f"记录指标: {metric_type.value} = {value}{unit} from {source}")
        except Exception as e:
            self.logger.error(f"保存指标失败: {e}")

    async def _collect_application_metrics(self):
        """收集应用级指标"""
        try:
            # 模拟应用指标收集
            # 实际应用中，这里会收集真实的业务指标

            # API响应时间（模拟）
            api_latency = self._simulate_api_latency()
            self._record_metric(MetricType.API_LATENCY, api_latency, "api", unit="s")

            # 错误率（模拟）
            error_rate = self._simulate_error_rate()
            self._record_metric(MetricType.ERROR_RATE, error_rate, "application", unit="%")

            # 吞吐量（模拟）
            throughput = self._simulate_throughput()
            self._record_metric(MetricType.THROUGHPUT, throughput, "application", unit="req/s")

        except Exception as e:
            self.logger.error(f"收集应用指标失败: {e}")

    def _simulate_api_latency(self) -> float:
        """模拟API延迟"""
        import random
        base_latency = 1.0
        variation = random.uniform(-0.5, 2.0)
        return max(0.1, base_latency + variation)

    def _simulate_error_rate(self) -> float:
        """模拟错误率"""
        import random
        # 大部分时间错误率很低
        if random.random() < 0.9:
            return random.uniform(0.0, 0.02)  # 0-2%
        else:
            return random.uniform(0.05, 0.15)  # 5-15%

    def _simulate_throughput(self) -> float:
        """模拟吞吐量"""
        import random
        base_throughput = 100.0
        variation = random.uniform(-20, 50)
        return max(10, base_throughput + variation)

    async def _check_alerts(self):
        """检查告警"""
        for metric_type, threshold in self.thresholds.items():
            try:
                recent_metrics = list(self.metrics_history[metric_type.value])
                if not recent_metrics:
                    continue

                latest_metric = recent_metrics[-1]
                alert_level = threshold.get_alert_level(latest_metric.value)

                if alert_level:
                    await self._handle_alert(latest_metric, alert_level, threshold)
                else:
                    # 检查是否需要解决现有告警
                    await self._check_alert_resolution(metric_type)

            except Exception as e:
                self.logger.error(f"检查{metric_type.value}告警失败: {e}")

    async def _handle_alert(self, metric: PerformanceMetric, level: AlertLevel, threshold: MetricThreshold):
        """处理告警"""
        alert_id = f"{metric.metric_type.value}_{metric.source}"

        # 检查是否已存在相同告警
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            # 如果级别升级，更新告警
            if self._compare_alert_levels(level, existing_alert.level) > 0:
                existing_alert.level = level
                existing_alert.current_value = metric.value
                self.logger.warning(f"告警升级: {existing_alert.title} -> {level.value}")
            return

        # 创建新告警
        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            title=f"{level.value.upper()}: {metric.metric_type.value}",
            description=f"{metric.source}的{metric.metric_type.value}为{metric.value:.2f}{metric.unit}，超过{level.value}阈值{threshold.warning_threshold}",
            metric_type=metric.metric_type,
            current_value=metric.value,
            threshold=threshold.warning_threshold
        )

        self.active_alerts[alert_id] = alert
        self.logger.warning(f"新告警: {alert.title} - {alert.description}")

        # 根据告警级别执行自动优化
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            await self._execute_automatic_optimization(alert)

    def _compare_alert_levels(self, level1: AlertLevel, level2: AlertLevel) -> int:
        """比较告警级别，返回1表示level1更严重，-1表示level2更严重，0表示相同"""
        level_order = {
            AlertLevel.INFO: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.CRITICAL: 3,
            AlertLevel.EMERGENCY: 4
        }
        return level_order[level1] - level_order[level2]

    async def _check_alert_resolution(self, metric_type: MetricType):
        """检查告警是否可以解决"""
        recent_metrics = list(self.metrics_history[metric_type.value])
        if len(recent_metrics) < 5:  # 需要足够的样本
            return

        # 检查最近5个指标是否都在阈值以下
        threshold = self.thresholds[metric_type]
        for metric in recent_metrics[-5:]:
            if threshold.get_alert_level(metric.value) is not None:
                return  # 仍然有告警

        # 可以解决告警
        alert_id = f"{metric_type.value}_system"  # 假设source是system
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            self.logger.info(f"告警已解决: {alert.title}")
            del self.active_alerts[alert_id]

    async def _check_optimization_opportunities(self):
        """检查优化机会"""
        try:
            # 检查响应时间趋势
            await self._check_response_time_trend()

            # 检查错误率趋势
            await self._check_error_rate_trend()

            # 检查资源利用率
            await self._check_resource_utilization()

        except Exception as e:
            self.logger.error(f"检查优化机会失败: {e}")

    async def _check_response_time_trend(self):
        """检查响应时间趋势"""
        api_metrics = list(self.metrics_history[MetricType.API_LATENCY.value])
        if len(api_metrics) < 10:
            return

        # 计算最近10个指标的平均值
        recent_avg = statistics.mean(m.value for m in api_metrics[-10:])
        historical_avg = statistics.mean(m.value for m in api_metrics[:-10]) if len(api_metrics) > 10 else recent_avg

        # 如果最近平均值比历史平均值高30%，建议优化
        if recent_avg > historical_avg * 1.3:
            await self._suggest_optimization(
                OptimizationAction.ENABLE_CACHING,
                "API响应时间增长，建议启用缓存",
                {"current_avg": recent_avg, "historical_avg": historical_avg}
            )

    async def _check_error_rate_trend(self):
        """检查错误率趋势"""
        error_metrics = list(self.metrics_history[MetricType.ERROR_RATE.value])
        if len(error_metrics) < 10:
            return

        recent_avg = statistics.mean(m.value for m in error_metrics[-10:])
        if recent_avg > 0.05:  # 5%
            await self._suggest_optimization(
                OptimizationAction.OPTIMIZE_QUERIES,
                "错误率较高，建议优化查询",
                {"current_error_rate": recent_avg}
            )

    async def _check_resource_utilization(self):
        """检查资源利用率"""
        cpu_metrics = list(self.metrics_history[MetricType.CPU_USAGE.value])
        memory_metrics = list(self.metrics_history[MetricType.MEMORY_USAGE.value])

        if len(cpu_metrics) < 5 or len(memory_metrics) < 5:
            return

        avg_cpu = statistics.mean(m.value for m in cpu_metrics[-5:])
        avg_memory = statistics.mean(m.value for m in memory_metrics[-5:])

        # 如果资源利用率持续较高，建议调整并发
        if avg_cpu > 80 or avg_memory > 80:
            await self._suggest_optimization(
                OptimizationAction.ADJUST_CONCURRENCY,
                "资源利用率高，建议调整并发设置",
                {"avg_cpu": avg_cpu, "avg_memory": avg_memory}
            )

    async def _suggest_optimization(self, action_type: OptimizationAction, description: str, details: Dict[str, Any]):
        """建议优化"""
        # 检查是否已经建议过相同的优化
        for result in self.optimization_history[-5:]:  # 检查最近5个结果
            if result.action_type == action_type and result.timestamp > datetime.now() - timedelta(minutes=30):
                return  # 最近已经建议过

        self.logger.info(f"优化建议: {description}")
        # 可以在这里添加通知逻辑

    async def _execute_automatic_optimization(self, alert: PerformanceAlert):
        """执行自动优化"""
        try:
            # 根据告警类型选择优化动作
            action_type = self._select_optimization_action(alert)

            if action_type and action_type in self.optimization_actions:
                # 获取优化前的指标
                before_metrics = self._get_current_metrics()

                # 执行优化动作
                success = await self.optimization_actions[action_type]()

                # 获取优化后的指标
                after_metrics = self._get_current_metrics()

                # 计算改进程度
                improvement = self._calculate_improvement(before_metrics, after_metrics, alert.metric_type)

                # 记录优化结果
                result = OptimizationResult(
                    action_id=f"{action_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    action_type=action_type,
                    description=f"自动执行{action_type.value}以响应{alert.level.value}告警",
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    improvement_percentage=improvement,
                    success=success,
                    details={"triggering_alert": alert.alert_id}
                )

                self.optimization_history.append(result)
                alert.actions_taken.append(action_type.value)

                if success:
                    self.logger.info(f"自动优化成功: {action_type.value}, 改进: {improvement:.1f}%")
                else:
                    self.logger.warning(f"自动优化失败: {action_type.value}")

        except Exception as e:
            self.logger.error(f"执行自动优化失败: {e}")

    def _select_optimization_action(self, alert: PerformanceAlert) -> Optional[OptimizationAction]:
        """选择优化动作"""
        if alert.metric_type == MetricType.API_LATENCY:
            return OptimizationAction.ENABLE_CACHING
        elif alert.metric_type == MetricType.CPU_USAGE:
            return OptimizationAction.ADJUST_CONCURRENCY
        elif alert.metric_type == MetricType.MEMORY_USAGE:
            return OptimizationAction.CLEANUP_RESOURCES
        elif alert.metric_type == MetricType.ERROR_RATE:
            return OptimizationAction.OPTIMIZE_QUERIES

        return None

    def _get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        current_metrics = {}
        for metric_type in [MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.API_LATENCY, MetricType.ERROR_RATE]:
            metrics = list(self.metrics_history[metric_type.value])
            if metrics:
                current_metrics[metric_type.value] = metrics[-1].value

        return current_metrics

    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float], target_metric: MetricType) -> float:
        """计算改进程度"""
        target_key = target_metric.value
        if target_key not in before or target_key not in after:
            return 0.0

        before_value = before[target_key]
        after_value = after[target_key]

        if before_value == 0:
            return 0.0

        # 对于大多数指标，值越小越好
        improvement = ((before_value - after_value) / before_value) * 100

        return max(-100, min(100, improvement))  # 限制在-100%到100%之间

    async def _adjust_timeouts(self) -> bool:
        """调整超时设置"""
        try:
            # 模拟超时调整
            self.logger.info("调整超时设置...")
            # 实际实现会修改配置文件或运行时参数
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"调整超时失败: {e}")
            return False

    async def _enable_caching(self) -> bool:
        """启用缓存"""
        try:
            self.logger.info("启用缓存...")
            # 实际实现会启用各种缓存机制
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"启用缓存失败: {e}")
            return False

    async def _adjust_concurrency(self) -> bool:
        """调整并发设置"""
        try:
            self.logger.info("调整并发设置...")
            # 实际实现会调整线程池、连接池等并发参数
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"调整并发失败: {e}")
            return False

    async def _cleanup_resources(self) -> bool:
        """清理资源"""
        try:
            self.logger.info("清理资源...")
            # 实际实现会清理临时文件、释放内存等
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")
            return False

    def _cleanup_expired_metrics(self):
        """清理过期指标"""
        cutoff_time = datetime.now() - timedelta(hours=self.metric_retention_hours)

        for metric_type, metrics in self.metrics_history.items():
            # 移除过期的指标
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()

    async def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            summary = {
                "monitoring_status": "active" if self.is_monitoring else "inactive",
                "total_metrics": sum(len(metrics) for metrics in self.metrics_history.values()),
                "active_alerts": len(self.active_alerts),
                "optimization_count": len(self.optimization_history),
                "current_metrics": {},
                "alert_summary": {},
                "recent_optimizations": []
            }

            # 当前指标
            for metric_type in MetricType:
                metrics = list(self.metrics_history[metric_type.value])
                if metrics:
                    latest = metrics[-1]
                    recent_avg = statistics.mean(m.value for m in metrics[-10:]) if len(metrics) >= 10 else latest.value

                    summary["current_metrics"][metric_type.value] = {
                        "current": latest.value,
                        "recent_average": recent_avg,
                        "unit": latest.unit
                    }

            # 告警摘要
            alert_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                alert_counts[alert.level.value] += 1

            summary["alert_summary"] = dict(alert_counts)

            # 最近的优化
            recent_optimizations = sorted(self.optimization_history, key=lambda x: x.timestamp, reverse=True)[:5]
            summary["recent_optimizations"] = [
                {
                    "action_type": opt.action_type.value,
                    "success": opt.success,
                    "improvement": opt.improvement_percentage,
                    "timestamp": opt.timestamp.isoformat()
                }
                for opt in recent_optimizations
            ]

            return summary

        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            return {"error": str(e)}

    async def get_detailed_metrics(self, metric_type: MetricType, hours: int = 1) -> List[Dict[str, Any]]:
        """获取详细指标"""
        try:
            metrics = list(self.metrics_history[metric_type.value])
            cutoff_time = datetime.now() - timedelta(hours=hours)

            filtered_metrics = [
                m.to_dict() for m in metrics if m.timestamp >= cutoff_time
            ]

            return filtered_metrics

        except Exception as e:
            self.logger.error(f"获取详细指标失败: {e}")
            return []

    async def get_alerts(self, level: Optional[AlertLevel] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """获取告警"""
        try:
            alerts = []

            if active_only:
                alerts_to_check = self.active_alerts.values()
            else:
                alerts_to_check = list(self.active_alerts.values()) + self.optimization_history

            for alert in alerts_to_check:
                if level is None or alert.level == level:
                    alerts.append(alert.to_dict())

            return sorted(alerts, key=lambda x: x["created_at"], reverse=True)

        except Exception as e:
            self.logger.error(f"获取告警失败: {e}")
            return []

    def record_custom_metric(self, metric_type: MetricType, value: float, source: str = "custom", tags: Optional[Dict[str, str]] = None):
        """记录自定义指标"""
        self._record_metric(metric_type, value, source, tags)

    async def trigger_manual_optimization(self, action_type: OptimizationAction) -> Dict[str, Any]:
        """手动触发优化"""
        try:
            if action_type not in self.optimization_actions:
                return {
                    "success": False,
                    "error": f"不支持的优化动作: {action_type.value}"
                }

            before_metrics = self._get_current_metrics()
            success = await self.optimization_actions[action_type]()
            after_metrics = self._get_current_metrics()

            result = OptimizationResult(
                action_id=f"manual_{action_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                action_type=action_type,
                description=f"手动执行{action_type.value}",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=0.0,  # 手动优化不计算改进
                success=success
            )

            self.optimization_history.append(result)

            return {
                "success": True,
                "action_id": result.action_id,
                "success": success
            }

        except Exception as e:
            self.logger.error(f"手动优化失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# 导出
__all__ = [
    "PerformanceMonitor",
    "PerformanceMetric",
    "PerformanceAlert",
    "OptimizationResult",
    "MetricType",
    "AlertLevel",
    "OptimizationAction",
    "MetricThreshold"
]