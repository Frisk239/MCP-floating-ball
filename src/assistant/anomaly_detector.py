"""
MCP Floating Ball - 异常检测和用户行为模式深度分析系统

使用多种机器学习算法检测异常行为，深度分析用户行为模式。
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略sklearn的警告
warnings.filterwarnings('ignore')

from src.core.logging import get_logger
from src.core.database import get_database

logger = get_logger("anomaly_detector")


class AnomalyType(Enum):
    """异常类型"""
    STATISTICAL = "statistical"  # 统计异常
    BEHAVIORAL = "behavioral"    # 行为异常
    TEMPORAL = "temporal"        # 时间异常
    PERFORMANCE = "performance"  # 性能异常
    SECURITY = "security"        # 安全异常
    SYSTEM = "system"           # 系统异常


class AnomalySeverity(Enum):
    """异常严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BehaviorPattern(Enum):
    """用户行为模式"""
    REGULAR_USER = "regular_user"        # 常规用户
    POWER_USER = "power_user"            # 高级用户
    BURST_USER = "burst_user"            # 突发用户
    NIGHT_OWL = "night_owl"              # 夜猫子用户
    WEEKEND_WARRIOR = "weekend_warrior"  # 周末战士
    NEW_USER = "new_user"                # 新用户
    INACTIVE_USER = "inactive_user"      # 不活跃用户


@dataclass
class UserBehaviorEvent:
    """用户行为事件"""
    timestamp: datetime
    user_id: str
    event_type: str
    action: str
    context: Dict[str, Any]
    duration: float
    success: bool
    features: Dict[str, float] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量"""
        features = [
            self.timestamp.hour / 24.0,
            self.timestamp.weekday() / 7.0,
            len(self.action) / 100.0,  # 动作长度
            self.duration / 300.0,      # 持续时间（5分钟归一化）
            float(self.success),        # 成功率
            self.context.get("complexity", 0.5),
            self.context.get("urgency", 0.5),
            len(self.context) / 10.0     # 上下文复杂度
        ]

        # 添加其他特征
        for key, value in self.features.items():
            if isinstance(value, (int, float)):
                features.append(float(value))

        return np.array(features, dtype=np.float32)


@dataclass
class AnomalyDetection:
    """异常检测结果"""
    detection_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    description: str
    affected_user_id: Optional[str]
    detected_at: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "affected_user_id": self.affected_user_id,
            "detected_at": self.detected_at.isoformat(),
            "recommendations": self.recommendations
        }


@dataclass
class BehaviorInsight:
    """行为洞察"""
    user_id: str
    pattern_type: BehaviorPattern
    confidence: float
    characteristics: Dict[str, Any]
    activity_schedule: Dict[str, float]
    preferred_actions: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    last_updated: datetime = field(default_factory=datetime.now)


class StatisticalAnomalyDetector:
    """统计异常检测器"""

    def __init__(self, window_size: int = 100, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def detect_anomaly(self, user_id: str, metric_name: str, value: float) -> Tuple[bool, float, str]:
        """检测统计异常"""
        key = f"{user_id}_{metric_name}"
        window = self.data_windows[key]

        # 添加新数据点
        window.append(value)

        if len(window) < 10:  # 数据不足，无法检测
            return False, 0.0, "数据不足"

        # 计算统计量
        mean_val = statistics.mean(window)
        std_val = statistics.stdev(window) if len(window) > 1 else 0

        if std_val == 0:
            return False, 0.0, "方差为零"

        # 计算Z分数
        z_score = abs((value - mean_val) / std_val)
        is_anomaly = z_score > self.z_threshold

        description = f"{metric_name}={value:.2f}, 历史均值={mean_val:.2f}, Z分数={z_score:.2f}"

        return is_anomaly, z_score, description


class MLAnomalyDetector:
    """机器学习异常检测器"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_buffer: List[np.ndarray] = []

    def add_training_data(self, features: np.ndarray):
        """添加训练数据"""
        self.feature_buffer.append(features)

    def train(self):
        """训练模型"""
        if len(self.feature_buffer) < 50:
            return False

        X = np.array(self.feature_buffer)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        return True

    def detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """检测异常"""
        if not self.is_trained:
            return False, 0.0, "模型未训练"

        try:
            features_scaled = self.scaler.transform([features])
            anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
            is_anomaly = self.isolation_forest.predict([features_scaled])[0] == -1

            # 转换异常分数到0-1范围
            confidence = max(0, min(1, (0.5 - anomaly_score) * 2))

            description = f"异常分数={anomaly_score:.3f}, 置信度={confidence:.3f}"

            return is_anomaly, confidence, description

        except Exception as e:
            return False, 0.0, f"检测失败: {str(e)}"


class BehaviorAnalyzer:
    """行为分析器"""

    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.activity_patterns: Dict[str, Dict[int, int]] = defaultdict(dict)  # user_id -> {hour: count}

    def analyze_behavior(self, user_id: str, events: List[UserBehaviorEvent]) -> BehaviorInsight:
        """分析用户行为"""
        if not events:
            return BehaviorInsight(
                user_id=user_id,
                pattern_type=BehaviorPattern.NEW_USER,
                confidence=0.0,
                characteristics={},
                activity_schedule={},
                preferred_actions=[],
                risk_factors=["活动数据不足"],
                recommendations=["需要更多数据来分析行为模式"]
            )

        # 分析时间模式
        activity_schedule = self._analyze_activity_schedule(events)

        # 分析偏好动作
        preferred_actions = self._analyze_preferred_actions(events)

        # 分析行为特征
        characteristics = self._analyze_characteristics(events)

        # 识别行为模式
        pattern_type, confidence = self._identify_pattern_type(events, characteristics, activity_schedule)

        # 识别风险因素
        risk_factors = self._identify_risk_factors(events, characteristics)

        # 生成建议
        recommendations = self._generate_behavior_recommendations(pattern_type, characteristics, risk_factors)

        return BehaviorInsight(
            user_id=user_id,
            pattern_type=pattern_type,
            confidence=confidence,
            characteristics=characteristics,
            activity_schedule=activity_schedule,
            preferred_actions=preferred_actions,
            risk_factors=risk_factors,
            recommendations=recommendations
        )

    def _analyze_activity_schedule(self, events: List[UserBehaviorEvent]) -> Dict[str, float]:
        """分析活动时间安排"""
        hour_counts = defaultdict(int)
        total_events = len(events)

        for event in events:
            hour = event.timestamp.hour
            hour_counts[hour] += 1

        # 计算每小时的活跃比例
        activity_schedule = {
            f"{hour:02d}:00": count / total_events
            for hour, count in hour_counts.items()
        }

        return activity_schedule

    def _analyze_preferred_actions(self, events: List[UserBehaviorEvent]) -> List[str]:
        """分析偏好动作"""
        action_counts = defaultdict(int)
        for event in events:
            action_counts[event.action] += 1

        # 返回前5个最常用的动作
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [action for action, count in sorted_actions[:5]]

    def _analyze_characteristics(self, events: List[UserBehaviorEvent]) -> Dict[str, Any]:
        """分析行为特征"""
        if not events:
            return {}

        # 基本统计
        durations = [event.duration for event in events]
        success_rate = sum(1 for event in events if event.success) / len(events)

        # 时间分析
        timestamps = [event.timestamp for event in events]
        hours = [ts.hour for ts in timestamps]
        weekdays = [ts.weekday() for ts in timestamps]

        # 活跃时间段
        peak_hour = max(set(hours), key=hours.count)
        peak_weekday = max(set(weekdays), key=weekdays.count)

        # 行为一致性
        unique_actions = len(set(event.action for event in events))
        action_diversity = unique_actions / len(events)

        return {
            "total_events": len(events),
            "avg_duration": statistics.mean(durations),
            "duration_std": statistics.stdev(durations) if len(durations) > 1 else 0,
            "success_rate": success_rate,
            "peak_hour": peak_hour,
            "peak_weekday": peak_weekday,
            "action_diversity": action_diversity,
            "unique_actions": unique_actions,
            "most_common_action": max(set(event.action for event in events), key=lambda x: [event.action for event in events].count(x))
        }

    def _identify_pattern_type(self, events: List[UserBehaviorEvent], characteristics: Dict[str, Any], activity_schedule: Dict[str, float]) -> Tuple[BehaviorPattern, float]:
        """识别行为模式类型"""
        scores = {}

        # 新用户模式
        if characteristics["total_events"] < 20:
            scores[BehaviorPattern.NEW_USER] = 0.8

        # 高级用户模式
        if characteristics["success_rate"] > 0.9 and characteristics["action_diversity"] > 0.3:
            scores[BehaviorPattern.POWER_USER] = 0.7

        # 夜猫子模式
        night_activity = sum(count for hour, count in activity_schedule.items() if int(hour.split(":")[0]) >= 22 or int(hour.split(":")[0]) <= 6)
        if night_activity > 0.3:
            scores[BehaviorPattern.NIGHT_OWL] = 0.6

        # 周末战士模式
        weekend_events = [event for event in events if event.timestamp.weekday() >= 5]
        if len(weekend_events) / len(events) > 0.5:
            scores[BehaviorPattern.WEEKEND_WARRIOR] = 0.6

        # 不活跃用户
        if characteristics["total_events"] < 5:
            scores[BehaviorPattern.INACTIVE_USER] = 0.9

        # 如果没有特殊模式，归为常规用户
        if not scores:
            scores[BehaviorPattern.REGULAR_USER] = 0.5

        # 选择得分最高的模式
        best_pattern = max(scores, key=scores.get)
        confidence = scores[best_pattern]

        return best_pattern, confidence

    def _identify_risk_factors(self, events: List[UserBehaviorEvent], characteristics: Dict[str, Any]) -> List[str]:
        """识别风险因素"""
        risk_factors = []

        # 成功率低
        if characteristics["success_rate"] < 0.7:
            risk_factors.append("成功率较低，可能存在操作困难")

        # 持续时间异常
        if characteristics["duration_std"] > characteristics["avg_duration"]:
            risk_factors.append("操作时间不稳定")

        # 行为模式单一
        if characteristics["action_diversity"] < 0.1:
            risk_factors.append("行为模式较为单一")

        # 活跃度异常
        if characteristics["total_events"] < 10:
            risk_factors.append("活跃度较低")

        return risk_factors

    def _generate_behavior_recommendations(self, pattern_type: BehaviorPattern, characteristics: Dict[str, Any], risk_factors: List[str]) -> List[str]:
        """生成行为建议"""
        recommendations = []

        if pattern_type == BehaviorPattern.NEW_USER:
            recommendations.extend([
                "建议提供新手引导",
                "推荐常用功能教程",
                "设置渐进式学习计划"
            ])
        elif pattern_type == BehaviorPattern.POWER_USER:
            recommendations.extend([
                "推荐高级功能",
                "提供快捷键和效率技巧",
                "邀请参与用户反馈"
            ])
        elif pattern_type == BehaviorPattern.NIGHT_OWL:
            recommendations.extend([
                "优化夜间使用体验",
                "提供夜间模式",
                "调整系统资源分配"
            ])
        elif pattern_type == BehaviorPattern.INACTIVE_USER:
            recommendations.extend([
                "发送重新激活通知",
                "推荐新功能和改进",
                "提供使用激励"
            ])

        # 基于风险因素的建议
        if "成功率较低" in str(risk_factors):
            recommendations.append("改进用户界面和操作流程")

        if "活跃度较低" in str(risk_factors):
            recommendations.append("增加用户互动和参与度")

        return recommendations


class AnomalyDetector:
    """异常检测和用户行为分析系统"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.db = get_database()

        # 检测器组件
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        self.behavior_analyzer = BehaviorAnalyzer()

        # 存储数据
        self.user_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.detection_history: List[AnomalyDetection] = []
        self.behavior_insights: Dict[str, BehaviorInsight] = {}

        # 配置
        self.detection_interval = 60  # 60秒检测一次
        self.is_running = False
        self.detection_task: Optional[asyncio.Task] = None

        self.logger.info("异常检测器初始化完成")

    async def start_detection(self):
        """启动异常检测"""
        if self.is_running:
            self.logger.warning("异常检测已在运行")
            return

        self.is_running = True
        self.detection_task = asyncio.create_task(self._detection_loop())
        self.logger.info("异常检测已启动")

    async def stop_detection(self):
        """停止异常检测"""
        if not self.is_running:
            return

        self.is_running = False
        if self.detection_task:
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass

        self.logger.info("异常检测已停止")

    async def _detection_loop(self):
        """检测循环"""
        while self.is_running:
            try:
                # 检测异常
                await self._detect_anomalies()

                # 分析用户行为
                await self._analyze_user_behaviors()

                # 训练ML模型
                await self._train_ml_models()

                # 清理过期数据
                self._cleanup_expired_data()

                await asyncio.sleep(self.detection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"检测循环出错: {e}")
                await asyncio.sleep(5)

    async def add_user_event(self, user_id: str, event_type: str, action: str, context: Dict[str, Any], duration: float, success: bool):
        """添加用户事件"""
        try:
            event = UserBehaviorEvent(
                timestamp=datetime.now(),
                user_id=user_id,
                event_type=event_type,
                action=action,
                context=context,
                duration=duration,
                success=success
            )

            # 添加特征
            event.features = {
                "event_type_encoded": hash(event_type) % 100,
                "action_length": len(action),
                "context_size": len(context),
                "complexity": context.get("complexity", 0.5),
                "urgency": context.get("urgency", 0.5)
            }

            # 存储事件
            self.user_events[user_id].append(event)

            # 为ML模型添加训练数据
            self.ml_detector.add_training_data(event.to_feature_vector())

            self.logger.debug(f"添加用户事件: {user_id} - {action}")

        except Exception as e:
            self.logger.error(f"添加用户事件失败: {e}")

    async def _detect_anomalies(self):
        """检测异常"""
        try:
            for user_id, events in self.user_events.items():
                if not events:
                    continue

                recent_events = list(events)[-20:]  # 分析最近20个事件

                for event in recent_events:
                    await self._detect_event_anomalies(event)

                    await self._detect_user_pattern_anomalies(user_id, recent_events)

        except Exception as e:
            self.logger.error(f"检测异常失败: {e}")

    async def _detect_event_anomalies(self, event: UserBehaviorEvent):
        """检测单个事件的异常"""
        try:
            # 统计异常检测
            for metric_name, value in event.features.items():
                is_anomaly, confidence, description = self.statistical_detector.detect_anomaly(
                    event.user_id, metric_name, value
                )

                if is_anomaly:
                    await self._create_anomaly_detection(
                        AnomalyType.STATISTICAL,
                        AnomalySeverity.MEDIUM,
                        confidence,
                        f"统计异常: {description}",
                        event.user_id,
                        {"metric": metric_name, "value": value}
                    )

            # ML异常检测
            if self.ml_detector.is_trained:
                features = event.to_feature_vector()
                is_anomaly, confidence, description = self.ml_detector.detect_anomaly(features)

                if is_anomaly:
                    await self._create_anomaly_detection(
                        AnomalyType.BEHAVIORAL,
                        AnomalySeverity.HIGH,
                        confidence,
                        f"行为异常: {description}",
                        event.user_id,
                        {"event_type": event.event_type, "action": event.action}
                    )

        except Exception as e:
            self.logger.error(f"检测事件异常失败: {e}")

    async def _detect_user_pattern_anomalies(self, user_id: str, events: List[UserBehaviorEvent]):
        """检测用户模式异常"""
        try:
            if len(events) < 5:
                return

            # 检测时间模式异常
            await self._detect_temporal_anomalies(user_id, events)

            # 检测性能异常
            await self._detect_performance_anomalies(user_id, events)

            # 检测安全异常
            await self._detect_security_anomalies(user_id, events)

        except Exception as e:
            self.logger.error(f"检测用户模式异常失败: {e}")

    async def _detect_temporal_anomalies(self, user_id: str, events: List[UserBehaviorEvent]):
        """检测时间模式异常"""
        try:
            # 分析事件时间间隔
            timestamps = [event.timestamp for event in events]
            intervals = []

            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)

            if len(intervals) < 3:
                return

            # 检测异常间隔
            mean_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals)

            for i, interval in enumerate(intervals):
                z_score = abs((interval - mean_interval) / std_interval) if std_interval > 0 else 0

                if z_score > 3.0:  # 3个标准差
                    severity = AnomalySeverity.HIGH if z_score > 4.0 else AnomalySeverity.MEDIUM
                    confidence = min(1.0, z_score / 4.0)

                    await self._create_anomaly_detection(
                        AnomalyType.TEMPORAL,
                        severity,
                        confidence,
                        f"时间间隔异常: {interval:.1f}秒 (正常: {mean_interval:.1f}±{std_interval:.1f}秒)",
                        user_id,
                        {"interval": interval, "mean_interval": mean_interval, "z_score": z_score}
                    )

        except Exception as e:
            self.logger.error(f"检测时间异常失败: {e}")

    async def _detect_performance_anomalies(self, user_id: str, events: List[UserBehaviorEvent]):
        """检测性能异常"""
        try:
            durations = [event.duration for event in events if event.duration > 0]

            if len(durations) < 3:
                return

            # 检测响应时间异常
            mean_duration = statistics.mean(durations)
            std_duration = statistics.stdev(durations)

            for event in events:
                if event.duration <= 0:
                    continue

                z_score = (event.duration - mean_duration) / std_duration if std_duration > 0 else 0

                if z_score > 2.5:  # 响应时间异常长
                    severity = AnomalySeverity.HIGH if z_score > 3.5 else AnomalySeverity.MEDIUM
                    confidence = min(1.0, z_score / 3.5)

                    await self._create_anomaly_detection(
                        AnomalyType.PERFORMANCE,
                        severity,
                        confidence,
                        f"性能异常: 响应时间{event.duration:.1f}秒 (正常: {mean_duration:.1f}±{std_duration:.1f}秒)",
                        user_id,
                        {"duration": event.duration, "mean_duration": mean_duration, "action": event.action}
                    )

        except Exception as e:
            self.logger.error(f"检测性能异常失败: {e}")

    async def _detect_security_anomalies(self, user_id: str, events: List[UserBehaviorEvent]):
        """检测安全异常"""
        try:
            # 检测失败率异常
            failed_events = [e for e in events if not e.success]
            failure_rate = len(failed_events) / len(events)

            if failure_rate > 0.3:  # 失败率超过30%
                confidence = min(1.0, failure_rate * 2)
                severity = AnomalySeverity.HIGH if failure_rate > 0.5 else AnomalySeverity.MEDIUM

                await self._create_anomaly_detection(
                    AnomalyType.SECURITY,
                    severity,
                    confidence,
                    f"安全异常: 失败率{failure_rate:.1%}",
                    user_id,
                    {"failure_rate": failure_rate, "failed_actions": [e.action for e in failed_events]}
                )

            # 检测频繁失败的动作
            action_failures = defaultdict(int)
            for event in failed_events:
                action_failures[event.action] += 1

            for action, failure_count in action_failures.items():
                if failure_count >= 3:  # 同一动作失败3次以上
                    await self._create_anomaly_detection(
                        AnomalyType.SECURITY,
                        AnomalySeverity.MEDIUM,
                        0.7,
                        f"重复失败: 动作'{action}'失败{failure_count}次",
                        user_id,
                        {"action": action, "failure_count": failure_count}
                    )

        except Exception as e:
            self.logger.error(f"检测安全异常失败: {e}")

    async def _create_anomaly_detection(self, anomaly_type: AnomalyType, severity: AnomalySeverity, confidence: float, description: str, user_id: Optional[str], raw_data: Dict[str, Any]):
        """创建异常检测结果"""
        try:
            detection = AnomalyDetection(
                detection_id=f"{anomaly_type.value}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=confidence,
                description=description,
                affected_user_id=user_id,
                raw_data=raw_data,
                recommendations=self._generate_anomaly_recommendations(anomaly_type, severity, raw_data)
            )

            self.detection_history.append(detection)

            # 限制历史记录数量
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]

            self.logger.warning(f"检测到异常: {description}")

        except Exception as e:
            self.logger.error(f"创建异常检测结果失败: {e}")

    def _generate_anomaly_recommendations(self, anomaly_type: AnomalyType, severity: AnomalySeverity, raw_data: Dict[str, Any]) -> List[str]:
        """生成异常处理建议"""
        recommendations = []

        if anomaly_type == AnomalyType.STATISTICAL:
            recommendations.extend([
                "检查数据质量",
                "验证数据采集过程",
                "分析异常产生原因"
            ])
        elif anomaly_type == AnomalyType.BEHAVIORAL:
            recommendations.extend([
                "联系用户确认行为",
                "检查系统状态",
                "审查安全策略"
            ])
        elif anomaly_type == AnomalyType.TEMPORAL:
            recommendations.extend([
                "检查系统负载",
                "优化处理流程",
                "调整资源配置"
            ])
        elif anomaly_type == AnomalyType.PERFORMANCE:
            recommendations.extend([
                "性能优化",
                "增加系统资源",
                "检查网络连接"
            ])
        elif anomaly_type == AnomalyType.SECURITY:
            recommendations.extend([
                "安全审计",
                "检查访问权限",
                "启用额外安全验证"
            ])

        if severity == AnomalySeverity.CRITICAL:
            recommendations.insert(0, "立即处理此异常")
        elif severity == AnomalySeverity.HIGH:
            recommendations.insert(0, "优先处理此异常")

        return recommendations

    async def _analyze_user_behaviors(self):
        """分析用户行为"""
        try:
            for user_id, events in self.user_events.items():
                if not events:
                    continue

                # 分析最近的用户行为
                recent_events = list(events)[-50:]  # 最近50个事件
                insight = self.behavior_analyzer.analyze_behavior(user_id, recent_events)

                self.behavior_insights[user_id] = insight

                self.logger.debug(f"用户行为分析完成: {user_id} - {insight.pattern_type.value}")

        except Exception as e:
            self.logger.error(f"分析用户行为失败: {e}")

    async def _train_ml_models(self):
        """训练机器学习模型"""
        try:
            # 每100个新数据点重新训练一次
            if len(self.ml_detector.feature_buffer) >= 100 and len(self.ml_detector.feature_buffer) % 100 == 0:
                success = self.ml_detector.train()
                if success:
                    self.logger.info("ML异常检测模型训练完成")

        except Exception as e:
            self.logger.error(f"训练ML模型失败: {e}")

    def _cleanup_expired_data(self):
        """清理过期数据"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)

            # 清理过期事件
            for user_id in list(self.user_events.keys()):
                events = self.user_events[user_id]
                while events and events[0].timestamp < cutoff_time:
                    events.popleft()

                # 如果用户没有事件了，删除用户记录
                if not events:
                    del self.user_events[user_id]

            # 清理过期检测记录
            self.detection_history = [
                detection for detection in self.detection_history
                if detection.detected_at > cutoff_time
            ]

        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")

    async def get_anomaly_summary(self) -> Dict[str, Any]:
        """获取异常检测摘要"""
        try:
            recent_detections = [
                detection for detection in self.detection_history
                if detection.detected_at > datetime.now() - timedelta(hours=24)
            ]

            # 按类型统计
            type_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for detection in recent_detections:
                type_counts[detection.anomaly_type.value] += 1
                severity_counts[detection.severity.value] += 1

            # 高危异常
            critical_detections = [
                detection for detection in recent_detections
                if detection.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
            ]

            return {
                "total_detections": len(recent_detections),
                "critical_detections": len(critical_detections),
                "type_distribution": dict(type_counts),
                "severity_distribution": dict(severity_counts),
                "affected_users": len(set(detection.affected_user_id for detection in recent_detections if detection.affected_user_id)),
                "ml_model_trained": self.ml_detector.is_trained,
                "total_events_analyzed": sum(len(events) for events in self.user_events.values())
            }

        except Exception as e:
            self.logger.error(f"获取异常摘要失败: {e}")
            return {"error": str(e)}

    async def get_user_behavior_insights(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取用户行为洞察"""
        try:
            insights = []

            if user_id:
                if user_id in self.behavior_insights:
                    insights = [self.behavior_insights[user_id]]
            else:
                insights = list(self.behavior_insights.values())

            return [
                {
                    "user_id": insight.user_id,
                    "pattern_type": insight.pattern_type.value,
                    "confidence": insight.confidence,
                    "characteristics": insight.characteristics,
                    "preferred_actions": insight.preferred_actions,
                    "risk_factors": insight.risk_factors,
                    "recommendations": insight.recommendations,
                    "last_updated": insight.last_updated.isoformat()
                }
                for insight in insights
            ]

        except Exception as e:
            self.logger.error(f"获取用户行为洞察失败: {e}")
            return []

    async def get_recent_anomalies(self, hours: int = 24, severity: Optional[AnomalySeverity] = None) -> List[Dict[str, Any]]:
        """获取最近的异常"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_anomalies = [
                detection for detection in self.detection_history
                if detection.detected_at > cutoff_time
            ]

            if severity:
                recent_anomalies = [
                    detection for detection in recent_anomalies
                    if detection.severity == severity
                ]

            return [detection.to_dict() for detection in recent_anomalies]

        except Exception as e:
            self.logger.error(f"获取最近异常失败: {e}")
            return []


# 导出
__all__ = [
    "AnomalyDetector",
    "UserBehaviorEvent",
    "AnomalyDetection",
    "BehaviorInsight",
    "StatisticalAnomalyDetector",
    "MLAnomalyDetector",
    "BehaviorAnalyzer",
    "AnomalyType",
    "AnomalySeverity",
    "BehaviorPattern"
]