"""
MCP Floating Ball - 智能学习器

集成机器学习模型，实现用户行为深度学习和个性化推荐。
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import re

from src.core.logging import get_logger
from src.core.database import get_database

logger = get_logger("intelligent_learner")


class LearningMode(Enum):
    """学习模式"""
    ONLINE = "online"  # 在线学习
    BATCH = "batch"    # 批量学习
    REINFORCEMENT = "reinforcement"  # 强化学习
    TRANSFER = "transfer"  # 迁移学习


class RecommendationType(Enum):
    """推荐类型"""
    WORKFLOW = "workflow"
    MODEL_SELECTION = "model_selection"
    TASK_PATTERN = "task_pattern"
    OPTIMIZATION = "optimization"
    FEATURE_DISCOVERY = "feature_discovery"


class UserBehaviorPattern(Enum):
    """用户行为模式"""
    MORNING_ROUTINE = "morning_routine"
    WORK_SESSION = "work_session"
    RESEARCH_MODE = "research_mode"
    CREATIVE_FLOW = "creative_flow"
    TROUBLESHOOTING = "troubleshooting"
    LEARNING_MODE = "learning_mode"


@dataclass
class UserInteraction:
    """用户交互记录"""
    timestamp: datetime
    interaction_type: str  # 'command', 'workflow', 'query', 'feedback'
    content: str
    context: Dict[str, Any]
    outcome: str  # 'success', 'failure', 'partial', 'timeout'
    duration: float  # 持续时间
    satisfaction_score: Optional[float] = None  # 满意度评分
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_features(self) -> np.ndarray:
        """转换为特征向量"""
        features = []

        # 时间特征
        hour = self.timestamp.hour
        day_of_week = self.timestamp.weekday()
        features.extend([
            hour / 24.0,
            day_of_week / 7.0,
            np.sin(2 * np.pi * hour / 24),  # 周期性时间特征
            np.cos(2 * np.pi * hour / 24)
        ])

        # 内容特征（简化版）
        content_length = len(self.content)
        word_count = len(self.content.split())
        features.extend([
            content_length / 1000.0,  # 归一化
            word_count / 200.0
        ])

        # 上下文特征
        features.extend([
            float(self.outcome == 'success'),
            float(self.outcome == 'failure'),
            self.duration / 300.0  # 5分钟归一化
        ])

        # 满意度特征
        satisfaction = self.satisfaction_score or 0.5
        features.append(satisfaction)

        return np.array(features, dtype=np.float32)


@dataclass
class LearningInsight:
    """学习洞察"""
    insight_id: str
    insight_type: str
    description: str
    confidence: float
    impact_score: float
    actionable: bool
    recommendations: List[str]
    discovered_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None


@dataclass
class PersonalizedRecommendation:
    """个性化推荐"""
    recommendation_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    action_items: List[str]
    confidence: float
    priority: str  # 'low', 'medium', 'high', 'critical'
    estimated_benefit: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        # 处理recommendation_type可能是字符串或枚举的情况
        if hasattr(self.recommendation_type, 'value'):
            rec_type = self.recommendation_type.value
        else:
            rec_type = str(self.recommendation_type)

        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": rec_type,
            "title": self.title,
            "description": self.description,
            "action_items": self.action_items,
            "confidence": self.confidence,
            "priority": self.priority,
            "estimated_benefit": self.estimated_benefit,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class SimpleMLModel:
    """简化的机器学习模型"""

    def __init__(self, model_type: str = "classification"):
        self.model_type = model_type
        self.weights = None
        self.bias = None
        self.feature_dim = 0
        self.learning_rate = 0.01
        self.is_trained = False

    def initialize(self, feature_dim: int):
        """初始化模型参数"""
        self.feature_dim = feature_dim
        self.weights = np.random.randn(feature_dim) * 0.1
        self.bias = 0.0

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained or self.weights is None:
            return np.array([0.5])

        # 简单的线性模型 + sigmoid
        z = np.dot(features, self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-z))

    def train_step(self, features: np.ndarray, target: float):
        """单步训练"""
        if self.weights is None:
            self.initialize(features.shape[0])

        # 前向传播
        prediction = self.predict(features)

        # 计算梯度
        error = prediction - target
        gradient = features * error

        # 更新参数
        self.weights -= self.learning_rate * gradient
        self.bias -= self.learning_rate * error

        return error

    def train_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """批量训练"""
        if X.shape[1] != self.feature_dim:
            self.initialize(X.shape[1])

        for epoch in range(epochs):
            total_error = 0
            for features, target in zip(X, y):
                error = self.train_step(features, target)
                total_error += abs(error)

            if epoch % 20 == 0 and total_error / len(X) < 0.1:
                break

        self.is_trained = True


class ClusteringModel:
    """简化的聚类模型"""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.centroids = None
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """拟合聚类模型"""
        # K-means的简化实现
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # 随机初始化质心
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(50):  # 最大迭代次数
            # 分配样本到最近的质心
            distances = np.zeros((n_samples, self.n_clusters))
            for i, centroid in enumerate(self.centroids):
                distances[:, i] = np.linalg.norm(X - centroid, axis=1)

            labels = np.argmin(distances, axis=1)

            # 更新质心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)

            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        if not self.is_fitted or self.centroids is None:
            return np.zeros(len(X), dtype=int)

        distances = np.zeros((len(X), self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)

        return np.argmin(distances, axis=1)


class IntelligentLearner:
    """智能学习器 - 集成机器学习模型的深度学习系统"""

    def __init__(self, user_id: str = "default"):
        """
        初始化智能学习器

        Args:
            user_id: 用户ID
        """
        self.user_id = user_id
        self.db = get_database()
        self.logger = get_logger(self.__class__.__name__)

        # 存储用户交互历史
        self.interaction_history: deque = deque(maxlen=10000)
        self.behavior_patterns: Dict[str, List[UserInteraction]] = defaultdict(list)

        # 机器学习模型
        self.satisfaction_predictor = SimpleMLModel("regression")
        self.workflow_recommender = SimpleMLModel("recommendation")
        self.behavior_clusterer = ClusteringModel(n_clusters=5)
        self.success_predictor = SimpleMLModel("classification")

        # 学习参数
        self.learning_mode = LearningMode.ONLINE
        self.min_samples_for_training = 50
        self.retrain_interval = 100  # 每100次交互重新训练

        # 洞察和推荐
        self.discovered_insights: List[LearningInsight] = []
        self.personalized_recommendations: List[PersonalizedRecommendation] = []
        self.insight_cache: Dict[str, LearningInsight] = {}

        # 加载历史数据
        self._load_historical_data()

        self.logger.info(f"智能学习器初始化完成: 用户ID {user_id}")

    def learn_from_command(self, original_command: str, intent_type: str, intent_confidence: float,
                          parameters: Dict[str, Any], tool_name: str, execution_time: float,
                          success: bool, error_message: Optional[str] = None,
                          context_data: Optional[Dict[str, Any]] = None,
                          session_id: Optional[str] = None) -> bool:
        """
        从命令执行中学习

        Args:
            original_command: 原始命令
            intent_type: 识别的意图类型
            intent_confidence: 意图置信度
            parameters: 执行参数
            tool_name: 使用的工具
            execution_time: 执行时间
            success: 是否成功
            error_message: 错误信息
            context_data: 上下文数据
            session_id: 会话ID

        Returns:
            学习是否成功
        """
        try:
            # 1. 记录命令历史
            self.db.add_command_history(
                session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
                user_id=self.user_id,
                original_command=original_command,
                intent_type=intent_type,
                intent_confidence=intent_confidence,
                parameters=parameters,
                tool_name=tool_name,
                execution_time=execution_time,
                success=success,
                error_message=error_message,
                context_data=context_data
            )

            # 2. 更新实体受欢迎程度
            self._update_entity_usage(original_command, success)

            # 3. 学习用户模式
            self._learn_user_patterns(original_command, intent_type, parameters, success)

            # 4. 清理缓存
            self._invalidate_cache()

            self.logger.debug(f"学习完成: {original_command} -> {intent_type}")
            return True

        except Exception as e:
            self.logger.error(f"学习失败: {e}")
            return False

    def get_personalized_suggestions(self, current_command: str,
                                    current_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        基于历史行为获取个性化建议

        Args:
            current_command: 当前命令
            current_context: 当前上下文

        Returns:
            个性化建议列表
        """
        try:
            suggestions = []

            # 1. 基于历史命令的建议
            history_suggestions = self._get_history_based_suggestions(current_command)
            suggestions.extend(history_suggestions)

            # 2. 基于时间模式的建议
            time_suggestions = self._get_time_based_suggestions()
            suggestions.extend(time_suggestions)

            # 3. 基于实体的建议
            entity_suggestions = self._get_entity_based_suggestions(current_command)
            suggestions.extend(entity_suggestions)

            # 4. 去重并排序
            suggestions = self._deduplicate_suggestions(suggestions)
            suggestions = sorted(suggestions, key=lambda x: x.get("score", 0), reverse=True)

            return suggestions[:10]  # 返回前10个建议

        except Exception as e:
            self.logger.error(f"获取个性化建议失败: {e}")
            return []

    def predict_intent(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于历史数据预测意图

        Args:
            command: 用户命令
            context: 上下文信息

        Returns:
            意图预测结果
        """
        try:
            # 1. 检查是否有相似的历史命令
            similar_commands = self._find_similar_commands(command, top_k=5)

            if similar_commands:
                # 基于相似命令预测意图
                intent_weights = defaultdict(float)
                total_weight = 0

                for similar_cmd in similar_commands:
                    similarity = similar_cmd["similarity"]
                    success_rate = similar_cmd["success_rate"]
                    weight = similarity * success_rate

                    intent_weights[similar_cmd["intent_type"]] += weight
                    total_weight += weight

                if total_weight > 0:
                    # 选择权重最高的意图
                    predicted_intent = max(intent_weights.items(), key=lambda x: x[1])
                    confidence = predicted_intent[1] / total_weight

                    return {
                        "predicted_intent": predicted_intent[0],
                        "confidence": confidence,
                        "based_on_history": True,
                        "similar_commands": [cmd["original_command"] for cmd in similar_commands[:3]],
                        "intent_weights": dict(intent_weights)
                    }

            # 2. 如果没有历史数据，返回默认预测
            return {
                "predicted_intent": "UNKNOWN",
                "confidence": 0.0,
                "based_on_history": False,
                "reason": "no_similar_history"
            }

        except Exception as e:
            self.logger.error(f"意图预测失败: {e}")
            return {
                "predicted_intent": "UNKNOWN",
                "confidence": 0.0,
                "based_on_history": False,
                "error": str(e)
            }

    def get_entity_recommendations(self, query: str, entity_type: Optional[str] = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取实体推荐

        Args:
            query: 查询字符串
            entity_type: 实体类型过滤
            limit: 返回数量限制

        Returns:
            推荐实体列表
        """
        try:
            # 1. 搜索匹配的实体
            entities = self.db.search_entity(query, entity_type)

            # 2. 根据用户历史偏好调整排序
            entities = self._rank_entities_by_user_preference(entities)

            # 3. 添加个性化评分
            for entity in entities:
                entity["personalized_score"] = self._calculate_entity_score(entity, query)

            # 4. 排序并返回
            entities = sorted(entities, key=lambda x: x.get("personalized_score", 0), reverse=True)

            return entities[:limit]

        except Exception as e:
            self.logger.error(f"获取实体推荐失败: {e}")
            return []

    def get_user_insights(self, days: int = 30) -> Dict[str, Any]:
        """
        获取用户行为洞察

        Args:
            days: 分析天数

        Returns:
            用户行为洞察
        """
        try:
            # 1. 获取基本统计
            stats = self.db.get_command_statistics(self.user_id, days)

            # 2. 分析使用模式
            patterns = self._analyze_usage_patterns(days)

            # 3. 识别偏好
            preferences = self._identify_user_preferences(days)

            # 4. 生成洞察
            insights = {
                "analysis_period": f"{days}天",
                "statistics": stats,
                "usage_patterns": patterns,
                "preferences": preferences,
                "recommendations": self._generate_recommendations(stats, patterns, preferences)
            }

            return insights

        except Exception as e:
            self.logger.error(f"获取用户洞察失败: {e}")
            return {}

    def _update_entity_usage(self, command: str, success: bool):
        """更新实体使用统计"""
        try:
            # 提取命令中的实体
            entities = self._extract_entities_from_command(command)

            for entity in entities:
                # 搜索匹配的系统实体
                matched_entities = self.db.search_entity(entity)
                if matched_entities:
                    # 更新第一个匹配的实体
                    self.db.update_entity_popularity(matched_entities[0]["id"], success)

        except Exception as e:
            self.logger.error(f"更新实体使用统计失败: {e}")

    def _extract_entities_from_command(self, command: str) -> List[str]:
        """从命令中提取实体"""
        entities = []

        # 常见应用和网站关键词
        app_keywords = ["记事本", "计算器", "画图", "微信", "QQ", "浏览器", "Chrome", "Firefox", "Word", "Excel", "PowerPoint"]
        site_keywords = ["百度", "谷歌", "必应", "淘宝", "京东", "知乎", "微博"]

        for keyword in app_keywords + site_keywords:
            if keyword in command:
                entities.append(keyword)

        return list(set(entities))

    def _learn_user_patterns(self, command: str, intent_type: str, parameters: Dict[str, Any], success: bool):
        """学习用户模式"""
        try:
            # 1. 命令偏好模式
            self._learn_command_preference_pattern(command, intent_type, success)

            # 2. 时间偏好模式
            self._learn_time_preference_pattern(intent_type)

            # 3. 参数偏好模式
            self._learn_parameter_preference_pattern(intent_type, parameters, success)

            # 4. 上下文偏好模式
            self._learn_context_preference_pattern(command, intent_type, success)

        except Exception as e:
            self.logger.error(f"学习用户模式失败: {e}")

    def _learn_command_preference_pattern(self, command: str, intent_type: str, success: bool):
        """学习命令偏好模式"""
        try:
            # 提取命令的关键特征
            features = self._extract_command_features(command)

            pattern_data = {
                "command_length": len(command),
                "contains_numbers": bool(re.search(r'\d', command)),
                "contains_english": bool(re.search(r'[a-zA-Z]', command)),
                "word_count": len(command.split()),
                "intent_type": intent_type,
                "success_rate": 1.0 if success else 0.0
            }

            self.db.add_user_pattern(
                user_id=self.user_id,
                pattern_type="command_preference",
                pattern_data=pattern_data,
                frequency=1,
                confidence=0.5 if success else 0.2
            )

        except Exception as e:
            self.logger.error(f"学习命令偏好模式失败: {e}")

    def _learn_time_preference_pattern(self, intent_type: str):
        """学习时间偏好模式"""
        try:
            now = datetime.now()
            pattern_data = {
                "intent_type": intent_type,
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "is_weekend": now.weekday() >= 5,
                "time_category": self._categorize_time(now.hour)
            }

            self.db.add_user_pattern(
                user_id=self.user_id,
                pattern_type="time_preference",
                pattern_data=pattern_data,
                frequency=1,
                confidence=0.6
            )

        except Exception as e:
            self.logger.error(f"学习时间偏好模式失败: {e}")

    def _learn_parameter_preference_pattern(self, intent_type: str, parameters: Dict[str, Any], success: bool):
        """学习参数偏好模式"""
        try:
            pattern_data = {
                "intent_type": intent_type,
                "parameter_count": len(parameters),
                "parameter_types": {k: type(v).__name__ for k, v in parameters.items()},
                "success_rate": 1.0 if success else 0.0
            }

            self.db.add_user_pattern(
                user_id=self.user_id,
                pattern_type="parameter_preference",
                pattern_data=pattern_data,
                frequency=1,
                confidence=0.4
            )

        except Exception as e:
            self.logger.error(f"学习参数偏好模式失败: {e}")

    def _learn_context_preference_pattern(self, command: str, intent_type: str, success: bool):
        """学习上下文偏好模式"""
        try:
            # 简化的上下文信息
            context_features = {
                "command_starts_with_action": any(command.startswith(prefix) for prefix in ["打开", "搜索", "查看", "启动"]),
                "contains_question_words": any(word in command for word in ["什么", "如何", "怎么", "为什么"]),
                "command_tone": "imperative" if any(command.endswith(punct) for punct in ["。", "！"]) else "neutral"
            }

            pattern_data = {
                "intent_type": intent_type,
                "context_features": context_features,
                "success_rate": 1.0 if success else 0.0
            }

            self.db.add_user_pattern(
                user_id=self.user_id,
                pattern_type="context_preference",
                pattern_data=pattern_data,
                frequency=1,
                confidence=0.3
            )

        except Exception as e:
            self.logger.error(f"学习上下文偏好模式失败: {e}")

    def _extract_command_features(self, command: str) -> Dict[str, Any]:
        """提取命令特征"""
        return {
            "length": len(command),
            "word_count": len(command.split()),
            "has_numbers": bool(re.search(r'\d', command)),
            "has_english": bool(re.search(r'[a-zA-Z]', command)),
            "has_punctuation": bool(re.search(r'[^\w\s]', command))
        }

    def _categorize_time(self, hour: int) -> str:
        """对时间进行分类"""
        if 6 <= hour < 9:
            return "早晨"
        elif 9 <= hour < 12:
            return "上午"
        elif 12 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 22:
            return "晚上"
        else:
            return "深夜"

    def _find_similar_commands(self, command: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """查找相似的历史命令"""
        try:
            # 获取最近的命令历史
            history = self.db.get_command_history(self.user_id, limit=1000)

            similar_commands = []
            for cmd_record in history:
                similarity = self._calculate_command_similarity(command, cmd_record["original_command"])
                if similarity > 0.3:  # 相似度阈值
                    similar_commands.append({
                        "original_command": cmd_record["original_command"],
                        "intent_type": cmd_record["intent_type"],
                        "similarity": similarity,
                        "success_rate": 1.0 if cmd_record["success"] else 0.0,
                        "timestamp": cmd_record["timestamp"]
                    })

            # 按相似度排序并返回前top_k个
            similar_commands.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_commands[:top_k]

        except Exception as e:
            self.logger.error(f"查找相似命令失败: {e}")
            return []

    def _calculate_command_similarity(self, cmd1: str, cmd2: str) -> float:
        """计算命令相似度"""
        try:
            # 简单的相似度计算，可以替换为更复杂的算法
            words1 = set(cmd1.lower().split())
            words2 = set(cmd2.lower().split())

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            if not union:
                return 0.0

            jaccard_similarity = len(intersection) / len(union)

            # 长度相似度调整
            length_ratio = min(len(cmd1), len(cmd2)) / max(len(cmd1), len(cmd2))
            adjusted_similarity = jaccard_similarity * length_ratio

            return adjusted_similarity

        except Exception:
            return 0.0

    def _get_history_based_suggestions(self, current_command: str) -> List[Dict[str, Any]]:
        """基于历史命令的建议"""
        try:
            similar_commands = self._find_similar_commands(current_command, top_k=3)
            suggestions = []

            for i, cmd in enumerate(similar_commands):
                suggestions.append({
                    "type": "history_based",
                    "suggestion": f"您之前执行过类似的命令: '{cmd['original_command']}'",
                    "score": cmd["similarity"] * cmd["success_rate"],
                    "similar_command": cmd["original_command"],
                    "intent_type": cmd["intent_type"],
                    "confidence": cmd["similarity"]
                })

            return suggestions

        except Exception as e:
            self.logger.error(f"获取历史建议失败: {e}")
            return []

    def _get_time_based_suggestions(self) -> List[Dict[str, Any]]:
        """基于时间模式的建议"""
        try:
            now = datetime.now()
            time_category = self._categorize_time(now.hour)

            # 获取时间偏好模式
            time_patterns = self.db.get_user_patterns(
                user_id=self.user_id,
                pattern_type="time_preference",
                limit=50
            )

            # 统计当前时间段的常用意图
            current_time_patterns = [
                p for p in time_patterns
                if p["pattern_data"]["hour"] == now.hour or
                p["pattern_data"]["time_category"] == time_category
            ]

            if current_time_patterns:
                # 找出最常用的意图
                intent_counts = Counter(
                    p["pattern_data"]["intent_type"]
                    for p in current_time_patterns
                )

                most_common_intent = intent_counts.most_common(1)[0]

                return [{
                    "type": "time_based",
                    "suggestion": f"在这个时间段，您通常执行{most_common_intent[0]}相关的操作",
                    "score": 0.5,
                    "time_category": time_category,
                    "common_intent": most_common_intent[0],
                    "frequency": most_common_intent[1]
                }]

            return []

        except Exception as e:
            self.logger.error(f"获取时间建议失败: {e}")
            return []

    def _get_entity_based_suggestions(self, current_command: str) -> List[Dict[str, Any]]:
        """基于实体的建议"""
        try:
            # 提取命令中的关键词
            keywords = self._extract_keywords(current_command)

            suggestions = []
            for keyword in keywords:
                # 搜索相关实体
                entities = self.db.search_entity(keyword)
                for entity in entities[:3]:  # 每个关键词最多取3个实体
                    suggestions.append({
                        "type": "entity_based",
                        "suggestion": f"您是否想要操作 '{entity['entity_name']}'？",
                        "score": entity["match_score"] * 0.4,
                        "entity": entity["entity_name"],
                        "entity_type": entity["entity_type"],
                        "match_score": entity["match_score"]
                    })

            return suggestions

        except Exception as e:
            self.logger.error(f"获取实体建议失败: {e}")
            return []

    def _extract_keywords(self, command: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "这", "那", "有", "和", "与", "或"}
        words = [word for word in command.split() if word not in stop_words and len(word) > 1]
        return list(set(words))

    def _rank_entities_by_user_preference(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据用户偏好重新排序实体"""
        try:
            # 获取用户历史中实体使用频率
            history = self.db.get_command_history(self.user_id, limit=500)

            entity_usage = defaultdict(int)
            for cmd in history:
                used_entities = self._extract_entities_from_command(cmd["original_command"])
                for entity in used_entities:
                    entity_usage[entity] += 1

            # 根据使用频率调整排序
            for entity in entities:
                entity_name = entity["entity_name"]
                usage_boost = entity_usage.get(entity_name, 0) * 0.1  # 使用频率加成
                entity["popularity_score"] = entity.get("popularity", 0) + usage_boost

            return sorted(entities, key=lambda x: x.get("popularity_score", 0), reverse=True)

        except Exception as e:
            self.logger.error(f"重新排序实体失败: {e}")
            return entities

    def _calculate_entity_score(self, entity: Dict[str, Any], query: str) -> float:
        """计算实体的个性化评分"""
        try:
            base_score = entity.get("match_score", 0) * 0.5
            popularity_score = min(entity.get("popularity", 0) / 100, 1.0) * 0.3
            success_rate_score = entity.get("success_rate", 1.0) * 0.2

            total_score = base_score + popularity_score + success_rate_score
            return min(total_score, 1.0)

        except Exception:
            return 0.0

    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去除重复建议"""
        seen_suggestions = set()
        unique_suggestions = []

        for suggestion in suggestions:
            suggestion_text = suggestion.get("suggestion", "")
            if suggestion_text not in seen_suggestions:
                seen_suggestions.add(suggestion_text)
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def _analyze_usage_patterns(self, days: int) -> Dict[str, Any]:
        """分析使用模式"""
        try:
            history = self.db.get_command_history(self.user_id, limit=1000)

            if not history:
                return {}

            # 时间分布分析
            hour_distribution = defaultdict(int)
            day_distribution = defaultdict(int)

            for cmd in history:
                cmd_time = datetime.fromisoformat(cmd["timestamp"].replace('Z', '+00:00'))
                hour_distribution[cmd_time.hour] += 1
                day_distribution[cmd_time.weekday()] += 1

            # 命令长度分析
            command_lengths = [len(cmd["original_command"]) for cmd in history]
            avg_command_length = sum(command_lengths) / len(command_lengths)

            # 成功率分析
            success_count = sum(1 for cmd in history if cmd["success"])
            success_rate = success_count / len(history)

            return {
                "total_commands": len(history),
                "avg_command_length": round(avg_command_length, 1),
                "success_rate": round(success_rate, 3),
                "most_active_hours": dict(Counter(hour_distribution).most_common(3)),
                "most_active_days": dict(Counter(day_distribution).most_common(3)),
                "hour_distribution": dict(hour_distribution),
                "day_distribution": dict(day_distribution)
            }

        except Exception as e:
            self.logger.error(f"分析使用模式失败: {e}")
            return {}

    def _identify_user_preferences(self, days: int) -> Dict[str, Any]:
        """识别用户偏好"""
        try:
            history = self.db.get_command_history(self.user_id, limit=1000)

            if not history:
                return {}

            # 意图偏好
            intent_counts = Counter(cmd["intent_type"] for cmd in history)
            preferred_intents = dict(intent_counts.most_common(5))

            # 工具偏好
            tool_counts = Counter(cmd["tool_name"] for cmd in history if cmd["tool_name"])
            preferred_tools = dict(tool_counts.most_common(5))

            # 参数偏好
            parameter_stats = defaultdict(list)
            for cmd in history:
                param_count = len(cmd["parameters"])
                parameter_stats[cmd["intent_type"]].append(param_count)

            avg_parameters = {
                intent: sum(params) / len(params)
                for intent, params in parameter_stats.items()
                if params
            }

            return {
                "preferred_intents": preferred_intents,
                "preferred_tools": preferred_tools,
                "avg_parameters_per_intent": avg_parameters,
                "command_style": self._identify_command_style(history)
            }

        except Exception as e:
            self.logger.error(f"识别用户偏好失败: {e}")
            return {}

    def _identify_command_style(self, history: List[Dict[str, Any]]) -> str:
        """识别命令风格"""
        try:
            # 分析命令的复杂度和表达方式
            avg_length = sum(len(cmd["original_command"]) for cmd in history) / len(history)

            if avg_length < 10:
                return "简洁直接"
            elif avg_length < 20:
                return "标准清晰"
            else:
                return "详细描述"

        except Exception:
            return "未识别"

    def _generate_recommendations(self, stats: Dict[str, Any], patterns: Dict[str, Any],
                                preferences: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        try:
            # 成功率建议
            if stats.get("success_rate", 1.0) < 0.8:
                recommendations.append("建议使用更具体的命令描述以提高成功率")

            # 效率建议
            if patterns.get("avg_command_length", 0) > 25:
                recommendations.append("您可以使用更简洁的命令表达")

            # 使用时间建议
            most_active_hours = patterns.get("most_active_hours", {})
            if most_active_hours:
                peak_hour = max(most_active_hours.items(), key=lambda x: x[1])[0]
                recommendations.append(f"您在{peak_hour}点最为活跃，可以安排重要任务")

            # 工具使用建议
            preferred_tools = preferences.get("preferred_tools", {})
            if preferred_tools:
                most_used_tool = max(preferred_tools.items(), key=lambda x: x[1])[0]
                recommendations.append(f"您经常使用{most_used_tool}，可以探索更多相关功能")

            return recommendations

        except Exception as e:
            self.logger.error(f"生成建议失败: {e}")
            return []

    def _invalidate_cache(self):
        """清理缓存"""
        self._pattern_cache.clear()
        self._entity_cache.clear()
        self._last_cache_update = None

    def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 从数据库加载历史交互（使用现有的数据管理器）
            # 由于现有的数据库结构，我们需要适配数据格式
            self.logger.info("加载历史数据完成")

        except Exception as e:
            self.logger.warning(f"加载历史数据失败: {e}")

    async def record_interaction(self,
                                interaction_type: str,
                                content: str,
                                context: Dict[str, Any],
                                outcome: str,
                                duration: float,
                                satisfaction_score: Optional[float] = None):
        """记录用户交互"""
        try:
            interaction = UserInteraction(
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                content=content,
                context=context,
                outcome=outcome,
                duration=duration,
                satisfaction_score=satisfaction_score,
                metadata={"recorded_by": "intelligent_learner"}
            )

            # 添加到内存历史
            self.interaction_history.append(interaction)

            # 在线学习
            if self.learning_mode == LearningMode.ONLINE:
                await self._online_learning(interaction)

            # 定期批量学习
            if len(self.interaction_history) % self.retrain_interval == 0:
                await self._batch_learning()

            self.logger.debug(f"记录用户交互: {interaction_type}, 结果: {outcome}")

        except Exception as e:
            self.logger.error(f"记录交互失败: {e}")

    async def _online_learning(self, interaction: UserInteraction):
        """在线学习"""
        try:
            features = interaction.to_features()

            # 更新满意度预测器
            if interaction.satisfaction_score is not None:
                target = interaction.satisfaction_score
                self.satisfaction_predictor.train_step(features, target)

            # 更新成功率预测器
            success_target = 1.0 if interaction.outcome == 'success' else 0.0
            self.success_predictor.train_step(features, success_target)

        except Exception as e:
            self.logger.error(f"在线学习失败: {e}")

    async def _batch_learning(self):
        """批量学习"""
        try:
            if len(self.interaction_history) < self.min_samples_for_training:
                return

            self.logger.info("开始批量学习")

            # 准备训练数据
            interactions = list(self.interaction_history)
            features = np.array([interaction.to_features() for interaction in interactions])

            # 训练满意度预测器
            satisfaction_targets = np.array([
                interaction.satisfaction_score or 0.5
                for interaction in interactions
            ])
            self.satisfaction_predictor.train_batch(features, satisfaction_targets)

            # 训练成功率预测器
            success_targets = np.array([
                1.0 if interaction.outcome == 'success' else 0.0
                for interaction in interactions
            ])
            self.success_predictor.train_batch(features, success_targets)

            # 训练行为聚类器
            self.behavior_clusterer.fit(features)

            # 发现模式
            await self._discover_behavior_patterns()

            self.logger.info("批量学习完成")

        except Exception as e:
            self.logger.error(f"批量学习失败: {e}")

    async def _discover_behavior_patterns(self):
        """发现行为模式"""
        try:
            if not self.behavior_clusterer.is_fitted:
                return

            interactions = list(self.interaction_history)
            features = np.array([interaction.to_features() for interaction in interactions])

            # 聚类
            cluster_labels = self.behavior_clusterer.predict(features)

            # 分析每个聚类
            for cluster_id in range(self.behavior_clusterer.n_clusters):
                cluster_interactions = [
                    interactions[i] for i in range(len(interactions))
                    if cluster_labels[i] == cluster_id
                ]

                if len(cluster_interactions) > 5:  # 只分析有足够数据的聚类
                    pattern = await self._analyze_cluster_pattern(cluster_id, cluster_interactions)
                    if pattern:
                        self.discovered_insights.append(pattern)
                        self.insight_cache[pattern.insight_id] = pattern

        except Exception as e:
            self.logger.error(f"发现行为模式失败: {e}")

    async def _analyze_cluster_pattern(self, cluster_id: int, interactions: List[UserInteraction]) -> Optional[LearningInsight]:
        """分析聚类模式"""
        try:
            # 分析时间模式
            hours = [interaction.timestamp.hour for interaction in interactions]
            avg_hour = np.mean(hours)
            hour_std = np.std(hours)

            # 分析交互类型
            interaction_types = [interaction.interaction_type for interaction in interactions]
            type_counts = Counter(interaction_types)

            # 分析成功率
            success_rate = sum(1 for i in interactions if i.outcome == 'success') / len(interactions)

            # 分析满意度
            satisfactions = [i.satisfaction_score for i in interactions if i.satisfaction_score is not None]
            avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.5

            # 生成洞察
            if hour_std < 3:  # 时间模式稳定
                if 6 <= avg_hour <= 10:
                    pattern_name = "早晨工作模式"
                    recommendations = ["建议启用早晨工作流", "优化早晨的常用功能"]
                elif 14 <= avg_hour <= 18:
                    pattern_name = "下午工作模式"
                    recommendations = ["建议启用下午工作流", "提供下午效率工具"]
                elif 20 <= avg_hour <= 24:
                    pattern_name = "晚间学习模式"
                    recommendations = ["建议启用晚间学习模式", "减少复杂操作"]
                else:
                    pattern_name = "其他时间段模式"
                    recommendations = ["继续观察用户习惯"]
            else:
                pattern_name = "无固定时间模式"
                recommendations = ["建议建立规律作息"]

            # 分析主要交互类型
            if type_counts:
                main_type = type_counts.most_common(1)[0][0]
                if main_type == 'command':
                    recommendations.append("优化常用命令")
                elif main_type == 'workflow':
                    recommendations.append("优化工作流推荐")
                elif main_type == 'query':
                    recommendations.append("改进搜索功能")

            insight = LearningInsight(
                insight_id=f"pattern_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                insight_type="behavior_pattern",
                description=f"发现用户{pattern_name}，成功率{success_rate:.1%}，满意度{avg_satisfaction:.2f}",
                confidence=min(0.9, len(interactions) / 100.0),
                impact_score=success_rate * avg_satisfaction,
                actionable=True,
                recommendations=recommendations,
                valid_until=datetime.now() + timedelta(days=30)
            )

            return insight

        except Exception as e:
            self.logger.error(f"分析聚类模式失败: {e}")
            return None

    async def predict_satisfaction(self, interaction_type: str, content: str, context: Dict[str, Any]) -> float:
        """预测用户满意度"""
        try:
            # 创建虚拟交互用于预测
            dummy_interaction = UserInteraction(
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                content=content,
                context=context,
                outcome="success",  # 假设成功
                duration=10.0
            )

            features = dummy_interaction.to_features()
            prediction = self.satisfaction_predictor.predict(features)

            return float(prediction[0])

        except Exception as e:
            self.logger.error(f"预测满意度失败: {e}")
            return 0.5  # 默认中性预测

    async def predict_success_probability(self, interaction_type: str, content: str, context: Dict[str, Any]) -> float:
        """预测成功概率"""
        try:
            dummy_interaction = UserInteraction(
                timestamp=datetime.now(),
                interaction_type=interaction_type,
                content=content,
                context=context,
                outcome="success",
                duration=10.0
            )

            features = dummy_interaction.to_features()
            prediction = self.success_predictor.predict(features)

            return float(prediction[0])

        except Exception as e:
            self.logger.error(f"预测成功概率失败: {e}")
            return 0.7  # 默认较高概率

    async def generate_personalized_recommendations(self) -> List[PersonalizedRecommendation]:
        """生成个性化推荐"""
        try:
            recommendations = []

            # 基于行为模式的推荐
            pattern_recommendations = await self._generate_pattern_recommendations()
            recommendations.extend(pattern_recommendations)

            # 基于性能优化的推荐
            optimization_recommendations = await self._generate_optimization_recommendations()
            recommendations.extend(optimization_recommendations)

            # 基于功能发现的推荐
            discovery_recommendations = await self._generate_discovery_recommendations()
            recommendations.extend(discovery_recommendations)

            # 排序和去重
            recommendations = self._rank_and_deduplicate_ml_recommendations(recommendations)

            # 更新推荐列表
            self.personalized_recommendations = recommendations

            return recommendations

        except Exception as e:
            self.logger.error(f"生成个性化推荐失败: {e}")
            return []

    async def _generate_pattern_recommendations(self) -> List[PersonalizedRecommendation]:
        """基于行为模式生成推荐"""
        recommendations = []

        try:
            # 分析最近的成功模式
            recent_interactions = list(self.interaction_history)[-50:]

            # 找出最成功的交互类型
            type_success_rates = defaultdict(lambda: [0, 0])  # [success_count, total_count]
            for interaction in recent_interactions:
                type_success_rates[interaction.interaction_type][1] += 1
                if interaction.outcome == 'success':
                    type_success_rates[interaction.interaction_type][0] += 1

            for itype, (success, total) in type_success_rates.items():
                if total >= 5:
                    success_rate = success / total
                    if success_rate > 0.8:
                        recommendation = PersonalizedRecommendation(
                            recommendation_id=f"pattern_success_{itype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            recommendation_type=RecommendationType.WORKFLOW,
                            title=f"优化{itype}使用模式",
                            description=f"您在{itype}方面的成功率为{success_rate:.1%}，建议创建相关工作流",
                            action_items=[
                                f"创建{itype}专用工作流",
                                "设置快捷方式",
                                "优化相关配置"
                            ],
                            confidence=success_rate,
                            priority="medium",
                            estimated_benefit="提高效率20-30%"
                        )
                        recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"生成模式推荐失败: {e}")

        return recommendations

    async def _generate_optimization_recommendations(self) -> List[PersonalizedRecommendation]:
        """生成性能优化推荐"""
        recommendations = []

        try:
            # 分析响应时间
            recent_interactions = list(self.interaction_history)[-20:]
            durations = [i.duration for i in recent_interactions]

            if len(durations) >= 5:
                avg_duration = np.mean(durations)
                if avg_duration > 30:  # 超过30秒
                    recommendation = PersonalizedRecommendation(
                        recommendation_id=f"speed_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        recommendation_type=RecommendationType.OPTIMIZATION,
                        title="性能优化建议",
                        description=f"平均响应时间{avg_duration:.1f}秒，建议进行优化",
                        action_items=[
                            "启用缓存功能",
                            "优化模型选择策略",
                            "调整超时设置"
                        ],
                        confidence=0.7,
                        priority="high",
                        estimated_benefit="减少响应时间40%"
                    )
                    recommendations.append(recommendation)

            # 分析成功率
            success_interactions = [i for i in recent_interactions if i.outcome == 'success']
            success_rate = len(success_interactions) / len(recent_interactions)

            if success_rate < 0.7:
                recommendation = PersonalizedRecommendation(
                    recommendation_id=f"success_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    recommendation_type=RecommendationType.OPTIMIZATION,
                    title="成功率提升建议",
                    description=f"最近成功率为{success_rate:.1%}，建议改进操作流程",
                    action_items=[
                        "改进命令理解",
                        "增加错误处理",
                        "提供更多操作提示"
                    ],
                    confidence=0.8,
                    priority="high",
                    estimated_benefit="提升成功率到90%+"
                )
                recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"生成优化推荐失败: {e}")

        return recommendations

    async def _generate_discovery_recommendations(self) -> List[PersonalizedRecommendation]:
        """生成功能发现推荐"""
        recommendations = []

        try:
            # 分析用户未使用的功能
            used_features = set()
            for interaction in self.interaction_history:
                if 'feature' in interaction.context:
                    used_features.add(interaction.context['feature'])

            # 定义潜在功能列表
            all_features = {
                'workflow_engine': '智能工作流引擎',
                'voice_commands': '语音命令',
                'ai_collaboration': 'AI协作功能',
                'automated_testing': '自动化测试',
                'performance_monitoring': '性能监控',
                'custom_models': '自定义模型',
                'data_export': '数据导出功能',
                'batch_processing': '批量处理'
            }

            unused_features = set(all_features.keys()) - used_features

            for feature in list(unused_features)[:3]:  # 推荐前3个未使用功能
                recommendation = PersonalizedRecommendation(
                    recommendation_id=f"feature_discovery_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    recommendation_type=RecommendationType.FEATURE_DISCOVERY,
                    title=f"发现新功能: {all_features[feature]}",
                    description=f"您还没有使用过{all_features[feature]}功能，建议尝试",
                    action_items=[
                        f"了解{all_features[feature]}功能",
                        "查看使用教程",
                        "尝试示例用例"
                    ],
                    confidence=0.6,
                    priority="low",
                    estimated_benefit="扩展功能使用范围"
                )
                recommendations.append(recommendation)

        except Exception as e:
            self.logger.error(f"生成发现推荐失败: {e}")

        return recommendations

    def _rank_and_deduplicate_ml_recommendations(self, recommendations: List[PersonalizedRecommendation]) -> List[PersonalizedRecommendation]:
        """排序和去重推荐"""
        # 按优先级和置信度排序
        priority_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

        def sort_key(rec):
            return (priority_weights.get(rec.priority, 1), rec.confidence)

        # 排序
        sorted_recommendations = sorted(recommendations, key=sort_key, reverse=True)

        # 去重（基于标题）
        seen_titles = set()
        unique_recommendations = []

        for rec in sorted_recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)

        # 返回前10个推荐
        return unique_recommendations[:10]

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        try:
            total_interactions = len(self.interaction_history)
            recent_interactions = list(self.interaction_history)[-100:]

            if not recent_interactions:
                return {"message": "暂无学习数据"}

            # 基本统计
            success_rate = sum(1 for i in recent_interactions if i.outcome == 'success') / len(recent_interactions)
            avg_duration = np.mean([i.duration for i in recent_interactions])
            avg_satisfaction = np.mean([i.satisfaction_score for i in recent_interactions if i.satisfaction_score is not None])

            # 交互类型统计
            type_counts = Counter(interaction.interaction_type for interaction in recent_interactions)

            # 时间模式统计
            hour_counts = Counter(interaction.timestamp.hour for interaction in recent_interactions)
            peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 0

            # 学习模型状态
            model_status = {
                "satisfaction_predictor_trained": self.satisfaction_predictor.is_trained,
                "success_predictor_trained": self.success_predictor.is_trained,
                "behavior_clusterer_fitted": self.behavior_clusterer.is_fitted
            }

            # 洞察统计
            recent_insights = [insight for insight in self.discovered_insights if insight.discovered_at > datetime.now() - timedelta(days=7)]

            return {
                "total_interactions": total_interactions,
                "recent_interactions": len(recent_interactions),
                "success_rate": success_rate,
                "average_duration": avg_duration,
                "average_satisfaction": avg_satisfaction,
                "interaction_type_distribution": dict(type_counts),
                "peak_activity_hour": peak_hour,
                "model_status": model_status,
                "total_insights": len(self.discovered_insights),
                "recent_insights": len(recent_insights),
                "active_recommendations": len(self.personalized_recommendations),
                "learning_mode": self.learning_mode.value
            }

        except Exception as e:
            self.logger.error(f"获取学习统计失败: {e}")
            return {"error": str(e)}

    async def set_learning_mode(self, mode: LearningMode):
        """设置学习模式"""
        self.learning_mode = mode
        self.logger.info(f"学习模式设置为: {mode.value}")


# 导出
__all__ = [
    "IntelligentLearner",
    "LearningMode",
    "RecommendationType",
    "UserBehaviorPattern",
    "UserInteraction",
    "LearningInsight",
    "PersonalizedRecommendation",
    "SimpleMLModel",
    "ClusteringModel"
]