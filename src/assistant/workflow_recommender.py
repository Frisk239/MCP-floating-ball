"""
MCP Floating Ball - 个性化工作流推荐系统

基于用户行为模式和偏好，智能推荐最适合的工作流。
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics
import math

from src.core.logging import get_logger
from src.core.database import get_database
from src.assistant.intelligent_learner import IntelligentLearner, UserInteraction
from src.assistant.workflow_manager import get_workflow_manager
from src.assistant.task_dispatcher import TaskDispatcher, TaskProfile, TaskCategory

logger = get_logger("workflow_recommender")


class RecommendationStrategy(Enum):
    """推荐策略"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    POPULARITY_BASED = "popularity_based"
    CONTEXT_AWARE = "context_aware"
    ML_PREDICTIVE = "ml_predictive"


class RecommendationType(Enum):
    """推荐类型"""
    WORKFLOW_SUGGESTION = "workflow_suggestion"
    SIMILAR_WORKFLOW = "similar_workflow"
    TRENDING_WORKFLOW = "trending_workflow"
    PERSONALIZED_WORKFLOW = "personalized_workflow"
    CONTEXT_WORKFLOW = "context_workflow"


@dataclass
class WorkflowUsageStats:
    """工作流使用统计"""
    workflow_id: str
    usage_count: int = 0
    success_count: int = 0
    total_duration: float = 0.0
    avg_satisfaction: float = 0.0
    last_used: Optional[datetime] = None
    user_ratings: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def avg_duration(self) -> float:
        """平均执行时间"""
        return self.total_duration / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def avg_rating(self) -> float:
        """平均评分"""
        return statistics.mean(self.user_ratings) if self.user_ratings else 0.0


@dataclass
class UserWorkflowProfile:
    """用户工作流档案"""
    user_id: str
    preferred_categories: Dict[str, float] = field(default_factory=dict)
    usage_patterns: Dict[str, Any] = field(default_factory=dict)
    skill_level: float = 0.5
    preferred_complexity: str = "medium"
    time_preferences: Dict[str, float] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkflowRecommendation:
    """工作流推荐"""
    recommendation_id: str
    workflow_id: str
    workflow_name: str
    recommendation_type: RecommendationType
    confidence: float
    explanation: str
    predicted_satisfaction: float
    estimated_duration: float
    tags: List[str] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    similar_users_used: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "recommendation_type": self.recommendation_type.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "predicted_satisfaction": self.predicted_satisfaction,
            "estimated_duration": self.estimated_duration,
            "tags": self.tags,
            "similar_users_used": self.similar_users_used,
            "created_at": self.created_at.isoformat()
        }


class CollaborativeFilteringRecommender:
    """协同过滤推荐器"""

    def __init__(self):
        self.user_item_matrix: Dict[str, Dict[str, float]] = {}  # user_id -> workflow_id -> rating
        self.workflow_similarity: Dict[str, Dict[str, float]] = {}

    def update_user_rating(self, user_id: str, workflow_id: str, rating: float):
        """更新用户评分"""
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = {}
        self.user_item_matrix[user_id][workflow_id] = rating

    def calculate_user_similarity(self, user1: str, user2: str) -> float:
        """计算用户相似度"""
        if user1 not in self.user_item_matrix or user2 not in self.user_item_matrix:
            return 0.0

        ratings1 = self.user_item_matrix[user1]
        ratings2 = self.user_item_matrix[user2]

        # 找到共同评分的工作流
        common_workflows = set(ratings1.keys()) & set(ratings2.keys())

        if not common_workflows:
            return 0.0

        # 计算皮尔逊相关系数
        numerator = 0.0
        sum1 = sum2 = sum1_sq = sum2_sq = 0.0

        for workflow_id in common_workflows:
            r1 = ratings1[workflow_id]
            r2 = ratings2[workflow_id]

            numerator += r1 * r2
            sum1 += r1
            sum2 += r2
            sum1_sq += r1 * r2
            sum2_sq += r2 * r2

        if len(common_workflows) == 0:
            return 0.0

        denominator = math.sqrt((sum1_sq - sum1 * sum1 / len(common_workflows)) *
                               (sum2_sq - sum2 * sum2 / len(common_workflows)))

        if denominator == 0:
            return 0.0

        return (numerator - sum1 * sum2 / len(common_workflows)) / denominator

    def recommend_workflows(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """基于协同过滤推荐工作流"""
        if user_id not in self.user_item_matrix:
            return []

        # 找到相似用户
        similar_users = []
        for other_user in self.user_item_matrix:
            if other_user != user_id:
                similarity = self.calculate_user_similarity(user_id, other_user)
                if similarity > 0:
                    similar_users.append((other_user, similarity))

        # 按相似度排序
        similar_users.sort(key=lambda x: x[1], reverse=True)

        # 生成推荐
        user_workflows = set(self.user_item_matrix[user_id].keys())
        workflow_scores = defaultdict(float)

        for similar_user, similarity in similar_users[:10]:  # 取前10个相似用户
            for workflow_id, rating in self.user_item_matrix[similar_user].items():
                if workflow_id not in user_workflows:
                    workflow_scores[workflow_id] += similarity * rating

        # 排序并返回前n个推荐
        recommendations = sorted(workflow_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n]


class ContentBasedRecommender:
    """基于内容的推荐器"""

    def __init__(self):
        self.workflow_features: Dict[str, Dict[str, Any]] = {}
        self.user_profiles: Dict[str, Dict[str, float]] = {}

    def add_workflow_features(self, workflow_id: str, features: Dict[str, Any]):
        """添加工作流特征"""
        self.workflow_features[workflow_id] = features

    def update_user_profile(self, user_id: str, workflow_id: str, rating: float):
        """更新用户档案"""
        if workflow_id not in self.workflow_features:
            return

        features = self.workflow_features[workflow_id]
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = defaultdict(float)

        # 基于评分更新特征偏好
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                self.user_profiles[user_id][feature_name] += rating * feature_value
            elif isinstance(feature_value, str):
                # 处理文本特征
                self.user_profiles[user_id][f"text_{feature_name}"] += rating

    def recommend_workflows(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """基于内容推荐工作流"""
        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]
        workflow_scores = []

        for workflow_id, features in self.workflow_features.items():
            score = 0.0
            feature_count = 0

            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)) and feature_name in user_profile:
                    score += user_profile[feature_name] * feature_value
                    feature_count += 1

            if feature_count > 0:
                workflow_scores.append((workflow_id, score / feature_count))

        # 排序并返回前n个推荐
        workflow_scores.sort(key=lambda x: x[1], reverse=True)
        return workflow_scores[:n]


class WorkflowRecommender:
    """个性化工作流推荐系统"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.db = get_database()

        # 获取其他组件
        self.workflow_manager = get_workflow_manager()
        self.intelligent_learner = IntelligentLearner()
        self.task_dispatcher = TaskDispatcher()

        # 推荐器组件
        self.collaborative_filter = CollaborativeFilteringRecommender()
        self.content_based = ContentBasedRecommender()

        # 数据存储
        self.workflow_stats: Dict[str, WorkflowUsageStats] = defaultdict(WorkflowUsageStats)
        self.user_profiles: Dict[str, UserWorkflowProfile] = defaultdict(UserWorkflowProfile)

        # 推荐缓存
        self.recommendation_cache: Dict[str, List[WorkflowRecommendation]] = {}
        self.cache_ttl = timedelta(hours=1)

        # 加载历史数据
        self._load_historical_data()

        self.logger.info("工作流推荐系统初始化完成")

    def _load_historical_data(self):
        """加载历史数据"""
        try:
            # 从数据库加载工作流使用记录
            # 这里是模拟数据，实际应该从数据库加载
            self._populate_mock_data()

            self.logger.info("历史数据加载完成")

        except Exception as e:
            self.logger.error(f"加载历史数据失败: {e}")

    def _populate_mock_data(self):
        """填充模拟数据"""
        # 模拟一些工作流特征
        workflow_features = {
            "smart_document_processor": {
                "category": "document_processing",
                "complexity": 0.7,
                "automation_level": 0.9,
                "ai_features": 0.8,
                "estimated_duration": 45.0
            },
            "web_researcher": {
                "category": "research",
                "complexity": 0.6,
                "automation_level": 0.8,
                "ai_features": 0.9,
                "estimated_duration": 60.0
            },
            "system_maintenance": {
                "category": "system",
                "complexity": 0.4,
                "automation_level": 0.95,
                "ai_features": 0.5,
                "estimated_duration": 30.0
            },
            "voice_assistant_automation": {
                "category": "automation",
                "complexity": 0.3,
                "automation_level": 0.85,
                "ai_features": 0.7,
                "estimated_duration": 15.0
            }
        }

        for workflow_id, features in workflow_features.items():
            self.content_based.add_workflow_features(workflow_id, features)

    async def record_workflow_usage(self, user_id: str, workflow_id: str, success: bool, duration: float, satisfaction: Optional[float] = None, rating: Optional[float] = None):
        """记录工作流使用情况"""
        try:
            # 更新使用统计
            stats = self.workflow_stats[workflow_id]
            stats.usage_count += 1
            if success:
                stats.success_count += 1
            stats.total_duration += duration
            stats.last_used = datetime.now()

            if satisfaction is not None:
                # 更新平均满意度
                total_satisfaction = stats.avg_satisfaction * (stats.usage_count - 1) + satisfaction
                stats.avg_satisfaction = total_satisfaction / stats.usage_count

            if rating is not None:
                stats.user_ratings.append(rating)

            # 更新协同过滤数据
            if rating is not None:
                self.collaborative_filter.update_user_rating(user_id, workflow_id, rating)

            # 更新基于内容的用户档案
            if rating is not None:
                self.content_based.update_user_profile(user_id, workflow_id, rating)

            # 更新用户工作流档案
            await self._update_user_workflow_profile(user_id, workflow_id, success, duration, satisfaction, rating)

            self.logger.debug(f"记录工作流使用: {user_id} -> {workflow_id}")

        except Exception as e:
            self.logger.error(f"记录工作流使用失败: {e}")

    async def _update_user_workflow_profile(self, user_id: str, workflow_id: str, success: bool, duration: float, satisfaction: Optional[float], rating: Optional[float]):
        """更新用户工作流档案"""
        try:
            profile = self.user_profiles[user_id]

            # 获取工作流信息
            workflows = await self.workflow_manager.get_available_workflows()
            workflow_info = next((w for w in workflows if w["id"] == workflow_id), None)

            if workflow_info:
                # 更新类别偏好
                for tag in workflow_info.get("tags", []):
                    if tag not in profile.preferred_categories:
                        profile.preferred_categories[tag] = 0.0
                    profile.preferred_categories[tag] += (satisfaction or rating or 0.5) * 0.1

                # 更新技能水平（基于成功率）
                if success:
                    profile.skill_level = min(1.0, profile.skill_level + 0.01)

            # 更新使用模式
            if "usage_frequency" not in profile.usage_patterns:
                profile.usage_patterns["usage_frequency"] = defaultdict(int)
            profile.usage_patterns["usage_frequency"][workflow_id] += 1

            if rating is not None:
                profile.feedback_history.append({
                    "workflow_id": workflow_id,
                    "rating": rating,
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "duration": duration
                })

        except Exception as e:
            self.logger.error(f"更新用户工作流档案失败: {e}")

    async def get_personalized_recommendations(self, user_id: str, context: Optional[Dict[str, Any]] = None, n: int = 10, strategy: RecommendationStrategy = RecommendationStrategy.HYBRID) -> List[WorkflowRecommendation]:
        """获取个性化推荐"""
        try:
            # 检查缓存
            cache_key = f"{user_id}_{strategy.value}_{n}"
            if cache_key in self.recommendation_cache:
                cached_recommendations = self.recommendation_cache[cache_key]
                if cached_recommendations and cached_recommendations[0].created_at > datetime.now() - self.cache_ttl:
                    return cached_recommendations

            # 根据策略生成推荐
            if strategy == RecommendationStrategy.COLLABORATIVE_FILTERING:
                recommendations = await self._collaborative_filtering_recommendations(user_id, n, context)
            elif strategy == RecommendationStrategy.CONTENT_BASED:
                recommendations = await self._content_based_recommendations(user_id, n, context)
            elif strategy == RecommendationStrategy.HYBRID:
                recommendations = await self._hybrid_recommendations(user_id, n, context)
            elif strategy == RecommendationStrategy.POPULARITY_BASED:
                recommendations = await self._popularity_based_recommendations(user_id, n, context)
            elif strategy == RecommendationStrategy.CONTEXT_AWARE:
                recommendations = await self._context_aware_recommendations(user_id, n, context)
            elif strategy == RecommendationStrategy.ML_PREDICTIVE:
                recommendations = await self._ml_predictive_recommendations(user_id, n, context)
            else:
                recommendations = await self._hybrid_recommendations(user_id, n, context)

            # 缓存结果
            self.recommendation_cache[cache_key] = recommendations

            return recommendations

        except Exception as e:
            self.logger.error(f"获取个性化推荐失败: {e}")
            return []

    async def _collaborative_filtering_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """协同过滤推荐"""
        try:
            cf_recommendations = self.collaborative_filter.recommend_workflows(user_id, n)

            recommendations = []
            for workflow_id, score in cf_recommendations:
                stats = self.workflow_stats[workflow_id]

                recommendation = WorkflowRecommendation(
                    recommendation_id=f"cf_{user_id}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    workflow_id=workflow_id,
                    workflow_name=await self._get_workflow_name(workflow_id),
                    recommendation_type=RecommendationType.SIMILAR_WORKFLOW,
                    confidence=min(1.0, score),
                    explanation=f"与您相似的用户也喜欢这个工作流",
                    predicted_satisfaction=stats.avg_rating or 0.7,
                    estimated_duration=stats.avg_duration or 30.0,
                    similar_users_used=int(score * 10)  # 估算相似用户数量
                )

                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            self.logger.error(f"协同过滤推荐失败: {e}")
            return []

    async def _content_based_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """基于内容的推荐"""
        try:
            cb_recommendations = self.content_based.recommend_workflows(user_id, n)

            recommendations = []
            for workflow_id, score in cb_recommendations:
                stats = self.workflow_stats[workflow_id]
                profile = self.user_profiles[user_id]

                recommendation = WorkflowRecommendation(
                    recommendation_id=f"cb_{user_id}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    workflow_id=workflow_id,
                    workflow_name=await self._get_workflow_name(workflow_id),
                    recommendation_type=RecommendationType.PERSONALIZED_WORKFLOW,
                    confidence=min(1.0, score / 10.0),
                    explanation=f"基于您的历史偏好推荐",
                    predicted_satisfaction=stats.avg_rating or 0.7,
                    estimated_duration=stats.avg_duration or 30.0,
                    context_factors={
                        "user_skill_level": profile.skill_level,
                        "match_score": score
                    }
                )

                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            self.logger.error(f"基于内容的推荐失败: {e}")
            return []

    async def _hybrid_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """混合推荐"""
        try:
            # 获取多种推荐策略的结果
            cf_recs = await self._collaborative_filtering_recommendations(user_id, n, context)
            cb_recs = await self._content_based_recommendations(user_id, n, context)
            popular_recs = await self._popularity_based_recommendations(user_id, n, context)

            # 合并推荐结果
            all_recommendations = defaultdict(lambda: {"score": 0, "sources": []})

            # 协同过滤权重 0.4
            for rec in cf_recs:
                all_recommendations[rec.workflow_id]["score"] += rec.confidence * 0.4
                all_recommendations[rec.workflow_id]["sources"].append("collaborative")

            # 基于内容权重 0.4
            for rec in cb_recs:
                all_recommendations[rec.workflow_id]["score"] += rec.confidence * 0.4
                all_recommendations[rec.workflow_id]["sources"].append("content")

            # 热门推荐权重 0.2
            for rec in popular_recs:
                all_recommendations[rec.workflow_id]["score"] += rec.confidence * 0.2
                all_recommendations[rec.workflow_id]["sources"].append("popularity")

            # 排序并生成最终推荐
            sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1]["score"], reverse=True)

            hybrid_recommendations = []
            for workflow_id, data in sorted_recommendations[:n]:
                stats = self.workflow_stats[workflow_id]

                recommendation = WorkflowRecommendation(
                    recommendation_id=f"hybrid_{user_id}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    workflow_id=workflow_id,
                    workflow_name=await self._get_workflow_name(workflow_id),
                    recommendation_type=RecommendationType.WORKFLOW_SUGGESTION,
                    confidence=min(1.0, data["score"]),
                    explanation=f"基于多种算法的智能推荐（{', '.join(data['sources'])}）",
                    predicted_satisfaction=stats.avg_rating or 0.7,
                    estimated_duration=stats.avg_duration or 30.0,
                    context_factors={
                        "recommendation_sources": data["sources"],
                        "combined_score": data["score"]
                    }
                )

                hybrid_recommendations.append(recommendation)

            return hybrid_recommendations

        except Exception as e:
            self.logger.error(f"混合推荐失败: {e}")
            return []

    async def _popularity_based_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """基于热门度的推荐"""
        try:
            # 按使用量和评分排序
            sorted_workflows = sorted(
                self.workflow_stats.items(),
                key=lambda x: (x[1].usage_count * x[1].avg_rating),
                reverse=True
            )

            recommendations = []
            for workflow_id, stats in sorted_workflows[:n]:
                recommendation = WorkflowRecommendation(
                    recommendation_id=f"popular_{user_id}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    workflow_id=workflow_id,
                    workflow_name=await self._get_workflow_name(workflow_id),
                    recommendation_type=RecommendationType.TRENDING_WORKFLOW,
                    confidence=min(1.0, (stats.usage_count / 100.0) * stats.avg_rating),
                    explanation=f"热门工作流，已被使用{stats.usage_count}次",
                    predicted_satisfaction=stats.avg_rating,
                    estimated_duration=stats.avg_duration,
                    similar_users_used=stats.usage_count
                )

                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            self.logger.error(f"热门推荐失败: {e}")
            return []

    async def _context_aware_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """上下文感知推荐"""
        try:
            if not context:
                return await self._hybrid_recommendations(user_id, n, context)

            # 基于上下文信息调整推荐
            time_of_day = datetime.now().hour
            user_profile = self.user_profiles[user_id]

            # 根据时间调整
            if 6 <= time_of_day <= 10:  # 早晨
                preferred_complexity = 0.3  # 简单任务
                time_context = "早晨"
            elif 14 <= time_of_day <= 18:  # 下午
                preferred_complexity = 0.7  # 复杂任务
                time_context = "下午"
            else:  # 其他时间
                preferred_complexity = 0.5
                time_context = "其他时间"

            # 获取基础推荐
            base_recommendations = await self._hybrid_recommendations(user_id, n * 2, context)

            # 根据上下文重新排序
            context_scores = {}
            for rec in base_recommendations:
                context_score = 1.0

                # 复杂度匹配
                workflow_stats = self.workflow_stats[rec.workflow_id]
                if workflow_stats.avg_duration < 30:  # 简单
                    if preferred_complexity < 0.5:
                        context_score += 0.2
                else:  # 复杂
                    if preferred_complexity > 0.5:
                        context_score += 0.2

                # 用户技能匹配
                if user_profile.skill_level > 0.7:  # 高级用户
                    context_score += 0.1

                context_scores[rec.workflow_id] = context_score

            # 重新排序
            base_recommendations.sort(key=lambda x: x.confidence * context_scores.get(x.workflow_id, 1.0), reverse=True)

            # 生成上下文感知推荐
            context_recommendations = []
            for rec in base_recommendations[:n]:
                rec.recommendation_type = RecommendationType.CONTEXT_WORKFLOW
                rec.explanation += f"（适合{time_context}使用）"
                rec.context_factors["time_context"] = time_context
                rec.context_factors["preferred_complexity"] = preferred_complexity

                context_recommendations.append(rec)

            return context_recommendations

        except Exception as e:
            self.logger.error(f"上下文感知推荐失败: {e}")
            return []

    async def _ml_predictive_recommendations(self, user_id: str, n: int, context: Optional[Dict[str, Any]]) -> List[WorkflowRecommendation]:
        """机器学习预测推荐"""
        try:
            # 使用智能学习器预测满意度
            user_profile = self.user_profiles[user_id]
            workflows = await self.workflow_manager.get_available_workflows()

            ml_recommendations = []
            for workflow in workflows:
                workflow_id = workflow["id"]

                # 预测满意度
                predicted_satisfaction = await self.intelligent_learner.predict_satisfaction(
                    "workflow_execution",
                    f"执行工作流 {workflow['name']}",
                    {"workflow_id": workflow_id, "complexity": workflow.get("task_count", 0)}
                )

                # 预测成功概率
                success_probability = await self.intelligent_learner.predict_success_probability(
                    "workflow_execution",
                    f"执行工作流 {workflow['name']}",
                    {"workflow_id": workflow_id, "complexity": workflow.get("task_count", 0)}
                )

                # 综合评分
                confidence = (predicted_satisfaction + success_probability) / 2

                if confidence > 0.6:  # 只推荐高置信度的
                    recommendation = WorkflowRecommendation(
                        recommendation_id=f"ml_{user_id}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        workflow_id=workflow_id,
                        workflow_name=workflow["name"],
                        recommendation_type=RecommendationType.PERSONALIZED_WORKFLOW,
                        confidence=confidence,
                        explanation=f"基于机器学习预测，满意度{predicted_satisfaction:.2f}，成功率{success_probability:.2f}",
                        predicted_satisfaction=predicted_satisfaction,
                        estimated_duration=workflow.get("estimated_duration", 30.0),
                        context_factors={
                            "ml_confidence": confidence,
                            "success_probability": success_probability
                        }
                    )

                    ml_recommendations.append(recommendation)

            # 排序并返回前n个
            ml_recommendations.sort(key=lambda x: x.confidence, reverse=True)
            return ml_recommendations[:n]

        except Exception as e:
            self.logger.error(f"机器学习推荐失败: {e}")
            return []

    async def _get_workflow_name(self, workflow_id: str) -> str:
        """获取工作流名称"""
        try:
            workflows = await self.workflow_manager.get_available_workflows()
            workflow = next((w for w in workflows if w["id"] == workflow_id), None)
            return workflow["name"] if workflow else workflow_id
        except Exception:
            return workflow_id

    async def get_recommendation_explanation(self, recommendation_id: str) -> Dict[str, Any]:
        """获取推荐解释"""
        try:
            # 在所有缓存推荐中查找
            for recommendations in self.recommendation_cache.values():
                for rec in recommendations:
                    if rec.recommendation_id == recommendation_id:
                        return {
                            "recommendation": rec.to_dict(),
                            "explanation_details": {
                                "algorithm": rec.recommendation_type.value,
                                "confidence_factors": self._get_confidence_factors(rec),
                                "similar_workflows": await self._get_similar_workflows(rec.workflow_id),
                                "user_feedback": await self._get_user_feedback_for_workflow(rec.workflow_id)
                            }
                        }

            return {"error": "推荐不存在"}

        except Exception as e:
            self.logger.error(f"获取推荐解释失败: {e}")
            return {"error": str(e)}

    def _get_confidence_factors(self, recommendation: WorkflowRecommendation) -> List[str]:
        """获取置信度因素"""
        factors = []

        if recommendation.confidence > 0.8:
            factors.append("高置信度推荐")
        elif recommendation.confidence > 0.6:
            factors.append("中等置信度推荐")
        else:
            factors.append("探索性推荐")

        if recommendation.similar_users_used > 10:
            factors.append(f"{recommendation.similar_users_used}个相似用户使用过")

        if recommendation.predicted_satisfaction > 0.8:
            factors.append("预测满意度高")

        return factors

    async def _get_similar_workflows(self, workflow_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """获取相似工作流"""
        try:
            # 简化实现：基于类别找相似工作流
            workflows = await self.workflow_manager.get_available_workflows()
            target_workflow = next((w for w in workflows if w["id"] == workflow_id), None)

            if not target_workflow:
                return []

            similar_workflows = []
            for workflow in workflows:
                if workflow["id"] != workflow_id:
                    # 计算相似度（基于标签）
                    target_tags = set(target_workflow.get("tags", []))
                    workflow_tags = set(workflow.get("tags", []))
                    similarity = len(target_tags & workflow_tags) / len(target_tags | workflow_tags) if target_tags | workflow_tags else 0

                    if similarity > 0.3:  # 相似度阈值
                        similar_workflows.append({
                            "workflow_id": workflow["id"],
                            "workflow_name": workflow["name"],
                            "similarity": similarity
                        })

            # 按相似度排序
            similar_workflows.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_workflows[:limit]

        except Exception as e:
            self.logger.error(f"获取相似工作流失败: {e}")
            return []

    async def _get_user_feedback_for_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流的用户反馈"""
        try:
            stats = self.workflow_stats[workflow_id]

            return {
                "usage_count": stats.usage_count,
                "success_rate": stats.success_rate,
                "avg_rating": stats.avg_rating,
                "avg_satisfaction": stats.avg_satisfaction,
                "avg_duration": stats.avg_duration,
                "last_used": stats.last_used.isoformat() if stats.last_used else None
            }

        except Exception as e:
            self.logger.error(f"获取用户反馈失败: {e}")
            return {}

    async def get_user_recommendation_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """获取用户推荐历史"""
        try:
            # 这里应该从数据库获取实际的历史记录
            # 目前返回模拟数据
            return [
                {
                    "recommendation_id": f"hist_{i}",
                    "workflow_id": f"workflow_{i}",
                    "workflow_name": f"工作流 {i}",
                    "recommendation_type": "workflow_suggestion",
                    "clicked": i % 3 == 0,  # 模拟点击
                    "executed": i % 4 == 0,  # 模拟执行
                    "rating": (i % 5) + 1,
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat()
                }
                for i in range(min(limit, 10))
            ]

        except Exception as e:
            self.logger.error(f"获取推荐历史失败: {e}")
            return []

    async def update_recommendation_feedback(self, user_id: str, recommendation_id: str, feedback: Dict[str, Any]):
        """更新推荐反馈"""
        try:
            # 记录用户对推荐的反馈
            self.logger.info(f"更新推荐反馈: {user_id} -> {recommendation_id}, 反馈: {feedback}")

            # 根据反馈调整推荐算法参数
            if feedback.get("rating"):
                # 可以根据评分调整推荐权重
                pass

            if feedback.get("clicked"):
                # 记录点击行为
                pass

            # 清除相关缓存
            keys_to_remove = [key for key in self.recommendation_cache.keys() if user_id in key]
            for key in keys_to_remove:
                del self.recommendation_cache[key]

        except Exception as e:
            self.logger.error(f"更新推荐反馈失败: {e}")


# 导出
__all__ = [
    "WorkflowRecommender",
    "WorkflowRecommendation",
    "WorkflowUsageStats",
    "UserWorkflowProfile",
    "RecommendationStrategy",
    "RecommendationType",
    "CollaborativeFilteringRecommender",
    "ContentBasedRecommender"
]