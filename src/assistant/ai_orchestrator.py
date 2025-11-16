"""
MCP Floating Ball - AI模型编排器

实现多AI模型协作，基于Eachere项目的智能调度理念。
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import random
import statistics
import logging

from src.core.logging import get_logger
from src.core.database import get_database_manager
from src.core.exceptions import MCPFloatingBallError


class FusionStrategy(Enum):
    """融合策略"""
    CONFIDENCE_BASED = "confidence_based"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    EXPERT_WEIGHTED = "expert_weighted"
    CONDORCET = "condorcet"
    BORDA_COUNT = "borda_count"
    DEMPSTER_SHAFER = "dempster_shafer"
    BAYESIAN_FUSION = "bayesian_fusion"
    NEURAL_ENSEMBLE = "neural_ensemble"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    CONFLICT_RESOLUTION = "conflict_resolution"
    HIERARCHICAL_FUSION = "hierarchical_fusion"


@dataclass
class AIModelResponse:
    """AI模型响应"""
    model_id: str
    model_output: str
    confidence: Optional[float] = None
    execution_time: Optional[float] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)



logger = get_logger("ai_orchestrator")


class AIModelCapabilities(Enum):
    """AI模型能力枚举"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CONTEXT_UNDERSTANDING = "context_understanding"
    MULTIMODAL = "multimodal"
    REAL_TIME_PROCESSING = "real_time_processing"
    LONG_CONTEXT = "long_context"
    FACTUAL_ACCURACY = "factual_accuracy"
    VISION_UNDERSTANDING = "vision_understanding"
    SPEECH_PROCESSING = "speech_processing"
    WEB_SEARCH = "web_search"


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"


class FusionStrategy(Enum):
    """融合策略"""
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    SEQUENTIAL_REFINEMENT = "sequential_refinement"
    EXPERTISE_WEIGHTED = "expertise_weighted"


@dataclass
class AIModelProfile:
    """AI模型配置文件"""
    model_id: str
    model_name: str
    provider: str  # kimi, dashscope, metaso, etc.
    capabilities: List[AIModelCapabilities]
    strengths: List[str]  # 模型优势
    weaknesses: List[str]  # 模型弱点
    cost_tier: str  # low, medium, high
    performance_tier: str  # basic, standard, premium
    max_tokens: int
    response_time: float  # 平均响应时间（秒）
    accuracy_score: float  # 准确性评分 (0-1)
    reliability_score: float  # 可靠性评分 (0-1)
    context_window: int  # 上下文窗口大小
    specialized_domains: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TaskRequirements:
    """任务需求"""
    task_id: str
    task_type: str
    complexity: TaskComplexity
    required_capabilities: List[AIModelCapabilities]
    specialized_domain: Optional[str] = None
    max_response_time: Optional[float] = None
    accuracy_requirement: Optional[float] = None
    cost_sensitivity: Optional[float] = None  # 成本敏感度 0-1
    priority: str = "normal"  # low, normal, high, urgent
    context_size: Optional[int] = None
    expected_output_length: Optional[int] = None


@dataclass
class ModelExecutionResult:
    """模型执行结果"""
    model_id: str
    success: bool
    response: Any
    confidence: float
    execution_time: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def model_output(self) -> Any:
        """兼容性属性，返回response"""
        return self.response


@dataclass
class FusionResult:
    """融合结果"""
    fused_response: Any
    confidence: float
    contributing_models: List[str]
    fusion_strategy: str
    execution_time: float = 0.0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fused_output(self) -> Any:
        """兼容性属性，返回fused_response"""
        return self.fused_response

    @property
    def fused_result(self) -> Any:
        """兼容性属性，返回fused_response"""
        return self.fused_response

    @property
    def fusion_method(self) -> str:
        """兼容性属性，返回fusion_strategy"""
        return self.fusion_strategy


class AIOrchestrator:
    """AI模型编排器"""

    def __init__(self):
        self.db = get_database_manager()
        self.models: Dict[str, AIModelProfile] = {}
        self.performance_history: Dict[str, List[ModelExecutionResult]] = {}
        self.ab_testing_results: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger(self.__class__.__name__)

        # 初始化模型配置
        self._initialize_models()

        # 启动后台监控
        self._start_background_monitoring()

        self.logger.info("AI模型编排器初始化完成")

    def _initialize_models(self):
        """初始化AI模型配置"""
        try:
            # Kimi模型 - 长上下文和推理优势
            self.models["kimi"] = AIModelProfile(
                model_id="kimi",
                model_name="Moonshot AI (Kimi)",
                provider="kimi",
                capabilities=[
                    AIModelCapabilities.LONG_CONTEXT,
                    AIModelCapabilities.REASONING,
                    AIModelCapabilities.CODE_GENERATION,
                    AIModelCapabilities.DATA_ANALYSIS,
                    AIModelCapabilities.CONTEXT_UNDERSTANDING
                ],
                strengths=[
                    "超长上下文处理",
                    "深度推理能力",
                    "代码生成和分析",
                    "数据分析和处理",
                    "上下文理解"
                ],
                weaknesses=[
                    "实时处理能力一般",
                    "创造性写作中等",
                    "成本较高"
                ],
                cost_tier="medium",
                performance_tier="premium",
                max_tokens=200000,
                response_time=3.5,
                accuracy_score=0.92,
                reliability_score=0.89,
                context_window=200000,
                specialized_domains=["programming", "data_analysis", "research"]
            )

            # DashScope模型 - 实时和多模态优势
            self.models["dashscope"] = AIModelProfile(
                model_id="dashscope",
                model_name="阿里云DashScope",
                provider="dashscope",
                capabilities=[
                    AIModelCapabilities.MULTIMODAL,
                    AIModelCapabilities.REAL_TIME_PROCESSING,
                    AIModelCapabilities.VISION_UNDERSTANDING,
                    AIModelCapabilities.SPEECH_PROCESSING,
                    AIModelCapabilities.QUESTION_ANSWERING
                ],
                strengths=[
                    "实时处理能力强",
                    "多模态支持",
                    "语音识别和合成",
                    "图像理解",
                    "实时响应"
                ],
                weaknesses=[
                    "上下文窗口较小",
                    "深度推理能力有限",
                    "长文档处理能力一般"
                ],
                cost_tier="low",
                performance_tier="standard",
                max_tokens=8000,
                response_time=1.2,
                accuracy_score=0.85,
                reliability_score=0.94,
                context_window=8000,
                specialized_domains=["multimodal", "real_time", "speech", "vision"]
            )

            # Metaso模型 - 网络搜索和实时信息优势
            self.models["metaso"] = AIModelProfile(
                model_id="metaso",
                model_name="秘塔AI搜索",
                provider="metaso",
                capabilities=[
                    AIModelCapabilities.WEB_SEARCH,
                    AIModelCapabilities.REAL_TIME_PROCESSING,
                    AIModelCapabilities.FACTUAL_ACCURACY,
                    AIModelCapabilities.QUESTION_ANSWERING
                ],
                strengths=[
                    "强大的网络搜索能力",
                    "实时信息获取",
                    "事实准确性高",
                    "问答能力强"
                ],
                weaknesses=[
                    "创造性能力有限",
                    "长文档处理弱",
                    "深度推理一般"
                ],
                cost_tier="medium",
                performance_tier="standard",
                max_tokens=4000,
                response_time=1.8,
                accuracy_score=0.88,
                reliability_score=0.91,
                context_window=4000,
                specialized_domains=["web_search", "real_time_info", "fact_checking"]
            )

            # 添加本地模型配置（如果有的话）
            # self.models["local_llm"] = AIModelProfile(...)

            self.logger.info(f"AI模型初始化完成，共注册 {len(self.models)} 个模型")

        except Exception as e:
            self.logger.error(f"AI模型初始化失败: {e}")

    async def select_optimal_model(self, task_requirements: TaskRequirements) -> List[AIModelProfile]:
        """
        选择最优模型

        Args:
            task_requirements: 任务需求

        Returns:
            推荐的模型列表（按优先级排序）
        """
        try:
            model_scores = {}

            for model_id, model in self.models.items():
                score = await self._calculate_model_suitability(model, task_requirements)
                model_scores[model_id] = score

            # 按分数排序
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            # 应用学习优化
            optimized_models = await self._apply_learning_optimization(sorted_models, task_requirements)

            return [self.models[model_id] for model_id, score in optimized_models[:3]]

        except Exception as e:
            self.logger.error(f"选择最优模型失败: {e}")
            return list(self.models.values())[:3]  # 返回前3个作为备选

    async def _calculate_model_suitability(self, model: AIModelProfile,
                                          requirements: TaskRequirements) -> float:
        """计算模型适合度分数"""
        try:
            score = 0.0

            # 能力匹配度 (40%)
            capability_match = self._calculate_capability_match(model, requirements)
            score += capability_match * 0.4

            # 专业化领域匹配 (20%)
            domain_match = self._calculate_domain_match(model, requirements)
            score += domain_match * 0.2

            # 性能匹配 (15%)
            performance_match = self._calculate_performance_match(model, requirements)
            score += performance_match * 0.15

            # 成本效益 (15%)
            cost_efficiency = self._calculate_cost_efficiency(model, requirements)
            score += cost_efficiency * 0.15

            # 可靠性和历史表现 (10%)
            reliability_score = self._get_reliability_score(model, requirements.task_type)
            score += reliability_score * 0.1

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"计算模型适合度失败: {e}")
            return 0.5  # 默认中等分数

    def _calculate_capability_match(self, model: AIModelProfile,
                                  requirements: TaskRequirements) -> float:
        """计算能力匹配度"""
        required_capabilities = set(requirements.required_capabilities)
        model_capabilities = set(model.capabilities)

        if not required_capabilities:
            return 1.0

        # 计算交集
        intersection = required_capabilities.intersection(model_capabilities)
        match_ratio = len(intersection) / len(required_capabilities)

        # 关键能力加权
        critical_capabilities = [
            AIModelCapabilities.CONTEXT_UNDERSTANDING,
            AIModelCapabilities.REASONING,
            AIModelCapabilities.ACCURACY
        ]

        critical_match = any(cap in model_capabilities for cap in critical_capabilities)
        if critical_match:
            match_ratio *= 1.1  # 10%加权

        return min(match_ratio, 1.0)

    def _calculate_domain_match(self, model: AIModelProfile,
                               requirements: TaskRequirements) -> float:
        """计算领域匹配度"""
        if not requirements.specialized_domain:
            return 1.0

        if requirements.specialized_domain in model.specialized_domains:
            return 1.0

        # 模糊匹配
        domain_keywords = {
            "programming": ["code", "software", "development"],
            "data_analysis": ["data", "analytics", "statistics"],
            "research": ["academic", "research", "study"],
            "multimodal": ["image", "vision", "audio", "speech"],
            "real_time": ["realtime", "instant", "immediate"]
        }

        domain_keywords_lower = {k: [kw.lower() for kw in v] for k, v in domain_keywords.items()}
        specialized_domain_lower = requirements.specialized_domain.lower()

        # 检查模型优势中是否包含相关关键词
        model_strengths_text = " ".join(model.strengths).lower()
        for domain, keywords in domain_keywords_lower.items():
            if specialized_domain_lower in keywords or any(kw in model_strengths_text for kw in keywords):
                if domain in model.specialized_domains:
                    return 0.8  # 部分匹配
                else:
                    return 0.6  # 关键词匹配

        return 0.3  # 无明显匹配

    def _calculate_performance_match(self, model: AIModelProfile,
                                   requirements: TaskRequirements) -> float:
        """计算性能匹配度"""
        score = 0.0

        # 响应时间要求
        if requirements.max_response_time:
            if model.response_time <= requirements.max_response_time:
                score += 0.5
            else:
                score += max(0, 0.5 - (model.response_time - requirements.max_response_time) * 0.1)

        # 准确性要求
        if requirements.accuracy_requirement:
            if model.accuracy_score >= requirements.accuracy_requirement:
                score += 0.3
            else:
                score += model.accuracy_score * 0.3

        # 上下文大小要求
        if requirements.context_size:
            if model.context_window >= requirements.context_size:
                score += 0.2
            else:
                score += (model.context_window / requirements.context_size) * 0.2

        return min(score, 1.0)

    def _calculate_cost_efficiency(self, model: AIModelProfile,
                                  requirements: TaskRequirements) -> float:
        """计算成本效益"""
        if not requirements.cost_sensitivity:
            return 0.7  # 默认中等成本效益

        cost_scores = {"low": 1.0, "medium": 0.7, "high": 0.4}
        base_score = cost_scores.get(model.cost_tier, 0.5)

        # 根据成本敏感度调整
        if requirements.cost_sensitivity > 0.7:  # 高成本敏感
            # 优先选择低成本模型
            return base_score
        elif requirements.cost_sensitivity < 0.3:  # 低成本敏感
            # 可以选择高性能模型
            return min(base_score * 1.2, 1.0)
        else:
            return base_score

    def _get_reliability_score(self, model: AIModelProfile, task_type: str) -> float:
        """获取可靠性分数"""
        base_reliability = model.reliability_score

        # 获取历史表现数据
        historical_performance = self.performance_history.get(model.model_id, [])
        if historical_performance:
            # 计算最近的成功率
            recent_results = historical_performance[-20:]  # 最近20次执行
            success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)

            # 结合基础可靠性和历史表现
            combined_score = (base_reliability * 0.6 + success_rate * 0.4)
            return combined_score

        return base_reliability

    async def _apply_learning_optimization(self, sorted_models: List[Tuple[str, float]],
                                          requirements: TaskRequirements) -> List[Tuple[str, float]]:
        """应用学习优化"""
        try:
            optimized_models = []

            for model_id, base_score in sorted_models:
                # 获取A/B测试结果
                ab_results = self.ab_testing_results.get(model_id, {})

                if ab_results:
                    # 根据A/B测试结果调整分数
                    ab_score = ab_results.get(requirements.task_type, {}).get("success_rate", 0.5)
                    adjusted_score = base_score * (0.7 + ab_score * 0.6)  # 70%基础分 + 30%-60%测试分
                    optimized_models.append((model_id, min(adjusted_score, 1.0)))
                else:
                    optimized_models.append((model_id, base_score))

            return sorted(optimized_models, key=lambda x: x[1], reverse=True)

        except Exception as e:
            self.logger.error(f"学习优化失败: {e}")
            return sorted_models

    async def execute_with_single_model(self, model_id: str, prompt: str,
                                       task_requirements: Optional[TaskRequirements] = None) -> ModelExecutionResult:
        """使用单个模型执行任务"""
        try:
            model = self.models.get(model_id)
            if not model:
                raise ValueError(f"模型不存在: {model_id}")

            start_time = time.time()

            # 根据模型提供商调用相应的API
            if model.provider == "kimi":
                response = await self._execute_kimi_model(prompt, model)
            elif model.provider == "dashscope":
                response = await self._execute_dashscope_model(prompt, model)
            elif model.provider == "metaso":
                response = await self._execute_metaso_model(prompt, model)
            else:
                raise ValueError(f"不支持的模型提供商: {model.provider}")

            execution_time = time.time() - start_time
            cost = self._calculate_execution_cost(model, len(prompt), execution_time)

            result = ModelExecutionResult(
                model_id=model_id,
                success=True,
                response=response,
                confidence=self._extract_confidence(response),
                execution_time=execution_time,
                cost=cost,
                metadata={"model_name": model.model_name, "provider": model.provider}
            )

            # 记录历史表现
            self._record_performance(result)

            return result

        except Exception as e:
            self.logger.error(f"模型执行失败 {model_id}: {e}")
            return ModelExecutionResult(
                model_id=model_id,
                success=False,
                response=None,
                confidence=0.0,
                execution_time=0.0,
                cost=0.0,
                error=str(e)
            )

    async def execute_with_model_ensemble(self, model_ids: List[str], prompt: str,
                                          fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_BASED,
                                          task_requirements: Optional[TaskRequirements] = None) -> FusionResult:
        """使用模型集成执行任务"""
        try:
            if not model_ids:
                raise ValueError("模型ID列表不能为空")

            start_time = time.time()
            total_cost = 0.0
            execution_results = []

            # 并行执行多个模型
            tasks = []
            for model_id in model_ids:
                task = asyncio.create_task(
                    self.execute_with_single_model(model_id, prompt, task_requirements)
                )
                tasks.append(task)

            # 等待所有模型执行完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"模型 {model_ids[i]} 执行异常: {result}")
                elif result.success:
                    successful_results.append(result)
                    total_cost += result.cost
                else:
                    self.logger.warning(f"模型 {model_ids[i]} 执行失败: {result.error}")

            if not successful_results:
                raise ValueError("所有模型执行都失败了")

            # 融合结果
            fused_result = await self._fuse_results(successful_results, fusion_strategy)
            fused_result.execution_time = time.time() - start_time
            fused_result.total_cost = total_cost

            return fused_result

        except Exception as e:
            self.logger.error(f"模型集成执行失败: {e}")
            raise

    async def _fuse_results(self, results: List[ModelExecutionResult],
                          strategy: FusionStrategy) -> FusionResult:
        """融合多个模型的结果"""
        try:
            if not results:
                raise ValueError("结果列表不能为空")

            if strategy == FusionStrategy.MAJORITY_VOTING:
                return await self._majority_voting_fusion(results)
            elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
                return await self._weighted_average_fusion(results)
            elif strategy == FusionStrategy.CONFIDENCE_BASED:
                return await self._confidence_based_fusion(results)
            elif strategy == FusionStrategy.SEQUENTIAL_REFINEMENT:
                return await self._sequential_refinement_fusion(results)
            elif strategy == FusionStrategy.EXPERTISE_WEIGHTED:
                return await self._expertise_weighted_fusion(results)
            else:
                # 默认使用置信度融合
                return await self._confidence_based_fusion(results)

        except Exception as e:
            self.logger.error(f"结果融合失败: {e}")
            # 返回置信度最高的结果
            best_result = max(results, key=lambda x: x.confidence)
            return FusionResult(
                fused_response=best_result.response,
                confidence=best_result.confidence,
                contributing_models=[best_result.model_id],
                fusion_strategy="fallback_best",
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

    async def _confidence_based_fusion(self, results: List[ModelExecutionResult]) -> FusionResult:
        """基于置信度的融合"""
        try:
            # 按置信度加权
            total_confidence = sum(r.confidence for r in results)

            if total_confidence == 0:
                # 所有置信度都为0，使用平均
                return await self._weighted_average_fusion(results)

            fused_response = None
            contributing_models = []

            for result in results:
                weight = result.confidence / total_confidence
                contributing_models.append(result.model_id)

                # 这里需要根据响应类型进行融合
                # 简化处理：返回置信度最高的响应
                if fused_response is None or result.confidence > fused_response.get("confidence", 0):
                    fused_response = result.response

            # 添加融合元数据
            if isinstance(fused_response, dict):
                fused_response["fusion_metadata"] = {
                    "strategy": "confidence_based",
                    "contributing_models": contributing_models,
                    "confidence_weights": {r.model_id: r.confidence/total_confidence for r in results}
                }

            avg_confidence = total_confidence / len(results)

            return FusionResult(
                fused_response=fused_response,
                confidence=avg_confidence,
                contributing_models=contributing_models,
                fusion_strategy="confidence_based",
                metadata={"weights": {r.model_id: r.confidence/total_confidence for r in results}},
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

        except Exception as e:
            self.logger.error(f"置信度融合失败: {e}")
            raise

    async def _weighted_average_fusion(self, results: List[ModelExecutionResult]) -> FusionResult:
        """加权平均融合"""
        try:
            # 计算平均权重（基于可靠性和置信度）
            weights = []
            for result in results:
                model = self.models.get(result.model_id)
                reliability = model.reliability_score if model else 0.5
                weight = (result.confidence * 0.6 + reliability * 0.4)
                weights.append(weight)

            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0] * len(results)
                total_weight = len(results)

            # 这里需要根据响应类型进行实际的加权平均
            # 简化处理：返回第一个结果作为代表
            best_result = max(results, key=lambda x: x.confidence)
            contributing_models = [r.model_id for r in results]

            return FusionResult(
                fused_response=best_result.response,
                confidence=sum(r.confidence * w / total_weight for r, w in zip(results, weights)),
                contributing_models=contributing_models,
                fusion_strategy="weighted_average",
                metadata={"weights": {r.model_id: w/total_weight for r, w in zip(results, weights)}},
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

        except Exception as e:
            self.logger.error(f"加权平均融合失败: {e}")
            raise

    async def _majority_voting_fusion(self, results: List[ModelExecutionResult]) -> FusionResult:
        """多数投票融合"""
        try:
            # 收集所有响应
            responses = [r.response for r in results]
            contributing_models = [r.model_id for r in results]

            # 简化处理：使用第一个响应
            # 在实际实现中，这里应该进行真正的投票逻辑
            majority_response = responses[0]
            avg_confidence = statistics.mean([r.confidence for r in results])

            return FusionResult(
                fused_response=majority_response,
                confidence=avg_confidence,
                contributing_models=contributing_models,
                fusion_strategy="majority_voting",
                metadata={"vote_counts": {r.model_id: 1 for r in results}},
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

        except Exception as e:
            self.logger.error(f"多数投票融合失败: {e}")
            raise

    async def _sequential_refinement_fusion(self, results: List[ModelExecutionResult]) -> FusionResult:
        """顺序精炼融合"""
        try:
            # 按置信度排序
            sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)

            # 使用第一个结果作为基础
            refined_response = sorted_results[0].response

            # 顺序精炼（简化处理）
            contributing_models = [sorted_results[0].model_id]

            # 如果有多个结果，尝试精炼
            if len(sorted_results) > 1:
                # 这里可以实现真正的精炼逻辑
                refined_response["refinement_metadata"] = {
                    "base_model": sorted_results[0].model_id,
                    "refinement_count": len(sorted_results) - 1
                }
                contributing_models.extend([r.model_id for r in sorted_results[1:]])

            avg_confidence = statistics.mean([r.confidence for r in sorted_results])

            return FusionResult(
                fused_response=refined_response,
                confidence=avg_confidence,
                contributing_models=contributing_models,
                fusion_strategy="sequential_refinement",
                metadata={"refinement_order": [r.model_id for r in sorted_results]},
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

        except Exception as e:
            self.logger.error(f"顺序精炼融合失败: {e}")
            raise

    async def _expertise_weighted_fusion(self, results: List[ModelExecutionResult]) -> FusionResult:
        """专业权重融合"""
        try:
            # 根据任务类型和专业领域计算权重
            weights = []

            for result in results:
                model = self.models.get(result.model_id)
                if model:
                    # 基于模型专业性和任务相关性计算权重
                    expertise_weight = 0.5 + (model.accuracy_score * 0.3)
                    confidence_weight = result.confidence * 0.2
                    total_weight = expertise_weight + confidence_weight
                    weights.append(total_weight)
                else:
                    weights.append(0.5)  # 默认权重

            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0] * len(results)
                total_weight = len(results)

            # 选择权重最高的结果
            weighted_scores = [(r, w/total_weight) for r, w in zip(results, weights)]
            best_result = max(weighted_scores, key=lambda x: x[1])
            contributing_models = [r.model_id for r in results]

            return FusionResult(
                fused_response=best_result[0].response,
                confidence=best_result[0].confidence,
                contributing_models=contributing_models,
                fusion_strategy="expertise_weighted",
                metadata={"expertise_weights": {r.model_id: w for r, (r, w) in zip(results, weighted_scores)}},
                execution_time=sum(r.execution_time for r in results),
                total_cost=sum(r.cost for r in results)
            )

        except Exception as e:
            self.logger.error(f"专业权重融合失败: {e}")
            raise

    def _calculate_execution_cost(self, model: AIModelProfile, input_length: int,
                                   execution_time: float) -> float:
        """计算执行成本"""
        try:
            cost_tiers = {"low": 0.001, "medium": 0.005, "high": 0.01}
            base_cost = cost_tiers.get(model.cost_tier, 0.005)

            # 基于输入长度和执行时间调整成本
            length_factor = min(input_length / 1000, 5.0)  # 每1000字符的成本因子
            time_factor = execution_time / 10.0  # 每10秒的时间因子

            total_cost = base_cost * (1 + length_factor + time_factor)
            return total_cost

        except Exception:
            return 0.005  # 默认成本

    def _extract_confidence(self, response: Any) -> float:
        """从响应中提取置信度"""
        try:
            if isinstance(response, dict):
                return response.get("confidence", 0.8)
            else:
                return 0.8  # 默认置信度
        except Exception:
            return 0.8

    def _record_performance(self, result: ModelExecutionResult):
        """记录模型性能"""
        try:
            if result.model_id not in self.performance_history:
                self.performance_history[result.model_id] = []

            self.performance_history[result.model_id].append(result)

            # 保持历史记录在合理大小
            if len(self.performance_history[result.model_id]) > 1000:
                self.performance_history[result.model_id] = self.performance_history[result.model_id][-500:]

            # 保存到数据库
            self._save_performance_to_db(result)

        except Exception as e:
            self.logger.error(f"记录性能失败: {e}")

    def _save_performance_to_db(self, result: ModelExecutionResult):
        """保存性能数据到数据库"""
        try:
            # 这里应该调用数据库API保存
            pass
        except Exception as e:
            self.logger.error(f"保存性能数据到数据库失败: {e}")

    async def _execute_kimi_model(self, prompt: str, model: AIModelProfile) -> Any:
        """执行Kimi模型"""
        try:
            # 这里应该调用Kimi API
            # 模拟实现
            await asyncio.sleep(1.5)  # 模拟API调用延迟

            return {
                "content": f"Kimi响应: {prompt[:100]}...",
                "confidence": 0.9,
                "model": model.model_name
            }
        except Exception as e:
            self.logger.error(f"Kimi模型执行失败: {e}")
            raise

    async def _execute_dashscope_model(self, prompt: str, model: AIModelProfile) -> Any:
        """执行DashScope模型"""
        try:
            # 这里应该调用DashScope API
            # 模拟实现
            await asyncio.sleep(0.8)

            return {
                "content": f"DashScope响应: {prompt[:100]}...",
                "confidence": 0.85,
                "model": model.model_name
            }
        except Exception as e:
            self.logger.error(f"DashScope模型执行失败: {e}")
            raise

    async def _execute_metaso_model(self, prompt: str, model: AIModelProfile) -> Any:
        """执行Metaso模型"""
        try:
            # 这里应该调用Metaso API
            # 模拟实现
            await asyncio.sleep(1.2)

            return {
                "content": f"Metaso响应: {prompt[:100]}...",
                "confidence": 0.88,
                "model": model.model_name
            }
        except Exception as e:
            self.logger.error(f"Metaso模型执行失败: {e}")
            raise

    def _start_background_monitoring(self):
        """启动后台监控"""
        def monitor():
            while True:
                try:
                    # 清理旧数据
                    self._cleanup_old_data()

                    # 更新模型状态
                    self._update_model_status()

                    # 生成性能报告
                    self._generate_performance_report()

                    # 等待下一次检查
                    time.sleep(3600)  # 每小时检查一次

                except Exception as e:
                    self.logger.error(f"后台监控异常: {e}")
                    time.sleep(300)  # 出错后等待5分钟再试

        import threading
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            # 清理超过30天的性能记录
            cutoff_date = datetime.now() - timedelta(days=30)

            for model_id, results in self.performance_history.items():
                filtered_results = [
                    r for r in results
                    if r.timestamp > cutoff_date
                ]
                self.performance_history[model_id] = filtered_results

        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}")

    def _update_model_status(self):
        """更新模型状态"""
        try:
            for model_id, model in self.models.items():
                # 检查模型可用性
                # 这里可以实现健康检查逻辑
                pass

        except Exception as e:
            self.logger.error(f"更新模型状态失败: {e}")

    def _generate_performance_report(self):
        """生成性能报告"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "models": {}
            }

            for model_id, model in self.models.items():
                history = self.performance_history.get(model_id, [])
                if history:
                    recent_results = history[-100:]  # 最近100次

                    success_rate = sum(1 for r in recent_results if r.success) / len(recent_results)
                    avg_confidence = statistics.mean([r.confidence for r in recent_results])
                    avg_execution_time = statistics.mean([r.execution_time for r in recent_results])

                    report["models"][model_id] = {
                        "success_rate": success_rate,
                        "avg_confidence": avg_confidence,
                        "avg_execution_time": avg_execution_time,
                        "total_executions": len(recent_results),
                        "last_updated": datetime.now().isoformat()
                    }

            # 这里可以保存报告到数据库或文件
            self.logger.info(f"性能报告生成完成，包含 {len(report['models'])} 个模型")

        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")

    def get_model_performance_stats(self, model_id: str, days: int = 7) -> Dict[str, Any]:
        """获取模型性能统计"""
        try:
            history = self.performance_history.get(model_id, [])
            if not history:
                return {}

            cutoff_date = datetime.now() - timedelta(days=days)
            recent_results = [
                r for r in history
                if r.timestamp > cutoff_date
            ]

            if not recent_results:
                return {}

            return {
                "model_id": model_id,
                "period_days": days,
                "total_executions": len(recent_results),
                "success_rate": sum(1 for r in recent_results if r.success) / len(recent_results),
                "avg_confidence": statistics.mean([r.confidence for r in recent_results]),
                "avg_execution_time": statistics.mean([r.execution_time for r in recent_results]),
                "total_cost": sum(r.cost for r in recent_results),
                "last_execution": recent_results[-1].timestamp.isoformat()
            }

        except Exception as e:
            self.logger.error(f"获取性能统计失败: {e}")
            return {}

    def get_all_models_info(self) -> List[Dict[str, Any]]:
        """获取所有模型信息"""
        return [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "provider": model.provider,
                "capabilities": [cap.value for cap in model.capabilities],
                "strengths": model.strengths,
                "weaknesses": model.weaknesses,
                "cost_tier": model.cost_tier,
                "performance_tier": model.performance_tier,
                "max_tokens": model.max_tokens,
                "response_time": model.response_time,
                "accuracy_score": model.accuracy_score,
                "reliability_score": model.reliability_score,
                "context_window": model.context_window,
                "specialized_domains": model.specialized_domains,
                "last_updated": model.last_updated.isoformat()
            }
            for model in self.models.values()
        ]


# 全局AI编排器实例
_ai_orchestrator: Optional[AIOrchestrator] = None


def get_ai_orchestrator() -> AIOrchestrator:
    """获取全局AI编排器实例"""
    global _ai_orchestrator
    if _ai_orchestrator is None:
        _ai_orchestrator = AIOrchestrator()
    return _ai_orchestrator


# 导出
__all__ = [
    "AIOrchestrator",
    "AIModelProfile",
    "TaskRequirements",
    "ModelExecutionResult",
    "FusionResult",
    "AIModelCapabilities",
    "TaskComplexity",
    "FusionStrategy",
    "get_ai_orchestrator"
]