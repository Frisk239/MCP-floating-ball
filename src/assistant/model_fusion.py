"""
MCP Floating Ball - 高级模型输出融合系统

提供多种先进的融合算法，实现多AI模型的智能输出组合。
"""

import asyncio
import json
import math
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import statistics
import hashlib

from src.core.logging import get_logger
from src.assistant.ai_orchestrator import FusionResult, AIModelResponse

logger = get_logger("model_fusion")


class FusionMethod(Enum):
    """融合方法"""
    WEIGHTED_AVERAGE = "weighted_average"
    Borda_COUNT = "borda_count"
    CONDORCET = "condorcet"
    EXPERT_WEIGHTING = "expert_weighting"
    DEMPSTER_SHAFER = "dempster_shafer"
    BAYESIAN_FUSION = "bayesian_fusion"
    NEURAL_ENSEMBLE = "neural_ensemble"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    CONFLICT_RESOLUTION = "conflict_resolution"
    HIERARCHICAL_FUSION = "hierarchical_fusion"


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_PRIORITY = "expert_priority"
    CONSENSUS_BUILDING = "consensus_building"
    ARBITRATION = "arbitration"


@dataclass
class ModelWeight:
    """模型权重配置"""
    model_id: str
    base_weight: float
    expert_weight: float  # 基于专业能力的权重
    performance_weight: float  # 基于历史表现的权重
    task_specific_weight: float  # 任务特定权重
    dynamic_weight: float = 0.0  # 动态调整权重
    confidence_factor: float = 1.0  # 置信度因子

    def calculate_final_weight(self) -> float:
        """计算最终权重"""
        return (self.base_weight * 0.2 +
                self.expert_weight * 0.3 +
                self.performance_weight * 0.3 +
                self.task_specific_weight * 0.2) * self.confidence_factor * self.dynamic_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_weight": self.base_weight,
            "expert_weight": self.expert_weight,
            "performance_weight": self.performance_weight,
            "task_specific_weight": self.task_specific_weight,
            "dynamic_weight": self.dynamic_weight,
            "confidence_factor": self.confidence_factor,
            "final_weight": self.calculate_final_weight()
        }


@dataclass
class Evidence:
    """证据（用于Dempster-Shafer理论）"""
    hypothesis: str
    belief_mass: float
    supporting_models: List[str]

    def __post_init__(self):
        # 确保信念质量在合理范围内
        self.belief_mass = max(0.0, min(1.0, self.belief_mass))


@dataclass
class ConflictInfo:
    """冲突信息"""
    conflict_level: float  # 0-1，冲突程度
    conflicting_models: List[Tuple[str, str]]  # 冲突的模型对
    conflict_reasons: Dict[str, str]  # 冲突原因
    resolution_suggestion: str


class ModelFusionEngine:
    """高级模型融合引擎"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.model_weights: Dict[str, ModelWeight] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.expertise_matrix: Dict[str, Dict[str, float]] = {}  # 模型对不同任务类型的专业度
        self.fusion_cache: Dict[str, FusionResult] = {}
        self.conflict_threshold = 0.3

        # 初始化专业知识矩阵
        self._initialize_expertise_matrix()

        self.logger.info("模型融合引擎初始化完成")

    def _initialize_expertise_matrix(self):
        """初始化专业知识矩阵"""
        # 预定义的模型专业度评估
        default_expertise = {
            "kimi": {
                "code_generation": 0.9,
                "text_analysis": 0.8,
                "reasoning": 0.85,
                "creative_writing": 0.7,
                "data_analysis": 0.8,
                "problem_solving": 0.85
            },
            "dashscope": {
                "code_generation": 0.8,
                "text_analysis": 0.9,
                "reasoning": 0.8,
                "creative_writing": 0.85,
                "data_analysis": 0.75,
                "problem_solving": 0.8
            },
            "metaso": {
                "code_generation": 0.7,
                "text_analysis": 0.85,
                "reasoning": 0.9,
                "creative_writing": 0.9,
                "data_analysis": 0.7,
                "problem_solving": 0.85
            }
        }
        self.expertise_matrix.update(default_expertise)

    async def advanced_fusion(self,
                             responses: List[AIModelResponse],
                             fusion_method: FusionMethod,
                             task_type: str,
                             context: Optional[Dict[str, Any]] = None) -> FusionResult:
        """高级融合方法"""
        try:
            self.logger.info(f"开始高级融合，方法: {fusion_method.value}")

            # 更新模型权重
            await self._update_model_weights(responses, task_type)

            # 检测冲突
            conflict_info = await self._detect_conflicts(responses)

            if conflict_info.conflict_level > self.conflict_threshold:
                self.logger.warning(f"检测到高冲突级别: {conflict_info.conflict_level}")
                # 应用冲突解决策略
                responses = await self._resolve_conflicts(responses, conflict_info)

            # 根据融合方法执行融合
            if fusion_method == FusionMethod.WEIGHTED_AVERAGE:
                result = await self._weighted_average_fusion(responses)
            elif fusion_method == FusionMethod.BORDA_COUNT:
                result = await self._borda_count_fusion(responses)
            elif fusion_method == FusionMethod.CONDORCET:
                result = await self._condorcet_fusion(responses)
            elif fusion_method == FusionMethod.EXPERT_WEIGHTING:
                result = await self._expert_weighting_fusion(responses, task_type)
            elif fusion_method == FusionMethod.DEMPSTER_SHAFER:
                result = await self._dempster_shafer_fusion(responses)
            elif fusion_method == FusionMethod.BAYESIAN_FUSION:
                result = await self._bayesian_fusion(responses, context)
            elif fusion_method == FusionMethod.ADAPTIVE_THRESHOLD:
                result = await self._adaptive_threshold_fusion(responses)
            elif fusion_method == FusionMethod.CONFLICT_RESOLUTION:
                result = await self._conflict_resolution_fusion(responses, conflict_info)
            elif fusion_method == FusionMethod.HIERARCHICAL_FUSION:
                result = await self._hierarchical_fusion(responses, task_type)
            else:
                # 默认使用加权平均
                result = await self._weighted_average_fusion(responses)

            # 记录性能历史
            await self._record_performance(responses, result)

            return result

        except Exception as e:
            self.logger.error(f"高级融合失败: {e}")
            # 返回简单的加权平均作为后备
            return await self._weighted_average_fusion(responses)

    async def _update_model_weights(self, responses: List[AIModelResponse], task_type: str):
        """更新模型权重"""
        for response in responses:
            model_id = response.model_id

            # 初始化权重
            if model_id not in self.model_weights:
                self.model_weights[model_id] = ModelWeight(
                    model_id=model_id,
                    base_weight=1.0 / len(responses),  # 平均初始权重
                    expert_weight=0.0,
                    performance_weight=0.5,
                    task_specific_weight=0.0
                )

            weight = self.model_weights[model_id]

            # 更新专业权重
            weight.expert_weight = self.expertise_matrix.get(model_id, {}).get(task_type, 0.5)

            # 更新任务特定权重
            weight.task_specific_weight = self._calculate_task_specific_weight(response, task_type)

            # 更新置信度因子
            weight.confidence_factor = response.confidence or 0.5

            # 动态权重调整
            weight.dynamic_weight = self._calculate_dynamic_weight(response, responses)

    def _calculate_task_specific_weight(self, response: AIModelResponse, task_type: str) -> float:
        """计算任务特定权重"""
        # 基于响应长度、结构化程度等计算权重
        text = response.model_output or ""

        if not text:
            return 0.1

        weight = 0.5  # 基础权重

        # 代码生成任务
        if task_type == "code_generation":
            if "```" in text or any(keyword in text.lower() for keyword in ["def ", "function", "class ", "import"]):
                weight += 0.3

        # 文本分析任务
        elif task_type == "text_analysis":
            if any(keyword in text.lower() for keyword in ["分析", "总结", "要点", "关键"]):
                weight += 0.2

        # 创意写作任务
        elif task_type == "creative_writing":
            if len(text) > 100:  # 较长的响应
                weight += 0.2

        # 数据分析任务
        elif task_type == "data_analysis":
            if any(char.isdigit() for char in text) and any(keyword in text.lower() for keyword in ["数据", "统计", "分析"]):
                weight += 0.3

        return min(1.0, weight)

    def _calculate_dynamic_weight(self, response: AIModelResponse, all_responses: List[AIModelResponse]) -> float:
        """计算动态权重"""
        if not all_responses:
            return 1.0

        # 基于置信度的相对权重
        confidences = [r.confidence or 0.5 for r in all_responses]
        avg_confidence = statistics.mean(confidences)

        if avg_confidence == 0:
            return 1.0

        relative_confidence = (response.confidence or 0.5) / avg_confidence
        return min(2.0, max(0.5, relative_confidence))

    async def _detect_conflicts(self, responses: List[AIModelResponse]) -> ConflictInfo:
        """检测模型间的冲突"""
        if len(responses) < 2:
            return ConflictInfo(0.0, [], {}, "")

        conflicting_models = []
        conflict_reasons = {}
        total_conflicts = 0
        possible_conflicts = len(responses) * (len(responses) - 1) // 2

        for i, resp1 in enumerate(responses):
            for j, resp2 in enumerate(responses[i+1:], i+1):
                conflict_level, reason = await self._calculate_conflict_level(resp1, resp2)

                if conflict_level > self.conflict_threshold:
                    conflicting_models.append((resp1.model_id, resp2.model_id))
                    conflict_reasons[f"{resp1.model_id}_vs_{resp2.model_id}"] = reason
                    total_conflicts += 1

        overall_conflict = total_conflicts / possible_conflicts if possible_conflicts > 0 else 0.0

        resolution_suggestion = self._suggest_conflict_resolution(conflicting_models, conflict_reasons)

        return ConflictInfo(
            conflict_level=overall_conflict,
            conflicting_models=conflicting_models,
            conflict_reasons=conflict_reasons,
            resolution_suggestion=resolution_suggestion
        )

    async def _calculate_conflict_level(self, resp1: AIModelResponse, resp2: AIModelResponse) -> Tuple[float, str]:
        """计算两个响应间的冲突程度"""
        text1 = (resp1.model_output or "").lower().strip()
        text2 = (resp2.model_output or "").lower().strip()

        if not text1 or not text2:
            return 0.0, "空响应"

        # 简单的相似度计算
        similarity = self._calculate_text_similarity(text1, text2)
        conflict_level = 1.0 - similarity

        # 分析冲突原因
        if conflict_level > 0.7:
            reason = "完全不同的回答"
        elif conflict_level > 0.5:
            reason = "部分不一致"
        elif conflict_level > 0.3:
            reason = "轻微差异"
        else:
            reason = "基本一致"

        return conflict_level, reason

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简化的文本相似度计算
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _suggest_conflict_resolution(self, conflicting_models: List[Tuple[str, str]], reasons: Dict[str, str]) -> str:
        """建议冲突解决方法"""
        if not conflicting_models:
            return "无冲突"

        if len(conflicting_models) > len(conflicting_models) * 0.5:
            return "高冲突情况，建议使用专家仲裁或人工干预"

        # 分析主要原因
        reason_counts = {}
        for reason in reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        most_common_reason = max(reason_counts, key=reason_counts.get)

        if "完全不同" in most_common_reason:
            return "建议重新评估问题或采用置信度加权"
        elif "部分不一致" in most_common_reason:
            return "建议使用多数投票或专家优先"
        else:
            return "建议使用加权平均融合"

    async def _resolve_conflicts(self, responses: List[AIModelResponse], conflict_info: ConflictInfo) -> List[AIModelResponse]:
        """解决冲突"""
        if conflict_info.conflict_level <= self.conflict_threshold:
            return responses

        # 根据冲突解决策略处理响应
        strategy = ConflictResolutionStrategy.CONFIDENCE_WEIGHTED  # 默认策略

        if "专家仲裁" in conflict_info.resolution_suggestion:
            strategy = ConflictResolutionStrategy.EXPERT_PRIORITY
        elif "多数投票" in conflict_info.resolution_suggestion:
            strategy = ConflictResolutionStrategy.MAJORITY_VOTE

        # 应用策略
        if strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
            # 按置信度排序，保留高置信度的响应
            responses.sort(key=lambda x: x.confidence or 0.5, reverse=True)
            # 保留前70%的响应
            keep_count = max(1, int(len(responses) * 0.7))
            return responses[:keep_count]

        elif strategy == ConflictResolutionStrategy.EXPERT_PRIORITY:
            # 优先保留专家模型
            expert_models = ["kimi", "metaso"]  # 预定义专家模型
            expert_responses = [r for r in responses if r.model_id in expert_models]
            non_expert_responses = [r for r in responses if r.model_id not in expert_models]

            # 保留所有专家响应和最好的非专家响应
            if non_expert_responses:
                best_non_expert = max(non_expert_responses, key=lambda x: x.confidence or 0.5)
                expert_responses.append(best_non_expert)

            return expert_responses

        elif strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            # 简化的多数投票，保留相似的响应
            similar_responses = [responses[0]]  # 保留第一个作为基准

            for resp in responses[1:]:
                # 检查是否与已有响应相似
                is_similar = any(
                    self._calculate_text_similarity(resp.model_output or "", existing.model_output or "") > 0.5
                    for existing in similar_responses
                )
                if is_similar:
                    similar_responses.append(resp)

            return similar_responses

        return responses

    async def _weighted_average_fusion(self, responses: List[AIModelResponse]) -> FusionResult:
        """加权平均融合"""
        if not responses:
            return FusionResult(
                fused_response="",
                confidence=0.0,
                contributing_models=[],
                fusion_strategy=FusionMethod.WEIGHTED_AVERAGE.value
            )

        # 计算权重
        weights = []
        for response in responses:
            weight = self.model_weights.get(response.model_id)
            if weight:
                weights.append(weight.calculate_final_weight())
            else:
                weights.append(1.0 / len(responses))

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # 融合文本
        if len(responses) == 1:
            fused_text = responses[0].model_output or ""
        else:
            # 基于权重选择主要响应，补充其他响应的信息
            sorted_responses = sorted(zip(responses, weights), key=lambda x: x[1], reverse=True)
            primary_response, primary_weight = sorted_responses[0]

            fused_text = primary_response.model_output or ""

            # 如果其他响应有重要信息，添加补充
            for response, weight in sorted_responses[1:]:
                if weight > 0.2:  # 权重阈值
                    additional_info = self._extract_complementary_info(
                        primary_response.model_output or "",
                        response.model_output or ""
                    )
                    if additional_info:
                        fused_text += f"\n\n补充信息: {additional_info}"

        # 计算融合置信度
        fused_confidence = sum(
            (response.confidence or 0.5) * weight
            for response, weight in zip(responses, weights)
        )

        return FusionResult(
            fused_response=fused_text,
            confidence=fused_confidence,
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.WEIGHTED_AVERAGE.value,
            metadata={
                "weights": dict(zip([r.model_id for r in responses], weights)),
                "fusion_details": "加权平均融合"
            }
        )

    def _extract_complementary_info(self, primary_text: str, additional_text: str) -> str:
        """提取补充信息"""
        # 简化的信息提取
        primary_words = set(primary_text.lower().split())
        additional_words = set(additional_text.lower().split())

        complementary_words = additional_words - primary_words

        if len(complementary_words) > 3:  # 有足够的新信息
            # 返回包含新信息的句子片段
            sentences = additional_text.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if len(complementary_words.intersection(sentence_words)) > 2:
                    return sentence.strip()

        return ""

    async def _borda_count_fusion(self, responses: List[AIModelResponse]) -> FusionResult:
        """Borda计数融合"""
        if len(responses) < 2:
            return await self._weighted_average_fusion(responses)

        # 按置信度排序
        sorted_responses = sorted(responses, key=lambda x: x.confidence or 0.5, reverse=True)

        # 分配Borda分数
        borda_scores = {}
        for i, response in enumerate(sorted_responses):
            score = len(responses) - i - 1
            borda_scores[response.model_id] = score

        # 选择得分最高的响应作为主要输出
        winner = max(borda_scores, key=borda_scores.get)
        winner_response = next(r for r in responses if r.model_id == winner)

        return FusionResult(
            fused_response=winner_response.model_output or "",
            confidence=winner_response.confidence or 0.5,
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.BORDA_COUNT.value,
            metadata={
                "borda_scores": borda_scores,
                "fusion_details": "Borda计数融合"
            }
        )

    async def _condorcet_fusion(self, responses: List[AIModelResponse]) -> FusionResult:
        """Condorcet融合"""
        if len(responses) < 3:
            return await self._weighted_average_fusion(responses)

        # 两两比较
        wins = {r.model_id: 0 for r in responses}

        for i, resp1 in enumerate(responses):
            for j, resp2 in enumerate(responses):
                if i != j:
                    # 基于置信度和内容质量比较
                    if (resp1.confidence or 0.5) > (resp2.confidence or 0.5):
                        wins[resp1.model_id] += 1

        # 找到Condorcet胜者
        condorcet_winner = max(wins, key=wins.get)
        winner_response = next(r for r in responses if r.model_id == condorcet_winner)

        return FusionResult(
            fused_response=winner_response.model_output or "",
            confidence=winner_response.confidence or 0.5,
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.CONDORCET.value,
            metadata={
                "pairwise_wins": wins,
                "fusion_details": "Condorcet融合"
            }
        )

    async def _expert_weighting_fusion(self, responses: List[AIModelResponse], task_type: str) -> FusionResult:
        """专家权重融合"""
        # 基于专业知识矩阵计算权重
        expert_weights = []
        for response in responses:
            expertise = self.expertise_matrix.get(response.model_id, {}).get(task_type, 0.5)
            expert_weights.append(expertise * (response.confidence or 0.5))

        # 归一化权重
        total_weight = sum(expert_weights)
        if total_weight > 0:
            expert_weights = [w / total_weight for w in expert_weights]

        # 选择权重最高的响应
        best_idx = max(range(len(expert_weights)), key=lambda i: expert_weights[i])
        best_response = responses[best_idx]

        return FusionResult(
            fused_response=best_response.model_output or "",
            confidence=best_response.confidence or 0.5,
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.EXPERT_WEIGHTING.value,
            metadata={
                "expert_weights": dict(zip([r.model_id for r in responses], expert_weights)),
                "task_type": task_type,
                "fusion_details": "专家权重融合"
            }
        )

    async def _dempster_shafer_fusion(self, responses: List[AIModelResponse]) -> FusionResult:
        """Dempster-Shafer证据理论融合"""
        # 简化的Dempster-Shafer实现
        evidences = []

        for response in responses:
            # 将响应转换为证据
            hypothesis = response.model_output or ""[:100]  # 取前100字符作为假设
            belief_mass = (response.confidence or 0.5)

            evidence = Evidence(
                hypothesis=hypothesis,
                belief_mass=belief_mass,
                supporting_models=[response.model_id]
            )
            evidences.append(evidence)

        # 简化的证据组合
        if not evidences:
            return await self._weighted_average_fusion(responses)

        # 选择信念质量最高的证据
        best_evidence = max(evidences, key=lambda e: e.belief_mass)

        return FusionResult(
            fused_response=best_evidence.hypothesis,
            confidence=best_evidence.belief_mass,
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.DEMPSTER_SHAFER.value,
            metadata={
                "evidence_count": len(evidences),
                "fusion_details": "Dempster-Shafer证据理论融合"
            }
        )

    async def _bayesian_fusion(self, responses: List[AIModelResponse], context: Optional[Dict[str, Any]] = None) -> FusionResult:
        """贝叶斯融合"""
        # 简化的贝叶斯融合
        priors = {r.model_id: 1.0 / len(responses) for r in responses}

        # 计算似然
        likelihoods = {}
        for response in responses:
            # 基于置信度和响应质量计算似然
            likelihood = (response.confidence or 0.5) * self._calculate_response_quality(response)
            likelihoods[response.model_id] = likelihood

        # 计算后验概率
        posteriors = {}
        total_probability = 0
        for model_id in priors:
            posterior = priors[model_id] * likelihoods[model_id]
            posteriors[model_id] = posterior
            total_probability += posterior

        # 归一化
        if total_probability > 0:
            posteriors = {k: v / total_probability for k, v in posteriors.items()}

        # 选择后验概率最高的响应
        best_model = max(posteriors, key=posteriors.get)
        best_response = next(r for r in responses if r.model_id == best_model)

        return FusionResult(
            fused_response=best_response.model_output or "",
            confidence=posteriors[best_model],
            contributing_models=[r.model_id for r in responses],
            fusion_strategy=FusionMethod.BAYESIAN_FUSION.value,
            metadata={
                "posteriors": posteriors,
                "fusion_details": "贝叶斯融合"
            }
        )

    def _calculate_response_quality(self, response: AIModelResponse) -> float:
        """计算响应质量"""
        text = response.model_output or ""
        if not text:
            return 0.1

        quality = 0.5  # 基础质量

        # 长度质量
        if 50 <= len(text) <= 1000:
            quality += 0.2

        # 结构质量
        if any(punct in text for punct in [".", "!", "?", "，", "。"]):
            quality += 0.1

        # 内容质量（关键词）
        positive_keywords = ["正确", "准确", "详细", "完整", "清楚", "明确"]
        if any(keyword in text for keyword in positive_keywords):
            quality += 0.2

        return min(1.0, quality)

    async def _adaptive_threshold_fusion(self, responses: List[AIModelResponse]) -> FusionResult:
        """自适应阈值融合"""
        # 计算平均置信度
        avg_confidence = statistics.mean([r.confidence or 0.5 for r in responses])

        # 设置动态阈值
        threshold = max(0.6, avg_confidence + 0.1)

        # 筛选高置信度响应
        high_confidence_responses = [r for r in responses if (r.confidence or 0.5) >= threshold]

        if not high_confidence_responses:
            # 如果没有高置信度响应，使用最高置信度的
            best_response = max(responses, key=lambda x: x.confidence or 0.5)
            high_confidence_responses = [best_response]

        # 使用加权平均融合筛选后的响应
        return await self._weighted_average_fusion(high_confidence_responses)

    async def _conflict_resolution_fusion(self, responses: List[AIModelResponse], conflict_info: ConflictInfo) -> FusionResult:
        """冲突解决融合"""
        if conflict_info.conflict_level <= self.conflict_threshold:
            return await self._weighted_average_fusion(responses)

        # 使用建议的解决方法
        if "置信度加权" in conflict_info.resolution_suggestion:
            return await self._weighted_average_fusion(responses)
        elif "专家优先" in conflict_info.resolution_suggestion:
            # 优先专家模型
            expert_models = ["kimi", "metaso"]
            expert_responses = [r for r in responses if r.model_id in expert_models]
            if expert_responses:
                return await self._weighted_average_fusion(expert_responses)
        elif "多数投票" in conflict_info.resolution_suggestion:
            return await self._borda_count_fusion(responses)

        # 默认使用加权平均
        return await self._weighted_average_fusion(responses)

    async def _hierarchical_fusion(self, responses: List[AIModelResponse], task_type: str) -> FusionResult:
        """分层融合"""
        if len(responses) < 2:
            return await self._weighted_average_fusion(responses)

        # 第一层：按专业度分组
        expert_responses = []
        general_responses = []

        for response in responses:
            expertise = self.expertise_matrix.get(response.model_id, {}).get(task_type, 0.5)
            if expertise >= 0.7:
                expert_responses.append(response)
            else:
                general_responses.append(response)

        # 第二层：组内融合
        if expert_responses:
            expert_fusion = await self._weighted_average_fusion(expert_responses)
        else:
            expert_fusion = None

        if general_responses:
            general_fusion = await self._weighted_average_fusion(general_responses)
        else:
            general_fusion = None

        # 第三层：跨组融合
        final_fusions = [fusion for fusion in [expert_fusion, general_fusion] if fusion]

        if len(final_fusions) == 1:
            return final_fusions[0]
        elif len(final_fusions) == 2:
            # 创建虚拟响应进行最终融合
            virtual_responses = []
            for fusion in final_fusions:
                virtual_response = AIModelResponse(
                    model_id=f"hierarchical_{fusion.fusion_method}",
                    model_output=fusion.fused_output,
                    confidence=fusion.confidence
                )
                virtual_responses.append(virtual_response)

            return await self._weighted_average_fusion(virtual_responses)
        else:
            return await self._weighted_average_fusion(responses)

    async def _record_performance(self, responses: List[AIModelResponse], fusion_result: FusionResult):
        """记录性能历史"""
        for response in responses:
            model_id = response.model_id
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []

            # 记录置信度作为性能指标
            confidence = response.confidence or 0.5
            self.performance_history[model_id].append(confidence)

            # 保持历史记录在合理范围内
            if len(self.performance_history[model_id]) > 100:
                self.performance_history[model_id] = self.performance_history[model_id][-50:]

    def get_model_weights_info(self) -> Dict[str, Any]:
        """获取模型权重信息"""
        return {
            model_id: weight.to_dict()
            for model_id, weight in self.model_weights.items()
        }

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要"""
        summary = {}
        for model_id, history in self.performance_history.items():
            if history:
                summary[model_id] = {
                    "avg_performance": statistics.mean(history),
                    "std_performance": statistics.stdev(history) if len(history) > 1 else 0,
                    "min_performance": min(history),
                    "max_performance": max(history),
                    "sample_count": len(history)
                }
        return summary


# 导出
__all__ = [
    "ModelFusionEngine",
    "FusionMethod",
    "ConflictResolutionStrategy",
    "ModelWeight",
    "Evidence",
    "ConflictInfo"
]