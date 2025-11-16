"""
MCP Floating Ball - 专业化任务分配系统

智能分析任务特征，为最适合的AI模型分配任务。
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.core.logging import get_logger
from src.assistant.ai_orchestrator import (
    AIModelProfile,
    TaskRequirements,
    AIModelCapabilities,
    TaskComplexity
)

logger = get_logger("task_dispatcher")


class TaskCategory(Enum):
    """任务类别"""
    CODE_GENERATION = "code_generation"
    TEXT_ANALYSIS = "text_analysis"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    PROBLEM_SOLVING = "problem_solving"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    DOCUMENT_PROCESSING = "document_processing"
    REASONING = "reasoning"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TaskUrgency(Enum):
    """任务紧急程度"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AllocationStrategy(Enum):
    """分配策略"""
    EXPERTISE_MATCH = "expertise_match"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCING = "load_balancing"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_FIRST = "quality_first"
    SPEED_FIRST = "speed_first"
    ADAPTIVE = "adaptive"


@dataclass
class TaskProfile:
    """任务档案"""
    task_id: str
    category: TaskCategory
    complexity: TaskComplexity
    urgency: TaskUrgency
    description: str
    requirements: TaskRequirements
    estimated_duration: float  # 预估执行时间（秒）
    estimated_cost: float  # 预估成本
    keywords: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category.value,
            "complexity": self.complexity.value,
            "urgency": self.urgency.value,
            "description": self.description,
            "estimated_duration": self.estimated_duration,
            "estimated_cost": self.estimated_cost,
            "keywords": self.keywords,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AllocationResult:
    """分配结果"""
    task_id: str
    assigned_model: str
    allocation_strategy: AllocationStrategy
    confidence: float
    reasoning: str
    alternative_models: List[str]
    estimated_success_rate: float
    cost_estimate: float
    time_estimate: float
    success: bool = True  # 添加success属性

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_model": self.assigned_model,
            "allocation_strategy": self.allocation_strategy.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternative_models": self.alternative_models,
            "estimated_success_rate": self.estimated_success_rate,
            "cost_estimate": self.cost_estimate,
            "time_estimate": self.time_estimate,
            "success": self.success
        }


@dataclass
class ModelPerformance:
    """模型性能记录"""
    model_id: str
    task_category: TaskCategory
    success_count: int = 0
    total_count: int = 0
    avg_duration: float = 0.0
    avg_cost: float = 0.0
    avg_quality_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    recent_performances: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.success_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def recent_performance(self) -> float:
        """最近性能"""
        if not self.recent_performances:
            return 0.5
        return sum(self.recent_performances) / len(self.recent_performances)

    def update_performance(self, success: bool, duration: float, cost: float, quality: float):
        """更新性能记录"""
        self.total_count += 1
        if success:
            self.success_count += 1

        # 更新平均值
        self.avg_duration = (self.avg_duration * (self.total_count - 1) + duration) / self.total_count
        self.avg_cost = (self.avg_cost * (self.total_count - 1) + cost) / self.total_count
        self.avg_quality_score = (self.avg_quality_score * (self.total_count - 1) + quality) / self.total_count

        # 更新最近性能
        self.recent_performances.append(quality)
        if len(self.recent_performances) > 20:  # 保持最近20次记录
            self.recent_performances = self.recent_performances[-10:]

        self.last_updated = datetime.now()


class TaskAnalyzer:
    """任务分析器"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        # 预定义的关键词映射
        self.category_keywords = {
            TaskCategory.CODE_GENERATION: [
                "代码", "编程", "函数", "类", "算法", "实现", "开发", "程序", "软件",
                "python", "javascript", "java", "code", "function", "class", "algorithm"
            ],
            TaskCategory.TEXT_ANALYSIS: [
                "分析", "解释", "理解", "总结", "提取", "识别", "分类", "判断",
                "analysis", "understand", "explain", "interpret", "extract"
            ],
            TaskCategory.CREATIVE_WRITING: [
                "创作", "写作", "故事", "诗歌", "小说", "文章", "文案", "创意",
                "write", "create", "story", "poem", "novel", "creative"
            ],
            TaskCategory.DATA_ANALYSIS: [
                "数据", "统计", "图表", "计算", "分析", "报告", "趋势", "模式",
                "data", "statistics", "chart", "calculate", "report", "trend"
            ],
            TaskCategory.PROBLEM_SOLVING: [
                "解决", "问题", "方案", "方法", "策略", "建议", "优化", "改进",
                "solve", "problem", "solution", "method", "strategy", "improve"
            ],
            TaskCategory.TRANSLATION: [
                "翻译", "英文", "中文", "转换", "语言", "translate", "translation", "language"
            ],
            TaskCategory.SUMMARIZATION: [
                "摘要", "总结", "概要", "要点", "大纲", "summary", "summarize", "outline"
            ],
            TaskCategory.QUESTION_ANSWERING: [
                "问题", "回答", "疑问", "解答", "询问", "question", "answer", "ask"
            ],
            TaskCategory.REASONING: [
                "推理", "逻辑", "判断", "推导", "证明", "reasoning", "logic", "deduction"
            ],
            TaskCategory.DEBUGGING: [
                "调试", "错误", "bug", "修复", "纠错", "debug", "error", "fix"
            ],
            TaskCategory.OPTIMIZATION: [
                "优化", "改进", "提升", "效率", "性能", "optimize", "improve", "efficiency"
            ]
        }

        # 复杂度关键词
        self.complexity_keywords = {
            TaskComplexity.SIMPLE: [
                "简单", "基础", "入门", "初步", "basic", "simple", "beginner"
            ],
            TaskComplexity.MEDIUM: [
                "中等", "一般", "常规", "标准", "medium", "normal", "standard"
            ],
            TaskComplexity.COMPLEX: [
                "复杂", "高级", "困难", "挑战", "complex", "advanced", "difficult"
            ],
            TaskComplexity.VERY_COMPLEX: [
                "非常复杂", "极难", "专家级", "深度", "very complex", "expert", "deep"
            ]
        }

    def _infer_required_capabilities(self, category: 'TaskCategory') -> List[AIModelCapabilities]:
        """根据任务类别推断所需能力"""
        capability_mapping = {
            TaskCategory.CODE_GENERATION: [
                AIModelCapabilities.CODE_GENERATION,
                AIModelCapabilities.REASONING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.DATA_ANALYSIS: [
                AIModelCapabilities.DATA_ANALYSIS,
                AIModelCapabilities.REASONING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.WRITING: [
                AIModelCapabilities.TEXT_GENERATION,
                AIModelCapabilities.CREATIVE_WRITING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.RESEARCH: [
                AIModelCapabilities.CONTEXT_UNDERSTANDING,
                AIModelCapabilities.FACTUAL_ACCURACY,
                AIModelCapabilities.REASONING
            ],
            TaskCategory.TRANSLATION: [
                AIModelCapabilities.TRANSLATION,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.SUMMARIZATION: [
                AIModelCapabilities.SUMMARIZATION,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.QUESTION_ANSWERING: [
                AIModelCapabilities.QUESTION_ANSWERING,
                AIModelCapabilities.REASONING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.REASONING: [
                AIModelCapabilities.REASONING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.DEBUGGING: [
                AIModelCapabilities.CODE_GENERATION,
                AIModelCapabilities.REASONING,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ],
            TaskCategory.OPTIMIZATION: [
                AIModelCapabilities.REASONING,
                AIModelCapabilities.DATA_ANALYSIS,
                AIModelCapabilities.CONTEXT_UNDERSTANDING
            ]
        }

        return capability_mapping.get(category, [
            AIModelCapabilities.CONTEXT_UNDERSTANDING,
            AIModelCapabilities.REASONING
        ])

    async def analyze_task(self, description: str, context: Optional[Dict[str, Any]] = None) -> TaskProfile:
        """分析任务"""
        # 清理和分析描述
        cleaned_description = self._clean_text(description)
        keywords = self._extract_keywords(cleaned_description)

        # 分类任务
        category = self._classify_task(cleaned_description, keywords)
        complexity = self._estimate_complexity(cleaned_description, keywords)
        urgency = self._estimate_urgency(cleaned_description, context)

        # 估算成本和时间
        estimated_duration = self._estimate_duration(complexity, category)
        estimated_cost = self._estimate_cost(complexity, estimated_duration)

        # 创建任务需求
        task_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(cleaned_description) % 10000}"
        required_capabilities = self._infer_required_capabilities(category)
        requirements = TaskRequirements(
            task_id=task_id,
            task_type=category.value,
            complexity=complexity,
            required_capabilities=required_capabilities
        )

        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(cleaned_description) % 10000}"

        return TaskProfile(
            task_id=task_id,
            category=category,
            complexity=complexity,
            urgency=urgency,
            description=description,
            requirements=requirements,
            estimated_duration=estimated_duration,
            estimated_cost=estimated_cost,
            keywords=keywords,
            context=context or {}
        )

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除特殊字符，保留中英文和基本标点
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()（）。，！？；：]', ' ', text)
        # 移除多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = text.lower().split()
        # 过滤停用词
        stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'}
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        return keywords

    def _classify_task(self, text: str, keywords: List[str]) -> TaskCategory:
        """分类任务"""
        scores = {}
        text_lower = text.lower()

        for category, category_keywords in self.category_keywords.items():
            score = 0
            # 检查关键词匹配
            for keyword in category_keywords:
                if keyword in text_lower:
                    score += 1

            # 检查关键词列表匹配
            for keyword in keywords:
                if keyword in [kw.lower() for kw in category_keywords]:
                    score += 0.5

            scores[category] = score

        # 选择得分最高的类别
        if not scores or max(scores.values()) == 0:
            return TaskCategory.TEXT_ANALYSIS  # 默认类别

        best_category = max(scores, key=scores.get)
        return best_category

    def _estimate_complexity(self, text: str, keywords: List[str]) -> TaskComplexity:
        """估算复杂度"""
        text_lower = text.lower()
        scores = {}

        for complexity, complexity_keywords in self.complexity_keywords.items():
            score = sum(1 for keyword in complexity_keywords if keyword in text_lower)
            scores[complexity] = score

        # 基于文本长度和关键词数量调整
        length_factor = min(2.0, len(text) / 200)  # 基于长度
        keyword_factor = min(1.5, len(keywords) / 10)  # 基于关键词数量

        if scores:
            best_complexity = max(scores, key=scores.get)
            # 如果没有明确的关键词，基于长度和关键词数量判断
            if scores[best_complexity] == 0:
                if length_factor < 0.5 and keyword_factor < 0.5:
                    return TaskComplexity.SIMPLE
                elif length_factor < 1.0 and keyword_factor < 1.0:
                    return TaskComplexity.MEDIUM
                elif length_factor < 1.5:
                    return TaskComplexity.COMPLEX
                else:
                    return TaskComplexity.VERY_COMPLEX
            return best_complexity

        return TaskComplexity.MEDIUM  # 默认复杂度

    def _estimate_urgency(self, text: str, context: Optional[Dict[str, Any]]) -> TaskUrgency:
        """估算紧急程度"""
        urgency_keywords = {
            TaskUrgency.CRITICAL: ["紧急", "立即", "马上", "急", "urgent", "immediately", "asap", "critical"],
            TaskUrgency.HIGH: ["尽快", "优先", "重要", "高", "soon", "priority", "important", "high"],
            TaskUrgency.NORMAL: ["正常", "一般", "standard", "normal"],
            TaskUrgency.LOW: ["不急", "有空", "低", "low", "when possible", "no rush"]
        }

        text_lower = text.lower()
        scores = {}

        for urgency, keywords in urgency_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[urgency] = score

        # 检查上下文
        if context:
            if context.get("deadline"):
                deadline = datetime.fromisoformat(context["deadline"]) if isinstance(context["deadline"], str) else context["deadline"]
                if deadline and deadline < datetime.now() + timedelta(hours=1):
                    return TaskUrgency.CRITICAL
                elif deadline and deadline < datetime.now() + timedelta(days=1):
                    return TaskUrgency.HIGH

        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)

        return TaskUrgency.NORMAL  # 默认紧急程度

    def _estimate_duration(self, complexity: TaskComplexity, category: TaskCategory) -> float:
        """估算执行时间（秒）"""
        base_duration = {
            TaskComplexity.SIMPLE: 10,
            TaskComplexity.MEDIUM: 30,
            TaskComplexity.COMPLEX: 60,
            TaskComplexity.VERY_COMPLEX: 120
        }

        # 类别调整系数
        category_multiplier = {
            TaskCategory.CODE_GENERATION: 1.2,
            TaskCategory.DATA_ANALYSIS: 1.1,
            TaskCategory.CREATIVE_WRITING: 1.3,
            TaskCategory.REASONING: 1.4,
            TaskCategory.PROBLEM_SOLVING: 1.2,
            TaskCategory.TRANSLATION: 0.8,
            TaskCategory.SUMMARIZATION: 0.7,
            TaskCategory.DEBUGGING: 1.5,
            TaskCategory.OPTIMIZATION: 1.3
        }

        base = base_duration.get(complexity, 30)
        multiplier = category_multiplier.get(category, 1.0)

        return base * multiplier

    def _estimate_cost(self, complexity: TaskComplexity, duration: float) -> float:
        """估算成本（相对单位）"""
        base_cost = {
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MEDIUM: 2.0,
            TaskComplexity.COMPLEX: 4.0,
            TaskComplexity.VERY_COMPLEX: 8.0
        }

        # 基于时间的成本调整
        time_factor = duration / 30  # 30秒作为基准

        base = base_cost.get(complexity, 2.0)
        return base * max(0.5, min(2.0, time_factor))


class TaskDispatcher:
    """任务分配器"""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.analyzer = TaskAnalyzer()
        self.model_profiles: Dict[str, AIModelProfile] = {}
        self.performance_history: Dict[str, Dict[TaskCategory, ModelPerformance]] = defaultdict(dict)
        self.current_load: Dict[str, int] = defaultdict(int)  # 当前负载
        self.allocation_history: List[AllocationResult] = []

        # 初始化模型配置
        self._initialize_models()

    def _initialize_models(self):
        """初始化模型配置"""
        # 预定义的模型档案
        model_configs = {
            "kimi": AIModelProfile(
                model_id="kimi",
                model_name="Kimi AI",
                provider="moonshot",
                capabilities=["code_generation", "reasoning", "problem_solving", "data_analysis"],
                strengths=["代码", "逻辑推理", "问题解决"],
                weaknesses=["长文本生成"],
                cost_tier="medium",
                performance_tier="premium",
                max_tokens=8000,
                response_time=5.0,
                accuracy_score=0.85,
                reliability_score=0.9,
                context_window=4000
            ),
            "dashscope": AIModelProfile(
                model_id="dashscope",
                model_name="DashScope",
                provider="alibaba",
                capabilities=["text_analysis", "translation", "summarization", "question_answering"],
                strengths=["文本分析", "翻译"],
                weaknesses=["复杂推理"],
                cost_tier="low",
                performance_tier="standard",
                max_tokens=6000,
                response_time=4.0,
                accuracy_score=0.8,
                reliability_score=0.85,
                context_window=3000
            ),
            "metaso": AIModelProfile(
                model_id="metaso",
                model_name="Metaso AI",
                provider="metaso",
                capabilities=["creative_writing", "document_processing", "reasoning", "planning"],
                strengths=["创意写作", "文档处理"],
                weaknesses=["实时响应"],
                cost_tier="medium",
                performance_tier="premium",
                max_tokens=10000,
                response_time=6.0,
                accuracy_score=0.82,
                reliability_score=0.88,
                context_window=5000
            )
        }

        self.model_profiles.update(model_configs)

    async def dispatch_task(self,
                          description: str,
                          context: Optional[Dict[str, Any]] = None,
                          preferred_strategy: AllocationStrategy = AllocationStrategy.ADAPTIVE) -> AllocationResult:
        """分配任务"""
        try:
            self.logger.info(f"开始分配任务: {description[:50]}...")

            # 分析任务
            task_profile = await self.analyzer.analyze_task(description, context)

            # 选择分配策略
            strategy = preferred_strategy if preferred_strategy != AllocationStrategy.ADAPTIVE else self._select_optimal_strategy(task_profile)

            # 执行分配
            allocation_result = await self._allocate_task(task_profile, strategy)

            # 更新负载
            self.current_load[allocation_result.assigned_model] += 1

            # 记录分配历史
            self.allocation_history.append(allocation_result)

            self.logger.info(f"任务分配完成: {task_profile.task_id} -> {allocation_result.assigned_model}")

            return allocation_result

        except Exception as e:
            self.logger.error(f"任务分配失败: {e}")
            # 返回默认分配
            default_model = list(self.model_profiles.keys())[0] if self.model_profiles else "kimi"
            return AllocationResult(
                task_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                assigned_model=default_model,
                allocation_strategy=AllocationStrategy.EXPERTISE_MATCH,
                confidence=0.3,
                reasoning="分配失败，使用默认模型",
                alternative_models=[],
                estimated_success_rate=0.5,
                cost_estimate=1.0,
                time_estimate=10.0
            )

    def _select_optimal_strategy(self, task_profile: TaskProfile) -> AllocationStrategy:
        """选择最优分配策略"""
        # 基于任务特征选择策略
        if task_profile.urgency in [TaskUrgency.HIGH, TaskUrgency.CRITICAL]:
            return AllocationStrategy.SPEED_FIRST
        elif task_profile.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            return AllocationStrategy.QUALITY_FIRST
        elif task_profile.category in [TaskCategory.CODE_GENERATION, TaskCategory.DEBUGGING]:
            return AllocationStrategy.EXPERTISE_MATCH
        else:
            return AllocationStrategy.PERFORMANCE_BASED

    async def _allocate_task(self, task_profile: TaskProfile, strategy: AllocationStrategy) -> AllocationResult:
        """执行任务分配"""
        if strategy == AllocationStrategy.EXPERTISE_MATCH:
            return await self._expertise_match_allocation(task_profile)
        elif strategy == AllocationStrategy.PERFORMANCE_BASED:
            return await self._performance_based_allocation(task_profile)
        elif strategy == AllocationStrategy.LOAD_BALANCING:
            return await self._load_balancing_allocation(task_profile)
        elif strategy == AllocationStrategy.COST_OPTIMIZATION:
            return await self._cost_optimization_allocation(task_profile)
        elif strategy == AllocationStrategy.QUALITY_FIRST:
            return await self._quality_first_allocation(task_profile)
        elif strategy == AllocationStrategy.SPEED_FIRST:
            return await self._speed_first_allocation(task_profile)
        else:
            return await self._expertise_match_allocation(task_profile)

    async def _expertise_match_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """专业知识匹配分配"""
        scores = {}
        category = task_profile.category

        for model_id, profile in self.model_profiles.items():
            # 检查模型是否有该类别的专业能力
            if category.value in profile.capabilities:
                # 专业度评分
                specialty_score = 1.0
                for specialty in profile.specialties:
                    if any(keyword in task_profile.description.lower() for keyword in specialty.lower().split()):
                        specialty_score += 0.2

                # 可靠性评分
                reliability_score = profile.reliability

                # 历史性能评分
                performance = self.performance_history.get(model_id, {}).get(category)
                if performance:
                    performance_score = performance.recent_performance
                else:
                    performance_score = 0.5

                # 综合评分
                total_score = (specialty_score * 0.4 + reliability_score * 0.3 + performance_score * 0.3)
                scores[model_id] = total_score

        if not scores:
            # 如果没有专业匹配，使用第一个可用模型
            best_model = list(self.model_profiles.keys())[0]
            confidence = 0.3
            reasoning = "没有专业匹配模型"
        else:
            best_model = max(scores, key=scores.get)
            confidence = scores[best_model]
            reasoning = f"专业匹配，评分: {scores[best_model]:.2f}"

        # 获取备选模型
        alternatives = [model for model in scores.keys() if model != best_model][:2]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.EXPERTISE_MATCH,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=min(0.95, confidence),
            cost_estimate=task_profile.estimated_cost,
            time_estimate=task_profile.estimated_duration
        )

    async def _performance_based_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """基于性能的分配"""
        scores = {}
        category = task_profile.category

        for model_id in self.model_profiles.keys():
            performance = self.performance_history.get(model_id, {}).get(category)

            if performance and performance.total_count > 0:
                # 基于历史性能评分
                success_rate = performance.success_rate
                quality_score = performance.recent_performance
                speed_score = 1.0 / max(1.0, performance.avg_duration)  # 速度评分（越快越高）

                total_score = success_rate * 0.4 + quality_score * 0.4 + speed_score * 0.2
                scores[model_id] = total_score
            else:
                # 新模型给予默认评分
                profile = self.model_profiles[model_id]
                scores[model_id] = profile.reliability * 0.6

        if not scores:
            best_model = list(self.model_profiles.keys())[0]
            confidence = 0.5
            reasoning = "无历史数据，使用默认选择"
        else:
            best_model = max(scores, key=scores.get)
            confidence = scores[best_model]
            reasoning = f"基于历史性能，评分: {scores[best_model]:.2f}"

        alternatives = [model for model in scores.keys() if model != best_model][:2]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.PERFORMANCE_BASED,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=min(0.95, confidence),
            cost_estimate=task_profile.estimated_cost,
            time_estimate=task_profile.estimated_duration
        )

    async def _load_balancing_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """负载均衡分配"""
        # 计算负载评分（负载越低评分越高）
        load_scores = {}
        max_load = max(self.current_load.values()) if self.current_load else 1

        for model_id in self.model_profiles.keys():
            current_load = self.current_load.get(model_id, 0)
            load_score = (max_load - current_load) / max_load
            load_scores[model_id] = load_score

        # 结合基础性能
        final_scores = {}
        for model_id, load_score in load_scores.items():
            profile = self.model_profiles[model_id]
            performance_score = profile.reliability
            final_scores[model_id] = load_score * 0.6 + performance_score * 0.4

        best_model = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_model]
        reasoning = f"负载均衡，当前负载: {self.current_load.get(best_model, 0)}"

        alternatives = sorted(final_scores, key=final_scores.get, reverse=True)[1:3]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.LOAD_BALANCING,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=0.8,
            cost_estimate=task_profile.estimated_cost,
            time_estimate=task_profile.estimated_duration
        )

    async def _cost_optimization_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """成本优化分配"""
        cost_scores = {}

        for model_id, profile in self.model_profiles.items():
            # 成本评分（成本越低评分越高）
            max_cost = max(p.cost_per_token for p in self.model_profiles.values())
            cost_score = 1.0 - (profile.cost_per_token / max_cost)

            # 考虑性能
            performance_score = profile.reliability

            total_score = cost_score * 0.6 + performance_score * 0.4
            cost_scores[model_id] = total_score

        best_model = max(cost_scores, key=cost_scores.get)
        confidence = cost_scores[best_model]
        reasoning = f"成本优化，模型成本: {self.model_profiles[best_model].cost_per_token}"

        alternatives = sorted(cost_scores, key=cost_scores.get, reverse=True)[1:3]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.COST_OPTIMIZATION,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=0.75,
            cost_estimate=task_profile.estimated_cost * 0.8,  # 成本优化
            time_estimate=task_profile.estimated_duration
        )

    async def _quality_first_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """质量优先分配"""
        quality_scores = {}

        for model_id, profile in self.model_profiles.items():
            # 可靠性评分
            reliability_score = profile.reliability

            # 专业度评分
            category = task_profile.category
            expertise_score = 1.0 if category.value in profile.capabilities else 0.5

            # 历史质量评分
            performance = self.performance_history.get(model_id, {}).get(category)
            if performance:
                historical_quality = performance.avg_quality_score
            else:
                historical_quality = 0.5

            total_score = reliability_score * 0.3 + expertise_score * 0.3 + historical_quality * 0.4
            quality_scores[model_id] = total_score

        best_model = max(quality_scores, key=quality_scores.get)
        confidence = quality_scores[best_model]
        reasoning = f"质量优先，综合评分: {quality_scores[best_model]:.2f}"

        alternatives = sorted(quality_scores, key=quality_scores.get, reverse=True)[1:3]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.QUALITY_FIRST,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=0.9,
            cost_estimate=task_profile.estimated_cost * 1.2,  # 质量优先可能需要更多成本
            time_estimate=task_profile.estimated_duration * 1.1
        )

    async def _speed_first_allocation(self, task_profile: TaskProfile) -> AllocationResult:
        """速度优先分配"""
        speed_scores = {}

        for model_id, profile in self.model_profiles.items():
            # 响应时间评分（时间越短评分越高）
            max_time = max(p.average_response_time for p in self.model_profiles.values())
            speed_score = 1.0 - (profile.average_response_time / max_time)

            # 考虑可靠性
            reliability_score = profile.reliability

            total_score = speed_score * 0.7 + reliability_score * 0.3
            speed_scores[model_id] = total_score

        best_model = max(speed_scores, key=speed_scores.get)
        confidence = speed_scores[best_model]
        reasoning = f"速度优先，响应时间: {self.model_profiles[best_model].average_response_time}s"

        alternatives = sorted(speed_scores, key=speed_scores.get, reverse=True)[1:3]

        return AllocationResult(
            task_id=task_profile.task_id,
            assigned_model=best_model,
            allocation_strategy=AllocationStrategy.SPEED_FIRST,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_success_rate=0.8,
            cost_estimate=task_profile.estimated_cost,
            time_estimate=self.model_profiles[best_model].average_response_time
        )

    async def update_task_result(self, task_id: str, model_id: str, success: bool, duration: float, cost: float, quality: float):
        """更新任务结果"""
        # 减少负载
        if self.current_load[model_id] > 0:
            self.current_load[model_id] -= 1

        # 更新性能历史
        # 需要从任务档案中获取类别信息
        task_category = TaskCategory.TEXT_ANALYSIS  # 默认值，实际应该从分配历史中获取

        if model_id not in self.performance_history:
            self.performance_history[model_id] = {}

        if task_category not in self.performance_history[model_id]:
            self.performance_history[model_id][task_category] = ModelPerformance(
                model_id=model_id,
                task_category=task_category
            )

        self.performance_history[model_id][task_category].update_performance(success, duration, cost, quality)

        self.logger.info(f"任务结果更新: {task_id} -> {model_id}, 成功: {success}, 质量: {quality:.2f}")

    def get_allocation_statistics(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        if not self.allocation_history:
            return {"message": "暂无分配历史"}

        # 策略统计
        strategy_counts = defaultdict(int)
        model_counts = defaultdict(int)

        for allocation in self.allocation_history[-100:]:  # 最近100次分配
            strategy_counts[allocation.allocation_strategy.value] += 1
            model_counts[allocation.assigned_model] += 1

        # 平均成功率估算
        avg_confidence = sum(a.confidence for a in self.allocation_history[-100:]) / min(100, len(self.allocation_history))

        return {
            "total_allocations": len(self.allocation_history),
            "recent_allocations": min(100, len(self.allocation_history)),
            "strategy_distribution": dict(strategy_counts),
            "model_distribution": dict(model_counts),
            "average_confidence": avg_confidence,
            "current_load": dict(self.current_load),
            "performance_summary": self._get_performance_summary()
        }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        for model_id, categories in self.performance_history.items():
            model_summary = {}
            for category, performance in categories.items():
                model_summary[category.value] = {
                    "success_rate": performance.success_rate,
                    "avg_quality": performance.avg_quality_score,
                    "total_tasks": performance.total_count
                }
            summary[model_id] = model_summary
        return summary


# 导出
__all__ = [
    "TaskDispatcher",
    "TaskAnalyzer",
    "TaskProfile",
    "AllocationResult",
    "TaskCategory",
    "TaskComplexity",
    "TaskUrgency",
    "AllocationStrategy"
]