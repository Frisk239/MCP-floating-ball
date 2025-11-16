#!/usr/bin/env python3
"""
增强NLP处理器 - 基于Everywhere项目的设计理念

实现多层次意图识别、智能参数提取和复杂任务分解
"""

import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.core.logging import get_logger
from src.assistant.nlp_processor import IntentType, Intent, CommandType

logger = get_logger("enhanced_nlp_processor")


class PrimaryIntent(Enum):
    """一级意图分类"""
    NAVIGATION = "navigation"       # 导航类：打开、访问、前往
    SEARCH = "search"              # 搜索类：搜索、查找、寻找
    INFORMATION = "information"    # 信息类：查询、查看、了解
    OPERATION = "operation"        # 操作类：操作、执行、处理
    CREATION = "creation"          # 创建类：创建、新建、生成
    SYSTEM = "system"              # 系统类：系统相关操作


class SecondaryIntent(Enum):
    """二级意图细分"""
    # 导航类细分
    WEB_NAVIGATION = "web_navigation"    # 网页导航
    APP_LAUNCH = "app_launch"           # 应用启动
    FILE_OPEN = "file_open"             # 文件打开

    # 搜索类细分
    WEB_SEARCH = "web_search"           # 网页搜索
    LOCAL_SEARCH = "local_search"       # 本地搜索
    CONTENT_SEARCH = "content_search"   # 内容搜索

    # 信息类细分
    WEATHER_QUERY = "weather_query"     # 天气查询
    SYSTEM_INFO = "system_info"         # 系统信息
    FILE_INFO = "file_info"             # 文件信息

    # 操作类细分
    SCREEN_CAPTURE = "screen_capture"   # 屏幕截图
    OCR_OPERATION = "ocr_operation"     # OCR识别
    IMAGE_ANALYSIS = "image_analysis"   # 图像分析


@dataclass
class HierarchicalIntent:
    """层次化意图结构"""
    primary: PrimaryIntent
    secondary: SecondaryIntent
    action: str
    parameters: Dict[str, Any]
    confidence: float
    context_aware: bool = False
    disambiguation: Optional[str] = None

    def to_legacy_intent(self) -> Intent:
        """转换为兼容的Intent对象"""
        # 完善的意图映射表
        intent_mapping = {
            # 导航类意图
            (PrimaryIntent.NAVIGATION, SecondaryIntent.WEB_NAVIGATION): IntentType.WEB_SCRAPING,
            (PrimaryIntent.NAVIGATION, SecondaryIntent.APP_LAUNCH): IntentType.APP_LAUNCH,
            (PrimaryIntent.NAVIGATION, SecondaryIntent.FILE_OPEN): IntentType.FILE_OPERATION,

            # 搜索类意图
            (PrimaryIntent.SEARCH, SecondaryIntent.WEB_SEARCH): IntentType.WEB_SEARCH,
            (PrimaryIntent.SEARCH, SecondaryIntent.LOCAL_SEARCH): IntentType.FILE_OPERATION,
            (PrimaryIntent.SEARCH, SecondaryIntent.CONTENT_SEARCH): IntentType.FILE_OPERATION,

            # 信息类意图
            (PrimaryIntent.INFORMATION, SecondaryIntent.WEATHER_QUERY): IntentType.WEB_SEARCH,  # 天气查询通过网页搜索实现
            (PrimaryIntent.INFORMATION, SecondaryIntent.SYSTEM_INFO): IntentType.SYSTEM_INFO,
            (PrimaryIntent.INFORMATION, SecondaryIntent.FILE_INFO): IntentType.FILE_OPERATION,

            # 操作类意图
            (PrimaryIntent.OPERATION, SecondaryIntent.SCREEN_CAPTURE): IntentType.SCREEN_CAPTURE,
            (PrimaryIntent.OPERATION, SecondaryIntent.OCR_OPERATION): IntentType.OCR,
            (PrimaryIntent.OPERATION, SecondaryIntent.IMAGE_ANALYSIS): IntentType.IMAGE_ANALYSIS,

            # 系统类意图（默认映射）
            (PrimaryIntent.SYSTEM, SecondaryIntent.SYSTEM_INFO): IntentType.SYSTEM_INFO,
        }

        # 获取映射的意图类型
        legacy_intent_type = intent_mapping.get((self.primary, self.secondary), IntentType.UNKNOWN)

        # 如果映射失败，尝试基于主要意图的默认映射
        if legacy_intent_type == IntentType.UNKNOWN:
            default_mapping = {
                PrimaryIntent.NAVIGATION: IntentType.WEB_SCRAPING,
                PrimaryIntent.SEARCH: IntentType.WEB_SEARCH,
                PrimaryIntent.INFORMATION: IntentType.SYSTEM_INFO,
                PrimaryIntent.OPERATION: IntentType.SCREEN_CAPTURE,
                PrimaryIntent.CREATION: IntentType.FILE_OPERATION,
                PrimaryIntent.SYSTEM: IntentType.SYSTEM_INFO,
            }
            legacy_intent_type = default_mapping.get(self.primary, IntentType.UNKNOWN)

        return Intent(
            intent_type=legacy_intent_type,
            confidence=self.confidence,
            action=self.action,
            parameters=self.parameters,
            raw_text="",  # 将在调用处设置
            matched_keywords=[]
        )


class HierarchicalIntentRecognizer:
    """多层次意图识别器"""

    def __init__(self):
        # 一级意图关键词
        self.primary_intents = {
            PrimaryIntent.NAVIGATION: ["打开", "访问", "前往", "go", "open", "visit", "启动", "运行", "launch"],
            PrimaryIntent.SEARCH: ["搜索", "查找", "寻找", "search", "find", "查询"],
            PrimaryIntent.INFORMATION: ["查询", "查看", "了解", "query", "check", "信息", "怎么样"],
            PrimaryIntent.OPERATION: ["操作", "执行", "处理", "截图", "识别", "capture", "operate"],
            PrimaryIntent.CREATION: ["创建", "新建", "生成", "create", "generate", "make"],
            PrimaryIntent.SYSTEM: ["系统", "窗口", "应用", "程序", "system", "window", "app"]
        }

        # 二级意图关键词
        self.secondary_intents = {
            PrimaryIntent.NAVIGATION: {
                SecondaryIntent.WEB_NAVIGATION: ["网址", "网站", "网页", "百度", "谷歌", "必应", "website", "web"],
                SecondaryIntent.APP_LAUNCH: ["应用", "程序", "软件", "app", "program"],
                SecondaryIntent.FILE_OPEN: ["文件", "文档", "file", "document"]
            },
            PrimaryIntent.SEARCH: {
                SecondaryIntent.WEB_SEARCH: ["网页", "网络", "互联网", "web", "internet"],
                SecondaryIntent.LOCAL_SEARCH: ["本地", "电脑", "文件", "local", "computer"],
                SecondaryIntent.CONTENT_SEARCH: ["内容", "文本", "content", "text"]
            },
            PrimaryIntent.INFORMATION: {
                SecondaryIntent.WEATHER_QUERY: ["天气", "气温", "下雨", "晴天", "weather", "温度"],
                SecondaryIntent.SYSTEM_INFO: ["系统", "电脑", "硬件", "software", "hardware"],
                SecondaryIntent.FILE_INFO: ["文件", "文档", "信息", "file", "document"]
            },
            PrimaryIntent.OPERATION: {
                SecondaryIntent.SCREEN_CAPTURE: ["截图", "截屏", "抓屏", "screenshot", "capture"],
                SecondaryIntent.OCR_OPERATION: ["识别", "文字", "图片", "ocr", "text"],
                SecondaryIntent.IMAGE_ANALYSIS: ["分析", "图片", "图像", "analyze", "image"]
            }
        }

        # 上下文相关意图模式
        self.context_patterns = {
            "WEATHER_CONTEXT": ["天气", "气温", "下雨", "晴天", "weather", "温度"],
            "WEB_CONTEXT": ["网址", "网站", "网页", "搜索", "website", "web", "百度", "谷歌"],
            "APP_CONTEXT": ["应用", "程序", "软件", "app", "program"],
            "FILE_CONTEXT": ["文件", "文档", "保存", "打开", "file"]
        }

        # 网站名称映射
        self.site_mapping = {
            "百度": "https://www.baidu.com",
            "谷歌": "https://www.google.com",
            "google": "https://www.google.com",
            "必应": "https://www.bing.com",
            "淘宝": "https://www.taobao.com",
            "京东": "https://www.jd.com"
        }

        logger.info("多层次意图识别器初始化完成")

    def recognize_intent(self, text: str, context: Optional[Dict] = None) -> HierarchicalIntent:
        """
        多层次意图识别

        Args:
            text: 用户输入文本
            context: 上下文信息

        Returns:
            HierarchicalIntent: 层次化意图对象
        """
        try:
            # 文本预处理
            processed_text = self._preprocess_text(text)

            # 一级意图识别
            primary_intent = self._detect_primary_intent(processed_text)

            # 二级意图细化
            secondary_intent = self._refine_secondary_intent(primary_intent, processed_text, context)

            # 具体动作和参数提取
            action, parameters = self._extract_action_and_params(
                primary_intent, secondary_intent, processed_text
            )

            # 计算置信度
            confidence = self._calculate_confidence(primary_intent, secondary_intent, processed_text)

            # 歧义消解
            disambiguation = self._resolve_disambiguation(
                primary_intent, secondary_intent, processed_text
            )

            intent = HierarchicalIntent(
                primary=primary_intent,
                secondary=secondary_intent,
                action=action,
                parameters=parameters,
                confidence=confidence,
                context_aware=context is not None,
                disambiguation=disambiguation
            )

            logger.info(f"意图识别完成: {primary_intent.value} -> {secondary_intent.value}, 置信度: {confidence:.2f}")
            return intent

        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return HierarchicalIntent(
                primary=PrimaryIntent.SYSTEM,
                secondary=SecondaryIntent.SYSTEM_INFO,
                action="unknown",
                parameters={},
                confidence=0.0
            )

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 转换为小写并去除空格
        text = text.lower().strip()
        # 统一标点符号
        text = re.sub(r'[，。！？]', ',', text)
        return text

    def _detect_primary_intent(self, text: str) -> PrimaryIntent:
        """检测一级意图"""
        max_score = 0
        best_intent = PrimaryIntent.SYSTEM

        for primary_intent, keywords in self.primary_intents.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > max_score:
                max_score = score
                best_intent = primary_intent

        return best_intent

    def _refine_secondary_intent(self, primary_intent: PrimaryIntent, text: str, context: Optional[Dict]) -> SecondaryIntent:
        """基于上下文细化二级意图"""
        if primary_intent == PrimaryIntent.NAVIGATION:
            # 特殊处理：优先检查是否为网页导航
            if any(keyword in text for keyword in ["百度", "谷歌", "必应", "网站", "网页", "网址"]):
                return SecondaryIntent.WEB_NAVIGATION
            elif any(keyword in text for keyword in ["应用", "程序", "软件"]):
                return SecondaryIntent.APP_LAUNCH
            elif "文件" in text or "文档" in text:
                return SecondaryIntent.FILE_OPEN
            else:
                # 智能判断：通过后续内容推测
                # 提取"打开"后面的内容进行分析
                import re
                match = re.search(r'打开[\s]*([^，。！？\s]+)', text)
                if match:
                    entity = match.group(1)
                    # 检查是否为常见应用名称
                    common_apps = ["记事本", "计算器", "画图", "微信", "QQ", "浏览器", "Chrome", "Firefox", "Word", "Excel", "PowerPoint"]
                    if entity in common_apps:
                        return SecondaryIntent.APP_LAUNCH
                    # 检查是否为网站名称
                    elif entity in ["百度", "谷歌", "必应", "淘宝", "京东", "知乎", "微博"]:
                        return SecondaryIntent.WEB_NAVIGATION

                # 如果无法确定，默认为应用启动（更常见的操作）
                return SecondaryIntent.APP_LAUNCH

        elif primary_intent == PrimaryIntent.INFORMATION:
            # 天气查询优先级高
            if any(keyword in text for keyword in ["天气", "气温", "下雨", "晴天", "weather", "温度"]):
                return SecondaryIntent.WEATHER_QUERY
            elif any(keyword in text for keyword in ["系统", "电脑", "硬件"]):
                return SecondaryIntent.SYSTEM_INFO
            else:
                return SecondaryIntent.SYSTEM_INFO

        elif primary_intent == PrimaryIntent.SEARCH:
            # 网页搜索优先级高
            if any(keyword in text for keyword in ["网页", "网络", "互联网", "百度", "谷歌"]):
                return SecondaryIntent.WEB_SEARCH
            else:
                return SecondaryIntent.WEB_SEARCH  # 默认网页搜索

        elif primary_intent == PrimaryIntent.OPERATION:
            if any(keyword in text for keyword in ["截图", "截屏", "抓屏", "screenshot"]):
                return SecondaryIntent.SCREEN_CAPTURE
            elif any(keyword in text for keyword in ["识别", "文字", "ocr", "文本"]):
                return SecondaryIntent.OCR_OPERATION
            elif any(keyword in text for keyword in ["分析", "图片", "图像", "analyze"]):
                return SecondaryIntent.IMAGE_ANALYSIS

        # 默认值
        default_mapping = {
            PrimaryIntent.NAVIGATION: SecondaryIntent.WEB_NAVIGATION,
            PrimaryIntent.SEARCH: SecondaryIntent.WEB_SEARCH,
            PrimaryIntent.INFORMATION: SecondaryIntent.SYSTEM_INFO,
            PrimaryIntent.OPERATION: SecondaryIntent.SCREEN_CAPTURE
        }

        return default_mapping.get(primary_intent, SecondaryIntent.SYSTEM_INFO)

    def _extract_action_and_params(self, primary_intent: PrimaryIntent, secondary_intent: SecondaryIntent, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取具体动作和参数"""
        if secondary_intent == SecondaryIntent.WEB_NAVIGATION:
            return self._extract_web_navigation_params(text)
        elif secondary_intent == SecondaryIntent.APP_LAUNCH:
            return self._extract_app_launch_params(text)
        elif secondary_intent == SecondaryIntent.WEB_SEARCH:
            return self._extract_search_params(text)
        elif secondary_intent == SecondaryIntent.WEATHER_QUERY:
            return self._extract_weather_params(text)
        elif secondary_intent == SecondaryIntent.SCREEN_CAPTURE:
            return self._extract_screenshot_params(text)
        elif secondary_intent == SecondaryIntent.OCR_OPERATION:
            return self._extract_ocr_params(text)

        # 默认处理
        return "execute", {"input": text}

    def _extract_web_navigation_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取网页导航参数 - 解决"打开百度"被错误识别的问题"""
        # 检查特定网站
        for site_name, site_url in self.site_mapping.items():
            if site_name in text:
                return "open_website", {"url": site_url, "site_name": site_name}

        # 检查是否包含URL模式
        url_pattern = r'https?://[^\s,\'"]+|www\.[^\s,\'"]+'
        url_match = re.search(url_pattern, text)
        if url_match:
            return "open_website", {"url": url_match.group(0)}

        # 默认打开百度
        return "open_website", {"url": "https://www.baidu.com", "site_name": "百度"}

    def _extract_app_launch_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取应用启动参数"""
        app_patterns = [
            r'打开["\']([^"\']+)["\']',
            r'启动["\']([^"\']+)["\']',
            r'运行["\']([^"\']+)["\']',
            r'打开(.+?)(?:应用|程序|软件)?',
            r'启动(.+?)(?:应用|程序|软件)?'
        ]

        for pattern in app_patterns:
            match = re.search(pattern, text)
            if match:
                app_name = match.group(1).strip()
                if app_name and len(app_name) > 1:
                    return "launch_app", {"app_name": app_name}

        return "launch_app", {"app_name": text}  # 使用原始文本作为应用名

    def _extract_search_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取搜索参数"""
        params = {"action": "search"}

        # 搜索引擎识别
        if "百度" in text:
            params["engine"] = "baidu"
            params["url"] = "https://www.baidu.com"
        elif "谷歌" in text or "google" in text:
            params["engine"] = "google"
            params["url"] = "https://www.google.com"
        elif "必应" in text:
            params["engine"] = "bing"
            params["url"] = "https://www.bing.com"
        else:
            params["engine"] = "baidu"
            params["url"] = "https://www.baidu.com"

        # 搜索内容提取
        search_patterns = [
            r'搜索["\']([^"\']+)["\']',
            r'查找["\']([^"\']+)["\']',
            r'搜索(.+)',
            r'查找(.+)',
            r'关于(.+)',
            r'(.+)的?信息'
        ]

        for pattern in search_patterns:
            match = re.search(pattern, text)
            if match:
                query = match.group(1) if match.groups() else match.group(0)
                # 清理查询词
                query = re.sub(r'(搜索|查找|关于|的?信息).*$', '', query).strip()
                if query and len(query) > 1:
                    params["query"] = query
                    break

        if "query" not in params:
            # 如果没有提取到查询词，尝试从整个文本中推断
            clean_text = re.sub(r'(百度|谷歌|必应|搜索|查找)', '', text).strip()
            if clean_text:
                params["query"] = clean_text
            else:
                params["query"] = "默认搜索"

        return "web_search", params

    def _extract_weather_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取天气查询参数 - 解决"福州天气"识别问题"""
        params = {"action": "get_weather"}

        # 城市名称提取模式
        city_patterns = [
            r'(.+?)的?天气',
            r'查询(.+?)的?天气',
            r'(.+?)的?气温',
            r'(.+?)的?天气预报',
            r'(.+?)怎么样',
            r'(?:今天|明天|后天)?(.+?)天气'
        ]

        city_found = False
        for pattern in city_patterns:
            match = re.search(pattern, text)
            if match:
                city = match.group(1).strip()
                # 过滤无效的匹配
                if city and len(city) > 1 and city not in ["今天", "明天", "后天", "查询", "查看"]:
                    params["city"] = city
                    city_found = True
                    break

        if not city_found:
            # 尝试简单的地名匹配
            common_cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都", "西安", "福州", "厦门"]
            for city in common_cities:
                if city in text:
                    params["city"] = city
                    city_found = True
                    break

        if not city_found:
            params["city"] = "当前位置"

        # 时间信息
        if "今天" in text:
            params["time"] = "today"
        elif "明天" in text:
            params["time"] = "tomorrow"
        elif "后天" in text:
            params["time"] = "day_after_tomorrow"
        else:
            params["time"] = "today"

        return "weather_query", params

    def _extract_screenshot_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取截图参数"""
        params = {"action": "capture_screen"}

        if "全屏" in text or "整个屏幕" in text:
            params["capture_type"] = "full"
        elif "区域" in text or "部分" in text:
            params["capture_type"] = "region"
        elif "窗口" in text:
            params["capture_type"] = "window"
        else:
            params["capture_type"] = "full"  # 默认全屏

        return "screenshot", params

    def _extract_ocr_params(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """提取OCR参数"""
        params = {"action": "recognize_text"}

        # OCR引擎
        if "tesseract" in text:
            params["engine"] = "tesseract"
        elif "easyocr" in text:
            params["engine"] = "easyocr"
        else:
            params["engine"] = "tesseract"  # 默认

        return "ocr", params

    def _calculate_confidence(self, primary_intent: PrimaryIntent, secondary_intent: SecondaryIntent, text: str) -> float:
        """计算意图识别置信度"""
        confidence = 0.0

        # 一级意图匹配度 - 使用最高匹配而不是平均
        primary_keywords = self.primary_intents.get(primary_intent, [])
        primary_matches = sum(1 for keyword in primary_keywords if keyword in text)
        if primary_matches > 0:
            confidence += 0.4  # 有一级匹配就有基础分

        # 二级意图匹配度
        if primary_intent in self.secondary_intents:
            secondary_keywords = self.secondary_intents[primary_intent].get(secondary_intent, [])
            secondary_matches = sum(1 for keyword in secondary_keywords if keyword in text)
            if secondary_matches > 0:
                confidence += 0.4  # 有二级匹配就加分

        # 智能实体匹配奖励
        entity_match_confidence = self._calculate_entity_match_confidence(text, secondary_intent)
        confidence += entity_match_confidence * 0.2

        # 基础置信度保证（至少有一定置信度，因为我们已经通过了意图识别）
        if confidence > 0:
            confidence = max(confidence, 0.6)  # 确保最小置信度

        return min(confidence, 1.0)

    def _calculate_entity_match_confidence(self, text: str, secondary_intent: SecondaryIntent) -> float:
        """计算实体匹配置信度"""
        if secondary_intent == SecondaryIntent.WEB_NAVIGATION:
            # 检查网站名称匹配
            for site_name in self.site_mapping.keys():
                if site_name in text:
                    return 1.0
            # 检查URL模式
            import re
            url_pattern = r'https?://[^\s,\'"]+|www\.[^\s,\'"]+'
            if re.search(url_pattern, text):
                return 1.0

        elif secondary_intent == SecondaryIntent.APP_LAUNCH:
            # 检查常见应用名称
            common_apps = ["记事本", "计算器", "画图", "微信", "QQ", "浏览器", "Chrome", "Firefox", "Word", "Excel", "PowerPoint"]
            for app in common_apps:
                if app in text:
                    return 1.0

        elif secondary_intent == SecondaryIntent.WEATHER_QUERY:
            # 检查城市名称
            common_cities = ["北京", "上海", "广州", "深圳", "福州", "厦门", "杭州", "南京", "武汉", "成都", "西安"]
            for city in common_cities:
                if city in text:
                    return 1.0
            # 检查天气关键词
            weather_keywords = ["天气", "气温", "下雨", "晴天", "温度", "weather"]
            for keyword in weather_keywords:
                if keyword in text:
                    return 0.8

        return 0.0

    def _resolve_disambiguation(self, primary_intent: PrimaryIntent, secondary_intent: SecondaryIntent, text: str) -> Optional[str]:
        """歧义消解"""
        # 如果意图识别置信度较低，提供消解建议
        if (primary_intent == PrimaryIntent.NAVIGATION and
            secondary_intent == SecondaryIntent.WEB_NAVIGATION and
            not any(site in text for site in self.site_mapping.keys())):
            return "您是想打开网站还是启动本地应用？"

        return None


class EnhancedNLPProcessor:
    """增强NLP处理器 - 整合多层次意图识别"""

    def __init__(self, user_id: str = "default"):
        self.intent_recognizer = HierarchicalIntentRecognizer()
        self.user_id = user_id

        # 初始化智能学习器
        try:
            from src.assistant.intelligent_learner import IntelligentLearner
            self.learner = IntelligentLearner(user_id)
            self.learning_enabled = True
            logger.info("智能学习器已启用")
        except Exception as e:
            logger.warning(f"智能学习器初始化失败，将使用基础功能: {e}")
            self.learner = None
            self.learning_enabled = False

        logger.info(f"增强NLP处理器初始化完成 (用户ID: {user_id}, 学习功能: {'启用' if self.learning_enabled else '禁用'})")

    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """
        处理文本并返回意图（兼容原接口）

        Args:
            text: 用户输入文本
            context: 上下文信息

        Returns:
            Intent: 兼容的意图对象
        """
        try:
            # 1. 使用增强的意图识别器
            hierarchical_intent = self.intent_recognizer.recognize_intent(text, context)

            # 2. 如果启用了学习功能，使用智能预测进行优化
            if self.learning_enabled and self.learner:
                try:
                    prediction = self.learner.predict_intent(text, context)

                    # 如果历史预测置信度更高，且有足够的历史支持，使用历史预测
                    if (prediction.get("based_on_history", False) and
                        prediction.get("confidence", 0) > hierarchical_intent.confidence * 1.2):

                        # 更新意图类型
                        from src.assistant.nlp_processor import IntentType
                        try:
                            predicted_intent_type = IntentType[prediction["predicted_intent"]]
                            hierarchical_intent.primary = self._map_to_primary_intent(predicted_intent_type)
                            hierarchical_intent.confidence = prediction["confidence"]

                            logger.info(f"使用历史预测优化意图: {prediction['predicted_intent']} (置信度: {prediction['confidence']:.2f})")
                        except (KeyError, AttributeError):
                            # 如果映射失败，保持原有意图
                            pass

                except Exception as e:
                    logger.debug(f"智能预测失败，使用原有意图: {e}")

            # 3. 转换为兼容的Intent对象
            legacy_intent = hierarchical_intent.to_legacy_intent()
            legacy_intent.raw_text = text

            # 4. 添加学习相关信息到结果
            if self.learning_enabled:
                legacy_intent.parameters["learning_enabled"] = True
                legacy_intent.parameters["user_id"] = self.user_id

            logger.info(f"增强NLP处理完成: {hierarchical_intent.primary.value} -> {hierarchical_intent.secondary.value} (置信度: {hierarchical_intent.confidence:.2f})")
            return legacy_intent

        except Exception as e:
            logger.error(f"增强NLP处理失败: {e}")
            # 返回未知意图
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                action="unknown",
                parameters={},
                raw_text=text,
                matched_keywords=[]
            )

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        解析命令（兼容性方法）

        Args:
            text: 用户输入的命令文本

        Returns:
            解析结果字典
        """
        try:
            # 使用增强的意图识别
            hierarchical_intent = self.intent_recognizer.recognize_intent(text)

            # 转换为兼容格式
            legacy_intent = hierarchical_intent.to_legacy_intent()
            legacy_intent.raw_text = text

            from src.assistant.nlp_processor import CompatibleCommand, CommandType

            compatible_command = CompatibleCommand(
                intent_type=legacy_intent.intent_type,
                parameters=legacy_intent.parameters,
                command_type=CommandType.SINGLE,
                original_text=text,
                confidence=hierarchical_intent.confidence
            )

            return {
                "success": True,
                "commands": [compatible_command],
                "hierarchical_intent": hierarchical_intent,
                "confidence": hierarchical_intent.confidence
            }

        except Exception as e:
            logger.error(f"命令解析失败: {e}")
            import traceback
            logger.error(f"完整错误信息: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "commands": []
            }

    def get_personalized_suggestions(self, current_command: str,
                                   context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        获取个性化建议

        Args:
            current_command: 当前命令
            context: 上下文信息

        Returns:
            个性化建议列表
        """
        if not self.learning_enabled or not self.learner:
            return []

        try:
            return self.learner.get_personalized_suggestions(current_command, context)
        except Exception as e:
            logger.error(f"获取个性化建议失败: {e}")
            return []

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
        if not self.learning_enabled or not self.learner:
            return []

        try:
            return self.learner.get_entity_recommendations(query, entity_type, limit)
        except Exception as e:
            logger.error(f"获取实体推荐失败: {e}")
            return []

    def learn_from_execution(self, original_command: str, intent_type: str, intent_confidence: float,
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
        if not self.learning_enabled or not self.learner:
            return False

        try:
            return self.learner.learn_from_command(
                original_command, intent_type, intent_confidence,
                parameters, tool_name, execution_time, success,
                error_message, context_data, session_id
            )
        except Exception as e:
            logger.error(f"学习执行失败: {e}")
            return False

    def get_user_insights(self, days: int = 30) -> Dict[str, Any]:
        """
        获取用户行为洞察

        Args:
            days: 分析天数

        Returns:
            用户行为洞察
        """
        if not self.learning_enabled or not self.learner:
            return {}

        try:
            return self.learner.get_user_insights(days)
        except Exception as e:
            logger.error(f"获取用户洞察失败: {e}")
            return {}

    def _map_to_primary_intent(self, intent_type):
        """将IntentType映射到PrimaryIntent"""
        from src.assistant.enhanced_nlp_processor import PrimaryIntent

        # 简化的映射关系
        intent_mapping = {
            "APP_LAUNCH": PrimaryIntent.NAVIGATION,
            "WEB_SCRAPING": PrimaryIntent.NAVIGATION,
            "WEB_SEARCH": PrimaryIntent.SEARCH,
            "SYSTEM_INFO": PrimaryIntent.INFORMATION,
            "SCREEN_CAPTURE": PrimaryIntent.OPERATION,
            "OCR": PrimaryIntent.OPERATION,
            "IMAGE_ANALYSIS": PrimaryIntent.OPERATION,
            "FILE_FORMAT_CONVERT": PrimaryIntent.OPERATION,
            "FILE_TEXT_PROCESS": PrimaryIntent.OPERATION,
            "UNKNOWN": PrimaryIntent.NAVIGATION
        }

        return intent_mapping.get(intent_type.name, PrimaryIntent.NAVIGATION)