"""
MCP Floating Ball - NLP处理器

实现自然语言处理，包括意图识别、关键词匹配、参数提取等功能。
"""

import re
import time
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher
from dataclasses import dataclass
from enum import Enum

from src.core.logging import get_logger
from src.core.exceptions import AssistantError


class IntentType(Enum):
    """意图类型枚举"""
    SCREEN_CAPTURE = "screen_capture"
    OCR = "ocr"
    IMAGE_ANALYSIS = "image_analysis"
    APP_LAUNCH = "app_launch"
    SYSTEM_INFO = "system_info"
    FILE_OPERATION = "file_operation"
    WEB_SEARCH = "web_search"
    WEB_SCRAPING = "web_scraping"
    WINDOW_CONTROL = "window_control"
    VOICE_WAKE = "voice_wake"
    UNKNOWN = "unknown"


class CommandType(Enum):
    """命令类型枚举"""
    SINGLE = "single"        # 单个命令
    SEQUENCE = "sequence"    # 串行多个命令
    PARALLEL = "parallel"    # 并行多个命令
    COMPLEX = "complex"      # 复杂命令


@dataclass
class Intent:
    """意图识别结果"""
    intent_type: IntentType
    confidence: float
    action: str
    parameters: Dict[str, Any]
    raw_text: str
    matched_keywords: List[str]


@dataclass
class Command:
    """解析后的命令"""
    intent: Intent
    tools_required: List[str]
    execution_order: List[str]  # 工具执行顺序
    context_info: Dict[str, Any]


# 兼容性Command类，用于测试和其他模块
@dataclass
class CompatibleCommand:
    """兼容性命令类，用于向后兼容"""
    intent_type: IntentType
    parameters: Dict[str, Any]
    command_type: CommandType
    original_text: str
    confidence: float = 0.0


class NLPProcessor:
    """NLP处理器"""

    def __init__(self):
        """初始化NLP处理器"""
        self.logger = get_logger("assistant.nlp")

        # 初始化关键词库
        self._initialize_keyword_database()

        # 同义词映射
        self._initialize_synonyms()

        # 配置参数
        self.fuzzy_threshold = 0.7  # 模糊匹配阈值
        self.min_keyword_length = 2  # 最小关键词长度

        self.logger.info("NLP处理器初始化完成")

    def _initialize_keyword_database(self):
        """初始化关键词数据库"""
        self.keyword_database = {
            # 截图相关
            IntentType.SCREEN_CAPTURE: [
                "截图", "截屏", "屏幕截图", "全屏截图", "抓屏", "screen capture",
                "screenshot", "capture", "capture screen", "take screenshot",
                "截取", "屏幕抓取"
            ],

            # OCR相关
            IntentType.OCR: [
                "识别文字", "文字识别", "OCR", "图片文字", "图像文字", "文字提取",
                "extract text", "recognize text", "文字识别", "图片文字识别",
                "看图识字", "读图识文", "提取文字", "文字内容", "文本提取"
            ],

            # 图像分析相关
            IntentType.IMAGE_ANALYSIS: [
                "分析图像", "图像分析", "图片分析", "分析图片", "图像检测",
                "图片检测", "image analysis", "analyze image", "图片内容",
                "图像内容", "图像信息", "分析这张图", "查看图片"
            ],

            # 应用启动相关
            IntentType.APP_LAUNCH: [
                "启动", "打开", "运行", "执行", "launch", "open", "run", "start",
                "start app", "run program", "open application", "启动应用",
                "打开应用", "运行程序", "开应用"
            ],

            # 系统信息相关
            IntentType.SYSTEM_INFO: [
                "系统信息", "系统状态", "查看系统", "系统详情", "系统配置",
                "system info", "system status", "check system", "view system",
                "电脑信息", "系统规格", "硬件信息", "软件信息",
                "CPU信息", "内存信息", "磁盘信息", "性能信息"
            ],

            # 文件操作相关
            IntentType.FILE_OPERATION: [
                "读取文件", "写入文件", "编辑文件", "创建文件", "删除文件",
                "read file", "write file", "edit file", "create file", "delete file",
                "文件操作", "文件处理", "文本编辑", "文档处理",
                "打开文件", "保存文件", "复制文件", "移动文件"
            ],

            # 网络搜索相关
            IntentType.WEB_SEARCH: [
                "搜索", "查找", "搜索内容", "网络搜索", "在线搜索",
                "search", "find", "lookup", "web search", "online search",
                "搜索一下", "查一下", "找一下", "百度", "谷歌", "搜一下"
            ],

            # 网页抓取相关
            IntentType.WEB_SCRAPING: [
                "抓取网页", "网页抓取", "爬虫", "抓取内容", "网页内容",
                "scrape web", "web scraping", "crawl web", "extract content",
                "下载网页", "保存网页", "获取网页", "网页数据",
                "抓取网站", "网站抓取", "爬取网站"
            ],

            # 窗口控制相关
            IntentType.WINDOW_CONTROL: [
                "窗口", "激活窗口", "最小化", "最大化", "关闭窗口",
                "window", "activate window", "minimize", "maximize", "close window",
                "窗口管理", "窗口操作", "控制窗口", "窗口列表",
                "查找窗口", "移动窗口", "调整窗口"
            ]
        }

    def _initialize_synonyms(self):
        """初始化同义词映射"""
        self.synonyms = {
            # 截图同义词
            "截图": ["截屏", "抓屏", "screen", "screenshot", "capture"],
            "识别": ["提取", "获取", "检测", "recognize", "extract", "detect"],
            "启动": ["打开", "运行", "执行", "launch", "open", "run", "start"],
            "搜索": ["查找", "寻找", "找", "search", "find", "lookup"],
            "分析": ["检查", "检测", "查看", "analyze", "check", "examine"],
            "窗口": ["视窗", "对话窗口", "对话框", "window", "dialog"],
            "应用": ["程序", "软件", "app", "application", "software", "program"],
            "文件": ["文档", "档案", "档案", "file", "document", "archive"],
            "文字": ["文本", "内容", "字符", "text", "content", "character"],
        }

    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Intent:
        """
        处理用户输入文本，识别意图和提取参数

        Args:
            text: 用户输入文本
            context: 上下文信息（对话历史等）

        Returns:
            意图识别结果
        """
        try:
            self.logger.info(f"处理文本: {text[:50]}...")

            # 文本预处理
            processed_text = self._preprocess_text(text)

            # 意图识别 - 先用快速匹配，再用模糊匹配
            intent = self._detect_intent(processed_text)

            # 参数提取
            parameters = self._extract_parameters(processed_text, intent.intent_type)

            # 构建意图结果
            result = Intent(
                intent_type=intent.intent_type,
                confidence=intent.confidence,
                action=intent.action,
                parameters=parameters,
                raw_text=text,
                matched_keywords=intent.matched_keywords
            )

            self.logger.info(f"意图识别完成: {intent.intent_type.value}, 置信度: {intent.confidence:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"文本处理失败: {e}")
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                action="unknown",
                parameters={},
                raw_text=text,
                matched_keywords=[]
            )

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 转换为小写
        text = text.lower().strip()

        # 移除多余的空格和标点
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

        return text

    def _detect_intent(self, text: str) -> Intent:
        """
        意图检测 - 结合关键词匹配和模糊匹配
        """
        best_match = None
        best_confidence = 0.0

        # 遍历所有意图类型
        for intent_type, keywords in self.keyword_database.items():
            confidence, matched_keywords = self._calculate_match_confidence(text, keywords)

            if confidence > best_confidence:
                best_match = IntentType(intent_type)
                best_confidence = confidence

        # 如果匹配度太低，返回未知
        if best_confidence < 0.3:
            best_match = IntentType.UNKNOWN
            best_confidence = 0.0

        # 确定具体动作
        action = self._determine_action(best_match, text)

        return Intent(
            intent_type=best_match,
            confidence=best_confidence,
            action=action,
            parameters={},
            raw_text=text,
            matched_keywords=[]
        )

    def _calculate_match_confidence(self, text: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """计算匹配置信度"""
        matched_keywords = []
        max_confidence = 0.0

        for keyword in keywords:
            # 直接匹配
            if keyword in text:
                matched_keywords.append(keyword)
                max_confidence = max(max_confidence, 1.0)
                continue

            # 模糊匹配
            similarity = self._fuzzy_match(text, keyword)
            if similarity >= self.fuzzy_threshold:
                matched_keywords.append(keyword)
                max_confidence = max(max_confidence, similarity)

        # 使用最高匹配置信度，而不是平均值
        confidence = max_confidence

        return confidence, matched_keywords

    def _fuzzy_match(self, text: str, keyword: str) -> float:
        """模糊匹配"""
        try:
            # 使用SequenceMatcher进行模糊匹配
            matcher = SequenceMatcher(None, text, keyword)
            return matcher.ratio()
        except Exception as e:
            self.logger.error(f"模糊匹配失败: {e}")
            return 0.0

    def _determine_action(self, intent_type: IntentType, text: str) -> str:
        """根据意图类型和文本确定具体动作"""
        action_map = {
            IntentType.SCREEN_CAPTURE: self._determine_capture_action,
            IntentType.OCR: self._determine_ocr_action,
            IntentType.IMAGE_ANALYSIS: self._determine_analysis_action,
            IntentType.APP_LAUNCH: self._determine_launch_action,
            IntentType.SYSTEM_INFO: self._determine_info_action,
            IntentType.FILE_OPERATION: self._determine_file_action,
            IntentType.WEB_SEARCH: self._determine_search_action,
            IntentType.WEB_SCRAPING: self._determine_scraping_action,
            IntentType.WINDOW_CONTROL: self._determine_window_action,
            IntentType.VOICE_WAKE: self._determine_wake_action
        }

        action_func = action_map.get(intent_type)
        if action_func:
            return action_func(text)

        return intent_type.value

    def _determine_capture_action(self, text: str) -> str:
        """确定截图动作"""
        if "全屏" in text or "整个" in text:
            return "capture_full_screen"
        elif "区域" in text or "部分" in text:
            return "capture_region"
        elif "窗口" in text:
            return "capture_window"
        elif "光标" in text:
            return "capture_with_cursor"
        else:
            return "capture_full_screen"  # 默认

    def _determine_ocr_action(self, text: str) -> str:
        """确定OCR动作"""
        return "recognize_text"  # OCR只有一种动作

    def _determine_analysis_action(self, text: str) -> str:
        """确定分析动作"""
        if "颜色" in text:
            return "analyze_colors"
        elif "边缘" in text:
            return "analyze_edges"
        elif "基础" in text:
            return "analyze_basic"
        else:
            return "analyze_all"  # 默认全部分析

    def _determine_launch_action(self, text: str) -> str:
        """确定启动动作"""
        return "launch_app"  # 启动只有一种动作

    def _determine_info_action(self, text: str) -> str:
        """确定信息获取动作"""
        if "cpu" in text:
            return "get_cpu_info"
        elif "内存" in text:
            return "get_memory_info"
        elif "磁盘" in text:
            return "get_disk_info"
        elif "网络" in text:
            return "get_network_info"
        else:
            return "get_system_info"  # 默认系统信息

    def _determine_file_action(self, text: str) -> str:
        """确定文件操作动作"""
        if "读取" in text or "查看" in text:
            return "read_file"
        elif "写入" in text or "保存" in text:
            return "write_file"
        elif "编辑" in text:
            return "edit_file"
        elif "删除" in text or "移除" in text:
            return "delete_file"
        elif "创建" in text or "新建" in text:
            return "create_file"
        else:
            return "read_file"  # 默认读取

    def _determine_search_action(self, text: str) -> str:
        """确定搜索动作"""
        return "search_web"  # 搜索只有一种动作

    def _determine_scraping_action(self, text: str) -> str:
        """确定抓取动作"""
        return "scrape_content"  # 抓取只有一种动作

    def _determine_window_action(self, text: str) -> str:
        """确定窗口操作动作"""
        if "列表" in text or "查看" in text:
            return "list_windows"
        elif "激活" in text:
            return "activate_window"
        elif "最小化" in text:
            return "minimize_window"
        elif "最大化" in text:
            return "maximize_window"
        elif "关闭" in text:
            return "close_window"
        elif "移动" in text:
            return "move_window"
        elif "调整" in text or "大小" in text:
            return "resize_window"
        else:
            return "list_windows"  # 默认列表

    def _determine_wake_action(self, text: str) -> str:
        """确定唤醒动作"""
        return "activate"  # 唤醒只有一种动作

    def _extract_parameters(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """从文本中提取参数"""
        try:
            if intent_type == IntentType.SCREEN_CAPTURE:
                return self._extract_capture_parameters(text)
            elif intent_type == IntentType.OCR:
                return self._extract_ocr_parameters(text)
            elif intent_type == IntentType.IMAGE_ANALYSIS:
                return self._extract_analysis_parameters(text)
            elif intent_type == IntentType.APP_LAUNCH:
                return self._extract_launch_parameters(text)
            elif intent_type == IntentType.SYSTEM_INFO:
                return self._extract_info_parameters(text)
            elif intent_type == IntentType.FILE_OPERATION:
                return self._extract_file_parameters(text)
            elif intent_type == IntentType.WEB_SEARCH:
                return self._extract_search_parameters(text)
            elif intent_type == IntentType.WEB_SCRAPING:
                return self._extract_scraping_parameters(text)
            elif intent_type == IntentType.WINDOW_CONTROL:
                return self._extract_window_parameters(text)
            else:
                return {}
        except Exception as e:
            self.logger.error(f"参数提取失败: {e}")
            return {}

    def _extract_capture_parameters(self, text: str) -> Dict[str, Any]:
        """提取截图参数"""
        params = {}

        # 区域截图参数 (x, y, width, height)
        region_pattern = r'(\d+)\s*[,\s]* (\d+)\s*[,\s]* (\d+)\s*[,\s]* (\d+)'
        region_match = re.search(region_pattern, text)
        if region_match:
            try:
                x, y, width, height = map(int, region_match.groups())
                params["region"] = {"x": x, "y": y, "width": width, "height": height}
            except ValueError:
                pass

        # 窗口标题
        window_pattern = r'窗口["\']([^"\'"]*)["\']'
        window_match = re.search(window_pattern, text)
        if window_match:
            params["window_title"] = window_match.group(1)

        return params

    def _extract_ocr_parameters(self, text: str) -> Dict[str, Any]:
        """提取OCR参数"""
        params = {}

        # OCR语言
        if "中文" in text:
            params["lang"] = "chi_sim+eng"
        elif "英文" in text or "english" in text:
            params["lang"] = "eng"

        return params

    def _extract_analysis_parameters(self, text: str) -> Dict[str, Any]:
        """提取分析参数"""
        params = {}

        # 分析类型
        analysis_types = []
        if "颜色" in text:
            analysis_types.append("colors")
        if "边缘" in text:
            analysis_types.append("edges")
        if "基础" in text or "基本信息" in text:
            analysis_types.append("basic")
        if "面部" in text or "人脸" in text:
            analysis_types.append("faces")
        if "手部" in text or "手势" in text:
            analysis_types.append("hands")
        if "姿态" in text or "姿势" in text:
            analysis_types.append("pose")

        if analysis_types:
            params["analysis_types"] = analysis_types

        return params

    def _extract_launch_parameters(self, text: str) -> Dict[str, Any]:
        """提取启动参数"""
        params = {}

        # 应用名称
        app_patterns = [
            r'打开["\']([^"\'"]*)["\']',
            r'启动["\']([^"\'"]*)["\']',
            r'运行["\']([^"\'"]*)["\']'
        ]

        for pattern in app_patterns:
            match = re.search(pattern, text)
            if match:
                params["app_name"] = match.group(1)
                break

        return params

    def _extract_info_parameters(self, text: str) -> Dict[str, Any]:
        """提取信息参数"""
        params = {}

        # 信息类型
        if "cpu" in text or "处理器" in text:
            params["category"] = "cpu"
        elif "内存" in text or "内存条" in text:
            params["category"] = "memory"
        elif "磁盘" in text or "硬盘" in text:
            params["category"] = "disk"
        elif "网络" in text:
            params["category"] = "network"
        elif "详细" in text or "全部" in text:
            params["detailed"] = True

        return params

    def _extract_file_parameters(self, text: str) -> Dict[str, Any]:
        """提取文件参数"""
        params = {}

        # 文件路径
        file_patterns = [
            r'["\']([^"\'"]*)["\']',
            r'文件["\']([^"\'"]*)["\']',
            r'路径["\']([^"\'"]*)["\']'
        ]

        for pattern in file_patterns:
            match = re.search(pattern, text)
            if match:
                params["file_path"] = match.group(1)
                break

        # 操作类型
        if "写入" in text or "保存" in text:
            params["operation"] = "write"
        elif "编辑" in text or "修改" in text:
            params["operation"] = "edit"
        elif "删除" in text or "移除" in text:
            params["operation"] = "delete"
        elif "创建" in text or "新建" in text:
            params["operation"] = "create"
        else:
            params["operation"] = "read"

        return params

    def _extract_search_parameters(self, text: str) -> Dict[str, Any]:
        """提取搜索参数"""
        params = {}

        # 搜索关键词
        search_patterns = [
            r'搜索["\']([^"\'"]*)["\']',
            r'查找["\']([^"\'"]*)["\']',
            r'搜["\']([^"\'"]*)["\']',
            r'找["\']([^"\'"]*)["\']'
        ]

        for pattern in search_patterns:
            match = re.search(pattern, text)
            if match:
                params["query"] = match.group(1)
                break

        # 搜索引擎
        if "百度" in text:
            params["engine"] = "baidu"
        elif "谷歌" in text or "google" in text:
            params["engine"] = "google"
        elif "必应" in text or "bing" in text:
            params["engine"] = "bing"
        else:
            params["engine"] = "google"  # 默认

        return params

    def _extract_scraping_parameters(self, text: str) -> Dict[str, Any]:
        """提取抓取参数"""
        params = {}

        # URL
        url_patterns = [
            r'https?://[^\s]+',
            r'网址["\']([^"\'"]*)["\']',
            r'链接["\']([^"\'"]*)["\']'
        ]

        for pattern in url_patterns:
            match = re.search(pattern, text)
            if match:
                params["url"] = match.group(0)
                break

        return params

    def _extract_window_parameters(self, text: str) -> Dict[str, Any]:
        """提取窗口参数"""
        params = {}

        # 窗口标题
        window_patterns = [
            r'窗口["\']([^"\'"]*)["\']',
            r'标题["\']([^"\'"]*)["\']'
        ]

        for pattern in window_patterns:
            match = re.search(pattern, text)
            if match:
                params["title"] = match.group(1)
                break

        # 窗口操作
        if "最小化" in text:
            params["action"] = "minimize"
        elif "最大化" in text:
            params["action"] = "maximize"
        elif "关闭" in text:
            params["action"] = "close"
        elif "激活" in text:
            params["action"] = "activate"
        elif "移动" in text:
            params["action"] = "move"
        elif "调整" in text or "大小" in text:
            params["action"] = "resize"
        else:
            params["action"] = "list"

        return params

    def get_keyword_database(self) -> Dict[IntentType, List[str]]:
        """获取关键词数据库"""
        return self.keyword_database.copy()

    def add_keywords(self, intent_type: IntentType, keywords: List[str]):
        """添加关键词"""
        if intent_type not in self.keyword_database:
            self.keyword_database[intent_type] = []

        self.keyword_database[intent_type].extend(keywords)
        self.logger.info(f"为意图 {intent_type.value} 添加了 {len(keywords)} 个关键词")

    def add_synonym(self, word: str, synonyms: List[str]):
        """添加同义词"""
        self.synonyms[word] = synonyms
        self.logger.info(f"为词语 '{word}' 添加了 {len(synonyms)} 个同义词")

    def process_conversation(self, text: str, context: Dict[str, Any]) -> Command:
        """处理对话（带上下文）"""
        try:
            # 基础意图识别
            intent = self.process_text(text, context)

            # 识别复合命令（如"截图然后OCR"）
            commands = self._parse_complex_commands(text, intent)

            # 确定需要的工具
            tools_needed = self._determine_required_tools(commands)

            # 确定执行顺序
            execution_order = self._determine_execution_order(commands)

            return Command(
                intent=intent,
                tools_required=tools_needed,
                execution_order=execution_order,
                context_info=context
            )

        except Exception as e:
            self.logger.error(f"对话处理失败: {e}")
            return Command(
                intent=Intent(intent_type=IntentType.UNKNOWN, confidence=0.0, action="unknown", parameters={}, raw_text=text, matched_keywords=[]),
                tools_required=[],
                execution_order=[],
                context_info={}
            )

    def _parse_complex_commands(self, text: str, primary_intent: Intent) -> List[Intent]:
        """解析复合命令"""
        commands = [primary_intent]

        # 检测连接词
        connectors = ["然后", "接着", "之后", "再", "and", "然后", "之后", "再"]

        for connector in connectors:
            parts = text.split(connector)
            if len(parts) > 1:
                # 处理后面的部分
                for part in parts[1:]:
                    part = part.strip()
                    if part:
                        secondary_intent = self._detect_intent(part)
                        commands.append(secondary_intent)
                break

        return commands

    def _determine_required_tools(self, commands: List[Intent]) -> List[str]:
        """确定需要的工具"""
        tools = []

        tool_map = {
            IntentType.SCREEN_CAPTURE: ["screen_capture"],
            IntentType.OCR: ["ocr_engine"],
            IntentType.IMAGE_ANALYSIS: ["image_analyzer"],
            IntentType.APP_LAUNCH: ["application_launcher"],
            IntentType.SYSTEM_INFO: ["system_info"],
            IntentType.FILE_OPERATION: ["text_operations"],
            IntentType.WEB_SEARCH: ["multi_search"],
            IntentType.WEB_SCRAPING: ["web_scraper"],
            IntentType.WINDOW_CONTROL: ["window_manager"]
        }

        for intent in commands:
            tool = tool_map.get(intent.intent_type)
            if tool and tool not in tools:
                tools.extend(tool)

        return tools

    def _determine_execution_order(self, commands: List[Intent]) -> List[str]:
        """确定执行顺序"""
        order = []

        # 根据依赖关系确定顺序
        # 截图 -> OCR -> 分析的逻辑顺序
        intent_order = {
            IntentType.SCREEN_CAPTURE: 1,
            IntentType.OCR: 2,
            IntentType.IMAGE_ANALYSIS: 3,
            IntentType.APP_LAUNCH: 4,
            IntentType.SYSTEM_INFO: 5,
            IntentType.FILE_OPERATION: 6,
            IntentType.WEB_SEARCH: 7,
            IntentType.WEB_SCRAPING: 8,
            IntentType.WINDOW_CONTROL: 9,
            IntentType.VOICE_WAKE: 10
        }

        # 按优先级排序
        sorted_commands = sorted(commands, key=lambda x: intent_order.get(x.intent_type, 999))

        for intent in sorted_commands:
            tool_map = {
                IntentType.SCREEN_CAPTURE: "screen_capture",
                IntentType.OCR: "ocr_engine",
                IntentType.IMAGE_ANALYSIS: "image_analyzer",
                IntentType.APP_LAUNCH: "application_launcher",
                IntentType.SYSTEM_INFO: "system_info",
                IntentType.FILE_OPERATION: "text_operations",
                IntentType.WEB_SEARCH: "multi_search",
                IntentType.WEB_SCRAPING: "web_scraper",
                IntentType.WINDOW_CONTROL: "window_manager",
                IntentType.VOICE_WAKE: "voice_activation"
            }

            tool = tool_map.get(intent.intent_type)
            if tool and tool not in order:
                order.append(tool)

        return order

    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        解析命令 (兼容性方法)

        Args:
            text: 用户输入的命令文本

        Returns:
            解析结果字典
        """
        try:
            # 使用现有的process_conversation方法
            # 提供默认的空上下文
            default_context = {
                "session_id": "default",
                "user_history": [],
                "current_state": "active"
            }
            command = self.process_conversation(text, default_context)

            # 将Command对象转换为兼容性格式
            if hasattr(command, 'intent') and hasattr(command.intent, 'intent_type'):
                compatible_cmd = CompatibleCommand(
                    intent_type=command.intent.intent_type,
                    parameters=command.intent.parameters,
                    command_type=CommandType.SINGLE,  # 默认为单命令
                    original_text=text,
                    confidence=command.intent.confidence
                )

                return {
                    "success": True,
                    "commands": [compatible_cmd],
                    "original_command": command
                }
            else:
                # 未知意图处理
                compatible_cmd = CompatibleCommand(
                    intent_type=IntentType.UNKNOWN,
                    parameters={},
                    command_type=CommandType.SINGLE,
                    original_text=text,
                    confidence=0.0
                )

                return {
                    "success": True,
                    "commands": [compatible_cmd],
                    "original_command": command
                }

        except Exception as e:
            self.logger.error(f"解析命令失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "commands": []
            }