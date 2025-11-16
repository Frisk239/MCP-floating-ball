"""
MCP Floating Ball - 工具调用器

负责工具的调用、执行和管理，连接NLP处理器和实际工具执行。
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import traceback
from pathlib import Path

from src.core.logging import get_logger
from src.core.exceptions import AssistantError
from src.tools.base import ToolParameter, ParameterType
from src.tools.system.application_launcher import ApplicationLauncherTool
from src.tools.system.window_manager import WindowManagerTool
from src.tools.system.system_info import SystemInfoTool
from src.tools.file.format_converter import FormatConverterTool
from src.tools.file.text_operations import TextOperationsTool
from src.tools.network.multi_search import MultiSearchTool
from src.tools.network.web_scraper import WebScraperTool
from src.assistant.nlp_processor import IntentType, Command, CommandType, CompatibleCommand
from src.core.config_manager import get_config_manager

logger = get_logger("assistant.tool_caller")


class ToolCaller:
    """工具调用器类"""

    def __init__(self):
        """初始化工具调用器"""
        self.logger = get_logger("assistant.tool_caller")
        self.config_manager = get_config_manager()

        # 工具注册表
        self.tools: Dict[str, Any] = {}
        self.tool_categories: Dict[str, List[str]] = {}

        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []

        # 初始化工具
        self._initialize_tools()

        self.logger.info("工具调用器初始化完成")

    def _initialize_tools(self):
        """初始化所有工具"""
        try:
            # 系统控制工具
            self.tools["application_launcher"] = ApplicationLauncherTool()
            self.tools["window_manager"] = WindowManagerTool()
            self.tools["system_info"] = SystemInfoTool()

            # 文件处理工具
            self.tools["format_converter"] = FormatConverterTool()
            self.tools["text_operations"] = TextOperationsTool()

            # 网络工具
            self.tools["multi_search"] = MultiSearchTool()
            self.tools["web_scraper"] = WebScraperTool()

            # 分类管理
            self.tool_categories["system"] = ["application_launcher", "window_manager", "system_info"]
            self.tool_categories["file"] = ["format_converter", "text_operations"]
            self.tool_categories["network"] = ["multi_search", "web_scraper"]

            self.logger.info(f"成功初始化 {len(self.tools)} 个工具")

        except Exception as e:
            self.logger.error(f"工具初始化失败: {e}")
            raise AssistantError(f"工具初始化失败: {e}")

    async def execute_command(self, command: Union[Command, CompatibleCommand]) -> Dict[str, Any]:
        """
        执行单个命令

        Args:
            command: 要执行的命令

        Returns:
            执行结果
        """
        start_time = datetime.now()

        try:
            # 兼容性处理：支持Command和CompatibleCommand
            if hasattr(command, 'intent'):
                # 原始Command对象
                intent_type = command.intent.intent_type
                parameters = command.intent.parameters
            else:
                # CompatibleCommand对象
                intent_type = command.intent_type
                parameters = command.parameters

            self.logger.info(f"开始执行命令: {intent_type.name} - {parameters}")

            # 验证参数
            if not parameters:
                raise AssistantError(f"命令缺少必要参数: {intent_type.name}")

            # 根据意图类型执行相应的工具
            result = await self._execute_by_intent(intent_type, parameters)

            execution_time = (datetime.now() - start_time).total_seconds()

            # 记录执行历史
            history_entry = {
                "timestamp": start_time.isoformat(),
                "intent_type": intent_type.name,
                "tool_name": result.get("tool_name", "unknown"),
                "parameters": parameters,
                "success": result.get("success", False),
                "execution_time": execution_time,
                "result_summary": result.get("summary", "")
            }

            self.execution_history.append(history_entry)

            self.logger.info(f"命令执行完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }

            # 记录失败历史
            history_entry = {
                "timestamp": start_time.isoformat(),
                "intent_type": command.intent_type.name,
                "tool_name": "unknown",
                "parameters": command.parameters,
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }

            self.execution_history.append(history_entry)

            self.logger.error(f"命令执行失败: {e}")
            return error_result

    async def execute_commands(self, commands: List[Command]) -> List[Dict[str, Any]]:
        """
        执行多个命令（串行或并行）

        Args:
            commands: 要执行的命令列表

        Returns:
            执行结果列表
        """
        if not commands:
            return []

        results = []

        try:
            # 检查是否可以并行执行
            if len(commands) == 1:
                # 单个命令直接执行
                result = await self.execute_command(commands[0])
                results.append(result)
            else:
                # 多个命令检查依赖关系
                can_parallel = self._can_execute_parallel(commands)

                if can_parallel:
                    self.logger.info(f"并行执行 {len(commands)} 个命令")
                    # 并行执行
                    tasks = [self.execute_command(cmd) for cmd in commands]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    self.logger.info(f"串行执行 {len(commands)} 个命令")
                    # 串行执行
                    for cmd in commands:
                        result = await self.execute_command(cmd)
                        results.append(result)

                        # 如果有严重错误，停止执行
                        if not result.get("success", True) and result.get("error_type") == "critical":
                            self.logger.warning("遇到严重错误，停止执行后续命令")
                            break

        except Exception as e:
            self.logger.error(f"批量命令执行失败: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            results.append(error_result)

        return results

    async def _execute_by_intent(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图类型执行相应的工具"""
        try:
            if intent_type == IntentType.APPLICATION_LAUNCH:
                return await self._execute_application_launcher(intent_type, parameters)
            elif intent_type == IntentType.WINDOW_MANAGE:
                return await self._execute_window_manager(intent_type, parameters)
            elif intent_type == IntentType.SYSTEM_INFO:
                return await self._execute_system_info(intent_type, parameters)
            elif intent_type == IntentType.FILE_FORMAT_CONVERT:
                return await self._execute_format_converter(intent_type, parameters)
            elif intent_type == IntentType.FILE_TEXT_PROCESS:
                return await self._execute_text_operations(intent_type, parameters)
            elif intent_type == IntentType.WEB_SEARCH:
                return await self._execute_multi_search(intent_type, parameters)
            elif intent_type == IntentType.WEB_SCRAPING:
                return await self._execute_web_scraper(intent_type, parameters)
            elif intent_type == IntentType.SCREEN_CAPTURE:
                return await self._execute_screen_capture(intent_type, parameters)
            elif intent_type == IntentType.OCR:
                return await self._execute_ocr(intent_type, parameters)
            elif intent_type == IntentType.IMAGE_ANALYSIS:
                return await self._execute_image_analysis(intent_type, parameters)
            else:
                raise AssistantError(f"不支持的意图类型: {intent_type.name}")

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "intent_type": intent_type.name,
                "timestamp": datetime.now().isoformat()
            }

    async def _execute_application_launcher(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行应用启动器工具"""
        tool = self.tools["application_launcher"]

        # 提取应用名称 - 参数映射
        app_name = parameters.get("app_name") or parameters.get("application")
        if not app_name:
            raise AssistantError("缺少应用名称参数")

        # 执行工具
        result = await tool.launch_app(app_name)

        return {
            "success": result["success"],
            "tool_name": "application_launcher",
            "result": result,
            "summary": f"启动应用 {app_name}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_window_manager(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行窗口管理工具"""
        tool = self.tools["window_manager"]

        # 提取操作类型
        action = parameters.get("action")
        window_title = parameters.get("window_title")

        if not action:
            raise AssistantError("缺少窗口操作类型")

        # 执行相应的窗口操作
        if action == "list":
            result = await tool.list_windows()
        elif action == "activate":
            result = await tool.activate_window(window_title)
        elif action == "minimize":
            result = await tool.minimize_window(window_title)
        elif action == "maximize":
            result = await tool.maximize_window(window_title)
        elif action == "close":
            result = await tool.close_window(window_title)
        else:
            raise AssistantError(f"不支持的窗口操作: {action}")

        return {
            "success": result["success"],
            "tool_name": "window_manager",
            "result": result,
            "summary": f"窗口操作 {action}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_system_info(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行系统信息工具"""
        tool = self.tools["system_info"]

        # 提取信息类型
        info_type = parameters.get("info_type", "basic")

        # 执行相应的信息查询
        if info_type == "basic":
            result = await tool.get_basic_info()
        elif info_type == "hardware":
            result = await tool.get_hardware_info()
        elif info_type == "network":
            result = await tool.get_network_info()
        elif info_type == "processes":
            result = await tool.get_process_list()
        else:
            raise AssistantError(f"不支持的系统信息类型: {info_type}")

        return {
            "success": result["success"],
            "tool_name": "system_info",
            "result": result,
            "summary": f"获取系统信息 {info_type}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_format_converter(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行格式转换工具"""
        tool = self.tools["format_converter"]

        # 提取参数
        input_file = parameters.get("input_file")
        output_file = parameters.get("output_file")

        if not input_file or not output_file:
            raise AssistantError("缺少输入或输出文件路径")

        # 执行转换
        result = await tool.convert_file(input_file, output_file)

        return {
            "success": result["success"],
            "tool_name": "format_converter",
            "result": result,
            "summary": f"文件转换 {input_file} -> {output_file}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_text_operations(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行文本操作工具"""
        tool = self.tools["text_operations"]

        # 提取操作类型
        operation = parameters.get("operation")

        if not operation:
            raise AssistantError("缺少文本操作类型")

        # 执行相应的文本操作
        if operation == "read":
            file_path = parameters.get("file_path")
            result = await tool.read_file(file_path)
        elif operation == "write":
            file_path = parameters.get("file_path")
            content = parameters.get("content")
            result = await tool.write_file(file_path, content)
        elif operation == "search":
            file_path = parameters.get("file_path")
            pattern = parameters.get("pattern")
            result = await tool.search_in_file(file_path, pattern)
        elif operation == "replace":
            file_path = parameters.get("file_path")
            old_text = parameters.get("old_text")
            new_text = parameters.get("new_text")
            result = await tool.replace_in_file(file_path, old_text, new_text)
        else:
            raise AssistantError(f"不支持的文本操作: {operation}")

        return {
            "success": result["success"],
            "tool_name": "text_operations",
            "result": result,
            "summary": f"文本操作 {operation}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_multi_search(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行多搜索引擎工具"""
        tool = self.tools["multi_search"]

        # 提取参数
        query = parameters.get("query")
        engine = parameters.get("engine", "all")

        if not query:
            raise AssistantError("缺少搜索查询")

        # 执行搜索
        result = await tool.search(query, engine)

        return {
            "success": result["success"],
            "tool_name": "multi_search",
            "result": result,
            "summary": f"搜索 '{query}': {'成功' if result['success'] else '失败'}"
        }

    async def _execute_web_scraper(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行网页抓取工具"""
        tool = self.tools["web_scraper"]

        # 提取操作类型
        action = parameters.get("action")
        url = parameters.get("url")

        if not action or not url:
            raise AssistantError("缺少操作类型或URL")

        # 执行相应的网页操作
        if action == "scrape":
            result = await tool.scrape_content(url)
        elif action == "analyze_links":
            result = await tool.analyze_links(url)
        elif action == "download_images":
            result = await tool.download_images(url)
        else:
            raise AssistantError(f"不支持的网页操作: {action}")

        return {
            "success": result["success"],
            "tool_name": "web_scraper",
            "result": result,
            "summary": f"网页操作 {action} {url}: {'成功' if result['success'] else '失败'}"
        }

    async def _execute_screen_capture(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行屏幕截图功能"""
        from src.vision.screen_capture import ScreenCapture

        try:
            screen_capture = ScreenCapture()

            # 提取参数
            capture_type = parameters.get("capture_type", "full")
            save_path = parameters.get("save_path")

            if capture_type == "full":
                result = screen_capture.capture_full_screen(save_path=save_path)
            elif capture_type == "region":
                x = parameters.get("x")
                y = parameters.get("y")
                width = parameters.get("width")
                height = parameters.get("height")
                result = screen_capture.capture_region(
                    x, y, width, height, save_path=save_path
                )
            elif capture_type == "window":
                window_title = parameters.get("window_title")
                result = screen_capture.capture_window(
                    window_title=window_title, save_path=save_path
                )
            else:
                raise AssistantError(f"不支持的截图类型: {capture_type}")

            return {
                "success": result["success"],
                "tool_name": "screen_capture",
                "result": result,
                "summary": f"屏幕截图 {capture_type}: {'成功' if result['success'] else '失败'}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": "screen_capture",
                "timestamp": datetime.now().isoformat()
            }

    async def _execute_ocr(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行OCR功能"""
        from src.vision.ocr_engine import OCREngine

        try:
            # 提取参数
            image_path = parameters.get("image_path")
            engine = parameters.get("engine", "tesseract")

            if not image_path:
                raise AssistantError("缺少图片路径")

            ocr_engine = OCREngine(engine=engine)
            result = ocr_engine.recognize_text(image_path)

            return {
                "success": result["success"],
                "tool_name": "ocr",
                "result": result,
                "summary": f"OCR识别 {image_path}: {'成功' if result['success'] else '失败'}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": "ocr",
                "timestamp": datetime.now().isoformat()
            }

    async def _execute_image_analysis(self, intent_type: IntentType, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行图像分析功能"""
        from src.vision.image_analyzer import ImageAnalyzer

        try:
            # 提取参数
            image_path = parameters.get("image_path")
            analysis_types = parameters.get("analysis_types", ["basic", "colors"])

            if not image_path:
                raise AssistantError("缺少图片路径")

            analyzer = ImageAnalyzer()
            result = analyzer.analyze_image(image_path, analysis_types)

            return {
                "success": result["success"],
                "tool_name": "image_analyzer",
                "result": result,
                "summary": f"图像分析 {image_path}: {'成功' if result['success'] else '失败'}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": "image_analyzer",
                "timestamp": datetime.now().isoformat()
            }

    def _can_execute_parallel(self, commands: List[Command]) -> bool:
        """判断命令是否可以并行执行"""
        # 简单的并行判断逻辑
        # 不同类别的工具通常可以并行执行

        categories = set()
        for cmd in commands:
            category = self._get_tool_category_by_intent(cmd.intent_type)
            categories.add(category)

            # 如果有多个命令属于同一类别，则串行执行
            if len(categories) < len(commands):
                return False

        return True

    def _get_tool_category_by_intent(self, intent_type: IntentType) -> str:
        """根据意图类型获取工具类别"""
        if intent_type in [IntentType.APPLICATION_LAUNCH, IntentType.WINDOW_MANAGE, IntentType.SYSTEM_INFO]:
            return "system"
        elif intent_type in [IntentType.FILE_FORMAT_CONVERT, IntentType.FILE_TEXT_PROCESS]:
            return "file"
        elif intent_type in [IntentType.WEB_SEARCH, IntentType.WEB_SCRAPING]:
            return "network"
        elif intent_type in [IntentType.SCREEN_CAPTURE, IntentType.OCR, IntentType.IMAGE_ANALYSIS]:
            return "vision"
        else:
            return "unknown"

    def get_tool_status(self) -> Dict[str, Any]:
        """获取工具状态"""
        status = {
            "total_tools": len(self.tools),
            "categories": {},
            "execution_history_count": len(self.execution_history)
        }

        for category, tools in self.tool_categories.items():
            status["categories"][category] = {
                "tools": tools,
                "count": len(tools),
                "available": len([t for t in tools if t in self.tools])
            }

        return status

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取执行历史"""
        if limit:
            return self.execution_history[-limit:]
        return self.execution_history.copy()

    def clear_execution_history(self):
        """清空执行历史"""
        self.execution_history.clear()
        self.logger.info("执行历史已清空")

    async def test_tool(self, tool_name: str) -> Dict[str, Any]:
        """测试指定工具"""
        try:
            if tool_name not in self.tools:
                raise AssistantError(f"工具 '{tool_name}' 不存在")

            tool = self.tools[tool_name]

            # 获取工具元数据
            metadata = tool.get_metadata()

            # 构建测试命令
            test_command = self._build_test_command(tool_name)

            if test_command:
                # 执行测试
                result = await self.execute_command(test_command)
                return {
                    "tool_name": tool_name,
                    "metadata": metadata,
                    "test_result": result,
                    "test_passed": result.get("success", False)
                }
            else:
                return {
                    "tool_name": tool_name,
                    "metadata": metadata,
                    "test_skipped": True,
                    "reason": "没有可用的测试参数"
                }

        except Exception as e:
            return {
                "tool_name": tool_name,
                "test_failed": True,
                "error": str(e)
            }

    def _build_test_command(self, tool_name: str) -> Optional[Command]:
        """为工具构建测试命令"""
        test_params = {
            "application_launcher": Command(
                intent_type=IntentType.APPLICATION_LAUNCH,
                parameters={"app_name": "notepad"},  # 测试记事本
                command_type=CommandType.SINGLE,
                original_text="测试启动记事本"
            ),
            "system_info": Command(
                intent_type=IntentType.SYSTEM_INFO,
                parameters={"info_type": "basic"},
                command_type=CommandType.SINGLE,
                original_text="测试获取系统基本信息"
            )
        }

        return test_params.get(tool_name)

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        直接执行指定工具

        Args:
            tool_name: 工具名称
            parameters: 执行参数

        Returns:
            执行结果
        """
        try:
            # 创建兼容的Command对象
            from src.assistant.nlp_processor import Command, CommandType
            command = Command(
                intent_type=self._get_intent_type_for_tool(tool_name),
                parameters=parameters,
                command_type=CommandType.SINGLE,
                original_text=f"执行工具: {tool_name}"
            )

            # 执行命令
            result = await self.execute_command(command)
            return result

        except Exception as e:
            self.logger.error(f"执行工具 {tool_name} 失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }

    def _get_intent_type_for_tool(self, tool_name: str) -> 'IntentType':
        """根据工具名称获取对应的意图类型"""
        from src.assistant.nlp_processor import IntentType

        tool_intent_mapping = {
            "application_launcher": IntentType.APP_LAUNCH,
            "window_manager": IntentType.SYSTEM_INFO,
            "system_info": IntentType.SYSTEM_INFO,
            "format_converter": IntentType.FILE_FORMAT_CONVERT,
            "text_operations": IntentType.FILE_TEXT_PROCESS,
            "multi_search": IntentType.WEB_SEARCH,
            "web_scraper": IntentType.WEB_SCRAPING,
            "screen_capture": IntentType.SCREEN_CAPTURE,
            "ocr_engine": IntentType.OCR,
            "image_analyzer": IntentType.IMAGE_ANALYSIS
        }

        return tool_intent_mapping.get(tool_name, IntentType.UNKNOWN)

    def cleanup(self):
        """清理资源"""
        try:
            # 清理工具资源
            for tool_name, tool in self.tools.items():
                if hasattr(tool, 'cleanup'):
                    try:
                        tool.cleanup()
                    except Exception as e:
                        self.logger.warning(f"清理工具 {tool_name} 失败: {e}")

            self.logger.info("工具调用器资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")