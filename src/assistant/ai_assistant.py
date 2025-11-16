"""
MCP Floating Ball - AI助手主类

统一的AI助手核心类，整合所有功能模块。
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import threading
import time
from pathlib import Path

from src.core.logging import get_logger
from src.core.exceptions import AssistantError
from src.assistant.command_handler import CommandHandler
from src.voice.voice_activation import VoiceActivation
from src.vision.vision_integration import VisionIntegration
from src.core.config_manager import get_config_manager

logger = get_logger("assistant.ai_assistant")


class AIAssistant:
    """AI助手主类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化AI助手

        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger("assistant.ai_assistant")

        # 配置管理
        self.config_manager = get_config_manager()
        if config_path:
            self.config_manager.load_config(config_path)

        # 核心组件
        self.command_handler: Optional[CommandHandler] = None
        self.voice_activation: Optional[VoiceActivation] = None
        self.vision_integration: Optional[VisionIntegration] = None

        # 运行状态
        self.is_running = False
        self.is_voice_active = False

        # 线程管理
        self.voice_thread: Optional[threading.Thread] = None
        self.background_tasks: List[asyncio.Task] = []

        # 回调函数
        self.response_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None

        # 会话管理
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()

        # 初始化组件
        self._initialize_components()

        self.logger.info(f"AI助手初始化完成，会话ID: {self.session_id}")

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化命令处理器
            self.command_handler = CommandHandler()

            # 初始化语音功能（如果启用）
            if self.config_manager.get("voice.enabled", False):
                self._initialize_voice()

            # 初始化视觉功能（如果启用）
            if self.config_manager.get("vision.enabled", True):
                self._initialize_vision()

            self.logger.info("所有组件初始化完成")

        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise AssistantError(f"组件初始化失败: {e}")

    def _initialize_voice(self):
        """初始化语音功能"""
        try:
            self.voice_activation = VoiceActivation()

            # 设置语音回调
            self.voice_activation.set_command_callback(self._on_voice_command)
            self.voice_activation.set_status_callback(self._on_voice_status)

            self.logger.info("语音功能初始化成功")

        except Exception as e:
            self.logger.warning(f"语音功能初始化失败: {e}")
            self.voice_activation = None

    def _initialize_vision(self):
        """初始化视觉功能"""
        try:
            self.vision_integration = VisionIntegration()

            # 设置视觉回调
            self.vision_integration.add_screenshot_callback(self._on_screenshot)
            self.vision_integration.add_ocr_callback(self._on_ocr_result)
            self.vision_integration.add_analysis_callback(self._on_analysis_result)

            self.logger.info("视觉功能初始化成功")

        except Exception as e:
            self.logger.warning(f"视觉功能初始化失败: {e}")
            self.vision_integration = None

    async def process_text_command(self, command_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        处理文本命令

        Args:
            command_text: 命令文本
            user_id: 用户ID

        Returns:
            处理结果
        """
        try:
            if not self.command_handler:
                raise AssistantError("命令处理器未初始化")

            self.logger.info(f"处理文本命令: {command_text[:50]}...")

            # 使用命令处理器处理命令
            result = await self.command_handler.process_command(command_text, user_id)

            # 触发响应回调
            if self.response_callback:
                await self._safe_callback(self.response_callback, result)

            return result

        except Exception as e:
            self.logger.error(f"文本命令处理失败: {e}")
            error_result = {
                "success": False,
                "response": f"命令处理失败：{str(e)}",
                "response_type": "error",
                "timestamp": datetime.now().isoformat()
            }

            if self.response_callback:
                await self._safe_callback(self.response_callback, error_result)

            return error_result

    def start_voice_listening(self) -> bool:
        """
        启动语音监听

        Returns:
            是否启动成功
        """
        try:
            if not self.voice_activation:
                self.logger.warning("语音功能未启用")
                return False

            if self.is_voice_active:
                self.logger.warning("语音监听已在运行")
                return True

            # 创建语音监听线程
            self.voice_thread = threading.Thread(
                target=self._voice_listening_loop,
                daemon=True,
                name="VoiceListening"
            )

            self.is_voice_active = True
            self.voice_thread.start()

            self.logger.info("语音监听已启动")
            return True

        except Exception as e:
            self.logger.error(f"启动语音监听失败: {e}")
            return False

    def stop_voice_listening(self):
        """停止语音监听"""
        try:
            if not self.is_voice_active:
                return

            self.is_voice_active = False

            if self.voice_activation:
                self.voice_activation.stop_listening()

            if self.voice_thread and self.voice_thread.is_alive():
                self.voice_thread.join(timeout=2.0)

            self.logger.info("语音监听已停止")

        except Exception as e:
            self.logger.error(f"停止语音监听失败: {e}")

    def _voice_listening_loop(self):
        """语音监听循环"""
        try:
            while self.is_voice_active:
                if self.voice_activation:
                    self.voice_activation.listen_once()
                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"语音监听循环错误: {e}")

    async def capture_screenshot(self, capture_type: str = "full", **kwargs) -> Dict[str, Any]:
        """
        截图

        Args:
            capture_type: 截图类型 (full, region, window)
            **kwargs: 其他参数

        Returns:
            截图结果
        """
        try:
            if not self.vision_integration:
                raise AssistantError("视觉功能未启用")

            self.logger.info(f"执行截图: {capture_type}")

            if capture_type == "full":
                result = self.vision_integration.capture_full_screen(**kwargs)
            elif capture_type == "region":
                result = self.vision_integration.capture_region(**kwargs)
            elif capture_type == "window":
                result = self.vision_integration.capture_window(**kwargs)
            else:
                raise ValueError(f"不支持的截图类型: {capture_type}")

            return result

        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def perform_ocr(self, image_path: str, engine: str = "tesseract") -> Dict[str, Any]:
        """
        执行OCR识别

        Args:
            image_path: 图片路径
            engine: OCR引擎

        Returns:
            OCR结果
        """
        try:
            if not self.vision_integration:
                raise AssistantError("视觉功能未启用")

            self.logger.info(f"执行OCR: {image_path}")

            result = self.vision_integration.recognize_text(image_path, engine)
            return result

        except Exception as e:
            self.logger.error(f"OCR失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_image(self, image_path: str, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析图像

        Args:
            image_path: 图片路径
            analysis_types: 分析类型列表

        Returns:
            分析结果
        """
        try:
            if not self.vision_integration:
                raise AssistantError("视觉功能未启用")

            self.logger.info(f"分析图像: {image_path}")

            if analysis_types is None:
                analysis_types = ["basic", "colors", "edges"]

            result = self.vision_integration.analyze_image(image_path, analysis_types)
            return result

        except Exception as e:
            self.logger.error(f"图像分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def full_vision_analysis(self, capture_type: str = "full",
                                 analysis_types: Optional[List[str]] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        完整视觉分析（截图+OCR+图像分析）

        Args:
            capture_type: 截图类型
            analysis_types: 分析类型
            **kwargs: 截图参数

        Returns:
            完整分析结果
        """
        try:
            if not self.vision_integration:
                raise AssistantError("视觉功能未启用")

            self.logger.info("执行完整视觉分析")

            if analysis_types is None:
                analysis_types = ["basic", "colors"]

            result = self.vision_integration.full_vision_analysis(
                capture_type=capture_type,
                analysis_types=analysis_types,
                **kwargs
            )

            return result

        except Exception as e:
            self.logger.error(f"完整视觉分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def set_response_callback(self, callback: Callable):
        """设置响应回调函数"""
        self.response_callback = callback

    def set_status_callback(self, callback: Callable):
        """设置状态回调函数"""
        self.status_callback = callback

    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"回调函数执行失败: {e}")

    async def _on_voice_command(self, command: str):
        """语音命令回调"""
        try:
            self.logger.info(f"收到语音命令: {command}")

            # 处理语音命令
            result = await self.process_text_command(command, "voice_user")

            # 触发响应回调（文字转语音）
            if self.response_callback:
                await self._safe_callback(self.response_callback, result)

        except Exception as e:
            self.logger.error(f"语音命令处理失败: {e}")

    async def _on_voice_status(self, status: str, message: str):
        """语音状态回调"""
        try:
            self.logger.info(f"语音状态: {status} - {message}")

            if self.status_callback:
                await self._safe_callback(
                    self.status_callback,
                    "voice", status, message
                )

        except Exception as e:
            self.logger.warning(f"语音状态回调失败: {e}")

    def _on_screenshot(self, screenshot_result: Dict[str, Any]):
        """截图结果回调"""
        try:
            self.logger.info(f"截图完成: {screenshot_result.get('filename', 'unknown')}")

            if self.status_callback:
                asyncio.create_task(self._safe_callback(
                    self.status_callback,
                    "vision", "screenshot", screenshot_result
                ))

        except Exception as e:
            self.logger.warning(f"截图回调失败: {e}")

    def _on_ocr_result(self, ocr_result: Dict[str, Any]):
        """OCR结果回调"""
        try:
            word_count = ocr_result.get("word_count", 0)
            self.logger.info(f"OCR完成，识别到 {word_count} 个文字")

            if self.status_callback:
                asyncio.create_task(self._safe_callback(
                    self.status_callback,
                    "vision", "ocr", ocr_result
                ))

        except Exception as e:
            self.logger.warning(f"OCR回调失败: {e}")

    def _on_analysis_result(self, analysis_result: Dict[str, Any]):
        """分析结果回调"""
        try:
            self.logger.info("图像分析完成")

            if self.status_callback:
                asyncio.create_task(self._safe_callback(
                    self.status_callback,
                    "vision", "analysis", analysis_result
                ))

        except Exception as e:
            self.logger.warning(f"分析回调失败: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                "session_id": self.session_id,
                "session_start_time": self.session_start_time.isoformat(),
                "uptime": str(datetime.now() - self.session_start_time).split('.')[0],
                "is_running": self.is_running,
                "is_voice_active": self.is_voice_active,
                "components": {
                    "command_handler": self.command_handler is not None,
                    "voice_activation": self.voice_activation is not None,
                    "vision_integration": self.vision_integration is not None
                },
                "config": {
                    "voice_enabled": self.config_manager.get("voice.enabled", False),
                    "vision_enabled": self.config_manager.get("vision.enabled", True)
                }
            }

            # 获取命令处理器状态
            if self.command_handler:
                status["command_stats"] = self.command_handler.get_stats()

            # 获取语音系统状态
            if self.voice_activation:
                status["voice_status"] = self.voice_activation.get_status()

            # 获取视觉系统状态
            if self.vision_integration:
                status["vision_status"] = self.vision_integration.get_system_status()

            return status

        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_capabilities(self) -> Dict[str, Any]:
        """获取助手能力"""
        try:
            capabilities = {
                "text_commands": True,
                "voice_commands": self.voice_activation is not None,
                "vision_capabilities": {
                    "screen_capture": self.vision_integration is not None,
                    "ocr": self.vision_integration is not None,
                    "image_analysis": self.vision_integration is not None
                },
                "tools": {}
            }

            # 获取工具能力
            if self.command_handler and self.command_handler.tool_caller:
                tool_status = self.command_handler.tool_caller.get_tool_status()
                capabilities["tools"] = tool_status

            return capabilities

        except Exception as e:
            self.logger.error(f"获取助手能力失败: {e}")
            return {"error": str(e)}

    def start(self):
        """启动AI助手"""
        try:
            if self.is_running:
                self.logger.warning("AI助手已在运行")
                return

            self.is_running = True

            # 启动语音监听（如果启用）
            if self.config_manager.get("voice.auto_start", False):
                self.start_voice_listening()

            self.logger.info("AI助手已启动")

            if self.status_callback:
                asyncio.create_task(self._safe_callback(
                    self.status_callback,
                    "assistant", "started", "AI助手已启动"
                ))

        except Exception as e:
            self.logger.error(f"启动AI助手失败: {e}")
            raise AssistantError(f"启动AI助手失败: {e}")

    def stop(self):
        """停止AI助手"""
        try:
            if not self.is_running:
                return

            self.is_running = False

            # 停止语音监听
            self.stop_voice_listening()

            # 取消后台任务
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()

            # 清理资源
            self.cleanup()

            self.logger.info("AI助手已停止")

            if self.status_callback:
                asyncio.create_task(self._safe_callback(
                    self.status_callback,
                    "assistant", "stopped", "AI助手已停止"
                ))

        except Exception as e:
            self.logger.error(f"停止AI助手失败: {e}")

    def cleanup(self):
        """清理资源"""
        try:
            # 清理命令处理器
            if self.command_handler:
                self.command_handler.cleanup()

            # 清理语音功能
            if self.voice_activation:
                self.voice_activation.cleanup()

            # 清理视觉功能
            if self.vision_integration:
                self.vision_integration.cleanup()

            self.logger.info("AI助手资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()

    def __del__(self):
        """析构函数"""
        self.stop()