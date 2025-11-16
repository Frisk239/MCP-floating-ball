"""
MCP Floating Ball - 语音激活模块

整合语音识别和唤醒词检测，提供完整的语音激活功能。
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path

from src.voice.speech_recognition import VoiceRecognition
from src.voice.wake_word_detector import WakeWordDetector
from src.core.logging import get_logger
from src.core.exceptions import VoiceError


class VoiceActivation:
    """语音激活管理器"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 wake_words: Optional[List[str]] = None,
                 auto_start: bool = False):
        """
        初始化语音激活

        Args:
            model_path: Vosk模型路径
            wake_words: 唤醒词列表
            auto_start: 是否自动开始监听
        """
        self.logger = get_logger("voice.activation")

        try:
            # 初始化语音识别
            self.speech_recognition = VoiceRecognition(model_path=model_path)

            # 初始化唤醒词检测
            self.wake_detector = WakeWordDetector(wake_words)

            # 设置回调
            self.speech_recognition.add_recognition_callback(self._on_speech_result)

            # 状态变量
            self._is_running = False
            self._is_activated = False
            self._activation_callbacks = []
            self._deactivation_callbacks = []
            self._command_callbacks = []

            # 监听模式
            self._listen_mode = "wake_word"  # wake_word, continuous, manual
            self._command_timeout = 10.0     # 命令超时时间

            # 统计信息
            self.activation_history = []
            self.recognition_history = []

            self.logger.info("语音激活系统初始化完成")

            if auto_start:
                self.start()

        except Exception as e:
            self.logger.error(f"语音激活系统初始化失败: {e}")
            raise VoiceError(f"语音激活系统初始化失败: {e}")

    def _on_speech_result(self, text: str):
        """
        语音识别结果回调

        Args:
            text: 识别的文本
        """
        timestamp = time.time()
        self.recognition_history.append({
            "text": text,
            "timestamp": timestamp,
            "is_activated": self._is_activated
        })

        # 限制历史记录数量
        if len(self.recognition_history) > 100:
            self.recognition_history.pop(0)

        self.logger.debug(f"语音识别: {text}")

        if self._listen_mode == "wake_word" and not self._is_activated:
            # 唤醒词模式：检测唤醒词
            wake_word = self.wake_detector.detect_wake_word(text)
            if wake_word:
                self._activate(wake_word)

        elif self._is_activated:
            # 激活状态：处理命令
            self._handle_command(text)

    def _activate(self, wake_word: str):
        """
        激活语音助手

        Args:
            wake_word: 触发激活的唤醒词
        """
        if self._is_activated:
            return  # 已经激活

        self._is_activated = True

        # 记录激活历史
        self.activation_history.append({
            "wake_word": wake_word,
            "timestamp": time.time(),
            "trigger": "voice"
        })

        self.logger.info(f"语音助手已激活 (唤醒词: {wake_word})")

        # 通知激活回调
        for callback in self._activation_callbacks:
            try:
                callback(wake_word)
            except Exception as e:
                self.logger.error(f"激活回调执行失败: {e}")

        # 设置命令超时
        self._schedule_command_timeout()

    def _handle_command(self, text: str):
        """
        处理用户命令

        Args:
            text: 用户命令文本
        """
        self.logger.info(f"收到用户命令: {text}")

        # 通知命令回调
        for callback in self._command_callbacks:
            try:
                callback(text, self)
            except Exception as e:
                self.logger.error(f"命令回调执行失败: {e}")

    def _schedule_command_timeout(self):
        """设置命令超时"""
        def timeout_callback():
            if self._is_activated:
                self.logger.info("命令超时，停用语音助手")
                self.deactivate()

        timer = threading.Timer(self._command_timeout, timeout_callback, daemon=True)
        timer.start()

    def start(self):
        """开始语音激活"""
        if self._is_running:
            self.logger.warning("语音激活已在运行")
            return

        try:
            self.speech_recognition.start_listening()
            self._is_running = True
            self.logger.info("语音激活已启动")

        except Exception as e:
            self.logger.error(f"启动语音激活失败: {e}")
            raise VoiceError(f"启动语音激活失败: {e}")

    def stop(self):
        """停止语音激活"""
        if not self._is_running:
            return

        try:
            self.speech_recognition.stop_listening()
            self._is_running = False
            self.deactivate()  # 确保停用状态
            self.logger.info("语音激活已停止")

        except Exception as e:
            self.logger.error(f"停止语音激活失败: {e}")

    def activate_manually(self, reason: str = "manual"):
        """
        手动激活

        Args:
            reason: 激活原因
        """
        if self._is_activated:
            return

        self._is_activated = True

        # 记录激活历史
        self.activation_history.append({
            "wake_word": None,
            "timestamp": time.time(),
            "trigger": reason
        })

        self.logger.info(f"语音助手已手动激活")

        # 通知激活回调
        for callback in self._activation_callbacks:
            try:
                callback(None)
            except Exception as e:
                self.logger.error(f"激活回调执行失败: {e}")

        # 设置命令超时
        self._schedule_command_timeout()

    def deactivate(self):
        """停用语音助手"""
        if not self._is_activated:
            return

        self._is_activated = False
        self.wake_detector.deactivate()

        self.logger.info("语音助手已停用")

        # 通知停用回调
        for callback in self._deactivation_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"停用回调执行失败: {e}")

    def set_listen_mode(self, mode: str):
        """
        设置监听模式

        Args:
            mode: 监听模式 (wake_word, continuous, manual)
        """
        if mode not in ["wake_word", "continuous", "manual"]:
            raise ValueError("无效的监听模式")

        self._listen_mode = mode
        self.logger.info(f"设置监听模式: {mode}")

    def set_command_timeout(self, timeout: float):
        """
        设置命令超时时间

        Args:
            timeout: 超时时间（秒）
        """
        self._command_timeout = timeout
        self.logger.info(f"设置命令超时时间: {timeout}秒")

    def add_activation_callback(self, callback: Callable[[Optional[str]], None]):
        """
        添加激活回调

        Args:
            callback: 回调函数，参数为唤醒词（手动激活时为None）
        """
        self._activation_callbacks.append(callback)

    def add_deactivation_callback(self, callback: Callable[[], None]):
        """
        添加停用回调

        Args:
            callback: 回调函数
        """
        self._deactivation_callbacks.append(callback)

    def add_command_callback(self, callback: Callable[[str, 'VoiceActivation'], None]):
        """
        添加命令回调

        Args:
            callback: 回调函数，参数为命令文本和VoiceActivation实例
        """
        self._command_callbacks.append(callback)

    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        单次语音识别

        Args:
            timeout: 超时时间

        Returns:
            识别的文本，失败返回None
        """
        if not self._is_running:
            self.logger.warning("语音激活未启动，请先调用start()")
            return None

        return self.speech_recognition.recognize_once(timeout)

    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            状态信息字典
        """
        return {
            "is_running": self._is_running,
            "is_activated": self._is_activated,
            "listen_mode": self._listen_mode,
            "command_timeout": self._command_timeout,
            "wake_detector_status": self.wake_detector.get_status(),
            "speech_recognition_status": self.speech_recognition.get_available_models(),
            "activation_count": len(self.activation_history),
            "recognition_count": len(self.recognition_history),
            "recent_activations": self.activation_history[-5:] if self.activation_history else [],
            "recent_recognitions": self.recognition_history[-10:] if self.recognition_history else []
        }

    def add_wake_word(self, wake_word: str):
        """
        添加唤醒词

        Args:
            wake_word: 要添加的唤醒词
        """
        self.wake_detector.add_wake_word(wake_word)

    def remove_wake_word(self, wake_word: str):
        """
        移除唤醒词

        Args:
            wake_word: 要移除的唤醒词
        """
        self.wake_detector.remove_wake_word(wake_word)

    def set_wake_word_threshold(self, threshold: float):
        """
        设置唤醒词相似度阈值

        Args:
            threshold: 阈值，范围0-1
        """
        self.wake_detector.set_similarity_threshold(threshold)

    def test_system(self) -> Dict[str, Any]:
        """
        测试语音激活系统

        Returns:
            测试结果
        """
        results = {
            "speech_recognition": {
                "available": True,
                "model_loaded": self.speech_recognition.model is not None,
                "details": self.speech_recognition.get_available_models()
            },
            "wake_detector": {
                "available": True,
                "wake_words": self.wake_detector.wake_words,
                "threshold": self.wake_detector.similarity_threshold
            },
            "overall": {
                "ready": False,
                "issues": []
            }
        }

        # 检查整体就绪状态
        issues = []
        if not results["speech_recognition"]["model_loaded"]:
            issues.append("语音模型未加载")
        if not results["wake_detector"]["wake_words"]:
            issues.append("未设置唤醒词")

        results["overall"]["ready"] = len(issues) == 0
        results["overall"]["issues"] = issues

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "activation_count": len(self.activation_history),
            "recognition_count": len(self.recognition_history),
            "wake_detector_stats": self.wake_detector.get_status(),
            "uptime": time.time() - (self.activation_history[0]["timestamp"] if self.activation_history else time.time()),
            "activation_rate": len(self.activation_history) / max(1, len(self.recognition_history)),
            "average_activation_interval": self._calculate_average_activation_interval()
        }

    def _calculate_average_activation_interval(self) -> float:
        """计算平均激活间隔"""
        if len(self.activation_history) < 2:
            return 0

        intervals = []
        for i in range(1, len(self.activation_history)):
            interval = self.activation_history[i]["timestamp"] - self.activation_history[i-1]["timestamp"]
            intervals.append(interval)

        return sum(intervals) / len(intervals)

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()