"""
MCP Floating Ball - 唤醒词检测模块

检测特定的唤醒词，用于激活AI助手。
"""

import re
import time
from typing import List, Optional, Callable, Dict, Any
from difflib import SequenceMatcher

from src.core.logging import get_logger
from src.core.exceptions import VoiceError


class WakeWordDetector:
    """唤醒词检测器"""

    def __init__(self, wake_words: Optional[List[str]] = None):
        """
        初始化唤醒词检测器

        Args:
            wake_words: 唤醒词列表，默认使用常见的唤醒词
        """
        self.logger = get_logger("voice.wake_word")

        # 默认唤醒词（中文和英文）
        self.default_wake_words = [
            "你好小助手",
            "小助手",
            "助手",
            "AI助手",
            "智能助手",
            "hello assistant",
            "assistant",
            "ai assistant",
            "computer",
            "hey computer"
        ]

        # 设置唤醒词
        self.wake_words = wake_words or self.default_wake_words

        # 配置参数
        self.similarity_threshold = 0.7  # 相似度阈值
        self.activation_timeout = 5.0    # 激活超时时间（秒）
        self.cooldown_period = 2.0       # 冷却期（秒）

        # 状态变量
        self._is_active = False
        self._last_activation_time = 0
        self._activation_callbacks = []
        self._deactivation_callbacks = []

        # 统计信息
        self.detection_count = 0
        self.false_positive_count = 0

        self.logger.info(f"唤醒词检测器初始化完成，唤醒词: {self.wake_words}")

    def add_wake_word(self, wake_word: str):
        """
        添加唤醒词

        Args:
            wake_word: 要添加的唤醒词
        """
        if wake_word.lower() not in [w.lower() for w in self.wake_words]:
            self.wake_words.append(wake_word)
            self.logger.info(f"添加唤醒词: {wake_word}")

    def remove_wake_word(self, wake_word: str):
        """
        移除唤醒词

        Args:
            wake_word: 要移除的唤醒词
        """
        for i, word in enumerate(self.wake_words):
            if word.lower() == wake_word.lower():
                removed = self.wake_words.pop(i)
                self.logger.info(f"移除唤醒词: {removed}")
                break

    def set_similarity_threshold(self, threshold: float):
        """
        设置相似度阈值

        Args:
            threshold: 阈值，范围0-1
        """
        if 0 <= threshold <= 1:
            self.similarity_threshold = threshold
            self.logger.info(f"设置相似度阈值: {threshold}")
        else:
            raise ValueError("相似度阈值必须在0-1之间")

    def add_activation_callback(self, callback: Callable[[str], None]):
        """
        添加激活回调函数

        Args:
            callback: 回调函数，接收匹配的唤醒词
        """
        self._activation_callbacks.append(callback)

    def add_deactivation_callback(self, callback: Callable[[], None]):
        """
        添加停用回调函数

        Args:
            callback: 回调函数
        """
        self._deactivation_callbacks.append(callback)

    def _notify_activation_callbacks(self, wake_word: str):
        """通知激活回调"""
        for callback in self._activation_callbacks:
            try:
                callback(wake_word)
            except Exception as e:
                self.logger.error(f"激活回调执行失败: {e}")

    def _notify_deactivation_callbacks(self):
        """通知停用回调"""
        for callback in self._deactivation_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"停用回调执行失败: {e}")

    def detect_wake_word(self, text: str) -> Optional[str]:
        """
        检测唤醒词

        Args:
            text: 输入文本

        Returns:
            匹配的唤醒词，未找到返回None
        """
        if not text or not text.strip():
            return None

        text = text.strip().lower()

        # 检查是否在冷却期内
        current_time = time.time()
        if current_time - self._last_activation_time < self.cooldown_period:
            return None

        # 首先检查精确匹配
        for wake_word in self.wake_words:
            if wake_word.lower() in text:
                self._activate(wake_word)
                return wake_word

        # 模糊匹配
        for wake_word in self.wake_words:
            similarity = self._calculate_similarity(text, wake_word.lower())
            if similarity >= self.similarity_threshold:
                self._activate(wake_word, similarity)
                return wake_word

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数，范围0-1
        """
        # 使用SequenceMatcher计算相似度
        return SequenceMatcher(None, text1, text2).ratio()

    def _activate(self, wake_word: str, similarity: float = 1.0):
        """
        激活检测

        Args:
            wake_word: 匹配的唤醒词
            similarity: 相似度分数
        """
        self._is_active = True
        self._last_activation_time = time.time()
        self.detection_count += 1

        self.logger.info(f"唤醒词检测成功: '{wake_word}' (相似度: {similarity:.2f})")
        self._notify_activation_callbacks(wake_word)

        # 设置自动停用定时器
        self._schedule_deactivation()

    def _schedule_deactivation(self):
        """安排自动停用"""
        import threading

        def deactivate_timer():
            time.sleep(self.activation_timeout)
            if self._is_active:
                self.deactivate()

        timer = threading.Timer(self.activation_timeout, deactivate_timer, daemon=True)
        timer.start()

    def deactivate(self):
        """手动停用"""
        if self._is_active:
            self._is_active = False
            self.logger.info("唤醒词检测器已停用")
            self._notify_deactivation_callbacks()

    def is_active(self) -> bool:
        """检查是否处于激活状态"""
        return self._is_active

    def get_status(self) -> Dict[str, Any]:
        """
        获取检测器状态

        Returns:
            状态信息字典
        """
        return {
            "is_active": self._is_active,
            "wake_words": self.wake_words,
            "similarity_threshold": self.similarity_threshold,
            "activation_timeout": self.activation_timeout,
            "cooldown_period": self.cooldown_period,
            "detection_count": self.detection_count,
            "false_positive_count": self.false_positive_count,
            "last_activation_time": self._last_activation_time,
            "time_since_last_activation": time.time() - self._last_activation_time if self._last_activation_time > 0 else None
        }

    def test_wake_word(self, test_text: str) -> Dict[str, Any]:
        """
        测试唤醒词检测

        Args:
            test_text: 测试文本

        Returns:
            测试结果字典
        """
        results = []

        text_lower = test_text.lower()

        for wake_word in self.wake_words:
            wake_word_lower = wake_word.lower()

            # 精确匹配
            exact_match = wake_word_lower in text_lower
            similarity = self._calculate_similarity(text_lower, wake_word_lower)
            fuzzy_match = similarity >= self.similarity_threshold

            results.append({
                "wake_word": wake_word,
                "exact_match": exact_match,
                "fuzzy_match": fuzzy_match,
                "similarity": similarity,
                "would_activate": exact_match or fuzzy_match
            })

        # 找出最佳匹配
        best_match = max(results, key=lambda x: x["similarity"])

        return {
            "test_text": test_text,
            "results": results,
            "best_match": best_match,
            "would_activate": best_match["would_activate"]
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.detection_count = 0
        self.false_positive_count = 0
        self._last_activation_time = 0
        self.logger.info("统计信息已重置")

    def export_config(self) -> Dict[str, Any]:
        """
        导出配置

        Returns:
            配置字典
        """
        return {
            "wake_words": self.wake_words,
            "similarity_threshold": self.similarity_threshold,
            "activation_timeout": self.activation_timeout,
            "cooldown_period": self.cooldown_period
        }

    def import_config(self, config: Dict[str, Any]):
        """
        导入配置

        Args:
            config: 配置字典
        """
        if "wake_words" in config:
            self.wake_words = config["wake_words"]
        if "similarity_threshold" in config:
            self.similarity_threshold = config["similarity_threshold"]
        if "activation_timeout" in config:
            self.activation_timeout = config["activation_timeout"]
        if "cooldown_period" in config:
            self.cooldown_period = config["cooldown_period"]

        self.logger.info("配置导入完成")