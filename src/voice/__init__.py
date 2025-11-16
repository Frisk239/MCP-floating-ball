# MCP Floating Ball - 语音处理模块
# 提供语音识别、语音唤醒、语音合成等功能

from src.voice.speech_recognition import VoiceRecognition
from src.voice.wake_word_detector import WakeWordDetector
from src.voice.voice_activation import VoiceActivation

__all__ = [
    "VoiceRecognition",
    "WakeWordDetector",
    "VoiceActivation"
]