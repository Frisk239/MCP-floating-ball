"""
MCP Floating Ball - 语音识别模块

基于Vosk的离线语音识别功能。
"""

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

try:
    import vosk
    import sounddevice as sd
    import numpy as np
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

from src.core.logging import get_logger
from src.core.exceptions import VoiceError


class VoiceRecognition:
    """语音识别类"""

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        """
        初始化语音识别

        Args:
            model_path: Vosk模型路径
            sample_rate: 采样率
        """
        self.logger = get_logger("voice.recognition")

        if not VOSK_AVAILABLE:
            raise VoiceError("Vosk或sounddevice库未安装，请安装: pip install vosk sounddevice")

        # 设置模型路径
        if model_path is None:
            # 默认模型路径
            base_dir = Path(__file__).parent.parent.parent
            model_path = base_dir / "model" / "vosk-model-small-en-us-0.15"

        self.model_path = Path(model_path)
        self.sample_rate = sample_rate

        # 初始化模型
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()

        # 控制变量
        self._is_listening = False
        self._listen_thread = None
        self._recognition_callbacks = []

        self._initialize_model()
        self.logger.info(f"语音识别初始化完成，模型路径: {self.model_path}")

    def _initialize_model(self):
        """初始化Vosk模型"""
        try:
            if not self.model_path.exists():
                raise VoiceError(f"模型文件不存在: {self.model_path}")

            self.logger.info(f"正在加载语音模型: {self.model_path}")
            self.model = vosk.Model(str(self.model_path))
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

            self.logger.info("语音模型加载成功")

        except Exception as e:
            self.logger.error(f"语音模型加载失败: {e}")
            raise VoiceError(f"语音模型加载失败: {e}")

    def add_recognition_callback(self, callback: Callable[[str], None]):
        """
        添加识别结果回调

        Args:
            callback: 回调函数，接收识别文本
        """
        self._recognition_callbacks.append(callback)

    def remove_recognition_callback(self, callback: Callable[[str], None]):
        """
        移除识别结果回调

        Args:
            callback: 要移除的回调函数
        """
        if callback in self._recognition_callbacks:
            self._recognition_callbacks.remove(callback)

    def _notify_callbacks(self, text: str):
        """通知所有回调函数"""
        for callback in self._recognition_callbacks:
            try:
                callback(text)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")

    def start_listening(self):
        """开始语音监听"""
        if self._is_listening:
            self.logger.warning("语音监听已在运行中")
            return

        self._is_listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

        self.logger.info("开始语音监听")

    def stop_listening(self):
        """停止语音监听"""
        if not self._is_listening:
            return

        self._is_listening = False

        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2)

        self.logger.info("停止语音监听")

    def _listen_loop(self):
        """监听循环"""
        def audio_callback(indata, frames, time, status):
            """音频数据回调"""
            if status:
                self.logger.warning(f"音频输入状态: {status}")

            # 将音频数据放入队列
            self.audio_queue.put(bytes(indata))

        try:
            # 开始音频流
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=8000,
                              dtype='int16'):

                self.logger.info("音频流已启动")

                while self._is_listening:
                    try:
                        # 从队列获取音频数据
                        data = self.audio_queue.get(timeout=1)

                        # 进行语音识别
                        if self.recognizer.AcceptWaveform(data):
                            result = json.loads(self.recognizer.Result())
                            text = result.get('text', '').strip()

                            if text:
                                self.logger.info(f"识别结果: {text}")
                                self._notify_callbacks(text)
                        else:
                            # 获取部分结果（实时识别）
                            partial = json.loads(self.recognizer.PartialResult())
                            partial_text = partial.get('partial', '')

                            # 可以在这里处理实时识别结果
                            # self.logger.debug(f"部分识别: {partial_text}")

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"语音识别处理失败: {e}")

        except Exception as e:
            self.logger.error(f"音频流启动失败: {e}")
            self._is_listening = False
            raise

    def recognize_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        单次语音识别

        Args:
            timeout: 超时时间（秒）

        Returns:
            识别的文本，失败返回None
        """
        try:
            self.logger.info("开始单次语音识别")

            # 录音数据缓存
            audio_data = []

            def audio_callback(indata, frames, time, status):
                """音频数据回调"""
                audio_data.extend(indata.tolist())

            # 录音
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              dtype='int16'):
                self.logger.info(f"录音中，超时时间: {timeout}秒")
                time.sleep(timeout)

            if not audio_data:
                self.logger.warning("未录制到音频数据")
                return None

            # 转换为字节数据
            audio_bytes = np.array(audio_data, dtype='int16').tobytes()

            # 识别
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip()
                self.logger.info(f"单次识别结果: {text}")
                return text
            else:
                self.logger.info("未识别到有效语音")
                return None

        except Exception as e:
            self.logger.error(f"单次语音识别失败: {e}")
            return None

    def get_available_models(self) -> Dict[str, Any]:
        """
        获取可用模型信息

        Returns:
            模型信息字典
        """
        try:
            model_info = {
                "model_path": str(self.model_path),
                "sample_rate": self.sample_rate,
                "model_exists": self.model_path.exists(),
                "vosk_available": VOSK_AVAILABLE,
                "model_loaded": self.model is not None
            }

            # 检查模型文件
            if self.model_path.exists():
                model_files = []
                for file_path in self.model_path.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(self.model_path)
                        model_files.append(str(rel_path))
                model_info["model_files"] = model_files

            return model_info

        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {"error": str(e)}

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_listening()