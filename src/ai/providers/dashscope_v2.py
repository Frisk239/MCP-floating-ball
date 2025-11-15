"""
MCP Floating Ball - 阿里云DashScope服务提供商 V2

基于实际DashScope模块结构重新实现的适配器，提供语音识别、语音合成、视觉理解等AI能力。
"""

import asyncio
import base64
import io
import json
import time
from typing import Any, Dict, List, Optional, Union
import tempfile
import os
import numpy as np

try:
    import dashscope
    DASHSCOPE_AVAILABLE = True

    # 根据实际的DashScope结构导入正确的模块
    try:
        from dashscope import SpeechSynthesizer, Transcription, Understanding
        SPEECH_AVAILABLE = True
    except ImportError:
        SPEECH_AVAILABLE = False
        SpeechSynthesizer = None
        Transcription = None
        Understanding = None

    try:
        from dashscope.audio import Audio
        AUDIO_AVAILABLE = True
    except ImportError:
        AUDIO_AVAILABLE = False
        Audio = None

    try:
        from dashscope import ImageSynthesis
        IMAGE_AVAILABLE = True
    except ImportError:
        IMAGE_AVAILABLE = False
        ImageSynthesis = None

    try:
        from dashscope import Generation
        GENERATION_AVAILABLE = True
    except ImportError:
        GENERATION_AVAILABLE = False
        Generation = None

except ImportError as e:
    print(f"警告: DashScope模块导入失败: {e}")
    dashscope = None
    DASHSCOPE_AVAILABLE = False
    SPEECH_AVAILABLE = False
    AUDIO_AVAILABLE = False
    IMAGE_AVAILABLE = False
    GENERATION_AVAILABLE = False

import httpx

from ...core.config import DashScopeSettings
from ...core.logging import get_logger
from ...core.exceptions import APIError, AIServiceError, AuthenticationError, VoiceError, VisionError

logger = get_logger(__name__)


class DashScopeV2Provider:
    """阿里云DashScope服务提供商 V2 - 基于实际模块结构"""

    def __init__(self, config: DashScopeSettings):
        """
        初始化DashScope服务提供商

        Args:
            config: DashScope配置
        """
        if not DASHSCOPE_AVAILABLE:
            raise AIServiceError("DashScope模块未安装，请运行: pip install dashscope")

        self.config = config

        # 设置API密钥
        if hasattr(dashscope, 'api_key') and config.access_key_id:
            dashscope.api_key = config.access_key_id

        self.logger = get_logger(f"ai.provider.dashscope_v2")

        # 验证API密钥
        self._validate_api_key()

        # 记录可用功能
        self._log_available_features()

    def _validate_api_key(self) -> None:
        """验证API密钥"""
        if not self.config.access_key_id:
            raise AuthenticationError("DashScope API密钥不能为空", provider="dashscope")

        if dashscope is None:
            raise AIServiceError("DashScope库未安装，请安装: pip install dashscope", service="dashscope")

        try:
            # 设置API密钥
            dashscope.api_key = self.config.access_key_id
            self.logger.info("DashScope API密钥配置成功")
        except Exception as e:
            raise AuthenticationError(f"DashScope API密钥配置失败: {e}", provider="dashscope")

    def _log_available_features(self) -> None:
        """记录可用功能"""
        available_features = []
        if SPEECH_AVAILABLE:
            available_features.append("语音合成")
            available_features.append("语音转录")
        if AUDIO_AVAILABLE:
            available_features.append("音频处理")
        if IMAGE_AVAILABLE:
            available_features.append("图像合成")
        if GENERATION_AVAILABLE:
            available_features.append("文本生成")

        self.logger.info(f"DashScope可用功能: {', '.join(available_features)}")

    async def speech_to_text(
        self,
        audio_data: Union[bytes, str, np.ndarray],
        model: Optional[str] = None,
        format: str = "wav",
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        语音识别（语音转文字）

        Args:
            audio_data: 音频数据
            model: 模型名称
            format: 音频格式
            sample_rate: 采样率

        Returns:
            识别结果
        """
        if not SPEECH_AVAILABLE:
            raise VoiceError("DashScope语音转录功能不可用", provider="dashscope")

        try:
            model = model or self.config.asr_model

            self.logger.info(
                "开始语音识别",
                model=model,
                format=format,
                sample_rate=sample_rate,
                data_size=len(audio_data) if isinstance(audio_data, (bytes, np.ndarray)) else "file_path"
            )

            start_time = time.time()

            # 处理音频数据
            if isinstance(audio_data, str):
                # 文件路径
                audio_file = audio_data
            elif isinstance(audio_data, bytes):
                # 字节数据，需要保存为临时文件
                with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
                    f.write(audio_data)
                    audio_file = f.name
            elif isinstance(audio_data, np.ndarray):
                # numpy数组，需要转换为音频文件
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
                    sf.write(f.name, audio_data, sample_rate)
                    audio_file = f.name
            else:
                raise VoiceError("不支持的音频数据类型", provider="dashscope")

            try:
                # 使用DashScope的Transcription进行语音识别
                response = Transcription.call(
                    model=model,
                    audio_file=audio_file,
                    format=format,
                    sample_rate=sample_rate
                )

                execution_time = time.time() - start_time

                if response.status_code == 200:
                    result = {
                        "success": True,
                        "text": response.output.text,
                        "confidence": getattr(response.output, 'confidence', None),
                        "model": model,
                        "execution_time": execution_time,
                        "raw_response": response.output
                    }
                    self.logger.info(
                        "语音识别成功",
                        text=result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"],
                        execution_time=execution_time
                    )
                    return result
                else:
                    raise VoiceError(f"语音识别失败: {response.message}", provider="dashscope")

            finally:
                # 清理临时文件
                if isinstance(audio_data, (bytes, np.ndarray)):
                    try:
                        os.unlink(audio_file)
                    except OSError:
                        pass

        except Exception as e:
            error_msg = f"语音识别失败: {e}"
            self.logger.error(error_msg)
            raise VoiceError(error_msg, provider="dashscope")

    async def text_to_speech(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "xiaoyun",
        format: str = "mp3",
        sample_rate: int = 22050
    ) -> bytes:
        """
        文本转语音

        Args:
            text: 要转换的文本
            model: 模型名称
            voice: 语音类型
            format: 音频格式
            sample_rate: 采样率

        Returns:
            音频数据的字节流
        """
        if not SPEECH_AVAILABLE:
            raise VoiceError("DashScope语音合成功能不可用", provider="dashscope")

        try:
            model = model or self.config.tts_model

            self.logger.info(
                "开始文本转语音",
                text=text[:50] + "..." if len(text) > 50 else text,
                model=model,
                voice=voice,
                format=format
            )

            start_time = time.time()

            # 使用DashScope的SpeechSynthesizer进行语音合成
            response = SpeechSynthesizer.call(
                model=model,
                text=text,
                voice=voice,
                format=format,
                sample_rate=sample_rate
            )

            execution_time = time.time() - start_time

            if response.status_code == 200:
                self.logger.info(
                    "文本转语音成功",
                    model=model,
                    voice=voice,
                    execution_time=execution_time
                )
                return response.output
            else:
                raise VoiceError(f"文本转语音失败: {response.message}", provider="dashscope")

        except Exception as e:
            error_msg = f"文本转语音失败: {e}"
            self.logger.error(error_msg)
            raise VoiceError(error_msg, provider="dashscope")

    async def analyze_image(
        self,
        image_data: Union[bytes, str],
        text: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        图像理解分析

        Args:
            image_data: 图像数据
            text: 提问文本
            model: 模型名称

        Returns:
            分析结果
        """
        if not GENERATION_AVAILABLE:
            raise VisionError("DashScope图像理解功能不可用", provider="dashscope")

        try:
            model = model or self.config.vision_model

            self.logger.info(
                "开始图像理解分析",
                model=model,
                text=text[:50] + "..." if text and len(text) > 50 else text,
                image_size=len(image_data) if isinstance(image_data, bytes) else "file_path"
            )

            start_time = time.time()

            # 处理图像数据
            if isinstance(image_data, str):
                # 文件路径
                image_url = image_data
            elif isinstance(image_data, bytes):
                # 字节数据，需要编码为base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{image_base64}"
            else:
                raise VisionError("不支持的图像数据类型", provider="dashscope")

            # 构建对话消息
            messages = []
            if text:
                messages.append({"role": "user", "content": text})

            # 添加图像消息
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url}
                ]
            })

            # 使用DashScope的Generation进行多模态理解
            response = Generation.call(
                model=model,
                messages=messages
            )

            execution_time = time.time() - start_time

            if response.status_code == 200:
                result = {
                    "success": True,
                    "text": response.output.text,
                    "model": model,
                    "execution_time": execution_time,
                    "raw_response": response.output
                }
                self.logger.info(
                    "图像理解成功",
                    text=result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"],
                    execution_time=execution_time
                )
                return result
            else:
                raise VisionError(f"图像理解失败: {response.message}", provider="dashscope")

        except Exception as e:
            error_msg = f"图像理解失败: {e}"
            self.logger.error(error_msg)
            raise VisionError(error_msg, provider="dashscope")

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务信息字典
        """
        return {
            "provider": "dashscope_v2",
            "dashscope_available": DASHSCOPE_AVAILABLE,
            "speech_available": SPEECH_AVAILABLE,
            "audio_available": AUDIO_AVAILABLE,
            "image_available": IMAGE_AVAILABLE,
            "generation_available": GENERATION_AVAILABLE,
            "asr_model": self.config.asr_model,
            "tts_model": self.config.tts_model,
            "vision_model": self.config.vision_model,
            "timeout": self.config.timeout,
            "api_key_configured": bool(self.config.access_key_id)
        }

    async def test_connection(self) -> bool:
        """
        测试连接

        Returns:
            连接是否成功
        """
        try:
            # 执行一个简单的语音合成测试
            result = await self.text_to_speech("测试", format="wav")
            return bool(result and len(result) > 0)
        except Exception as e:
            self.logger.error(f"DashScope连接测试失败: {e}")
            return False


# 导出的类和函数
__all__ = [
    "DashScopeV2Provider",
]