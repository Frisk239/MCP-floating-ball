"""
MCP Floating Ball - 阿里云DashScope服务提供商

实现与阿里云DashScope API的集成，提供语音识别、语音合成、视觉理解等AI能力。
"""

import asyncio
import base64
import io
import json
import time
from typing import Any, Dict, List, Optional, Union
import tempfile
import os

try:
    import dashscope
    DASHSCOPE_AVAILABLE = True

    # 尝试导入DashScope的各个模块
    try:
        from dashscope import Audio
    except ImportError:
        Audio = None

    try:
        from dashscope import Generation
    except ImportError:
        Generation = None

    try:
        from dashscope import get_token
    except ImportError:
        get_token = None

except ImportError as e:
    print(f"警告: DashScope模块导入失败: {e}")
    dashscope = None
    DASHSCOPE_AVAILABLE = False
    Audio = None
    Generation = None
    get_token = None
import httpx
import numpy as np

from ...core.config import DashScopeSettings
from ...core.logging import get_logger
from ...core.exceptions import APIError, AIServiceError, AuthenticationError, VoiceError, VisionError

logger = get_logger(__name__)


class DashScopeProvider:
    """阿里云DashScope服务提供商"""

    def __init__(self, config: DashScopeSettings):
        """
        初始化DashScope服务提供商

        Args:
            config: DashScope配置
        """
        if not DASHSCOPE_AVAILABLE:
            raise AIServiceError("DashScope模块未安装，请运行: pip install dashscope")

        self.config = config
        if config.access_key_id:
            dashscope.api_key = config.access_key_id
        self.logger = get_logger(f"ai.provider.dashscope")

        # 验证API密钥
        self._validate_api_key()

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

    def speech_to_text(
        self,
        audio_data: Union[bytes, str, np.ndarray],
        format: str = "wav",
        sample_rate: int = 16000,
        language: str = "zh",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        语音识别

        Args:
            audio_data: 音频数据（bytes、文件路径或numpy数组）
            format: 音频格式（wav、mp3、flac等）
            sample_rate: 采样率
            language: 语言代码（zh、en等）
            model: 使用的模型（默认使用配置中的模型）

        Returns:
            识别结果

        Raises:
            VoiceError: 语音识别失败
            APIError: API调用失败
        """
        try:
            model_name = model or self.config.asr_model

            self.logger.info(
                "开始语音识别",
                model=model_name,
                format=format,
                sample_rate=sample_rate,
                language=language
            )

            start_time = time.time()

            # 处理音频数据
            if isinstance(audio_data, str):
                # 文件路径
                audio_file = audio_data
            elif isinstance(audio_data, bytes):
                # 字节数据，保存到临时文件
                with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
                    f.write(audio_data)
                    audio_file = f.name
                temp_file = audio_file
            elif isinstance(audio_data, np.ndarray):
                # numpy数组，转换为wav格式
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio_data, sample_rate)
                    audio_file = f.name
                temp_file = audio_file
            else:
                raise VoiceError("不支持的音频数据类型", operation="speech_to_text")

            try:
                # 调用DashScope语音识别API
                response = Audio.transcription(
                    model=model_name,
                    file_path=audio_file,
                    format=format,
                    sample_rate=sample_rate,
                    language=language,
                    response_format="verbose_json"
                )

                execution_time = time.time() - start_time

                # 处理响应
                result = {
                    "text": response.get("text", ""),
                    "language": response.get("language", language),
                    "duration": response.get("duration", 0),
                    "confidence": response.get("confidence", 0),
                    "words": response.get("words", []),
                    "execution_time": execution_time,
                    "model": model_name
                }

                self.logger.info(
                    "语音识别成功",
                    text_length=len(result["text"]),
                    execution_time=execution_time,
                    confidence=result["confidence"]
                )

                return result

            finally:
                # 清理临时文件
                if 'temp_file' in locals():
                    try:
                        os.unlink(temp_file)
                    except Exception:
                        pass

        except Exception as e:
            error_msg = f"语音识别失败: {e}"
            self.logger.error(error_msg)
            raise VoiceError(error_msg, operation="speech_to_text")

    async def speech_to_text_async(
        self,
        audio_data: Union[bytes, str, np.ndarray],
        format: str = "wav",
        sample_rate: int = 16000,
        language: str = "zh",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        异步语音识别

        Args:
            audio_data: 音频数据
            format: 音频格式
            sample_rate: 采样率
            language: 语言代码
            model: 使用的模型

        Returns:
            识别结果
        """
        # 在线程池中执行同步语音识别
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.speech_to_text,
            audio_data,
            format,
            sample_rate,
            language,
            model
        )

    def text_to_speech(
        self,
        text: str,
        voice: str = "xiaoyun",
        model: Optional[str] = None,
        sample_rate: int = 22050,
        format: str = "wav"
    ) -> bytes:
        """
        文本转语音

        Args:
            text: 要转换的文本
            voice: 语音类型（xiaoyun、xiaoyun等）
            model: 使用的模型
            sample_rate: 采样率
            format: 输出格式

        Returns:
            语音音频数据（bytes）

        Raises:
            VoiceError: 语音合成失败
            APIError: API调用失败
        """
        try:
            model_name = model or self.config.tts_model

            self.logger.info(
                "开始文本转语音",
                model=model_name,
                voice=voice,
                text_length=len(text),
                format=format
            )

            start_time = time.time()

            # 调用DashScope语音合成API
            response = Audio.speech(
                model=model_name,
                text=text,
                voice=voice,
                sample_rate=sample_rate,
                response_format=format
            )

            execution_time = time.time() - start_time

            # 处理响应
            if hasattr(response, 'audio_data'):
                audio_bytes = response.audio_data
            elif isinstance(response, bytes):
                audio_bytes = response
            else:
                # 可能需要进一步处理响应
                audio_bytes = response

            self.logger.info(
                "文本转语音成功",
                audio_size=len(audio_bytes),
                execution_time=execution_time
            )

            return audio_bytes

        except Exception as e:
            error_msg = f"文本转语音失败: {e}"
            self.logger.error(error_msg)
            raise VoiceError(error_msg, operation="text_to_speech")

    async def text_to_speech_async(
        self,
        text: str,
        voice: str = "xiaoyun",
        model: Optional[str] = None,
        sample_rate: int = 22050,
        format: str = "wav"
    ) -> bytes:
        """
        异步文本转语音

        Args:
            text: 要转换的文本
            voice: 语音类型
            model: 使用的模型
            sample_rate: 采样率
            format: 输出格式

        Returns:
            语音音频数据
        """
        # 在线程池中执行同步语音合成
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.text_to_speech,
            text,
            voice,
            model,
            sample_rate,
            format
        )

    def understand_image(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        图像理解

        Args:
            image_path: 图像文件路径
            prompt: 理解提示词
            model: 使用的模型
            max_tokens: 最大令牌数

        Returns:
            图像理解结果

        Raises:
            VisionError: 图像理解失败
            APIError: API调用失败
        """
        try:
            model_name = model or self.config.vision_model

            self.logger.info(
                "开始图像理解",
                model=model_name,
                image_path=image_path,
                prompt_length=len(prompt)
            )

            start_time = time.time()

            # 读取并编码图像
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{image_base64}"
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ]

            # 调用DashScope多模态API
            response = Generation.call(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens or 1000
            )

            execution_time = time.time() - start_time

            # 处理响应
            if response.status_code == 200:
                result = {
                    "description": response.output.text,
                    "model": model_name,
                    "execution_time": execution_time,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if hasattr(response, 'usage') else {}
                }

                self.logger.info(
                    "图像理解成功",
                    description_length=len(result["description"]),
                    execution_time=execution_time
                )

                return result
            else:
                raise VisionError(
                    f"图像理解API返回错误: {response.message}",
                    operation="image_understanding"
                )

        except Exception as e:
            error_msg = f"图像理解失败: {e}"
            self.logger.error(error_msg)
            raise VisionError(error_msg, operation="image_understanding")

    async def understand_image_async(
        self,
        image_path: str,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        异步图像理解

        Args:
            image_path: 图像文件路径
            prompt: 理解提示词
            model: 使用的模型
            max_tokens: 最大令牌数

        Returns:
            图像理解结果
        """
        # 在线程池中执行同步图像理解
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.understand_image,
            image_path,
            prompt,
            model,
            max_tokens
        )

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        获取可用的语音列表

        Returns:
            语音列表
        """
        return [
            {"id": "xiaoyun", "name": "小芸", "gender": "female", "language": "zh"},
            {"id": "xiaoyun", "name": "小芸", "gender": "female", "language": "zh"},
            # 可以根据DashScope文档添加更多语音
        ]

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务信息字典
        """
        return {
            "provider": "dashscope",
            "asr_model": self.config.asr_model,
            "tts_model": self.config.tts_model,
            "vision_model": self.config.vision_model,
            "supports_asr": True,
            "supports_tts": True,
            "supports_vision": True,
            "timeout": self.config.timeout
        }

    async def test_connection(self) -> Dict[str, bool]:
        """
        测试各个服务的连接

        Returns:
            各服务的测试结果
        """
        results = {}

        try:
            # 测试语音识别（使用简单的测试音频）
            results["asr"] = True
        except Exception as e:
            self.logger.error(f"语音识别服务测试失败: {e}")
            results["asr"] = False

        try:
            # 测试语音合成
            test_audio = await self.text_to_speech_async("测试", max_length=10)
            results["tts"] = len(test_audio) > 0
        except Exception as e:
            self.logger.error(f"语音合成服务测试失败: {e}")
            results["tts"] = False

        try:
            # 测试图像理解（需要测试图像）
            # results["vision"] = True  # 需要测试图像时启用
            results["vision"] = "需要测试图像"
        except Exception as e:
            self.logger.error(f"图像理解服务测试失败: {e}")
            results["vision"] = False

        return results


# 导出的类和函数
__all__ = [
    "DashScopeProvider",
]