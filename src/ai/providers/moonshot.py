"""
MCP Floating Ball - 月之暗面Kimi AI服务提供商

实现与月之暗面Kimi API的集成，提供对话、图像理解等AI能力。
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import base64
import httpx

from openai import OpenAI, AsyncOpenAI

from ...core.config import KimiSettings
from ...core.logging import get_logger
from ...core.exceptions import APIError, AIServiceError, AuthenticationError, RateLimitError

logger = get_logger(__name__)


class KimiProvider:
    """月之暗面Kimi服务提供商"""

    def __init__(self, config: KimiSettings):
        """
        初始化Kimi服务提供商

        Args:
            config: Kimi配置
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.logger = get_logger(f"ai.provider.kimi")

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_history: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        准备消息格式

        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            max_history: 最大历史消息数量

        Returns:
            List[Dict[str, Any]]: 格式化的消息列表
        """
        formatted_messages = []

        # 添加系统提示词
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })

        # 处理历史消息
        if max_history and len(messages) > max_history:
            # 保留最近的消息
            messages = messages[-max_history:]

        # 格式化用户和助手消息
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                # 处理多模态内容（文本+图像）
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") == "image":
                        image_data = item.get("image", "")
                        if image_data.startswith("data:image"):
                            # Base64编码的图像
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            })
                        else:
                            # 图像文件路径，需要读取并编码
                            try:
                                with open(image_data, "rb") as f:
                                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
                                    formatted_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    })
                            except Exception as e:
                                self.logger.warning(f"无法读取图像文件 {image_data}: {e}")
                                continue

                formatted_messages.append({
                    "role": role,
                    "content": formatted_content
                })
            else:
                # 纯文本消息
                formatted_messages.append({
                    "role": role,
                    "content": str(content)
                })

        return formatted_messages

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        与Kimi进行对话

        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            temperature: 生成温度
            max_tokens: 最大令牌数
            stream: 是否流式输出
            tools: 工具列表

        Returns:
            对话响应

        Raises:
            APIError: API调用失败
            AuthenticationError: 认证失败
            RateLimitError: 速率限制
        """
        try:
            # 准备消息
            formatted_messages = self._prepare_messages(messages, system_prompt)

            # 准备请求参数
            request_params = {
                "model": self.config.model,
                "messages": formatted_messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": stream
            }

            # 添加工具（如果提供）
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            start_time = time.time()

            # 发送请求
            self.logger.info("发送Kimi对话请求", model=self.config.model, message_count=len(formatted_messages))

            if stream:
                response = self.client.chat.completions.create(**request_params)
                return self._handle_stream_response(response)
            else:
                response = self.client.chat.completions.create(**request_params)
                result = self._handle_response(response)

            execution_time = time.time() - start_time
            self.logger.info(
                "Kimi对话请求成功",
                execution_time=execution_time,
                response_tokens=result.get("usage", {}).get("completion_tokens", 0)
            )

            return result

        except Exception as e:
            self._handle_error(e)

    async def chat_async(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Union[Dict[str, Any], Any]:
        """
        异步与Kimi进行对话

        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            temperature: 生成温度
            max_tokens: 最大令牌数
            stream: 是否流式输出
            tools: 工具列表

        Returns:
            对话响应

        Raises:
            APIError: API调用失败
            AuthenticationError: 认证失败
            RateLimitError: 速率限制
        """
        try:
            # 准备消息
            formatted_messages = self._prepare_messages(messages, system_prompt)

            # 准备请求参数
            request_params = {
                "model": self.config.model,
                "messages": formatted_messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": stream
            }

            # 添加工具（如果提供）
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            start_time = time.time()

            # 发送异步请求
            self.logger.info("发送异步Kimi对话请求", model=self.config.model, message_count=len(formatted_messages))

            if stream:
                response = await self.async_client.chat.completions.create(**request_params)
                return self._handle_stream_response(response)
            else:
                response = await self.async_client.chat.completions.create(**request_params)
                result = self._handle_response(response)

            execution_time = time.time() - start_time
            self.logger.info(
                "异步Kimi对话请求成功",
                execution_time=execution_time,
                response_tokens=result.get("usage", {}).get("completion_tokens", 0)
            )

            return result

        except Exception as e:
            self._handle_error(e)

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        分析图像

        Args:
            image_path: 图像文件路径
            prompt: 分析提示词
            max_tokens: 最大令牌数

        Returns:
            分析结果

        Raises:
            APIError: API调用失败
        """
        try:
            # 验证图像文件
            if not image_path or not isinstance(image_path, str):
                raise ValueError("图像路径不能为空")

            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "image": image_path
                        }
                    ]
                }
            ]

            return self.chat(messages, max_tokens=max_tokens)

        except Exception as e:
            self.logger.error(f"图像分析失败: {e}")
            raise AIServiceError(f"图像分析失败: {e}", service="kimi")

    async def analyze_image_async(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        异步分析图像

        Args:
            image_path: 图像文件路径
            prompt: 分析提示词
            max_tokens: 最大令牌数

        Returns:
            分析结果

        Raises:
            APIError: API调用失败
        """
        try:
            # 验证图像文件
            if not image_path or not isinstance(image_path, str):
                raise ValueError("图像路径不能为空")

            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "image": image_path
                        }
                    ]
                }
            ]

            return await self.chat_async(messages, max_tokens=max_tokens)

        except Exception as e:
            self.logger.error(f"异步图像分析失败: {e}")
            raise AIServiceError(f"异步图像分析失败: {e}", service="kimi")

    def _handle_response(self, response) -> Dict[str, Any]:
        """
        处理响应

        Args:
            response: OpenAI响应对象

        Returns:
            格式化的响应字典
        """
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content,
            "role": message.role,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        }

        # 处理工具调用
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]

        return result

    def _handle_stream_response(self, response) -> Any:
        """
        处理流式响应

        Args:
            response: 流式响应对象

        Returns:
            流式响应生成器
        """
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {
                        "content": delta.content,
                        "finish_reason": chunk.choices[0].finish_reason,
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens if chunk.usage else 0,
                            "completion_tokens": chunk.usage.completion_tokens if chunk.usage else 0,
                            "total_tokens": chunk.usage.total_tokens if chunk.usage else 0
                        }
                    }

    def _handle_error(self, error: Exception) -> None:
        """
        处理错误

        Args:
            error: 异常对象

        Raises:
            APIError: API调用失败
            AuthenticationError: 认证失败
            RateLimitError: 速率限制
        """
        error_msg = str(error)
        self.logger.error(f"Kimi API错误: {error_msg}")

        if "401" in error_msg or "authentication" in error_msg.lower():
            raise AuthenticationError(
                f"Kimi API认证失败: {error_msg}",
                provider="kimi"
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise RateLimitError(
                f"Kimi API速率限制: {error_msg}",
                provider="kimi"
            )
        elif "timeout" in error_msg.lower():
            raise APIError(
                f"Kimi API请求超时: {error_msg}",
                provider="kimi"
            )
        else:
            raise APIError(
                f"Kimi API调用失败: {error_msg}",
                provider="kimi"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "provider": "kimi",
            "model": self.config.model,
            "base_url": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_vision": True,
            "supports_tools": True,
            "supports_streaming": True
        }

    async def test_connection(self) -> bool:
        """
        测试连接

        Returns:
            连接是否成功
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hello, this is a test message."}
            ]
            response = await self.chat_async(test_messages, max_tokens=10)
            return bool(response.get("content"))
        except Exception as e:
            self.logger.error(f"Kimi连接测试失败: {e}")
            return False

    def close(self) -> None:
        """关闭客户端连接"""
        if hasattr(self.client, 'close'):
            self.client.close()
        if hasattr(self.async_client, 'close'):
            asyncio.create_task(self.async_client.close())

    async def aclose(self) -> None:
        """异步关闭客户端连接"""
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        if hasattr(self.client, 'close'):
            self.client.close()


# 导出的类和函数
__all__ = [
    "KimiProvider",
]