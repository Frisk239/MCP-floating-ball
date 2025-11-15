"""
MCP Floating Ball - AI服务编排器

智能调度和管理不同的AI服务提供商，提供统一的AI服务接口。
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time

from .providers.moonshot import KimiProvider
from .providers.dashscope import DashScopeProvider
from .providers.metaso import MetasoProvider
from ..core.config import Settings, settings
from ..core.logging import get_logger
from ..core.dependencies import inject_config
from ..core.exceptions import AIServiceError, ValidationError

logger = get_logger(__name__)


class AIProvider(str, Enum):
    """AI服务提供商枚举"""
    KIMI = "kimi"
    DASHSCOPE = "dashscope"
    METASO = "metaso"


class ServiceType(str, Enum):
    """服务类型枚举"""
    CHAT = "chat"
    VOICE_RECOGNITION = "asr"
    VOICE_SYNTHESIS = "tts"
    VISION = "vision"
    SEARCH = "search"


class AIServiceOrchestrator:
    """AI服务编排器"""

    def __init__(self, config: Optional[Settings] = None):
        """
        初始化AI服务编排器

        Args:
            config: 配置对象
        """
        from ..core.config import get_settings
        self.config = config or get_settings()
        self.logger = get_logger(f"ai.orchestrator")

        # 初始化服务提供商
        self.providers = {}
        self._initialize_providers()

        # 服务状态
        self.service_status = {}
        self._last_health_check = 0

        # 默认提供商映射
        self.default_providers = {
            ServiceType.CHAT: AIProvider.KIMI,
            ServiceType.VOICE_RECOGNITION: AIProvider.DASHSCOPE,
            ServiceType.VOICE_SYNTHESIS: AIProvider.DASHSCOPE,
            ServiceType.VISION: AIProvider.DASHSCOPE,
            ServiceType.SEARCH: AIProvider.METASO
        }

        self.logger.info("AI服务编排器初始化完成")

    def _initialize_providers(self) -> None:
        """初始化服务提供商"""
        try:
            # 初始化Kimi提供商
            if self.config.ai.kimi.api_key:
                self.providers[AIProvider.KIMI] = KimiProvider(self.config.ai.kimi)
                self.logger.info("Kimi服务提供商初始化成功")

            # 初始化DashScope提供商
            try:
                if self.config.ai.dashscope and self.config.ai.dashscope.access_key_id:
                    self.providers[AIProvider.DASHSCOPE] = DashScopeProvider(self.config.ai.dashscope)
                    self.logger.info("DashScope服务提供商初始化成功")
            except AIServiceError as e:
                self.logger.warning(f"DashScope服务提供商初始化失败: {e}")
                self.logger.info("继续使用其他AI服务提供商")

            # 初始化秘塔提供商
            try:
                if self.config.ai.metaso and self.config.ai.metaso.api_key:
                    self.providers[AIProvider.METASO] = MetasoProvider(self.config.ai.metaso)
                    self.logger.info("秘塔服务提供商初始化成功")
            except Exception as e:
                self.logger.warning(f"秘塔服务提供商初始化失败: {e}")
                self.logger.info("继续使用其他AI服务提供商")

        except Exception as e:
            self.logger.error(f"初始化服务提供商失败: {e}")
            raise AIServiceError(f"初始化服务提供商失败: {e}")

    def get_provider(self, provider: AIProvider) -> Union[KimiProvider, DashScopeProvider, MetasoProvider]:
        """
        获取服务提供商

        Args:
            provider: 提供商枚举

        Returns:
            服务提供商实例

        Raises:
            AIServiceError: 提供商不存在
        """
        if provider not in self.providers:
            raise AIServiceError(f"服务提供商 {provider.value} 不存在")

        return self.providers[provider]

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        provider: Optional[AIProvider] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        对话服务

        Args:
            messages: 对话消息列表
            provider: 指定的提供商（可选）
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            对话结果

        Raises:
            AIServiceError: 对话失败
        """
        provider = provider or self.default_providers[ServiceType.CHAT]

        try:
            self.logger.info(f"开始对话服务", provider=provider.value, message_count=len(messages))

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的对话方法
            if provider == AIProvider.KIMI:
                result = await provider_instance.chat_async(messages, system_prompt, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持对话服务")

            self.logger.info(f"对话服务成功", provider=provider.value, response_length=len(result.get("content", "")))

            return result

        except Exception as e:
            error_msg = f"对话服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def speech_to_text(
        self,
        audio_data: Union[bytes, str],
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        语音识别服务

        Args:
            audio_data: 音频数据
            provider: 指定的提供商（可选）
            **kwargs: 其他参数

        Returns:
            识别结果

        Raises:
            AIServiceError: 语音识别失败
        """
        provider = provider or self.default_providers[ServiceType.VOICE_RECOGNITION]

        try:
            self.logger.info(f"开始语音识别服务", provider=provider.value)

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的语音识别方法
            if provider == AIProvider.DASHSCOPE:
                result = await provider_instance.speech_to_text_async(audio_data, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持语音识别服务")

            self.logger.info(f"语音识别服务成功", provider=provider.value, text_length=len(result.get("text", "")))

            return result

        except Exception as e:
            error_msg = f"语音识别服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def text_to_speech(
        self,
        text: str,
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> bytes:
        """
        文本转语音服务

        Args:
            text: 要转换的文本
            provider: 指定的提供商（可选）
            **kwargs: 其他参数

        Returns:
            语音音频数据

        Raises:
            AIServiceError: 语音合成失败
        """
        provider = provider or self.default_providers[ServiceType.VOICE_SYNTHESIS]

        try:
            self.logger.info(f"开始文本转语音服务", provider=provider.value, text_length=len(text))

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的语音合成方法
            if provider == AIProvider.DASHSCOPE:
                result = await provider_instance.text_to_speech_async(text, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持语音合成服务")

            self.logger.info(f"文本转语音服务成功", provider=provider.value, audio_size=len(result))

            return result

        except Exception as e:
            error_msg = f"文本转语音服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def understand_image(
        self,
        image_path: str,
        prompt: str,
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        图像理解服务

        Args:
            image_path: 图像文件路径
            prompt: 理解提示词
            provider: 指定的提供商（可选）
            **kwargs: 其他参数

        Returns:
            理解结果

        Raises:
            AIServiceError: 图像理解失败
        """
        # 尝试使用Kimi的视觉理解能力，如果不可用则使用DashScope
        if provider is None:
            provider = AIProvider.KIMI if AIProvider.KIMI in self.providers else AIProvider.DASHSCOPE

        try:
            self.logger.info(f"开始图像理解服务", provider=provider.value, image_path=image_path)

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的图像理解方法
            if provider == AIProvider.KIMI:
                result = await provider_instance.analyze_image_async(image_path, prompt, **kwargs)
            elif provider == AIProvider.DASHSCOPE:
                result = await provider_instance.understand_image_async(image_path, prompt, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持图像理解服务")

            self.logger.info(f"图像理解服务成功", provider=provider.value, description_length=len(result.get("description", "")))

            return result

        except Exception as e:
            error_msg = f"图像理解服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def search(
        self,
        query: str,
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        搜索服务

        Args:
            query: 搜索查询
            provider: 指定的提供商（可选）
            **kwargs: 其他参数

        Returns:
            搜索结果

        Raises:
            AIServiceError: 搜索失败
        """
        provider = provider or self.default_providers[ServiceType.SEARCH]

        try:
            self.logger.info(f"开始搜索服务", provider=provider.value, query=query)

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的搜索方法
            if provider == AIProvider.METASO:
                result = await provider_instance.search_async(query, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持搜索服务")

            self.logger.info(f"搜索服务成功", provider=provider.value, total_results=result.get("total_results", 0))

            return result

        except Exception as e:
            error_msg = f"搜索服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def search_websites(
        self,
        query: str,
        websites: List[str],
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        多网站搜索服务

        Args:
            query: 搜索查询
            websites: 网站列表
            provider: 指定的提供商（可选）
            **kwargs: 其他参数

        Returns:
            搜索结果

        Raises:
            AIServiceError: 搜索失败
        """
        provider = provider or self.default_providers[ServiceType.SEARCH]

        try:
            self.logger.info(f"开始多网站搜索服务", provider=provider.value, websites=websites)

            provider_instance = self.get_provider(provider)

            # 根据提供商类型调用相应的多网站搜索方法
            if provider == AIProvider.METASO:
                result = await provider_instance.search_websites_async(query, websites, **kwargs)
            else:
                raise AIServiceError(f"提供商 {provider.value} 不支持多网站搜索服务")

            self.logger.info(f"多网站搜索服务成功", provider=provider.value, total_results=result.get("total_results", 0))

            return result

        except Exception as e:
            error_msg = f"多网站搜索服务失败: {e}"
            self.logger.error(error_msg)
            raise AIServiceError(error_msg)

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            各服务的健康状态
        """
        try:
            self.logger.info("开始健康检查")

            # 如果上次检查时间小于5分钟，返回缓存结果
            current_time = time.time()
            if current_time - self._last_health_check < 300:  # 5分钟
                return self.service_status

            health_results = {}

            # 检查各个提供商
            for provider_name, provider in self.providers.items():
                try:
                    is_healthy = await provider.test_connection()
                    health_results[provider_name.value] = {
                        "status": "healthy" if is_healthy else "unhealthy",
                        "last_check": current_time
                    }
                except Exception as e:
                    health_results[provider_name.value] = {
                        "status": "error",
                        "error": str(e),
                        "last_check": current_time
                    }

            self.service_status = {
                "providers": health_results,
                "overall_status": "healthy" if all(
                    result["status"] == "healthy" for result in health_results.values()
                ) else "degraded",
                "last_check": current_time
            }

            self._last_health_check = current_time

            self.logger.info("健康检查完成", overall_status=self.service_status["overall_status"])

            return self.service_status

        except Exception as e:
            error_msg = f"健康检查失败: {e}"
            self.logger.error(error_msg)
            return {
                "providers": {},
                "overall_status": "error",
                "error": error_msg,
                "last_check": time.time()
            }

    def get_available_providers(self, service_type: Optional[ServiceType] = None) -> List[AIProvider]:
        """
        获取可用的服务提供商

        Args:
            service_type: 服务类型（可选）

        Returns:
            可用的提供商列表
        """
        if service_type is None:
            return list(self.providers.keys())

        # 返回支持指定服务类型的提供商
        capable_providers = []
        for provider in self.providers.keys():
            # 这里可以根据不同服务类型检查提供商是否支持
            if service_type == ServiceType.CHAT and provider in [AIProvider.KIMI]:
                capable_providers.append(provider)
            elif service_type == ServiceType.VOICE_RECOGNITION and provider == AIProvider.DASHSCOPE:
                capable_providers.append(provider)
            elif service_type == ServiceType.VOICE_SYNTHESIS and provider == AIProvider.DASHSCOPE:
                capable_providers.append(provider)
            elif service_type == ServiceType.VISION and provider in [AIProvider.KIMI, AIProvider.DASHSCOPE]:
                capable_providers.append(provider)
            elif service_type == ServiceType.SEARCH and provider == AIProvider.METASO:
                capable_providers.append(provider)

        return capable_providers

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务信息字典
        """
        providers_info = {}
        for provider_name, provider in self.providers.items():
            try:
                if hasattr(provider, 'get_model_info'):
                    providers_info[provider_name.value] = provider.get_model_info()
                elif hasattr(provider, 'get_service_info'):
                    providers_info[provider_name.value] = provider.get_service_info()
                else:
                    providers_info[provider_name.value] = {"name": provider_name.value}
            except Exception as e:
                providers_info[provider_name.value] = {"name": provider_name.value, "error": str(e)}

        return {
            "orchestrator_version": "1.0.0",
            "available_providers": list(self.providers.keys()),
            "default_providers": {k.value: v.value for k, v in self.default_providers.items()},
            "providers_info": providers_info,
            "config_validation": self.config.validate_api_keys()
        }

    async def close(self) -> None:
        """关闭所有服务提供商"""
        try:
            for provider in self.providers.values():
                if hasattr(provider, 'close'):
                    provider.close()
                elif hasattr(provider, 'aclose'):
                    await provider.aclose()

            self.logger.info("AI服务编排器已关闭")

        except Exception as e:
            self.logger.error(f"关闭AI服务编排器时出错: {e}")


# 全局AI服务编排器实例（延迟加载）
ai_orchestrator = None

def get_ai_orchestrator():
    """获取AI服务编排器实例"""
    global ai_orchestrator
    if ai_orchestrator is None:
        ai_orchestrator = AIServiceOrchestrator()
    return ai_orchestrator


# 便捷函数
async def chat(
    messages: List[Dict[str, Any]],
    provider: Optional[AIProvider] = None,
    **kwargs
) -> Dict[str, Any]:
    """便捷的对话函数"""
    orchestrator = get_ai_orchestrator()
    return await orchestrator.chat(messages, provider, **kwargs)


async def speech_to_text(
    audio_data: Union[bytes, str],
    provider: Optional[AIProvider] = None,
    **kwargs
) -> Dict[str, Any]:
    """便捷的语音识别函数"""
    orchestrator = get_ai_orchestrator()
    return await orchestrator.speech_to_text(audio_data, provider, **kwargs)


async def text_to_speech(
    text: str,
    provider: Optional[AIProvider] = None,
    **kwargs
) -> bytes:
    """便捷的文本转语音函数"""
    orchestrator = get_ai_orchestrator()
    return await orchestrator.text_to_speech(text, provider, **kwargs)


async def understand_image(
    image_path: str,
    prompt: str,
    provider: Optional[AIProvider] = None,
    **kwargs
) -> Dict[str, Any]:
    """便捷的图像理解函数"""
    orchestrator = get_ai_orchestrator()
    return await orchestrator.understand_image(image_path, prompt, provider, **kwargs)


async def search(
    query: str,
    provider: Optional[AIProvider] = None,
    **kwargs
) -> Dict[str, Any]:
    """便捷的搜索函数"""
    orchestrator = get_ai_orchestrator()
    return await orchestrator.search(query, provider, **kwargs)


# 导出的类和函数
__all__ = [
    "AIServiceOrchestrator",
    "AIProvider",
    "ServiceType",
    "ai_orchestrator",
    "chat",
    "speech_to_text",
    "text_to_speech",
    "understand_image",
    "search",
]