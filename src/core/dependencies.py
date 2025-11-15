"""
MCP Floating Ball 依赖注入模块

使用dependency-injector实现依赖注入，提供更好的模块解耦和测试支持。
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from .config import Settings, settings


class CoreContainer(containers.DeclarativeContainer):
    """核心依赖容器"""

    # 配置
    config = providers.Object(settings)

    # 基础工具
    path = providers.Configuration()

    # 日志配置
    logging_config = providers.Configuration()


class AIServicesContainer(containers.DeclarativeContainer):
    """AI服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # AI服务提供商
    kimi_client = providers.Singleton(
        # 将在实现时替换为实际的Kimi客户端
        lambda: None
    )

    dashscope_client = providers.Singleton(
        # 将在实现时替换为实际的DashScope客户端
        lambda: None
    )

    metaso_client = providers.Singleton(
        # 将在实现时替换为实际的秘塔客户端
        lambda: None
    )


class VoiceContainer(containers.DeclarativeContainer):
    """语音服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # AI服务
    ai_services = providers.DependenciesContainer()

    # 语音服务
    asr_service = providers.Singleton(
        # 将在实现时替换为实际的ASR服务
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )

    tts_service = providers.Singleton(
        # 将在实现时替换为实际的TTS服务
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )

    audio_player = providers.Singleton(
        # 将在实现时替换为实际的音频播放器
        lambda config: None,
        config=config
    )


class VisionContainer(containers.DeclarativeContainer):
    """视觉服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # AI服务
    ai_services = providers.DependenciesContainer()

    # 视觉服务
    screenshot_service = providers.Singleton(
        # 将在实现时替换为实际的截图服务
        lambda config: None,
        config=config
    )

    ocr_service = providers.Singleton(
        # 将在实现时替换为实际的OCR服务
        lambda config: None,
        config=config
    )

    image_analyzer = providers.Singleton(
        # 将在实现时替换为实际的图像分析服务
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )


class WebContainer(containers.DeclarativeContainer):
    """网页服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # AI服务
    ai_services = providers.DependenciesContainer()

    # 网页服务
    scraper_service = providers.Singleton(
        # 将在实现时替换为实际的网页抓取服务
        lambda config: None,
        config=config
    )

    search_service = providers.Singleton(
        # 将在实现时替换为实际的搜索服务
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )


class WeatherContainer(containers.DeclarativeContainer):
    """天气服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # 天气服务
    weather_client = providers.Singleton(
        # 将在实现时替换为实际的天气客户端
        lambda config: None,
        config=config
    )


class SystemContainer(containers.DeclarativeContainer):
    """系统服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # 系统服务
    app_launcher = providers.Singleton(
        # 将在实现时替换为实际的应用启动器
        lambda config: None,
        config=config
    )

    window_manager = providers.Singleton(
        # 将在实现时替换为实际的窗口管理器
        lambda config: None,
        config=config
    )

    shortcuts_manager = providers.Singleton(
        # 将在实现时替换为实际的快捷键管理器
        lambda config: None,
        config=config
    )


class FilesContainer(containers.DeclarativeContainer):
    """文件服务依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # AI服务
    ai_services = providers.DependenciesContainer()

    # 文件服务
    file_manager = providers.Singleton(
        # 将在实现时替换为实际的文件管理器
        lambda config: None,
        config=config
    )

    document_converter = providers.Singleton(
        # 将在实现时替换为实际的文档转换器
        lambda config: None,
        config=config
    )

    file_analyzer = providers.Singleton(
        # 将在实现时替换为实际的文件分析器
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )


class ToolsContainer(containers.DeclarativeContainer):
    """工具系统依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # 各种服务容器
    ai_services = providers.DependenciesContainer()
    voice_services = providers.DependenciesContainer()
    vision_services = providers.DependenciesContainer()
    web_services = providers.DependenciesContainer()
    weather_services = providers.DependenciesContainer()
    system_services = providers.DependenciesContainer()
    files_services = providers.DependenciesContainer()

    # 工具注册器
    tool_registry = providers.Singleton(
        # 将在实现时替换为实际的工具注册器
        lambda: None
    )


class AgentsContainer(containers.DeclarativeContainer):
    """智能代理依赖容器"""

    # 核心配置
    config = providers.Dependency()

    # 服务容器
    ai_services = providers.DependenciesContainer()
    tools = providers.DependenciesContainer()

    # 代理服务
    intent_classifier = providers.Singleton(
        # 将在实现时替换为实际的意图分类器
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )

    task_planner = providers.Singleton(
        # 将在实现时替换为实际的任务规划器
        lambda config, ai_services: None,
        config=config,
        ai_services=ai_services
    )

    orchestrator = providers.Singleton(
        # 将在实现时替换为实际的编排器
        lambda config, ai_services, tools: None,
        config=config,
        ai_services=ai_services,
        tools=tools
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """应用程序主容器"""

    # 核心配置
    config = providers.Object(settings)

    # 核心容器
    core = providers.Container(CoreContainer)

    # 服务容器
    ai_services = providers.Container(
        AIServicesContainer,
        config=config
    )

    voice_services = providers.Container(
        VoiceContainer,
        config=config,
        ai_services=ai_services
    )

    vision_services = providers.Container(
        VisionContainer,
        config=config,
        ai_services=ai_services
    )

    web_services = providers.Container(
        WebContainer,
        config=config,
        ai_services=ai_services
    )

    weather_services = providers.Container(
        WeatherContainer,
        config=config
    )

    system_services = providers.Container(
        SystemContainer,
        config=config
    )

    files_services = providers.Container(
        FilesContainer,
        config=config,
        ai_services=ai_services
    )

    tools = providers.Container(
        ToolsContainer,
        config=config,
        ai_services=ai_services,
        voice_services=voice_services,
        vision_services=vision_services,
        web_services=web_services,
        weather_services=weather_services,
        system_services=system_services,
        files_services=files_services
    )

    agents = providers.Container(
        AgentsContainer,
        config=config,
        ai_services=ai_services,
        tools=tools
    )


# 全局应用容器
app_container = ApplicationContainer()


# 便捷的依赖注入装饰器
def inject_config(func):
    """注入配置依赖"""
    return inject(func, config=Provide[ApplicationContainer.config])


def inject_ai_services(func):
    """注入AI服务依赖"""
    return inject(func, ai_services=Provide[ApplicationContainer.ai_services])


def inject_voice_services(func):
    """注入语音服务依赖"""
    return inject(func, voice_services=Provide[ApplicationContainer.voice_services])


def inject_vision_services(func):
    """注入视觉服务依赖"""
    return inject(func, vision_services=Provide[ApplicationContainer.vision_services])


def inject_web_services(func):
    """注入网页服务依赖"""
    return inject(func, web_services=Provide[ApplicationContainer.web_services])


def inject_weather_services(func):
    """注入天气服务依赖"""
    return inject(func, weather_services=Provide[ApplicationContainer.weather_services])


def inject_system_services(func):
    """注入系统服务依赖"""
    return inject(func, system_services=Provide[ApplicationContainer.system_services])


def inject_files_services(func):
    """注入文件服务依赖"""
    return inject(func, files_services=Provide[ApplicationContainer.files_services])


def inject_tools(func):
    """注入工具系统依赖"""
    return inject(func, tools=Provide[ApplicationContainer.tools])


def inject_agents(func):
    """注入代理系统依赖"""
    return inject(func, agents=Provide[ApplicationContainer.agents])


# 导出的依赖
__all__ = [
    "ApplicationContainer",
    "app_container",
    "inject_config",
    "inject_ai_services",
    "inject_voice_services",
    "inject_vision_services",
    "inject_web_services",
    "inject_weather_services",
    "inject_system_services",
    "inject_files_services",
    "inject_tools",
    "inject_agents",
    "Provide",
]