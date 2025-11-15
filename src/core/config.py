"""
MCP Floating Ball 核心配置模块

提供统一的应用配置管理，支持环境变量、YAML配置文件等多种配置源。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 文件
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # 如果项目根目录没有，尝试当前工作目录
    load_dotenv()


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    url: str = Field(default="sqlite:///./data/mcp_assistant.db", description="数据库连接URL")
    echo: bool = Field(default=False, description="是否打印SQL语句")
    pool_size: int = Field(default=5, description="连接池大小")
    max_overflow: int = Field(default=10, description="连接池最大溢出")


class RedisSettings(BaseSettings):
    """Redis配置"""
    url: str = Field(default="redis://localhost:6379/0", description="Redis连接URL")
    decode_responses: bool = Field(default=True, description="是否解码响应")
    socket_timeout: int = Field(default=5, description="Socket超时时间")
    socket_connect_timeout: int = Field(default=5, description="Socket连接超时时间")


class KimiSettings(BaseSettings):
    """月之暗面Kimi配置"""
    api_key: Optional[str] = Field(default=None, description="Kimi API密钥")
    base_url: str = Field(default="https://api.moonshot.cn/v1", description="Kimi API基础URL")
    model: str = Field(default="kimi-k2-turbo-preview", description="Kimi模型名称")
    max_tokens: int = Field(default=4096, description="最大生成令牌数")
    temperature: float = Field(default=0.7, description="生成温度", ge=0.0, le=2.0)
    timeout: int = Field(default=30, description="请求超时时间（秒）")

    model_config = SettingsConfigDict(env_prefix="KIMI_")


class DashScopeSettings(BaseSettings):
    """阿里云DashScope配置"""
    access_key_id: Optional[str] = Field(default=None, description="阿里云Access Key ID")
    asr_model: str = Field(default="paraformer-realtime-v2", description="语音识别模型")
    tts_model: str = Field(default="sambert-zhiwei-v1", description="语音合成模型")
    vision_model: str = Field(default="qwen-vl-plus", description="视觉理解模型")
    timeout: int = Field(default=30, description="请求超时时间（秒）")

    model_config = SettingsConfigDict(env_prefix="DASHSCOPE_")


class MetasoSettings(BaseSettings):
    """秘塔AI搜索配置"""
    api_key: Optional[str] = Field(default=None, description="秘塔API密钥")
    timeout: int = Field(default=30, description="请求超时时间（秒）")

    model_config = SettingsConfigDict(env_prefix="METASO_")


class AISettings(BaseSettings):
    """AI服务配置"""
    kimi: KimiSettings = Field(default_factory=KimiSettings)
    dashscope: DashScopeSettings = Field(default_factory=DashScopeSettings)
    metaso: MetasoSettings = Field(default_factory=MetasoSettings)


class VoiceSettings(BaseSettings):
    """语音配置"""
    sample_rate: int = Field(default=16000, description="采样率")
    channels: int = Field(default=1, description="声道数")
    chunk_size: int = Field(default=1024, description="数据块大小")
    format: str = Field(default="int16", description="音频格式")
    wake_word_enabled: bool = Field(default=True, description="是否启用语音唤醒")
    wake_word_sensitivity: float = Field(default=0.5, description="唤醒词灵敏度", ge=0.0, le=1.0)
    wake_word_keyword: str = Field(default="hello", description="唤醒关键词")

    model_config = SettingsConfigDict(env_prefix="VOICE_")


class VisionSettings(BaseSettings):
    """视觉配置"""
    screenshot_format: str = Field(default="png", description="截图格式")
    screenshot_quality: int = Field(default=95, description="截图质量", ge=1, le=100)
    screenshot_max_size: int = Field(default=1920, description="截图最大尺寸")
    ocr_languages: List[str] = Field(default=["chi_sim", "eng"], description="OCR识别语言")
    ocr_confidence_threshold: int = Field(default=60, description="OCR置信度阈值", ge=0, le=100)

    model_config = SettingsConfigDict(env_prefix="VISION_")


class SystemSettings(BaseSettings):
    """系统配置"""
    platform: str = Field(default_factory=lambda: sys.platform, description="操作系统平台")
    max_concurrent_tasks: int = Field(default=10, description="最大并发任务数", ge=1)
    request_timeout: int = Field(default=30, description="请求超时时间（秒）", ge=1)
    max_retries: int = Field(default=3, description="最大重试次数", ge=0)
    retry_delay: float = Field(default=1.0, description="重试延迟（秒）", ge=0.0)

    model_config = SettingsConfigDict(env_prefix="SYSTEM_")


class CacheSettings(BaseSettings):
    """缓存配置"""
    ttl: int = Field(default=3600, description="缓存生存时间（秒）", ge=0)
    max_size: int = Field(default=1000, description="缓存最大条目数", ge=1)
    cleanup_interval: int = Field(default=300, description="清理间隔（秒）", ge=60)

    model_config = SettingsConfigDict(env_prefix="CACHE_")


class FileSettings(BaseSettings):
    """文件处理配置"""
    max_file_size: str = Field(default="50MB", description="最大文件大小")
    allowed_extensions: List[str] = Field(
        default=[".txt", ".docx", ".pdf", ".xlsx", ".png", ".jpg", ".jpeg"],
        description="允许的文件扩展名"
    )
    temp_dir: str = Field(default="./data/temp", description="临时文件目录")

    model_config = SettingsConfigDict(env_prefix="FILE_")

    @validator("temp_dir", pre=True)
    def ensure_temp_dir_exists(cls, v: str) -> str:
        """确保临时目录存在"""
        temp_path = Path(v)
        temp_path.mkdir(parents=True, exist_ok=True)
        return str(temp_path.absolute())


class WeatherSettings(BaseSettings):
    """天气配置"""
    default_city: str = Field(default="北京", description="默认城市")
    update_interval: int = Field(default=1800, description="更新间隔（秒）", ge=300)
    forecast_days: int = Field(default=7, description="预报天数", ge=1, le=15)

    model_config = SettingsConfigDict(env_prefix="WEATHER_")


class GestureSettings(BaseSettings):
    """手势识别配置"""
    detection_confidence: float = Field(default=0.5, description="检测置信度", ge=0.0, le=1.0)
    tracking_confidence: float = Field(default=0.5, description="跟踪置信度", ge=0.0, le=1.0)
    max_hands: int = Field(default=2, description="最大手数", ge=1, le=4)

    model_config = SettingsConfigDict(env_prefix="GESTURE_")


class WebSettings(BaseSettings):
    """网页配置"""
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        description="用户代理字符串"
    )
    timeout: int = Field(default=30, description="请求超时时间（秒）", ge=1)
    max_retries: int = Field(default=3, description="最大重试次数", ge=0)
    headless: bool = Field(default=True, description="是否无头模式")

    model_config = SettingsConfigDict(env_prefix="WEB_")


class LoggingSettings(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="json", description="日志格式")
    file_path: str = Field(default="./data/logs/app.log", description="日志文件路径")
    max_file_size: str = Field(default="10MB", description="日志文件最大大小")
    backup_count: int = Field(default=5, description="日志备份数量", ge=1)
    enable_console: bool = Field(default=True, description="是否启用控制台输出")
    enable_file: bool = Field(default=True, description="是否启用文件输出")
    enable_error_file: bool = Field(default=True, description="是否启用错误日志文件")

    model_config = SettingsConfigDict(env_prefix="LOG_")

    @validator("level", pre=True)
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是以下之一: {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    """主配置类"""

    model_config = SettingsConfigDict(
        env_file=[".env", "config/.env"],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # 应用基本信息
    name: str = Field(default="MCP Floating Ball", description="应用名称")
    version: str = Field(default="1.0.0", description="应用版本")
    description: str = Field(default="现代化AI助手项目", description="应用描述")
    debug: bool = Field(default=False, description="调试模式")
    development: bool = Field(default=False, description="开发模式")
    testing: bool = Field(default=False, description="测试模式")

    # 各模块配置
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ai: AISettings = Field(default_factory=AISettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    weather: WeatherSettings = Field(default_factory=WeatherSettings)
    gesture: GestureSettings = Field(default_factory=GestureSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def load_from_yaml(cls, config_path: Union[str, Path] = "config/settings.yaml") -> "Settings":
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return cls(**config_data)
        except Exception as e:
            # 如果YAML加载失败，返回默认配置
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
            return cls()

    def save_to_yaml(self, config_path: Union[str, Path] = "config/settings.yaml") -> None:
        """保存配置到YAML文件"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.model_dump(exclude_unset=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def ensure_directories(self) -> None:
        """确保所有必要的目录存在"""
        directories = [
            "data/logs",
            "data/cache",
            "data/temp",
            Path(self.files.temp_dir),
            Path(self.logging.file_path).parent,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_ai_config(self, provider: str) -> Dict[str, Any]:
        """获取指定AI提供商的配置"""
        if provider == "kimi":
            return self.ai.kimi.model_dump()
        elif provider == "dashscope":
            return self.ai.dashscope.model_dump()
        elif provider == "metaso":
            return self.ai.metaso.model_dump()
        else:
            raise ValueError(f"未知的AI提供商: {provider}")

    def validate_api_keys(self) -> Dict[str, bool]:
        """验证所有API密钥是否已配置"""
        results = {}

        # 检查Kimi API密钥
        results["kimi"] = bool(self.ai.kimi.api_key and self.ai.kimi.api_key != "your_moonshot_api_key_here")

        # 检查DashScope API密钥
        results["dashscope"] = bool(self.ai.dashscope.access_key_id and self.ai.dashscope.access_key_id != "your_alibaba_api_key_here")

        # 检查秘塔API密钥
        results["metaso"] = bool(self.ai.metaso.api_key and self.ai.metaso.api_key != "your_metaso_api_key_here")

        return results


# 延迟加载配置实例
def _get_settings():
    """获取配置实例，支持延迟加载和错误处理"""
    try:
        # 暂时跳过YAML配置，直接使用环境变量
        return Settings()
    except Exception as e:
        print(f"⚠️ 配置加载失败，使用默认配置: {e}")
        return Settings()

# 全局配置实例（延迟加载）
settings = None

def get_settings():
    """获取全局配置实例"""
    global settings
    if settings is None:
        settings = _get_settings()
        if settings:
            settings.ensure_directories()
    return settings

# 导出配置实例
__all__ = [
    "Settings",
    "settings",
    "DatabaseSettings",
    "RedisSettings",
    "KimiSettings",
    "DashScopeSettings",
    "MetasoSettings",
    "AISettings",
    "VoiceSettings",
    "VisionSettings",
    "SystemSettings",
    "CacheSettings",
    "FileSettings",
    "WeatherSettings",
    "GestureSettings",
    "WebSettings",
    "LoggingSettings",
]