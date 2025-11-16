"""
MCP Floating Ball - 配置管理器

提供统一的应用配置管理接口，兼容现有代码的导入方式。
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

# 导入现有的配置系统
from src.core.config import settings

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器类"""

    def __init__(self):
        """初始化配置管理器"""
        self._settings = settings
        self._cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键 (如 "assistant.max_history_size")
            default: 默认值

        Returns:
            配置值
        """
        try:
            # 尝试从缓存获取
            if key in self._cache:
                return self._cache[key]

            # 处理嵌套键
            if "." in key:
                parts = key.split(".")
                value = self._settings
                for part in parts:
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = default
                        break
            else:
                # 直接获取属性
                if hasattr(self._settings, key):
                    value = getattr(self._settings, key)
                else:
                    value = default

            # 缓存结果
            self._cache[key] = value
            return value

        except Exception as e:
            self.logger.warning(f"获取配置 '{key}' 失败: {e}，使用默认值 {default}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """
        设置配置值

        Args:
            key: 配置键
            value: 配置值

        Returns:
            是否设置成功
        """
        try:
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
            else:
                # 添加到缓存
                self._cache[key] = value

            self.logger.debug(f"设置配置 '{key}' = {value}")
            return True

        except Exception as e:
            self.logger.error(f"设置配置 '{key}' 失败: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        try:
            result = {}
            # 获取所有属性
            for attr_name in dir(self._settings):
                if not attr_name.startswith("_") and not callable(getattr(self._settings, attr_name)):
                    try:
                        attr_value = getattr(self._settings, attr_name)
                        if not isinstance(attr_value, (type, logging.Logger)):
                            result[attr_name] = attr_value
                    except Exception:
                        continue

            # 添加缓存的配置
            result.update(self._cache)
            return result

        except Exception as e:
            self.logger.error(f"获取所有配置失败: {e}")
            return {}

    def reload(self) -> bool:
        """
        重新加载配置

        Returns:
            是否重载成功
        """
        try:
            self._cache.clear()
            self.logger.info("配置已重新加载")
            return True

        except Exception as e:
            self.logger.error(f"重新加载配置失败: {e}")
            return False

    def load_config(self, config_path: Optional[str] = None) -> bool:
        """
        加载外部配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            是否加载成功
        """
        try:
            if config_path:
                config_file = Path(config_path)
                if config_file.exists():
                    # 这里可以扩展支持加载额外的配置文件
                    self.logger.info(f"加载外部配置文件: {config_file}")
                    return True
                else:
                    self.logger.warning(f"配置文件不存在: {config_file}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return False

    def get_assistant_config(self) -> Dict[str, Any]:
        """获取助手相关配置"""
        return {
            "max_history_size": self.get("assistant.max_history_size", 1000),
            "voice_enabled": self.get("voice.enabled", False),
            "vision_enabled": self.get("vision.enabled", True),
            "auto_start_voice": self.get("voice.auto_start", False),
            "show_timestamps": self.get("console.show_timestamps", False),
            "wake_word": self.get("voice.wake_word", "hello")
        }

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """获取工具相关配置"""
        try:
            if hasattr(self._settings, f"{tool_name}_settings"):
                tool_settings = getattr(self._settings, f"{tool_name}_settings")
                if hasattr(tool_settings, "model_dump"):
                    return tool_settings.model_dump()
                elif isinstance(tool_settings, dict):
                    return tool_settings
                else:
                    return {"settings": tool_settings}
            return {}
        except Exception as e:
            self.logger.warning(f"获取工具 '{tool_name}' 配置失败: {e}")
            return {}

    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """获取API配置"""
        try:
            api_keys = self.get("api_keys", {})
            if isinstance(api_keys, dict) and provider in api_keys:
                return {"api_key": api_keys[provider]}

            # 尝试从环境变量获取
            import os
            env_key = f"{provider.upper()}_API_KEY"
            env_value = os.getenv(env_key)
            if env_value:
                return {"api_key": env_value}

            return {}
        except Exception as e:
            self.logger.warning(f"获取API '{provider}' 配置失败: {e}")
            return {}

    def validate_config(self) -> Dict[str, Any]:
        """验证配置完整性"""
        issues = []
        warnings = []

        try:
            # 检查必要配置
            required_configs = [
                "assistant.max_history_size",
                "voice.enabled",
                "vision.enabled"
            ]

            for config in required_configs:
                value = self.get(config)
                if value is None:
                    issues.append(f"缺少必要配置: {config}")

            # 检查API配置
            if self.get("ai_service.enabled", True):
                ai_providers = ["kimi", "dashscope", "metaso"]
                for provider in ai_providers:
                    api_config = self.get_api_config(provider)
                    if not api_config.get("api_key"):
                        warnings.append(f"AI服务 '{provider}' 未配置API密钥")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings
            }

        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return {
                "valid": False,
                "issues": [f"配置验证过程出错: {e}"],
                "warnings": []
            }


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    获取全局配置管理器实例

    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reload_config() -> bool:
    """重新加载配置"""
    global _config_manager
    if _config_manager:
        return _config_manager.reload()
    return True


# 导出
__all__ = [
    "ConfigManager",
    "get_config_manager",
    "reload_config"
]