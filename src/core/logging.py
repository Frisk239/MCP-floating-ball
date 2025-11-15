"""
MCP Floating Ball 日志模块

使用Loguru提供现代化的结构化日志记录功能，支持多种输出格式和日志级别。
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from .config import LoggingSettings


class LoggerManager:
    """日志管理器"""

    def __init__(self):
        self._configured = False
        self._handlers = {}

    def configure(
        self,
        config: Optional[LoggingSettings] = None,
        log_config_file: Optional[Union[str, Path]] = None
    ) -> None:
        """
        配置日志系统

        Args:
            config: 日志配置对象
            log_config_file: 日志配置文件路径
        """
        if self._configured:
            return

        if config is None:
            # 使用默认配置，避免循环依赖
            config = LoggingSettings(
                level="INFO",
                format="text",
                file_path="data/logs/app.log",
                max_file_size="10 MB",
                backup_count=5,
                enable_console=True
            )

        # 移除默认处理器
        logger.remove()

        # 确保日志目录存在
        log_file_path = Path(config.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 根据配置添加处理器
        if config.enable_console:
            if config.format.lower() == "json":
                self._add_json_handler(config)
            else:
                self._add_console_handler(config)

        if config.enable_file:
            self._add_file_handler(config)

        if config.enable_error_file:
            self._add_error_file_handler(config)

        self._configured = True

    def _add_console_handler(self, config: LoggingSettings) -> None:
        """添加控制台处理器"""
        handler_id = logger.add(
            sys.stdout,
            level=config.level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        self._handlers["console"] = handler_id

    def _add_json_handler(self, config: LoggingSettings) -> None:
        """添加JSON格式控制台处理器"""
        handler_id = logger.add(
            sys.stdout,
            level=config.level,
            format="{message}",  # JSON格式将在序列化时处理
            serialize=True,
            backtrace=True,
            diagnose=True
        )
        self._handlers["console_json"] = handler_id

    def _add_file_handler(self, config: LoggingSettings) -> None:
        """添加文件处理器"""
        log_file = Path(config.file_path)
        handler_id = logger.add(
            str(log_file),
            level=config.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | "
                   "{level: <8} | "
                   "{name}:{function}:{line} | "
                   "{message}",
            rotation=self._parse_rotation(config.max_file_size),
            retention=self._parse_retention(config.backup_count),
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
        self._handlers["file"] = handler_id

    def _add_error_file_handler(self, config: LoggingSettings) -> None:
        """添加错误文件处理器"""
        error_log_file = Path(config.file_path).parent / "error.log"
        handler_id = logger.add(
            str(error_log_file),
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | "
                   "{level: <8} | "
                   "{name}:{function}:{line} | "
                   "{message}\n"
                   "Exception: {exception}\n",
            rotation=self._parse_rotation(config.max_file_size),
            retention=self._parse_retention(config.backup_count * 2),  # 错误日志保留更久
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
        self._handlers["error_file"] = handler_id

    def _parse_rotation(self, size_str: str) -> str:
        """解析文件轮转配置为Loguru格式"""
        if not size_str:
            return "10 MB"  # 默认值

        size_str = str(size_str).upper().strip()

        # 支持的格式: "10 MB", "10MB", "1 GB", "500KB" 等
        if size_str.endswith("KB") or size_str.endswith("K"):
            size = int(size_str.rstrip("KB").rstrip("K").strip())
            return f"{size} KB"
        elif size_str.endswith("MB") or size_str.endswith("M"):
            size = int(size_str.rstrip("MB").rstrip("M").strip())
            return f"{size} MB"
        elif size_str.endswith("GB") or size_str.endswith("G"):
            size = int(size_str.rstrip("GB").rstrip("G").strip())
            return f"{size} GB"
        elif size_str.endswith("B"):
            # 纯字节数
            size = int(size_str.rstrip("B").strip())
            return f"{size} bytes"
        else:
            # 尝试解析为数字，默认为MB
            try:
                size = int(size_str)
                if size < 1024:
                    return f"{size} MB"
                else:
                    return f"{size // 1024} GB"
            except ValueError:
                return "10 MB"  # 解析失败时的默认值

    def _parse_retention(self, count: int) -> str:
        """解析日志保留配置为Loguru格式"""
        if count <= 0:
            return "7 days"  # 默认保留7天

        # 如果是较小的数字，理解为天数；较大的数字理解为文件数量
        if count <= 30:
            return f"{count} days"
        elif count <= 365:
            return f"{count // 30} weeks" if count % 30 == 0 else f"{count} days"
        else:
            return f"{count // 365} years" if count % 365 == 0 else f"{count} days"

    def get_logger(self, name: str) -> Any:
        """
        获取指定名称的日志记录器

        Args:
            name: 日志记录器名称

        Returns:
            日志记录器对象
        """
        return logger.bind(name=name)

    def add_custom_handler(
        self,
        sink: Union[str, Path, Any],
        level: str = "INFO",
        format_string: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        添加自定义处理器

        Args:
            sink: 输出目标
            level: 日志级别
            format_string: 格式字符串
            **kwargs: 其他参数

        Returns:
            处理器ID
        """
        if format_string is None:
            format_string = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"

        handler_id = logger.add(
            sink,
            level=level,
            format=format_string,
            **kwargs
        )
        return handler_id

    def remove_handler(self, handler_id: str) -> None:
        """
        移除处理器

        Args:
            handler_id: 处理器ID
        """
        logger.remove(handler_id)

    def set_level(self, level: str) -> None:
        """
        设置日志级别

        Args:
            level: 日志级别
        """
        # 更新所有处理器的级别
        for handler_id in self._handlers.values():
            # Loguru不支持动态修改级别，需要重新配置
            pass

    def configure_from_yaml(self, config_file: Union[str, Path]) -> None:
        """
        从YAML配置文件配置日志

        Args:
            config_file: 配置文件路径
        """
        import yaml

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"日志配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # TODO: 实现从YAML配置文件的详细解析
        logger.info(f"从配置文件加载日志配置: {config_path}")


# 全局日志管理器实例
logger_manager = LoggerManager()


def setup_logging(
    config: Optional[LoggingSettings] = None,
    config_file: Optional[Union[str, Path]] = None
) -> None:
    """
    设置日志系统

    Args:
        config: 日志配置
        config_file: 配置文件路径
    """
    logger_manager.configure(config, config_file)


def get_logger(name: str) -> Any:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器对象
    """
    return logger_manager.get_logger(name)


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None) -> None:
    """
    记录函数调用

    Args:
        func_name: 函数名称
        args: 位置参数
        kwargs: 关键字参数
    """
    logger.debug(f"调用函数: {func_name}, args: {args}, kwargs: {kwargs or {}}")


def log_tool_execution(tool_name: str, args: dict, result: Any = None, error: Exception = None) -> None:
    """
    记录工具执行

    Args:
        tool_name: 工具名称
        args: 工具参数
        result: 执行结果
        error: 错误信息
    """
    if error:
        logger.error(f"工具执行失败: {tool_name}, 参数: {args}, 错误: {error}")
    else:
        logger.info(f"工具执行成功: {tool_name}, 参数: {args}")


def log_api_call(
    provider: str,
    endpoint: str,
    method: str = "POST",
    status_code: Optional[int] = None,
    response_time: Optional[float] = None,
    error: Optional[Exception] = None
) -> None:
    """
    记录API调用

    Args:
        provider: API提供商
        endpoint: API端点
        method: HTTP方法
        status_code: 响应状态码
        response_time: 响应时间（毫秒）
        error: 错误信息
    """
    log_data = {
        "provider": provider,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time": response_time
    }

    if error:
        log_data["error"] = str(error)
        logger.error(f"API调用失败: {log_data}")
    else:
        logger.info(f"API调用成功: {log_data}")


def log_user_interaction(
    user_id: str,
    interaction_type: str,
    content: str,
    response: Optional[str] = None
) -> None:
    """
    记录用户交互

    Args:
        user_id: 用户ID
        interaction_type: 交互类型（如：voice, text, gesture）
        content: 交互内容
        response: 系统响应
    """
    log_data = {
        "user_id": user_id,
        "interaction_type": interaction_type,
        "content": content,
        "response": response
    }
    logger.info(f"用户交互: {log_data}")


class LoggerAdapter:
    """日志适配器，用于特定的模块或组件"""

    def __init__(self, name: str, extra_context: Optional[Dict[str, Any]] = None):
        """
        初始化日志适配器

        Args:
            name: 日志记录器名称
            extra_context: 额外的上下文信息
        """
        self.logger = get_logger(name)
        self.extra_context = extra_context or {}

    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        self.logger.debug(message, **self.extra_context, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        self.logger.info(message, **self.extra_context, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        self.logger.warning(message, **self.extra_context, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        self.logger.error(message, **self.extra_context, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        self.logger.critical(message, **self.extra_context, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """异常日志"""
        self.logger.exception(message, **self.extra_context, **kwargs)


# 兼容标准库logging接口
class StandardLoggerAdapter:
    """标准库日志适配器"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)

    def debug(self, message: str, *args, **kwargs) -> None:
        self.logger.debug(message.format(*args, **kwargs) if args else message)

    def info(self, message: str, *args, **kwargs) -> None:
        self.logger.info(message.format(*args, **kwargs) if args else message)

    def warning(self, message: str, *args, **kwargs) -> None:
        self.logger.warning(message.format(*args, **kwargs) if args else message)

    def error(self, message: str, *args, **kwargs) -> None:
        self.logger.error(message.format(*args, **kwargs) if args else message)

    def critical(self, message: str, *args, **kwargs) -> None:
        self.logger.critical(message.format(*args, **kwargs) if args else message)

    def exception(self, message: str, *args, **kwargs) -> None:
        self.logger.exception(message.format(*args, **kwargs) if args else message)


# 初始化日志系统
def init_logging() -> None:
    """初始化日志系统"""
    try:
        setup_logging()
        logger.info("日志系统初始化完成")
    except Exception as e:
        # 如果日志系统初始化失败，使用标准库logging
        logging.basicConfig(level=logging.INFO)
        logging.error(f"日志系统初始化失败: {e}")


# 导出的接口
__all__ = [
    "logger",
    "LoggerManager",
    "logger_manager",
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    "StandardLoggerAdapter",
    "init_logging",
    "log_function_call",
    "log_tool_execution",
    "log_api_call",
    "log_user_interaction",
]