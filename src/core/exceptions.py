"""
MCP Floating Ball 自定义异常模块

定义项目中使用的所有自定义异常类，提供统一的错误处理机制。
"""

from typing import Any, Dict, Optional, Union


class MCPFloatingBallError(Exception):
    """MCP Floating Ball 基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(MCPFloatingBallError):
    """配置错误"""
    pass


class APIError(MCPFloatingBallError):
    """API调用错误"""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.provider = provider
        self.status_code = status_code
        self.response_data = response_data or {}

        # 更新详细信息
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        if response_data:
            details["response_data"] = response_data

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AuthenticationError(APIError):
    """认证错误"""
    pass


class RateLimitError(APIError):
    """速率限制错误"""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        if retry_after:
            details = kwargs.get("details", {})
            details["retry_after"] = retry_after
            kwargs["details"] = details
        super().__init__(message, **kwargs)


class QuotaExceededError(APIError):
    """配额超限错误"""
    pass


class AIServiceError(MCPFloatingBallError):
    """AI服务错误"""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        self.service = service
        self.model = model

        details = kwargs.get("details", {})
        if service:
            details["service"] = service
        if model:
            details["model"] = model

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class VoiceError(MCPFloatingBallError):
    """语音服务错误"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation

        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class VisionError(MCPFloatingBallError):
    """视觉处理错误"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation

        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class WebScrapingError(MCPFloatingBallError):
    """网页抓取错误"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        **kwargs
    ):
        self.url = url

        details = kwargs.get("details", {})
        if url:
            details["url"] = url

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class SystemOperationError(MCPFloatingBallError):
    """系统操作错误"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        target: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation
        self.target = target

        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if target:
            details["target"] = target

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class FileOperationError(MCPFloatingBallError):
    """文件操作错误"""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        self.file_path = file_path
        self.operation = operation

        details = kwargs.get("details", {})
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ToolError(MCPFloatingBallError):
    """工具执行错误"""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.tool_name = tool_name
        self.tool_args = tool_args or {}

        details = kwargs.get("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        if tool_args:
            details["tool_args"] = tool_args

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ToolNotFoundError(ToolError):
    """工具未找到错误"""
    pass


class ToolExecutionError(ToolError):
    """工具执行错误"""
    pass


class ValidationError(MCPFloatingBallError):
    """验证错误"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        self.field = field
        self.value = value

        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class NetworkError(MCPFloatingBallError):
    """网络错误"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        self.url = url
        self.timeout = timeout

        details = kwargs.get("details", {})
        if url:
            details["url"] = url
        if timeout:
            details["timeout"] = timeout

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class TimeoutError(NetworkError):
    """超时错误"""
    pass


class ConnectionError(NetworkError):
    """连接错误"""
    pass


class DependencyError(MCPFloatingBallError):
    """依赖错误"""
    pass


class PermissionError(MCPFloatingBallError):
    """权限错误"""

    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        self.resource = resource
        self.action = action

        details = kwargs.get("details", {})
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class WeatherError(MCPFloatingBallError):
    """天气服务错误"""

    def __init__(
        self,
        message: str,
        city: Optional[str] = None,
        **kwargs
    ):
        self.city = city

        details = kwargs.get("details", {})
        if city:
            details["city"] = city

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class GestureError(MCPFloatingBallError):
    """手势识别错误"""

    def __init__(
        self,
        message: str,
        gesture_type: Optional[str] = None,
        **kwargs
    ):
        self.gesture_type = gesture_type

        details = kwargs.get("details", {})
        if gesture_type:
            details["gesture_type"] = gesture_type

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class ClipboardError(MCPFloatingBallError):
    """剪贴板错误"""
    pass


class WindowError(MCPFloatingBallError):
    """窗口操作错误"""

    def __init__(
        self,
        message: str,
        window_title: Optional[str] = None,
        window_pid: Optional[int] = None,
        **kwargs
    ):
        self.window_title = window_title
        self.window_pid = window_pid

        details = kwargs.get("details", {})
        if window_title:
            details["window_title"] = window_title
        if window_pid:
            details["window_pid"] = window_pid

        kwargs["details"] = details
        super().__init__(message, **kwargs)


class AssistantError(MCPFloatingBallError):
    """AI助手错误"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation
        self.component = component

        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        if component:
            details["component"] = component

        kwargs["details"] = details
        super().__init__(message, **kwargs)


# 异常处理工具函数
def handle_exception(
    exception: Exception,
    default_message: str = "发生未知错误",
    error_code: Optional[str] = None
) -> MCPFloatingBallError:
    """
    将任意异常转换为MCPFloatingBallError

    Args:
        exception: 原始异常
        default_message: 默认错误消息
        error_code: 错误代码

    Returns:
        MCPFloatingBallError: 转换后的异常
    """
    if isinstance(exception, MCPFloatingBallError):
        return exception

    # 根据异常类型创建相应的错误
    if isinstance(exception, FileNotFoundError):
        return FileOperationError(
            message=f"文件未找到: {str(exception)}",
            error_code=error_code,
            cause=exception
        )
    elif isinstance(exception, PermissionError):
        return PermissionError(
            message=f"权限不足: {str(exception)}",
            error_code=error_code,
            cause=exception
        )
    elif isinstance(exception, ConnectionError):
        return ConnectionError(
            message=f"连接错误: {str(exception)}",
            error_code=error_code,
            cause=exception
        )
    elif isinstance(exception, TimeoutError):
        return TimeoutError(
            message=f"操作超时: {str(exception)}",
            error_code=error_code,
            cause=exception
        )
    elif isinstance(exception, ValueError):
        return ValidationError(
            message=f"数据验证错误: {str(exception)}",
            error_code=error_code,
            cause=exception
        )
    else:
        return MCPFloatingBallError(
            message=f"{default_message}: {str(exception)}",
            error_code=error_code,
            cause=exception
        )


def create_error_response(
    error: Union[Exception, MCPFloatingBallError],
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    创建错误响应字典

    Args:
        error: 异常对象
        include_traceback: 是否包含堆栈跟踪

    Returns:
        Dict[str, Any]: 错误响应字典
    """
    if isinstance(error, MCPFloatingBallError):
        response = error.to_dict()
    else:
        mcp_error = handle_exception(error)
        response = mcp_error.to_dict()

    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()

    response["success"] = False
    response["timestamp"] = __import__("datetime").datetime.now().isoformat()

    return response


# 导出的异常类
__all__ = [
    # 基础异常
    "MCPFloatingBallError",

    # 配置和API异常
    "ConfigurationError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "QuotaExceededError",

    # 服务异常
    "AIServiceError",
    "VoiceError",
    "VisionError",
    "WebScrapingError",
    "SystemOperationError",
    "FileOperationError",
    "WeatherError",
    "GestureError",
    "ClipboardError",
    "WindowError",
    "AssistantError",

    # 工具异常
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",

    # 通用异常
    "ValidationError",
    "NetworkError",
    "TimeoutError",
    "ConnectionError",
    "DependencyError",
    "PermissionError",

    # 工具函数
    "handle_exception",
    "create_error_response",
]