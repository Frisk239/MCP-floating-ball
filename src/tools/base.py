"""
MCP Floating Ball 工具基类模块

定义工具系统的基础接口和抽象类，为所有工具提供统一的实现规范。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
import inspect
import time

from ..core.logging import get_logger
from ..core.exceptions import ToolError, ToolExecutionError, ValidationError

logger = get_logger(__name__)


class ToolCategory(str, Enum):
    """工具类别枚举"""
    WEATHER = "weather"
    WEB = "web"
    VISION = "vision"
    SYSTEM = "system"
    FILE = "file"
    AI = "ai"
    VOICE = "voice"
    MEDIA = "media"
    NETWORK = "network"
    UTILITY = "utility"


class ParameterType(str, Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    URL = "url"


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str = Field(..., description="参数名称")
    type: ParameterType = Field(..., description="参数类型")
    description: str = Field(..., description="参数描述")
    required: bool = Field(default=True, description="是否必需")
    default: Optional[Any] = Field(default=None, description="默认值")
    enum: Optional[List[Any]] = Field(default=None, description="枚举值")
    min_value: Optional[Union[int, float]] = Field(default=None, description="最小值")
    max_value: Optional[Union[int, float]] = Field(default=None, description="最大值")
    pattern: Optional[str] = Field(default=None, description="正则表达式模式")


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool = Field(..., description="执行是否成功")
    data: Optional[Any] = Field(default=None, description="返回数据")
    message: Optional[str] = Field(default=None, description="执行消息")
    error: Optional[str] = Field(default=None, description="错误信息")
    execution_time: float = Field(..., description="执行时间（秒）")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class ToolMetadata(BaseModel):
    """工具元数据"""
    name: str = Field(..., description="工具名称")
    display_name: str = Field(..., description="显示名称")
    description: str = Field(..., description="工具描述")
    category: ToolCategory = Field(..., description="工具类别")
    version: str = Field(default="1.0.0", description="版本号")
    author: str = Field(default="MCP Team", description="作者")
    tags: List[str] = Field(default_factory=list, description="标签")
    parameters: List[ToolParameter] = Field(default_factory=list, description="参数列表")
    examples: List[str] = Field(default_factory=list, description="使用示例")
    dependencies: List[str] = Field(default_factory=list, description="依赖项")
    async_execution: bool = Field(default=False, description="是否异步执行")
    requires_internet: bool = Field(default=False, description="是否需要网络")
    requires_permission: bool = Field(default=False, description="是否需要特殊权限")


class BaseTool(ABC):
    """工具基类"""

    def __init__(self):
        self.metadata = self.get_metadata()
        self._logger = get_logger(f"tool.{self.metadata.name}")
        self._execution_count = 0
        self._last_execution_time = 0
        self._total_execution_time = 0

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """
        获取工具元数据

        Returns:
            ToolMetadata: 工具元数据对象
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        pass

    async def execute_async(self, **kwargs) -> ToolResult:
        """
        异步执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        if self.metadata.async_execution:
            return await self._execute_async_internal(**kwargs)
        else:
            # 如果工具不支持异步，在线程池中执行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, **kwargs)

    async def _execute_async_internal(self, **kwargs) -> ToolResult:
        """
        内部异步执行方法，子类可以重写

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        return self.execute(**kwargs)

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        验证参数

        Args:
            parameters: 参数字典

        Raises:
            ValidationError: 参数验证失败
        """
        for param in self.metadata.parameters:
            param_name = param.name
            param_value = parameters.get(param_name)

            # 检查必需参数
            if param.required and param_value is None:
                raise ValidationError(
                    f"缺少必需参数: {param_name}",
                    field=param_name,
                    value=param_value
                )

            # 跳过未提供且非必需的参数
            if param_value is None and not param.required:
                continue

            # 类型验证
            self._validate_parameter_type(param, param_value)

            # 枚举值验证
            if param.enum and param_value not in param.enum:
                raise ValidationError(
                    f"参数 {param_name} 的值必须是以下之一: {param.enum}",
                    field=param_name,
                    value=param_value
                )

            # 数值范围验证
            if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                if param.min_value is not None and param_value < param.min_value:
                    raise ValidationError(
                        f"参数 {param_name} 的值不能小于 {param.min_value}",
                        field=param_name,
                        value=param_value
                    )
                if param.max_value is not None and param_value > param.max_value:
                    raise ValidationError(
                        f"参数 {param_name} 的值不能大于 {param.max_value}",
                        field=param_name,
                        value=param_value
                    )

            # 正则表达式验证
            if param.pattern and isinstance(param_value, str):
                import re
                if not re.match(param.pattern, param_value):
                    raise ValidationError(
                        f"参数 {param_name} 的值不匹配模式 {param.pattern}",
                        field=param_name,
                        value=param_value
                    )

    def _validate_parameter_type(self, param: ToolParameter, value: Any) -> None:
        """验证参数类型"""
        type_mapping = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: (list, tuple),
            ParameterType.OBJECT: dict,
        }

        expected_types = type_mapping.get(param.type)
        if expected_types and not isinstance(value, expected_types):
            raise ValidationError(
                f"参数 {param.name} 的类型应为 {param.type.value}，实际为 {type(value).__name__}",
                field=param.name,
                value=value
            )

    def _wrap_execution(self, func: Callable, **kwargs) -> ToolResult:
        """
        包装工具执行，添加日志和异常处理

        Args:
            func: 执行函数
            **kwargs: 参数

        Returns:
            ToolResult: 执行结果
        """
        start_time = time.time()
        self._execution_count += 1

        try:
            # 验证参数
            self.validate_parameters(kwargs)

            # 记录执行开始
            self._logger.info(f"开始执行工具: {self.metadata.name}", kwargs=kwargs)

            # 执行工具
            result = func(**kwargs)

            # 确保返回类型正确
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=True,
                    data=result,
                    message=f"工具 {self.metadata.name} 执行成功"
                )

            execution_time = time.time() - start_time
            self._last_execution_time = execution_time
            self._total_execution_time += execution_time

            result.execution_time = execution_time
            self._logger.info(
                f"工具执行成功: {self.metadata.name}",
                execution_time=execution_time,
                success=result.success
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)

            self._logger.error(
                f"工具执行失败: {self.metadata.name}",
                error=error_message,
                execution_time=execution_time
            )

            return ToolResult(
                success=False,
                error=error_message,
                execution_time=execution_time,
                message=f"工具 {self.metadata.name} 执行失败: {error_message}"
            )

    def execute_with_validation(self, **kwargs) -> ToolResult:
        """
        带验证的执行

        Args:
            **kwargs: 参数

        Returns:
            ToolResult: 执行结果
        """
        return self._wrap_execution(self.execute, **kwargs)

    async def execute_async_with_validation(self, **kwargs) -> ToolResult:
        """
        带验证的异步执行

        Args:
            **kwargs: 参数

        Returns:
            ToolResult: 执行结果
        """
        return await self._wrap_execution(self.execute_async, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取工具统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        avg_execution_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0 else 0
        )

        return {
            "name": self.metadata.name,
            "execution_count": self._execution_count,
            "last_execution_time": self._last_execution_time,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_execution_time,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._execution_count = 0
        self._last_execution_time = 0
        self._total_execution_time = 0

    def __str__(self) -> str:
        return f"Tool({self.metadata.name})"

    def __repr__(self) -> str:
        return f"Tool(name='{self.metadata.name}', category='{self.metadata.category}')"


class FunctionTool(BaseTool):
    """函数工具 - 将普通函数转换为工具"""

    def __init__(
        self,
        func: Callable,
        metadata: ToolMetadata,
        async_func: Optional[Callable] = None
    ):
        """
        初始化函数工具

        Args:
            func: 同步执行函数
            metadata: 工具元数据
            async_func: 异步执行函数（可选）
        """
        self._func = func
        self._async_func = async_func
        super().__init__()

    def get_metadata(self) -> ToolMetadata:
        return self.metadata

    def execute(self, **kwargs) -> ToolResult:
        try:
            result = self._func(**kwargs)
            return ToolResult(
                success=True,
                data=result,
                message=f"工具 {self.metadata.name} 执行成功"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message=f"工具 {self.metadata.name} 执行失败"
            )

    async def execute_async(self, **kwargs) -> ToolResult:
        if self._async_func:
            try:
                result = await self._async_func(**kwargs)
                return ToolResult(
                    success=True,
                    data=result,
                    message=f"工具 {self.metadata.name} 执行成功"
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    message=f"工具 {self.metadata.name} 执行失败"
                )
        else:
            return await super().execute_async(**kwargs)


class ToolDecorator:
    """工具装饰器，用于将函数转换为工具"""

    @staticmethod
    def tool(
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Optional[List[Dict[str, Any]]] = None,
        examples: Optional[List[str]] = None,
        async_execution: bool = False,
        **metadata_kwargs
    ):
        """
        工具装饰器

        Args:
            name: 工具名称
            description: 工具描述
            category: 工具类别
            parameters: 参数定义列表
            examples: 使用示例
            async_execution: 是否异步执行
            **metadata_kwargs: 其他元数据参数
        """
        def decorator(func: Callable) -> FunctionTool:
            # 转换参数定义
            tool_parameters = []
            if parameters:
                for param in parameters:
                    tool_parameters.append(ToolParameter(**param))

            # 创建工具元数据
            metadata = ToolMetadata(
                name=name,
                display_name=name.replace("_", " ").title(),
                description=description,
                category=category,
                parameters=tool_parameters,
                examples=examples or [],
                async_execution=async_execution,
                **metadata_kwargs
            )

            # 检查是否为异步函数
            async_func = None
            if inspect.iscoroutinefunction(func):
                async_func = func
                # 创建同步包装器
                def sync_wrapper(**kwargs):
                    # 这里需要在线程池中运行异步函数
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    return loop.run_until_complete(async_func(**kwargs))

                func = sync_wrapper

            return FunctionTool(func, metadata, async_func)

        return decorator


# 导出的类型和函数
__all__ = [
    "BaseTool",
    "FunctionTool",
    "ToolDecorator",
    "ToolCategory",
    "ParameterType",
    "ToolParameter",
    "ToolResult",
    "ToolMetadata",
    "tool",  # 装饰器别名
]

# 创建装饰器别名
tool = ToolDecorator.tool