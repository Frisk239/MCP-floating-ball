"""
MCP Floating Ball 工具注册器模块

提供工具的注册、发现、调用和管理功能，支持36个工具函数的统一管理。
"""

from typing import Any, Dict, List, Optional, Type, Union, Callable
from collections import defaultdict
import asyncio
import inspect
import json

from .base import (
    BaseTool, ToolCategory, ToolResult, ToolMetadata,
    ParameterType, ToolParameter, FunctionTool, tool
)
from ..core.logging import get_logger
from ..core.exceptions import (
    ToolError, ToolNotFoundError, ToolExecutionError,
    ValidationError
)

logger = get_logger(__name__)


class ToolRegistry:
    """工具注册器"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._category_tools: Dict[ToolCategory, List[str]] = defaultdict(list)
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        self._logger = get_logger(f"registry")

    def register(self, tool: BaseTool) -> None:
        """
        注册工具

        Args:
            tool: 工具实例

        Raises:
            ToolError: 工具注册失败
        """
        try:
            metadata = tool.get_metadata()
            tool_name = metadata.name

            # 检查工具名称是否已存在
            if tool_name in self._tools:
                self._logger.warning(f"工具 {tool_name} 已存在，将被覆盖")
                self.unregister(tool_name)

            # 注册工具
            self._tools[tool_name] = tool
            self._category_tools[metadata.category].append(tool_name)
            self._tool_stats[tool_name] = tool.get_stats()

            self._logger.info(f"工具注册成功: {tool_name} ({metadata.category.value})")

        except Exception as e:
            error_msg = f"注册工具失败: {str(e)}"
            self._logger.error(error_msg)
            raise ToolError(error_msg)

    def register_function(
        self,
        func: Callable,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Optional[List[Dict[str, Any]]] = None,
        examples: Optional[List[str]] = None,
        async_execution: bool = False,
        **metadata_kwargs
    ) -> FunctionTool:
        """
        注册函数为工具

        Args:
            func: 函数对象
            name: 工具名称
            description: 工具描述
            category: 工具类别
            parameters: 参数定义
            examples: 使用示例
            async_execution: 是否异步执行
            **metadata_kwargs: 其他元数据

        Returns:
            FunctionTool: 创建的工具对象
        """
        # 使用装饰器创建工具
        tool_decorator = tool(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            examples=examples,
            async_execution=async_execution,
            **metadata_kwargs
        )

        tool_obj = tool_decorator(func)
        self.register(tool_obj)

        return tool_obj

    def unregister(self, tool_name: str) -> None:
        """
        注销工具

        Args:
            tool_name: 工具名称
        """
        if tool_name not in self._tools:
            self._logger.warning(f"工具 {tool_name} 不存在")
            return

        tool = self._tools[tool_name]
        metadata = tool.get_metadata()

        # 移除工具
        del self._tools[tool_name]
        self._category_tools[metadata.category].remove(tool_name)
        if tool_name in self._tool_stats:
            del self._tool_stats[tool_name]

        self._logger.info(f"工具注销成功: {tool_name}")

    def get_tool(self, tool_name: str) -> BaseTool:
        """
        获取工具

        Args:
            tool_name: 工具名称

        Returns:
            BaseTool: 工具对象

        Raises:
            ToolNotFoundError: 工具未找到
        """
        if tool_name not in self._tools:
            raise ToolNotFoundError(f"工具未找到: {tool_name}")

        return self._tools[tool_name]

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[BaseTool]:
        """
        列出工具

        Args:
            category: 工具类别（可选）

        Returns:
            List[BaseTool]: 工具列表
        """
        if category is None:
            return list(self._tools.values())
        else:
            tool_names = self._category_tools.get(category, [])
            return [self._tools[name] for name in tool_names]

    def list_tool_names(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        列出工具名称

        Args:
            category: 工具类别（可选）

        Returns:
            List[str]: 工具名称列表
        """
        if category is None:
            return list(self._tools.keys())
        else:
            return list(self._category_tools.get(category, []))

    def get_tool_metadata(self, tool_name: str) -> ToolMetadata:
        """
        获取工具元数据

        Args:
            tool_name: 工具名称

        Returns:
            ToolMetadata: 工具元数据

        Raises:
            ToolNotFoundError: 工具未找到
        """
        tool = self.get_tool(tool_name)
        return tool.get_metadata()

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        执行工具

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            ToolResult: 执行结果

        Raises:
            ToolNotFoundError: 工具未找到
            ToolExecutionError: 工具执行失败
        """
        try:
            tool = self.get_tool(tool_name)
            result = tool.execute_with_validation(**parameters)

            # 更新统计信息
            self._tool_stats[tool_name] = tool.get_stats()

            # 记录工具执行
            logger.info(f"工具执行完成", tool=tool_name, success=result.success)

            return result

        except ToolNotFoundError:
            raise
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            self._logger.error(error_msg, tool=tool_name, parameters=parameters)
            raise ToolExecutionError(error_msg, tool_name=tool_name, tool_args=parameters)

    async def execute_tool_async(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        异步执行工具

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            ToolResult: 执行结果

        Raises:
            ToolNotFoundError: 工具未找到
            ToolExecutionError: 工具执行失败
        """
        try:
            tool = self.get_tool(tool_name)
            result = await tool.execute_async_with_validation(**parameters)

            # 更新统计信息
            self._tool_stats[tool_name] = tool.get_stats()

            # 记录工具执行
            logger.info(f"工具异步执行完成", tool=tool_name, success=result.success)

            return result

        except ToolNotFoundError:
            raise
        except Exception as e:
            error_msg = f"工具异步执行失败: {str(e)}"
            self._logger.error(error_msg, tool=tool_name, parameters=parameters)
            raise ToolExecutionError(error_msg, tool_name=tool_name, tool_args=parameters)

    def search_tools(self, query: str, category: Optional[ToolCategory] = None) -> List[BaseTool]:
        """
        搜索工具

        Args:
            query: 搜索关键词
            category: 工具类别（可选）

        Returns:
            List[BaseTool]: 匹配的工具列表
        """
        query_lower = query.lower()
        results = []

        tools = self.list_tools(category)
        for tool in tools:
            metadata = tool.get_metadata()

            # 检查名称、描述和标签
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(tool)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        获取注册器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        category_counts = {}
        for category, tool_names in self._category_tools.items():
            category_counts[category.value] = len(tool_names)

        total_executions = sum(
            stats.get("execution_count", 0)
            for stats in self._tool_stats.values()
        )

        total_execution_time = sum(
            stats.get("total_execution_time", 0)
            for stats in self._tool_stats.values()
        )

        return {
            "total_tools": len(self._tools),
            "categories": category_counts,
            "total_executions": total_executions,
            "total_execution_time": total_execution_time,
            "tool_stats": dict(self._tool_stats)
        }

    def export_tools_schema(self) -> Dict[str, Any]:
        """
        导出工具架构

        Returns:
            Dict[str, Any]: 工具架构字典
        """
        schema = {
            "tools": {},
            "categories": list(self._category_tools.keys())
        }

        for tool_name, tool in self._tools.items():
            metadata = tool.get_metadata()
            schema["tools"][tool_name] = {
                "name": metadata.name,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "category": metadata.category.value,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type.value,
                        "description": param.description,
                        "required": param.required,
                        "default": param.default,
                        "enum": param.enum
                    }
                    for param in metadata.parameters
                ],
                "examples": metadata.examples,
                "tags": metadata.tags,
                "async_execution": metadata.async_execution
            }

        return schema

    def validate_tools(self) -> Dict[str, List[str]]:
        """
        验证所有工具

        Returns:
            Dict[str, List[str]]: 验证结果，键为工具名称，值为错误列表
        """
        results = {}

        for tool_name, tool in self._tools.items():
            errors = []

            try:
                metadata = tool.get_metadata()

                # 检查元数据完整性
                if not metadata.name:
                    errors.append("工具名称不能为空")
                if not metadata.description:
                    errors.append("工具描述不能为空")
                if not metadata.category:
                    errors.append("工具类别不能为空")

                # 检查参数定义
                for param in metadata.parameters:
                    if not param.name:
                        errors.append("参数名称不能为空")
                    if not param.description:
                        errors.append(f"参数 {param.name} 的描述不能为空")

            except Exception as e:
                errors.append(f"元数据获取失败: {str(e)}")

            results[tool_name] = errors

        return results

    def clear(self) -> None:
        """清空所有注册的工具"""
        self._tools.clear()
        self._category_tools.clear()
        self._tool_stats.clear()
        self._logger.info("工具注册器已清空")

    def __len__(self) -> int:
        """返回注册的工具数量"""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """检查工具是否已注册"""
        return tool_name in self._tools

    def __iter__(self):
        """迭代所有工具"""
        return iter(self._tools.values())


# 全局工具注册器实例
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> None:
    """
    注册工具的便捷函数

    Args:
        tool: 工具实例
    """
    tool_registry.register(tool)


def register_function(
    func: Callable,
    name: str,
    description: str,
    category: ToolCategory,
    parameters: Optional[List[Dict[str, Any]]] = None,
    examples: Optional[List[str]] = None,
    async_execution: bool = False,
    **metadata_kwargs
) -> FunctionTool:
    """
    注册函数为工具的便捷函数

    Args:
        func: 函数对象
        name: 工具名称
        description: 工具描述
        category: 工具类别
        parameters: 参数定义
        examples: 使用示例
        async_execution: 是否异步执行
        **metadata_kwargs: 其他元数据

    Returns:
        FunctionTool: 创建的工具对象
    """
    return tool_registry.register_function(
        func=func,
        name=name,
        description=description,
        category=category,
        parameters=parameters,
        examples=examples,
        async_execution=async_execution,
        **metadata_kwargs
    )


def get_tool(tool_name: str) -> BaseTool:
    """
    获取工具的便捷函数

    Args:
        tool_name: 工具名称

    Returns:
        BaseTool: 工具对象
    """
    return tool_registry.get_tool(tool_name)


def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """
    执行工具的便捷函数

    Args:
        tool_name: 工具名称
        parameters: 工具参数

    Returns:
        ToolResult: 执行结果
    """
    return tool_registry.execute_tool(tool_name, parameters)


async def execute_tool_async(tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
    """
    异步执行工具的便捷函数

    Args:
        tool_name: 工具名称
        parameters: 工具参数

    Returns:
        ToolResult: 执行结果
    """
    return await tool_registry.execute_tool_async(tool_name, parameters)


# 导出的接口
__all__ = [
    "ToolRegistry",
    "tool_registry",
    "register_tool",
    "register_function",
    "get_tool",
    "execute_tool",
    "execute_tool_async",
]