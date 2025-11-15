"""
MCP Floating Ball - 窗口管理工具

提供窗口操作功能，包括窗口查找、最大化、最小化、关闭等。
"""

import time
from typing import Dict, List, Optional, Any
import platform

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ...core.logging import get_logger
from ...core.exceptions import ToolError
from ..base import BaseTool, ToolMetadata, ToolCategory, ParameterType, ToolParameter

logger = get_logger(__name__)


class WindowManagerTool(BaseTool):
    """窗口管理工具"""

    def __init__(self):
        """初始化窗口管理工具"""
        super().__init__()
        self.logger = get_logger("tool.window_manager")

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="window_manager",
            display_name="窗口管理器",
            description="管理系统窗口，包括查找、移动、调整大小、最小化、最大化、关闭等操作",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["system", "window", "manager", "gui"],
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="窗口操作类型",
                    required=True,
                    enum=["list", "find", "activate", "minimize", "maximize", "restore", "close", "move", "resize", "info"]
                ),
                ToolParameter(
                    name="title",
                    type=ParameterType.STRING,
                    description="窗口标题（用于查找窗口）",
                    required=False
                ),
                ToolParameter(
                    name="pid",
                    type=ParameterType.INTEGER,
                    description="进程ID（用于查找窗口）",
                    required=False
                ),
                ToolParameter(
                    name="x",
                    type=ParameterType.INTEGER,
                    description="窗口X坐标（用于move操作）",
                    required=False
                ),
                ToolParameter(
                    name="y",
                    type=ParameterType.INTEGER,
                    description="窗口Y坐标（用于move操作）",
                    required=False
                ),
                ToolParameter(
                    name="width",
                    type=ParameterType.INTEGER,
                    description="窗口宽度（用于resize操作）",
                    required=False
                ),
                ToolParameter(
                    name="height",
                    type=ParameterType.INTEGER,
                    description="窗口高度（用于resize操作）",
                    required=False
                )
            ],
            examples=["列出所有窗口", "查找记事本窗口", "移动Chrome窗口到坐标100,100"]
        )

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="action",
                type=ParameterType.STRING.value,
                description="窗口操作类型",
                required=True,
                choices=["list", "find", "activate", "minimize", "maximize", "restore", "close", "move", "resize", "info"],
                examples=["list", "minimize", "maximize", "close"]
            ),
            ToolParameter(
                name="title",
                type=ParameterType.STRING.value,
                description="窗口标题（用于查找窗口）",
                required=False,
                examples=["Chrome", "Notepad", "Calculator"]
            ),
            ToolParameter(
                name="pid",
                type=ParameterType.INTEGER.value,
                description="进程ID（用于查找窗口）",
                required=False,
                examples=[1234, 5678]
            ),
            ToolParameter(
                name="x",
                type=ParameterType.INTEGER.value,
                description="窗口X坐标（用于move操作）",
                required=False,
                examples=[100, 200]
            ),
            ToolParameter(
                name="y",
                type=ParameterType.INTEGER.value,
                description="窗口Y坐标（用于move操作）",
                required=False,
                examples=[100, 200]
            ),
            ToolParameter(
                name="width",
                type=ParameterType.INTEGER.value,
                description="窗口宽度（用于resize操作）",
                required=False,
                examples=[800, 1024]
            ),
            ToolParameter(
                name="height",
                type=ParameterType.INTEGER.value,
                description="窗口高度（用于resize操作）",
                required=False,
                examples=[600, 768]
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行窗口管理操作

        Args:
            action: 操作类型
            title: 窗口标题
            pid: 进程ID
            x, y: 坐标
            width, height: 尺寸

        Returns:
            操作结果
        """
        try:
            if not PYGETWINDOW_AVAILABLE:
                raise ToolError("pygetwindow模块不可用，请安装: pip install pygetwindow")

            action = kwargs.get("action", "")
            if not action:
                raise ToolError("操作类型不能为空")

            self.logger.info("执行窗口操作", action=action, kwargs=kwargs)

            start_time = time.time()

            # 根据操作类型执行相应的方法
            if action == "list":
                result = self._list_windows()
            elif action == "find":
                result = self._find_windows(kwargs)
            elif action == "info":
                result = self._get_window_info(kwargs)
            elif action in ["activate", "minimize", "maximize", "restore", "close"]:
                result = self._window_action(action, kwargs)
            elif action == "move":
                result = self._move_window(kwargs)
            elif action == "resize":
                result = self._resize_window(kwargs)
            else:
                raise ToolError(f"不支持的操作类型: {action}")

            execution_time = time.time() - start_time

            self.logger.info(
                "窗口操作完成",
                action=action,
                success=result.get("success", False),
                execution_time=execution_time
            )

            result["execution_time"] = execution_time
            return result

        except Exception as e:
            error_msg = f"窗口操作失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0
            }

    def _list_windows(self) -> Dict[str, Any]:
        """列出所有窗口"""
        try:
            windows = gw.getAllWindows()
            window_list = []

            for window in windows:
                try:
                    window_list.append({
                        "title": window.title or "",
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "is_visible": window.visible,
                        "is_maximized": window.isMaximized,
                        "is_minimized": window.isMinimized
                    })
                except Exception:
                    continue

            return {
                "success": True,
                "windows": window_list,
                "count": len(window_list)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"获取窗口列表失败: {e}",
                "windows": []
            }

    def _find_windows(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """查找窗口"""
        try:
            title = params.get("title", "")
            pid = params.get("pid")

            if title:
                # 按标题查找
                windows = gw.getWindowsWithTitle(title)
                if not windows:
                    # 模糊匹配
                    all_windows = gw.getAllWindows()
                    windows = [w for w in all_windows if title.lower() in (w.title or "").lower()]
            elif pid:
                # 按进程ID查找
                all_windows = gw.getAllWindows()
                windows = [w for w in all_windows if self._get_window_pid(w) == pid]
            else:
                raise ToolError("必须提供title或pid参数")

            window_list = []
            for window in windows:
                try:
                    window_list.append({
                        "title": window.title or "",
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "is_visible": window.visible,
                        "is_maximized": window.isMaximized,
                        "is_minimized": window.isMinimized
                    })
                except Exception:
                    continue

            return {
                "success": True,
                "windows": window_list,
                "count": len(window_list),
                "search_params": params
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"查找窗口失败: {e}",
                "windows": []
            }

    def _get_window_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取窗口详细信息"""
        try:
            windows = self._find_target_windows(params)
            if not windows:
                raise ToolError("找不到指定的窗口")

            window = windows[0]  # 使用第一个匹配的窗口

            # 尝试获取进程信息
            pid = self._get_window_pid(window)
            process_info = None
            if pid and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process(pid)
                    process_info = {
                        "pid": pid,
                        "name": process.name(),
                        "exe": process.exe(),
                        "cmdline": process.cmdline(),
                        "cpu_percent": process.cpu_percent(),
                        "memory_percent": process.memory_percent()
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return {
                "success": True,
                "window": {
                    "title": window.title or "",
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height,
                    "is_visible": window.visible,
                    "is_maximized": window.isMaximized,
                    "is_minimized": window.isMinimized
                },
                "process_info": process_info
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"获取窗口信息失败: {e}"
            }

    def _window_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行窗口操作"""
        try:
            windows = self._find_target_windows(params)
            if not windows:
                raise ToolError("找不到指定的窗口")

            results = []
            success_count = 0

            for window in windows:
                try:
                    window_title = window.title or ""
                    if action == "activate":
                        window.activate()
                        status = "已激活"
                    elif action == "minimize":
                        window.minimize()
                        status = "已最小化"
                    elif action == "maximize":
                        window.maximize()
                        status = "已最大化"
                    elif action == "restore":
                        window.restore()
                        status = "已恢复"
                    elif action == "close":
                        window.close()
                        status = "已关闭"
                    else:
                        continue

                    results.append({
                        "title": window_title,
                        "action": action,
                        "status": status,
                        "success": True
                    })
                    success_count += 1

                except Exception as e:
                    results.append({
                        "title": window.title or "",
                        "action": action,
                        "status": "失败",
                        "success": False,
                        "error": str(e)
                    })

            return {
                "success": success_count > 0,
                "results": results,
                "success_count": success_count,
                "total_count": len(windows)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"窗口操作失败: {e}"
            }

    def _move_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """移动窗口"""
        try:
            x = params.get("x")
            y = params.get("y")

            if x is None or y is None:
                raise ToolError("移动窗口需要提供x和y坐标")

            windows = self._find_target_windows(params)
            if not windows:
                raise ToolError("找不到指定的窗口")

            results = []
            success_count = 0

            for window in windows:
                try:
                    window.moveTo(x, y)
                    results.append({
                        "title": window.title or "",
                        "action": "move",
                        "new_position": {"x": x, "y": y},
                        "success": True
                    })
                    success_count += 1
                except Exception as e:
                    results.append({
                        "title": window.title or "",
                        "action": "move",
                        "success": False,
                        "error": str(e)
                    })

            return {
                "success": success_count > 0,
                "results": results,
                "success_count": success_count,
                "total_count": len(windows)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"移动窗口失败: {e}"
            }

    def _resize_window(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """调整窗口大小"""
        try:
            width = params.get("width")
            height = params.get("height")

            if width is None or height is None:
                raise ToolError("调整窗口大小需要提供width和height参数")

            if width <= 0 or height <= 0:
                raise ToolError("窗口宽度和高度必须大于0")

            windows = self._find_target_windows(params)
            if not windows:
                raise ToolError("找不到指定的窗口")

            results = []
            success_count = 0

            for window in windows:
                try:
                    window.resizeTo(width, height)
                    results.append({
                        "title": window.title or "",
                        "action": "resize",
                        "new_size": {"width": width, "height": height},
                        "success": True
                    })
                    success_count += 1
                except Exception as e:
                    results.append({
                        "title": window.title or "",
                        "action": "resize",
                        "success": False,
                        "error": str(e)
                    })

            return {
                "success": success_count > 0,
                "results": results,
                "success_count": success_count,
                "total_count": len(windows)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"调整窗口大小失败: {e}"
            }

    def _find_target_windows(self, params: Dict[str, Any]) -> List:
        """查找目标窗口"""
        title = params.get("title", "")
        pid = params.get("pid")

        if title:
            windows = gw.getWindowsWithTitle(title)
            if not windows:
                # 模糊匹配
                all_windows = gw.getAllWindows()
                windows = [w for w in all_windows if title.lower() in (w.title or "").lower()]
        elif pid:
            all_windows = gw.getAllWindows()
            windows = [w for w in all_windows if self._get_window_pid(w) == pid]
        else:
            raise ToolError("必须提供title或pid参数")

        return windows

    def _get_window_pid(self, window) -> Optional[int]:
        """获取窗口对应的进程ID"""
        # pygetwindow本身不支持直接获取PID，这里使用替代方案
        try:
            # 在Windows上，可以使用窗口句柄来获取进程ID
            if platform.system() == "Windows":
                import win32gui
                import win32process

                hwnd = window._hWnd if hasattr(window, '_hWnd') else None
                if hwnd:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    return pid
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def get_screen_info(self) -> Dict[str, Any]:
        """获取屏幕信息"""
        try:
            if platform.system() == "Windows":
                import win32api
                import win32con

                # 获取主显示器信息
                hdc = win32api.GetDC(0)
                width = win32api.GetDeviceCaps(hdc, win32con.HORZRES)
                height = win32api.GetDeviceCaps(hdc, win32con.VERTRES)
                win32api.ReleaseDC(0, hdc)

                return {
                    "success": True,
                    "primary_screen": {
                        "width": width,
                        "height": height,
                        "dpi": 96  # 默认DPI
                    }
                }
            else:
                # 跨平台方式
                windows = gw.getAllWindows()
                if windows:
                    # 找到一个可见的窗口来推断屏幕大小
                    for window in windows:
                        if window.visible:
                            return {
                                "success": True,
                                "primary_screen": {
                                    "width": window.left + window.width,
                                    "height": window.top + window.height,
                                    "method": "window_inference"
                                }
                            }

                return {
                    "success": False,
                    "error": "无法获取屏幕信息"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"获取屏幕信息失败: {e}"
            }


# 注册工具
from ..registry import tool_registry
tool_registry.register(WindowManagerTool())