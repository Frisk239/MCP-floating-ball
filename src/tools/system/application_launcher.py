"""
MCP Floating Ball - 应用启动器工具

提供快速启动各种应用程序的功能，支持常见的Windows应用程序。
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

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


class ApplicationLauncherTool(BaseTool):
    """应用启动器工具"""

    def __init__(self):
        """初始化应用启动器工具"""
        super().__init__()
        # 预定义的应用程序映射
        self.applications = self._load_applications()

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="application_launcher",
            display_name="应用启动器",
            description="快速启动各种应用程序",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["system", "application", "launcher", "windows"],
            parameters=[],
            examples=["启动计算器", "启动Chrome浏览器", "启动记事本"]
        )

    def _load_applications(self) -> Dict[str, Dict[str, Any]]:
        """加载预定义的应用程序映射"""
        apps = {
            # 系统工具
            "calculator": {
                "name": "计算器",
                "commands": {
                    "windows": ["calc"],
                    "macos": ["open", "-a", "Calculator"],
                    "linux": ["gnome-calculator"]  # GNOME
                },
                "description": "打开计算器"
            },
            "notepad": {
                "name": "记事本",
                "commands": {
                    "windows": ["notepad"],
                    "macos": ["open", "-a", "TextEdit"],
                    "linux": ["gedit"]  # GNOME
                },
                "description": "打开记事本"
            },
            "cmd": {
                "name": "命令提示符",
                "commands": {
                    "windows": ["cmd"],
                    "macos": ["open", "-a", "Terminal"],
                    "linux": ["gnome-terminal"]
                },
                "description": "打开命令提示符/终端"
            },
            "powershell": {
                "name": "PowerShell",
                "commands": {
                    "windows": ["powershell"],
                    "macos": ["open", "-a", "Terminal"],
                    "linux": ["gnome-terminal", "--", "bash"]
                },
                "description": "打开PowerShell"
            },

            # 浏览器
            "chrome": {
                "name": "Google Chrome",
                "commands": {
                    "windows": ["chrome", "--new-tab"],
                    "macos": ["open", "-a", "Google Chrome"],
                    "linux": ["google-chrome", "--new-tab"]
                },
                "description": "打开Google Chrome浏览器"
            },
            "firefox": {
                "name": "Mozilla Firefox",
                "commands": {
                    "windows": ["firefox", "--new-tab"],
                    "macos": ["open", "-a", "Firefox"],
                    "linux": ["firefox", "--new-tab"]
                },
                "description": "打开Mozilla Firefox浏览器"
            },
            "edge": {
                "name": "Microsoft Edge",
                "commands": {
                    "windows": ["msedge", "--new-tab"],
                    "macos": ["open", "-a", "Microsoft Edge"],
                    "linux": []  # Edge在Linux上可能不可用
                },
                "description": "打开Microsoft Edge浏览器"
            },

            # 常用软件
            "explorer": {
                "name": "文件资源管理器",
                "commands": {
                    "windows": ["explorer"],
                    "macos": ["open", "."],
                    "linux": ["nautilus"]
                },
                "description": "打开文件资源管理器"
            },
            "task_manager": {
                "name": "任务管理器",
                "commands": {
                    "windows": ["taskmgr"],
                    "macos": ["open", "-a", "Activity Monitor"],
                    "linux": ["gnome-system-monitor"]
                },
                "description": "打开任务管理器"
            },
            "control_panel": {
                "name": "控制面板",
                "commands": {
                    "windows": ["control"],
                    "macos": ["open", "/System/Applications/System Preferences.app"],
                    "linux": ["gnome-control-center"]
                },
                "description": "打开控制面板/系统设置"
            },

            # 开发工具
            "vscode": {
                "name": "Visual Studio Code",
                "commands": {
                    "windows": ["code"],
                    "macos": ["code"],
                    "linux": ["code"]
                },
                "description": "打开Visual Studio Code"
            },
            "git_bash": {
                "name": "Git Bash",
                "commands": {
                    "windows": ["git", "bash"],
                    "macos": ["open", "-a", "Terminal"],
                    "linux": ["gnome-terminal", "--", "bash"]
                },
                "description": "打开Git Bash"
            },

            # 娱乐软件
            "spotify": {
                "name": "Spotify",
                "commands": {
                    "windows": ["spotify"],
                    "macos": ["open", "-a", "Spotify"],
                    "linux": ["spotify"]
                },
                "description": "打开Spotify音乐播放器"
            }
        }

        # 添加平台特定的应用程序
        if platform.system() == "Windows":
            apps.update({
                "word": {
                    "name": "Microsoft Word",
                    "commands": {"windows": ["winword"]},
                    "description": "打开Microsoft Word"
                },
                "excel": {
                    "name": "Microsoft Excel",
                    "commands": {"windows": ["excel"]},
                    "description": "打开Microsoft Excel"
                },
                "powerpoint": {
                    "name": "Microsoft PowerPoint",
                    "commands": {"windows": ["powerpnt"]},
                    "description": "打开Microsoft PowerPoint"
                },
                "paint": {
                    "name": "画图",
                    "commands": {"windows": ["mspaint"]},
                    "description": "打开画图工具"
                }
            })

        return apps

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="application",
                type=ParameterType.STRING.value,
                description="应用程序名称或可执行文件路径",
                required=True,
                choices=list(self.applications.keys()) + ["custom"],
                examples=["chrome", "calculator", "notepad", "C:\\path\\to\\app.exe"]
            ),
            ToolParameter(
                name="arguments",
                type=ParameterType.STRING.value,
                description="启动参数（可选）",
                required=False,
                examples=["--new-tab", "https://www.google.com", "--help"]
            ),
            ToolParameter(
                name="wait",
                type=ParameterType.BOOLEAN.value,
                description="是否等待应用启动完成",
                required=False,
                default=False
            ),
            ToolParameter(
                name="custom_command",
                type=ParameterType.STRING.value,
                description="自定义命令（当application='custom'时使用）",
                required=False,
                examples=["notepad myfile.txt", "cmd /c dir"]
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行应用启动

        Args:
            application: 应用程序名称或路径
            arguments: 启动参数
            wait: 是否等待启动完成
            custom_command: 自定义命令

        Returns:
            启动结果
        """
        try:
            application = kwargs.get("application", "")
            arguments = kwargs.get("arguments", "")
            wait = kwargs.get("wait", False)
            custom_command = kwargs.get("custom_command", "")

            if not application:
                raise ToolError("应用程序名称不能为空")

            self.logger.info(
                "启动应用程序",
                application=application,
                arguments=arguments,
                wait=wait,
                custom_command=custom_command
            )

            start_time = time.time()

            # 执行启动
            result = self._launch_application(application, arguments, wait, custom_command)

            execution_time = time.time() - start_time

            self.logger.info(
                "应用程序启动完成",
                application=application,
                success=result["success"],
                execution_time=execution_time
            )

            return {
                "success": result["success"],
                "application": result["application"],
                "command": result["command"],
                "pid": result.get("pid"),
                "window_title": result.get("window_title"),
                "execution_time": execution_time,
                "message": result["message"],
                "error": result.get("error")
            }

        except Exception as e:
            error_msg = f"启动应用程序失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0
            }

    def _launch_application(
        self,
        application: str,
        arguments: str,
        wait: bool,
        custom_command: str
    ) -> Dict[str, Any]:
        """启动应用程序"""

        if application == "custom":
            if not custom_command:
                raise ToolError("使用自定义命令时必须提供custom_command参数")
            command_parts = custom_command.split()
            app_name = "自定义应用程序"
            cmd_list = command_parts
        else:
            # 查找预定义的应用程序
            app_info = self.applications.get(application.lower())
            if not app_info:
                # 尝试作为可执行文件路径处理
                if os.path.exists(application):
                    app_name = os.path.basename(application)
                    cmd_list = [application]
                    if arguments:
                        cmd_list.extend(arguments.split())
                else:
                    raise ToolError(f"未知的应用程序: {application}")
            else:
                app_name = app_info["name"]
                # 获取平台特定的命令
                platform_commands = app_info["commands"].get(platform.system().lower(), [])
                if not platform_commands:
                    # 尝试使用Windows命令作为默认
                    platform_commands = app_info["commands"].get("windows", [])

                if not platform_commands:
                    raise ToolError(f"平台 {platform.system()} 不支持应用程序 {application}")

                cmd_list = platform_commands.copy()
                if arguments:
                    cmd_list.extend(arguments.split())

        try:
            # 启动应用程序
            if wait:
                # 同步启动，等待启动完成
                process = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False
                )
                pid = process.pid

                # 等待一小段时间让应用程序启动
                time.sleep(1)

                # 尝试获取窗口标题
                window_title = self._get_window_title(pid)

                return {
                    "success": True,
                    "application": app_name,
                    "command": " ".join(cmd_list),
                    "pid": pid,
                    "window_title": window_title,
                    "message": f"成功启动 {app_name}"
                }
            else:
                # 异步启动
                if platform.system() == "Windows":
                    # Windows上使用start命令后台启动
                    subprocess.Popen(
                        cmd_list,
                        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                        shell=False
                    )
                else:
                    # Unix-like系统
                    subprocess.Popen(
                        cmd_list,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        shell=False
                    )

                return {
                    "success": True,
                    "application": app_name,
                    "command": " ".join(cmd_list),
                    "message": f"已启动 {app_name}"
                }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "application": app_name,
                "command": " ".join(cmd_list),
                "error": f"启动失败: {e}",
                "message": f"启动 {app_name} 失败"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "application": app_name,
                "command": " ".join(cmd_list),
                "error": "找不到指定的应用程序",
                "message": f"找不到应用程序 {app_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "application": app_name,
                "command": " ".join(cmd_list),
                "error": str(e),
                "message": f"启动 {app_name} 时发生错误"
            }

    def _get_window_title(self, pid: int) -> Optional[str]:
        """获取进程对应的窗口标题"""
        if not PYGETWINDOW_AVAILABLE:
            return None

        try:
            # 获取所有窗口
            windows = gw.getAllWindows()

            # 查找匹配的窗口
            for window in windows:
                # 窗口PID可能不直接可用，但可以通过标题匹配
                if window.title and any(keyword in window.title.lower()
                                       for keyword in ["chrome", "firefox", "edge", "notepad", "calculator"]):
                    return window.title

            return None
        except Exception:
            return None

    def list_applications(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有可用的应用程序

        Returns:
            应用程序列表
        """
        return {
            "applications": self.applications,
            "platform": platform.system(),
            "total_count": len(self.applications)
        }

    def search_applications(self, query: str) -> Dict[str, Any]:
        """
        搜索应用程序

        Args:
            query: 搜索关键词

        Returns:
            搜索结果
        """
        query = query.lower()
        results = {}

        for app_id, app_info in self.applications.items():
            if (query in app_id.lower() or
                query in app_info["name"].lower() or
                query in app_info["description"].lower()):
                results[app_id] = app_info

        return {
            "query": query,
            "results": results,
            "count": len(results)
        }

    def get_running_processes(self) -> Dict[str, Any]:
        """
        获取正在运行的进程列表

        Returns:
            进程列表
        """
        if not PSUTIL_AVAILABLE:
            return {
                "error": "psutil模块不可用，无法获取进程信息",
                "processes": []
            }

        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['name']:  # 过滤掉没有名称的进程
                        processes.append({
                            "pid": proc_info["pid"],
                            "name": proc_info["name"],
                            "cmdline": " ".join(proc_info["cmdline"] or [])
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                "processes": processes[:50],  # 限制返回数量
                "total_count": len(processes)
            }
        except Exception as e:
            return {
                "error": f"获取进程信息失败: {e}",
                "processes": []
            }


# 注册工具
from ..registry import tool_registry
tool_registry.register(ApplicationLauncherTool())