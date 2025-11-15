"""
MCP Floating Ball - 系统信息查询工具

提供系统信息查询功能，包括硬件信息、软件信息、网络状态等。
"""

import platform
import os
import sys
import time
from typing import Dict, List, Optional, Any
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from ...core.logging import get_logger
from ...core.exceptions import ToolError
from ..base import BaseTool, ToolMetadata, ToolCategory, ParameterType, ToolParameter

logger = get_logger(__name__)


class SystemInfoTool(BaseTool):
    """系统信息查询工具"""

    def __init__(self):
        """初始化系统信息工具"""
        super().__init__()
        self.logger = get_logger("tool.system_info")

    def get_metadata(self) -> ToolMetadata:
        """获取工具元数据"""
        return ToolMetadata(
            name="system_info",
            display_name="系统信息查询",
            description="查询系统信息，包括硬件、软件、网络状态等",
            category=ToolCategory.SYSTEM,
            version="1.0.0",
            author="MCP Floating Ball",
            tags=["system", "info", "hardware", "software", "network"],
            parameters=[
                ToolParameter(
                    name="category",
                    type=ParameterType.STRING,
                    description="信息类别",
                    required=True,
                    enum=["basic", "cpu", "memory", "disk", "gpu", "network", "processes", "environment", "hardware", "software", "all"]
                ),
                ToolParameter(
                    name="detailed",
                    type=ParameterType.BOOLEAN,
                    description="是否显示详细信息",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="format",
                    type=ParameterType.STRING,
                    description="输出格式",
                    required=False,
                    enum=["json", "text", "table"],
                    default="json"
                )
            ],
            examples=["查询基本信息", "查询CPU使用情况", "查询内存使用情况", "查询所有系统信息"]
        )

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="category",
                type=ParameterType.STRING.value,
                description="信息类别",
                required=True,
                choices=[
                    "basic", "cpu", "memory", "disk", "gpu", "network",
                    "processes", "environment", "hardware", "software", "all"
                ],
                examples=["basic", "cpu", "memory", "all"]
            ),
            ToolParameter(
                name="detailed",
                type=ParameterType.BOOLEAN.value,
                description="是否显示详细信息",
                required=False,
                default=False
            ),
            ToolParameter(
                name="format",
                type=ParameterType.STRING.value,
                description="输出格式",
                required=False,
                choices=["dict", "json", "text"],
                default="dict"
            )
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行系统信息查询

        Args:
            category: 信息类别
            detailed: 是否显示详细信息
            format: 输出格式

        Returns:
            系统信息
        """
        try:
            category = kwargs.get("category", "basic")
            detailed = kwargs.get("detailed", False)
            format_type = kwargs.get("format", "dict")

            self.logger.info(
                "查询系统信息",
                category=category,
                detailed=detailed
            )

            start_time = time.time()

            # 根据类别获取相应信息
            if category == "all":
                result = self._get_all_info(detailed)
            elif category == "basic":
                result = self._get_basic_info(detailed)
            elif category == "cpu":
                result = self._get_cpu_info(detailed)
            elif category == "memory":
                result = self._get_memory_info(detailed)
            elif category == "disk":
                result = self._get_disk_info(detailed)
            elif category == "gpu":
                result = self._get_gpu_info(detailed)
            elif category == "network":
                result = self._get_network_info(detailed)
            elif category == "processes":
                result = self._get_process_info(detailed)
            elif category == "environment":
                result = self._get_environment_info(detailed)
            elif category == "hardware":
                result = self._get_hardware_info(detailed)
            elif category == "software":
                result = self._get_software_info(detailed)
            else:
                raise ToolError(f"不支持的信息类别: {category}")

            execution_time = time.time() - start_time

            self.logger.info(
                "系统信息查询完成",
                category=category,
                execution_time=execution_time
            )

            # 格式化输出
            result["query_info"] = {
                "category": category,
                "detailed": detailed,
                "execution_time": execution_time
            }

            if format_type == "json":
                return {
                    "success": True,
                    "data": json.dumps(result, indent=2, ensure_ascii=False),
                    "format": "json"
                }
            elif format_type == "text":
                return {
                    "success": True,
                    "data": self._format_as_text(result),
                    "format": "text"
                }
            else:
                return {
                    "success": True,
                    "data": result,
                    "format": "dict"
                }

        except Exception as e:
            error_msg = f"查询系统信息失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }

    def _get_all_info(self, detailed: bool) -> Dict[str, Any]:
        """获取所有系统信息"""
        return {
            "basic": self._get_basic_info(detailed),
            "cpu": self._get_cpu_info(detailed),
            "memory": self._get_memory_info(detailed),
            "disk": self._get_disk_info(detailed),
            "gpu": self._get_gpu_info(detailed),
            "network": self._get_network_info(detailed),
            "hardware": self._get_hardware_info(detailed),
            "software": self._get_software_info(detailed)
        }

    def _get_basic_info(self, detailed: bool) -> Dict[str, Any]:
        """获取基本信息"""
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }

        if detailed:
            info.update({
                "platform_freedesktop": platform.freedesktop_os_release() if hasattr(platform, 'freedesktop_os_release') else {},
                "platform_libc_ver": platform.libc_ver(),
                "time_boot": psutil.boot_time() if PSUTIL_AVAILABLE else None,
                "time_current": time.time(),
                "users": self._get_users_info() if PSUTIL_AVAILABLE else []
            })

        return info

    def _get_cpu_info(self, detailed: bool) -> Dict[str, Any]:
        """获取CPU信息"""
        info = {}

        if PSUTIL_AVAILABLE:
            info.update({
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "usage_percent": psutil.cpu_percent(interval=1),
                "frequency": {
                    "current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    "min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
                    "max": psutil.cpu_freq().max if psutil.cpu_freq() else None
                }
            })

        if detailed and WMI_AVAILABLE and platform.system() == "Windows":
            try:
                wmi_client = wmi.WMI()
                cpu_info = wmi_client.Win32_Processor()[0]
                info.update({
                    "name": cpu_info.Name,
                    "manufacturer": cpu_info.Manufacturer,
                    "description": cpu_info.Description,
                    "family": cpu_info.Family,
                    "max_clock_speed": cpu_info.MaxClockSpeed,
                    "current_clock_speed": cpu_info.CurrentClockSpeed
                })
            except Exception:
                pass

        if detailed:
            info.update({
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
                "usage_per_core": psutil.cpu_percent(interval=1, percpu=True) if PSUTIL_AVAILABLE else None,
                "times": psutil.cpu_times()._asdict() if PSUTIL_AVAILABLE else None
            })

        return info

    def _get_memory_info(self, detailed: bool) -> Dict[str, Any]:
        """获取内存信息"""
        info = {}

        if PSUTIL_AVAILABLE:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            info = {
                "virtual": {
                    "total": virtual_mem.total,
                    "available": virtual_mem.available,
                    "used": virtual_mem.used,
                    "free": virtual_mem.free,
                    "percent": virtual_mem.percent
                },
                "swap": {
                    "total": swap_mem.total,
                    "used": swap_mem.used,
                    "free": swap_mem.free,
                    "percent": swap_mem.percent
                }
            }

        if detailed and PSUTIL_AVAILABLE:
            info["virtual"].update({
                "active": getattr(virtual_mem, 'active', None),
                "inactive": getattr(virtual_mem, 'inactive', None),
                "buffers": getattr(virtual_mem, 'buffers', None),
                "cached": getattr(virtual_mem, 'cached', None)
            })

        return info

    def _get_disk_info(self, detailed: bool) -> Dict[str, Any]:
        """获取磁盘信息"""
        info = {
            "partitions": [],
            "usage": {}
        }

        if PSUTIL_AVAILABLE:
            # 磁盘分区信息
            partitions = psutil.disk_partitions()
            info["partitions"] = [
                {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "opts": partition.opts
                }
                for partition in partitions
            ]

            # 磁盘使用情况
            usage_info = {}
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    usage_info[partition.device] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    }
                except Exception:
                    continue

            info["usage"] = usage_info

        if detailed and PSUTIL_AVAILABLE:
            # 磁盘IO统计
            try:
                disk_io = psutil.disk_io_counters(perdisk=True)
                info["io_stats"] = {
                    device: {
                        "read_count": stats.read_count,
                        "write_count": stats.write_count,
                        "read_bytes": stats.read_bytes,
                        "write_bytes": stats.write_bytes
                    }
                    for device, stats in disk_io.items()
                }
            except Exception:
                pass

        return info

    def _get_gpu_info(self, detailed: bool) -> Dict[str, Any]:
        """获取GPU信息"""
        info = {"gpus": []}

        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info = {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "temperature": gpu.temperature,
                        "uuid": gpu.uuid
                    }
                    info["gpus"].append(gpu_info)
            except Exception:
                pass

        if detailed and WMI_AVAILABLE and platform.system() == "Windows":
            try:
                wmi_client = wmi.WMI()
                gpu_devices = wmi_client.Win32_VideoController()
                for gpu in gpu_devices:
                    gpu_info = {
                        "name": gpu.Name,
                        "adapter_ram": gpu.AdapterRAM,
                        "driver_version": gpu.DriverVersion,
                        "driver_date": str(gpu.DriverVersion),
                        "video_mode_description": gpu.VideoModeDescription
                    }
                    info["gpus"].append(gpu_info)
            except Exception:
                pass

        return info

    def _get_network_info(self, detailed: bool) -> Dict[str, Any]:
        """获取网络信息"""
        info = {
            "interfaces": [],
            "io_stats": None
        }

        if PSUTIL_AVAILABLE:
            # 网络接口
            interfaces = psutil.net_if_addrs()
            for interface_name, addresses in interfaces.items():
                interface_info = {
                    "name": interface_name,
                    "addresses": [
                        {
                            "family": str(address.family),
                            "address": address.address,
                            "netmask": address.netmask,
                            "broadcast": address.broadcast
                        }
                        for address in addresses
                    ]
                }
                info["interfaces"].append(interface_info)

            # 网络IO统计
            io_stats = psutil.net_io_counters()
            info["io_stats"] = {
                "bytes_sent": io_stats.bytes_sent,
                "bytes_recv": io_stats.bytes_recv,
                "packets_sent": io_stats.packets_sent,
                "packets_recv": io_stats.packets_recv,
                "errin": io_stats.errin,
                "errout": io_stats.errout,
                "dropin": io_stats.dropin,
                "dropout": io_stats.dropout
            }

        if detailed:
            # 获取连接信息
            if PSUTIL_AVAILABLE:
                connections = psutil.net_connections()
                info["connections"] = [
                    {
                        "local_address": conn.laddr,
                        "remote_address": conn.raddr,
                        "status": conn.status,
                        "pid": conn.pid
                    }
                    for conn in connections[:50]  # 限制数量
                ]

        return info

    def _get_process_info(self, detailed: bool) -> Dict[str, Any]:
        """获取进程信息"""
        info = {
            "processes": [],
            "count": 0
        }

        if PSUTIL_AVAILABLE:
            processes = []
            process_count = 0

            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    process_count += 1
                    proc_info = {
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "status": proc.info['status'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    }

                    if detailed:
                        # 获取更多详细信息
                        try:
                            process = psutil.Process(proc.info['pid'])
                            proc_info.update({
                                "exe": process.exe(),
                                "cmdline": process.cmdline(),
                                "create_time": process.create_time(),
                                "num_threads": process.num_threads(),
                                "username": process.username()
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            info["processes"] = processes[:100]  # 限制返回数量
            info["count"] = process_count

        return info

    def _get_environment_info(self, detailed: bool) -> Dict[str, Any]:
        """获取环境信息"""
        env_info = {
            "path": os.environ.get("PATH", "").split(os.pathsep),
            "python_path": sys.path,
            "current_directory": os.getcwd(),
            "home_directory": os.path.expanduser("~")
        }

        if detailed:
            # 包含更多环境变量
            important_env_vars = [
                "USER", "USERNAME", "HOME", "USERPROFILE", "TEMP", "TMP",
                "COMPUTERNAME", "LANG", "LC_ALL", "SHELL"
            ]
            env_info["environment_variables"] = {
                var: os.environ.get(var) for var in important_env_vars
                if var in os.environ
            }

        return env_info

    def _get_hardware_info(self, detailed: bool) -> Dict[str, Any]:
        """获取硬件信息"""
        info = {}

        if WMI_AVAILABLE and platform.system() == "Windows":
            try:
                wmi_client = wmi.WMI()

                # 主板信息
                board_info = wmi_client.Win32_BaseBoard()[0]
                info["motherboard"] = {
                    "manufacturer": board_info.Manufacturer,
                    "product": board_info.Product,
                    "serial_number": board_info.SerialNumber
                }

                # BIOS信息
                bios_info = wmi_client.Win32_BIOS()[0]
                info["bios"] = {
                    "manufacturer": bios_info.Manufacturer,
                    "version": bios_info.SMBIOSBIOSVersion,
                    "release_date": str(bios_info.ReleaseDate)
                }

                if detailed:
                    # 内存条信息
                    memory_info = wmi_client.Win32_PhysicalMemory()
                    memory_modules = []
                    for mem in memory_info:
                        memory_modules.append({
                            "capacity": mem.Capacity,
                            "speed": mem.Speed,
                            "manufacturer": mem.Manufacturer,
                            "part_number": mem.PartNumber
                        })
                    info["memory_modules"] = memory_modules

            except Exception:
                pass

        return info

    def _get_software_info(self, detailed: bool) -> Dict[str, Any]:
        """获取软件信息"""
        info = {
            "installed_packages": self._get_installed_packages() if detailed else [],
            "running_services": []
        }

        if PSUTIL_AVAILABLE:
            try:
                # 获取服务信息
                services = psutil.win_service_iter() if platform.system() == "Windows" else []
                info["running_services"] = [
                    {
                        "name": service.name(),
                        "display_name": service.display_name(),
                        "status": service.status()
                    }
                    for service in list(services)[:50]  # 限制数量
                ]
            except Exception:
                pass

        return info

    def _get_users_info(self) -> List[Dict[str, Any]]:
        """获取用户信息"""
        try:
            users = psutil.users()
            return [
                {
                    "name": user.name,
                    "terminal": user.terminal,
                    "host": user.host,
                    "started": user.started
                }
                for user in users
            ]
        except Exception:
            return []

    def _get_installed_packages(self) -> List[Dict[str, Any]]:
        """获取已安装的Python包"""
        try:
            import pkg_resources
            installed_packages = pkg_resources.working_set
            return [
                {
                    "name": pkg.project_name,
                    "version": pkg.version,
                    "location": pkg.location
                }
                for pkg in installed_packages[:100]  # 限制数量
            ]
        except Exception:
            return []

    def _format_as_text(self, data: Dict[str, Any]) -> str:
        """格式化为文本"""
        lines = []
        lines.append("=== 系统信息报告 ===")
        lines.append(f"查询时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本信息格式化
        if "basic" in data:
            basic = data["basic"]
            lines.append("【基本信息】")
            lines.append(f"操作系统: {basic.get('platform', 'Unknown')}")
            lines.append(f"计算机名: {basic.get('hostname', 'Unknown')}")
            lines.append(f"处理器: {basic.get('processor', 'Unknown')}")
            lines.append(f"架构: {basic.get('architecture', 'Unknown')}")
            lines.append(f"Python版本: {basic.get('python_version', 'Unknown')}")
            lines.append("")

        # CPU信息格式化
        if "cpu" in data:
            cpu = data["cpu"]
            lines.append("【CPU信息】")
            lines.append(f"逻辑核心数: {cpu.get('logical_cores', 'Unknown')}")
            lines.append(f"物理核心数: {cpu.get('physical_cores', 'Unknown')}")
            lines.append(f"CPU使用率: {cpu.get('usage_percent', 'Unknown')}%")
            lines.append("")

        # 内存信息格式化
        if "memory" in data:
            memory = data["memory"]
            if "virtual" in memory:
                vm = memory["virtual"]
                lines.append("【内存信息】")
                lines.append(f"总内存: {self._format_bytes(vm.get('total', 0))}")
                lines.append(f"已使用: {self._format_bytes(vm.get('used', 0))}")
                lines.append(f"可用: {self._format_bytes(vm.get('available', 0))}")
                lines.append(f"使用率: {vm.get('percent', 0):.1f}%")
                lines.append("")

        # 磁盘信息格式化
        if "disk" in data:
            disk = data["disk"]
            if "usage" in disk:
                lines.append("【磁盘信息】")
                for device, usage in disk["usage"].items():
                    lines.append(f"{device}:")
                    lines.append(f"  总容量: {self._format_bytes(usage.get('total', 0))}")
                    lines.append(f"  已使用: {self._format_bytes(usage.get('used', 0))}")
                    lines.append(f"  使用率: {usage.get('percent', 0):.1f}%")
                lines.append("")

        return "\n".join(lines)

    def _format_bytes(self, bytes_value: int) -> str:
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f} PB"


# 注册工具
from ..registry import tool_registry
tool_registry.register(SystemInfoTool())