"""
MCP Floating Ball - 屏幕截图模块

提供屏幕截图、窗口截图、区域截图等功能。
"""

import time
import platform
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
from datetime import datetime

try:
    import pyautogui
    import numpy as np
    from PIL import Image, ImageDraw
    from PIL.ImageGrab import grab
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pygetwindow as gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    PYGETWINDOW_AVAILABLE = False

from src.core.logging import get_logger
from src.core.exceptions import VisionError


class ScreenCapture:
    """屏幕截图类"""

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化屏幕截图

        Args:
            output_dir: 截图输出目录
        """
        self.logger = get_logger("vision.screen_capture")

        if not GUI_AVAILABLE:
            raise VisionError("GUI库未安装，请安装: pip install pyautogui pillow numpy")

        self.output_dir = Path(output_dir) if output_dir else Path("./screenshots")
        self.output_dir.mkdir(exist_ok=True)

        # 默认配置
        self.default_format = "png"
        self.default_quality = 95
        self.cursor_capture = True

        # 屏幕信息
        self.screen_size = self._get_screen_size()
        self.system_info = self._get_system_info()

        self.logger.info(f"屏幕截图初始化完成，屏幕尺寸: {self.screen_size}")

    def _get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕尺寸"""
        try:
            width, height = pyautogui.size()
            return (width, height)
        except Exception as e:
            self.logger.error(f"获取屏幕尺寸失败: {e}")
            return (1920, 1080)  # 默认尺寸

    def _get_system_info(self) -> Dict[str, str]:
        """获取系统信息"""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        }

    def capture_full_screen(self, save: bool = True, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        截取全屏

        Args:
            save: 是否保存到文件
            filename: 自定义文件名

        Returns:
            截图结果字典
        """
        start_time = time.time()

        try:
            self.logger.info("开始截取全屏")

            # 截图
            screenshot = pyautogui.screenshot()
            if not self.cursor_capture:
                screenshot = grab()  # 不包含光标

            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fullscreen_{timestamp}.{self.default_format}"

            filepath = self.output_dir / filename if save else None

            # 保存图片
            if save:
                if self.default_format.lower() == 'jpg' or self.default_format.lower() == 'jpeg':
                    screenshot.save(filepath, quality=self.default_quality, optimize=True)
                else:
                    screenshot.save(filepath, optimize=True)
                self.logger.info(f"全屏截图已保存: {filepath}")

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "type": "full_screen",
                "size": screenshot.size,
                "mode": screenshot.mode,
                "filepath": str(filepath) if filepath else None,
                "filename": filename,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            # 保存图片对象用于后续处理
            result["image"] = screenshot

            self.logger.info(f"全屏截图完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"全屏截图失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "type": "full_screen",
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def capture_region(self, x: int, y: int, width: int, height: int, save: bool = True,
                      filename: Optional[str] = None) -> Dict[str, Any]:
        """
        截取指定区域

        Args:
            x: 左上角X坐标
            y: 左上角Y坐标
            width: 宽度
            height: 高度
            save: 是否保存到文件
            filename: 自定义文件名

        Returns:
            截图结果字典
        """
        start_time = time.time()

        try:
            self.logger.info(f"截取区域: ({x}, {y}, {width}, {height})")

            # 验证区域
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("无效的截取区域参数")

            if x + width > self.screen_size[0] or y + height > self.screen_size[1]:
                self.logger.warning("截取区域超出屏幕范围，将自动调整")

            # 截图
            screenshot = pyautogui.screenshot(region=(x, y, width, height))

            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"region_{x}_{y}_{width}_{height}_{timestamp}.{self.default_format}"

            filepath = self.output_dir / filename if save else None

            # 保存图片
            if save:
                if self.default_format.lower() == 'jpg' or self.default_format.lower() == 'jpeg':
                    screenshot.save(filepath, quality=self.default_quality, optimize=True)
                else:
                    screenshot.save(filepath, optimize=True)
                self.logger.info(f"区域截图已保存: {filepath}")

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "type": "region",
                "region": {"x": x, "y": y, "width": width, "height": height},
                "size": screenshot.size,
                "mode": screenshot.mode,
                "filepath": str(filepath) if filepath else None,
                "filename": filename,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            result["image"] = screenshot

            self.logger.info(f"区域截图完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"区域截图失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "type": "region",
                "region": {"x": x, "y": y, "width": width, "height": height},
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def capture_window(self, window_title: Optional[str] = None, save: bool = True,
                      filename: Optional[str] = None) -> Dict[str, Any]:
        """
        截取指定窗口

        Args:
            window_title: 窗口标题，None表示活动窗口
            save: 是否保存到文件
            filename: 自定义文件名

        Returns:
            截图结果字典
        """
        start_time = time.time()

        try:
            if not PYGETWINDOW_AVAILABLE:
                raise VisionError("pygetwindow库未安装，请安装: pip install pygetwindow")

            self.logger.info(f"截取窗口: {window_title or '活动窗口'}")

            # 获取窗口
            if window_title:
                windows = gw.getWindowsWithTitle(window_title)
                if not windows:
                    # 尝试模糊匹配
                    all_windows = gw.getAllWindows()
                    windows = [w for w in all_windows if window_title.lower() in (w.title or "").lower()]

                if not windows:
                    raise VisionError(f"找不到窗口: {window_title}")

                window = windows[0]  # 使用第一个匹配的窗口
            else:
                window = gw.getActiveWindow()
                if not window:
                    raise VisionError("无法获取活动窗口")

            # 确保窗口可见
            if window.isMinimized:
                window.restore()

            window.activate()
            time.sleep(0.5)  # 等待窗口激活

            # 获取窗口区域
            left, top = window.left, window.top
            width, height = window.width, window.height

            # 截图
            screenshot = pyautogui.screenshot(region=(left, top, width, height))

            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                window_name = window.title.replace(" ", "_").replace("/", "_")[:20]
                filename = f"window_{window_name}_{timestamp}.{self.default_format}"

            filepath = self.output_dir / filename if save else None

            # 保存图片
            if save:
                if self.default_format.lower() == 'jpg' or self.default_format.lower() == 'jpeg':
                    screenshot.save(filepath, quality=self.default_quality, optimize=True)
                else:
                    screenshot.save(filepath, optimize=True)
                self.logger.info(f"窗口截图已保存: {filepath}")

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "type": "window",
                "window_info": {
                    "title": window.title,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                },
                "size": screenshot.size,
                "mode": screenshot.mode,
                "filepath": str(filepath) if filepath else None,
                "filename": filename,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            result["image"] = screenshot

            self.logger.info(f"窗口截图完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"窗口截图失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "type": "window",
                "window_title": window_title,
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def capture_with_cursor(self, save: bool = True, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        截取屏幕并包含鼠标光标

        Args:
            save: 是否保存到文件
            filename: 自定义文件名

        Returns:
            截图结果字典
        """
        start_time = time.time()

        try:
            self.logger.info("截取屏幕（包含光标）")

            # 截取屏幕
            screenshot = pyautogui.screenshot()

            # 获取鼠标位置
            cursor_x, cursor_y = pyautogui.position()

            # 在截图上绘制光标
            draw = ImageDraw.Draw(screenshot)
            cursor_size = 20
            cursor_color = "red"

            # 绘制十字光标
            draw.line([(cursor_x - cursor_size, cursor_y),
                      (cursor_x + cursor_size, cursor_y)], fill=cursor_color, width=2)
            draw.line([(cursor_x, cursor_y - cursor_size),
                      (cursor_x, cursor_y + cursor_size)], fill=cursor_color, width=2)

            # 绘制光标圆圈
            draw.ellipse([cursor_x - 10, cursor_y - 10,
                         cursor_x + 10, cursor_y + 10], outline=cursor_color, width=2)

            # 生成文件名
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cursor_{timestamp}.{self.default_format}"

            filepath = self.output_dir / filename if save else None

            # 保存图片
            if save:
                if self.default_format.lower() == 'jpg' or self.default_format.lower() == 'jpeg':
                    screenshot.save(filepath, quality=self.default_quality, optimize=True)
                else:
                    screenshot.save(filepath, optimize=True)
                self.logger.info(f"含光标截图已保存: {filepath}")

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "type": "cursor",
                "cursor_position": {"x": cursor_x, "y": cursor_y},
                "size": screenshot.size,
                "mode": screenshot.mode,
                "filepath": str(filepath) if filepath else None,
                "filename": filename,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            result["image"] = screenshot

            self.logger.info(f"含光标截图完成，耗时: {execution_time:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"含光标截图失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "type": "cursor",
                "error": error_msg,
                "execution_time": time.time() - start_time
            }

    def list_windows(self) -> Dict[str, Any]:
        """
        列出所有可见窗口

        Returns:
            窗口列表字典
        """
        try:
            if not PYGETWINDOW_AVAILABLE:
                return {
                    "success": False,
                    "error": "pygetwindow库未安装",
                    "windows": []
                }

            windows = gw.getAllWindows()
            window_list = []

            for window in windows:
                if window.title and window.visible:  # 只包含有标题且可见的窗口
                    window_list.append({
                        "title": window.title,
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "is_active": window.isActive,
                        "is_maximized": window.isMaximized,
                        "is_minimized": window.isMinimized
                    })

            return {
                "success": True,
                "windows": window_list,
                "count": len(window_list)
            }

        except Exception as e:
            error_msg = f"获取窗口列表失败: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "windows": []
            }

    def get_screen_info(self) -> Dict[str, Any]:
        """
        获取屏幕信息

        Returns:
            屏幕信息字典
        """
        return {
            "screen_size": self.screen_size,
            "system_info": self.system_info,
            "output_dir": str(self.output_dir),
            "default_format": self.default_format,
            "default_quality": self.default_quality,
            "cursor_capture": self.cursor_capture,
            "available_features": {
                "gui_available": GUI_AVAILABLE,
                "opencv_available": OPENCV_AVAILABLE,
                "pygetwindow_available": PYGETWINDOW_AVAILABLE
            }
        }

    def set_output_format(self, format: str, quality: int = 95):
        """
        设置输出格式和质量

        Args:
            format: 图片格式 (png, jpg, jpeg)
            quality: 图片质量 (1-100, 仅对JPG有效)
        """
        if format.lower() in ['png', 'jpg', 'jpeg']:
            self.default_format = format.lower()
            if quality and 1 <= quality <= 100:
                self.default_quality = quality
            self.logger.info(f"输出格式已设置: {format}, 质量: {quality}")
        else:
            raise ValueError("不支持的图片格式")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        pass