"""
MCP Floating Ball - 视觉识别集成模块

整合截图、OCR、图像分析等功能的统一接口。
"""

import time
from typing import Optional, Dict, Any, List, Union, Callable
from pathlib import Path
from datetime import datetime

from src.vision.screen_capture import ScreenCapture
from src.vision.ocr_engine import OCREngine
from src.vision.image_analyzer import ImageAnalyzer
from src.core.logging import get_logger
from src.core.exceptions import VisionError


class VisionIntegration:
    """视觉识别集成类"""

    def __init__(self, output_dir: Optional[str] = None, ocr_engine: str = "tesseract"):
        """
        初始化视觉识别集成

        Args:
            output_dir: 输出目录
            ocr_engine: OCR引擎类型
        """
        self.logger = get_logger("vision.integration")

        try:
            # 初始化各个组件
            self.screen_capture = ScreenCapture(output_dir)
            self.ocr_engine = OCREngine(engine=ocr_engine)
            self.image_analyzer = ImageAnalyzer()

            # 配置
            self.output_dir = Path(output_dir) if output_dir else Path("./vision_output")
            self.output_dir.mkdir(exist_ok=True)

            # 回调函数
            self.screenshot_callbacks = []
            self.ocr_callbacks = []
            self.analysis_callbacks = []

            # 统计信息
            self.operation_history = []

            self.logger.info("视觉识别集成系统初始化完成")

        except Exception as e:
            self.logger.error(f"视觉识别集成系统初始化失败: {e}")
            raise VisionError(f"视觉识别集成系统初始化失败: {e}")

    def capture_and_ocr(self, capture_type: str = "full", **capture_kwargs) -> Dict[str, Any]:
        """
        截图并识别文字

        Args:
            capture_type: 截图类型 (full, region, window)
            **capture_kwargs: 截图参数

        Returns:
            截图和OCR结果
        """
        start_time = time.time()
        operation_id = f"capture_ocr_{int(start_time)}"

        try:
            self.logger.info(f"开始截图和OCR: {capture_type}")

            # 截图
            if capture_type == "full":
                capture_result = self.screen_capture.capture_full_screen(**capture_kwargs)
            elif capture_type == "region":
                capture_result = self.screen_capture.capture_region(**capture_kwargs)
            elif capture_type == "window":
                capture_result = self.screen_capture.capture_window(**capture_kwargs)
            else:
                raise ValueError(f"不支持的截图类型: {capture_type}")

            if not capture_result["success"]:
                raise VisionError(f"截图失败: {capture_result.get('error', '未知错误')}")

            # OCR识别
            image = capture_result["image"]
            ocr_result = self.ocr_engine.recognize_text(
                image,
                save_result=True,
                output_dir=str(self.output_dir / "ocr_results")
            )

            # 合并结果
            result = {
                "operation_id": operation_id,
                "success": ocr_result["success"],
                "capture_type": capture_type,
                "capture_result": capture_result,
                "ocr_result": ocr_result,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            # 记录操作历史
            self.operation_history.append({
                "operation": "capture_and_ocr",
                "operation_id": operation_id,
                "success": result["success"],
                "timestamp": datetime.now().isoformat()
            })

            # 通知回调
            self._notify_callbacks("screenshot", result["capture_result"])
            self._notify_callbacks("ocr", result["ocr_result"])

            self.logger.info(f"截图和OCR完成，耗时: {result['execution_time']:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"截图和OCR失败: {e}"
            self.logger.error(error_msg)

            error_result = {
                "operation_id": operation_id,
                "success": False,
                "capture_type": capture_type,
                "error": error_msg,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            self.operation_history.append({
                "operation": "capture_and_ocr",
                "operation_id": operation_id,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })

            return error_result

    def capture_and_analyze(self, capture_type: str = "full", analysis_types: List[str] = None,
                           **capture_kwargs) -> Dict[str, Any]:
        """
        截图并分析图像

        Args:
            capture_type: 截图类型
            analysis_types: 分析类型列表
            **capture_kwargs: 截图参数

        Returns:
            截图和分析结果
        """
        start_time = time.time()
        operation_id = f"capture_analyze_{int(start_time)}"

        try:
            self.logger.info(f"开始截图和分析: {capture_type}")

            # 截图
            if capture_type == "full":
                capture_result = self.screen_capture.capture_full_screen(**capture_kwargs)
            elif capture_type == "region":
                capture_result = self.screen_capture.capture_region(**capture_kwargs)
            elif capture_type == "window":
                capture_result = self.screen_capture.capture_window(**capture_kwargs)
            else:
                raise ValueError(f"不支持的截图类型: {capture_type}")

            if not capture_result["success"]:
                raise VisionError(f"截图失败: {capture_result.get('error', '未知错误')}")

            # 图像分析
            image = capture_result["image"]
            analysis_result = self.image_analyzer.analyze_image(
                image,
                analysis_types=analysis_types or ["basic", "colors"]
            )

            # 合并结果
            result = {
                "operation_id": operation_id,
                "success": analysis_result["success"],
                "capture_type": capture_type,
                "capture_result": capture_result,
                "analysis_result": analysis_result,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            # 记录操作历史
            self.operation_history.append({
                "operation": "capture_and_analyze",
                "operation_id": operation_id,
                "success": result["success"],
                "timestamp": datetime.now().isoformat()
            })

            # 通知回调
            self._notify_callbacks("screenshot", result["capture_result"])
            self._notify_callbacks("analysis", result["analysis_result"])

            self.logger.info(f"截图和分析完成，耗时: {result['execution_time']:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"截图和分析失败: {e}"
            self.logger.error(error_msg)

            error_result = {
                "operation_id": operation_id,
                "success": False,
                "capture_type": capture_type,
                "error": error_msg,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            self.operation_history.append({
                "operation": "capture_and_analyze",
                "operation_id": operation_id,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })

            return error_result

    def full_vision_analysis(self, capture_type: str = "full", analysis_types: List[str] = None,
                            **capture_kwargs) -> Dict[str, Any]:
        """
        完整的视觉分析（截图 + OCR + 图像分析）

        Args:
            capture_type: 截图类型
            analysis_types: 图像分析类型
            **capture_kwargs: 截图参数

        Returns:
            完整分析结果
        """
        start_time = time.time()
        operation_id = f"full_analysis_{int(start_time)}"

        try:
            self.logger.info(f"开始完整视觉分析: {capture_type}")

            # 截图
            if capture_type == "full":
                capture_result = self.screen_capture.capture_full_screen(**capture_kwargs)
            elif capture_type == "region":
                capture_result = self.screen_capture.capture_region(**capture_kwargs)
            elif capture_type == "window":
                capture_result = self.screen_capture.capture_window(**capture_kwargs)
            else:
                raise ValueError(f"不支持的截图类型: {capture_type}")

            if not capture_result["success"]:
                raise VisionError(f"截图失败: {capture_result.get('error', '未知错误')}")

            image = capture_result["image"]

            # 并行执行OCR和图像分析
            import threading
            results = {}

            def run_ocr():
                try:
                    results["ocr"] = self.ocr_engine.recognize_text(
                        image,
                        save_result=True,
                        output_dir=str(self.output_dir / "ocr_results")
                    )
                except Exception as e:
                    results["ocr"] = {"success": False, "error": str(e)}

            def run_analysis():
                try:
                    results["analysis"] = self.image_analyzer.analyze_image(
                        image,
                        analysis_types=analysis_types or ["basic", "colors", "edges"]
                    )
                except Exception as e:
                    results["analysis"] = {"success": False, "error": str(e)}

            # 启动线程
            ocr_thread = threading.Thread(target=run_ocr)
            analysis_thread = threading.Thread(target=run_analysis)

            ocr_thread.start()
            analysis_thread.start()

            ocr_thread.join()
            analysis_thread.join()

            # 合并结果
            result = {
                "operation_id": operation_id,
                "success": True,
                "capture_type": capture_type,
                "capture_result": capture_result,
                "ocr_result": results.get("ocr", {}),
                "analysis_result": results.get("analysis", {}),
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            # 记录操作历史
            self.operation_history.append({
                "operation": "full_vision_analysis",
                "operation_id": operation_id,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })

            # 通知回调
            self._notify_callbacks("screenshot", result["capture_result"])
            if "ocr" in results:
                self._notify_callbacks("ocr", results["ocr"])
            if "analysis" in results:
                self._notify_callbacks("analysis", results["analysis"])

            self.logger.info(f"完整视觉分析完成，耗时: {result['execution_time']:.2f}秒")
            return result

        except Exception as e:
            error_msg = f"完整视觉分析失败: {e}"
            self.logger.error(error_msg)

            error_result = {
                "operation_id": operation_id,
                "success": False,
                "capture_type": capture_type,
                "error": error_msg,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

            self.operation_history.append({
                "operation": "full_vision_analysis",
                "operation_id": operation_id,
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })

            return error_result

    def ocr_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """从文件进行OCR识别"""
        try:
            return self.ocr_engine.recognize_text(
                file_path,
                save_result=True,
                output_dir=str(self.output_dir / "ocr_results")
            )
        except Exception as e:
            self.logger.error(f"文件OCR失败: {e}")
            return {"success": False, "error": str(e)}

    def analyze_from_file(self, file_path: Union[str, Path],
                         analysis_types: List[str] = None) -> Dict[str, Any]:
        """从文件进行图像分析"""
        try:
            return self.image_analyzer.analyze_image(
                file_path,
                analysis_types=analysis_types
            )
        except Exception as e:
            self.logger.error(f"文件分析失败: {e}")
            return {"success": False, "error": str(e)}

    def add_screenshot_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加截图回调"""
        self.screenshot_callbacks.append(callback)

    def add_ocr_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加OCR回调"""
        self.ocr_callbacks.append(callback)

    def add_analysis_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加分析回调"""
        self.analysis_callbacks.append(callback)

    def _notify_callbacks(self, callback_type: str, data: Dict[str, Any]):
        """通知回调函数"""
        try:
            if callback_type == "screenshot":
                for callback in self.screenshot_callbacks:
                    callback(data)
            elif callback_type == "ocr":
                for callback in self.ocr_callbacks:
                    callback(data)
            elif callback_type == "analysis":
                for callback in self.analysis_callbacks:
                    callback(data)
        except Exception as e:
            self.logger.error(f"回调通知失败: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "components": {
                "screen_capture": {
                    "available": True,
                    "screen_info": self.screen_capture.get_screen_info()
                },
                "ocr_engine": {
                    "available": True,
                    "engine_info": self.ocr_engine.get_engine_info()
                },
                "image_analyzer": {
                    "available": True,
                    "capabilities": self.image_analyzer.get_capabilities()
                }
            },
            "output_dir": str(self.output_dir),
            "operation_count": len(self.operation_history),
            "last_operation": self.operation_history[-1] if self.operation_history else None
        }

    def get_operation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self.operation_history[-limit:] if self.operation_history else []

    def clear_history(self):
        """清空操作历史"""
        self.operation_history.clear()
        self.logger.info("操作历史已清空")

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.image_analyzer, 'cleanup'):
                self.image_analyzer.cleanup()

            self.logger.info("视觉识别集成系统资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()