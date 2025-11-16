#!/usr/bin/env python3
"""
MCP Floating Ball - 视觉识别集成示例

演示如何在主应用中集成视觉识别功能。
"""

import time
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.vision_integration import VisionIntegration
from src.core.logging import get_logger

logger = get_logger(__name__)


class VisionIntegratedAssistant:
    """视觉集成助手示例"""

    def __init__(self):
        """初始化视觉集成助手"""
        self.logger = get_logger("vision.integrated_assistant")

        # 初始化视觉识别系统
        self.vision_system = VisionIntegration()

        # 设置回调
        self.vision_system.add_screenshot_callback(self.on_screenshot)
        self.vision_system.add_ocr_callback(self.on_ocr_result)
        self.vision_system.add_analysis_callback(self.on_analysis_result)

        self.logger.info("视觉集成助手初始化完成")

    def on_screenshot(self, screenshot_result):
        """截图结果回调"""
        if screenshot_result["success"]:
            self.logger.info(f"截图完成: {screenshot_result['filename']}")
            print(f"📸 截图完成: {screenshot_result['filename']}")

    def on_ocr_result(self, ocr_result):
        """OCR结果回调"""
        if ocr_result["success"]:
            self.logger.info(f"OCR完成，识别到 {ocr_result['word_count']} 个文字")
            print(f"📝 OCR完成，识别到 {ocr_result['word_count']} 个文字")
            if ocr_result['text']:
                print(f"   文字预览: {ocr_result['text'][:100]}...")

    def on_analysis_result(self, analysis_result):
        """分析结果回调"""
        if analysis_result["success"]:
            self.logger.info("图像分析完成")
            print("🔬 图像分析完成")

            if "basic_analysis" in analysis_result:
                basic = analysis_result["basic_analysis"]
                print(f"   图片尺寸: {basic['dimensions']['width']} x {basic['dimensions']['height']}")
                print(f"   亮度: {basic['brightness']:.1f}")

            if "edge_analysis" in analysis_result:
                edge = analysis_result["edge_analysis"]
                print(f"   边缘密度: {edge['edge_density']:.4f}")

    def capture_and_describe(self, capture_type="full", **kwargs):
        """截图并描述"""
        print(f"\n🎯 开始{capture_type}截图并描述...")

        result = self.vision_system.full_vision_analysis(
            capture_type=capture_type,
            analysis_types=["basic", "colors", "edges"],
            **kwargs
        )

        if result["success"]:
            print("✅ 完整分析完成！")

            # 生成描述
            description = self.generate_description(result)
            print(f"\n🤖 AI描述: {description}")

            return result
        else:
            print(f"❌ 分析失败: {result.get('error')}")
            return result

    def generate_description(self, result):
        """生成图片描述"""
        try:
            description_parts = []

            # 基础信息
            if result.get("capture_result", {}).get("success"):
                capture = result["capture_result"]
                description_parts.append(f"这是一张{capture['size'][0]}x{capture['size'][1]}像素的图片")

            # OCR文字
            if result.get("ocr_result", {}).get("success"):
                ocr = result["ocr_result"]
                if ocr['text']:
                    description_parts.append(f"图片中包含文字，识别到{ocr['word_count']}个文字")
                    if ocr['confidence_avg'] > 80:
                        description_parts.append("文字识别置信度较高")
                else:
                    description_parts.append("图片中未检测到文字")

            # 图像分析
            if result.get("analysis_result", {}).get("success"):
                analysis = result["analysis_result"]

                if "basic_analysis" in analysis:
                    basic = analysis["basic_analysis"]
                    if basic['brightness'] > 128:
                        description_parts.append("图片整体较亮")
                    else:
                        description_parts.append("图片整体较暗")

                if "edge_analysis" in analysis:
                    edge = analysis["edge_analysis"]
                    if edge['has_significant_edges']:
                        description_parts.append("图片包含明显的边缘特征")

                if "color_analysis" in analysis and not analysis.get("color_analysis_error"):
                    color = analysis["color_analysis"]
                    if color['dominant_colors']:
                        main_color = color['dominant_colors'][0]
                        description_parts.append(f"主要颜色为RGB({main_color['color'][0]},{main_color['color'][1]},{main_color['color'][2]})")

            return "。".join(description_parts) + "。"

        except Exception as e:
            self.logger.error(f"生成描述失败: {e}")
            return "无法生成图片描述"

    def interactive_demo(self):
        """交互式演示"""
        print("🎮 视觉识别交互式演示")
        print("=" * 30)
        print("可用命令:")
        print("  1 - 全屏截图分析")
        print("  2 - 区域截图分析")
        print("  3 - 窗口截图分析")
        print("  4 - 查看系统状态")
        print("  5 - 查看操作历史")
        print("  q - 退出")

        while True:
            try:
                command = input("\n请输入命令: ").strip().lower()

                if command == "1":
                    self.capture_and_describe("full")
                elif command == "2":
                    try:
                        print("请输入区域坐标 (x,y,width,height):")
                        coords = input("例如: 100,100,400,300: ").strip()
                        x, y, w, h = map(int, coords.split(','))
                        self.capture_and_describe("region", x=x, y=y, width=w, height=h)
                    except Exception as e:
                        print(f"❌ 区域坐标格式错误: {e}")
                elif command == "3":
                    try:
                        window_title = input("请输入窗口标题（留空使用活动窗口）: ").strip()
                        kwargs = {}
                        if window_title:
                            kwargs["window_title"] = window_title
                        self.capture_and_describe("window", **kwargs)
                    except Exception as e:
                        print(f"❌ 窗口截图失败: {e}")
                elif command == "4":
                    status = self.vision_system.get_system_status()
                    print("\n📊 系统状态:")
                    for component, info in status["components"].items():
                        available = "✅" if info["available"] else "❌"
                        print(f"  {available} {component}")
                    print(f"📁 输出目录: {status['output_dir']}")
                elif command == "5":
                    history = self.vision_system.get_operation_history()
                    print(f"\n📜 操作历史 ({len(history)} 条):")
                    for i, op in enumerate(history[-5:]):
                        status = "✅" if op["success"] else "❌"
                        print(f"  {i+1}. {status} {op['operation']} - {op['timestamp'][:19]}")
                elif command == "q":
                    break
                else:
                    print("❌ 未知命令")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"交互式演示失败: {e}")
                print(f"❌ 操作失败: {e}")

        print("\n👋 演示结束")

    def cleanup(self):
        """清理资源"""
        if self.vision_system:
            self.vision_system.cleanup()


def main():
    """主函数"""
    print("👁️ MCP Floating Ball 视觉识别集成演示")
    print("=" * 40)

    try:
        assistant = VisionIntegratedAssistant()

        print("选择演示模式:")
        print("1. 快速演示（全屏截图分析）")
        print("2. 交互式演示")
        print("0. 退出")

        choice = input("\n请输入选择 (0-2): ").strip()

        if choice == "1":
            # 快速演示
            result = assistant.capture_and_describe("full")
            if result["success"]:
                print("\n🎉 演示成功！视觉识别功能正常工作。")
        elif choice == "2":
            # 交互式演示
            assistant.interactive_demo()
        elif choice == "0":
            print("👋 退出演示")
        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        logger.error(f"演示失败: {e}")
        print(f"❌ 演示失败: {e}")


if __name__ == "__main__":
    main()