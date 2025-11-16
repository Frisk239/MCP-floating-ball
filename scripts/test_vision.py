#!/usr/bin/env python3
"""
MCP Floating Ball - 视觉识别功能测试脚本

测试截图、OCR、图像分析等功能。
"""

import time
import signal
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision.vision_integration import VisionIntegration
from src.core.logging import get_logger

logger = get_logger(__name__)


class VisionTester:
    """视觉功能测试器"""

    def __init__(self):
        """初始化视觉功能测试器"""
        self.logger = get_logger("vision.tester")
        self.vision_system = None
        self.running = True

    def signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info("收到停止信号")
        print("\n🛑 正在停止视觉测试...")
        self.running = False
        if self.vision_system:
            self.vision_system.cleanup()

    def test_system_status(self):
        """测试系统状态"""
        print("🧪 测试系统状态")
        print("=" * 30)

        try:
            self.vision_system = VisionIntegration()
            status = self.vision_system.get_system_status()

            print("📊 系统组件状态:")
            for component, info in status["components"].items():
                available = info["available"]
                print(f"  - {component}: {'✅' if available else '❌'}")

            print(f"📁 输出目录: {status['output_dir']}")
            print(f"📈 操作历史: {status['operation_count']} 条")

            return status["components"]["screen_capture"]["available"] and \
                   status["components"]["ocr_engine"]["available"] and \
                   status["components"]["image_analyzer"]["available"]

        except Exception as e:
            self.logger.error(f"系统状态测试失败: {e}")
            print(f"❌ 测试失败: {e}")
            return False

    def test_screen_capture(self):
        """测试截图功能"""
        print("\n📸 测试截图功能")
        print("=" * 30)

        try:
            # 全屏截图测试
            print("🖥️  全屏截图测试...")
            result = self.vision_system.screen_capture.capture_full_screen(save=True)

            if result["success"]:
                print(f"✅ 全屏截图成功: {result['filename']}")
                print(f"   尺寸: {result['size']}")
                print(f"   文件: {result['filepath']}")
            else:
                print(f"❌ 全屏截图失败: {result.get('error', '未知错误')}")

            # 区域截图测试
            print("\n✂️  区域截图测试...")
            region_result = self.vision_system.screen_capture.capture_region(
                x=100, y=100, width=400, height=300, save=True
            )

            if region_result["success"]:
                print(f"✅ 区域截图成功: {region_result['filename']}")
                print(f"   区域: {region_result['region']}")
            else:
                print(f"❌ 区域截图失败: {region_result.get('error', '未知错误')}")

            return result["success"] or region_result["success"]

        except Exception as e:
            self.logger.error(f"截图测试失败: {e}")
            print(f"❌ 测试失败: {e}")
            return False

    def test_ocr_function(self):
        """测试OCR功能"""
        print("\n📝 测试OCR功能")
        print("=" * 30)

        try:
            # 先截图再OCR
            print("📸 截图用于OCR测试...")
            capture_result = self.vision_system.screen_capture.capture_full_screen(save=True)

            if not capture_result["success"]:
                print("❌ 截图失败，无法进行OCR测试")
                return False

            print("🔍 开始OCR识别...")
            ocr_result = self.vision_system.ocr_engine.recognize_text(
                capture_result["image"],
                save_result=True
            )

            if ocr_result["success"]:
                print("✅ OCR识别成功")
                print(f"   识别文字数量: {ocr_result['word_count']} 个")
                print(f"   平均置信度: {ocr_result['confidence_avg']:.1f}%")
                print(f"   识别文字: {ocr_result['text'][:100]}...")
                if ocr_result.get('result_file'):
                    print(f"   结果文件: {ocr_result['result_file']}")
            else:
                print(f"❌ OCR识别失败: {ocr_result.get('error', '未知错误')}")

            return ocr_result["success"]

        except Exception as e:
            self.logger.error(f"OCR测试失败: {e}")
            print(f"❌ 测试失败: {e}")
            return False

    def test_image_analysis(self):
        """测试图像分析功能"""
        print("\n🎨 测试图像分析功能")
        print("=" * 30)

        try:
            # 先截图再分析
            print("📸 截图用于图像分析测试...")
            capture_result = self.vision_system.screen_capture.capture_full_screen(save=True)

            if not capture_result["success"]:
                print("❌ 截图失败，无法进行图像分析测试")
                return False

            print("🔬 开始图像分析...")
            analysis_result = self.vision_system.image_analyzer.analyze_image(
                capture_result["image"],
                analysis_types=["basic", "colors", "edges"]
            )

            if analysis_result["success"]:
                print("✅ 图像分析成功")

                if "basic_analysis" in analysis_result:
                    basic = analysis_result["basic_analysis"]
                    print(f"   图片尺寸: {basic['dimensions']['width']} x {basic['dimensions']['height']}")
                    print(f"   是否彩色: {basic['is_color']}")
                    print(f"   亮度: {basic['brightness']:.1f}")
                    print(f"   对比度: {basic['contrast']:.1f}")

                if "color_analysis" in analysis_result:
                    color = analysis_result["color_analysis"]
                    print(f"   主要颜色数量: {len(color['dominant_colors'])}")
                    if color['dominant_colors']:
                        print(f"   主要颜色: {color['dominant_colors'][0]['color']}")

                if "edge_analysis" in analysis_result:
                    edge = analysis_result["edge_analysis"]
                    print(f"   边缘密度: {edge['edge_density']:.4f}")
                    print(f"   显著边缘: {edge['has_significant_edges']}")
            else:
                print(f"❌ 图像分析失败: {analysis_result.get('error', '未知错误')}")

            return analysis_result["success"]

        except Exception as e:
            self.logger.error(f"图像分析测试失败: {e}")
            print(f"❌ 测试失败: {e}")
            return False

    def test_integrated_workflow(self):
        """测试集成工作流程"""
        print("\n🔄 测试集成工作流程")
        print("=" * 30)

        try:
            print("🔥 开始完整视觉分析...")
            result = self.vision_system.full_vision_analysis(
                capture_type="full",
                analysis_types=["basic", "colors", "edges"]
            )

            if result["success"]:
                print("✅ 完整视觉分析成功")
                print(f"   截图类型: {result['capture_type']}")
                print(f"   执行时间: {result['execution_time']:.2f}秒")

                if result["ocr_result"]["success"]:
                    print(f"   OCR文字数: {result['ocr_result']['word_count']}")

                if result["analysis_result"]["success"]:
                    print(f"   分析类型: {result['analysis_result'].get('analysis_types', [])}")
            else:
                print(f"❌ 完整视觉分析失败: {result.get('error', '未知错误')}")

            return result["success"]

        except Exception as e:
            self.logger.error(f"集成工作流程测试失败: {e}")
            print(f"❌ 测试失败: {e}")
            return False

    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n🎮 交互式测试模式")
        print("=" * 30)
        print("可用命令:")
        print("  1 - 截图测试")
        print("  2 - OCR测试")
        print("  3 - 图像分析测试")
        print("  4 - 完整分析测试")
        print("  5 - 窗口列表")
        print("  q - 退出")

        while self.running:
            try:
                command = input("\n请输入命令: ").strip().lower()

                if command == "1":
                    self.test_screen_capture()
                elif command == "2":
                    self.test_ocr_function()
                elif command == "3":
                    self.test_image_analysis()
                elif command == "4":
                    self.test_integrated_workflow()
                elif command == "5":
                    self.list_windows()
                elif command == "q":
                    break
                else:
                    print("❌ 未知命令")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"交互式测试失败: {e}")
                print(f"❌ 测试失败: {e}")

    def list_windows(self):
        """列出窗口"""
        try:
            windows = self.vision_system.screen_capture.list_windows()
            if windows["success"]:
                print(f"\n🪟 找到 {windows['count']} 个窗口:")
                for i, window in enumerate(windows["windows"][:10]):  # 只显示前10个
                    print(f"  {i+1}. {window['title']} ({window['width']}x{window['height']})")
            else:
                print(f"❌ 获取窗口列表失败: {windows.get('error')}")
        except Exception as e:
            print(f"❌ 窗口列表失败: {e}")

    def run_full_test(self):
        """运行完整测试"""
        print("🚀 启动MCP Floating Ball视觉功能完整测试")
        print("=" * 50)

        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # 初始化系统
            print("🔧 初始化视觉识别系统...")
            if not self.test_system_status():
                print("❌ 系统初始化失败")
                return

            # 运行各项测试
            tests = [
                ("截图功能", self.test_screen_capture),
                ("OCR功能", self.test_ocr_function),
                ("图像分析功能", self.test_image_analysis),
                ("集成工作流程", self.test_integrated_workflow)
            ]

            results = []
            for test_name, test_func in tests:
                print(f"\n🧪 {test_name}")
                print("-" * 30)
                try:
                    result = test_func()
                    results.append((test_name, result))
                except Exception as e:
                    self.logger.error(f"{test_name}测试异常: {e}")
                    results.append((test_name, False))

            # 测试结果汇总
            print("\n📊 测试结果汇总")
            print("=" * 50)
            success_count = 0
            for test_name, result in results:
                status = "✅" if result else "❌"
                print(f"{status} {test_name}")
                if result:
                    success_count += 1

            print(f"\n🎯 总体结果: {success_count}/{len(results)} 项测试通过")
            print(f"📈 成功率: {success_count/len(results)*100:.1f}%")

            # 询问是否继续交互式测试
            if self.running:
                choice = input("\n是否进入交互式测试模式? (y/n): ").strip().lower()
                if choice == 'y' or choice == 'yes':
                    self.run_interactive_test()

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        except Exception as e:
            self.logger.error(f"完整测试失败: {e}")
            print(f"❌ 测试失败: {e}")
        finally:
            if self.vision_system:
                print("🛑 清理资源...")
                self.vision_system.cleanup()

        print("\n✅ 视觉功能测试完成")


def main():
    """主函数"""
    print("👁️ MCP Floating Ball 视觉识别功能测试")
    print("=" * 40)
    print("请选择测试模式:")
    print("1. 完整功能测试（推荐）")
    print("2. 系统状态检查")
    print("3. 截图功能测试")
    print("4. OCR功能测试")
    print("5. 图像分析测试")
    print("6. 交互式测试")
    print("0. 退出")

    try:
        choice = input("\n请输入选择 (0-6): ").strip()

        tester = VisionTester()

        if choice == "1":
            tester.run_full_test()
        elif choice == "2":
            tester.test_system_status()
        elif choice == "3":
            if tester.test_system_status():
                tester.test_screen_capture()
        elif choice == "4":
            if tester.test_system_status():
                tester.test_ocr_function()
        elif choice == "5":
            if tester.test_system_status():
                tester.test_image_analysis()
        elif choice == "6":
            if tester.test_system_status():
                tester.run_interactive_test()
        elif choice == "0":
            print("👋 退出测试")
        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()