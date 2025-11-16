#!/usr/bin/env python3
"""
MCP Floating Ball - 语音功能测试脚本

测试语音识别、唤醒词检测等功能。
"""

import time
import signal
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.voice.voice_activation import VoiceActivation
from src.core.logging import get_logger

logger = get_logger(__name__)


class VoiceTester:
    """语音功能测试器"""

    def __init__(self):
        self.voice_activation = None
        self.running = True

    def on_activation(self, wake_word):
        """激活回调"""
        logger.info(f"🎯 语音助手已激活！唤醒词: {wake_word}")
        print(f"\n🎯 语音助手已激活！唤醒词: {wake_word}")
        print("💬 请说出您的命令（将在10秒后自动停用）...")

    def on_deactivation(self):
        """停用回调"""
        logger.info("😴 语音助手已停用")
        print("\n😴 语音助手已停用")
        print("🎤 请说出唤醒词来激活助手...")

    def on_command(self, command, voice_activation):
        """命令回调"""
        logger.info(f"📝 收到命令: {command}")
        print(f"\n📝 收到命令: {command}")

        # 处理一些简单的测试命令
        if "你好" in command or "hello" in command.lower():
            print("🤖 你好！我是MCP Floating Ball助手。")
        elif "停止" in command or "stop" in command.lower():
            print("🛑 正在停止语音测试...")
            voice_activation.deactivate()
        elif "测试" in command or "test" in command.lower():
            print("✅ 语音识别测试成功！")
        elif "状态" in command or "status" in command.lower():
            status = voice_activation.get_status()
            print(f"📊 系统状态: {status}")
        else:
            print(f"🤔 收到未知命令: {command}")

    def signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info("收到停止信号")
        print("\n🛑 正在停止语音测试...")
        self.running = False
        if self.voice_activation:
            self.voice_activation.stop()

    def run_test(self):
        """运行语音测试"""
        print("🚀 启动MCP Floating Ball语音功能测试")
        print("=" * 50)

        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # 初始化语音激活系统
            print("🔧 正在初始化语音激活系统...")
            self.voice_activation = VoiceActivation(
                wake_words=[
                    "你好小助手",
                    "小助手",
                    "助手",
                    "hello assistant",
                    "assistant",
                    "computer"
                ],
                auto_start=False
            )

            # 添加回调
            self.voice_activation.add_activation_callback(self.on_activation)
            self.voice_activation.add_deactivation_callback(self.on_deactivation)
            self.voice_activation.add_command_callback(self.on_command)

            # 测试系统
            print("🧪 正在测试语音系统...")
            test_results = self.voice_activation.test_system()

            print("📊 测试结果:")
            print(f"  - 语音识别: {'✅' if test_results['speech_recognition']['available'] else '❌'}")
            print(f"  - 模型加载: {'✅' if test_results['speech_recognition']['model_loaded'] else '❌'}")
            print(f"  - 唤醒词检测: {'✅' if test_results['wake_detector']['available'] else '❌'}")
            print(f"  - 系统就绪: {'✅' if test_results['overall']['ready'] else '❌'}")

            if not test_results['overall']['ready']:
                print("❌ 系统未就绪，问题:")
                for issue in test_results['overall']['issues']:
                    print(f"  - {issue}")
                return

            # 启动语音监听
            print("\n🎤 启动语音监听...")
            self.voice_activation.start()

            print("\n💡 使用说明:")
            print("  - 说'你好小助手'、'小助手'或'assistant'来激活")
            print("  - 激活后可以说出命令")
            print("  - 按 Ctrl+C 退出测试")
            print("🎤 请说出唤醒词来激活助手...")

            # 主循环
            while self.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        except Exception as e:
            logger.error(f"语音测试失败: {e}")
            print(f"❌ 语音测试失败: {e}")
        finally:
            if self.voice_activation:
                print("🛑 停止语音监听...")
                self.voice_activation.stop()

        print("\n✅ 语音测试完成")


def run_single_recognition_test():
    """运行单次语音识别测试"""
    print("🧪 单次语音识别测试")
    print("=" * 30)

    try:
        from src.voice.speech_recognition import VoiceRecognition

        with VoiceRecognition() as recognizer:
            print("🎤 请说话（5秒录音）...")
            result = recognizer.recognize_once(timeout=5.0)

            if result:
                print(f"✅ 识别结果: {result}")
            else:
                print("❌ 未识别到语音内容")

    except Exception as e:
        logger.error(f"单次识别测试失败: {e}")
        print(f"❌ 测试失败: {e}")


def run_wake_word_test():
    """运行唤醒词测试"""
    print("🧪 唤醒词测试")
    print("=" * 30)

    try:
        from src.voice.wake_word_detector import WakeWordDetector

        detector = WakeWordDetector([
            "你好小助手",
            "小助手",
            "hello assistant"
        ])

        # 测试文本
        test_texts = [
            "你好小助手，帮我打开文件",
            "小助手在哪里",
            "hello assistant",
            "今天天气不错",
            "助手助手助手"
        ]

        print("测试唤醒词检测:")
        for text in test_texts:
            result = detector.test_wake_word(text)
            would_activate = result["would_activate"]
            best_match = result["best_match"]

            print(f"  文本: '{text}'")
            print(f"  激活: {'✅' if would_activate else '❌'}")
            print(f"  最佳匹配: {best_match['wake_word']} (相似度: {best_match['similarity']:.2f})")
            print()

    except Exception as e:
        logger.error(f"唤醒词测试失败: {e}")
        print(f"❌ 测试失败: {e}")


def main():
    """主函数"""
    print("🎙️ MCP Floating Ball 语音功能测试")
    print("=" * 40)
    print("请选择测试模式:")
    print("1. 完整语音激活测试（推荐）")
    print("2. 单次语音识别测试")
    print("3. 唤醒词检测测试")
    print("0. 退出")

    try:
        choice = input("\n请输入选择 (0-3): ").strip()

        if choice == "1":
            tester = VoiceTester()
            tester.run_test()
        elif choice == "2":
            run_single_recognition_test()
        elif choice == "3":
            run_wake_word_test()
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