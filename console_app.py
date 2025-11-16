#!/usr/bin/env python3
"""
MCP Floating Ball - 控制台交互界面

提供用户友好的命令行界面与AI助手交互。
"""

import asyncio
import sys
import signal
import threading
import time
from typing import Optional
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.assistant.ai_assistant import AIAssistant
from src.core.logging import get_logger
from src.core.config_manager import get_config_manager

logger = get_logger("console_app")


class ConsoleApp:
    """控制台应用程序类"""

    def __init__(self):
        """初始化控制台应用"""
        self.logger = get_logger("console_app")
        self.config_manager = get_config_manager()

        # AI助手
        self.assistant: Optional[AIAssistant] = None

        # 运行状态
        self.is_running = False
        self.voice_enabled = False

        # 控制台配置
        self.prompt = "🤖 MCP> "
        self.show_timestamps = self.config_manager.get("console.show_timestamps", False)

        # ANSI颜色代码
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'dim': '\033[2m'
        }

        self.logger.info("控制台应用初始化完成")

    def color_print(self, text: str, color: str = 'white', bold: bool = False):
        """彩色打印"""
        try:
            prefix = ""
            if bold:
                prefix += self.colors['bold']
            if color in self.colors:
                prefix += self.colors[color]

            print(f"{prefix}{text}{self.colors['reset']}")
        except Exception:
            # 如果彩色输出不支持，使用普通输出
            print(text)

    def print_header(self):
        """打印程序头部信息"""
        self.color_print("=" * 60, 'cyan', True)
        self.color_print("🎈 MCP Floating Ball AI助手", 'cyan', True)
        self.color_print("=" * 60, 'cyan', True)
        print()
        self.color_print("✨ 功能特性：", 'green')
        print("   • 🖥️  系统控制 - 应用启动、窗口管理、系统信息")
        print("   • 📁 文件处理 - 格式转换、文本操作")
        print("   • 🌐 网络工具 - 搜索引擎、网页抓取")
        print("   • 👁️  视觉识别 - 屏幕截图、OCR识别、图像分析")
        print("   • 🎤 语音控制 - 语音命令识别（可选）")
        print()

    def print_help(self):
        """打印帮助信息"""
        self.color_print("📋 控制台命令帮助：", 'blue', True)
        print()
        self.color_print("🔧 特殊命令：", 'yellow')
        print("   help          - 显示此帮助信息")
        print("   status        - 查看系统状态")
        print("   voice         - 切换语音控制")
        print("   screenshot    - 快速截图")
        print("   ocr           - 截图并识别文字")
        print("   config        - 显示配置信息")
        print("   cls/clear     - 清空屏幕")
        print("   exit/quit     - 退出程序")
        print()

        self.color_print("💬 自然语言命令示例：", 'yellow')
        print("   • 打开记事本")
        print("   • 搜索Python教程")
        print("   • 截取全屏并识别上面的文字")
        print("   • 帮我查看系统信息")
        print("   • 将test.pdf转换为Word文档")
        print("   • 最小化所有窗口")
        print()

    async def initialize_assistant(self) -> bool:
        """初始化AI助手"""
        try:
            self.color_print("🔄 正在初始化AI助手...", 'blue')

            # 创建AI助手实例
            self.assistant = AIAssistant()

            # 设置回调函数
            self.assistant.set_response_callback(self.on_assistant_response)
            self.assistant.set_status_callback(self.on_assistant_status)

            # 启动AI助手
            self.assistant.start()

            # 检查语音功能
            capabilities = self.assistant.get_capabilities()
            self.voice_enabled = capabilities.get("voice_commands", False)

            if self.voice_enabled:
                self.color_print("🎤 语音控制功能已启用", 'green')
            else:
                self.color_print("🔇 语音控制功能未启用", 'yellow')

            self.color_print("✅ AI助手初始化完成", 'green')
            return True

        except Exception as e:
            self.color_print(f"❌ AI助手初始化失败：{e}", 'red')
            self.logger.error(f"AI助手初始化失败: {e}")
            return False

    async def on_assistant_response(self, response: dict):
        """AI助手响应回调"""
        try:
            success = response.get("success", False)
            response_text = response.get("response", "")
            response_type = response.get("response_type", "info")

            if success:
                if response_type == "success":
                    self.color_print(response_text, 'green')
                elif response_type == "help":
                    self.color_print(response_text, 'cyan')
                elif response_type == "status":
                    self.color_print(response_text, 'blue')
                elif response_type == "info":
                    self.color_print(response_text, 'yellow')
                elif response_type == "exit":
                    self.color_print(response_text, 'magenta')
                    self.is_running = False
                else:
                    print(response_text)
            else:
                self.color_print(f"❌ {response_text}", 'red')

                # 显示建议
                suggestions = response.get("suggestions", [])
                if suggestions:
                    self.color_print("💡 建议：", 'yellow')
                    for suggestion in suggestions:
                        print(f"   • {suggestion}")

        except Exception as e:
            self.logger.error(f"响应回调失败: {e}")

    async def on_assistant_status(self, component: str, status: str, message: str):
        """AI助手状态回调"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            status_icon = "🟢" if status in ["started", "active", "success"] else "🟡"

            if component == "voice":
                self.color_print(f"[{timestamp}] {status_icon} 语音：{message}", 'dim')
            elif component == "vision":
                self.color_print(f"[{timestamp}] {status_icon} 视觉：{message}", 'dim')
            elif component == "assistant":
                self.color_print(f"[{timestamp}] {status_icon} 助手：{message}", 'dim')

        except Exception as e:
            self.logger.error(f"状态回调失败: {e}")

    async def handle_special_command(self, command: str) -> bool:
        """处理特殊控制台命令"""
        try:
            cmd_lower = command.lower().strip()

            if cmd_lower in ["help", "帮助"]:
                self.print_help()
                return True

            elif cmd_lower in ["status", "状态"]:
                await self.show_status()
                return True

            elif cmd_lower in ["voice", "语音"]:
                await self.toggle_voice()
                return True

            elif cmd_lower in ["screenshot", "截图", "screen", "屏幕"]:
                await self.quick_screenshot()
                return True

            elif cmd_lower in ["ocr", "识别"]:
                await self.quick_ocr()
                return True

            elif cmd_lower in ["config", "配置"]:
                self.show_config()
                return True

            elif cmd_lower in ["cls", "clear", "清空", "清屏"]:
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                self.print_header()
                return True

            elif cmd_lower in ["exit", "quit", "退出", "再见"]:
                self.color_print("👋 正在退出程序...", 'yellow')
                self.is_running = False
                return True

            return False

        except Exception as e:
            self.color_print(f"❌ 特殊命令处理失败：{e}", 'red')
            return False

    async def show_status(self):
        """显示系统状态"""
        try:
            if not self.assistant:
                self.color_print("❌ AI助手未初始化", 'red')
                return

            status = self.assistant.get_system_status()

            self.color_print("📊 系统状态：", 'blue', True)
            print(f"🔧 会话ID: {status['session_id']}")
            print(f"⏱️  运行时长: {status['uptime']}")
            print(f"🎤 语音状态: {'启用' if status['is_voice_active'] else '禁用'}")

            # 组件状态
            self.color_print("\n🔧 组件状态：", 'blue')
            for component, enabled in status['components'].items():
                icon = "✅" if enabled else "❌"
                print(f"   {icon} {component}")

            # 命令统计
            if 'command_stats' in status:
                stats = status['command_stats']
                self.color_print(f"\n📈 命令统计：", 'blue')
                print(f"   总命令数: {stats['total_commands']}")
                print(f"   成功执行: {stats['successful_commands']}")
                print(f"   执行失败: {stats['failed_commands']}")
                print(f"   成功率: {stats['success_rate']:.1f}%")

        except Exception as e:
            self.color_print(f"❌ 获取状态失败：{e}", 'red')

    async def toggle_voice(self):
        """切换语音控制"""
        try:
            if not self.assistant or not self.voice_enabled:
                self.color_print("❌ 语音功能不可用", 'red')
                return

            if self.assistant.is_voice_active:
                self.assistant.stop_voice_listening()
                self.color_print("🔇 语音控制已停止", 'yellow')
            else:
                if self.assistant.start_voice_listening():
                    self.color_print("🎤 语音控制已启动，请说出唤醒词", 'green')
                else:
                    self.color_print("❌ 语音控制启动失败", 'red')

        except Exception as e:
            self.color_print(f"❌ 语音控制切换失败：{e}", 'red')

    async def quick_screenshot(self):
        """快速截图"""
        try:
            if not self.assistant:
                self.color_print("❌ AI助手未初始化", 'red')
                return

            self.color_print("📸 正在截图...", 'blue')
            result = await self.assistant.capture_screenshot("full")

            if result.get("success"):
                filename = result.get("filename", "")
                self.color_print(f"✅ 截图完成：{filename}", 'green')
            else:
                self.color_print(f"❌ 截图失败：{result.get('error')}", 'red')

        except Exception as e:
            self.color_print(f"❌ 截图失败：{e}", 'red')

    async def quick_ocr(self):
        """快速OCR识别"""
        try:
            if not self.assistant:
                self.color_print("❌ AI助手未初始化", 'red')
                return

            self.color_print("📸 正在截图并识别文字...", 'blue')
            result = await self.assistant.full_vision_analysis(
                capture_type="full",
                analysis_types=["basic"],
                perform_ocr=True
            )

            if result.get("success"):
                # 显示截图结果
                capture_result = result.get("capture_result", {})
                if capture_result.get("success"):
                    filename = capture_result.get("filename", "")
                    self.color_print(f"📸 截图完成：{filename}", 'green')

                # 显示OCR结果
                ocr_result = result.get("ocr_result", {})
                if ocr_result.get("success"):
                    text = ocr_result.get("text", "")
                    word_count = ocr_result.get("word_count", 0)
                    confidence = ocr_result.get("confidence_avg", 0)

                    self.color_print(f"📝 识别到 {word_count} 个文字", 'green')
                    self.color_print(f"🎯 识别置信度：{confidence:.1f}%", 'blue')

                    if text.strip():
                        self.color_print("📄 识别内容：", 'yellow')
                        print(f"   {text}")
                    else:
                        self.color_print("📄 未识别到文字", 'yellow')
                else:
                    self.color_print(f"❌ OCR失败：{ocr_result.get('error')}", 'red')
            else:
                self.color_print(f"❌ 操作失败：{result.get('error')}", 'red')

        except Exception as e:
            self.color_print(f"❌ OCR操作失败：{e}", 'red')

    def show_config(self):
        """显示配置信息"""
        try:
            self.color_print("⚙️  配置信息：", 'blue', True)

            # 显示主要配置项
            config_items = [
                ("voice.enabled", "语音功能"),
                ("voice.wake_word", "唤醒词"),
                ("vision.enabled", "视觉功能"),
                ("console.show_timestamps", "显示时间戳"),
                ("assistant.max_history_size", "最大历史记录")
            ]

            for key, description in config_items:
                value = self.config_manager.get(key, "未设置")
                print(f"   {description}: {value}")

        except Exception as e:
            self.color_print(f"❌ 获取配置失败：{e}", 'red')

    def setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            def signal_handler(signum, frame):
                self.color_print("\n🛑 收到中断信号，正在退出...", 'yellow')
                self.is_running = False

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        except Exception as e:
            self.logger.error(f"设置信号处理器失败: {e}")

    async def run(self):
        """运行控制台应用"""
        try:
            # 打印头部信息
            self.print_header()

            # 初始化AI助手
            if not await self.initialize_assistant():
                return 1

            # 设置信号处理器
            self.setup_signal_handlers()

            # 启动主循环
            self.is_running = True
            self.color_print("\n🎉 AI助手已就绪！您可以开始对话了。", 'green')
            self.color_print('💡 输入 "help" 查看可用命令', 'yellow')
            print()

            # 主循环
            while self.is_running:
                try:
                    # 获取用户输入
                    try:
                        user_input = input(self.prompt).strip()
                    except (EOFError, KeyboardInterrupt):
                        break

                    # 跳过空输入
                    if not user_input:
                        continue

                    # 处理特殊命令
                    if await self.handle_special_command(user_input):
                        continue

                    # 处理AI助手命令
                    await self.assistant.process_text_command(user_input)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.color_print(f"❌ 处理输入失败：{e}", 'red')
                    self.logger.error(f"处理输入失败: {e}")

            return 0

        except Exception as e:
            self.color_print(f"❌ 应用程序运行失败：{e}", 'red')
            self.logger.error(f"应用程序运行失败: {e}")
            return 1

        finally:
            # 清理资源
            await self.cleanup()

    async def cleanup(self):
        """清理资源"""
        try:
            self.color_print("🔄 正在清理资源...", 'blue')

            if self.assistant:
                self.assistant.stop()

            self.color_print("✅ 资源清理完成", 'green')

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")


async def main():
    """主函数"""
    app = ConsoleApp()
    return await app.run()


if __name__ == "__main__":
    # 运行控制台应用
    exit_code = asyncio.run(main())
    sys.exit(exit_code)