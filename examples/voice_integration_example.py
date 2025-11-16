#!/usr/bin/env python3
"""
MCP Floating Ball - 语音集成示例

演示如何在主应用中集成语音唤醒功能。
"""

import time
import threading
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from voice.voice_activation import VoiceActivation
from core.logging import get_logger
from interfaces.ai_assistant import AIAssistant

logger = get_logger(__name__)


class VoiceIntegratedAssistant:
    """语音集成助手"""

    def __init__(self):
        """初始化语音集成助手"""
        self.logger = get_logger("voice.integrated_assistant")

        # 初始化AI助手
        self.ai_assistant = AIAssistant()

        # 初始化语音激活系统
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

        # 设置语音回调
        self.voice_activation.add_activation_callback(self.on_voice_activation)
        self.voice_activation.add_deactivation_callback(self.on_voice_deactivation)
        self.voice_activation.add_command_callback(self.on_voice_command)

        # 状态变量
        self.is_active = False
        self.current_session = None

        self.logger.info("语音集成助手初始化完成")

    def start(self):
        """启动语音助手"""
        try:
            # 测试系统
            test_results = self.voice_activation.test_system()
            if not test_results["overall"]["ready"]:
                self.logger.error("语音系统未就绪")
                print("❌ 语音系统未就绪，请检查配置")
                return False

            # 启动语音监听
            self.voice_activation.start()
            self.is_active = True

            print("🎙️ 语音助手已启动")
            print("💡 说出唤醒词来激活助手，如：'你好小助手'")
            print("🛑 按 Ctrl+C 退出")

            return True

        except Exception as e:
            self.logger.error(f"启动语音助手失败: {e}")
            print(f"❌ 启动失败: {e}")
            return False

    def stop(self):
        """停止语音助手"""
        try:
            self.voice_activation.stop()
            self.is_active = False
            print("🛑 语音助手已停止")

        except Exception as e:
            self.logger.error(f"停止语音助手失败: {e}")

    def on_voice_activation(self, wake_word):
        """语音激活回调"""
        self.logger.info(f"语音助手已激活，唤醒词: {wake_word}")
        print(f"\n🎯 语音助手已激活！（唤醒词: {wake_word}）")
        print("💬 请说出您的命令...")

        # 创建新的对话会话
        self.current_session = self.ai_assistant.create_session()

    def on_voice_deactivation(self):
        """语音停用回调"""
        self.logger.info("语音助手已停用")
        print("\n😴 语音助手已停用")
        print("🎤 说出唤醒词重新激活助手...")

        # 结束当前对话会话
        if self.current_session:
            self.current_session = None

    def on_voice_command(self, command_text, voice_activation):
        """语音命令回调"""
        self.logger.info(f"收到语音命令: {command_text}")
        print(f"📝 您: {command_text}")

        try:
            if not self.current_session:
                self.current_session = self.ai_assistant.create_session()

            # 处理命令
            response = self.ai_assistant.process_command(
                command_text,
                session_id=self.current_session
            )

            print(f"🤖 助手: {response}")

            # 检查是否需要停用助手
            if self._should_deactivate(command_text):
                voice_activation.deactivate()

        except Exception as e:
            self.logger.error(f"处理语音命令失败: {e}")
            print(f"❌ 处理命令失败: {e}")

    def _should_deactivate(self, command_text: str) -> bool:
        """判断是否应该停用助手"""
        deactivation_keywords = [
            "再见", "bye", "结束", "退出", "停止", "拜拜",
            "goodbye", "exit", "quit", "stop", "end"
        ]

        return any(keyword in command_text.lower() for keyword in deactivation_keywords)

    def get_status(self):
        """获取助手状态"""
        voice_status = self.voice_activation.get_status()
        ai_status = {
            "is_active": self.is_active,
            "current_session": self.current_session is not None
        }

        return {
            "voice": voice_status,
            "ai": ai_status,
            "overall": {
                "ready": self.is_active and voice_status["is_running"]
            }
        }


def main():
    """主函数"""
    print("🤖 MCP Floating Ball 语音集成助手")
    print("=" * 40)

    assistant = VoiceIntegratedAssistant()

    try:
        # 启动助手
        if not assistant.start():
            return

        # 主循环
        while True:
            time.sleep(1)

            # 可以在这里添加其他后台任务
            # 例如检查系统状态、处理定时任务等

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        logger.error(f"语音助手运行失败: {e}")
        print(f"❌ 运行失败: {e}")
    finally:
        assistant.stop()


if __name__ == "__main__":
    main()