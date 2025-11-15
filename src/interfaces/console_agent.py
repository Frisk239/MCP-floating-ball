"""
MCP Floating Ball - 控制台智能代理

提供命令行界面的AI交互功能，可以测试所有AI服务。
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..ai.orchestrator import AIProvider, get_ai_orchestrator, chat, search, understand_image
from ..tools.registry import tool_registry, execute_tool
from ..core.config import get_settings
from ..core.logging import get_logger, setup_logging
from ..core.exceptions import MCPFloatingBallError

logger = get_logger(__name__)


class ConsoleAgent:
    """控制台智能代理"""

    def __init__(self):
        """初始化控制台代理"""
        self.logger = get_logger(f"console_agent")
        self.running = False
        self.conversation_history = []
        self.user_id = "console_user"

    async def initialize(self) -> None:
        """初始化代理"""
        try:
            self.logger.info("初始化控制台智能代理")

            # 设置日志系统
            from ..core.config import get_settings
            config = get_settings()
            if config and hasattr(config, 'logging'):
                setup_logging(config.logging)
            else:
                setup_logging()

            # 检查AI服务状态
            orchestrator = get_ai_orchestrator()
            health_status = await orchestrator.health_check()
            self.logger.info("AI服务健康检查完成", status=health_status["overall_status"])

            # 显示欢迎信息
            self._print_welcome_message()

            # 显示服务状态
            self._print_service_status(health_status)

        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)

    def _print_welcome_message(self) -> None:
        """打印欢迎信息"""
        welcome_text = """
🎉 欢迎使用 MCP Floating Ball AI助手！

🚀 这是一个现代化的AI助手，集成了多个AI服务提供商：
   • 月之暗面 Kimi (主要对话AI)
   • 阿里云 DashScope (语音服务)
   • 秘塔 AI搜索 (增强搜索)

💬 输入帮助命令查看可用功能
🔧 输入 'exit' 或 'quit' 退出
        """
        print(welcome_text)

    def _print_service_status(self, health_status: Dict[str, Any]) -> None:
        """打印服务状态"""
        print("\n🔍 服务状态检查:")
        for provider, status in health_status.get("providers", {}).items():
            status_emoji = "✅" if status["status"] == "healthy" else "❌"
            print(f"   {status_emoji} {provider.upper()}: {status['status']}")

        overall_status = health_status.get("overall_status", "unknown")
        status_emoji = "✅" if overall_status == "healthy" else "⚠️"
        print(f"\n📊 整体状态: {status_emoji} {overall_status}\n")

    def _get_user_input(self) -> str:
        """获取用户输入"""
        try:
            user_input = input("💬 您: ").strip()
            return user_input
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            return "exit"
        except EOFError:
            print("\n\n👋 再见！")
            return "exit"

    async def _process_chat_command(self, message: str) -> None:
        """处理聊天命令"""
        try:
            print("🤖 AI: 正在思考...", end="", flush=True)

            # 构建消息
            messages = []
            if self.conversation_history:
                messages.extend(self.conversation_history[-6:])  # 保持最近6轮对话
            messages.append({"role": "user", "content": message})

            # 调用AI服务
            response = await chat(messages)
            ai_message = response.get("content", "抱歉，我无法回答这个问题。")

            # 清除思考提示
            print("\r" + " " * 50 + "\r", end="")

            # 显示AI回复
            print(f"🤖 AI: {ai_message}")

            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": ai_message})

        except Exception as e:
            print(f"\r❌ 对话失败: {e}")

    async def _process_search_command(self, query: str) -> None:
        """处理搜索命令"""
        try:
            print(f"🔍 正在搜索: {query}", end="", flush=True)

            # 执行搜索
            result = await search(query, max_results=5)

            print("\r" + " " * 50 + "\r", end="")

            # 显示搜索结果
            print(f"🔍 搜索结果 (共 {result.get('total_results', 0)} 条):")
            print("-" * 60)

            for i, item in enumerate(result.get("results", [])[:5], 1):
                title = item.get("title", "无标题")
                url = item.get("url", "")
                snippet = item.get("snippet", "")

                print(f"{i}. {title}")
                if url:
                    print(f"   🔗 {url}")
                if snippet:
                    print(f"   📝 {snippet[:100]}...")
                print()

        except Exception as e:
            print(f"\r❌ 搜索失败: {e}")

    async def _process_image_command(self, image_path: str, prompt: str) -> None:
        """处理图像分析命令"""
        try:
            print(f"🖼️ 正在分析图像: {image_path}", end="", flush=True)

            # 验证图像文件存在
            if not Path(image_path).exists():
                print(f"\r❌ 图像文件不存在: {image_path}")
                return

            # 执行图像分析
            result = await understand_image(image_path, prompt)

            print("\r" + " " * 50 + "\r", end="")

            # 显示分析结果
            print(f"🖼️ 图像分析结果:")
            print("-" * 60)
            description = result.get("description", "无法分析图像内容")
            print(f"📝 {description}")

        except Exception as e:
            print(f"\r❌ 图像分析失败: {e}")

    async def _process_tool_command(self, tool_name: str, args: Dict[str, Any]) -> None:
        """处理工具命令"""
        try:
            print(f"🔧 正在执行工具: {tool_name}", end="", flush=True)

            # 执行工具
            result = await execute_tool_async(tool_name, args)

            print("\r" + " " * 50 + "\r", end="")

            # 显示工具执行结果
            if result.success:
                print(f"✅ 工具执行成功!")
                if result.data:
                    print(f"📊 结果: {result.data}")
                if result.message:
                    print(f"💬 说明: {result.message}")
            else:
                print(f"❌ 工具执行失败: {result.error}")

        except Exception as e:
            print(f"\r❌ 工具执行失败: {e}")

    def _print_help_message(self) -> None:
        """打印帮助信息"""
        help_text = """
📖 MCP Floating Ball 使用帮助:

💬 基本对话:
   直接输入您的问题，AI会智能回答

🔍 搜索功能:
   /search <关键词> - 执行网络搜索
   例: /search Python教程

🖼️ 图像分析:
   /image <图片路径> <描述问题>
   例: /image ./photo.jpg 这张图片里有什么？

🔧 工具调用:
   /tool <工具名> <参数JSON>
   例: /tool get_city_weather '{"city": "北京"}'

📋 查看工具:
   /tools - 列出所有可用工具
   /tools <类别> - 列出指定类别的工具

ℹ️ 其他命令:
   /help - 显示帮助信息
   /status - 查看服务状态
   /history - 显示对话历史
   /clear - 清除对话历史
   /exit, /quit - 退出程序

💡 提示: 所有命令都可以使用缩写，如 /s 代替 /search
        """
        print(help_text)

    async def _list_tools(self, category: Optional[str] = None) -> None:
        """列出工具"""
        try:
            if category:
                from ..tools.base import ToolCategory
                try:
                    cat = ToolCategory(category.lower())
                    tools = tool_registry.list_tools(cat)
                    print(f"🔧 {cat.value.upper()} 类别工具:")
                except ValueError:
                    print(f"❌ 未知的工具类别: {category}")
                    print(f"可用类别: {[c.value for c in ToolCategory]}")
                    return
            else:
                tools = tool_registry.list_tools()
                print("🔧 所有可用工具:")

            if not tools:
                print("   暂无可用工具")
                return

            print("-" * 60)
            for i, tool in enumerate(tools, 1):
                metadata = tool.get_metadata()
                print(f"{i:2d}. {metadata.name}")
                print(f"     📝 {metadata.description}")
                print(f"     🏷️  类别: {metadata.category.value}")
                if metadata.parameters:
                    params = ", ".join([p.name for p in metadata.parameters if p.required])
                    print(f"     🔧 参数: {params}")
                print()

        except Exception as e:
            print(f"❌ 获取工具列表失败: {e}")

    def _show_conversation_history(self) -> None:
        """显示对话历史"""
        if not self.conversation_history:
            print("📝 暂无对话历史")
            return

        print("📝 对话历史:")
        print("-" * 60)
        for i, message in enumerate(self.conversation_history, 1):
            role = "👤 用户" if message["role"] == "user" else "🤖 AI"
            content = message["content"]
            print(f"{i:2d}. {role}: {content[:100]}{'...' if len(content) > 100 else ''}")

    def _clear_history(self) -> None:
        """清除对话历史"""
        self.conversation_history.clear()
        print("✅ 对话历史已清除")

    async def _process_command(self, user_input: str) -> bool:
        """
        处理命令

        Args:
            user_input: 用户输入

        Returns:
            是否继续运行
        """
        input_lower = user_input.lower().strip()

        # 退出命令
        if input_lower in ['exit', 'quit', '退出']:
            self.running = False
            return False

        # 空输入
        if not input_lower:
            return True

        # 帮助命令
        if input_lower in ['help', '/help', '帮助']:
            self._print_help_message()
            return True

        # 状态命令
        if input_lower in ['status', '/status', '状态']:
            orchestrator = get_ai_orchestrator()
            health_status = await orchestrator.health_check()
            self._print_service_status(health_status)
            return True

        # 工具列表命令
        if input_lower in ['tools', '/tools']:
            await self._list_tools()
            return True
        elif input_lower.startswith('/tools '):
            category = input_lower[8:].strip()
            await self._list_tools(category)
            return True

        # 历史记录命令
        if input_lower in ['history', '/history', '历史']:
            self._show_conversation_history()
            return True

        # 清除历史命令
        if input_lower in ['clear', '/clear', '清除']:
            self._clear_history()
            return True

        # 搜索命令
        if input_lower.startswith('/search ') or input_lower.startswith('/s '):
            query = user_input.split(' ', 1)[1] if ' ' in user_input else ""
            if query:
                await self._process_search_command(query)
            else:
                print("❌ 请提供搜索关键词")
            return True

        # 图像分析命令
        if input_lower.startswith('/image '):
            parts = user_input.split(' ', 2)
            if len(parts) >= 3:
                image_path = parts[1]
                prompt = parts[2]
                await self._process_image_command(image_path, prompt)
            else:
                print("❌ 用法: /image <图片路径> <描述问题>")
            return True

        # 工具命令
        if input_lower.startswith('/tool ') or input_lower.startswith('/t '):
            parts = user_input.split(' ', 2)
            if len(parts) >= 3:
                tool_name = parts[1]
                try:
                    args = json.loads(parts[2])
                    await self._process_tool_command(tool_name, args)
                except json.JSONDecodeError:
                    print("❌ 参数格式错误，请使用有效的JSON格式")
            else:
                print("❌ 用法: /tool <工具名> <参数JSON>")
            return True

        # 默认处理为聊天
        await self._process_chat_command(user_input)
        return True

    async def run(self) -> None:
        """运行控制台代理"""
        try:
            await self.initialize()
            self.running = True

            print("🚀 AI助手已启动，开始对话吧！\n")

            while self.running:
                user_input = self._get_user_input()
                await self._process_command(user_input)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        except Exception as e:
            self.logger.error(f"运行时错误: {e}")
            print(f"❌ 运行时错误: {e}")
        finally:
            orchestrator = get_ai_orchestrator()
            await orchestrator.close()


async def execute_tool_async(tool_name: str, args: Dict[str, Any]):
    """异步执行工具的辅助函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, execute_tool, tool_name, args)


async def main():
    """主函数"""
    agent = ConsoleAgent()
    await agent.run()


if __name__ == "__main__":
    # 运行控制台代理
    asyncio.run(main())