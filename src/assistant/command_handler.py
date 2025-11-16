"""
MCP Floating Ball - 命令处理器

负责处理用户命令，协调NLP处理器和工具调用器。
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import re
import traceback

from src.core.logging import get_logger
from src.core.exceptions import AssistantError
from src.assistant.nlp_processor import NLPProcessor, IntentType, Command, CommandType, CompatibleCommand
from src.assistant.enhanced_nlp_processor import EnhancedNLPProcessor
from src.assistant.tool_caller import ToolCaller
from src.core.config_manager import get_config_manager

logger = get_logger("assistant.command_handler")


class CommandHandler:
    """命令处理器类"""

    def __init__(self):
        """初始化命令处理器"""
        self.logger = get_logger("assistant.command_handler")
        self.config_manager = get_config_manager()

        # 初始化组件
        self.nlp_processor = NLPProcessor()
        self.enhanced_nlp_processor = EnhancedNLPProcessor()  # 使用增强处理器
        self.tool_caller = ToolCaller()

        # 会话状态
        self.session_context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []

        # 命令处理统计
        self.stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "intent_distribution": {},
            "start_time": datetime.now()
        }

        # 特殊命令处理器
        self.special_commands = {
            "help": self._handle_help_command,
            "status": self._handle_status_command,
            "history": self._handle_history_command,
            "clear": self._handle_clear_command,
            "tools": self._handle_tools_command,
            "exit": self._handle_exit_command,
            "quit": self._handle_exit_command
        }

        self.logger.info("命令处理器初始化完成")

    async def process_command(self, user_input: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        处理用户命令

        Args:
            user_input: 用户输入的命令文本
            user_id: 用户ID（可选）

        Returns:
            处理结果
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"处理用户命令: {user_input[:50]}...")

            # 更新统计
            self.stats["total_commands"] += 1

            # 检查特殊命令
            special_result = await self._check_special_commands(user_input)
            if special_result:
                return special_result

            # 检查是否为空输入
            if not user_input.strip():
                return {
                    "success": True,
                    "response": "请输入您的命令或问题。",
                    "response_type": "info",
                    "execution_time": 0.01,
                    "timestamp": start_time.isoformat()
                }

            # 使用增强NLP处理器解析命令
            parse_result = self.enhanced_nlp_processor.parse_command(user_input)

            if not parse_result["success"]:
                self.stats["failed_commands"] += 1
                return {
                    "success": False,
                    "response": f"无法理解您的命令：{parse_result.get('error', '未知错误')}",
                    "response_type": "error",
                    "suggestions": self._get_command_suggestions(user_input),
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }

            commands = parse_result["commands"]
            self.logger.info(f"解析出 {len(commands)} 个命令")

            # 更新意图分布统计
            for cmd in commands:
                # 兼容性处理：获取意图类型
                if hasattr(cmd, 'intent_type'):  # CompatibleCommand
                    intent_type = cmd.intent_type
                elif hasattr(cmd, 'intent'):  # Command
                    intent_type = cmd.intent.intent_type
                else:
                    intent_type = IntentType.UNKNOWN

                intent_name = intent_type.name
                self.stats["intent_distribution"][intent_name] = \
                    self.stats["intent_distribution"].get(intent_name, 0) + 1

            # 验证命令参数
            validation_result = await self._validate_commands(commands)
            if not validation_result["valid"]:
                self.stats["failed_commands"] += 1
                return {
                    "success": False,
                    "response": f"命令参数验证失败：{validation_result['error']}",
                    "response_type": "error",
                    "suggestions": validation_result.get("suggestions", []),
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat()
                }

            # 执行命令
            execution_results = await self.tool_caller.execute_commands(commands)

            # 处理执行结果
            response_result = await self._process_execution_results(execution_results, user_input)

            # 更新成功统计
            successful_count = sum(1 for result in execution_results if result.get("success", False))
            if successful_count == len(execution_results):
                self.stats["successful_commands"] += 1
            else:
                self.stats["failed_commands"] += 1

            # 添加到对话历史
            self._add_to_conversation_history(user_input, response_result, user_id)

            # 更新会话上下文
            self._update_session_context(commands, execution_results)

            execution_time = (datetime.now() - start_time).total_seconds()
            response_result["execution_time"] = execution_time
            response_result["timestamp"] = start_time.isoformat()

            self.logger.info(f"命令处理完成，耗时: {execution_time:.2f}秒")
            return response_result

        except Exception as e:
            self.stats["failed_commands"] += 1
            self.logger.error(f"命令处理失败: {e}")

            # 添加到对话历史
            error_response = {
                "success": False,
                "response": f"命令处理失败：{str(e)}",
                "response_type": "error",
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": start_time.isoformat()
            }

            self._add_to_conversation_history(user_input, error_response, user_id)
            return error_response

    async def _check_special_commands(self, user_input: str) -> Optional[Dict[str, Any]]:
        """检查并处理特殊命令"""
        input_lower = user_input.strip().lower()

        # 检查是否为帮助命令
        help_patterns = [
            r'^帮助$', r'^help$', r'^怎么用', r'^使用说明', r'^指令帮助'
        ]
        for pattern in help_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_help_command()

        # 检查是否为状态命令
        status_patterns = [
            r'^状态$', r'^status$', r'^系统状态', r'^运行状态'
        ]
        for pattern in status_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_status_command()

        # 检查是否为历史命令
        history_patterns = [
            r'^历史$', r'^history$', r'^历史记录', r'^操作历史'
        ]
        for pattern in history_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_history_command()

        # 检查是否为清空命令
        clear_patterns = [
            r'^清空$', r'^clear$', r'^清屏', r'^清除对话'
        ]
        for pattern in clear_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_clear_command()

        # 检查是否为工具命令
        tools_patterns = [
            r'^工具$', r'^tools$', r'^可用工具', r'^工具列表'
        ]
        for pattern in tools_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_tools_command()

        # 检查是否为退出命令
        exit_patterns = [
            r'^退出$', r'^exit$', r'^quit$', r'^再见$'
        ]
        for pattern in exit_patterns:
            if re.match(pattern, input_lower):
                return await self._handle_exit_command()

        return None

    async def _handle_help_command(self) -> Dict[str, Any]:
        """处理帮助命令"""
        help_text = """
🤖 MCP Floating Ball AI助手 使用指南

📋 支持的功能类别：
• 系统控制 - 启动应用、窗口管理、系统信息查询
• 文件处理 - 格式转换、文本操作
• 网络工具 - 网页搜索、内容抓取
• 视觉识别 - 屏幕截图、OCR文字识别、图像分析

💡 使用示例：
• "打开记事本" - 启动应用程序
• "最小化所有窗口" - 窗口管理
• "帮我搜索Python教程" - 网络搜索
• "截取全屏并识别上面的文字" - 视觉识别
• "将PDF转换为Word" - 文件格式转换

🔧 特殊命令：
• 帮助/Help - 显示此帮助信息
• 状态/Status - 查看系统状态
• 历史/History - 查看操作历史
• 工具/Tools - 查看可用工具
• 清空/Clear - 清空对话历史
• 退出/Exit - 退出程序

💬 提示：您可以随时用自然语言描述您的需求，系统会自动理解并执行相应的操作。
        """.strip()

        return {
            "success": True,
            "response": help_text,
            "response_type": "help",
            "execution_time": 0.01,
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_status_command(self) -> Dict[str, Any]:
        """处理状态命令"""
        try:
            # 获取工具状态
            tool_status = self.tool_caller.get_tool_status()

            # 获取统计信息
            stats = self.get_stats()

            # 获取系统信息
            from src.tools.system.system_info import SystemInfoTool
            system_tool = SystemInfoTool()
            basic_info = await system_tool.get_basic_info()

            status_text = f"""
📊 系统状态报告

🔧 工具状态：
• 总工具数：{tool_status['total_tools']}
• 可用工具：{sum(cat['available'] for cat in tool_status['categories'].values())}
  - 系统工具：{tool_status['categories'].get('system', {}).get('available', 0)}/{tool_status['categories'].get('system', {}).get('count', 0)}
  - 文件工具：{tool_status['categories'].get('file', {}).get('available', 0)}/{tool_status['categories'].get('file', {}).get('count', 0)}
  - 网络工具：{tool_status['categories'].get('network', {}).get('available', 0)}/{tool_status['categories'].get('network', {}).get('count', 0)}

📈 运行统计：
• 总命令数：{stats['total_commands']}
• 成功执行：{stats['successful_commands']}
• 执行失败：{stats['failed_commands']}
• 成功率：{stats['success_rate']:.1f}%
• 运行时长：{stats['uptime']}

💻 系统信息：
• 操作系统：{basic_info.get('result', {}).get('platform', 'Unknown')}
• Python版本：{basic_info.get('result', {}).get('python_version', 'Unknown')}
• CPU使用率：{basic_info.get('result', {}).get('cpu_usage', 0):.1f}%
• 内存使用：{basic_info.get('result', {}).get('memory_usage', 0):.1f}%

📜 对话历史：{len(self.conversation_history)} 条记录
            """.strip()

            return {
                "success": True,
                "response": status_text,
                "response_type": "status",
                "detailed_info": {
                    "tool_status": tool_status,
                    "stats": stats,
                    "system_info": basic_info
                },
                "execution_time": 0.5,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"获取系统状态失败：{str(e)}",
                "response_type": "error",
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_history_command(self) -> Dict[str, Any]:
        """处理历史命令"""
        try:
            if not self.conversation_history:
                return {
                    "success": True,
                    "response": "暂无对话历史记录。",
                    "response_type": "info",
                    "execution_time": 0.01,
                    "timestamp": datetime.now().isoformat()
                }

            # 获取最近10条历史
            recent_history = self.conversation_history[-10:]

            history_text = "📜 最近对话历史：\n\n"
            for i, entry in enumerate(recent_history, 1):
                user_input = entry.get("user_input", "")
                response = entry.get("assistant_response", {})
                success = response.get("success", False)
                timestamp = entry.get("timestamp", "")

                history_text += f"{i}. 🧑 {user_input[:50]}{'...' if len(user_input) > 50 else ''}\n"
                history_text += f"   {'✅' if success else '❌'} {response.get('response', '无响应')[:60]}{'...' if len(response.get('response', '')) > 60 else ''}\n"
                history_text += f"   🕐 {timestamp[:19]}\n\n"

            return {
                "success": True,
                "response": history_text.strip(),
                "response_type": "history",
                "history_count": len(self.conversation_history),
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"获取历史记录失败：{str(e)}",
                "response_type": "error",
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_clear_command(self) -> Dict[str, Any]:
        """处理清空命令"""
        try:
            # 清空对话历史
            self.conversation_history.clear()
            self.session_context.clear()

            # 清空工具执行历史
            self.tool_caller.clear_execution_history()

            return {
                "success": True,
                "response": "✨ 对话历史和执行记录已清空。",
                "response_type": "info",
                "execution_time": 0.01,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"清空失败：{str(e)}",
                "response_type": "error",
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_tools_command(self) -> Dict[str, Any]:
        """处理工具命令"""
        try:
            tool_status = self.tool_caller.get_tool_status()

            tools_text = "🔧 可用工具列表：\n\n"

            for category, info in tool_status["categories"].items():
                tools_text += f"📁 {category.upper()}类工具 ({info['available']}/{info['count']})\n"

                for tool_name in info["tools"]:
                    if tool_name in self.tool_caller.tools:
                        tool = self.tool_caller.tools[tool_name]
                        metadata = tool.get_metadata()
                        tools_text += f"  ✅ {metadata.name}: {metadata.description}\n"
                    else:
                        tools_text += f"  ❌ {tool_name}: 不可用\n"

                tools_text += "\n"

            return {
                "success": True,
                "response": tools_text.strip(),
                "response_type": "tools",
                "tool_status": tool_status,
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"获取工具列表失败：{str(e)}",
                "response_type": "error",
                "execution_time": 0.1,
                "timestamp": datetime.now().isoformat()
            }

    async def _handle_exit_command(self) -> Dict[str, Any]:
        """处理退出命令"""
        return {
            "success": True,
            "response": "👋 感谢使用 MCP Floating Ball AI助手，再见！",
            "response_type": "exit",
            "should_exit": True,
            "execution_time": 0.01,
            "timestamp": datetime.now().isoformat()
        }

    async def _validate_commands(self, commands: List[Union[Command, CompatibleCommand]]) -> Dict[str, Any]:
        """验证命令参数"""
        try:
            for cmd in commands:
                # 兼容性处理：获取意图类型和参数
                if hasattr(cmd, 'intent_type'):  # CompatibleCommand
                    intent_type = cmd.intent_type
                    parameters = cmd.parameters
                elif hasattr(cmd, 'intent'):  # Command
                    intent_type = cmd.intent.intent_type
                    parameters = cmd.intent.parameters
                else:
                    return {
                        "valid": False,
                        "error": f"未知的命令类型: {type(cmd)}",
                        "suggestions": ["请重新输入命令"]
                    }

                # 调试信息：记录实际的意图类型
                self.logger.info(f"验证命令 - 意图类型: {intent_type}, 参数: {parameters}")

                # 检查必要参数
                if intent_type == IntentType.APP_LAUNCH:
                    if not parameters.get("app_name"):
                        return {
                            "valid": False,
                            "error": "启动应用需要指定应用名称",
                            "suggestions": ["请说 '打开记事本' 或 '启动计算器'"]
                        }

                elif intent_type == IntentType.WEB_SEARCH:
                    if not parameters.get("query"):
                        return {
                            "valid": False,
                            "error": "搜索需要指定查询内容",
                            "suggestions": ["请说 '搜索Python教程' 或 '帮我找一下天气预报'"]
                        }

                elif intent_type == IntentType.WEB_SCRAPING:
                    # 网页抓取的验证逻辑
                    if not parameters.get("url") and not parameters.get("site_name"):
                        return {
                            "valid": False,
                            "error": "网页操作需要指定URL或网站名称",
                            "suggestions": ["请说 '打开百度' 或 '访问https://www.google.com'"]
                        }

                elif intent_type == IntentType.WEB_SEARCH:
                    # 网页搜索的验证逻辑
                    if not parameters.get("query"):
                        return {
                            "valid": False,
                            "error": "搜索需要指定查询内容",
                            "suggestions": ["请说 '搜索Python教程' 或 '帮我找一下天气预报'"]
                        }

                elif intent_type in [IntentType.FILE_FORMAT_CONVERT, IntentType.FILE_TEXT_PROCESS]:
                    if not parameters.get("file_path"):
                        return {
                            "valid": False,
                            "error": "文件操作需要指定文件路径",
                            "suggestions": ["请提供完整的文件路径"]
                        }

                elif intent_type in [IntentType.OCR, IntentType.IMAGE_ANALYSIS]:
                    if not parameters.get("image_path"):
                        return {
                            "valid": False,
                            "error": "图像处理需要指定图片路径",
                            "suggestions": ["请提供图片文件路径，或先截图"]
                        }

            return {"valid": True}

        except Exception as e:
            return {
                "valid": False,
                "error": f"参数验证失败：{str(e)}",
                "suggestions": ["请检查命令格式是否正确"]
            }

    async def _process_execution_results(self, execution_results: List[Dict[str, Any]], original_input: str) -> Dict[str, Any]:
        """处理执行结果并生成用户友好的响应"""
        try:
            if not execution_results:
                return {
                    "success": False,
                    "response": "没有可执行的命令。",
                    "response_type": "error"
                }

            # 统计执行结果
            successful_results = [r for r in execution_results if r.get("success", False)]
            failed_results = [r for r in execution_results if not r.get("success", False)]

            # 生成响应文本
            if len(execution_results) == 1:
                # 单个命令
                result = execution_results[0]
                if result.get("success", False):
                    response_text = self._format_success_response(result, original_input)
                    response_type = "success"
                else:
                    response_text = self._format_error_response(result, original_input)
                    response_type = "error"
            else:
                # 多个命令
                response_text = self._format_multiple_results(execution_results, original_input)
                response_type = "success" if successful_results else "partial"

            return {
                "success": len(successful_results) > 0,
                "response": response_text,
                "response_type": response_type,
                "execution_results": execution_results,
                "summary": {
                    "total_commands": len(execution_results),
                    "successful": len(successful_results),
                    "failed": len(failed_results)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "response": f"结果处理失败：{str(e)}",
                "response_type": "error",
                "execution_results": execution_results
            }

    def _format_success_response(self, result: Dict[str, Any], original_input: str) -> str:
        """格式化成功响应"""
        tool_name = result.get("tool_name", "unknown")
        summary = result.get("summary", "")

        # 根据工具类型生成具体的响应
        if tool_name == "application_launcher":
            app_name = result.get("result", {}).get("app_name", "应用")
            return f"✅ 已成功启动 {app_name}"
        elif tool_name == "multi_search":
            query = result.get("result", {}).get("query", "")
            return f"🔍 已完成搜索：{query}"
        elif tool_name == "ocr":
            word_count = result.get("result", {}).get("word_count", 0)
            return f"📝 OCR识别完成，共识别到 {word_count} 个文字"
        elif tool_name == "screen_capture":
            filename = result.get("result", {}).get("filename", "")
            return f"📸 截图完成，已保存为 {filename}"
        elif summary:
            return f"✅ {summary}"
        else:
            return "✅ 命令执行成功"

    def _format_error_response(self, result: Dict[str, Any], original_input: str) -> str:
        """格式化错误响应"""
        error = result.get("error", "未知错误")
        tool_name = result.get("tool_name", "unknown")

        # 提供友好的错误信息和解决建议
        if "文件" in error and "不存在" in error:
            return f"❌ 文件操作失败：找不到指定的文件。请检查文件路径是否正确。"
        elif "网络" in error or "连接" in error:
            return f"❌ 网络操作失败：请检查网络连接是否正常。"
        elif "权限" in error:
            return f"❌ 权限不足：请以管理员身份运行程序。"
        elif "参数" in error:
            return f"❌ 参数错误：{error}\n💡 建议：请检查命令格式或使用'帮助'查看使用示例。"
        else:
            return f"❌ 操作失败：{error}"

    def _format_multiple_results(self, execution_results: List[Dict[str, Any]], original_input: str) -> str:
        """格式化多个执行结果"""
        successful = [r for r in execution_results if r.get("success", False)]
        failed = [r for r in execution_results if not r.get("success", False)]

        response_lines = []
        response_lines.append(f"📋 执行完成：{len(successful)} 成功，{len(failed)} 失败")

        # 添加成功结果
        for result in successful:
            summary = result.get("summary", "")
            if summary:
                response_lines.append(f"✅ {summary}")

        # 添加失败结果
        for result in failed:
            error = result.get("error", "未知错误")
            response_lines.append(f"❌ {error}")

        return "\n".join(response_lines)

    def _get_command_suggestions(self, user_input: str) -> List[str]:
        """根据用户输入提供建议"""
        suggestions = []

        # 基于关键词提供建议
        if any(keyword in user_input.lower() for keyword in ["打开", "启动", "运行"]):
            suggestions.extend([
                "请说：打开记事本",
                "请说：启动计算器",
                "请说：运行浏览器"
            ])
        elif any(keyword in user_input.lower() for keyword in ["搜索", "查找", "找"]):
            suggestions.extend([
                "请说：搜索Python教程",
                "请说：查找天气预报",
                "请说：帮我搜索AI相关内容"
            ])
        elif any(keyword in user_input.lower() for keyword in ["截图", "截屏", "屏幕"]):
            suggestions.extend([
                "请说：截取全屏",
                "请说：截图并识别文字",
                "请说：截取指定区域"
            ])

        # 添加通用建议
        suggestions.append("使用'帮助'查看所有可用功能")

        return suggestions[:5]  # 返回最多5个建议

    def _add_to_conversation_history(self, user_input: str, response: Dict[str, Any], user_id: Optional[str] = None):
        """添加到对话历史"""
        try:
            entry = {
                "user_input": user_input,
                "assistant_response": response,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }

            self.conversation_history.append(entry)

            # 限制历史记录数量
            max_history = self.config_manager.get("assistant.max_history_size", 1000)
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]

        except Exception as e:
            self.logger.warning(f"添加对话历史失败: {e}")

    def _update_session_context(self, commands: List[Command], execution_results: List[Dict[str, Any]]):
        """更新会话上下文"""
        try:
            # 更新最后一次操作类型
            if commands:
                self.session_context["last_intent"] = commands[0].intent_type.name
                self.session_context["last_tool"] = execution_results[0].get("tool_name", "unknown")

            # 更新会话统计
            self.session_context["command_count"] = self.session_context.get("command_count", 0) + len(commands)

        except Exception as e:
            self.logger.warning(f"更新会话上下文失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        now = datetime.now()
        uptime = str(now - self.stats["start_time"]).split('.')[0]

        success_rate = 0.0
        if self.stats["total_commands"] > 0:
            success_rate = (self.stats["successful_commands"] / self.stats["total_commands"]) * 100

        return {
            "total_commands": self.stats["total_commands"],
            "successful_commands": self.stats["successful_commands"],
            "failed_commands": self.stats["failed_commands"],
            "success_rate": success_rate,
            "uptime": uptime,
            "intent_distribution": self.stats["intent_distribution"].copy(),
            "conversation_history_count": len(self.conversation_history),
            "session_context": self.session_context.copy()
        }

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        self.session_context.clear()
        self.logger.info("对话历史已清空")

    def cleanup(self):
        """清理资源"""
        try:
            # 清理工具调用器
            if self.tool_caller:
                self.tool_caller.cleanup()

            # 清空历史记录
            self.clear_history()

            self.logger.info("命令处理器资源清理完成")

        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")