"""
MCP Floating Ball - 工作流管理器

提供工作流的统一管理和执行接口。
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.logging import get_logger
from src.core.workflow import (
    get_workflow_engine, WorkflowDefinition, WorkflowContext,
    WorkflowState, TriggerType
)
from src.core.predefined_workflows import setup_predefined_workflows
from src.assistant.enhanced_nlp_processor import EnhancedNLPProcessor

logger = get_logger("workflow_manager")


class WorkflowManager:
    """工作流管理器"""

    def __init__(self):
        self.engine = get_workflow_engine()
        self.nlp_processor = EnhancedNLPProcessor()
        self.logger = get_logger(self.__class__.__name__)
        self.active_conversations: Dict[str, Dict[str, Any]] = {}

        # 添加状态监听器
        self.engine.add_state_listener(self._on_workflow_state_change)

        # 初始化预定义工作流
        self._initialize_workflows()

        self.logger.info("工作流管理器初始化完成")

    def _initialize_workflows(self):
        """初始化工作流"""
        try:
            setup_predefined_workflows()
            self.logger.info("预定义工作流初始化完成")
        except Exception as e:
            self.logger.error(f"预定义工作流初始化失败: {e}")

    async def execute_workflow_by_trigger(self, trigger_text: str,
                                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """通过触发器文本执行工作流"""
        try:
            self.logger.info(f"处理工作流触发: {trigger_text}")

            # 使用NLP处理器分析触发文本
            intent_result = self.nlp_processor.parse_command(trigger_text)
            if not intent_result.get("success"):
                return {
                    "success": False,
                    "error": "无法理解触发命令",
                    "suggestions": ["请尝试使用更明确的触发词", "例如: '处理文档' 或 '搜索资料'"]
                }

            # 获取意图和参数
            commands = intent_result.get("commands", [])
            if commands:
                intent_type = commands[0].intent_type
                parameters = commands[0].parameters
            else:
                intent_type = "UNKNOWN"
                parameters = {}

            # 查找匹配的工作流
            workflow_def = await self._find_matching_workflow(intent_type, trigger_text, parameters)
            if not workflow_def:
                return {
                    "success": False,
                    "error": "未找到匹配的工作流",
                    "intent_type": intent_type,
                    "suggestions": await self._get_workflow_suggestions(trigger_text)
                }

            # 执行工作流
            execution_id = await self.engine.execute_workflow(
                workflow_def.id,
                initial_variables=parameters,
                trigger_type=TriggerType.VOICE_COMMAND
            )

            return {
                "success": True,
                "workflow_id": workflow_def.id,
                "workflow_name": workflow_def.name,
                "execution_id": execution_id,
                "estimated_duration": self._estimate_workflow_duration(workflow_def),
                "message": f"工作流 '{workflow_def.name}' 已启动"
            }

        except Exception as e:
            self.logger.error(f"执行工作流失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _find_matching_workflow(self, intent_type: str,
                                     trigger_text: str,
                                     parameters: Dict[str, Any]) -> Optional[WorkflowDefinition]:
        """查找匹配的工作流"""
        workflows = self.engine.get_workflow_definitions()

        # 优先级匹配规则
        for workflow in workflows:
            # 检查触发器
            for trigger in workflow.triggers:
                if trigger.trigger_type == TriggerType.VOICE_COMMAND:
                    patterns = trigger.config.get("patterns", [])
                    for pattern in patterns:
                        if pattern in trigger_text.lower():
                            return workflow

                # 检查意图类型匹配
                if intent_type and self._matches_workflow_intent(workflow, intent_type):
                    return workflow

                # 检查关键词匹配
                if self._matches_keywords(workflow, trigger_text):
                    return workflow

        return None

    def _matches_workflow_intent(self, workflow: WorkflowDefinition, intent_type: str) -> bool:
        """检查工作流是否匹配意图类型"""
        # 基于工作流标签和描述进行匹配
        workflow_keywords = {
            "APP_LAUNCH": ["应用", "启动", "打开", "程序"],
            "WEB_SEARCH": ["搜索", "查找", "资料", "研究"],
            "SCREEN_CAPTURE": ["截图", "捕获", "屏幕", "图像"],
            "SYSTEM_INFO": ["系统", "信息", "状态", "维护"],
            "OCR": ["文字", "识别", "文档", "读取"],
            "FILE_OPERATION": ["文件", "处理", "转换", "操作"]
        }

        intent_keywords = workflow_keywords.get(intent_type, [])
        return any(keyword in workflow.name.lower() or
                  keyword in workflow.description.lower() or
                  any(keyword in tag.lower() for tag in workflow.tags)
                  for keyword in intent_keywords)

    def _matches_keywords(self, workflow: WorkflowDefinition, text: str) -> bool:
        """检查工作流是否匹配关键词"""
        text_lower = text.lower()

        # 检查名称匹配
        if any(keyword in text_lower for keyword in workflow.name.lower().split()):
            return True

        # 检查标签匹配
        for tag in workflow.tags:
            if tag.lower() in text_lower:
                return True

        # 检查描述匹配
        description_words = workflow.description.lower().split()
        if any(word in text_lower for word in description_words if len(word) > 2):
            return True

        return False

    async def _get_workflow_suggestions(self, trigger_text: str) -> List[str]:
        """获取工作流建议"""
        workflows = self.engine.get_workflow_definitions()
        suggestions = []

        text_lower = trigger_text.lower()

        for workflow in workflows:
            # 基于相似度提供建议
            if self._calculate_similarity(text_lower, workflow.name.lower()) > 0.3:
                suggestions.append(f"'{workflow.name}' - {workflow.description}")
            elif any(tag in text_lower for tag in workflow.tags):
                suggestions.append(f"'{workflow.name}' - {workflow.description}")

        return suggestions[:5]  # 返回前5个建议

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _estimate_workflow_duration(self, workflow_def: WorkflowDefinition) -> float:
        """估算工作流执行时间（秒）"""
        base_time = 5.0  # 基础时间
        task_count = sum(len(stage.tasks) for stage in workflow_def.stages)
        estimated_time = base_time + (task_count * 2.0)  # 每个任务平均2秒

        return min(estimated_time, 60.0)  # 最大不超过60秒

    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """获取工作流执行状态"""
        try:
            context = self.engine.get_workflow_status(execution_id)
            if not context:
                return {
                    "success": False,
                    "error": "工作流执行不存在"
                }

            # 计算执行进度
            total_tasks = len(self.engine.definitions[context.workflow_id].stages)
            completed_tasks = sum(1 for result in context.task_results.values()
                                if result.state.value in ["completed", "skipped"])
            progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

            return {
                "success": True,
                "workflow_id": context.workflow_id,
                "execution_id": execution_id,
                "state": context.metadata.get("state", "unknown"),
                "progress": round(progress, 1),
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "start_time": context.start_time.isoformat(),
                "elapsed_time": (datetime.now() - context.start_time).total_seconds(),
                "task_results": {
                    task_id: {
                        "state": result.state.value,
                        "execution_time": result.execution_time,
                        "error": result.error
                    }
                    for task_id, result in context.task_results.items()
                }
            }

        except Exception as e:
            self.logger.error(f"获取工作流状态失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def cancel_workflow(self, execution_id: str) -> Dict[str, Any]:
        """取消工作流执行"""
        try:
            success = self.engine.cancel_workflow(execution_id)
            return {
                "success": success,
                "execution_id": execution_id,
                "message": "工作流已取消" if success else "取消失败"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_available_workflows(self) -> List[Dict[str, Any]]:
        """获取可用工作流列表"""
        try:
            workflows = self.engine.get_workflow_definitions()
            return [
                {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "version": workflow.version,
                    "tags": workflow.tags,
                    "author": workflow.author,
                    "trigger_types": [trigger.trigger_type.value for trigger in workflow.triggers],
                    "stage_count": len(workflow.stages),
                    "task_count": sum(len(stage.tasks) for stage in workflow.stages),
                    "estimated_duration": self._estimate_workflow_duration(workflow)
                }
                for workflow in workflows
            ]
        except Exception as e:
            self.logger.error(f"获取可用工作流失败: {e}")
            return []

    async def execute_workflow_directly(self, workflow_id: str,
                                       parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """直接执行指定工作流"""
        try:
            execution_id = await self.engine.execute_workflow(
                workflow_id,
                initial_variables=parameters or {},
                trigger_type=TriggerType.MANUAL
            )

            return {
                "success": True,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "message": "工作流执行已启动"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _on_workflow_state_change(self, workflow_id: str, execution_id: str, state: WorkflowState):
        """工作流状态变更处理"""
        try:
            self.logger.info(f"工作流状态变更: {workflow_id}/{execution_id} -> {state.value}")

            # 获取工作流上下文
            context = self.engine.get_workflow_status(execution_id)
            if context:
                # 更新对话上下文
                if execution_id in self.active_conversations:
                    conversation = self.active_conversations[execution_id]
                    conversation["workflow_state"] = state.value
                    conversation["last_update"] = datetime.now()

                # 生成通知
                await self._generate_workflow_notification(workflow_id, execution_id, state, context)

        except Exception as e:
            self.logger.error(f"处理工作流状态变更失败: {e}")

    async def _generate_workflow_notification(self, workflow_id: str, execution_id: str,
                                            state: WorkflowState, context: WorkflowContext):
        """生成工作流通知"""
        try:
            workflow_def = self.engine.definitions.get(workflow_id)
            if not workflow_def:
                return

            if state == WorkflowState.COMPLETED:
                message = f"✅ 工作流 '{workflow_def.name}' 执行完成"
            elif state == WorkflowState.FAILED:
                message = f"❌ 工作流 '{workflow_def.name}' 执行失败"
            elif state == WorkflowState.CANCELLED:
                message = f"⏹️ 工作流 '{workflow_def.name}' 已取消"
            else:
                return  # 其他状态不生成通知

            # 这里可以集成通知系统，如语音提示、桌面通知等
            self.logger.info(f"工作流通知: {message}")

            # 如果有活跃对话，添加到对话上下文
            if execution_id in self.active_conversations:
                self.active_conversations[execution_id]["notifications"].append({
                    "message": message,
                    "timestamp": datetime.now(),
                    "state": state.value
                })

        except Exception as e:
            self.logger.error(f"生成工作流通知失败: {e}")

    def start_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """开始新的对话"""
        try:
            self.active_conversations[conversation_id] = {
                "conversation_id": conversation_id,
                "start_time": datetime.now(),
                "last_update": datetime.now(),
                "workflow_executions": [],
                "notifications": [],
                "context": {}
            }

            return {
                "success": True,
                "conversation_id": conversation_id,
                "message": "对话已开始"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """结束对话"""
        try:
            if conversation_id in self.active_conversations:
                conversation = self.active_conversations[conversation_id]
                del self.active_conversations[conversation_id]

                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "duration": (datetime.now() - conversation["start_time"]).total_seconds(),
                    "workflow_count": len(conversation["workflow_executions"])
                }
            else:
                return {
                    "success": False,
                    "error": "对话不存在"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        try:
            if conversation_id not in self.active_conversations:
                return {
                    "success": False,
                    "error": "对话不存在"
                }

            conversation = self.active_conversations[conversation_id]
            return {
                "success": True,
                "conversation_id": conversation_id,
                "start_time": conversation["start_time"].isoformat(),
                "duration": (datetime.now() - conversation["start_time"]).total_seconds(),
                "workflow_executions": conversation["workflow_executions"],
                "notifications": conversation["notifications"][-5:],  # 最近5个通知
                "context": conversation["context"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# 全局工作流管理器实例
_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """获取全局工作流管理器实例"""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = WorkflowManager()
    return _workflow_manager


# 导出
__all__ = [
    "WorkflowManager",
    "get_workflow_manager"
]