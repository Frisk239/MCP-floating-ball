"""
MCP Floating Ball - 智能工作流引擎

基于Everywhere项目的任务编排理念，支持复杂的工作流自动化。
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.logging import get_logger
from src.core.database import get_database_manager
from src.core.exceptions import MCPFloatingBallError

logger = get_logger("workflow")


class WorkflowError(MCPFloatingBallError):
    """工作流错误"""
    pass


class WorkflowState(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(Enum):
    """触发器类型"""
    MANUAL = "manual"
    VOICE_COMMAND = "voice_command"
    SCHEDULE = "schedule"
    EVENT = "event"
    API_CALL = "api_call"


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    state: TaskState
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowContext:
    """工作流执行上下文"""
    workflow_id: str
    execution_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowTrigger:
    """工作流触发器"""
    trigger_type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class WorkflowTask:
    """工作流任务"""
    id: str
    name: str
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    parallel_group: Optional[str] = None
    description: str = ""


@dataclass
class WorkflowStage:
    """工作流阶段"""
    id: str
    name: str
    tasks: List[WorkflowTask] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # 依赖的前置阶段ID列表
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    parallel_execution: bool = False
    on_success: Optional[str] = None  # 成功后跳转的阶段ID
    on_failure: Optional[str] = None  # 失败后跳转的阶段ID


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = "system"
    tags: List[str] = field(default_factory=list)
    stages: List[WorkflowStage] = field(default_factory=list)
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class WorkflowEngine:
    """工作流执行引擎"""

    def __init__(self):
        self.db = get_database_manager()
        self.definitions: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowContext] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.state_listeners: List[Callable] = []
        self.running = False
        self.logger = get_logger(self.__class__.__name__)

        # 启动后台监控
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()

        self.logger.info("工作流引擎初始化完成")

    def register_definition(self, workflow_def: WorkflowDefinition) -> bool:
        """注册工作流定义"""
        try:
            self.definitions[workflow_def.id] = workflow_def

            # 保存到数据库
            success = self._save_workflow_definition(workflow_def)
            if success:
                self.logger.info(f"工作流定义已注册: {workflow_def.id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"注册工作流定义失败: {e}")
            return False

    async def execute_workflow(self, workflow_id: str,
                             initial_variables: Optional[Dict[str, Any]] = None,
                             trigger_type: TriggerType = TriggerType.MANUAL) -> str:
        """执行工作流"""
        try:
            if workflow_id not in self.definitions:
                raise WorkflowError(f"工作流定义不存在: {workflow_id}")

            workflow_def = self.definitions[workflow_id]
            execution_id = str(uuid.uuid4())

            # 创建执行上下文
            context = WorkflowContext(
                workflow_id=workflow_id,
                execution_id=execution_id,
                variables=initial_variables or {},
                metadata={
                    "trigger_type": trigger_type.value,
                    "workflow_name": workflow_def.name,
                    "start_time": datetime.now().isoformat()
                }
            )

            self.executions[execution_id] = context

            # 保存执行记录
            await self._save_execution_record(context)

            # 异步执行工作流
            asyncio.create_task(self._execute_workflow_async(context))

            self.logger.info(f"工作流执行已启动: {workflow_id} -> {execution_id}")
            return execution_id

        except Exception as e:
            self.logger.error(f"执行工作流失败: {e}")
            raise WorkflowError(f"执行工作流失败: {e}")

    async def _execute_workflow_async(self, context: WorkflowContext):
        """异步执行工作流"""
        try:
            workflow_def = self.definitions[context.workflow_id]

            # 通知状态变更
            await self._notify_state_change(context.workflow_id, context.execution_id, WorkflowState.RUNNING)

            # 按顺序执行各个阶段
            current_stage_index = 0
            while current_stage_index < len(workflow_def.stages):
                stage = workflow_def.stages[current_stage_index]

                # 检查阶段依赖
                if not await self._check_stage_dependencies(stage, context):
                    self.logger.info(f"阶段 {stage.id} 依赖不满足，等待前置阶段完成")
                    # 可以选择跳过或等待，这里选择跳过
                    current_stage_index += 1
                    continue

                # 检查阶段执行条件
                if not await self._evaluate_stage_conditions(stage, context):
                    self.logger.info(f"阶段 {stage.id} 条件不满足，跳过执行")
                    current_stage_index += 1
                    continue

                # 执行阶段
                stage_success = await self._execute_stage(stage, context)

                # 决定下一步执行
                if stage_success:
                    if stage.on_success:
                        # 跳转到指定阶段
                        next_stage_id = stage.on_success
                        next_stage_index = next(
                            i for i, s in enumerate(workflow_def.stages)
                            if s.id == next_stage_id
                        )
                        current_stage_index = next_stage_index
                    else:
                        current_stage_index += 1
                else:
                    if stage.on_failure:
                        # 跳转到失败处理阶段
                        next_stage_id = stage.on_failure
                        next_stage_index = next(
                            i for i, s in enumerate(workflow_def.stages)
                            if s.id == next_stage_id
                        )
                        current_stage_index = next_stage_index
                    else:
                        # 工作流失败
                        await self._notify_state_change(
                            context.workflow_id, context.execution_id, WorkflowState.FAILED
                        )
                        break

            # 工作流完成
            if current_stage_index >= len(workflow_def.stages):
                await self._notify_state_change(
                    context.workflow_id, context.execution_id, WorkflowState.COMPLETED
                )

            # 保存最终结果
            await self._save_execution_result(context)

        except Exception as e:
            self.logger.error(f"工作流执行异常: {e}")
            await self._notify_state_change(
                context.workflow_id, context.execution_id, WorkflowState.FAILED
            )
            await self._save_execution_result(context, error=str(e))

    async def _execute_stage(self, stage: WorkflowStage, context: WorkflowContext) -> bool:
        """执行工作流阶段"""
        try:
            self.logger.info(f"开始执行阶段: {stage.id}")

            if stage.parallel_execution:
                # 并行执行任务
                success = await self._execute_parallel_tasks(stage.tasks, context)
            else:
                # 串行执行任务
                success = await self._execute_sequential_tasks(stage.tasks, context)

            self.logger.info(f"阶段执行完成: {stage.id} -> {'成功' if success else '失败'}")
            return success

        except Exception as e:
            self.logger.error(f"阶段执行异常: {stage.id} - {e}")
            return False

    async def _execute_sequential_tasks(self, tasks: List[WorkflowTask], context: WorkflowContext) -> bool:
        """串行执行任务"""
        for task in tasks:
            # 检查任务依赖
            if not await self._check_task_dependencies(task, context):
                self.logger.warning(f"任务依赖不满足: {task.id}")
                continue

            # 检查执行条件
            if not await self._evaluate_task_conditions(task, context):
                self.logger.info(f"任务条件不满足，跳过: {task.id}")
                continue

            # 执行任务
            result = await self._execute_task(task, context)
            context.task_results[task.id] = result

            if result.state == TaskState.FAILED:
                # 检查是否应该继续执行
                if task.retry_policy.get("continue_on_failure", False):
                    continue
                else:
                    return False

        return True

    async def _execute_parallel_tasks(self, tasks: List[WorkflowTask], context: WorkflowContext) -> bool:
        """并行执行任务"""
        # 按并行分组组织任务
        parallel_groups = {}
        for task in tasks:
            group = task.parallel_group or task.id  # 没有分组则独立执行

            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append(task)

        # 按组顺序执行，组内并行执行
        for group_name, group_tasks in parallel_groups.items():
            # 检查组内所有任务的依赖
            ready_tasks = []
            for task in group_tasks:
                if await self._check_task_dependencies(task, context):
                    ready_tasks.append(task)

            if not ready_tasks:
                continue

            # 并行执行组内任务
            futures = []
            for task in ready_tasks:
                if await self._evaluate_task_conditions(task, context):
                    future = asyncio.create_task(self._execute_task(task, context))
                    futures.append((task.id, future))

            # 等待所有任务完成
            results = {}
            for task_id, future in futures:
                try:
                    result = await future
                    results[task_id] = result
                    context.task_results[task_id] = result
                except Exception as e:
                    self.logger.error(f"并行任务执行失败: {task_id} - {e}")
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        state=TaskState.FAILED,
                        error=str(e)
                    )
                    context.task_results[task_id] = results[task_id]

            # 检查是否有关键任务失败
            for task in ready_tasks:
                result = results.get(task.id)
                if result and result.state == TaskState.FAILED:
                    if not task.retry_policy.get("continue_on_failure", False):
                        return False

        return True

    async def _execute_task(self, task: WorkflowTask, context: WorkflowContext) -> TaskResult:
        """执行单个任务"""
        start_time = time.time()
        result = TaskResult(task_id=task.id, state=TaskState.PENDING)

        try:
            self.logger.debug(f"开始执行任务: {task.id}")
            result.state = TaskState.RUNNING

            # 准备任务参数
            task_params = await self._prepare_task_parameters(task, context)

            # 执行工具
            from src.assistant.tool_caller import ToolCaller
            tool_caller = ToolCaller()

            tool_result = await tool_caller.execute_tool(
                task.tool_name,
                task_params
            )

            if tool_result.get("success", False):
                result.state = TaskState.COMPLETED
                result.result = tool_result
            else:
                result.state = TaskState.FAILED
                result.error = tool_result.get("error", "未知错误")

        except Exception as e:
            result.state = TaskState.FAILED
            result.error = str(e)
            self.logger.error(f"任务执行异常: {task.id} - {e}")

        finally:
            result.execution_time = time.time() - start_time
            result.timestamp = datetime.now()

            # 保存任务结果
            await self._save_task_result(context.execution_id, result)

            self.logger.debug(f"任务执行完成: {task.id} -> {result.state.value}")

        return result

    async def _prepare_task_parameters(self, task: WorkflowTask, context: WorkflowContext) -> Dict[str, Any]:
        """准备任务参数"""
        # 合并工作流变量和任务参数
        params = context.variables.copy()
        params.update(task.parameters)

        # 处理参数替换
        for key, value in params.items():
            if isinstance(value, str):
                params[key] = await self._substitute_variables(value, context)

        return params

    async def _substitute_variables(self, text: str, context: WorkflowContext) -> str:
        """变量替换"""
        import re

        # 替换工作流变量
        def replace_var(match):
            var_name = match.group(1)
            return str(context.variables.get(var_name, match.group(0)))

        text = re.sub(r'\{\{(\w+)\}\}', replace_var, text)

        # 替换任务结果
        def replace_result(match):
            task_id = match.group(1)
            result = context.task_results.get(task_id)
            if result and result.result:
                return str(result.result)
            return match.group(0)

        text = re.sub(r'\{\{result\.(\w+)\}\}', replace_result, text)

        return text

    async def _check_task_dependencies(self, task: WorkflowTask, context: WorkflowContext) -> bool:
        """检查任务依赖"""
        for dep_id in task.depends_on:
            if dep_id not in context.task_results:
                return False

            dep_result = context.task_results[dep_id]
            if dep_result.state != TaskState.COMPLETED:
                return False

        return True

    async def _check_stage_dependencies(self, stage: WorkflowStage, context: WorkflowContext) -> bool:
        """检查阶段依赖"""
        if not stage.depends_on:
            return True

        # 需要扩展context来跟踪阶段完成状态
        # 暂时使用任务结果来推断阶段完成状态
        for dep_stage_id in stage.depends_on:
            # 查看依赖阶段的所有任务是否都已完成
            dep_stage_completed = True
            for task_id, result in context.task_results.items():
                # 通过任务ID前缀判断任务属于哪个阶段
                if task_id.startswith(dep_stage_id):
                    if result.state != TaskState.COMPLETED:
                        dep_stage_completed = False
                        break

            if not dep_stage_completed:
                return False

        return True

    async def _evaluate_task_conditions(self, task: WorkflowTask, context: WorkflowContext) -> bool:
        """评估任务执行条件"""
        for condition in task.conditions:
            if not await self._evaluate_condition(condition, context):
                return False
        return True

    async def _evaluate_stage_conditions(self, stage: WorkflowStage, context: WorkflowContext) -> bool:
        """评估阶段执行条件"""
        for condition in stage.conditions:
            if not await self._evaluate_condition(condition, context):
                return False
        return True

    async def _evaluate_condition(self, condition: Dict[str, Any], context: WorkflowContext) -> bool:
        """评估单个条件"""
        condition_type = condition.get("type")

        if condition_type == "variable_equals":
            var_name = condition.get("variable")
            expected_value = condition.get("value")
            return context.variables.get(var_name) == expected_value

        elif condition_type == "variable_exists":
            var_name = condition.get("variable")
            return var_name in context.variables

        elif condition_type == "task_success":
            task_id = condition.get("task_id")
            result = context.task_results.get(task_id)
            return result and result.state == TaskState.COMPLETED

        elif condition_type == "time_based":
            current_hour = datetime.now().hour
            hour_range = condition.get("hour_range", [0, 23])
            return hour_range[0] <= current_hour <= hour_range[1]

        elif condition_type == "expression":
            # 简单的表达式求值（生产环境建议使用更安全的方案）
            expr = condition.get("expression")
            try:
                # 创建安全的执行环境
                safe_dict = {
                    "context": context,
                    "variables": context.variables,
                    "results": context.task_results,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "datetime": datetime,
                }
                return eval(expr, {"__builtins__": {}}, safe_dict)
            except Exception as e:
                self.logger.error(f"条件表达式求值失败: {expr} - {e}")
                return False

        return True

    def add_state_listener(self, listener: Callable):
        """添加状态变更监听器"""
        self.state_listeners.append(listener)

    async def _notify_state_change(self, workflow_id: str, execution_id: str, state: WorkflowState):
        """通知状态变更"""
        for listener in self.state_listeners:
            try:
                await listener(workflow_id, execution_id, state)
            except Exception as e:
                self.logger.error(f"状态监听器异常: {e}")

    def _background_monitor(self):
        """后台监控线程"""
        while self.running:
            try:
                # 检查超时的工作流
                asyncio.run(self._check_timeouts())

                # 检查计划任务
                asyncio.run(self._check_scheduled_triggers())

                time.sleep(10)  # 每10秒检查一次
            except Exception as e:
                self.logger.error(f"后台监控异常: {e}")

    async def _check_timeouts(self):
        """检查工作流超时"""
        current_time = datetime.now()
        timeout_threshold = timedelta(hours=1)  # 1小时超时

        for execution_id, context in self.executions.items():
            if current_time - context.start_time > timeout_threshold:
                # 标记为超时
                await self._notify_state_change(
                    context.workflow_id, execution_id, WorkflowState.FAILED
                )
                await self._save_execution_result(context, error="执行超时")

    async def _check_scheduled_triggers(self):
        """检查计划触发器"""
        current_time = datetime.now()

        for workflow_id, workflow_def in self.definitions.items():
            for trigger in workflow_def.triggers:
                if trigger.trigger_type == TriggerType.SCHEDULE:
                    # 检查是否到了触发时间
                    cron_expr = trigger.config.get("cron")
                    if cron_expr and self._should_trigger_schedule(cron_expr, current_time):
                        await self.execute_workflow(workflow_id, trigger_type=TriggerType.SCHEDULE)
                        trigger.last_triggered = current_time
                        trigger.trigger_count += 1

    def _should_trigger_schedule(self, cron_expr: str, current_time: datetime) -> bool:
        """检查是否应该触发计划任务（简化版）"""
        # 这里应该使用完整的cron解析库
        # 简化实现：支持 "0 9 * * *" 格式
        try:
            parts = cron_expr.split()
            if len(parts) != 5:
                return False

            minute, hour, day, month, weekday = parts

            return (
                (minute == "*" or int(minute) == current_time.minute) and
                (hour == "*" or int(hour) == current_time.hour) and
                (day == "*" or int(day) == current_time.day) and
                (month == "*" or int(month) == current_time.month) and
                (weekday == "*" or int(weekday) == current_time.weekday())
            )
        except:
            return False

    async def _save_workflow_definition(self, workflow_def: WorkflowDefinition) -> bool:
        """保存工作流定义到数据库"""
        try:
            data = {
                "workflow_id": workflow_def.id,
                "name": workflow_def.name,
                "description": workflow_def.description,
                "version": workflow_def.version,
                "author": workflow_def.author,
                "tags": json.dumps(workflow_def.tags),
                "definition": json.dumps(workflow_def, default=str),
                "created_at": workflow_def.created_at,
                "updated_at": workflow_def.updated_at
            }

            # 这里应该调用数据库API保存
            return True
        except Exception as e:
            self.logger.error(f"保存工作流定义失败: {e}")
            return False

    async def _save_execution_record(self, context: WorkflowContext) -> bool:
        """保存执行记录"""
        try:
            data = {
                "execution_id": context.execution_id,
                "workflow_id": context.workflow_id,
                "variables": json.dumps(context.variables),
                "metadata": json.dumps(context.metadata),
                "start_time": context.start_time,
                "state": WorkflowState.RUNNING.value
            }

            # 这里应该调用数据库API保存
            return True
        except Exception as e:
            self.logger.error(f"保存执行记录失败: {e}")
            return False

    async def _save_task_result(self, execution_id: str, result: TaskResult) -> bool:
        """保存任务结果"""
        try:
            data = {
                "execution_id": execution_id,
                "task_id": result.task_id,
                "state": result.state.value,
                "result": json.dumps(result.result) if result.result else None,
                "error": result.error,
                "execution_time": result.execution_time,
                "retry_count": result.retry_count,
                "metadata": json.dumps(result.metadata),
                "timestamp": result.timestamp
            }

            # 这里应该调用数据库API保存
            return True
        except Exception as e:
            self.logger.error(f"保存任务结果失败: {e}")
            return False

    async def _save_execution_result(self, context: WorkflowContext, error: Optional[str] = None) -> bool:
        """保存执行结果"""
        try:
            data = {
                "execution_id": context.execution_id,
                "workflow_id": context.workflow_id,
                "end_time": datetime.now(),
                "duration": (datetime.now() - context.start_time).total_seconds(),
                "state": WorkflowState.COMPLETED.value if not error else WorkflowState.FAILED.value,
                "error": error,
                "task_count": len(context.task_results),
                "success_count": sum(1 for r in context.task_results.values() if r.state == TaskState.COMPLETED)
            }

            # 这里应该调用数据库API保存
            return True
        except Exception as e:
            self.logger.error(f"保存执行结果失败: {e}")
            return False

    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowContext]:
        """获取工作流执行状态"""
        return self.executions.get(execution_id)

    def get_workflow_definitions(self) -> List[WorkflowDefinition]:
        """获取所有工作流定义"""
        return list(self.definitions.values())

    def cancel_workflow(self, execution_id: str) -> bool:
        """取消工作流执行"""
        if execution_id in self.executions:
            context = self.executions[execution_id]
            # 这里应该实现取消逻辑
            asyncio.create_task(
                self._notify_state_change(context.workflow_id, execution_id, WorkflowState.CANCELLED)
            )
            return True
        return False

    def shutdown(self):
        """关闭工作流引擎"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("工作流引擎已关闭")


# 全局工作流引擎实例
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """获取全局工作流引擎实例"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


# 导出
__all__ = [
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStage",
    "WorkflowTask",
    "WorkflowTrigger",
    "WorkflowContext",
    "TaskResult",
    "WorkflowState",
    "TaskState",
    "TriggerType",
    "get_workflow_engine"
]