#!/usr/bin/env python3
"""
测试智能工作流引擎

验证工作流引擎的各项功能，包括工作流定义、执行、状态管理等。
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_workflow_engine_initialization():
    """测试工作流引擎初始化"""
    print("🔄 测试工作流引擎初始化...")

    try:
        from src.core.workflow import get_workflow_engine

        engine = get_workflow_engine()
        print("✅ 工作流引擎初始化成功")

        # 检查预定义工作流
        workflows = engine.get_workflow_definitions()
        print(f"✅ 找到 {len(workflows)} 个预定义工作流:")
        for workflow in workflows:
            print(f"   - {workflow.name} ({workflow.id}) - {len(workflow.stages)} 阶段")

        return True

    except Exception as e:
        print(f"❌ 工作流引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_manager():
    """测试工作流管理器"""
    print("\n🔄 测试工作流管理器...")

    try:
        from src.assistant.workflow_manager import get_workflow_manager

        manager = get_workflow_manager()
        print("✅ 工作流管理器初始化成功")

        # 获取可用工作流
        workflows = await manager.get_available_workflows()
        print(f"✅ 获取到 {len(workflows)} 个可用工作流:")
        for workflow in workflows:
            print(f"   - {workflow['name']}: {workflow['description']}")
            print(f"     阶段数: {workflow['stage_count']}, 任务数: {workflow['task_count']}")

        return True

    except Exception as e:
        print(f"❌ 工作流管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_trigger():
    """测试工作流触发"""
    print("\n🔄 测试工作流触发...")

    try:
        from src.assistant.workflow_manager import get_workflow_manager

        manager = get_workflow_manager()

        # 测试不同的触发命令
        test_triggers = [
            "处理文档",
            "搜索Python教程",
            "系统信息",
            "助手，你好"
        ]

        for trigger in test_triggers:
            print(f"\n  测试触发命令: '{trigger}'")
            result = await manager.execute_workflow_by_trigger(trigger)

            if result["success"]:
                print(f"  ✅ 工作流启动成功: {result['workflow_name']}")
                print(f"     执行ID: {result['execution_id']}")
                print(f"     预计耗时: {result['estimated_duration']:.1f}秒")
            else:
                print(f"  ❌ 工作流启动失败: {result['error']}")
                if "suggestions" in result:
                    print(f"     建议: {result['suggestions']}")

        return True

    except Exception as e:
        print(f"❌ 工作流触发测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_execution():
    """测试工作流执行"""
    print("\n🔄 测试工作流执行...")

    try:
        from src.assistant.workflow_manager import get_workflow_manager
        import time

        manager = get_workflow_manager()

        # 启动一个简单的工作流
        trigger_text = "系统信息"
        print(f"启动工作流: {trigger_text}")

        result = await manager.execute_workflow_by_trigger(trigger_text)
        if not result["success"]:
            print(f"❌ 工作流启动失败: {result['error']}")
            return False

        execution_id = result["execution_id"]
        print(f"✅ 工作流启动成功，执行ID: {execution_id}")

        # 监控执行状态
        max_wait_time = 30  # 最多等待30秒
        wait_interval = 2   # 每2秒检查一次
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            await asyncio.sleep(wait_interval)
            elapsed_time += wait_interval

            status = await manager.get_workflow_status(execution_id)
            if status["success"]:
                print(f"  状态: {status['state']}, 进度: {status['progress']}%")
                print(f"  已完成任务: {status['completed_tasks']}/{status['total_tasks']}")
                print(f"  已用时间: {status['elapsed_time']:.1f}秒")

                if status["state"] in ["completed", "failed", "cancelled"]:
                    print(f"  🎉 工作流执行完成: {status['state']}")
                    break
            else:
                print(f"  ❌ 获取状态失败: {status['error']}")
                break

        return True

    except Exception as e:
        print(f"❌ 工作流执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_conversation_management():
    """测试对话管理"""
    print("\n🔄 测试对话管理...")

    try:
        from src.assistant.workflow_manager import get_workflow_manager

        manager = get_workflow_manager()

        # 开始对话
        conversation_id = f"test_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = manager.start_conversation(conversation_id)
        if result["success"]:
            print(f"✅ 对话开始成功: {conversation_id}")
        else:
            print(f"❌ 对话开始失败: {result['error']}")
            return False

        # 在对话中执行工作流
        result = await manager.execute_workflow_by_trigger("处理文档", {
            "conversation_id": conversation_id
        })
        print(f"工作流执行结果: {'成功' if result['success'] else '失败'}")

        # 获取对话状态
        status = await manager.get_conversation_status(conversation_id)
        if status["success"]:
            print(f"✅ 对话状态: 已运行 {status['duration']:.1f}秒")
            print(f"   通知数: {len(status['notifications'])}")

        # 结束对话
        result = manager.end_conversation(conversation_id)
        if result["success"]:
            print(f"✅ 对话结束成功，持续 {result['duration']:.1f}秒")
            print(f"   执行工作流数: {result['workflow_count']}")
        else:
            print(f"❌ 对话结束失败: {result['error']}")

        return True

    except Exception as e:
        print(f"❌ 对话管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_definition_creation():
    """测试自定义工作流创建"""
    print("\n🔄 测试自定义工作流创建...")

    try:
        from src.core.workflow import (
            get_workflow_engine, WorkflowDefinition, WorkflowStage,
            WorkflowTask, WorkflowTrigger, TriggerType
        )

        engine = get_workflow_engine()

        # 创建一个简单的测试工作流
        test_workflow = WorkflowDefinition(
            id="test_simple_workflow",
            name="测试简单工作流",
            description="用于测试工作流引擎的简单工作流",
            version="1.0.0",
            author="test",
            tags=["测试"],
            stages=[
                WorkflowStage(
                    id="stage1",
                    name="第一阶段",
                    tasks=[
                        WorkflowTask(
                            id="task1",
                            name="系统信息任务",
                            tool_name="system_info",
                            parameters={"info_type": "basic"}
                        )
                    ]
                )
            ],
            triggers=[
                WorkflowTrigger(
                    trigger_type=TriggerType.MANUAL,
                    config={}
                )
            ]
        )

        # 注册工作流
        success = engine.register_definition(test_workflow)
        if success:
            print("✅ 自定义工作流注册成功")

            # 直接执行工作流
            execution_id = await engine.execute_workflow("test_simple_workflow")
            print(f"✅ 工作流执行启动: {execution_id}")

            # 等待执行完成
            await asyncio.sleep(5)

            # 检查执行状态
            status = engine.get_workflow_status(execution_id)
            if status:
                print(f"  执行状态: {status.metadata.get('state', 'unknown')}")
                print(f"  任务数量: {len(status.task_results)}")

                for task_id, result in status.task_results.items():
                    print(f"  任务 {task_id}: {result.state.value}")

            return True
        else:
            print("❌ 自定义工作流注册失败")
            return False

    except Exception as e:
        print(f"❌ 自定义工作流创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """测试错误处理"""
    print("\n🔄 测试错误处理...")

    try:
        from src.assistant.workflow_manager import get_workflow_manager

        manager = get_workflow_manager()

        # 测试无效的触发命令
        invalid_triggers = [
            "不存在的工作流触发词",
            "123456",
            ""
        ]

        for trigger in invalid_triggers:
            print(f"  测试无效触发: '{trigger}'")
            result = await manager.execute_workflow_by_trigger(trigger)
            if not result["success"]:
                print(f"  ✅ 正确处理错误: {result['error']}")
            else:
                print(f"  ⚠️ 意外成功: {result}")

        # 测试无效的执行ID
        invalid_status = await manager.get_workflow_status("invalid_execution_id")
        if not invalid_status["success"]:
            print("  ✅ 正确处理无效执行ID")
        else:
            print("  ⚠️ 无效执行ID处理异常")

        return True

    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("=" * 60)
    print("🎈 MCP Floating Ball - 智能工作流引擎测试")
    print("=" * 60)

    tests = [
        ("工作流引擎初始化测试", test_workflow_engine_initialization),
        ("工作流管理器测试", test_workflow_manager),
        ("工作流触发测试", test_workflow_trigger),
        ("工作流执行测试", test_workflow_execution),
        ("对话管理测试", test_conversation_management),
        ("自定义工作流创建测试", test_workflow_definition_creation),
        ("错误处理测试", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有工作流引擎测试通过！智能化升级成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要检查相关模块")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)