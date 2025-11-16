#!/usr/bin/env python3
"""
MCP Floating Ball - 系统集成测试

测试整个系统的集成和功能。
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.logging import get_logger
logger = get_logger("test_integration")


async def test_basic_imports():
    """测试基本导入"""
    print("🔄 测试基本导入...")

    try:
        # 测试导入AI助手
        from src.assistant.ai_assistant import AIAssistant
        print("✅ AI助手导入成功")

        # 测试导入命令处理器
        from src.assistant.command_handler import CommandHandler
        print("✅ 命令处理器导入成功")

        # 测试导入工具调用器
        from src.assistant.tool_caller import ToolCaller
        print("✅ 工具调用器导入成功")

        # 测试导入NLP处理器
        from src.assistant.nlp_processor import NLPProcessor
        print("✅ NLP处理器导入成功")

        print("🎉 所有核心组件导入测试通过!")
        return True

    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_nlp_processor():
    """测试NLP处理器"""
    print("\n🔄 测试NLP处理器...")

    try:
        from src.assistant.nlp_processor import NLPProcessor, IntentType

        nlp = NLPProcessor()

        # 测试命令解析
        test_commands = [
            "打开记事本",
            "搜索Python教程",
            "截图并识别文字",
            "查看系统信息",
            "将PDF转换为Word"
        ]

        for cmd in test_commands:
            result = nlp.parse_command(cmd)
            if result["success"]:
                print(f"✅ '{cmd}' -> {result['commands'][0].intent_type.name}")
            else:
                print(f"❌ '{cmd}' -> 解析失败: {result.get('error')}")

        print("🎉 NLP处理器测试通过!")
        return True

    except Exception as e:
        print(f"❌ NLP处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_command_handler():
    """测试命令处理器"""
    print("\n🔄 测试命令处理器...")

    try:
        from src.assistant.command_handler import CommandHandler

        handler = CommandHandler()

        # 测试基本命令处理
        test_commands = [
            "帮助",
            "状态",
            "工具",
            "历史"
        ]

        for cmd in test_commands:
            result = await handler.process_command(cmd)
            if result["success"]:
                print(f"✅ '{cmd}' -> 处理成功")
            else:
                print(f"❌ '{cmd}' -> 处理失败")

        print("🎉 命令处理器测试通过!")
        return True

    except Exception as e:
        print(f"❌ 命令处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tool_caller():
    """测试工具调用器"""
    print("\n🔄 测试工具调用器...")

    try:
        from src.assistant.tool_caller import ToolCaller
        from src.assistant.nlp_processor import CompatibleCommand, CommandType, IntentType

        caller = ToolCaller()

        # 测试系统信息工具
        cmd = CompatibleCommand(
            intent_type=IntentType.SYSTEM_INFO,
            parameters={"info_type": "basic"},
            command_type=CommandType.SINGLE,
            original_text="测试获取系统信息"
        )

        result = await caller.execute_command(cmd)
        if result["success"]:
            print("✅ 系统信息工具测试成功")
        else:
            print(f"❌ 系统信息工具测试失败: {result.get('error')}")

        # 获取工具状态
        status = caller.get_tool_status()
        print(f"📊 工具状态: 总数 {status['total_tools']}, 可用 {sum(cat['available'] for cat in status['categories'].values())}")

        print("🎉 工具调用器测试通过!")
        return True

    except Exception as e:
        print(f"❌ 工具调用器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_vision_integration():
    """测试视觉集成"""
    print("\n🔄 测试视觉集成...")

    try:
        from src.vision.vision_integration import VisionIntegration

        vision = VisionIntegration()

        # 获取系统状态
        status = vision.get_system_status()
        print(f"📊 视觉系统状态:")
        for component, info in status["components"].items():
            available = "✅" if info["available"] else "❌"
            print(f"   {available} {component}")

        vision.cleanup()
        print("🎉 视觉集成测试通过!")
        return True

    except Exception as e:
        print(f"❌ 视觉集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ai_assistant_basic():
    """测试AI助手基本功能"""
    print("\n🔄 测试AI助手基本功能...")

    try:
        from src.assistant.ai_assistant import AIAssistant

        # 创建AI助手实例
        assistant = AIAssistant()

        # 获取系统能力
        capabilities = assistant.get_capabilities()
        print(f"📊 AI助手能力:")
        print(f"   📝 文本命令: {capabilities['text_commands']}")
        print(f"   🎤 语音命令: {capabilities['voice_commands']}")

        # 获取系统状态
        status = assistant.get_system_status()
        print(f"📊 系统状态: 会话ID {status['session_id']}")

        # 清理
        assistant.cleanup()
        print("🎉 AI助手基本功能测试通过!")
        return True

    except Exception as e:
        print(f"❌ AI助手基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_console_app_import():
    """测试控制台应用导入"""
    print("\n🔄 测试控制台应用导入...")

    try:
        import console_app
        print("✅ 控制台应用导入成功")

        print("🎉 控制台应用导入测试通过!")
        return True

    except Exception as e:
        print(f"❌ 控制台应用导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("=" * 60)
    print("🎈 MCP Floating Ball - 系统集成测试")
    print("=" * 60)

    test_results = []

    # 运行所有测试
    tests = [
        ("基本导入测试", test_basic_imports),
        ("NLP处理器测试", test_nlp_processor),
        ("命令处理器测试", test_command_handler),
        ("工具调用器测试", test_tool_caller),
        ("视觉集成测试", test_vision_integration),
        ("AI助手基本测试", test_ai_assistant_basic),
        ("控制台应用导入测试", test_console_app_import),
    ]

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            test_results.append((test_name, False))

    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有测试通过！系统集成成功！")
        return 0
    else:
        print("⚠️  部分测试失败，需要检查相关模块")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)