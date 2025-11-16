#!/usr/bin/env python3
"""
快速测试脚本 - 测试基本导入
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_basic():
    """基本测试"""
    print("🔄 测试基本导入...")

    try:
        # 测试工具导入
        from src.tools.base import BaseTool, ToolParameter, ToolMetadata
        print("✅ 工具基类导入成功")

        # 测试异常类导入
        from src.core.exceptions import AssistantError
        print("✅ 异常类导入成功")

        # 测试核心模块导入
        from src.core.logging import get_logger
        from src.core.config_manager import get_config_manager
        print("✅ 核心模块导入成功")

        # 测试视觉模块
        from src.vision.vision_integration import VisionIntegration
        vision = VisionIntegration()
        print("✅ 视觉模块导入和初始化成功")
        vision.cleanup()

        # 测试工具模块
        from src.tools.system.system_info import SystemInfoTool
        tool = SystemInfoTool()
        print("✅ 系统信息工具导入成功")

        print("🎉 基本测试通过!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_basic())
    print(f"\n测试结果: {'成功' if result else '失败'}")
    sys.exit(0 if result else 1)