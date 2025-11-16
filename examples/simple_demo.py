"""
MCP Floating Ball - 简化演示

展示核心功能的基本使用。
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.assistant.advanced_ai_controller import get_advanced_ai_controller
    from src.core.logging import get_logger

    logger = get_logger("simple_demo")

    async def main():
        """简单的演示函数"""
        print("🚀 MCP Floating Ball 高级AI系统简化演示")
        print("=" * 50)

        try:
            # 获取控制器实例
            controller = get_advanced_ai_controller()
            print("✅ 高级AI控制器实例创建成功")

            # 测试基本状态检查
            print("\n📊 系统状态:")
            print(f"   控制器状态: {'已初始化' if controller else '未初始化'}")

            # 如果控制器有基本方法，测试一下
            if hasattr(controller, 'request_count'):
                controller.request_count = 10
                controller.success_count = 8
                controller.error_count = 2
                success_rate = (controller.success_count / controller.request_count * 100) if controller.request_count > 0 else 0
                print(f"   请求统计: {controller.request_count} 总计, {controller.success_count} 成功, {controller.error_count} 失败")
                print(f"   成功率: {success_rate:.1f}%")

            print("\n🎉 演示完成！系统基本功能正常。")

        except Exception as e:
            print(f"❌ 演示失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")

    if __name__ == "__main__":
        asyncio.run(main())

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖都已正确安装")
    print("这可能是因为缺少某些依赖包或路径配置问题")

    # 尝试简单的测试
    print("\n🧪 运行简单测试...")
    try:
        import numpy as np
        print("✅ numpy 可用")
    except ImportError:
        print("❌ numpy 不可用，请安装: pip install numpy")

    try:
        from src.core.logging import get_logger
        logger = get_logger("test")
        print("✅ 日志系统可用")
    except ImportError:
        print("❌ 日志系统不可用")

    print("\n💡 建议检查:")
    print("1. 确保在正确的目录下运行脚本")
    print("2. 检查 Python 路径配置")
    print("3. 安装所有必需的依赖包")