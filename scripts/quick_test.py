#!/usr/bin/env python3
"""
MCP Floating Ball - 快速测试脚本

快速验证系统基础功能是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")

    try:
        # 核心模块
        import src.core.config
        import src.core.logging
        import src.core.exceptions
        print("   ✅ 核心模块导入成功")

        # AI模块
        import src.ai.orchestrator
        print("   ✅ AI模块导入成功")

        # 工具模块
        import src.tools.base
        import src.tools.registry
        print("   ✅ 工具模块导入成功")

        # 界面模块
        import src.interfaces.console_agent
        print("   ✅ 界面模块导入成功")

        return True

    except ImportError as e:
        print(f"   ❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 模块导入异常: {e}")
        return False


def test_config():
    """测试配置系统"""
    print("\n⚙️ 测试配置系统...")

    try:
        from src.core.config import get_settings

        # 测试配置加载
        settings = get_settings()
        if settings is None:
            print("   ❌ 配置加载失败")
            return False

        print("   ✅ 配置系统正常")

        # 测试API密钥验证
        api_keys = settings.validate_api_keys()
        print(f"   🔑 API密钥验证: {api_keys}")

        return True

    except Exception as e:
        print(f"   ❌ 配置系统测试失败: {e}")
        return False


def test_logging():
    """测试日志系统"""
    print("\n📝 测试日志系统...")

    try:
        from src.core.logging import get_logger

        logger = get_logger("test")
        logger.info("测试日志消息")

        print("   ✅ 日志系统正常")
        return True

    except Exception as e:
        print(f"   ❌ 日志系统测试失败: {e}")
        return False


def test_tool_system():
    """测试工具系统"""
    print("\n🔧 测试工具系统...")

    try:
        from src.tools.registry import tool_registry

        # 测试工具注册器
        tools_count = len(tool_registry)
        print(f"   📊 已注册工具数量: {tools_count}")

        # 测试工具架构导出
        schema = tool_registry.export_tools_schema()
        if isinstance(schema, dict) and "tools" in schema:
            print("   ✅ 工具架构导出成功")
        else:
            print("   ❌ 工具架构导出失败")
            return False

        return True

    except Exception as e:
        print(f"   ❌ 工具系统测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")

    required_files = [
        "pyproject.toml",
        "requirements/base.txt",
        "src/core/config.py",
        "src/ai/orchestrator.py",
        "src/tools/base.py",
        "src/tools/registry.py",
        "src/interfaces/console_agent.py",
        "scripts/test_console.py",
        ".env"
    ]

    existing_files = 0
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files += 1
        else:
            print(f"   ❌ 缺失文件: {file_path}")

    success_rate = existing_files / len(required_files)
    print(f"   📊 文件完整性: {existing_files}/{len(required_files)} ({success_rate*100:.1f}%)")

    return success_rate >= 0.8


def main():
    """主函数"""
    print("🚀 MCP Floating Ball 快速测试")
    print("=" * 50)

    tests = [
        ("模块导入", test_imports),
        ("配置系统", test_config),
        ("日志系统", test_logging),
        ("工具系统", test_tool_system),
        ("文件结构", test_file_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ {test_name}测试异常: {e}")
            results.append(False)

    # 生成摘要
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 50)
    print("📊 快速测试摘要")
    print("=" * 50)
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    print(f"📈 成功率: {(passed/total*100):.1f}%")

    if passed == total:
        print("\n🎉 所有快速测试通过！")
        print("\n💡 建议下一步:")
        print("   1. 配置API密钥（如果尚未配置）")
        print("   2. 运行: python scripts/test_api_services.py")
        print("   3. 或者直接运行: python scripts/test_console.py")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查上述问题。")
        print("\n🔧 建议:")
        print("   1. 确保在项目根目录运行此脚本")
        print("   2. 安装依赖: pip install -r requirements/base.txt")
        print("   3. 检查Python版本是否 >= 3.11")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)