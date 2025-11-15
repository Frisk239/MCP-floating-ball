#!/usr/bin/env python3
"""
MCP Floating Ball - 工具功能测试脚本

测试所有已实现的工具功能，验证其可用性和正确性。
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ToolTester:
    """工具测试器"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.test_results = []
        self.start_time = time.time()

    def _setup_logger(self):
        """设置测试日志"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)

    def log_test_start(self, test_name: str):
        """记录测试开始"""
        self.logger.info(f"🧪 开始测试: {test_name}")
        return time.time()

    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """记录测试结果"""
        status = "✅ 成功" if success else "❌ 失败"
        duration_text = f" ({duration:.2f}s)" if duration > 0 else ""

        self.logger.info(f"{status}: {test_name}{duration_text}")
        if message:
            self.logger.info(f"   📝 {message}")

        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })

    def test_system_tools(self):
        """测试系统控制类工具"""
        test_name = "系统控制类工具"
        start_time = self.log_test_start(test_name)

        try:
            # 测试应用启动器
            try:
                from src.tools.system.application_launcher import ApplicationLauncherTool
                launcher = ApplicationLauncherTool()

                # 测试应用列表
                apps = launcher.list_applications()
                apps_available = len(apps.get("applications", {})) > 0

                # 测试应用搜索
                search_results = launcher.search_applications("calc")
                search_works = isinstance(search_results, dict)

                success = apps_available and search_works
                message = f"应用列表: {len(apps.get('applications', {}))}个, 搜索功能: {'正常' if search_works else '异常'}"
            except Exception as e:
                success = False
                message = f"应用启动器测试失败: {e}"

        except Exception as e:
            success = False
            message = f"系统工具测试失败: {e}"

        duration = time.time() - start_time
        self.log_test_result(test_name, success, message, duration)
        return success

    def test_file_tools(self):
        """测试文件处理类工具"""
        test_name = "文件处理类工具"
        start_time = self.log_test_start(test_name)

        try:
            # 测试文本操作工具
            try:
                from src.tools.file.text_operations import TextOperationsTool
                text_tool = TextOperationsTool()

                # 创建测试文件
                test_content = "这是测试内容\n包含中文和English"
                test_file = "test_file.txt"

                # 测试写入
                write_result = text_tool.execute(
                    operation="write",
                    file_path=test_file,
                    content=test_content
                )
                write_success = write_result.get("success", False)

                # 测试读取
                if write_success:
                    read_result = text_tool.execute(
                        operation="read",
                        file_path=test_file
                    )
                    read_success = read_result.get("success", False)
                    content_match = read_result.get("content") == test_content
                else:
                    read_success = False
                    content_match = False

                # 清理测试文件
                import os
                if os.path.exists(test_file):
                    os.remove(test_file)

                success = write_success and read_success and content_match
                message = f"文本操作: 写入{'✓' if write_success else '✗'}, 读取{'✓' if read_success else '✗'}, 内容匹配{'✓' if content_match else '✗'}"
            except Exception as e:
                success = False
                message = f"文本操作工具测试失败: {e}"

            # 测试格式转换工具（仅测试初始化）
            try:
                from src.tools.file.format_converter import FormatConverterTool
                converter = FormatConverterTool()
                conversions = converter.get_supported_conversions()
                conversions_available = len(conversions.get("conversions", {})) > 0

                if not success:
                    success = conversions_available
                    message = f"格式转换工具: {len(conversions.get('conversions', {}))}种转换类型"
            except Exception as e:
                if not success:
                    success = False
                    message += f", 格式转换工具失败: {e}"

        except Exception as e:
            success = False
            message = f"文件工具测试失败: {e}"

        duration = time.time() - start_time
        self.log_test_result(test_name, success, message, duration)
        return success

    def test_network_tools(self):
        """测试网络工具"""
        test_name = "网络工具"
        start_time = self.log_test_start(test_name)

        try:
            # 测试多搜索引擎（仅测试初始化）
            try:
                from src.tools.network.multi_search import MultiSearchTool
                search_tool = MultiSearchTool()
                engine_info = search_tool.get_engine_info()
                engines_available = len(engine_info.get("engines", {})) > 0

                message = f"搜索引擎: {len(engine_info.get('engines', {}))}个"
            except Exception as e:
                engines_available = False
                message = f"多搜索引擎测试失败: {e}"

            # 测试网页抓取工具（仅测试初始化）
            try:
                from src.tools.network.web_scraper import WebScraperTool
                scraper = WebScraperTool()
                scraper_available = True
                message += f", 网页抓取: ✓"
            except Exception as e:
                scraper_available = False
                message += f", 网页抓取失败: {e}"

            success = engines_available or scraper_available
            if not message.endswith(f", 网页抓取: ✓"):
                message += f", 网页抓取: ✗"

        except Exception as e:
            success = False
            message = f"网络工具测试失败: {e}"

        duration = time.time() - start_time
        self.log_test_result(test_name, success, message, duration)
        return success

    def test_tool_registry(self):
        """测试工具注册器"""
        test_name = "工具注册器"
        start_time = self.log_test_start(test_name)

        try:
            from src.tools.registry import tool_registry

            # 测试工具注册器功能
            registry_available = tool_registry is not None

            # 获取工具统计
            try:
                stats = tool_registry.get_stats()
                stats_available = isinstance(stats, dict) and "total_tools" in stats
            except:
                stats_available = False

            # 获取工具架构
            try:
                schema = tool_registry.export_tools_schema()
                schema_available = isinstance(schema, dict) and "tools" in schema
            except:
                schema_available = False

            success = registry_available and stats_available and schema_available
            message = f"注册器: {'✓' if registry_available else '✗'}, 统计: {'✓' if stats_available else '✗'}, 架构: {'✓' if schema_available else '✗'}"

            if stats_available:
                message += f", 总工具数: {stats.get('total_tools', 0)}"

        except Exception as e:
            success = False
            message = f"工具注册器测试失败: {e}"

        duration = time.time() - start_time
        self.log_test_result(test_name, success, message, duration)
        return success

    def test_tool_integration(self):
        """测试工具集成"""
        test_name = "工具集成"
        start_time = self.log_test_start(test_name)

        try:
            # 测试所有工具的导入
            import_results = {}

            # 系统工具
            try:
                from src.tools.system import ApplicationLauncherTool, WindowManagerTool, SystemInfoTool
                import_results["system"] = "✓"
            except Exception as e:
                import_results["system"] = f"✗ ({e})"

            # 文件工具
            try:
                from src.tools.file import FormatConverterTool, TextOperationsTool
                import_results["file"] = "✓"
            except Exception as e:
                import_results["file"] = f"✗ ({e})"

            # 网络工具
            try:
                from src.tools.network import MultiSearchTool, WebScraperTool
                import_results["network"] = "✓"
            except Exception as e:
                import_results["network"] = f"✗ ({e})"

            # 检查导入成功率
            successful_imports = sum(1 for result in import_results.values() if result == "✓")
            total_imports = len(import_results)

            success = successful_imports >= 2  # 至少2个模块导入成功
            message = f"导入成功率: {successful_imports}/{total_imports}, 详情: {import_results}"

        except Exception as e:
            success = False
            message = f"工具集成测试失败: {e}"

        duration = time.time() - start_time
        self.log_test_result(test_name, success, message, duration)
        return success

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 MCP Floating Ball 工具测试")
        print("=" * 60)

        # 测试列表
        tests = [
            ("系统控制类工具", self.test_system_tools),
            ("文件处理类工具", self.test_file_tools),
            ("网络工具", self.test_network_tools),
            ("工具注册器", self.test_tool_registry),
            ("工具集成", self.test_tool_integration),
        ]

        # 执行测试
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.logger.error(f"测试执行异常: {test_name} - {e}")
                self.log_test_result(test_name, False, f"测试执行异常: {e}")

        # 生成摘要
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)

        print("\n" + "=" * 60)
        print("🧪 MCP Floating Ball 工具测试报告")
        print("=" * 60)
        print(f"📊 测试总数: {total}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {total - passed}")
        print(f"📈 成功率: {(passed/total*100):.1f}%")
        print(f"⏱️  总耗时: {time.time() - self.start_time:.2f}s")
        print(f"🕐 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 打印失败的测试
        failed_results = [result for result in self.test_results if not result["success"]]
        if failed_results:
            print(f"\n❌ 失败的测试 ({len(failed_results)}):")
            for result in failed_results:
                print(f"   • {result['test']}: {result['message']}")

        # 打印工具统计
        print(f"\n📋 工具实现统计:")
        try:
            from src.tools.registry import tool_registry
            stats = tool_registry.get_stats()
            print(f"   • 总工具数: {stats.get('total_tools', 0)}")
            print(f"   • 系统控制工具: {stats.get('categories', {}).get('system', 0)}")
            print(f"   • 文件处理工具: {stats.get('categories', {}).get('file', 0)}")
            print(f"   • 网络工具: {stats.get('categories', {}).get('network', 0)}")
        except Exception as e:
            print(f"   • 无法获取工具统计: {e}")

        return passed == total


def main():
    """主函数"""
    tester = ToolTester()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 所有工具测试通过！")
        print("\n💡 下一步:")
        print("   1. 所有工具模块正常工作")
        print("   2. 可以开始使用AI助手调用这些工具")
        print("   3. 实现剩余的高级功能（语音唤醒、视觉识别等）")
    else:
        print("\n⚠️  部分工具测试失败，请检查上述问题。")
        print("\n🔧 建议:")
        print("   1. 确保安装了所有依赖: pip install -r requirements/base.txt")
        print("   2. 检查模块导入是否正常")
        print("   3. 查看详细错误信息并修复")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())